"""Resolve questions to automatically induced Concepts with local embeddings."""

from __future__ import annotations

from typing import Any

from automated_concept_induction import DEFAULT_RESOLVER_MODEL


def query_text(question: str, model_name: str) -> str:
    if "e5" in model_name.casefold():
        return f"query: {question}"
    return question


def passage_text(value: str, model_name: str) -> str:
    if "e5" in model_name.casefold():
        return f"passage: {value}"
    return value


def concept_fields(concept: dict[str, Any]) -> list[tuple[str, str, list[float] | None]]:
    """Return independently searchable labels learned from the corpus."""
    fields = [("concept_name", concept["concept_name"], concept.get("resolver_embedding"))]
    if concept.get("term_fields") is not None:
        fields.extend(
            ("term", term["text"], term.get("embedding"))
            for term in concept["term_fields"]
            if term.get("text")
        )
    else:
        fields.extend(("term", term, None) for term in concept.get("term_texts") or [])

    unique_fields = []
    seen = set()
    for field_type, value, embedding in fields:
        normalized = " ".join((value or "").casefold().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_fields.append((field_type, value, embedding))
    return unique_fields


def encode_ontology_fields(
    concepts: list[dict[str, Any]],
    terms: list[dict[str, Any]],
    model_name: str = DEFAULT_RESOLVER_MODEL,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Precompute local resolver vectors for leaf Concept names and Terms."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic concept resolution dependencies are missing. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    leaf_concepts = [concept for concept in concepts if concept.get("level") == 2]
    field_rows = [
        ("concept", concept["concept_id"], concept["name"])
        for concept in leaf_concepts
    ]
    field_rows.extend(("term", term["term_id"], term["text"]) for term in terms)
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        [passage_text(text, model_name) for _, _, text in field_rows],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    concept_embeddings = []
    term_embeddings = []
    for (field_type, field_id, _), embedding in zip(field_rows, embeddings):
        row = {"embedding": embedding.astype(float).tolist(), "resolver_model": model_name}
        if field_type == "concept":
            concept_embeddings.append({"concept_id": field_id, **row})
        else:
            term_embeddings.append({"term_id": field_id, **row})
    return {
        "concepts": concept_embeddings,
        "terms": term_embeddings,
        "dimensions": int(embeddings.shape[1]),
        "model": model_name,
    }


def resolve_with_embeddings(
    question: str,
    concepts: list[dict[str, Any]],
    model_name: str = DEFAULT_RESOLVER_MODEL,
    device: str = "cpu",
    threshold: float = 0.5,
    ambiguity_margin: float = 0.06,
    max_resolved_concepts: int = 1,
    candidate_limit: int = 5,
) -> dict[str, Any]:
    if not concepts:
        return {
            "status": "unavailable",
            "concepts": [],
            "candidates": [],
            "explanation": "No automatically induced Concept embeddings are available.",
        }

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic concept resolution dependencies are missing. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    searchable_fields = []
    stored_embeddings = []
    field_owners = []
    for concept_position, concept in enumerate(concepts):
        for field_type, value, embedding in concept_fields(concept):
            searchable_fields.append(passage_text(value, model_name))
            stored_embeddings.append(embedding)
            field_owners.append((concept_position, field_type, value))

    model = SentenceTransformer(model_name, device=device)
    question_embedding = model.encode(
        [query_text(question, model_name)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]
    if stored_embeddings and all(embedding is not None for embedding in stored_embeddings):
        field_matrix = np.asarray(stored_embeddings, dtype=np.float32)
    else:
        field_matrix = model.encode(
            searchable_fields,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
    field_scores = field_matrix @ question_embedding
    field_results: dict[int, list[tuple[float, str, str]]] = {}
    for score, (concept_position, field_type, value) in zip(field_scores, field_owners):
        field_results.setdefault(concept_position, []).append((float(score), field_type, value))

    best_fields = {}
    for concept_position, results in field_results.items():
        adjusted = [
            (score - (0.03 if field_type == "term" else 0.0), field_type, value)
            for score, field_type, value in results
        ]
        best_fields[concept_position] = max(adjusted, key=lambda item: item[0])
    ordered_positions = sorted(best_fields, key=lambda position: best_fields[position][0], reverse=True)

    candidates = [
        {
            "concept_id": concepts[position]["concept_id"],
            "concept_name": concepts[position]["concept_name"],
            "method": "multilingual_term_maxsim",
            "confidence": round(best_fields[position][0], 6),
            "trusted_mapping": True,
            "matched_field": best_fields[position][1],
            "matched_text": best_fields[position][2],
        }
        for position in ordered_positions[:candidate_limit]
    ]
    best_score = candidates[0]["confidence"]
    if best_score < threshold:
        return {
            "status": "unresolved",
            "concepts": [],
            "candidates": candidates,
            "explanation": (
                f"Best semantic Concept score {best_score:.3f} is below "
                f"the {threshold:.3f} threshold."
            ),
        }

    if (
        max_resolved_concepts == 1
        and len(candidates) > 1
        and best_score - candidates[1]["confidence"] < ambiguity_margin
    ):
        return {
            "status": "ambiguous",
            "concepts": [],
            "candidates": candidates,
            "explanation": (
                "The top multilingual Concept candidates are too close to select "
                f"automatically (margin < {ambiguity_margin:.3f})."
            ),
        }

    resolved = []
    for candidate in candidates:
        if len(resolved) >= max_resolved_concepts:
            break
        if candidate["confidence"] < threshold:
            break
        if best_score - candidate["confidence"] > ambiguity_margin:
            break
        resolved.append({**candidate, "matched_terms": [candidate["matched_text"]]})

    return {
        "status": "resolved",
        "concepts": resolved,
        "candidates": candidates,
        "explanation": (
            "No exact or fuzzy auto-generated Term matched; the question was "
            f"resolved by local {model_name} MaxSim over independently embedded "
            "Concept names and corpus-induced Terms."
        ),
    }
