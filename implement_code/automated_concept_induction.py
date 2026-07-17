"""Automatic, provenance-first concept induction for bilingual regulatory text.

The pipeline combines multilingual sentence embeddings, density clustering,
class-based TF-IDF term representation, and data-driven hierarchy induction.
It does not require a manually maintained concept or synonym dictionary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALGORITHM_VERSION = "embedding_hdbscan_ctfidf_v5"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_RESOLVER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CJK_RUN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
SECTION_PREFIX_PATTERN = re.compile(
    r"^(?:第\s*[一二三四五六七八九十百0-9]+\s*章\s*[-—–:：]*\s*|"
    r"chapter\s+\d+\s*[-—–:：]*\s*|"
    r"\d+(?:\.\d+){0,6}\.?\s*)",
    re.IGNORECASE,
)

ENGLISH_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "doing",
    "during",
    "each",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "must",
    "no",
    "nor",
    "not",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "same",
    "shall",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}

CHINESE_STOPWORDS = {
    "之",
    "了",
    "其",
    "及",
    "於",
    "與",
    "和",
    "一個",
    "一些",
    "一般",
    "不得",
    "以及",
    "任何",
    "但",
    "例如",
    "依據",
    "倘若",
    "其他",
    "其中",
    "則",
    "包括",
    "可以",
    "可能",
    "各",
    "同時",
    "因此",
    "如",
    "如果",
    "就",
    "應",
    "應當",
    "應該",
    "或",
    "所有",
    "所述",
    "按照",
    "根據",
    "此外",
    "然而",
    "為",
    "由",
    "的",
    "等",
    "而",
    "該",
    "該等",
    "這些",
    "這個",
    "進行",
    "須",
}

GENERIC_TERMS = {
    "applicable requirements",
    "authorized institution",
    "general requirements",
    "introduction",
    "overview",
    "purpose",
    "relevant requirements",
    "the authority",
    "the institution",
    "引言",
    "目的",
    "概覽",
    "範圍",
    "上述規定",
    "下列各項",
    "以下各項",
    "有關規定",
    "本章",
    "本節",
    "本部分",
    "本段",
    "本指引",
    "該機構",
    "該認可機構",
}


@dataclass(frozen=True)
class MLStack:
    np: Any
    jieba: Any
    SentenceTransformer: Any
    HDBSCAN: Any
    AgglomerativeClustering: Any
    PCA: Any
    silhouette_score: Any
    normalize: Any


def load_ml_stack() -> MLStack:
    """Import optional machine-learning dependencies with an actionable error."""
    try:
        import jieba
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering, HDBSCAN
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize
    except ImportError as error:
        raise RuntimeError(
            "Automatic concept induction dependencies are missing. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    jieba.setLogLevel(logging.ERROR)
    return MLStack(
        np=np,
        jieba=jieba,
        SentenceTransformer=SentenceTransformer,
        HDBSCAN=HDBSCAN,
        AgglomerativeClustering=AgglomerativeClustering,
        PCA=PCA,
        silhouette_score=silhouette_score,
        normalize=normalize,
    )


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_term(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff'\- ]+", " ", normalized)
    normalized = normalize_space(normalized).strip("-' ")
    if CJK_PATTERN.search(normalized) and not re.search(r"[a-z0-9]", normalized):
        normalized = normalized.replace(" ", "")
    return normalized


def detect_language(value: str) -> str:
    has_cjk = CJK_PATTERN.search(value or "") is not None
    has_latin = re.search(r"[A-Za-z]", value or "") is not None
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"


def term_evidence_weight(term: dict[str, Any], corpus_chunk_count: int) -> float:
    """Estimate how strongly an exact term occurrence identifies its Concept."""
    normalized = term["normalized_text"]
    if CJK_PATTERN.search(normalized):
        phrase_factor = min(1.0, len(normalized.replace(" ", "")) / 4.0)
    else:
        phrase_factor = min(1.0, len(normalized.split()) / 2.0)
    document_frequency = max(1, int(term["chunk_frequency"]))
    inverse_frequency = math.log((corpus_chunk_count + 1) / (document_frequency + 1))
    inverse_frequency /= math.log(corpus_chunk_count + 1)
    return round(float(term["score"]) * phrase_factor * inverse_frequency, 8)


def stable_hash(*values: str, length: int = 16) -> str:
    payload = "\x1f".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def l2_normalize(vector: Any, np: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _valid_candidate(value: str) -> bool:
    if not value or value in GENERIC_TERMS:
        return False
    compact = value.replace(" ", "")
    if len(compact) < 2 or len(value) > 80:
        return False
    if re.fullmatch(r"[\d.\-]+", value):
        return False
    if detect_language(value) == "en":
        words = value.split()
        if len(words) == 1 and len(words[0]) < 3:
            return False
        if all(word in ENGLISH_STOPWORDS for word in words):
            return False
    if detect_language(value) == "zh" and value in CHINESE_STOPWORDS:
        return False
    return True


def _candidate_display(original: str, normalized: str) -> str:
    stripped = normalize_space(original).strip(" .,:;()[]{}-—–")
    if re.fullmatch(r"[A-Z][A-Z0-9-]{1,10}", stripped):
        return stripped
    return normalized


class CandidateExtractor:
    """Extract bilingual term candidates without a domain concept dictionary."""

    def __init__(self, jieba_module: Any):
        self.jieba = jieba_module
        self.display_forms: dict[str, Counter[str]] = defaultdict(Counter)

    def _register(self, original: str, weight: int, output: Counter[str]) -> None:
        normalized = normalize_term(original)
        if not _valid_candidate(normalized):
            return
        output[normalized] += weight
        self.display_forms[normalized][_candidate_display(original, normalized)] += weight

    def _english_candidates(self, text: str, weight: int, output: Counter[str]) -> None:
        for segment in re.split(r"[\n。！？!?；;：:]", text or ""):
            raw_tokens = ENGLISH_TOKEN_PATTERN.findall(segment)
            current: list[str] = []

            def flush() -> None:
                if not current:
                    return
                for size in range(1, 5):
                    for index in range(len(current) - size + 1):
                        phrase = " ".join(current[index : index + size])
                        self._register(phrase, weight, output)
                current.clear()

            for token in raw_tokens:
                if token.casefold() in ENGLISH_STOPWORDS:
                    flush()
                else:
                    current.append(token)
            flush()

    def _chinese_candidates(self, text: str, weight: int, output: Counter[str]) -> None:
        for run in CJK_RUN_PATTERN.findall(text or ""):
            tokens = []
            for token in self.jieba.cut(run, HMM=True):
                token = normalize_term(token)
                if not token or token in CHINESE_STOPWORDS:
                    tokens.append("")
                elif CJK_PATTERN.search(token):
                    tokens.append(token)

            current: list[str] = []

            def flush() -> None:
                if not current:
                    return
                for size in range(1, 4):
                    for index in range(len(current) - size + 1):
                        phrase = "".join(current[index : index + size])
                        self._register(phrase, weight, output)
                current.clear()

            for token in tokens:
                if token:
                    current.append(token)
                else:
                    flush()
            flush()

    def extract(self, text: str, weight: int = 1) -> Counter[str]:
        output: Counter[str] = Counter()
        self._english_candidates(text, weight, output)
        self._chinese_candidates(text, weight, output)
        return output

    def extract_headings(self, chunk: dict[str, Any], weight: int = 4) -> Counter[str]:
        headings = []
        if chunk.get("section_title"):
            headings.append(chunk["section_title"])
        if chunk.get("section_path"):
            headings.extend(part.strip() for part in chunk["section_path"].split(">"))

        output: Counter[str] = Counter()
        for heading in dict.fromkeys(headings):
            cleaned = SECTION_PREFIX_PATTERN.sub("", normalize_space(heading)).strip(" -—–:：")
            if cleaned:
                self._register(cleaned, weight * 2, output)
                output.update(self.extract(cleaned, weight=weight))
        return output

    def preferred_display(self, normalized: str) -> str:
        variants = self.display_forms.get(normalized)
        if not variants:
            return normalized
        return variants.most_common(1)[0][0]


def corpus_fingerprint(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update((chunk.get("chunk_id") or "").encode("utf-8"))
        digest.update(b"\x00")
        digest.update((chunk.get("retrieval_text") or chunk.get("text") or "").encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def embedding_text(chunk: dict[str, Any], model_name: str) -> str:
    text = normalize_space(chunk.get("retrieval_text") or chunk.get("text") or "")
    if "e5" in model_name.casefold():
        return f"passage: {text}"
    return text


def _embedding_cache_path(cache_dir: Path, model_name: str, corpus_hash: str) -> Path:
    model_key = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return cache_dir / f"{model_key}_{corpus_hash[:20]}.npz"


def encode_chunks(
    chunks: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    use_cache: bool,
    stack: MLStack,
) -> tuple[Any, int, bool]:
    corpus_hash = corpus_fingerprint(chunks)
    cache_path = _embedding_cache_path(cache_dir, model_name, corpus_hash)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if use_cache and cache_path.exists():
        with stack.np.load(cache_path, allow_pickle=False) as cached:
            cached_ids = cached["chunk_ids"].tolist()
            if cached_ids == chunk_ids:
                embeddings = cached["embeddings"].astype(stack.np.float32)
                return embeddings, int(embeddings.shape[1]), True

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Loading local embedding model: {model_name}")
    model = stack.SentenceTransformer(model_name, device=device)
    texts = [embedding_text(chunk, model_name) for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(stack.np.float32)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stack.np.savez_compressed(
            cache_path,
            chunk_ids=stack.np.asarray(chunk_ids),
            embeddings=embeddings,
        )
    return embeddings, int(embeddings.shape[1]), False


def reduce_embeddings(embeddings: Any, random_state: int, stack: MLStack) -> Any:
    dimensions = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    if dimensions < 2 or dimensions >= embeddings.shape[1]:
        return embeddings
    reducer = stack.PCA(n_components=dimensions, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)
    return stack.normalize(reduced).astype(stack.np.float32)


def cluster_embeddings(
    embeddings: Any,
    min_cluster_size: int,
    min_samples: int,
    random_state: int,
    stack: MLStack,
) -> tuple[Any, Any, str]:
    reduced = reduce_embeddings(embeddings, random_state, stack)
    clusterer = stack.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        n_jobs=-1,
        copy=True,
    )
    labels = clusterer.fit_predict(reduced)
    probabilities = getattr(clusterer, "probabilities_", stack.np.ones(len(labels)))
    cluster_count = len(set(int(label) for label in labels if int(label) >= 0))

    if cluster_count >= 2:
        return labels, probabilities, "hdbscan"

    fallback_clusters = max(2, min(24, round(math.sqrt(len(embeddings)) / 2)))
    fallback = stack.AgglomerativeClustering(
        n_clusters=fallback_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = fallback.fit_predict(embeddings)
    probabilities = stack.np.ones(len(labels), dtype=stack.np.float32)
    return labels, probabilities, "agglomerative_fallback"


def calculate_centroids(labels: Any, embeddings: Any, stack: MLStack) -> dict[int, Any]:
    centroids = {}
    for label in sorted(set(int(value) for value in labels if int(value) >= 0)):
        member_embeddings = embeddings[labels == label]
        centroids[label] = l2_normalize(member_embeddings.mean(axis=0), stack.np)
    return centroids


def assign_primary_clusters(
    labels: Any,
    probabilities: Any,
    embeddings: Any,
    centroids: dict[int, Any],
    stack: MLStack,
) -> tuple[list[int], list[float], list[float]]:
    ordered_labels = sorted(centroids)
    centroid_matrix = stack.np.vstack([centroids[label] for label in ordered_labels])
    similarities = embeddings @ centroid_matrix.T
    primary_labels = []
    confidences = []
    primary_similarities = []

    for index, original_label in enumerate(labels):
        best_position = int(stack.np.argmax(similarities[index]))
        best_label = ordered_labels[best_position]
        best_similarity = float(similarities[index, best_position])
        if int(original_label) >= 0:
            primary_label = int(original_label)
            primary_position = ordered_labels.index(primary_label)
            semantic_similarity = float(similarities[index, primary_position])
            confidence = 0.5 * semantic_similarity + 0.5 * float(probabilities[index])
        else:
            primary_label = best_label
            semantic_similarity = best_similarity
            confidence = 0.8 * semantic_similarity
        primary_labels.append(primary_label)
        primary_similarities.append(round(semantic_similarity, 6))
        confidences.append(round(max(0.0, min(1.0, confidence)), 6))
    return primary_labels, confidences, primary_similarities


def _term_features(value: str) -> set[str]:
    if detect_language(value) == "en":
        return set(value.split())
    compact = value.replace(" ", "")
    if len(compact) <= 2:
        return {compact}
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def select_diverse_terms(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    selected_features: list[set[str]] = []
    for row in rows:
        features = _term_features(row["normalized_text"])
        too_similar = False
        for previous in selected_features:
            union = features | previous
            similarity = len(features & previous) / len(union) if union else 1.0
            if similarity >= 0.88:
                too_similar = True
                break
        if too_similar:
            continue
        selected.append(row)
        selected_features.append(features)
        if len(selected) >= limit:
            break
    return selected


def rank_cluster_terms(
    groups: dict[int, list[int]],
    candidate_counts: list[Counter[str]],
    heading_counts: list[Counter[str]],
    extractor: CandidateExtractor,
    terms_per_concept: int,
    min_term_chunk_frequency: int,
) -> dict[int, list[dict[str, Any]]]:
    group_counts: dict[int, Counter[str]] = {}
    group_document_frequency: dict[int, Counter[str]] = {}
    group_heading_counts: dict[int, Counter[str]] = {}

    for group, indices in groups.items():
        counts: Counter[str] = Counter()
        document_frequency: Counter[str] = Counter()
        headings: Counter[str] = Counter()
        for index in indices:
            counts.update(candidate_counts[index])
            headings.update(heading_counts[index])
            document_frequency.update(candidate_counts[index].keys())
        group_counts[group] = counts
        group_document_frequency[group] = document_frequency
        group_heading_counts[group] = headings

    global_counts: Counter[str] = Counter()
    for counts in group_counts.values():
        global_counts.update(counts)
    average_group_length = sum(sum(counts.values()) for counts in group_counts.values()) / max(len(groups), 1)

    output: dict[int, list[dict[str, Any]]] = {}
    for group, counts in group_counts.items():
        total = max(sum(counts.values()), 1)
        ranked = []
        for term, frequency in counts.items():
            chunk_frequency = group_document_frequency[group][term]
            if chunk_frequency < min_term_chunk_frequency:
                continue
            inverse_frequency = math.log1p(average_group_length / max(global_counts[term], 1))
            score = (frequency / total) * inverse_frequency
            score *= 1.0 + 0.12 * math.log1p(chunk_frequency)
            ranked.append(
                {
                    "text": extractor.preferred_display(term),
                    "normalized_text": term,
                    "score": round(score, 8),
                    "frequency": int(frequency),
                    "chunk_frequency": int(chunk_frequency),
                    "heading_frequency": int(group_heading_counts[group][term]),
                    "language": detect_language(term),
                }
            )
        ranked.sort(
            key=lambda row: (
                -int(row["heading_frequency"] > 0),
                -row["score"],
                -row["chunk_frequency"],
                row["normalized_text"],
            )
        )
        selected = select_diverse_terms(ranked, terms_per_concept)
        if not selected and counts:
            fallback_term = counts.most_common(1)[0][0]
            selected = [
                {
                    "text": extractor.preferred_display(fallback_term),
                    "normalized_text": fallback_term,
                    "score": 0.0,
                    "frequency": int(counts[fallback_term]),
                    "chunk_frequency": int(group_document_frequency[group][fallback_term]),
                    "heading_frequency": int(group_heading_counts[group][fallback_term]),
                    "language": detect_language(fallback_term),
                }
            ]
        output[group] = selected
    return output


def choose_parent_partition(centroids: Any, random_state: int, stack: MLStack) -> tuple[Any | None, float | None]:
    count = len(centroids)
    if count < 4:
        return None, None
    maximum_groups = min(8, max(2, count // 2))
    best_labels = None
    best_score = -1.0
    best_raw_score = None
    minimum_non_singleton_groups = min(3, max(2, count // 4))

    for group_count in range(2, maximum_groups + 1):
        model = stack.AgglomerativeClustering(
            n_clusters=group_count,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(centroids)
        if len(set(int(label) for label in labels)) < 2:
            continue
        group_sizes = Counter(int(label) for label in labels)
        non_singleton_groups = sum(size > 1 for size in group_sizes.values())
        largest_group_ratio = max(group_sizes.values()) / count
        if non_singleton_groups < minimum_non_singleton_groups or largest_group_ratio > 0.65:
            continue
        raw_score = float(stack.silhouette_score(centroids, labels, metric="cosine"))
        penalized_score = raw_score - 0.01 * group_count
        if penalized_score > best_score:
            best_score = penalized_score
            best_raw_score = raw_score
            best_labels = labels
    return best_labels, best_raw_score


def recursively_split_parent_group(
    concept_ids: list[str],
    centroid_by_concept: dict[str, Any],
    inherited_score: float,
    stack: MLStack,
) -> list[tuple[list[str], float]]:
    if len(concept_ids) < 5:
        return [(concept_ids, inherited_score)]

    centroids = stack.np.vstack([centroid_by_concept[concept_id] for concept_id in concept_ids])
    maximum_groups = min(4, len(concept_ids) // 2)
    best_labels = None
    best_score = -1.0
    for group_count in range(2, maximum_groups + 1):
        model = stack.AgglomerativeClustering(
            n_clusters=group_count,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(centroids)
        group_sizes = Counter(int(label) for label in labels)
        if min(group_sizes.values()) < 2 or max(group_sizes.values()) / len(concept_ids) > 0.75:
            continue
        score = float(stack.silhouette_score(centroids, labels, metric="cosine"))
        if score >= 0.12 and score > best_score:
            best_labels = labels
            best_score = score

    if best_labels is None:
        return [(concept_ids, inherited_score)]

    child_groups: dict[int, list[str]] = defaultdict(list)
    for position, label in enumerate(best_labels):
        child_groups[int(label)].append(concept_ids[position])
    output = []
    for child_ids in child_groups.values():
        output.extend(
            recursively_split_parent_group(
                child_ids,
                centroid_by_concept,
                inherited_score=best_score,
                stack=stack,
            )
        )
    return output


def count_term_occurrences(text: str, term: str, match_type: str) -> int:
    if not text or not term:
        return 0
    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    normalized_term = unicodedata.normalize("NFKC", term).casefold()
    if match_type == "word":
        pattern = rf"(?<![a-z0-9_]){re.escape(normalized_term)}(?![a-z0-9_])"
        return len(re.findall(pattern, normalized_text))
    return normalized_text.count(normalized_term)


def _match_type(term: dict[str, Any]) -> str:
    normalized = term["normalized_text"]
    if term["language"] == "en" and " " not in normalized:
        return "word"
    return "phrase"


def _concept_name(terms: list[dict[str, Any]], fallback: str) -> str:
    if not terms:
        return fallback
    ranked = sorted(terms, key=lambda term: term["score"], reverse=True)
    anchor = ranked[0]
    primary = _expand_name_phrase(anchor, ranked)
    anchor_length = len(anchor["normalized_text"].replace(" ", ""))
    if anchor_length <= 4:
        for secondary_anchor in ranked[1:8]:
            if secondary_anchor["score"] < anchor["score"] * 0.65:
                break
            secondary = _expand_name_phrase(secondary_anchor, ranked)
            if _term_similarity(normalize_term(primary), normalize_term(secondary)) >= 0.35:
                continue
            if normalize_term(primary) in normalize_term(secondary) or normalize_term(secondary) in normalize_term(primary):
                continue
            return f"{primary} / {secondary}"
    return primary


def _expand_name_phrase(anchor: dict[str, Any], terms: list[dict[str, Any]]) -> str:
    anchor_normalized = anchor["normalized_text"]
    anchor_compact = anchor_normalized.replace(" ", "")
    anchor_score = max(float(anchor["score"]), 1e-12)
    best_text = anchor["text"]
    best_quality = 1.0
    for candidate in terms:
        candidate_compact = candidate["normalized_text"].replace(" ", "")
        if anchor_compact not in candidate_compact or len(candidate_compact) > 32:
            continue
        relative_score = float(candidate["score"]) / anchor_score
        if relative_score < 0.55:
            continue
        added_length = min(max(0, len(candidate_compact) - len(anchor_compact)), 12)
        quality = relative_score + 0.08 * added_length
        if quality > best_quality:
            best_quality = quality
            best_text = candidate["text"]
    return best_text


def _term_similarity(first: str, second: str) -> float:
    first_features = _term_features(first)
    second_features = _term_features(second)
    union = first_features | second_features
    return len(first_features & second_features) / len(union) if union else 1.0


def _name_candidate_score(term: dict[str, Any]) -> tuple[float, int, str]:
    compact_length = len(term["normalized_text"].replace(" ", ""))
    specificity = 1.0 + 1.25 * math.log1p(min(compact_length, 24))
    term_score = term.get("score", term.get("ctfidf_score", 0.0))
    score = term_score * specificity
    return score, compact_length, term["normalized_text"]


def ensure_unique_concept_names(
    concepts: list[dict[str, Any]],
    terms: list[dict[str, Any]],
) -> None:
    terms_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for term in terms:
        terms_by_concept[term["concept_id"]].append(term)

    used_names = set()
    for concept in sorted(concepts, key=lambda row: (-row["chunk_count"], row["concept_id"])):
        candidates = sorted(
            terms_by_concept.get(concept["concept_id"], []),
            key=_name_candidate_score,
            reverse=True,
        )
        current_name = normalize_term(concept["name"])
        if current_name and current_name not in used_names:
            used_names.add(current_name)
            continue
        for candidate in candidates:
            normalized_name = candidate["normalized_text"]
            if normalized_name in used_names:
                continue
            concept["name"] = candidate["text"]
            used_names.add(normalized_name)
            break
        else:
            concept["name"] = f"{concept['name']} [{concept['concept_id'][-6:]}]"
            used_names.add(normalize_term(concept["name"]))


def _representative_rows(
    concept_id: str,
    member_indices: list[int],
    centroid: Any,
    embeddings: Any,
    chunks: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        ((float(embeddings[index] @ centroid), index) for index in member_indices),
        reverse=True,
    )
    return [
        {
            "concept_id": concept_id,
            "chunk_id": chunks[index]["chunk_id"],
            "source_file": chunks[index]["source_file"],
            "page": chunks[index]["page"],
            "section_id": chunks[index].get("section_id"),
            "rank": rank,
            "similarity": round(similarity, 6),
        }
        for rank, (similarity, index) in enumerate(ranked[:limit], start=1)
    ]


def build_preview(result: dict[str, Any]) -> dict[str, Any]:
    run = dict(result["run"])
    concepts = []
    representatives_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["representatives"]:
        representatives_by_concept[row["concept_id"]].append(
            {
                "chunk_id": row["chunk_id"],
                "source_file": row["source_file"],
                "page": row["page"],
                "section_id": row["section_id"],
                "rank": row["rank"],
                "similarity": row["similarity"],
            }
        )
    terms_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["terms"]:
        terms_by_concept[row["concept_id"]].append(
            {
                "text": row["text"],
                "score": row["ctfidf_score"],
                "chunk_frequency": row["chunk_frequency"],
            }
        )
    for concept in result["concepts"]:
        preview_concept = {key: value for key, value in concept.items() if key != "embedding"}
        preview_concept["terms"] = terms_by_concept.get(concept["concept_id"], [])
        preview_concept["representatives"] = representatives_by_concept.get(concept["concept_id"], [])
        concepts.append(preview_concept)
    return {
        "run": run,
        "summary": {
            "concepts": len(result["concepts"]),
            "terms": len(result["terms"]),
            "taxonomy_edges": len(result["taxonomy"]),
            "semantic_assignments": len(result["assignments"]),
            "exact_term_mentions": len(result["mentions"]),
        },
        "concepts": concepts,
        "taxonomy": result["taxonomy"],
    }


class AutomaticConceptInducer:
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        resolver_model: str = DEFAULT_RESOLVER_MODEL,
        embedding_batch_size: int = 8,
        device: str = "cpu",
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        terms_per_concept: int = 12,
        min_term_chunk_frequency: int = 2,
        representative_chunks: int = 3,
        max_concepts_per_chunk: int = 2,
        secondary_similarity: float = 0.78,
        secondary_margin: float = 0.035,
        random_state: int = 42,
        cache_dir: str | Path = ".cache/concept_induction",
        use_cache: bool = True,
    ):
        self.embedding_model = embedding_model
        self.resolver_model = resolver_model
        self.embedding_batch_size = embedding_batch_size
        self.device = device
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.terms_per_concept = terms_per_concept
        self.min_term_chunk_frequency = min_term_chunk_frequency
        self.representative_chunks = representative_chunks
        self.max_concepts_per_chunk = max_concepts_per_chunk
        self.secondary_similarity = secondary_similarity
        self.secondary_margin = secondary_margin
        self.random_state = random_state
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def induce(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if len(chunks) < 4:
            raise ValueError("At least four non-TOC chunks are required for automatic concept induction.")

        stack = load_ml_stack()
        corpus_hash = corpus_fingerprint(chunks)
        cluster_size = self.min_cluster_size or max(8, min(10, round(math.sqrt(len(chunks)) / 3)))
        min_samples = self.min_samples or max(2, round(cluster_size * 0.3))
        settings = {
            "embedding_model": self.embedding_model,
            "resolver_model": self.resolver_model,
            "min_cluster_size": cluster_size,
            "min_samples": min_samples,
            "terms_per_concept": self.terms_per_concept,
            "min_term_chunk_frequency": self.min_term_chunk_frequency,
            "representative_chunks": self.representative_chunks,
            "max_concepts_per_chunk": self.max_concepts_per_chunk,
            "secondary_similarity": self.secondary_similarity,
            "secondary_margin": self.secondary_margin,
            "random_state": self.random_state,
        }
        settings_json = json.dumps(settings, sort_keys=True, ensure_ascii=False)
        run_id = f"auto_run_{stable_hash(ALGORITHM_VERSION, corpus_hash, settings_json)}"

        embeddings, embedding_dimensions, cache_hit = encode_chunks(
            chunks,
            model_name=self.embedding_model,
            batch_size=self.embedding_batch_size,
            device=self.device,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            stack=stack,
        )
        labels, probabilities, clustering_method = cluster_embeddings(
            embeddings,
            min_cluster_size=cluster_size,
            min_samples=min_samples,
            random_state=self.random_state,
            stack=stack,
        )
        core_centroids = calculate_centroids(labels, embeddings, stack)
        primary_labels, primary_confidences, primary_similarities = assign_primary_clusters(
            labels,
            probabilities,
            embeddings,
            core_centroids,
            stack,
        )

        raw_groups: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(primary_labels):
            raw_groups[label].append(index)

        raw_label_to_concept_id = {}
        for label, indices in raw_groups.items():
            core_chunk_ids = sorted(
                chunks[index]["chunk_id"]
                for index, original_label in enumerate(labels)
                if int(original_label) == label
            )
            signature_ids = core_chunk_ids or sorted(chunks[index]["chunk_id"] for index in indices)
            raw_label_to_concept_id[label] = f"auto_topic_{stable_hash(*signature_ids, length=12)}"

        extractor = CandidateExtractor(stack.jieba)
        candidate_counts = []
        heading_counts = []
        for chunk in chunks:
            body_counts = extractor.extract(chunk.get("text") or "")
            headings = extractor.extract_headings(chunk)
            combined = body_counts.copy()
            combined.update(headings)
            candidate_counts.append(combined)
            heading_counts.append(headings)

        ranked_terms = rank_cluster_terms(
            raw_groups,
            candidate_counts,
            heading_counts,
            extractor,
            terms_per_concept=self.terms_per_concept,
            min_term_chunk_frequency=self.min_term_chunk_frequency,
        )

        leaf_concepts = []
        terms = []
        representatives = []
        leaf_centroids: dict[str, Any] = {}
        leaf_members: dict[str, list[int]] = {}

        for label, member_indices in sorted(raw_groups.items()):
            concept_id = raw_label_to_concept_id[label]
            centroid = l2_normalize(embeddings[member_indices].mean(axis=0), stack.np)
            leaf_centroids[concept_id] = centroid
            leaf_members[concept_id] = member_indices
            concept_terms = ranked_terms.get(label, [])
            concept_name = _concept_name(concept_terms, f"Topic {concept_id[-6:]}")
            document_count = len({chunks[index]["source_file"] for index in member_indices})
            mean_confidence = sum(primary_confidences[index] for index in member_indices) / len(member_indices)
            leaf_concepts.append(
                {
                    "concept_id": concept_id,
                    "name": concept_name,
                    "category": "auto_topic",
                    "description": "Automatically induced topic represented by: "
                    + "; ".join(term["text"] for term in concept_terms[:6]),
                    "level": 2,
                    "source": "automatic_induction",
                    "method": ALGORITHM_VERSION,
                    "run_id": run_id,
                    "embedding_model": self.embedding_model,
                    "confidence": round(mean_confidence, 6),
                    "chunk_count": len(member_indices),
                    "document_count": document_count,
                    "embedding": centroid.astype(float).tolist(),
                }
            )
            representatives.extend(
                _representative_rows(
                    concept_id,
                    member_indices,
                    centroid,
                    embeddings,
                    chunks,
                    self.representative_chunks,
                )
            )
            for term in concept_terms:
                term_id = f"auto_term_{stable_hash(concept_id, term['normalized_text'], length=14)}"
                terms.append(
                    {
                        "term_id": term_id,
                        "text": term["text"],
                        "normalized_text": term["normalized_text"],
                        "language": term["language"],
                        "match_type": _match_type(term),
                        "source": "automatic_ctfidf",
                        "concept_id": concept_id,
                        "ctfidf_score": term["score"],
                        "evidence_weight": term_evidence_weight(term, len(chunks)),
                        "frequency": term["frequency"],
                        "chunk_frequency": term["chunk_frequency"],
                        "heading_frequency": term["heading_frequency"],
                        "run_id": run_id,
                    }
                )

        ensure_unique_concept_names(leaf_concepts, terms)

        ordered_leaf_ids = sorted(leaf_centroids)
        leaf_matrix = stack.np.vstack([leaf_centroids[concept_id] for concept_id in ordered_leaf_ids])
        parent_labels, hierarchy_score = choose_parent_partition(leaf_matrix, self.random_state, stack)
        root_id = "auto_root_regulatory_corpus"
        root_centroid = l2_normalize(embeddings.mean(axis=0), stack.np)
        root_concept = {
            "concept_id": root_id,
            "name": "Regulatory knowledge corpus",
            "category": "auto_root",
            "description": "Automatically induced root for the loaded regulatory corpus.",
            "level": 0,
            "source": "automatic_induction",
            "method": ALGORITHM_VERSION,
            "run_id": run_id,
            "embedding_model": self.embedding_model,
            "confidence": 1.0,
            "chunk_count": len(chunks),
            "document_count": len({chunk["source_file"] for chunk in chunks}),
            "embedding": root_centroid.astype(float).tolist(),
        }

        parent_concepts = []
        taxonomy = []
        parent_by_leaf: dict[str, str] = {}
        if parent_labels is not None:
            parent_groups: dict[int, list[str]] = defaultdict(list)
            for position, parent_label in enumerate(parent_labels):
                parent_groups[int(parent_label)].append(ordered_leaf_ids[position])

            expanded_parent_groups = []
            for child_ids in parent_groups.values():
                if len(child_ids) <= 1:
                    continue
                expanded_parent_groups.extend(
                    recursively_split_parent_group(
                        child_ids,
                        leaf_centroids,
                        inherited_score=float(hierarchy_score or 0.0),
                        stack=stack,
                    )
                )

            terms_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for term in terms:
                terms_by_concept[term["concept_id"]].append(term)

            for child_ids, group_confidence in expanded_parent_groups:
                parent_id = f"auto_group_{stable_hash(*sorted(child_ids), length=12)}"
                child_terms = [term for child_id in child_ids for term in terms_by_concept[child_id]]
                term_coverage: dict[str, set[str]] = defaultdict(set)
                term_scores: Counter[str] = Counter()
                term_display = {}
                for term in child_terms:
                    normalized = term["normalized_text"]
                    term_coverage[normalized].add(term["concept_id"])
                    term_scores[normalized] += term["ctfidf_score"]
                    term_display[normalized] = term["text"]
                parent_term_keys = sorted(
                    term_scores,
                    key=lambda term: (-len(term_coverage[term]), -term_scores[term], term),
                )
                parent_names = [term_display[term] for term in parent_term_keys[:2]]
                parent_name = " / ".join(parent_names) if parent_names else f"Topic group {parent_id[-6:]}"
                child_centroids = stack.np.vstack([leaf_centroids[child_id] for child_id in child_ids])
                parent_centroid = l2_normalize(child_centroids.mean(axis=0), stack.np)
                child_members = [index for child_id in child_ids for index in leaf_members[child_id]]
                parent_concepts.append(
                    {
                        "concept_id": parent_id,
                        "name": parent_name,
                        "category": "auto_topic_group",
                        "description": "Automatically induced parent topic for: "
                        + "; ".join(
                            next(concept["name"] for concept in leaf_concepts if concept["concept_id"] == child_id)
                            for child_id in child_ids
                        ),
                        "level": 1,
                        "source": "automatic_induction",
                        "method": "agglomerative_silhouette",
                        "run_id": run_id,
                        "embedding_model": self.embedding_model,
                        "confidence": round(group_confidence, 6),
                        "chunk_count": len(child_members),
                        "document_count": len({chunks[index]["source_file"] for index in child_members}),
                        "embedding": parent_centroid.astype(float).tolist(),
                    }
                )
                taxonomy.append(
                    {
                        "parent_concept_id": root_id,
                        "child_concept_id": parent_id,
                        "method": "agglomerative_silhouette",
                        "confidence": round(group_confidence, 6),
                        "run_id": run_id,
                    }
                )
                for child_id in child_ids:
                    parent_by_leaf[child_id] = parent_id
                    taxonomy.append(
                        {
                            "parent_concept_id": parent_id,
                            "child_concept_id": child_id,
                            "method": "agglomerative_silhouette",
                            "confidence": round(group_confidence, 6),
                            "run_id": run_id,
                        }
                    )

        for child_id in ordered_leaf_ids:
            if child_id not in parent_by_leaf:
                taxonomy.append(
                    {
                        "parent_concept_id": root_id,
                        "child_concept_id": child_id,
                        "method": "automatic_root_attachment",
                        "confidence": 1.0,
                        "run_id": run_id,
                    }
                )

        centroid_labels = sorted(raw_groups)
        centroid_matrix = stack.np.vstack(
            [leaf_centroids[raw_label_to_concept_id[label]] for label in centroid_labels]
        )
        all_similarities = embeddings @ centroid_matrix.T
        assignments = []
        for index, chunk in enumerate(chunks):
            primary_label = primary_labels[index]
            primary_id = raw_label_to_concept_id[primary_label]
            assignments.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "concept_id": primary_id,
                    "primary": True,
                    "confidence": primary_confidences[index],
                    "semantic_similarity": primary_similarities[index],
                    "cluster_probability": round(float(probabilities[index]), 6),
                    "method": clustering_method,
                    "run_id": run_id,
                }
            )
            assigned_count = 1
            if self.max_concepts_per_chunk <= 1:
                continue
            ordered_positions = list(stack.np.argsort(all_similarities[index])[::-1])
            top_similarity = float(all_similarities[index, ordered_positions[0]])
            for position in ordered_positions:
                candidate_label = centroid_labels[int(position)]
                candidate_id = raw_label_to_concept_id[candidate_label]
                similarity = float(all_similarities[index, int(position)])
                if candidate_id == primary_id:
                    continue
                if similarity < self.secondary_similarity or top_similarity - similarity > self.secondary_margin:
                    continue
                assignments.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "concept_id": candidate_id,
                        "primary": False,
                        "confidence": round(similarity, 6),
                        "semantic_similarity": round(similarity, 6),
                        "cluster_probability": 0.0,
                        "method": "centroid_secondary",
                        "run_id": run_id,
                    }
                )
                assigned_count += 1
                if assigned_count >= self.max_concepts_per_chunk:
                    break

        mentions = []
        for chunk in chunks:
            source_text = chunk.get("text") or ""
            retrieval_text = chunk.get("retrieval_text") or source_text
            for term in terms:
                source_count = count_term_occurrences(source_text, term["text"], term["match_type"])
                retrieval_count = count_term_occurrences(retrieval_text, term["text"], term["match_type"])
                if retrieval_count <= 0:
                    continue
                mentions.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "term_id": term["term_id"],
                        "concept_id": term["concept_id"],
                        "matched_text": term["text"],
                        "count": retrieval_count,
                        "source_count": source_count,
                        "match_scope": "source_text" if source_count > 0 else "section_context",
                        "method": "automatic_exact_term",
                        "match_type": term["match_type"],
                        "run_id": run_id,
                    }
                )

        chunk_embedding_rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": embeddings[index].astype(float).tolist(),
                "embedding_model": self.embedding_model,
                "run_id": run_id,
            }
            for index, chunk in enumerate(chunks)
        ]
        noise_count = sum(1 for label in labels if int(label) < 0)
        run = {
            "run_id": run_id,
            "algorithm": ALGORITHM_VERSION,
            "embedding_model": self.embedding_model,
            "resolver_model": self.resolver_model,
            "embedding_dimensions": embedding_dimensions,
            "corpus_hash": corpus_hash,
            "chunk_count": len(chunks),
            "source_document_count": len({chunk["source_file"] for chunk in chunks}),
            "leaf_concept_count": len(leaf_concepts),
            "parent_concept_count": len(parent_concepts),
            "term_count": len(terms),
            "clustering_method": clustering_method,
            "noise_chunk_count_before_reassignment": noise_count,
            "hierarchy_silhouette": round(float(hierarchy_score), 6) if hierarchy_score is not None else None,
            "settings_json": settings_json,
            "embedding_cache_hit": cache_hit,
        }

        return {
            "run": run,
            "concepts": [root_concept, *parent_concepts, *leaf_concepts],
            "terms": terms,
            "taxonomy": taxonomy,
            "assignments": assignments,
            "representatives": representatives,
            "mentions": mentions,
            "chunk_embeddings": chunk_embedding_rows,
        }
