"""Deterministic and auditable question-to-concept resolution.

The resolver uses two local stages and never calls an LLM:

1. Exact matching against registered Term aliases.
2. Conservative fuzzy matching for spelling and punctuation variation.

Evidence-based fallback is handled by ``local_graph_rag.py`` because it needs
access to Chunk nodes. Every returned match records its method and confidence.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable


@dataclass(frozen=True)
class TermMatch:
    concept_id: str
    concept_name: str
    term_id: str
    term_text: str
    match_type: str
    method: str
    confidence: float


@dataclass(frozen=True)
class ResolvedConcept:
    concept_id: str
    concept_name: str
    method: str
    confidence: float
    matched_terms: tuple[TermMatch, ...]


@dataclass(frozen=True)
class ResolutionResult:
    status: str
    normalized_question: str
    concepts: tuple[ResolvedConcept, ...]
    explanation: str


INTENT_KEYWORDS = {
    "control_requirements": (
        "control",
        "controls",
        "requirement",
        "requirements",
        "required",
        "must",
        "should",
        "控制",
        "措施",
        "要求",
        "規定",
        "必須",
        "應當",
        "須要",
    ),
    "comparison": ("compare", "comparison", "difference", "versus", " vs ", "比較", "分別", "差異", "區別"),
    "definition": ("what is", "define", "definition", "meaning", "甚麼是", "什麼是", "定義", "意思"),
    "procedure": ("how to", "process", "procedure", "steps", "如何", "程序", "流程", "步驟"),
}

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "control",
    "controls",
    "for",
    "how",
    "is",
    "must",
    "of",
    "required",
    "requirement",
    "requirements",
    "should",
    "the",
    "to",
    "what",
    "which",
    "有哪些",
    "如何",
    "甚麼",
    "什麼",
    "控制",
    "措施",
    "要求",
    "規定",
}

ORTHOGRAPHIC_VARIANTS = {
    "身份": "身分",
    "帳戶": "戶口",
}


def normalize_text(value: str) -> str:
    """Normalize case, Unicode width, punctuation, and whitespace."""
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    characters = []
    for character in normalized:
        category = unicodedata.category(character)
        characters.append(" " if category.startswith("P") else character)
    result = re.sub(r"\s+", " ", "".join(characters)).strip()
    for variant, canonical in ORTHOGRAPHIC_VARIANTS.items():
        result = result.replace(variant, canonical)
    return result


def _contains_word(question: str, term: str) -> bool:
    escaped = re.escape(term)
    return re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", question) is not None


def term_matches_question(question: str, term_text: str, match_type: str) -> bool:
    normalized_term = normalize_text(term_text)
    if not normalized_term:
        return False
    if match_type == "word":
        return _contains_word(question, normalized_term)
    return normalized_term in question


def _latin_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value)


def _fuzzy_similarity(question: str, term: str) -> float:
    """Compare a term with similarly sized windows from the question."""
    term_tokens = _latin_tokens(term)
    question_tokens = _latin_tokens(question)
    if term_tokens and question_tokens:
        window_size = len(term_tokens)
        if len(question_tokens) < window_size:
            return SequenceMatcher(None, " ".join(question_tokens), " ".join(term_tokens)).ratio()
        return max(
            SequenceMatcher(None, " ".join(question_tokens[index : index + window_size]), " ".join(term_tokens)).ratio()
            for index in range(len(question_tokens) - window_size + 1)
        )

    compact_question = question.replace(" ", "")
    compact_term = term.replace(" ", "")
    if len(compact_term) < 4:
        return 0.0
    window_size = len(compact_term)
    if len(compact_question) < window_size:
        return SequenceMatcher(None, compact_question, compact_term).ratio()
    return max(
        SequenceMatcher(None, compact_question[index : index + window_size], compact_term).ratio()
        for index in range(len(compact_question) - window_size + 1)
    )


def _group_matches(matches: Iterable[TermMatch]) -> tuple[ResolvedConcept, ...]:
    matches_by_concept: dict[str, list[TermMatch]] = defaultdict(list)
    for match in matches:
        matches_by_concept[match.concept_id].append(match)

    concepts = []
    for concept_id, concept_matches in matches_by_concept.items():
        ordered_matches = tuple(
            sorted(concept_matches, key=lambda item: (-item.confidence, -len(normalize_text(item.term_text)), item.term_id))
        )
        best = ordered_matches[0]
        confidence = min(1.0, best.confidence + 0.02 * (len(ordered_matches) - 1))
        concepts.append(
            ResolvedConcept(
                concept_id=concept_id,
                concept_name=best.concept_name,
                method=best.method,
                confidence=round(confidence, 4),
                matched_terms=ordered_matches,
            )
        )
    return tuple(sorted(concepts, key=lambda item: (-item.confidence, item.concept_id)))


def resolve_question(
    question: str,
    terms: Iterable[dict],
    fuzzy_threshold: float = 0.88,
    ambiguity_margin: float = 0.05,
) -> ResolutionResult:
    """Resolve registered Concepts using exact aliases, then conservative fuzzy aliases."""
    normalized_question = normalize_text(question)
    term_rows = list(terms)
    exact_matches = []

    for term in term_rows:
        term_text = term.get("term_text") or ""
        match_type = term.get("match_type") or "phrase"
        if term_matches_question(normalized_question, term_text, match_type):
            exact_matches.append(
                TermMatch(
                    concept_id=term["concept_id"],
                    concept_name=term["concept_name"],
                    term_id=term["term_id"],
                    term_text=term_text,
                    match_type=match_type,
                    method="exact_term",
                    confidence=1.0 if match_type == "phrase" else 0.98,
                )
            )

    if exact_matches:
        concepts = _group_matches(exact_matches)
        return ResolutionResult(
            status="resolved",
            normalized_question=normalized_question,
            concepts=concepts,
            explanation="One or more registered terms occur in the normalized question.",
        )

    fuzzy_matches = []
    for term in term_rows:
        normalized_term = normalize_text(term.get("term_text") or "")
        if len(normalized_term.replace(" ", "")) < 4:
            continue
        similarity = _fuzzy_similarity(normalized_question, normalized_term)
        if similarity >= fuzzy_threshold:
            fuzzy_matches.append(
                TermMatch(
                    concept_id=term["concept_id"],
                    concept_name=term["concept_name"],
                    term_id=term["term_id"],
                    term_text=term["term_text"],
                    match_type=term.get("match_type") or "phrase",
                    method="fuzzy_term",
                    confidence=round(similarity, 4),
                )
            )

    concepts = _group_matches(fuzzy_matches)
    if not concepts:
        return ResolutionResult(
            status="unresolved",
            normalized_question=normalized_question,
            concepts=(),
            explanation="No registered term met the exact or fuzzy matching threshold.",
        )

    if len(concepts) > 1 and concepts[0].confidence - concepts[1].confidence < ambiguity_margin:
        return ResolutionResult(
            status="ambiguous",
            normalized_question=normalized_question,
            concepts=concepts,
            explanation="Multiple fuzzy concept candidates have similar confidence.",
        )

    return ResolutionResult(
        status="resolved",
        normalized_question=normalized_question,
        concepts=(concepts[0],),
        explanation="A registered term matched after conservative spelling normalization.",
    )


def classify_intent(question: str) -> str:
    """Classify retrieval intent; intent never changes Concept identity."""
    normalized = f" {normalize_text(question)} "
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(normalize_text(keyword) in normalized for keyword in keywords):
            return intent
    return "general_information"


def extract_query_keywords(question: str) -> list[str]:
    """Build deterministic lexical fallback terms from an unresolved question."""
    normalized = normalize_text(question)
    english_stopwords = {word for word in QUERY_STOPWORDS if re.fullmatch(r"[a-z ]+", word)}
    chinese_stopwords = QUERY_STOPWORDS - english_stopwords
    tokens = []
    for token in normalized.split():
        if token in english_stopwords:
            continue
        for phrase in sorted(chinese_stopwords, key=len, reverse=True):
            token = token.replace(phrase, " ")
        tokens.extend(part for part in token.split() if part)
    keywords = []
    for token in tokens:
        if len(token) < 2:
            continue
        keywords.append(token)
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", token) and len(token) >= 4:
            keywords.extend(token[index : index + 2] for index in range(len(token) - 1))
    return list(dict.fromkeys(keywords))
