"""Layered local Graph RAG retrieval without an LLM API."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from neo4j import GraphDatabase

from question_to_concept import classify_intent, extract_query_keywords, normalize_text, resolve_question

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


CONTROL_KEYWORDS = ("control", "requirement", "required", "must", "should", "控制", "措施", "要求", "規定", "必須")


class LocalGraphRetriever:
    def __init__(self, uri: str, user: str, password: str, database: str | None):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(query, **parameters)]

    def fetch_terms(self) -> list[dict[str, Any]]:
        query = """
        MATCH (term:Term)-[:NORMALIZED_TO]->(concept:Concept)
        RETURN term.term_id AS term_id, term.text AS term_text,
               term.match_type AS match_type, concept.concept_id AS concept_id,
               concept.name AS concept_name
        """
        return self.run(query)

    def fetch_concept_chunks(self, concept_ids: list[str], candidate_limit: int) -> list[dict[str, Any]]:
        query = """
        UNWIND $concept_ids AS concept_id
        MATCH (concept:Concept {concept_id: concept_id})<-[:MENTIONS_CONCEPT]-(chunk:Chunk)
        WITH chunk, collect(DISTINCT concept) AS concepts
        RETURN chunk.chunk_id AS chunk_id, chunk.source_file AS source_file,
               chunk.page AS page, chunk.text AS text,
               [concept IN concepts | {concept_id: concept.concept_id, name: concept.name}] AS concepts,
               0 AS lexical_score
        ORDER BY source_file, page, chunk_id
        LIMIT $candidate_limit
        """
        return self.run(query, concept_ids=concept_ids, candidate_limit=candidate_limit)

    def fetch_lexical_chunks(self, keywords: list[str], candidate_limit: int) -> list[dict[str, Any]]:
        if not keywords:
            return []
        query = """
        MATCH (chunk:Chunk)
        WHERE any(keyword IN $keywords WHERE toLower(chunk.text) CONTAINS keyword)
        OPTIONAL MATCH (chunk)-[:MENTIONS_CONCEPT]->(concept:Concept)
        WITH chunk, collect(DISTINCT concept) AS concepts,
             reduce(score = 0, keyword IN $keywords |
                 score + CASE WHEN toLower(chunk.text) CONTAINS keyword THEN 1 ELSE 0 END
             ) AS lexical_score
        RETURN chunk.chunk_id AS chunk_id, chunk.source_file AS source_file,
               chunk.page AS page, chunk.text AS text,
               [concept IN concepts WHERE concept IS NOT NULL |
                   {concept_id: concept.concept_id, name: concept.name}
               ] AS concepts,
               lexical_score
        ORDER BY lexical_score DESC, source_file, page, chunk_id
        LIMIT $candidate_limit
        """
        return self.run(query, keywords=keywords, candidate_limit=candidate_limit)

    def fetch_approved_facts(self, concept_ids: list[str], limit: int) -> list[dict[str, Any]]:
        if not concept_ids:
            return []
        query = """
        MATCH (subject:Concept)-[:SUBJECT_OF]->(fact:Fact {status: "approved"})-[:OBJECT_OF]->(object:Concept)
        WHERE subject.concept_id IN $concept_ids OR object.concept_id IN $concept_ids
        OPTIONAL MATCH (chunk:Chunk)-[support:SUPPORTS]->(fact)
        WITH subject, fact, object, collect({
            quote: support.quote, source_file: chunk.source_file, page: chunk.page,
            confidence: support.confidence
        })[0..3] AS evidence
        RETURN fact.fact_id AS fact_id, subject.concept_id AS subject_id,
               subject.name AS subject, fact.predicate AS predicate,
               object.concept_id AS object_id, object.name AS object, evidence
        ORDER BY fact_id
        LIMIT $limit
        """
        return self.run(query, concept_ids=concept_ids, limit=limit)


def rank_chunks(chunks: list[dict[str, Any]], matched_terms: list[str], intent: str) -> list[dict[str, Any]]:
    def score(chunk: dict[str, Any]) -> tuple[int, str, int, str]:
        text = normalize_text(chunk.get("text") or "")
        relevance = 60 * int(chunk.get("lexical_score") or 0)
        relevance += sum(100 * text.count(normalize_text(term)) for term in matched_terms)
        if intent == "control_requirements":
            relevance += sum(10 * text.count(keyword) for keyword in CONTROL_KEYWORDS)
        if "目錄" in text[:200]:
            relevance -= 1000
        return (-relevance, chunk.get("source_file") or "", chunk.get("page") or 0, chunk.get("chunk_id") or "")

    return sorted(chunks, key=score)


def infer_concept_candidates(chunks: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    votes: Counter[str] = Counter()
    names: dict[str, str] = {}
    for chunk in chunks:
        weight = max(1, int(chunk.get("lexical_score") or 1))
        for concept in chunk.get("concepts", []):
            votes[concept["concept_id"]] += weight
            names[concept["concept_id"]] = concept["name"]
    total = sum(votes.values()) or 1
    return [
        {
            "concept_id": concept_id,
            "concept_name": names[concept_id],
            "method": "chunk_evidence_vote",
            "vote": vote,
            "confidence": round(vote / total, 4),
            "trusted_mapping": False,
        }
        for concept_id, vote in votes.most_common(limit)
    ]


def build_result(retriever: LocalGraphRetriever, question: str, chunk_limit: int, fact_limit: int) -> dict[str, Any]:
    intent = classify_intent(question)
    resolution = resolve_question(question, retriever.fetch_terms())
    resolved_concepts = [
        {
            "concept_id": concept.concept_id,
            "concept_name": concept.concept_name,
            "method": concept.method,
            "confidence": concept.confidence,
            "matched_terms": [match.term_text for match in concept.matched_terms],
        }
        for concept in resolution.concepts
    ]
    concept_ids = [concept["concept_id"] for concept in resolved_concepts] if resolution.status == "resolved" else []
    matched_terms = [term for concept in resolved_concepts for term in concept["matched_terms"]]
    keywords = []
    candidate_concepts = []

    if concept_ids:
        candidates = retriever.fetch_concept_chunks(concept_ids, max(chunk_limit * 25, 100))
        retrieval_method = "concept_graph"
    else:
        keywords = extract_query_keywords(question)
        candidates = retriever.fetch_lexical_chunks(keywords, max(chunk_limit * 25, 100))
        candidate_concepts = infer_concept_candidates(candidates)
        retrieval_method = "lexical_chunk_fallback"

    chunks = rank_chunks(candidates, matched_terms or keywords, intent)[:chunk_limit]
    return {
        "question": question,
        "intent": intent,
        "resolution": {
            "status": resolution.status,
            "normalized_question": resolution.normalized_question,
            "explanation": resolution.explanation,
            "resolved_concepts": resolved_concepts,
            "candidate_concepts": candidate_concepts,
        },
        "retrieval_method": retrieval_method,
        "fallback_keywords": keywords,
        "approved_facts": retriever.fetch_approved_facts(concept_ids, fact_limit),
        "chunks": chunks,
    }


def print_result(result: dict[str, Any]) -> None:
    resolution = result["resolution"]
    print(f"Question: {result['question']}")
    print(f"Intent: {result['intent']}")
    print(f"Resolution status: {resolution['status']}")
    print(f"Resolution explanation: {resolution['explanation']}")
    print(f"Retrieval method: {result['retrieval_method']}")
    if resolution["resolved_concepts"]:
        print("Resolved concepts:")
        for concept in resolution["resolved_concepts"]:
            print(
                f"- {concept['concept_id']} ({concept['concept_name']}), "
                f"method={concept['method']}, confidence={concept['confidence']}, "
                f"terms={concept['matched_terms']}"
            )
    if resolution["candidate_concepts"]:
        print("Candidate concepts from retrieved evidence (not trusted mappings):")
        for concept in resolution["candidate_concepts"]:
            print(
                f"- {concept['concept_id']} ({concept['concept_name']}), "
                f"vote={concept['vote']}, confidence={concept['confidence']}"
            )
    if result["fallback_keywords"]:
        print(f"Fallback keywords: {result['fallback_keywords']}")
    print("Approved facts:")
    if not result["approved_facts"]:
        print("- none")
    for fact in result["approved_facts"]:
        print(f"- {fact['subject']} --{fact['predicate']}--> {fact['object']} ({fact['fact_id']})")
        for evidence in fact["evidence"]:
            print(f"  [{evidence['source_file']} p.{evidence['page']}] {evidence['quote']}")
    print("Supporting chunks:")
    if not result["chunks"]:
        print("- none")
    for chunk in result["chunks"]:
        print(f"- [{chunk['source_file']} p.{chunk['page']}] {chunk['chunk_id']}")
        print(f"  {chunk['text'][:800]}")


def main() -> None:
    if load_dotenv:
        load_dotenv(".env")
    parser = argparse.ArgumentParser(description="Retrieve Neo4j evidence without an LLM API.")
    parser.add_argument("question", nargs="+", help="Question to resolve and retrieve")
    parser.add_argument("--chunk-limit", type=int, default=8)
    parser.add_argument("--fact-limit", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE") or None)
    args = parser.parse_args()
    if not args.password:
        raise ValueError("Set NEO4J_PASSWORD before running this command.")
    retriever = LocalGraphRetriever(args.uri, args.user, args.password, args.database)
    try:
        result = build_result(retriever, " ".join(args.question), args.chunk_limit, args.fact_limit)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
