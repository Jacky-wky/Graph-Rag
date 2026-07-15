"""Extract reviewable, evidence-backed facts from Neo4j chunks.

The script never creates direct semantic relationships between concepts. Instead,
it stores each extracted statement as a :Fact node and attaches the exact source
chunk through :SUPPORTS. A fact is therefore traceable to Document -> Page ->
Chunk before it is approved for downstream use.
"""

import argparse
import hashlib
import json
import os
from itertools import combinations
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ALLOWED_PREDICATES = [
    "REQUIRES",
    "RESPONSIBLE_FOR",
    "MITIGATES",
    "APPLIES_TO",
    "PROHIBITS",
    "REPORTS_TO",
    "MONITORS",
    "RELATED_TO",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fact_id(subject_id: str, predicate: str, object_id: str) -> str:
    value = f"{subject_id}|{predicate}|{object_id}".lower()
    return "fact_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


class ProvenanceFactExtractor:
    def __init__(self, uri: str, user: str, password: str, database: str | None, model: str, mode: str):
        self.database = database
        self.model = model
        self.mode = mode
        self.client = OpenAI() if mode == "llm" else None
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def session(self):
        return self.driver.session(database=self.database)

    def create_constraints(self) -> None:
        with self.session() as session:
            session.run(
                "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS "
                "FOR (fact:Fact) REQUIRE fact.fact_id IS UNIQUE"
            )

    def fetch_concepts(self) -> list[dict[str, str]]:
        with self.session() as session:
            query = """
            MATCH (concept:Concept)
            RETURN concept.concept_id AS concept_id, concept.name AS name,
                   concept.description AS description
            ORDER BY concept.concept_id
            """
            return [dict(record) for record in session.run(query)]

    def fetch_chunks(self, limit: int) -> list[dict[str, Any]]:
        with self.session() as session:
            query = """
            MATCH (chunk:Chunk)-[:MENTIONS_CONCEPT]->(concept:Concept)
            WHERE NOT EXISTS {
                MATCH (chunk)-[:SUPPORTS]->(:Fact)
            }
            WITH chunk, collect(DISTINCT concept.concept_id) AS concept_ids
            RETURN chunk.chunk_id AS chunk_id, chunk.text AS text,
                   chunk.source_file AS source_file, chunk.page AS page,
                   chunk.chunk_index AS chunk_index, concept_ids
            ORDER BY source_file, page, chunk_index
            LIMIT $limit
            """
            return [dict(record) for record in session.run(query, limit=limit)]

    def extract_with_llm(self, chunk_text: str, concepts: list[dict[str, str]]) -> list[dict[str, Any]]:
        concept_catalog = [
            {"concept_id": item["concept_id"], "name": item["name"], "description": item.get("description")}
            for item in concepts
        ]
        prompt = f"""Extract explicit domain facts from the source chunk below.

Only use concept IDs in the supplied catalog. A fact must be directly stated,
not inferred. Use one predicate from: {', '.join(ALLOWED_PREDICATES)}.
The evidence_quote must be an exact, non-empty substring of the source chunk.
Return JSON only: {{"facts": [{{"subject_concept_id": "...", "predicate": "...",
"object_concept_id": "...", "evidence_quote": "...", "confidence": 0.0}}]}}.
Return an empty facts list when there is no supported relation.

Concept catalog:
{json.dumps(concept_catalog, ensure_ascii=False)}

Source chunk:
{chunk_text}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return payload.get("facts", [])

    @staticmethod
    def extract_with_rules(chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Create conservative co-occurrence candidates without calling an LLM.

        These candidates are intentionally limited to RELATED_TO and retain the
        source text as evidence. They must be reviewed before becoming trusted
        knowledge because co-occurrence does not prove a specific relation.
        """
        concept_ids = sorted(set(chunk.get("concept_ids", [])))
        quote = (chunk.get("text") or "").strip()[:500]
        if not quote:
            return []
        return [
            {
                "subject_concept_id": subject_id,
                "predicate": "RELATED_TO",
                "object_concept_id": object_id,
                "evidence_quote": quote,
                "confidence": 0.35,
            }
            for subject_id, object_id in combinations(concept_ids, 2)
        ]

    @staticmethod
    def validate(fact: dict[str, Any], chunk_text: str, concept_ids: set[str]) -> bool:
        quote = str(fact.get("evidence_quote", "")).strip()
        confidence = fact.get("confidence", 0)
        return (
            fact.get("subject_concept_id") in concept_ids
            and fact.get("object_concept_id") in concept_ids
            and fact.get("predicate") in ALLOWED_PREDICATES
            and quote in chunk_text
            and isinstance(confidence, (int, float))
            and 0 <= confidence <= 1
        )

    def write_facts(self, chunk_id: str, facts: list[dict[str, Any]], auto_approve: bool) -> int:
        rows = []
        for item in facts:
            rows.append(
                {
                    "fact_id": fact_id(item["subject_concept_id"], item["predicate"], item["object_concept_id"]),
                    "subject_id": item["subject_concept_id"],
                    "predicate": item["predicate"],
                    "object_id": item["object_concept_id"],
                    "quote": item["evidence_quote"].strip(),
                    "confidence": float(item["confidence"]),
                "status": "approved" if auto_approve else "pending",
                "extractor": f"{self.mode}_fact_extractor",
                }
            )
        if not rows:
            return 0

        query = """
        UNWIND $rows AS row
        MATCH (chunk:Chunk {chunk_id: $chunk_id})
        MATCH (subject:Concept {concept_id: row.subject_id})
        MATCH (object:Concept {concept_id: row.object_id})
        MERGE (fact:Fact {fact_id: row.fact_id})
        ON CREATE SET fact.created_at = $now, fact.status = row.status
        SET fact.predicate = row.predicate,
            fact.extractor = row.extractor,
            fact.model = $model,
            fact.updated_at = $now
        MERGE (subject)-[:SUBJECT_OF]->(fact)
        MERGE (fact)-[:OBJECT_OF]->(object)
        MERGE (chunk)-[support:SUPPORTS {extractor: row.extractor, quote: row.quote}]->(fact)
        SET support.confidence = row.confidence,
            support.model = $model,
            support.updated_at = $now
        SET support.created_at = coalesce(support.created_at, $now)
        """
        with self.session() as session:
            session.run(query, chunk_id=chunk_id, rows=rows, now=utc_now_iso(), model=self.model)
        return len(rows)

    def run(self, limit: int, auto_approve: bool) -> None:
        self.create_constraints()
        concepts = self.fetch_concepts()
        if not concepts:
            raise ValueError("No Concept nodes found. Run improve_relationships.py first.")
        concept_ids = {item["concept_id"] for item in concepts}
        chunks = self.fetch_chunks(limit)
        print(f"Processing {len(chunks)} chunks with {len(concepts)} concepts.")
        created = 0
        for index, chunk in enumerate(chunks, start=1):
            extracted = (
                self.extract_with_llm(chunk["text"], concepts)
                if self.mode == "llm"
                else self.extract_with_rules(chunk)
            )
            valid_facts = [item for item in extracted if self.validate(item, chunk["text"], concept_ids)]
            created += self.write_facts(chunk["chunk_id"], valid_facts, auto_approve)
            print(f"Processed {index}/{len(chunks)} chunks; accepted fact-evidence records: {created}")
        print(f"Done. Created or updated {created} fact-evidence records.")


def main() -> None:
    if load_dotenv:
        load_dotenv()
    parser = argparse.ArgumentParser(description="Extract evidence-backed candidate facts into Neo4j.")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE") or None)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--mode",
        choices=["rules", "llm"],
        default=os.getenv("FACT_EXTRACTION_MODE", "rules"),
        help="Extraction method. rules is local and free; llm requires OPENAI_API_KEY.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--auto-approve", action="store_true", help="Mark created facts as approved; default is pending review.")
    args = parser.parse_args()
    if not args.password:
        raise ValueError("Set NEO4J_PASSWORD before running this command.")
    if args.mode == "llm" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY before running this command.")
    extractor = ProvenanceFactExtractor(args.uri, args.user, args.password, args.database, args.model, args.mode)
    try:
        extractor.run(args.limit, args.auto_approve)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
