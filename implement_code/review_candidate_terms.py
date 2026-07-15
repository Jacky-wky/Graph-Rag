# review_candidate_terms.py
# -*- coding: utf-8 -*-

"""
Simple command-line human review for CandidateTerm.

Supports:

1. list pending candidates
2. approve as new Concept
3. merge into existing Concept
4. reject
5. show suggestions
6. rerun rule matching for newly approved/merged Term

Usage:

    python review_candidate_terms.py list
    python review_candidate_terms.py suggestions "風險為本方法"

    python review_candidate_terms.py reject "有關規定"

    python review_candidate_terms.py merge "盡職審查" cdd \
        --term-id term_due_diligence_zh \
        --language zh \
        --match-type phrase

    python review_candidate_terms.py approve "風險為本方法" \
        --concept-id risk_based_approach \
        --name "風險為本方法" \
        --category aml_cft \
        --description "根據風險程度調整管控措施的方法。" \
        --level 2 \
        --parent-concept-id aml_cft \
        --term-id term_risk_based_approach_zh \
        --language zh \
        --match-type phrase

Environment variables:

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
"""

import argparse
import os
from datetime import datetime, timezone

from neo4j import GraphDatabase


DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CandidateReviewer:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def list_pending(self, limit: int = 50):
        query = """
        MATCH (ct:CandidateTerm)
        WHERE ct.status = "pending"
        OPTIONAL MATCH (ct)-[s:SUGGESTED_FOR]->(c:Concept)
        WITH ct, c, s
        ORDER BY s.score DESC
        WITH ct, collect({
            concept_id: c.concept_id,
            name: c.name,
            score: s.score
        })[0..3] AS suggestions
        RETURN
            ct.text AS text,
            ct.frequency AS frequency,
            ct.document_count AS document_count,
            ct.chunk_count AS chunk_count,
            ct.score AS score,
            suggestions AS suggestions
        ORDER BY ct.document_count DESC, ct.frequency DESC, ct.score DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            rows = list(session.run(query, limit=limit))

        if not rows:
            print("No pending candidates.")
            return

        for idx, r in enumerate(rows, start=1):
            print("=" * 80)
            print(f"{idx}. {r['text']}")
            print(f"frequency={r['frequency']} document_count={r['document_count']} chunk_count={r['chunk_count']} score={r['score']}")
            suggestions = [s for s in r["suggestions"] if s.get("concept_id")]
            if suggestions:
                print("suggestions:")
                for s in suggestions:
                    print(f"  - {s['concept_id']} | {s['name']} | score={s['score']:.3f}")
            else:
                print("suggestions: none")

    def show_suggestions(self, text: str):
        query = """
        MATCH (ct:CandidateTerm {text: $text})
        OPTIONAL MATCH (ct)-[s:SUGGESTED_FOR]->(c:Concept)
        RETURN
            ct.text AS text,
            ct.status AS status,
            ct.frequency AS frequency,
            ct.document_count AS document_count,
            c.concept_id AS concept_id,
            c.name AS concept_name,
            s.score AS score,
            s.method AS method
        ORDER BY score DESC
        """

        with self.driver.session() as session:
            rows = list(session.run(query, text=text))

        if not rows:
            print(f"No CandidateTerm found for: {text}")
            return

        first = rows[0]
        print(f"Candidate: {first['text']}")
        print(f"status={first['status']} frequency={first['frequency']} document_count={first['document_count']}")
        print("Suggestions:")

        found = False
        for r in rows:
            if r["concept_id"]:
                found = True
                print(f"  - {r['concept_id']} | {r['concept_name']} | score={r['score']} | method={r['method']}")

        if not found:
            print("  none")

    def reject(self, text: str):
        query = """
        MATCH (ct:CandidateTerm {text: $text})
        SET ct.status = "rejected",
            ct.reviewed_at = $now,
            ct.updated_at = $now
        RETURN ct.text AS text, ct.status AS status
        """

        with self.driver.session() as session:
            row = session.run(query, text=text, now=utc_now_iso()).single()

        if row:
            print(f"Rejected: {row['text']}")
        else:
            print(f"No CandidateTerm found for: {text}")

    def merge(
        self,
        text: str,
        concept_id: str,
        term_id: str,
        language: str,
        match_type: str,
    ):
        query = """
        MATCH (ct:CandidateTerm {text: $text})
        MATCH (c:Concept {concept_id: $concept_id})
        SET ct.status = "merged",
            ct.normalized_to = $concept_id,
            ct.reviewed_at = $now,
            ct.updated_at = $now

        MERGE (t:Term {term_id: $term_id})
        SET t.text = $text,
            t.normalized_text = toLower($text),
            t.language = $language,
            t.match_type = $match_type,
            t.source = "human_review",
            t.concept_id = $concept_id,
            t.updated_at = $now
        SET t.created_at = coalesce(t.created_at, $now)

        MERGE (t)-[r:NORMALIZED_TO]->(c)
        SET r.method = "human_review",
            r.updated_at = $now
        SET r.created_at = coalesce(r.created_at, $now)

        RETURN ct.text AS text, c.concept_id AS concept_id, c.name AS concept_name, t.term_id AS term_id
        """

        params = {
            "text": text,
            "concept_id": concept_id,
            "term_id": term_id,
            "language": language,
            "match_type": match_type,
            "now": utc_now_iso(),
        }

        with self.driver.session() as session:
            row = session.run(query, **params).single()

        if row:
            print(f"Merged CandidateTerm '{row['text']}' into Concept '{row['concept_id']} | {row['concept_name']}'")
            print(f"Created/updated Term: {row['term_id']}")
        else:
            print("Merge failed. Check CandidateTerm text and Concept ID.")

    def approve(
        self,
        text: str,
        concept_id: str,
        name: str,
        category: str,
        description: str,
        level: int,
        parent_concept_id: str,
        term_id: str,
        language: str,
        match_type: str,
    ):
        query = """
        MATCH (ct:CandidateTerm {text: $text})

        MERGE (c:Concept {concept_id: $concept_id})
        SET c.name = $name,
            c.category = $category,
            c.description = $description,
            c.level = $level,
            c.source = "human_review",
            c.updated_at = $now
        SET c.created_at = coalesce(c.created_at, $now)

        SET ct.status = "approved",
            ct.normalized_to = $concept_id,
            ct.reviewed_at = $now,
            ct.updated_at = $now

        MERGE (t:Term {term_id: $term_id})
        SET t.text = $text,
            t.normalized_text = toLower($text),
            t.language = $language,
            t.match_type = $match_type,
            t.source = "human_review",
            t.concept_id = $concept_id,
            t.updated_at = $now
        SET t.created_at = coalesce(t.created_at, $now)

        MERGE (t)-[r:NORMALIZED_TO]->(c)
        SET r.method = "human_review",
            r.updated_at = $now
        SET r.created_at = coalesce(r.created_at, $now)

        WITH ct, c, t
        OPTIONAL MATCH (parent:Concept {concept_id: $parent_concept_id})
        FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END |
            MERGE (parent)-[:HAS_SUBCONCEPT]->(c)
        )

        RETURN ct.text AS text, c.concept_id AS concept_id, c.name AS concept_name, t.term_id AS term_id
        """

        params = {
            "text": text,
            "concept_id": concept_id,
            "name": name,
            "category": category,
            "description": description,
            "level": level,
            "parent_concept_id": parent_concept_id,
            "term_id": term_id,
            "language": language,
            "match_type": match_type,
            "now": utc_now_iso(),
        }

        with self.driver.session() as session:
            row = session.run(query, **params).single()

        if row:
            print(f"Approved CandidateTerm '{row['text']}' as new Concept '{row['concept_id']} | {row['concept_name']}'")
            print(f"Created/updated Term: {row['term_id']}")
            if parent_concept_id:
                print(f"Linked parent Concept if found: {parent_concept_id}")
        else:
            print("Approve failed. Check CandidateTerm text.")

    def list_concepts(self, limit: int = 200):
        query = """
        MATCH (c:Concept)
        RETURN c.concept_id AS concept_id, c.name AS name, c.category AS category, c.level AS level
        ORDER BY c.level, c.category, c.name
        LIMIT $limit
        """

        with self.driver.session() as session:
            rows = list(session.run(query, limit=limit))

        for r in rows:
            print(f"{r['concept_id']:30s} | level={r['level']} | {r['category']:15s} | {r['name']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=DEFAULT_NEO4J_URI)
    parser.add_argument("--user", default=DEFAULT_NEO4J_USER)
    parser.add_argument("--password", default=DEFAULT_NEO4J_PASSWORD)

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list")
    p_list.add_argument("--limit", type=int, default=50)

    p_concepts = subparsers.add_parser("concepts")
    p_concepts.add_argument("--limit", type=int, default=200)

    p_suggestions = subparsers.add_parser("suggestions")
    p_suggestions.add_argument("text")

    p_reject = subparsers.add_parser("reject")
    p_reject.add_argument("text")

    p_merge = subparsers.add_parser("merge")
    p_merge.add_argument("text")
    p_merge.add_argument("concept_id")
    p_merge.add_argument("--term-id", required=True)
    p_merge.add_argument("--language", default="zh")
    p_merge.add_argument("--match-type", default="phrase", choices=["phrase", "word"])

    p_approve = subparsers.add_parser("approve")
    p_approve.add_argument("text")
    p_approve.add_argument("--concept-id", required=True)
    p_approve.add_argument("--name", required=True)
    p_approve.add_argument("--category", required=True)
    p_approve.add_argument("--description", required=True)
    p_approve.add_argument("--level", type=int, default=2)
    p_approve.add_argument("--parent-concept-id", default="")
    p_approve.add_argument("--term-id", required=True)
    p_approve.add_argument("--language", default="zh")
    p_approve.add_argument("--match-type", default="phrase", choices=["phrase", "word"])

    return parser.parse_args()


def main():
    args = parse_args()

    reviewer = CandidateReviewer(args.uri, args.user, args.password)

    try:
        if args.command == "list":
            reviewer.list_pending(limit=args.limit)

        elif args.command == "concepts":
            reviewer.list_concepts(limit=args.limit)

        elif args.command == "suggestions":
            reviewer.show_suggestions(args.text)

        elif args.command == "reject":
            reviewer.reject(args.text)

        elif args.command == "merge":
            reviewer.merge(
                text=args.text,
                concept_id=args.concept_id,
                term_id=args.term_id,
                language=args.language,
                match_type=args.match_type,
            )

        elif args.command == "approve":
            reviewer.approve(
                text=args.text,
                concept_id=args.concept_id,
                name=args.name,
                category=args.category,
                description=args.description,
                level=args.level,
                parent_concept_id=args.parent_concept_id,
                term_id=args.term_id,
                language=args.language,
                match_type=args.match_type,
            )

    finally:
        reviewer.close()


if __name__ == "__main__":
    main()