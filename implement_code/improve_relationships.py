# improve_relationships.py
# -*- coding: utf-8 -*-

"""
Build rule-based Concept / Term / Taxonomy relationships in Neo4j.

Creates:

(:Concept)
(:Term)
(:Term)-[:NORMALIZED_TO]->(:Concept)
(:Concept)-[:HAS_SUBCONCEPT]->(:Concept)
(:Chunk)-[:MENTIONS_TERM]->(:Term)

Optional:
(:Chunk)-[:MENTIONS_CONCEPT]->(:Concept)

The direct MENTIONS_CONCEPT relationship is derived from Chunk -> Term -> Concept.
It is useful for simpler retrieval, but the source of truth remains:
Chunk -> Term -> Concept.

Expected existing Chunk nodes:

(:Chunk {
    chunk_id: "...",
    text: "...",
    source_file: "...",
    page: ...,
    chunk_index: ...
})

Usage:

    python improve_relationships.py

Environment variables:

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=password

Also supported:

    NEO4J_USERNAME=neo4j
    NEO4J_DATABASE=neo4j

Optional flags:

    python improve_relationships.py --clear-old
    python improve_relationships.py --no-direct-concept
    python improve_relationships.py --batch-size 500
    python improve_relationships.py --debug-config
"""

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from concept_config import CONCEPTS, TERMS, TAXONOMY


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------

def load_env_if_available() -> None:
    if load_dotenv is not None:
        load_dotenv()


def get_env_first(*names: str, default=None):
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


load_env_if_available()


DEFAULT_NEO4J_URI = get_env_first(
    "NEO4J_URI",
    "NEO4J_URL",
    default="bolt://localhost:7687",
)

DEFAULT_NEO4J_USER = get_env_first(
    "NEO4J_USER",
    "NEO4J_USERNAME",
    default="neo4j",
)

DEFAULT_NEO4J_PASSWORD = get_env_first(
    "NEO4J_PASSWORD",
    "NEO4J_PASS",
    default=None,
)

DEFAULT_NEO4J_DATABASE = get_env_first(
    "NEO4J_DATABASE",
    "NEO4J_DB",
    default=None,
)


# ---------------------------------------------------------------------
# Text matching helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text_for_phrase_match(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def count_phrase_matches(text: str, term: str) -> int:
    if not text or not term:
        return 0

    text_norm = normalize_text_for_phrase_match(text).lower()
    term_norm = normalize_text_for_phrase_match(term).lower()

    return text_norm.count(term_norm)


def count_word_matches(text: str, term: str) -> int:
    if not text or not term:
        return 0

    pattern = (
        rf"(?<![A-Za-z0-9_\u4e00-\u9fff])"
        rf"{re.escape(term)}"
        rf"(?![A-Za-z0-9_\u4e00-\u9fff])"
    )

    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches)


def count_matches(text: str, term: Dict) -> int:
    match_type = term.get("match_type", "phrase")

    if match_type == "word":
        return count_word_matches(text, term["text"])

    return count_phrase_matches(text, term["text"])


def batched(items: List, batch_size: int) -> Iterable[List]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------

class RelationshipImprover:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = None,
        batch_size: int = 500,
        create_direct_concept_edges: bool = True,
        debug_config: bool = False,
    ):
        self.neo4j_uri = uri
        self.neo4j_user = user
        self.neo4j_password = password
        self.neo4j_database = database

        self.batch_size = batch_size
        self.create_direct_concept_edges = create_direct_concept_edges

        self.term_by_id = {t["term_id"]: t for t in TERMS}
        self.concept_by_id = {c["concept_id"]: c for c in CONCEPTS}

        if not self.neo4j_password:
            raise ValueError(
                "Missing Neo4j password. Please set NEO4J_PASSWORD in your .env file "
                "or pass --password."
            )

        if debug_config:
            print("[Neo4j config]")
            print("NEO4J_URI =", repr(self.neo4j_uri))
            print("NEO4J_USER =", repr(self.neo4j_user))
            print(
                "NEO4J_PASSWORD length =",
                len(self.neo4j_password) if self.neo4j_password else None,
            )
            print(
                "NEO4J_PASSWORD startswith =",
                repr(self.neo4j_password[:2]) if self.neo4j_password else None,
            )
            print("NEO4J_DATABASE =", repr(self.neo4j_database))

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
        )

    def close(self):
        self.driver.close()

    def get_session(self):
        if self.neo4j_database:
            return self.driver.session(database=self.neo4j_database)
        return self.driver.session()

    def run(self, clear_old: bool = False):
        with self.get_session() as session:
            print("Dropping legacy uniqueness constraints if any...")
            self.drop_legacy_constraints(session)

            print("Creating constraints...")
            self.create_constraints(session)

            if clear_old:
                print("Clearing old concept/term relationships...")
                self.clear_old_relationships(session)

            print("Creating Concept nodes...")
            self.create_concepts(session)

            print("Creating Term nodes and NORMALIZED_TO relationships...")
            self.create_terms(session)

            print("Creating taxonomy relationships...")
            self.create_taxonomy(session)

            print("Matching chunks to terms...")
            stats = self.match_chunks_to_terms(session)

            print("\nDone.")
            print(f"Chunks scanned: {stats['chunks_scanned']}")
            print(f"Chunk-Term relationships created/updated: {stats['mention_edges']}")
            print(
                "Direct Chunk-Concept relationships created/updated: "
                f"{stats['direct_concept_edges']}"
            )
            print(f"Term match events: {stats['term_match_events']}")

    def drop_legacy_constraints(self, session):
        """
        Drop old uniqueness constraints that conflict with the new design.

        New design:
            Concept uniqueness: concept_id
            Term uniqueness: term_id

        Legacy constraints that may conflict:
            Concept.name unique
            Term.text unique

        We keep non-unique indexes on Concept.name and Term.text for search.
        """
        query = """
        SHOW CONSTRAINTS
        YIELD name, labelsOrTypes, properties, type
        WHERE type = "UNIQUENESS"
          AND (
            ("Concept" IN labelsOrTypes AND "name" IN properties)
            OR
            ("Term" IN labelsOrTypes AND "text" IN properties)
          )
        RETURN name
        """

        constraint_names = [
            record["name"]
            for record in session.run(query)
        ]

        for constraint_name in constraint_names:
            print(f"Dropping legacy constraint: {constraint_name}")
            session.run(f"DROP CONSTRAINT `{constraint_name}` IF EXISTS")

    def create_constraints(self, session):
        queries = [
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
            "CREATE CONSTRAINT term_id_unique IF NOT EXISTS FOR (t:Term) REQUIRE t.term_id IS UNIQUE",
            "CREATE CONSTRAINT candidate_id_unique IF NOT EXISTS FOR (ct:CandidateTerm) REQUIRE ct.candidate_id IS UNIQUE",
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (ch:Chunk) ON (ch.chunk_id)",
            "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX term_text_index IF NOT EXISTS FOR (t:Term) ON (t.text)",
        ]

        for q in queries:
            session.run(q)

    def clear_old_relationships(self, session):
        """
        Clear old generated concept/term structure.

        This does not delete Chunk, Document, Module, or NEXT nodes/relationships.

        It deletes:
            - all CandidateTerm nodes
            - all Term nodes
            - all Concept nodes

        DETACH DELETE is used because old Concept/Term nodes may still have
        unknown relationships from earlier experiments.
        """
        queries = [
            # Remove generated direct relationships from chunks first.
            "MATCH (:Chunk)-[r:MENTIONS_TERM]->(:Term) DELETE r",
            "MATCH (:Chunk)-[r:MENTIONS_CONCEPT]->(:Concept) DELETE r",
            "MATCH (:Chunk)-[r:SEMANTICALLY_RELATED_TO]->(:Concept) DELETE r",

            # Remove candidate terms if you previously used candidate discovery.
            "MATCH (ct:CandidateTerm) DETACH DELETE ct",

            # Remove all generated Term and Concept nodes and any remaining relationships.
            "MATCH (t:Term) DETACH DELETE t",
            "MATCH (c:Concept) DETACH DELETE c",
        ]

        for q in queries:
            session.run(q)
    def create_concepts(self, session):
        query = """
        UNWIND $concepts AS concept
        MERGE (c:Concept {concept_id: concept.concept_id})
        SET c.name = concept.name,
            c.category = concept.category,
            c.description = concept.description,
            c.level = concept.level,
            c.source = concept.source,
            c.updated_at = $now
        SET c.created_at = coalesce(c.created_at, $now)
        """

        session.run(query, concepts=CONCEPTS, now=utc_now_iso())

    def create_terms(self, session):
        query = """
        UNWIND $terms AS term
        MATCH (c:Concept {concept_id: term.concept_id})
        MERGE (t:Term {term_id: term.term_id})
        SET t.text = term.text,
            t.normalized_text = toLower(term.text),
            t.language = term.language,
            t.match_type = term.match_type,
            t.source = term.source,
            t.concept_id = term.concept_id,
            t.updated_at = $now
        SET t.created_at = coalesce(t.created_at, $now)
        MERGE (t)-[r:NORMALIZED_TO]->(c)
        SET r.method = "dictionary",
            r.updated_at = $now
        SET r.created_at = coalesce(r.created_at, $now)
        """

        session.run(query, terms=TERMS, now=utc_now_iso())

    def create_taxonomy(self, session):
        query = """
        UNWIND $taxonomy AS row
        MATCH (parent:Concept {concept_id: row.parent_concept_id})
        MATCH (child:Concept {concept_id: row.child_concept_id})
        MERGE (parent)-[r:HAS_SUBCONCEPT]->(child)
        SET r.source = "manual",
            r.updated_at = $now
        SET r.created_at = coalesce(r.created_at, $now)
        """

        session.run(query, taxonomy=TAXONOMY, now=utc_now_iso())

    def fetch_chunks_batch(self, session, skip: int, limit: int) -> List[Dict]:
        query = """
        MATCH (ch:Chunk)
        WHERE ch.text IS NOT NULL
        RETURN
            elementId(ch) AS element_id,
            ch.chunk_id AS chunk_id,
            ch.text AS text,
            ch.source_file AS source_file,
            ch.page AS page,
            ch.chunk_index AS chunk_index
        ORDER BY ch.source_file, ch.page, ch.chunk_index
        SKIP $skip
        LIMIT $limit
        """

        return [
            dict(record)
            for record in session.run(query, skip=skip, limit=limit)
        ]

    def count_chunks(self, session) -> int:
        result = session.run(
            "MATCH (ch:Chunk) WHERE ch.text IS NOT NULL RETURN count(ch) AS n"
        )
        return result.single()["n"]

    def match_chunks_to_terms(self, session) -> Dict[str, int]:
        total_chunks = self.count_chunks(session)

        stats = {
            "chunks_scanned": 0,
            "mention_edges": 0,
            "direct_concept_edges": 0,
            "term_match_events": 0,
        }

        skip = 0

        while skip < total_chunks:
            chunks = self.fetch_chunks_batch(session, skip, self.batch_size)

            if not chunks:
                break

            mention_rows = []
            direct_concept_counter = defaultdict(int)

            for chunk in chunks:
                text = chunk.get("text") or ""
                stats["chunks_scanned"] += 1

                for term in TERMS:
                    n = count_matches(text, term)

                    if n <= 0:
                        continue

                    mention_rows.append(
                        {
                            "chunk_element_id": chunk["element_id"],
                            "chunk_id": chunk.get("chunk_id"),
                            "term_id": term["term_id"],
                            "concept_id": term["concept_id"],
                            "matched_text": term["text"],
                            "count": n,
                            "method": "rule",
                            "match_type": term.get("match_type", "phrase"),
                        }
                    )

                    direct_key = (chunk["element_id"], term["concept_id"])
                    direct_concept_counter[direct_key] += n

            if mention_rows:
                self.write_mentions(session, mention_rows)
                stats["mention_edges"] += len(mention_rows)
                stats["term_match_events"] += sum(
                    row["count"] for row in mention_rows
                )

            if self.create_direct_concept_edges and direct_concept_counter:
                concept_rows = [
                    {
                        "chunk_element_id": chunk_element_id,
                        "concept_id": concept_id,
                        "count": count,
                        "method": "derived_from_term",
                    }
                    for (chunk_element_id, concept_id), count
                    in direct_concept_counter.items()
                ]

                self.write_direct_concept_edges(session, concept_rows)
                stats["direct_concept_edges"] += len(concept_rows)

            skip += self.batch_size
            print(f"Processed {min(skip, total_chunks)} / {total_chunks} chunks")

        return stats

    def write_mentions(self, session, rows: List[Dict]):
        query = """
        UNWIND $rows AS row
        MATCH (ch:Chunk)
        WHERE elementId(ch) = row.chunk_element_id
        MATCH (t:Term {term_id: row.term_id})
        MERGE (ch)-[r:MENTIONS_TERM]->(t)
        SET r.method = row.method,
            r.match_type = row.match_type,
            r.matched_text = row.matched_text,
            r.count = row.count,
            r.updated_at = $now
        SET r.created_at = coalesce(r.created_at, $now)
        """

        session.run(query, rows=rows, now=utc_now_iso())

    def write_direct_concept_edges(self, session, rows: List[Dict]):
        query = """
        UNWIND $rows AS row
        MATCH (ch:Chunk)
        WHERE elementId(ch) = row.chunk_element_id
        MATCH (c:Concept {concept_id: row.concept_id})
        MERGE (ch)-[r:MENTIONS_CONCEPT]->(c)
        SET r.method = row.method,
            r.count = row.count,
            r.updated_at = $now
        SET r.created_at = coalesce(r.created_at, $now)
        """

        session.run(query, rows=rows, now=utc_now_iso())


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--uri",
        default=DEFAULT_NEO4J_URI,
        help="Neo4j URI. Default reads NEO4J_URI or NEO4J_URL.",
    )

    parser.add_argument(
        "--user",
        default=DEFAULT_NEO4J_USER,
        help="Neo4j username. Default reads NEO4J_USER or NEO4J_USERNAME.",
    )

    parser.add_argument(
        "--password",
        default=DEFAULT_NEO4J_PASSWORD,
        help="Neo4j password. Default reads NEO4J_PASSWORD or NEO4J_PASS.",
    )

    parser.add_argument(
        "--database",
        default=DEFAULT_NEO4J_DATABASE,
        help="Neo4j database name. Optional. Reads NEO4J_DATABASE or NEO4J_DB.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of chunks to process per batch.",
    )

    parser.add_argument(
        "--clear-old",
        action="store_true",
        help="Delete old generated Term/Concept relationships before rebuilding.",
    )

    parser.add_argument(
        "--no-direct-concept",
        action="store_true",
        help="Do not create direct Chunk-[:MENTIONS_CONCEPT]->Concept edges.",
    )

    parser.add_argument(
        "--debug-config",
        action="store_true",
        help="Print Neo4j config debug information without exposing full password.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    improver = RelationshipImprover(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        batch_size=args.batch_size,
        create_direct_concept_edges=not args.no_direct_concept,
        debug_config=args.debug_config,
    )

    try:
        improver.run(clear_old=args.clear_old)
    finally:
        improver.close()


if __name__ == "__main__":
    main()