"""Induce a provenance-first concept graph from existing Neo4j Chunk nodes.

This command uses local multilingual embeddings and unsupervised clustering.
It does not use a manual concept dictionary, an LLM API, or human approval.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase

from automated_concept_induction import (
    ALGORITHM_VERSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RESOLVER_MODEL,
    AutomaticConceptInducer,
    build_preview,
)
from semantic_concept_resolver import encode_ontology_fields


IMPLEMENT_CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = IMPLEMENT_CODE_DIR / ".cache" / "concept_induction"
DEFAULT_PREVIEW_PATH = IMPLEMENT_CODE_DIR / "auto_concepts_preview.json"

load_dotenv(IMPLEMENT_CODE_DIR / ".env")

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", os.getenv("NEO4J_URL", "bolt://localhost:7687"))
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", os.getenv("NEO4J_PASS"))
DEFAULT_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", os.getenv("NEO4J_DB", "neo4j"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


class AutomaticRelationshipBuilder:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        batch_size: int,
        resolver_batch_size: int,
        inducer: AutomaticConceptInducer,
    ):
        if not password:
            raise ValueError("NEO4J_PASSWORD is missing. Check implement_code/.env.")
        self.database = database
        self.batch_size = batch_size
        self.resolver_batch_size = resolver_batch_size
        self.inducer = inducer
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def session(self):
        return self.driver.session(database=self.database)

    def fetch_chunks(self, session) -> list[dict[str, Any]]:
        query = """
        MATCH (chunk:Chunk)
        WHERE chunk.text IS NOT NULL
          AND coalesce(chunk.is_table_of_contents, false) = false
        RETURN chunk.chunk_id AS chunk_id,
               chunk.text AS text,
               coalesce(chunk.retrieval_text, chunk.text) AS retrieval_text,
               chunk.source_file AS source_file,
               chunk.page AS page,
               chunk.chunk_index AS chunk_index,
               chunk.section_id AS section_id,
               chunk.section_title AS section_title,
               chunk.section_path AS section_path
        ORDER BY source_file, page, chunk_index
        """
        return [dict(record) for record in session.run(query)]

    def existing_semantic_counts(self, session) -> dict[str, int]:
        query = """
        MATCH (node)
        RETURN count(CASE WHEN "Concept" IN labels(node) THEN 1 END) AS concepts,
               count(CASE WHEN "Term" IN labels(node) THEN 1 END) AS terms,
               count(CASE WHEN "Fact" IN labels(node) THEN 1 END) AS facts,
               count(CASE WHEN "OntologyRun" IN labels(node) THEN 1 END) AS ontology_runs
        """
        return dict(session.run(query).single())

    def run(
        self,
        clear_old: bool,
        dry_run: bool,
        preview_path: Path,
    ) -> dict[str, Any]:
        self.driver.verify_connectivity()
        with self.session() as session:
            existing = self.existing_semantic_counts(session)
            if not dry_run and existing["concepts"] and not clear_old:
                raise RuntimeError(
                    f"Neo4j already contains {existing['concepts']} Concept nodes. "
                    "Use --clear-old to replace the semantic layer, or --dry-run to preview."
                )
            chunks = self.fetch_chunks(session)

        print(f"Non-TOC chunks used for induction: {len(chunks)}")
        result = self.inducer.induce(chunks)
        print("Precomputing local Concept and Term resolver embeddings...")
        result["resolver_embeddings"] = encode_ontology_fields(
            result["concepts"],
            result["terms"],
            model_name=result["run"]["resolver_model"],
            batch_size=self.resolver_batch_size,
            device=self.inducer.device,
        )
        result["run"]["resolver_embedding_dimensions"] = result["resolver_embeddings"]["dimensions"]
        self.write_preview(preview_path, result)
        self.print_summary(result, preview_path)

        if dry_run:
            print("\nDry run complete. Neo4j was not modified.")
            return result

        with self.session() as session:
            self.drop_legacy_schema(session)
            if clear_old:
                self.clear_old_semantic_layer(session)
            self.create_constraints(session)
            self.write_ontology_run(session, result["run"])
            self.write_concepts(session, result["concepts"])
            self.write_terms(session, result["terms"])
            self.write_resolver_embeddings(session, result["resolver_embeddings"])
            self.write_taxonomy(session, result["taxonomy"])
            self.write_chunk_embeddings(session, result["chunk_embeddings"])
            self.write_semantic_assignments(session, result["assignments"])
            self.write_exact_mentions(session, result["mentions"])
            self.derive_exact_concept_mentions(session, result["run"]["run_id"])
            self.write_representatives(session, result["representatives"])
            self.write_run_provenance(session, result)
            self.create_vector_indexes(
                session,
                result["run"]["embedding_dimensions"],
                result["run"]["resolver_embedding_dimensions"],
            )
            self.verify_graph(session, result["run"]["run_id"])
        return result

    def write_preview(self, preview_path: Path, result: dict[str, Any]) -> None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(
            json.dumps(build_preview(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def print_summary(self, result: dict[str, Any], preview_path: Path) -> None:
        run = result["run"]
        leaf_concepts = [concept for concept in result["concepts"] if concept["level"] == 2]
        print("\nAutomatic concept induction summary")
        print("=" * 80)
        print("Run ID:", run["run_id"])
        print("Algorithm:", run["algorithm"])
        print("Embedding model:", run["embedding_model"])
        print("Resolver model:", run["resolver_model"])
        print("Embedding dimensions:", run["embedding_dimensions"])
        print("Resolver embedding dimensions:", run["resolver_embedding_dimensions"])
        print("Clustering method:", run["clustering_method"])
        print("Leaf concepts:", run["leaf_concept_count"])
        print("Parent concepts:", run["parent_concept_count"])
        print("Terms:", run["term_count"])
        print("Noise chunks reassigned:", run["noise_chunk_count_before_reassignment"])
        print("Embedding cache hit:", run["embedding_cache_hit"])
        print("\nLeaf concepts:")
        for concept in sorted(leaf_concepts, key=lambda row: (-row["chunk_count"], row["name"])):
            print(
                f"- {concept['concept_id']} | {concept['name']} | "
                f"chunks={concept['chunk_count']} | confidence={concept['confidence']:.3f}"
            )
        print("\nPreview:", preview_path.resolve())
        print("=" * 80)

    def drop_legacy_schema(self, session) -> None:
        statements = [
            "DROP CONSTRAINT concept_name_unique IF EXISTS",
            "DROP CONSTRAINT candidate_id_unique IF EXISTS",
            "DROP INDEX candidate_status_index IF EXISTS",
            "DROP INDEX candidate_text_index IF EXISTS",
            "DROP INDEX auto_chunk_embedding IF EXISTS",
            "DROP INDEX auto_concept_embedding IF EXISTS",
            "DROP INDEX auto_concept_resolver_embedding IF EXISTS",
            "DROP INDEX auto_term_resolver_embedding IF EXISTS",
        ]
        for statement in statements:
            session.run(statement).consume()

    def clear_old_semantic_layer(self, session) -> None:
        print("Replacing old Concept, Term, Fact, and ontology-run data...")
        statements = [
            "MATCH (fact:Fact) DETACH DELETE fact",
            "MATCH (candidate:CandidateTerm) DETACH DELETE candidate",
            "MATCH (run:OntologyRun) DETACH DELETE run",
            "MATCH (term:Term) DETACH DELETE term",
            "MATCH (concept:Concept) DETACH DELETE concept",
            "MATCH (chunk:Chunk) REMOVE chunk.embedding, chunk.embedding_model, chunk.embedding_run_id",
        ]
        for statement in statements:
            session.run(statement).consume()

    def create_constraints(self, session) -> None:
        statements = [
            "CREATE CONSTRAINT ontology_run_id_unique IF NOT EXISTS "
            "FOR (run:OntologyRun) REQUIRE run.run_id IS UNIQUE",
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS "
            "FOR (concept:Concept) REQUIRE concept.concept_id IS UNIQUE",
            "CREATE CONSTRAINT term_id_unique IF NOT EXISTS "
            "FOR (term:Term) REQUIRE term.term_id IS UNIQUE",
            "CREATE INDEX concept_name_index IF NOT EXISTS FOR (concept:Concept) ON (concept.name)",
            "CREATE INDEX term_text_index IF NOT EXISTS FOR (term:Term) ON (term.text)",
        ]
        for statement in statements:
            session.run(statement).consume()

    def write_ontology_run(self, session, run: dict[str, Any]) -> None:
        row = {**run, "created_at": utc_now_iso()}
        query = """
        MERGE (ontology_run:OntologyRun {run_id: $row.run_id})
        SET ontology_run += $row
        """
        session.run(query, row=row).consume()

    def write_concepts(self, session, concepts: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (concept:Concept {concept_id: row.concept_id})
        SET concept.name = row.name,
            concept.category = row.category,
            concept.description = row.description,
            concept.level = row.level,
            concept.source = row.source,
            concept.method = row.method,
            concept.run_id = row.run_id,
            concept.embedding_model = row.embedding_model,
            concept.confidence = row.confidence,
            concept.chunk_count = row.chunk_count,
            concept.document_count = row.document_count,
            concept.embedding = row.embedding,
            concept.updated_at = $now
        SET concept.created_at = coalesce(concept.created_at, $now)
        """
        now = utc_now_iso()
        for rows in batched(concepts, self.batch_size):
            session.run(query, rows=rows, now=now).consume()

    def write_terms(self, session, terms: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MATCH (concept:Concept {concept_id: row.concept_id})
        MERGE (term:Term {term_id: row.term_id})
        SET term.text = row.text,
            term.normalized_text = row.normalized_text,
            term.language = row.language,
            term.match_type = row.match_type,
            term.source = row.source,
            term.concept_id = row.concept_id,
            term.ctfidf_score = row.ctfidf_score,
            term.evidence_weight = row.evidence_weight,
            term.frequency = row.frequency,
            term.chunk_frequency = row.chunk_frequency,
            term.heading_frequency = row.heading_frequency,
            term.run_id = row.run_id,
            term.updated_at = $now
        SET term.created_at = coalesce(term.created_at, $now)
        MERGE (term)-[normalized:NORMALIZED_TO]->(concept)
        SET normalized.method = "automatic_ctfidf",
            normalized.score = row.ctfidf_score,
            normalized.run_id = row.run_id,
            normalized.updated_at = $now
        SET normalized.created_at = coalesce(normalized.created_at, $now)
        """
        now = utc_now_iso()
        for rows in batched(terms, self.batch_size):
            session.run(query, rows=rows, now=now).consume()

    def write_resolver_embeddings(self, session, resolver_embeddings: dict[str, Any]) -> None:
        concept_query = """
        UNWIND $rows AS row
        MATCH (concept:Concept {concept_id: row.concept_id})
        SET concept.resolver_embedding = row.embedding,
            concept.resolver_model = row.resolver_model
        """
        term_query = """
        UNWIND $rows AS row
        MATCH (term:Term {term_id: row.term_id})
        SET term.resolver_embedding = row.embedding,
            term.resolver_model = row.resolver_model
        """
        for rows in batched(resolver_embeddings["concepts"], min(self.batch_size, 100)):
            session.run(concept_query, rows=rows).consume()
        for rows in batched(resolver_embeddings["terms"], min(self.batch_size, 100)):
            session.run(term_query, rows=rows).consume()

    def write_taxonomy(self, session, taxonomy: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MATCH (parent:Concept {concept_id: row.parent_concept_id})
        MATCH (child:Concept {concept_id: row.child_concept_id})
        MERGE (parent)-[relationship:HAS_SUBCONCEPT]->(child)
        SET relationship.method = row.method,
            relationship.confidence = row.confidence,
            relationship.run_id = row.run_id,
            relationship.updated_at = $now
        SET relationship.created_at = coalesce(relationship.created_at, $now)
        """
        now = utc_now_iso()
        for rows in batched(taxonomy, self.batch_size):
            session.run(query, rows=rows, now=now).consume()

    def write_chunk_embeddings(self, session, embeddings: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MATCH (chunk:Chunk {chunk_id: row.chunk_id})
        SET chunk.embedding = row.embedding,
            chunk.embedding_model = row.embedding_model,
            chunk.embedding_run_id = row.run_id
        """
        for rows in batched(embeddings, min(self.batch_size, 100)):
            session.run(query, rows=rows).consume()

    def write_semantic_assignments(self, session, assignments: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MATCH (chunk:Chunk {chunk_id: row.chunk_id})
        MATCH (concept:Concept {concept_id: row.concept_id})
        MERGE (chunk)-[relationship:ABOUT_CONCEPT]->(concept)
        SET relationship.primary = row.primary,
            relationship.confidence = row.confidence,
            relationship.semantic_similarity = row.semantic_similarity,
            relationship.cluster_probability = row.cluster_probability,
            relationship.method = row.method,
            relationship.run_id = row.run_id,
            relationship.updated_at = $now
        SET relationship.created_at = coalesce(relationship.created_at, $now)
        """
        now = utc_now_iso()
        for rows in batched(assignments, self.batch_size):
            session.run(query, rows=rows, now=now).consume()

    def write_exact_mentions(self, session, mentions: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MATCH (chunk:Chunk {chunk_id: row.chunk_id})
        MATCH (term:Term {term_id: row.term_id})
        MERGE (chunk)-[mention:MENTIONS_TERM]->(term)
        SET mention.method = row.method,
            mention.match_type = row.match_type,
            mention.matched_text = row.matched_text,
            mention.count = row.count,
            mention.source_count = row.source_count,
            mention.match_scope = row.match_scope,
            mention.run_id = row.run_id,
            mention.updated_at = $now
        SET mention.created_at = coalesce(mention.created_at, $now)
        """
        now = utc_now_iso()
        for rows in batched(mentions, self.batch_size):
            session.run(query, rows=rows, now=now).consume()

    def derive_exact_concept_mentions(self, session, run_id: str) -> None:
        query = """
        MATCH (chunk:Chunk)-[mention:MENTIONS_TERM]->(term:Term)-[:NORMALIZED_TO]->(concept:Concept)
        WITH chunk, concept,
             sum(mention.count) AS match_count,
             sum(mention.source_count) AS source_count,
             collect(DISTINCT mention.matched_text) AS matched_terms,
             max(coalesce(term.evidence_weight, 0.0)) AS strongest_evidence
        MATCH (concept)<-[:NORMALIZED_TO]-(concept_term:Term)
        WITH chunk, concept, match_count, source_count, matched_terms, strongest_evidence,
             max(coalesce(concept_term.evidence_weight, 0.0)) AS concept_top_evidence
        WITH chunk, concept, match_count, source_count, matched_terms, strongest_evidence,
             CASE WHEN concept_top_evidence > 0.0
                  THEN strongest_evidence / concept_top_evidence
                  ELSE 0.0 END AS confidence
        WHERE confidence >= 0.45
        MERGE (chunk)-[relationship:MENTIONS_CONCEPT]->(concept)
        SET relationship.method = "derived_from_automatic_terms",
            relationship.count = match_count,
            relationship.source_count = source_count,
            relationship.matched_terms = matched_terms,
            relationship.confidence = confidence,
            relationship.evidence_weight = strongest_evidence,
            relationship.run_id = $run_id,
            relationship.updated_at = $now
        SET relationship.created_at = coalesce(relationship.created_at, $now)
        """
        session.run(query, run_id=run_id, now=utc_now_iso()).consume()

    def write_representatives(self, session, representatives: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MATCH (concept:Concept {concept_id: row.concept_id})
        MATCH (chunk:Chunk {chunk_id: row.chunk_id})
        MERGE (concept)-[relationship:REPRESENTED_BY]->(chunk)
        SET relationship.rank = row.rank,
            relationship.similarity = row.similarity,
            relationship.method = $method,
            relationship.updated_at = $now
        SET relationship.created_at = coalesce(relationship.created_at, $now)
        """
        now = utc_now_iso()
        for rows in batched(representatives, self.batch_size):
            session.run(query, rows=rows, method=ALGORITHM_VERSION, now=now).consume()

    def write_run_provenance(self, session, result: dict[str, Any]) -> None:
        run_id = result["run"]["run_id"]
        concept_ids = [concept["concept_id"] for concept in result["concepts"]]
        chunk_ids = [row["chunk_id"] for row in result["chunk_embeddings"]]
        concept_query = """
        MATCH (run:OntologyRun {run_id: $run_id})
        UNWIND $concept_ids AS concept_id
        MATCH (concept:Concept {concept_id: concept_id})
        MERGE (run)-[:GENERATED]->(concept)
        """
        chunk_query = """
        MATCH (run:OntologyRun {run_id: $run_id})
        UNWIND $chunk_ids AS chunk_id
        MATCH (chunk:Chunk {chunk_id: chunk_id})
        MERGE (run)-[:USED_CHUNK]->(chunk)
        """
        for rows in batched(concept_ids, self.batch_size):
            session.run(concept_query, run_id=run_id, concept_ids=rows).consume()
        for rows in batched(chunk_ids, self.batch_size):
            session.run(chunk_query, run_id=run_id, chunk_ids=rows).consume()

    def create_vector_indexes(self, session, dimensions: int, resolver_dimensions: int) -> None:
        statements = [
            f"""
            CREATE VECTOR INDEX auto_chunk_embedding IF NOT EXISTS
            FOR (node:Chunk) ON (node.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX auto_concept_embedding IF NOT EXISTS
            FOR (node:Concept) ON (node.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX auto_concept_resolver_embedding IF NOT EXISTS
            FOR (node:Concept) ON (node.resolver_embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {resolver_dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX auto_term_resolver_embedding IF NOT EXISTS
            FOR (node:Term) ON (node.resolver_embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {resolver_dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]
        for statement in statements:
            session.run(statement).consume()

    def verify_graph(self, session, run_id: str) -> None:
        query = """
        MATCH (run:OntologyRun {run_id: $run_id})
        RETURN count { (run)-[:GENERATED]->(:Concept) } AS concepts,
               count { (:Term {run_id: $run_id}) } AS terms,
               count { (concept:Concept {run_id: $run_id, level: 2})
                       WHERE concept.resolver_embedding IS NOT NULL } AS resolver_concepts,
               count { (term:Term {run_id: $run_id})
                       WHERE term.resolver_embedding IS NOT NULL } AS resolver_terms,
               count { (:Chunk)-[:ABOUT_CONCEPT {primary: true, run_id: $run_id}]->(:Concept) } AS primary_assignments,
               count { (:Chunk)-[:MENTIONS_TERM {run_id: $run_id}]->(:Term) } AS exact_mentions,
               count { (:Chunk)-[:MENTIONS_CONCEPT {run_id: $run_id}]->(:Concept) } AS exact_concept_mentions,
               count { (:Concept {run_id: $run_id})-[:REPRESENTED_BY]->(:Chunk) } AS representatives,
               count { (run)-[:USED_CHUNK]->(:Chunk) } AS provenance_chunks
        """
        record = dict(session.run(query, run_id=run_id).single())
        print("\nNeo4j semantic graph verification")
        print("=" * 80)
        for key, value in record.items():
            print(f"{key}: {value}")
        print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically induce Concept, Term, taxonomy, and chunk relationships."
    )
    parser.add_argument("--uri", default=DEFAULT_NEO4J_URI)
    parser.add_argument("--user", default=DEFAULT_NEO4J_USER)
    parser.add_argument("--password", default=DEFAULT_NEO4J_PASSWORD)
    parser.add_argument("--database", default=DEFAULT_NEO4J_DATABASE)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--resolver-model", default=DEFAULT_RESOLVER_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--resolver-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-cluster-size", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument("--terms-per-concept", type=int, default=12)
    parser.add_argument("--min-term-chunks", type=int, default=2)
    parser.add_argument("--representative-chunks", type=int, default=3)
    parser.add_argument("--max-concepts-per-chunk", type=int, default=2)
    parser.add_argument("--secondary-similarity", type=float, default=0.78)
    parser.add_argument("--secondary-margin", type=float, default=0.035)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-json", default=str(DEFAULT_PREVIEW_PATH))
    parser.add_argument(
        "--clear-old",
        action="store_true",
        help="Replace old Concept, Term, Fact, CandidateTerm, and OntologyRun data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Induce concepts and write JSON preview without modifying Neo4j.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.embedding_batch_size < 1 or args.resolver_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.min_cluster_size and args.min_cluster_size < 3:
        raise ValueError("--min-cluster-size must be 0 (auto) or at least 3.")
    if args.min_samples and args.min_samples < 1:
        raise ValueError("--min-samples must be 0 (auto) or positive.")
    if args.terms_per_concept < 1 or args.min_term_chunks < 1:
        raise ValueError("Term settings must be positive.")
    if args.max_concepts_per_chunk < 1:
        raise ValueError("--max-concepts-per-chunk must be positive.")
    if not 0.0 <= args.secondary_similarity <= 1.0:
        raise ValueError("--secondary-similarity must be between 0 and 1.")
    if not 0.0 <= args.secondary_margin <= 1.0:
        raise ValueError("--secondary-margin must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    inducer = AutomaticConceptInducer(
        embedding_model=args.embedding_model,
        resolver_model=args.resolver_model,
        embedding_batch_size=args.embedding_batch_size,
        device=args.device,
        min_cluster_size=args.min_cluster_size or None,
        min_samples=args.min_samples or None,
        terms_per_concept=args.terms_per_concept,
        min_term_chunk_frequency=args.min_term_chunks,
        representative_chunks=args.representative_chunks,
        max_concepts_per_chunk=args.max_concepts_per_chunk,
        secondary_similarity=args.secondary_similarity,
        secondary_margin=args.secondary_margin,
        random_state=args.random_state,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    builder = AutomaticRelationshipBuilder(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        batch_size=args.batch_size,
        resolver_batch_size=args.resolver_batch_size,
        inducer=inducer,
    )
    try:
        builder.run(
            clear_old=args.clear_old,
            dry_run=args.dry_run,
            preview_path=Path(args.output_json),
        )
    finally:
        builder.close()


if __name__ == "__main__":
    main()
