"""Review evidence-backed Fact nodes before they are used as trusted knowledge."""

import argparse
import os

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class FactReviewer:
    def __init__(self, uri: str, user: str, password: str, database: str | None):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **parameters):
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(query, **parameters)]

    def list_pending(self, limit: int) -> None:
        query = """
        MATCH (subject:Concept)-[:SUBJECT_OF]->(fact:Fact {status: "pending"})-[:OBJECT_OF]->(object:Concept)
        MATCH (chunk:Chunk)-[support:SUPPORTS]->(fact)
        WITH fact, subject, object,
             max(support.confidence) AS confidence,
             count(support) AS evidence_count,
             collect({
                 evidence_quote: support.quote,
                 source_file: chunk.source_file,
                 page: chunk.page
             })[0..3] AS evidence
        RETURN fact.fact_id AS fact_id, subject.name AS subject, fact.predicate AS predicate,
               object.name AS object, confidence, evidence_count, evidence
        ORDER BY confidence DESC
        LIMIT $limit
        """
        for row in self.run(query, limit=limit):
            print(f"\n{row['fact_id']}")
            print(f"  {row['subject']} --{row['predicate']}--> {row['object']}")
            print(f"  confidence: {row['confidence']}")
            print(f"  supporting chunks: {row['evidence_count']} (showing up to 3)")
            for evidence in row["evidence"]:
                print(f"  evidence: {evidence['evidence_quote']}")
                print(f"  source: {evidence['source_file']} p.{evidence['page']}")

    def set_status(self, fact_id: str, status: str) -> None:
        query = """
        MATCH (fact:Fact {fact_id: $fact_id})
        SET fact.status = $status, fact.reviewed_at = datetime()
        RETURN fact.fact_id AS fact_id
        """
        result = self.run(query, fact_id=fact_id, status=status)
        if not result:
            raise ValueError(f"Fact not found: {fact_id}")
        print(f"{fact_id} marked as {status}.")


def main() -> None:
    if load_dotenv:
        load_dotenv()
    parser = argparse.ArgumentParser(description="Review evidence-backed Neo4j facts.")
    parser.add_argument("command", choices=["list", "approve", "reject"])
    parser.add_argument("fact_id", nargs="?")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE") or None)
    args = parser.parse_args()
    if not args.password:
        raise ValueError("Set NEO4J_PASSWORD before running this command.")
    if args.command != "list" and not args.fact_id:
        parser.error("fact_id is required for approve and reject")
    reviewer = FactReviewer(args.uri, args.user, args.password, args.database)
    try:
        if args.command == "list":
            reviewer.list_pending(args.limit)
        else:
            reviewer.set_status(args.fact_id, "approved" if args.command == "approve" else "rejected")
    finally:
        reviewer.close()


if __name__ == "__main__":
    main()
