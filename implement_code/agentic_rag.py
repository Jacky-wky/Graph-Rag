"""Agentic RAG over the existing Neo4j Chunk / Term / Concept graph.

The agent can call two read-only tools:
* search_graph: expand concepts and retrieve their supporting chunks.
* search_chunks: run a Neo4j full-text search over chunk text.

It synthesizes an answer only from the retrieved evidence and includes source
citations in the form ``[source_file p.<page>]``.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from neo4j import GraphDatabase
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


SYSTEM_PROMPT = """You are a precise knowledge-graph RAG assistant. Use the
available tools before answering a factual question about the document corpus.
You may make multiple tool calls when evidence is incomplete. Answer only from
the retrieved evidence; say that the corpus does not establish a claim when it
does not. Cite every factual claim using the source citations returned by tools.
Do not expose Cypher or invent citations."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_graph",
            "description": "Find concepts matching the query, traverse their taxonomy, and retrieve chunks that mention them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Concept, term, or natural-language topic."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": "Run keyword full-text search over document chunks. Use this for exact wording, rules, or when graph search is insufficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or a short phrase."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_facts",
            "description": "Retrieve approved, evidence-backed concept relationships. Use this for questions about requirements, responsibilities, applicability, or controls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Concept, term, or natural-language topic."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class Settings:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str | None
    openai_model: str

    @classmethod
    def from_environment(cls) -> "Settings":
        if load_dotenv:
            load_dotenv()
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("Set NEO4J_PASSWORD in .env or the environment.")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY in .env or the environment.")
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")),
            neo4j_password=password,
            neo4j_database=os.getenv("NEO4J_DATABASE") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        )


class Neo4jRetriever:
    def __init__(self, settings: Settings):
        self.database = settings.neo4j_database
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(query, **parameters)]

    @staticmethod
    def _evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "concepts": row.get("concepts", []),
                "text": row["text"],
                "citation": f"[{row.get('source_file', 'unknown source')} p.{row.get('page', '?')}]",
            }
            for row in rows
        ]

    def search_graph(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        cypher = """
        MATCH (seed:Concept)
        WHERE toLower(seed.name) CONTAINS toLower($query)
           OR seed.concept_id CONTAINS toLower($query)
           OR EXISTS { MATCH (term:Term)-[:NORMALIZED_TO]->(seed)
                       WHERE toLower(term.text) CONTAINS toLower($query) }
        OPTIONAL MATCH (seed)-[:HAS_SUBCONCEPT*0..2]->(related:Concept)
        WITH collect(DISTINCT seed) + collect(DISTINCT related) AS concepts
        UNWIND concepts AS concept
        MATCH (chunk:Chunk)-[:MENTIONS_CONCEPT]->(concept)
        WITH chunk, collect(DISTINCT concept.name) AS concepts, sum(coalesce(chunk.chunk_index, 0)) AS ordering
        RETURN chunk.text AS text, chunk.source_file AS source_file, chunk.page AS page, concepts
        ORDER BY size(concepts) DESC, ordering ASC
        LIMIT $limit
        """
        return self._evidence(self._run(cypher, query=query, limit=limit))

    def search_chunks(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes('chunk_text', $query) YIELD node, score
        OPTIONAL MATCH (node)-[:MENTIONS_CONCEPT]->(concept:Concept)
        RETURN node.text AS text, node.source_file AS source_file, node.page AS page,
               collect(DISTINCT concept.name) AS concepts, score
        ORDER BY score DESC
        LIMIT $limit
        """
        try:
            return self._evidence(self._run(cypher, query=query, limit=limit))
        except Exception as error:
            if "chunk_text" not in str(error):
                raise
            fallback = """
            MATCH (chunk:Chunk)
            WHERE toLower(chunk.text) CONTAINS toLower($query)
            OPTIONAL MATCH (chunk)-[:MENTIONS_CONCEPT]->(concept:Concept)
            RETURN chunk.text AS text, chunk.source_file AS source_file, chunk.page AS page,
                   collect(DISTINCT concept.name) AS concepts
            LIMIT $limit
            """
            return self._evidence(self._run(fallback, query=query, limit=limit))

    def search_facts(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        cypher = """
        MATCH (subject:Concept)-[:SUBJECT_OF]->(fact:Fact {status: "approved"})-[:OBJECT_OF]->(object:Concept)
        WHERE toLower(subject.name) CONTAINS toLower($query)
           OR toLower(object.name) CONTAINS toLower($query)
           OR toLower(fact.predicate) CONTAINS toLower($query)
        MATCH (chunk:Chunk)-[support:SUPPORTS]->(fact)
        RETURN support.quote AS text, chunk.source_file AS source_file, chunk.page AS page,
               [subject.name, object.name] AS concepts, fact.predicate AS predicate,
               support.confidence AS confidence
        ORDER BY support.confidence DESC
        LIMIT $limit
        """
        return self._evidence(self._run(cypher, query=query, limit=limit))


class RAGAgent:
    def __init__(self, settings: Settings):
        self.client = OpenAI()
        self.model = settings.openai_model
        self.retriever = Neo4jRetriever(settings)
        self.tool_handlers: dict[str, Callable[..., list[dict[str, Any]]]] = {
            "search_graph": self.retriever.search_graph,
            "search_chunks": self.retriever.search_chunks,
            "search_facts": self.retriever.search_facts,
        }

    def close(self) -> None:
        self.retriever.close()

    def answer(self, question: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for _ in range(6):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return message.content or "No answer was generated."
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                handler = self.tool_handlers[call.function.name]
                result = handler(**arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
        return "I could not complete retrieval within the tool-call limit. Please refine the question."


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over a Neo4j knowledge graph.")
    parser.add_argument("question", nargs="+", help="Question to answer")
    args = parser.parse_args()
    agent = RAGAgent(Settings.from_environment())
    try:
        print(agent.answer(" ".join(args.question)))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
