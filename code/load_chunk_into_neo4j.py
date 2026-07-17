import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

CHUNKS_FILE = os.path.join(PROJECT_ROOT, "temp", "pdf_chunks.json")

load_dotenv(ENV_PATH)


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def create_constraints(driver):
    queries = [
        """
        CREATE CONSTRAINT document_id_unique IF NOT EXISTS
        FOR (d:Document)
        REQUIRE d.document_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
        FOR (c:Chunk)
        REQUIRE c.chunk_id IS UNIQUE
        """
    ]

    with driver.session(database=NEO4J_DATABASE) as session:
        for query in queries:
            session.run(query)


def load_chunks():
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from JSON.")

    return chunks


def insert_batch(tx, batch):
    query = """
    UNWIND $batch AS row

    MERGE (d:Document {document_id: row.source_file})
    ON CREATE SET
        d.file_name = row.source_file,
        d.created_at = datetime()
    ON MATCH SET
        d.updated_at = datetime()

    MERGE (c:Chunk {chunk_id: row.chunk_id})
    ON CREATE SET
        c.created_at = datetime()
    SET
        c.source_file = row.source_file,
        c.page = row.page,
        c.chunk_index = row.chunk_index,
        c.text = row.text,
        c.text_length = size(row.text),
        c.updated_at = datetime()

    MERGE (d)-[:HAS_CHUNK]->(c)
    """

    tx.run(query, batch=batch)


def insert_chunks(driver, chunks, batch_size=100):
    total = len(chunks)

    with driver.session(database=NEO4J_DATABASE) as session:
        for i in tqdm(range(0, total, batch_size), desc="Inserting chunks"):
            batch = chunks[i:i + batch_size]
            session.execute_write(insert_batch, batch)


def verify_insert(driver):
    query = """
    MATCH (d:Document)
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    RETURN d.file_name AS document, count(c) AS chunks
    ORDER BY document
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)

        print("\nInserted documents:")
        for record in result:
            print(f"{record['document']}: {record['chunks']} chunks")


def main():
    print("Connecting to Neo4j...")
    print("NEO4J_URI:", NEO4J_URI)
    print("NEO4J_DATABASE:", NEO4J_DATABASE)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        create_constraints(driver)

        chunks = load_chunks()

        insert_chunks(driver, chunks)

        verify_insert(driver)

        print("\nDone inserting chunks into Neo4j.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()