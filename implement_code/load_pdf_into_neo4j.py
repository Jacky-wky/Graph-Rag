import os
import re
import hashlib
from typing import List, Dict

import fitz
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase


# =========================
# Paths
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPLEMENT_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDER = os.path.join(PROJECT_ROOT, "dataset", "datasets")
ENV_PATH = os.path.join(IMPLEMENT_CODE_DIR, ".env")
if not os.path.exists(ENV_PATH):
    ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


# =========================
# Chunk settings
# =========================

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
BATCH_SIZE = 100


# =========================
# Neo4j settings
# =========================

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# =========================
# PDF extraction
# =========================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving Chinese and English content.
    """
    text = text.replace("\x00", " ")

    # Normalize spaces and tabs, but keep line breaks first.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove too many blank lines.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Strip each line.
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


def split_text_by_char(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Simple character-based chunking.
    Works reasonably well for Chinese because Chinese text does not always use spaces.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def list_pdf_files(folder: str) -> List[str]:
    pdf_files = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(folder, filename))

    pdf_files.sort()
    return pdf_files


def calculate_file_sha256(file_path: str) -> str:
    """Return a stable content fingerprint for document-level provenance."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    source_file = os.path.basename(pdf_path)
    source_stem = os.path.splitext(source_file)[0]
    document_sha256 = calculate_file_sha256(pdf_path)

    chunks_data = []

    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):
        page_number = page_idx + 1
        page = doc[page_idx]

        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        page_chunks = split_text_by_char(text)

        for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = f"{source_stem}_p{page_number}_c{chunk_idx}"

            chunks_data.append(
                {
                    "chunk_id": chunk_id,
                    "page_id": f"{source_stem}_p{page_number}",
                    "source_file": source_file,
                    "source_stem": source_stem,
                    "document_sha256": document_sha256,
                    "page": page_number,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "text_length": len(chunk_text),
                }
            )

    doc.close()
    return chunks_data


def extract_all_chunks() -> List[Dict]:
    print("Project root:", PROJECT_ROOT)
    print("PDF folder:", PDF_FOLDER)

    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    pdf_files = list_pdf_files(PDF_FOLDER)

    print(f"\nFound {len(pdf_files)} PDF files.")

    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        pdf_chunks = extract_chunks_from_pdf(pdf_path)
        all_chunks.extend(pdf_chunks)

        print(f"{os.path.basename(pdf_path)} -> {len(pdf_chunks)} chunks")

    print(f"\nTotal chunks extracted: {len(all_chunks)}")

    return all_chunks


# =========================
# Neo4j helpers
# =========================

def create_constraints(driver):
    """
    Create uniqueness constraints.
    This prevents duplicated documents/chunks when rerunning the script.
    """
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
        """,
        """
        CREATE CONSTRAINT page_id_unique IF NOT EXISTS
        FOR (p:Page)
        REQUIRE p.page_id IS UNIQUE
        """,
    ]

    with driver.session(database=NEO4J_DATABASE) as session:
        for query in queries:
            session.run(query)


def clear_existing_data(driver):
    """
    Optional reset.
    This deletes only project-related nodes.
    Turn this on if you want a clean reload.
    """
    query = """
    MATCH (n)
    WHERE n:Document OR n:Page OR n:Chunk OR n:Module
    DETACH DELETE n
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(query)


def insert_batch(tx, batch):
    query = """
    UNWIND $batch AS row

    MERGE (d:Document {document_id: row.source_file})
    ON CREATE SET
        d.file_name = row.source_file,
        d.source_stem = row.source_stem,
        d.sha256 = row.document_sha256,
        d.created_at = datetime()
    ON MATCH SET
        d.file_name = row.source_file,
        d.source_stem = row.source_stem,
        d.sha256 = row.document_sha256,
        d.updated_at = datetime()

    MERGE (p:Page {page_id: row.page_id})
    ON CREATE SET
        p.page_number = row.page,
        p.source_file = row.source_file,
        p.created_at = datetime()
    ON MATCH SET
        p.page_number = row.page,
        p.source_file = row.source_file,
        p.updated_at = datetime()

    MERGE (c:Chunk {chunk_id: row.chunk_id})
    ON CREATE SET
        c.created_at = datetime()
    SET
        c.source_file = row.source_file,
        c.source_stem = row.source_stem,
        c.page = row.page,
        c.chunk_index = row.chunk_index,
        c.text = row.text,
        c.text_length = row.text_length,
        c.updated_at = datetime()

    MERGE (d)-[:HAS_CHUNK]->(c)
    MERGE (d)-[:HAS_PAGE]->(p)
    MERGE (p)-[:HAS_CHUNK]->(c)
    MERGE (c)-[:IN_PAGE]->(p)
    """

    tx.run(query, batch=batch)


def insert_chunks(driver, chunks: List[Dict]):
    with driver.session(database=NEO4J_DATABASE) as session:
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Inserting chunks"):
            batch = chunks[i:i + BATCH_SIZE]
            session.execute_write(insert_batch, batch)


def create_next_relationships(driver):
    """
    Add NEXT relationship inside each document.

    Structure:
    (:Chunk)-[:NEXT]->(:Chunk)

    Ordering:
    source_file, page, chunk_index
    """
    query = """
    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
    WITH d, c
    ORDER BY d.document_id, c.page, c.chunk_index
    WITH d, collect(c) AS chunks
    UNWIND range(0, size(chunks) - 2) AS i
    WITH chunks[i] AS current_chunk, chunks[i + 1] AS next_chunk
    MERGE (current_chunk)-[:NEXT]->(next_chunk)
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(query)


def verify_graph(driver):
    queries = {
        "Document count": """
            MATCH (d:Document)
            RETURN count(d) AS count
        """,
        "Chunk count": """
            MATCH (c:Chunk)
            RETURN count(c) AS count
        """,
        "Page count": """
            MATCH (p:Page)
            RETURN count(p) AS count
        """,
        "HAS_CHUNK relationship count": """
            MATCH (:Document)-[r:HAS_CHUNK]->(:Chunk)
            RETURN count(r) AS count
        """,
        "NEXT relationship count": """
            MATCH (:Chunk)-[r:NEXT]->(:Chunk)
            RETURN count(r) AS count
        """,
    }

    with driver.session(database=NEO4J_DATABASE) as session:
        print("\nGraph verification:")
        print("=" * 80)

        for name, query in queries.items():
            result = session.run(query)
            record = result.single()
            print(f"{name}: {record['count']}")

        print("=" * 80)


def show_sample_chunk(driver):
    query = """
    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
    MATCH (c)-[:IN_PAGE]->(p:Page)
    RETURN 
        d.file_name AS document,
        c.chunk_id AS chunk_id,
        c.page AS page,
        c.chunk_index AS chunk_index,
        p.page_id AS page_id,
        c.text AS text
    ORDER BY d.file_name, c.page, c.chunk_index
    LIMIT 1
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(query).single()

        if record:
            print("\nSample chunk:")
            print("=" * 80)
            print("Document:", record["document"])
            print("Chunk ID:", record["chunk_id"])
            print("Page:", record["page"])
            print("Chunk index:", record["chunk_index"])
            print("Text preview:")
            print(record["text"][:1000])
            print("=" * 80)


def main():
    print("Connecting to Neo4j...")
    print("NEO4J_URI:", NEO4J_URI)
    print("NEO4J_DATABASE:", NEO4J_DATABASE)

    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD is missing. Please check your .env file.")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        create_constraints(driver)

        RESET_DATABASE = True

        if RESET_DATABASE:
            print("Clearing existing project data...")
            clear_existing_data(driver)

        chunks = extract_all_chunks()

        print("\nLoading chunks into Neo4j...")
        insert_chunks(driver, chunks)

        print("\nCreating NEXT relationships...")
        create_next_relationships(driver)

        verify_graph(driver)
        show_sample_chunk(driver)

        print("\nDone.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
