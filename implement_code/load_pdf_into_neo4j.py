import argparse
import os
import hashlib
from typing import List, Dict

import fitz
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase

from regulatory_chunking import chunk_document_pages


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

TARGET_TOKENS = 420
MAX_TOKENS = 650
OVERLAP_TOKENS = 70
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


def extract_chunks_from_pdf(
    pdf_path: str,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Dict]:
    source_file = os.path.basename(pdf_path)
    source_stem = os.path.splitext(source_file)[0]
    document_sha256 = calculate_file_sha256(pdf_path)

    chunks_data = []

    doc = fitz.open(pdf_path)
    try:
        page_texts = [page.get_text("text") for page in doc]
    finally:
        doc.close()

    document_pages = chunk_document_pages(
        page_texts,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    for page_idx, page_chunks in enumerate(document_pages):
        page_number = page_idx + 1
        for chunk_idx, chunk in enumerate(page_chunks, start=1):
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
                    "text": chunk.text,
                    "retrieval_text": chunk.retrieval_text,
                    "text_length": len(chunk.text),
                    "estimated_tokens": chunk.estimated_tokens,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "is_table_of_contents": chunk.is_table_of_contents,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "split_part": chunk.split_part,
                    "chunking_method": chunk.chunking_method,
                }
            )
    return chunks_data


def extract_all_chunks(
    pdf_folder: str = PDF_FOLDER,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Dict]:
    print("Project root:", PROJECT_ROOT)
    print("PDF folder:", pdf_folder)

    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    pdf_files = list_pdf_files(pdf_folder)

    print(f"\nFound {len(pdf_files)} PDF files.")

    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        pdf_chunks = extract_chunks_from_pdf(
            pdf_path,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(pdf_chunks)

        toc_chunks = sum(chunk["is_table_of_contents"] for chunk in pdf_chunks)
        print(f"{os.path.basename(pdf_path)} -> {len(pdf_chunks)} chunks ({toc_chunks} TOC)")

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
        """
        CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
        FOR (c:Chunk) ON EACH [c.text, c.retrieval_text]
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
    WHERE n:Document OR n:Page OR n:Chunk OR n:Fact OR n:Module
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
        p.is_table_of_contents = row.is_table_of_contents,
        p.created_at = datetime()
    ON MATCH SET
        p.page_number = row.page,
        p.source_file = row.source_file,
        p.is_table_of_contents = row.is_table_of_contents,
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
        c.retrieval_text = row.retrieval_text,
        c.text_length = row.text_length,
        c.estimated_tokens = row.estimated_tokens,
        c.section_id = row.section_id,
        c.section_title = row.section_title,
        c.section_path = row.section_path,
        c.is_table_of_contents = row.is_table_of_contents,
        c.start_line = row.start_line,
        c.end_line = row.end_line,
        c.split_part = row.split_part,
        c.chunking_method = row.chunking_method,
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
        c.section_id AS section_id,
        c.section_path AS section_path,
        c.estimated_tokens AS estimated_tokens,
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
            print("Section ID:", record["section_id"])
            print("Section path:", record["section_path"])
            print("Estimated tokens:", record["estimated_tokens"])
            print("Text preview:")
            print(record["text"][:1000])
            print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Load regulatory PDFs into Neo4j with structure-aware chunking.")
    parser.add_argument("--pdf-folder", default=PDF_FOLDER)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=OVERLAP_TOKENS)
    parser.add_argument("--reset", action="store_true", help="Delete existing Document, Page, Chunk, and Fact nodes before loading.")
    parser.add_argument("--dry-run", action="store_true", help="Extract and report chunks without connecting to Neo4j.")
    return parser.parse_args()


def validate_chunk_settings(target_tokens: int, max_tokens: int, overlap_tokens: int) -> None:
    if target_tokens <= 0 or max_tokens <= 0:
        raise ValueError("Token budgets must be positive.")
    if target_tokens > max_tokens:
        raise ValueError("--target-tokens must be less than or equal to --max-tokens.")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("--overlap-tokens must be non-negative and smaller than --max-tokens.")


def report_chunk_statistics(chunks: List[Dict]) -> None:
    token_counts = sorted(chunk["estimated_tokens"] for chunk in chunks)
    if not token_counts:
        print("No chunks extracted.")
        return
    percentile_index = min(len(token_counts) - 1, int(len(token_counts) * 0.95))
    print("\nChunk statistics:")
    print("Total:", len(chunks))
    print("TOC chunks:", sum(chunk["is_table_of_contents"] for chunk in chunks))
    print("With section ID:", sum(bool(chunk["section_id"]) for chunk in chunks))
    print("Average estimated tokens:", round(sum(token_counts) / len(token_counts), 1))
    print("Median estimated tokens:", token_counts[len(token_counts) // 2])
    print("95th percentile estimated tokens:", token_counts[percentile_index])
    print("Maximum estimated tokens:", token_counts[-1])


def count_existing_chunks(driver) -> int:
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]


def main():
    args = parse_args()
    validate_chunk_settings(args.target_tokens, args.max_tokens, args.overlap_tokens)

    chunks = extract_all_chunks(
        pdf_folder=args.pdf_folder,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    report_chunk_statistics(chunks)

    if args.dry_run:
        print("\nDry run complete. Neo4j was not modified.")
        return

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

        existing_chunks = count_existing_chunks(driver)
        if existing_chunks and not args.reset:
            raise RuntimeError(
                f"Neo4j already contains {existing_chunks} Chunk nodes. "
                "Use --reset to replace them with the new chunking model."
            )

        if args.reset:
            print("Clearing existing project data...")
            clear_existing_data(driver)

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
