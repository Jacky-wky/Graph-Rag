"""Export every structure-aware PDF chunk as a reviewable Markdown file."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from load_pdf_into_neo4j import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    PDF_FOLDER,
    TARGET_TOKENS,
    extract_chunks_from_pdf,
    list_pdf_files,
    validate_chunk_settings,
)


IMPLEMENT_CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = IMPLEMENT_CODE_DIR / "chunk_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one Markdown review file for every generated PDF chunk."
    )
    parser.add_argument("--pdf-folder", default=PDF_FOLDER, help="Folder containing source PDFs.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated Markdown files.",
    )
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=OVERLAP_TOKENS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing generated Markdown files to be replaced.",
    )
    return parser.parse_args()


def safe_path_component(value: str, fallback: str = "untitled") -> str:
    """Return a readable path component valid on Windows."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or fallback


def yaml_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def front_matter(metadata: list[tuple[str, object]]) -> str:
    lines = ["---"]
    lines.extend(f"{key}: {yaml_value(value)}" for key, value in metadata)
    lines.append("---")
    return "\n".join(lines)


def chunk_filename(chunk: dict) -> str:
    section = safe_path_component(chunk.get("section_id") or "no-section")
    split_suffix = f"_part-{chunk['split_part']:02d}" if chunk["split_part"] > 1 else ""
    return (
        f"p{chunk['page']:04d}_c{chunk['chunk_index']:03d}_"
        f"{section}{split_suffix}.md"
    )


def source_pdf_link(chunk_file: Path, pdf_path: Path, page: int) -> str:
    relative_path = Path(
        os.path.relpath(pdf_path.resolve(), start=chunk_file.parent.resolve())
    ).as_posix()
    return f"{quote(relative_path, safe='/')}#page={page}"


def render_chunk_markdown(chunk: dict, chunk_file: Path, pdf_path: Path) -> str:
    metadata = [
        ("chunk_id", chunk["chunk_id"]),
        ("source_file", chunk["source_file"]),
        ("document_sha256", chunk["document_sha256"]),
        ("pdf_page", chunk["page"]),
        ("chunk_index", chunk["chunk_index"]),
        ("section_id", chunk["section_id"]),
        ("section_title", chunk["section_title"]),
        ("section_path", chunk["section_path"]),
        ("is_table_of_contents", chunk["is_table_of_contents"]),
        ("estimated_tokens", chunk["estimated_tokens"]),
        ("source_start_line", chunk["start_line"]),
        ("source_end_line", chunk["end_line"]),
        ("split_part", chunk["split_part"]),
        ("chunking_method", chunk["chunking_method"]),
    ]
    pdf_link = source_pdf_link(chunk_file, pdf_path, chunk["page"])
    return (
        f"{front_matter(metadata)}\n\n"
        f"# {chunk['chunk_id']}\n\n"
        f"[Open source PDF at page {chunk['page']}]({pdf_link})\n\n"
        "## Retrieval Context\n\n"
        f"{chunk['retrieval_text']}\n\n"
        "## Source Text\n\n"
        f"{chunk['text']}\n"
    )


def ensure_output_is_safe(output_dir: Path, overwrite: bool) -> None:
    existing_markdown = next(output_dir.rglob("*.md"), None) if output_dir.exists() else None
    if existing_markdown and not overwrite:
        raise FileExistsError(
            f"Markdown files already exist in {output_dir}. "
            "Use --overwrite to replace generated files."
        )


def write_document_index(
    document_dir: Path,
    source_file: str,
    document_sha256: str,
    chunk_entries: list[tuple[dict, Path]],
) -> None:
    toc_count = sum(chunk["is_table_of_contents"] for chunk, _ in chunk_entries)
    lines = [
        f"# {source_file}",
        "",
        f"- SHA-256: `{document_sha256}`",
        f"- Chunks: {len(chunk_entries)}",
        f"- Table-of-contents chunks: {toc_count}",
        "",
        "| Page | Chunk | Section | Tokens | TOC | Review file |",
        "| ---: | ---: | --- | ---: | :---: | --- |",
    ]
    for chunk, chunk_file in chunk_entries:
        section = chunk["section_id"] or "—"
        toc = "yes" if chunk["is_table_of_contents"] else "no"
        lines.append(
            f"| {chunk['page']} | {chunk['chunk_index']} | {section} | "
            f"{chunk['estimated_tokens']} | {toc} | "
            f"[{chunk_file.name}]({quote(chunk_file.name)}) |"
        )
    (document_dir / "_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_index(
    output_dir: Path,
    document_entries: list[tuple[str, Path, int, int]],
    settings: dict[str, int],
) -> None:
    total_chunks = sum(chunk_count for _, _, chunk_count, _ in document_entries)
    total_toc = sum(toc_count for _, _, _, toc_count in document_entries)
    lines = [
        "# Chunk Review Index",
        "",
        "This directory is generated from the source PDFs without writing to Neo4j.",
        "",
        f"- Documents: {len(document_entries)}",
        f"- Chunks: {total_chunks}",
        f"- Table-of-contents chunks: {total_toc}",
        f"- Target tokens: {settings['target_tokens']}",
        f"- Maximum tokens: {settings['max_tokens']}",
        f"- Overlap tokens: {settings['overlap_tokens']}",
        "",
        "| Document | Chunks | TOC chunks | Review index |",
        "| --- | ---: | ---: | --- |",
    ]
    for source_file, document_dir, chunk_count, toc_count in document_entries:
        relative_index = (document_dir / "_index.md").relative_to(output_dir).as_posix()
        lines.append(
            f"| {source_file} | {chunk_count} | {toc_count} | "
            f"[Open]({quote(relative_index, safe='/')}) |"
        )
    (output_dir / "_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_chunks(args: argparse.Namespace) -> tuple[int, int]:
    validate_chunk_settings(args.target_tokens, args.max_tokens, args.overlap_tokens)
    pdf_folder = Path(args.pdf_folder).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not pdf_folder.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    ensure_output_is_safe(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    document_entries: list[tuple[str, Path, int, int]] = []
    total_chunks = 0
    for pdf_name in list_pdf_files(str(pdf_folder)):
        pdf_path = Path(pdf_name)
        chunks = extract_chunks_from_pdf(
            str(pdf_path),
            target_tokens=args.target_tokens,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
        )
        document_dir = output_dir / safe_path_component(pdf_path.stem)
        document_dir.mkdir(parents=True, exist_ok=True)

        chunk_entries = []
        for chunk in chunks:
            chunk_file = document_dir / chunk_filename(chunk)
            chunk_file.write_text(
                render_chunk_markdown(chunk, chunk_file, pdf_path),
                encoding="utf-8",
            )
            chunk_entries.append((chunk, chunk_file))

        write_document_index(
            document_dir,
            chunks[0]["source_file"] if chunks else pdf_path.name,
            chunks[0]["document_sha256"] if chunks else "",
            chunk_entries,
        )
        toc_count = sum(chunk["is_table_of_contents"] for chunk in chunks)
        document_entries.append((pdf_path.name, document_dir, len(chunks), toc_count))
        total_chunks += len(chunks)
        print(f"{pdf_path.name}: {len(chunks)} chunks")

    settings = {
        "target_tokens": args.target_tokens,
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
    }
    write_root_index(output_dir, document_entries, settings)
    return len(document_entries), total_chunks


def main() -> None:
    args = parse_args()
    document_count, chunk_count = export_chunks(args)
    print(f"\nExported {chunk_count} chunks from {document_count} PDFs.")
    print(f"Review index: {Path(args.output_dir).resolve() / '_index.md'}")


if __name__ == "__main__":
    main()
