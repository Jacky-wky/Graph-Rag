"""Structure-aware chunking for bilingual financial regulatory PDFs."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


CHAPTER_PATTERN = re.compile(r"^(第[一二三四五六七八九十百0-9]+\s*章\b.*|chapter\s+\d+\b.*)$", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^(?P<section>(?:\d+(?:\.\d+){1,5})|(?:\d+\.))\s*(?P<title>.*)$")
LIST_ITEM_PATTERN = re.compile(r"^(?:[•●▪◦]|\([a-zivx0-9]+\)|[a-zivx0-9]+[.)])\s*", re.IGNORECASE)
PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s+)?\d+$", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；!?;])|(?<=[.!?;])\s+(?=[A-Z0-9(])")


@dataclass(frozen=True)
class RegulatoryChunk:
    text: str
    retrieval_text: str
    section_id: str | None
    section_title: str | None
    section_path: str | None
    is_table_of_contents: bool
    estimated_tokens: int
    start_line: int
    end_line: int
    split_part: int
    chunking_method: str = "regulatory_section_v1"


@dataclass
class _Unit:
    lines: list[tuple[int, str]]
    section_id: str | None
    section_title: str | None
    section_path: str | None


def clean_lines(text: str) -> list[str]:
    """Normalize extracted lines without flattening document structure."""
    text = (text or "").replace("\x00", " ").replace("\u00a0", " ")
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        if line in {"\uf0b7", "\u2022", "\u25cf", "\u25aa"}:
            line = "•"
        lines.append(line)
    return lines


def _margin_key(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().casefold()


def detect_repeated_margin_lines(pages: list[list[str]], scan_lines: int = 6) -> set[str]:
    """Find repeated headers and footers that appear across many pages."""
    page_count = len(pages)
    if page_count < 3:
        return set()
    occurrences: Counter[str] = Counter()
    for lines in pages:
        candidates = {_margin_key(line) for line in lines[:scan_lines] + lines[-scan_lines:] if len(line) <= 140}
        occurrences.update(candidates)
    threshold = max(3, math.ceil(page_count * 0.25))
    return {line for line, count in occurrences.items() if line and count >= threshold}


def remove_page_margins(lines: list[str], repeated_lines: set[str], scan_lines: int = 6) -> list[str]:
    """Remove detected repeated margins and standalone page numbers near margins."""
    cleaned = []
    last_index = len(lines) - 1
    for index, line in enumerate(lines):
        is_margin = index < scan_lines or index > last_index - scan_lines
        if is_margin and _margin_key(line) in repeated_lines:
            continue
        if is_margin and PAGE_NUMBER_PATTERN.fullmatch(line):
            continue
        cleaned.append(line)
    return cleaned


def is_table_of_contents(lines: list[str]) -> bool:
    """Detect contents pages so retrieval can exclude or down-rank them."""
    preview = " ".join(lines[:20]).casefold()
    explicit_heading = "目錄" in preview or re.search(r"\bcontents\b", preview) is not None
    leader_lines = sum("..." in line or "……" in line for line in lines)
    return explicit_heading or leader_lines >= 3


def detect_table_of_contents_pages(pages: list[list[str]]) -> list[bool]:
    """Detect explicit contents pages and their immediate continuation pages."""
    flags = [is_table_of_contents(lines) for lines in pages]
    for index in range(1, len(pages)):
        if not flags[index - 1] or flags[index]:
            continue
        lines = pages[index]
        numbered_entries = sum(SECTION_PATTERN.match(line) is not None for line in lines)
        average_length = sum(len(line) for line in lines) / max(len(lines), 1)
        if numbered_entries >= 4 and average_length <= 55:
            flags[index] = True
    return flags


def estimate_tokens(text: str) -> int:
    """Estimate tokens consistently for mixed Chinese and English text."""
    cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))
    other = len(re.findall(r"[^\sA-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]", text))
    return max(1, math.ceil(cjk_count + latin_words * 1.3 + other * 0.35))


def _section_level(section_id: str) -> int:
    return section_id.count(".") + 1


def _build_section_path(chapter: str | None, section_titles: dict[int, str], level: int) -> str | None:
    values = []
    if chapter:
        values.append(chapter)
    values.extend(section_titles[index] for index in sorted(section_titles) if index < level)
    return " > ".join(values) or None


def _starts_structural_line(line: str) -> bool:
    return bool(SECTION_PATTERN.match(line) or CHAPTER_PATTERN.match(line) or LIST_ITEM_PATTERN.match(line))


def _needs_space(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous.endswith("-"):
        return False
    previous_character = previous[-1]
    current_character = current[0]
    cjk = r"[\u3400-\u4dbf\u4e00-\u9fff]"
    if re.match(cjk, previous_character) and re.match(cjk, current_character):
        return False
    if previous_character in "（([/" or current_character in "，。；：、！？,.!?;:)]）/":
        return False
    return True


def reflow_lines(lines: Iterable[str]) -> str:
    """Repair PDF line wrapping while preserving clauses and list boundaries."""
    output = ""
    for line in lines:
        if not output:
            output = line
            continue
        if _starts_structural_line(line):
            output += "\n" + line
            continue
        separator = " " if _needs_space(output.rstrip(), line) else ""
        output += separator + line
    return output.strip()


def _logical_segments(text: str) -> list[str]:
    segments = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentence_parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(paragraph) if part.strip()]
        segments.extend(sentence_parts or [paragraph])
    return segments


def _hard_split(segment: str, max_tokens: int) -> list[str]:
    if estimate_tokens(segment) <= max_tokens:
        return [segment]
    parts = []
    remaining = segment
    while estimate_tokens(remaining) > max_tokens:
        ratio = max_tokens / estimate_tokens(remaining)
        tentative = max(80, int(len(remaining) * ratio))
        search_start = max(1, int(tentative * 0.7))
        break_at = max(
            remaining.rfind(marker, search_start, tentative + 1)
            for marker in ("。", "；", ";", ".", "，", ",", " ")
        )
        if break_at <= 0:
            break_at = tentative
        else:
            break_at += 1
        parts.append(remaining[:break_at].strip())
        remaining = remaining[break_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def split_with_token_budget(
    text: str,
    target_tokens: int = 420,
    max_tokens: int = 650,
    overlap_tokens: int = 70,
) -> list[str]:
    """Split only oversized clauses, preferring sentence and list boundaries."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    source_segments = []
    for segment in _logical_segments(text):
        source_segments.extend(_hard_split(segment, max_tokens))

    chunks = []
    current: list[str] = []
    current_tokens = 0
    for segment in source_segments:
        segment_tokens = estimate_tokens(segment)
        if current and current_tokens + segment_tokens > max_tokens:
            chunks.append("\n".join(current).strip())
            overlap = []
            overlap_size = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous)
                if overlap and overlap_size + previous_tokens > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_size += previous_tokens
            current = overlap
            current_tokens = overlap_size
        current.append(segment)
        current_tokens += segment_tokens
        if current_tokens >= target_tokens:
            chunks.append("\n".join(current).strip())
            current = []
            current_tokens = 0
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _parse_page_units(
    lines: list[str],
    chapter: str | None,
    section_titles: dict[int, str],
    current_section_id: str | None,
) -> tuple[list[_Unit], str | None, dict[int, str], str | None]:
    units: list[_Unit] = []
    current_lines: list[tuple[int, str]] = []
    current_title: str | None = None
    current_path = _build_section_path(chapter, section_titles, 99)

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            units.append(_Unit(current_lines, current_section_id, current_title, current_path))
            current_lines = []

    for line_number, line in enumerate(lines, start=1):
        chapter_match = CHAPTER_PATTERN.match(line)
        if chapter_match:
            flush()
            chapter = line
            section_titles = {}
            current_section_id = None
            current_title = line
            current_path = chapter
            current_lines = [(line_number, line)]
            continue

        section_match = SECTION_PATTERN.match(line)
        if section_match:
            flush()
            section_id = section_match.group("section").rstrip(".")
            title = section_match.group("title").strip()
            level = _section_level(section_id)
            section_titles = {key: value for key, value in section_titles.items() if key < level}
            if title:
                section_titles[level] = f"{section_id} {title}"
            current_section_id = section_id
            current_title = title or None
            current_path = _build_section_path(chapter, section_titles, level)
            current_lines = [(line_number, line)]
            continue

        if (
            current_section_id
            and current_title is None
            and _section_level(current_section_id) <= 2
            and len(current_lines) == 1
            and len(line) <= 120
            and not _starts_structural_line(line)
            and not re.search(r"[。！？.!?]$", line)
        ):
            current_title = line
            level = _section_level(current_section_id)
            section_titles[level] = f"{current_section_id} {line}"
            current_path = _build_section_path(chapter, section_titles, level)
            current_lines.append((line_number, line))
            continue

        current_lines.append((line_number, line))

    flush()
    return units, chapter, section_titles, current_section_id


def chunk_document_pages(
    page_texts: list[str],
    target_tokens: int = 420,
    max_tokens: int = 650,
    overlap_tokens: int = 70,
) -> list[list[RegulatoryChunk]]:
    """Chunk a complete document while carrying section context across pages."""
    raw_pages = [clean_lines(text) for text in page_texts]
    repeated_lines = detect_repeated_margin_lines(raw_pages)
    prepared_pages = [remove_page_margins(lines, repeated_lines) for lines in raw_pages]
    toc_flags = detect_table_of_contents_pages(prepared_pages)

    chapter: str | None = None
    section_titles: dict[int, str] = {}
    current_section_id: str | None = None
    output: list[list[RegulatoryChunk]] = []

    for page_index, lines in enumerate(prepared_pages):
        toc = toc_flags[page_index]
        units, chapter, section_titles, current_section_id = _parse_page_units(
            lines,
            chapter,
            section_titles,
            current_section_id,
        )
        page_chunks: list[RegulatoryChunk] = []
        for unit in units:
            text = reflow_lines(line for _, line in unit.lines)
            if not text:
                continue
            section_heading = SECTION_PATTERN.fullmatch(text)
            if CHAPTER_PATTERN.match(text):
                continue
            if (
                section_heading
                and estimate_tokens(text) <= 35
                and not re.search(r"\b(?:must|should|required)\b|應|須|不得|禁止", text, re.IGNORECASE)
            ):
                continue
            split_texts = split_with_token_budget(text, target_tokens, max_tokens, overlap_tokens)
            for part_index, chunk_text in enumerate(split_texts, start=1):
                context_values = [value for value in (unit.section_path, unit.section_title) if value]
                context = " > ".join(dict.fromkeys(context_values))
                retrieval_text = f"{context}\n{chunk_text}" if context and context not in chunk_text else chunk_text
                page_chunks.append(
                    RegulatoryChunk(
                        text=chunk_text,
                        retrieval_text=retrieval_text,
                        section_id=unit.section_id,
                        section_title=unit.section_title,
                        section_path=unit.section_path,
                        is_table_of_contents=toc,
                        estimated_tokens=estimate_tokens(retrieval_text),
                        start_line=unit.lines[0][0],
                        end_line=unit.lines[-1][0],
                        split_part=part_index,
                    )
                )
        output.append(page_chunks)
    return output
