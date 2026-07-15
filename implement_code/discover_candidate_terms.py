# discover_candidate_terms.py
# -*- coding: utf-8 -*-

"""
Discover candidate terms from existing Chunk nodes in Neo4j.

This script scans (:Chunk {text: ...}) nodes and extracts frequent Chinese /
English candidate phrases that are not already present in concept_config.TERMS.

It creates:

(:CandidateTerm {
    candidate_id: "...",
    text: "...",
    normalized_text: "...",
    language: "...",
    frequency: ...,
    document_frequency: ...,
    source: "auto_discovery",
    status: "pending",
    examples: [...],
    created_at: "...",
    updated_at: "..."
})

Optional relationship if a candidate appears close to an existing known term:

(:CandidateTerm)-[:CANDIDATE_FOR]->(:Concept)

Expected existing Chunk nodes:

(:Chunk {
    chunk_id: "...",
    text: "...",
    source_file: "...",
    page: ...,
    chunk_index: ...
})

Usage:

    python implement_code/discover_candidate_terms.py

Common options:

    python implement_code/discover_candidate_terms.py --clear-pending
    python implement_code/discover_candidate_terms.py --min-freq 5
    python implement_code/discover_candidate_terms.py --max-candidates 300
    python implement_code/discover_candidate_terms.py --batch-size 500
    python implement_code/discover_candidate_terms.py --debug-config

Environment variables in project root .env:

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=password
    NEO4J_DATABASE=neo4j

Also supported:

    NEO4J_URL
    NEO4J_USERNAME
    NEO4J_PASS
    NEO4J_DB
"""

import argparse
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
IMPLEMENT_CODE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = IMPLEMENT_CODE_DIR.parent

# Make sure imports work when running from project root:
#     python implement_code/discover_candidate_terms.py
# or from implement_code:
#     python discover_candidate_terms.py
if str(IMPLEMENT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(IMPLEMENT_CODE_DIR))


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------

def load_env_if_available() -> None:
    """
    Load .env from project root.

    Because all code files are in implement_code/, this resolves:
        industrial_project/.env
    """
    if load_dotenv is None:
        return

    env_path = PROJECT_ROOT / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        # fallback: allow python-dotenv to search current working directory
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
# Import concept config
# ---------------------------------------------------------------------

try:
    from concept_config import TERMS, STOPWORDS
except ImportError as exc:
    raise ImportError(
        "Cannot import TERMS and STOPWORDS from concept_config.py. "
        "Please make sure concept_config.py is inside implement_code/."
    ) from exc


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_candidate_text(text: str) -> str:
    text = normalize_space(text)
    return text.lower()


def make_candidate_id(text: str) -> str:
    """
    Use hash to avoid illegal characters and over-long ids.
    """
    normalized = normalize_candidate_text(text)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"cand_{digest}"


def detect_language(text: str) -> str:
    """
    Very simple language detector.

    Returns:
        zh      contains Chinese characters
        en      mostly Latin letters / numbers
        mixed   contains both Chinese and Latin
        unknown otherwise
    """
    if not text:
        return "unknown"

    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))

    if has_zh and has_en:
        return "mixed"

    if has_zh:
        return "zh"

    if has_en:
        return "en"

    return "unknown"


def batched(items: List, batch_size: int) -> Iterable[List]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def compact_example(text: str, candidate: str, window: int = 60) -> str:
    """
    Return a short context example around candidate.
    """
    if not text or not candidate:
        return ""

    text_norm = normalize_space(text)
    idx = text_norm.lower().find(candidate.lower())

    if idx < 0:
        return text_norm[: window * 2]

    start = max(0, idx - window)
    end = min(len(text_norm), idx + len(candidate) + window)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text_norm) else ""

    return prefix + text_norm[start:end] + suffix


# ---------------------------------------------------------------------
# Candidate extraction helpers
# ---------------------------------------------------------------------

def get_known_term_texts() -> Set[str]:
    """
    Existing manually defined terms should not be rediscovered.
    """
    known = set()

    for term in TERMS:
        text = term.get("text")
        if not text:
            continue

        known.add(normalize_candidate_text(text))

    return known


def get_stopwords() -> Set[str]:
    """
    STOPWORDS comes from concept_config.py.

    These are not necessarily traditional NLP stopwords.
    They are domain noise words for candidate discovery.
    """
    return {
        normalize_candidate_text(x)
        for x in STOPWORDS
        if x and str(x).strip()
    }


def remove_noise_edges(s: str) -> str:
    """
    Remove punctuation and common bracket characters from both ends.
    """
    if not s:
        return ""

    s = s.strip()

    edge_chars = (
        " \t\r\n"
        ".,;:!?，。；：！？"
        "()[]{}（）【】「」『』《》〈〉"
        "\"'“”‘’"
        "/\\|"
        "、"
        "-"
        "—"
        "_"
    )

    return s.strip(edge_chars)


def is_probably_number_or_date(s: str) -> bool:
    if not s:
        return True

    value = s.strip()

    patterns = [
        r"^\d+$",
        r"^\d+\.\d+$",
        r"^\d+%$",
        r"^\d{4}$",
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",
        r"^\d{4}-\d{1,2}-\d{1,2}$",
        r"^\d+[\.\)]$",
        r"^\([a-zA-Z0-9]+\)$",
        r"^[ivxlcdmIVXLCDM]+$",
    ]

    return any(re.match(p, value) for p in patterns)


def is_bad_candidate(
    s: str,
    stopwords: Set[str],
    known_terms: Set[str],
    min_chars: int = 2,
    max_chars: int = 30,
) -> bool:
    """
    Return True if candidate should be rejected.

    Important:
    - STOPWORDS only affect candidate discovery.
    - STOPWORDS do not affect TERMS matching in improve_relationships.py.
    """
    if not s:
        return True

    s = remove_noise_edges(s)
    s_norm = normalize_candidate_text(s)

    if not s_norm:
        return True

    if len(s_norm) < min_chars:
        return True

    if len(s_norm) > max_chars:
        return True

    if s_norm in known_terms:
        return True

    if s_norm in stopwords:
        return True

    if is_probably_number_or_date(s_norm):
        return True

    # Reject candidates made only of punctuation / symbols.
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s_norm):
        return True

    # Reject if candidate is only English single character repeated or code-like.
    if re.match(r"^[a-zA-Z]$", s_norm):
        return True

    # Reject mostly punctuation.
    alnum_zh_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", s_norm))
    if alnum_zh_count / max(len(s_norm), 1) < 0.5:
        return True

    # Reject very generic short candidates containing stopword.
    # Example:
    #   stopword = "規定"
    #   candidate = "該規定" or "規定的"
    for sw in stopwords:
        if not sw:
            continue
        if sw in s_norm and len(s_norm) <= len(sw) + 1:
            return True

    # Reject phrases starting or ending with common function words.
    bad_prefixes = [
        "的",
        "及",
        "和",
        "與",
        "或",
        "在",
        "按",
        "由",
        "對",
        "就",
        "於",
        "其",
        "該",
        "有關",
        "相關",
    ]

    bad_suffixes = [
        "的",
        "及",
        "和",
        "與",
        "或",
        "在",
        "按",
        "由",
        "對",
        "就",
        "於",
        "等",
        "者",
    ]

    if any(s_norm.startswith(x) for x in bad_prefixes):
        return True

    if any(s_norm.endswith(x) for x in bad_suffixes):
        return True

    return False


def extract_chinese_candidates(text: str) -> List[str]:
    """
    Extract Chinese candidate phrases.

    Strategy:
    - Split text by punctuation and whitespace.
    - Keep continuous Chinese sequences.
    - Generate n-grams of length 2 to 8 Chinese characters.
    - Longer candidates are useful for terms like:
        客戶盡職審查
        可疑交易報告
        資本充足比率
    """
    if not text:
        return []

    candidates = []

    # Continuous Chinese sequences.
    sequences = re.findall(r"[\u4e00-\u9fff]{2,}", text)

    min_n = 2
    max_n = 8

    for seq in sequences:
        seq = remove_noise_edges(seq)

        if len(seq) < min_n:
            continue

        # Add whole sequence if not too long.
        if min_n <= len(seq) <= 12:
            candidates.append(seq)

        # Add n-grams.
        upper = min(max_n, len(seq))
        for n in range(min_n, upper + 1):
            for i in range(0, len(seq) - n + 1):
                candidates.append(seq[i : i + n])

    return candidates


def extract_english_candidates(text: str) -> List[str]:
    """
    Extract English / acronym candidate phrases.

    Examples:
        AML
        CFT
        customer due diligence
        beneficial owner
        risk-based approach
    """
    if not text:
        return []

    candidates = []

    # Acronyms, e.g. AML, CFT, KYC, PEPs.
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,9}s?\b", text)
    candidates.extend(acronyms)

    # English phrase chunks.
    # This keeps words with hyphen, e.g. risk-based.
    phrase_pattern = r"\b[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){1,5}\b"
    phrases = re.findall(phrase_pattern, text)

    for phrase in phrases:
        phrase = normalize_space(phrase)

        # Add whole phrase.
        candidates.append(phrase)

        # Add 2-4 word ngrams from phrase.
        words = phrase.split()
        for n in range(2, min(4, len(words)) + 1):
            for i in range(0, len(words) - n + 1):
                candidates.append(" ".join(words[i : i + n]))

    return candidates


def extract_mixed_candidates(text: str) -> List[str]:
    """
    Extract mixed candidates such as:
        AML風險
        CDD措施
        KYC程序
    """
    if not text:
        return []

    pattern = r"[A-Za-z]{2,10}[\u4e00-\u9fff]{1,8}|[\u4e00-\u9fff]{1,8}[A-Za-z]{2,10}"
    return re.findall(pattern, text)


def extract_candidates_from_text(text: str) -> List[str]:
    if not text:
        return []

    text = normalize_space(text)

    candidates = []
    candidates.extend(extract_chinese_candidates(text))
    candidates.extend(extract_english_candidates(text))
    candidates.extend(extract_mixed_candidates(text))

    cleaned = []

    for c in candidates:
        c = remove_noise_edges(c)
        c = normalize_space(c)
        if c:
            cleaned.append(c)

    return cleaned


# ---------------------------------------------------------------------
# Main discoverer
# ---------------------------------------------------------------------

class CandidateTermDiscoverer:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None,
        batch_size: int = 500,
        min_freq: int = 5,
        min_doc_freq: int = 2,
        max_candidates: int = 300,
        max_examples_per_candidate: int = 3,
        link_to_concepts: bool = True,
        debug_config: bool = False,
    ):
        self.neo4j_uri = uri
        self.neo4j_user = user
        self.neo4j_password = password
        self.neo4j_database = database

        self.batch_size = batch_size
        self.min_freq = min_freq
        self.min_doc_freq = min_doc_freq
        self.max_candidates = max_candidates
        self.max_examples_per_candidate = max_examples_per_candidate
        self.link_to_concepts = link_to_concepts

        self.known_terms = get_known_term_texts()
        self.stopwords = get_stopwords()

        if not self.neo4j_password:
            raise ValueError(
                "Missing Neo4j password. Please set NEO4J_PASSWORD in your .env file "
                "or pass --password."
            )

        if debug_config:
            print("[Project paths]")
            print("CURRENT_FILE =", str(CURRENT_FILE))
            print("IMPLEMENT_CODE_DIR =", str(IMPLEMENT_CODE_DIR))
            print("PROJECT_ROOT =", str(PROJECT_ROOT))
            print(".env path =", str(PROJECT_ROOT / ".env"))
            print(".env exists =", (PROJECT_ROOT / ".env").exists())

            print("\n[Neo4j config]")
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

            print("\n[Discovery config]")
            print("Known terms =", len(self.known_terms))
            print("Stopwords =", len(self.stopwords))
            print("Batch size =", self.batch_size)
            print("Min freq =", self.min_freq)
            print("Min document freq =", self.min_doc_freq)
            print("Max candidates =", self.max_candidates)

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

    def run(self, clear_pending: bool = False):
        with self.get_session() as session:
            print("Creating constraints...")
            self.create_constraints(session)

            if clear_pending:
                print("Clearing old pending CandidateTerm nodes...")
                self.clear_pending_candidates(session)

            print("Scanning chunks and extracting candidates...")
            candidate_rows = self.discover_candidates(session)

            if not candidate_rows:
                print("No candidate terms discovered.")
                return

            print(f"Writing {len(candidate_rows)} CandidateTerm nodes...")
            self.write_candidate_terms(session, candidate_rows)

            if self.link_to_concepts:
                print("Linking candidates to possible concepts...")
                link_count = self.link_candidates_to_concepts(session)
                print(f"Candidate-Concept links created/updated: {link_count}")

            print("\nDone.")
            print(f"Candidate terms written: {len(candidate_rows)}")

    def create_constraints(self, session):
        queries = [
            "CREATE CONSTRAINT candidate_id_unique IF NOT EXISTS FOR (ct:CandidateTerm) REQUIRE ct.candidate_id IS UNIQUE",
            "CREATE INDEX candidate_text_index IF NOT EXISTS FOR (ct:CandidateTerm) ON (ct.text)",
            "CREATE INDEX candidate_status_index IF NOT EXISTS FOR (ct:CandidateTerm) ON (ct.status)",
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (ch:Chunk) ON (ch.chunk_id)",
            "CREATE INDEX concept_id_index IF NOT EXISTS FOR (c:Concept) ON (c.concept_id)",
            "CREATE INDEX term_text_index IF NOT EXISTS FOR (t:Term) ON (t.text)",
        ]

        for q in queries:
            session.run(q)

    def clear_pending_candidates(self, session):
        """
        Delete only pending auto-discovered candidates.

        This preserves accepted/rejected candidates if you later build review flow.
        """
        query = """
        MATCH (ct:CandidateTerm)
        WHERE ct.status = "pending"
           OR ct.source = "auto_discovery"
        DETACH DELETE ct
        """

        session.run(query)

    def count_chunks(self, session) -> int:
        query = """
        MATCH (ch:Chunk)
        WHERE ch.text IS NOT NULL
        RETURN count(ch) AS n
        """

        result = session.run(query)
        record = result.single()
        return record["n"] if record else 0

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

    def discover_candidates(self, session) -> List[Dict]:
        total_chunks = self.count_chunks(session)

        if total_chunks == 0:
            print("No Chunk nodes with text found.")
            return []

        frequency = Counter()
        document_frequency = Counter()
        examples = defaultdict(list)
        source_files = defaultdict(set)

        skip = 0
        chunks_scanned = 0

        while skip < total_chunks:
            chunks = self.fetch_chunks_batch(session, skip, self.batch_size)

            if not chunks:
                break

            for chunk in chunks:
                text = chunk.get("text") or ""
                chunk_id = chunk.get("chunk_id")
                source_file = chunk.get("source_file")
                page = chunk.get("page")
                chunk_index = chunk.get("chunk_index")

                raw_candidates = extract_candidates_from_text(text)

                valid_candidates = []

                for candidate in raw_candidates:
                    candidate = remove_noise_edges(candidate)
                    candidate = normalize_space(candidate)

                    if is_bad_candidate(
                        candidate,
                        stopwords=self.stopwords,
                        known_terms=self.known_terms,
                    ):
                        continue

                    candidate_norm = normalize_candidate_text(candidate)

                    valid_candidates.append(candidate_norm)

                    frequency[candidate_norm] += 1

                    if len(examples[candidate_norm]) < self.max_examples_per_candidate:
                        examples[candidate_norm].append(
                            {
                                "chunk_id": chunk_id,
                                "source_file": source_file,
                                "page": page,
                                "chunk_index": chunk_index,
                                "example": compact_example(text, candidate),
                            }
                        )

                    if source_file:
                        source_files[candidate_norm].add(source_file)

                for candidate_norm in set(valid_candidates):
                    document_frequency[candidate_norm] += 1

                chunks_scanned += 1

            skip += self.batch_size
            print(f"Processed {min(skip, total_chunks)} / {total_chunks} chunks")

        print(f"Chunks scanned: {chunks_scanned}")
        print(f"Raw unique candidates after filtering: {len(frequency)}")

        rows = []

        for text_norm, freq in frequency.most_common():
            doc_freq = document_frequency[text_norm]

            if freq < self.min_freq:
                continue

            if doc_freq < self.min_doc_freq:
                continue

            display_text = self.choose_display_text(text_norm)

            row = {
                "candidate_id": make_candidate_id(text_norm),
                "text": display_text,
                "normalized_text": text_norm,
                "language": detect_language(display_text),
                "frequency": int(freq),
                "document_frequency": int(doc_freq),
                "source_file_count": len(source_files[text_norm]),
                "source": "auto_discovery",
                "status": "pending",
                "examples": examples[text_norm],
            }

            rows.append(row)

            if len(rows) >= self.max_candidates:
                break

        return rows

    def choose_display_text(self, normalized_text: str) -> str:
        """
        Currently normalized_text is lower-case for English.
        This method can be improved later by storing the most common original form.
        """
        return normalized_text

    def write_candidate_terms(self, session, rows: List[Dict]):
        """
        Write CandidateTerm nodes.

        Important:
        Neo4j properties cannot store arrays of maps.
        Therefore examples are converted to a JSON string before writing.
        """

        import json

        prepared_rows = []

        for row in rows:
            prepared = dict(row)

            # Neo4j cannot store List[Dict] as a property.
            # Store it as JSON string instead.
            prepared["examples_json"] = json.dumps(
                row.get("examples", []),
                ensure_ascii=False,
            )

            # Optional primitive list version for easier viewing in Neo4j Browser.
            example_texts = []
            for ex in row.get("examples", []):
                if isinstance(ex, dict):
                    example_text = ex.get("example")
                    if example_text:
                        example_texts.append(str(example_text))

            prepared["example_texts"] = example_texts

            # Remove map-list property before sending to Cypher.
            prepared.pop("examples", None)

            prepared_rows.append(prepared)

        query = """
        UNWIND $rows AS row
        MERGE (ct:CandidateTerm {candidate_id: row.candidate_id})
        SET ct.text = row.text,
            ct.normalized_text = row.normalized_text,
            ct.language = row.language,
            ct.frequency = row.frequency,
            ct.document_frequency = row.document_frequency,
            ct.source_file_count = row.source_file_count,
            ct.source = row.source,
            ct.status = coalesce(ct.status, row.status),
            ct.examples_json = row.examples_json,
            ct.example_texts = row.example_texts,
            ct.updated_at = $now
        SET ct.created_at = coalesce(ct.created_at, $now)
        """

        session.run(query, rows=prepared_rows, now=utc_now_iso())
    def link_candidates_to_concepts(self, session) -> int:
        """
        Weak heuristic:
        If a candidate text contains an existing Term text,
        link CandidateTerm to the Term's Concept.

        Example:
            candidate: 高風險客戶
            existing term: 客戶
            candidate -> concept of 客戶

        This is only a suggestion link for review.
        """
        query = """
        MATCH (ct:CandidateTerm)
        WHERE ct.status = "pending"
        MATCH (t:Term)-[:NORMALIZED_TO]->(c:Concept)
        WHERE ct.normalized_text CONTAINS toLower(t.text)
          AND size(toLower(t.text)) >= 2
        MERGE (ct)-[r:CANDIDATE_FOR]->(c)
        SET r.method = "contains_existing_term",
            r.updated_at = $now
        SET r.created_at = coalesce(r.created_at, $now)
        RETURN count(r) AS n
        """

        result = session.run(query, now=utc_now_iso())
        record = result.single()
        return record["n"] if record else 0


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
        "--min-freq",
        type=int,
        default=5,
        help="Minimum total frequency for a candidate term.",
    )

    parser.add_argument(
        "--min-doc-freq",
        type=int,
        default=2,
        help="Minimum number of chunks containing the candidate.",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=300,
        help="Maximum number of candidate terms to write.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Maximum examples stored per candidate.",
    )

    parser.add_argument(
        "--clear-pending",
        action="store_true",
        help="Delete old pending auto-discovered CandidateTerm nodes before discovery.",
    )

    parser.add_argument(
        "--no-link-concepts",
        action="store_true",
        help="Do not create CandidateTerm-[:CANDIDATE_FOR]->Concept suggestion links.",
    )

    parser.add_argument(
        "--debug-config",
        action="store_true",
        help="Print Neo4j config debug information without exposing full password.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    discoverer = CandidateTermDiscoverer(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        min_doc_freq=args.min_doc_freq,
        max_candidates=args.max_candidates,
        max_examples_per_candidate=args.max_examples,
        link_to_concepts=not args.no_link_concepts,
        debug_config=args.debug_config,
    )

    try:
        discoverer.run(clear_pending=args.clear_pending)
    finally:
        discoverer.close()


if __name__ == "__main__":
    main()