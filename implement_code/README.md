# Neo4j Agentic RAG

This folder contains an end-to-end document knowledge-graph RAG prototype:

1. `load_pdf_into_neo4j.py` creates `Document`, `Page`, and structure-aware `Chunk` nodes from PDFs.
2. `improve_relationships.py` automatically induces `Concept`, `Term`, and taxonomy nodes from Chunk embeddings.
3. `local_graph_rag.py` resolves questions against the induced ontology with a local cross-lingual model.
4. `extract_provenance_facts.py` optionally extracts `Fact` nodes, each supported by an exact chunk quote.
5. `review_provenance_facts.py` optionally approves or rejects facts before they are trusted knowledge.
6. `agentic_rag.py` lets an LLM decide whether to retrieve graph-grounded evidence, full-text chunk evidence, or both before composing a cited answer.

## Provenance-first graph model

Every extracted fact has an auditable evidence path:

```text
(:Document)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk)
(:OntologyRun)-[:USED_CHUNK]->(:Chunk)
(:OntologyRun)-[:GENERATED]->(:Concept)
(:Chunk)-[:ABOUT_CONCEPT {confidence, method}]->(:Concept)
(:Concept)-[:REPRESENTED_BY {rank, similarity}]->(:Chunk)
(:Chunk)-[:MENTIONS_TERM {match_scope, count}]->(:Term)-[:NORMALIZED_TO]->(:Concept)
(:Chunk)-[:SUPPORTS {quote, confidence}]->(:Fact {status})
(:Concept)-[:SUBJECT_OF]->(:Fact)-[:OBJECT_OF]->(:Concept)
```

`Fact` is the source of truth for semantic relations. Facts are `pending` by default and must be reviewed before application retrieval treats them as trusted knowledge.

## Graph prerequisites

Run ingestion and relationship construction before asking questions. Create the optional full-text index for better exact-keyword retrieval:

```cypher
CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
FOR (chunk:Chunk) ON EACH [chunk.text];
```

The agent falls back to a slower case-insensitive text match when this index does not exist.

## Setup

```powershell
cd implement_code
conda activate rag
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the Neo4j credentials in `.env`. OpenAI credentials are optional and only
needed for API-based fact extraction or answer generation. Do not commit this
file.

## Build the knowledge graph

The loader uses regulatory section-aware chunking. It removes repeated page
headers and footers, detects contents pages, keeps numbered clauses intact, and
splits only oversized clauses at sentence/list boundaries. It carries chapter
and parent-section context in `retrieval_text` while preserving exact source
text in `text`.

Test all PDFs without modifying Neo4j:

```powershell
python load_pdf_into_neo4j.py --dry-run
```

Optionally export every generated chunk to a separate Markdown file for audit.
This also does not connect to or modify Neo4j:

```powershell
python export_chunks_to_markdown.py
```

Open `chunk_review/_index.md` to browse all documents and chunks. Re-run with
`--overwrite` when review files already exist.

Chunk metadata includes `section_id`, `section_title`, `section_path`,
`estimated_tokens`, source line positions, `is_table_of_contents`, and
`chunking_method`. Defaults are tuned for the nine included HKMA documents:

```text
target_tokens = 420
max_tokens = 650
overlap_tokens = 70
```

To replace existing chunks, use the explicit reset flag. Reset also removes
Facts because their evidence links refer to the previous chunks. Rebuild the
taxonomy and mentions immediately afterwards:

```powershell
python load_pdf_into_neo4j.py --reset
python improve_relationships.py --clear-old
```

The loader refuses to overwrite an existing Chunk graph unless `--reset` is
provided.

## Automatic concept induction

The semantic layer does not use a manually maintained concept dictionary.
`improve_relationships.py` runs a local, reproducible pipeline:

1. Embed every non-TOC Chunk with `intfloat/multilingual-e5-small`.
2. Apply PCA and HDBSCAN to discover the number of leaf topics automatically.
3. Reassign HDBSCAN noise to its nearest semantic centroid.
4. Extract bilingual topic terms with automatic Chinese segmentation and
   class-based TF-IDF.
5. Build parent topics with agglomerative clustering selected by silhouette
   score.
6. Weight exact Term evidence automatically with phrase length, inverse corpus
   frequency, and class-based TF-IDF before deriving `MENTIONS_CONCEPT`.
7. Precompute resolver embeddings for every leaf Concept name and Term, then
   store all embeddings, confidence, exact matches, representative source
   Chunks, model names, settings, and corpus hash in Neo4j.

The first run downloads the local embedding models. Chunk embeddings are cached under
`.cache/concept_induction`, so an unchanged corpus does not need to be encoded
again.

Concept induction and question resolution deliberately use different local
models. E5-small is efficient for clustering all Chunks. The resolver uses
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which is tuned
for cross-lingual semantic similarity. Its Concept and Term vectors are stored
on the corresponding Neo4j nodes, so each query only encodes the question.
Override either model when needed:

```powershell
python improve_relationships.py --dry-run `
  --embedding-model intfloat/multilingual-e5-small `
  --resolver-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Preview the automatically induced ontology without changing Neo4j:

```powershell
python improve_relationships.py --dry-run
```

The preview is written to `auto_concepts_preview.json`. Apply it by replacing
the old semantic layer:

```powershell
python improve_relationships.py --clear-old
```

`--clear-old` preserves `Document`, `Page`, `Chunk`, and `NEXT`, but replaces
old Concept, Term, CandidateTerm, OntologyRun, and Fact data because those
semantic nodes refer to the previous ontology.

## Extract and review facts

Use a small limit for the first extraction run. The default `rules` mode is local and free: it creates low-confidence `RELATED_TO` candidates when two known concepts occur in the same chunk. The `llm` mode can extract typed predicates, but requires an OpenAI API key and available API quota. Both modes only use known `Concept` nodes and retain an exact source quote.

```powershell
python extract_provenance_facts.py --mode rules --limit 20
python review_provenance_facts.py list
python review_provenance_facts.py approve fact_your_fact_id
```

Use LLM extraction only when needed:

```powershell
python extract_provenance_facts.py --mode llm --limit 20
```

Use `--auto-approve` only after you have evaluated the extractor on representative documents.

## Ask a question

```powershell
python agentic_rag.py "What controls are required for customer due diligence?"
```

The answer is grounded in retrieved chunks and uses citations such as `[policy.pdf p.4]`.

## Local retrieval without an LLM API

Use the layered local resolver when working with Codex or when no OpenAI API
key is available. It tries exact and fuzzy auto-generated Terms, then uses the
local cross-lingual resolver model with MaxSim over each Concept name and its
automatically induced Terms, and finally uses a lexical Chunk fallback. Read
`RETRIEVAL_POLICY.md` for the complete confidence and matching policy.

```powershell
python local_graph_rag.py "customer due diligence 有哪些控制要求？"
```

## Retrieval design

- `search_graph` finds matching concepts/terms, expands up to two taxonomy levels, then retrieves related chunks.
- `search_chunks` uses the Neo4j full-text index for exact wording and unstructured details.
- `search_facts` retrieves only human-approved `Fact` nodes and their exact supporting quotes.
- The model receives only tool evidence, is instructed not to infer unsupported facts, and can iterate across tools up to six times.
