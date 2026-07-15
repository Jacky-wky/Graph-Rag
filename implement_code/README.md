# Neo4j Agentic RAG

This folder contains an end-to-end document knowledge-graph RAG prototype:

1. `load_pdf_into_neo4j.py` creates `Document`, `Page`, and `Chunk` nodes from PDFs.
2. `improve_relationships.py` creates the `Concept` / `Term` taxonomy and links chunks with `MENTIONS_CONCEPT`.
3. `extract_provenance_facts.py` extracts reviewable `Fact` nodes, each supported by an exact chunk quote.
4. `review_provenance_facts.py` approves or rejects facts before they are trusted knowledge.
5. `agentic_rag.py` lets an LLM decide whether to retrieve graph-grounded evidence, full-text chunk evidence, or both before composing a cited answer.

## Provenance-first graph model

Every extracted fact has an auditable evidence path:

```text
(:Document)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk)
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

Set the Neo4j and OpenAI credentials in `.env`. Do not commit this file.

## Build the knowledge graph

Adapt input arguments to the options offered by the loader, then run the taxonomy builder:

```powershell
python load_pdf_into_neo4j.py --help
python improve_relationships.py --clear-old
```

## Optional ontology maintenance

Use these scripts after the main graph is stable to find domain vocabulary that
is missing from `concept_config.py`. Candidate terms are not trusted knowledge
until they are reviewed and approved.

```powershell
python discover_candidate_terms.py --min-freq 5
python review_candidate_terms.py list
```

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
key is available. It first tries exact registered Terms, then conservative
spelling-tolerant Term matching, and finally a lexical Chunk fallback with
evidence-based Concept candidates. Read `RETRIEVAL_POLICY.md` for the complete
confidence and matching policy.

```powershell
python local_graph_rag.py "customer due diligence 有哪些控制要求？"
```

## Retrieval design

- `search_graph` finds matching concepts/terms, expands up to two taxonomy levels, then retrieves related chunks.
- `search_chunks` uses the Neo4j full-text index for exact wording and unstructured details.
- `search_facts` retrieves only human-approved `Fact` nodes and their exact supporting quotes.
- The model receives only tool evidence, is instructed not to infer unsupported facts, and can iterate across tools up to six times.
