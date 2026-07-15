# Layered Local Retrieval Policy

`local_graph_rag.py` retrieves Neo4j evidence without an LLM API. The policy is
designed to keep high-confidence Concept mappings explainable while still
handling spelling variation and long-tail questions.

## Layer 1: exact registered Term

The resolver normalizes Unicode width, case, punctuation, whitespace, and a
small set of controlled orthographic variants. It then checks whether any
registered `Term.text` occurs in the question.

```text
question contains registered Term
Term -[:NORMALIZED_TO]-> Concept
```

This is a trusted mapping. Multiple aliases and multiple explicitly mentioned
Concepts may be returned.

## Layer 2: conservative fuzzy Term

If no exact Term matches, the resolver compares registered terms with similarly
sized text windows. The default threshold is `0.88`, which is intended for
spelling and punctuation variation, not semantic synonym guessing.

Example: `customer due dilligence` can match the registered term `customer due
diligence`. Short abbreviations are excluded from fuzzy matching to reduce false
positives. Similar top candidates within the ambiguity margin are returned as
`ambiguous` instead of being silently selected.

## Layer 3: lexical Chunk fallback

If no Term reaches the fuzzy threshold, the question is converted into local
search keywords. English stopwords and intent words are removed. Long Chinese
sequences also produce character bigrams so that unsegmented text and common
orthographic differences can still retrieve evidence.

The system retrieves matching `Chunk` nodes and lets their existing
`MENTIONS_CONCEPT` relationships vote for candidate Concepts. These candidates
are explicitly marked `trusted_mapping: false`; they help exploration but do not
become ontology facts automatically.

## Intent policy

Intent classification is independent of Concept identity. Deterministic keyword
groups currently classify control requirements, comparison, definition, and
procedure questions. Intent only changes Chunk ranking.

## Confidence and failure behavior

- Exact Term: trusted Concept mapping.
- High-threshold fuzzy Term: trusted spelling-variation mapping.
- Chunk evidence vote: untrusted candidate Concept.
- No evidence: unresolved; return no invented Concept.

This policy avoids an unlimited alias dictionary without allowing the resolver
to present uncertain lexical evidence as a confirmed Concept mapping.
