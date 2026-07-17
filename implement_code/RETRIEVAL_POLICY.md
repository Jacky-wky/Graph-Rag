# Layered Local Retrieval Policy

`local_graph_rag.py` retrieves Neo4j evidence without an LLM API. The policy is
designed to keep high-confidence Concept mappings explainable while still
handling spelling variation and long-tail questions.

## Layer 1: exact auto-generated Term

The resolver normalizes Unicode width, case, punctuation, whitespace, and a
small set of controlled orthographic variants. It then checks whether any
automatically induced `Term.text` occurs in the question.

```text
question contains auto-generated Term
Term -[:NORMALIZED_TO]-> Concept
```

This is a trusted mapping. Multiple aliases and multiple explicitly mentioned
Concepts may be returned.

## Layer 2: conservative fuzzy auto-generated Term

If no exact Term matches, the resolver compares induced terms with similarly
sized text windows. The default threshold is `0.88`, which is intended for
spelling and punctuation variation, not semantic synonym guessing.

Short abbreviations are excluded from fuzzy matching to reduce false positives.
Similar top candidates within the ambiguity margin are returned as `ambiguous`
instead of being silently selected.

## Layer 3: multilingual Concept/Term MaxSim

When the corpus and question use different wording or languages, the resolver
uses the local `resolver_model` recorded by `OntologyRun`. Each leaf Concept
name and each automatically induced Term are embedded independently. The
Concept score is its best field score (MaxSim); Term-only matches receive a
small penalty because individual c-TF-IDF terms can be less specific than the
generated Concept name.

`improve_relationships.py` computes these Concept and Term vectors once and
stores them on the corresponding Neo4j nodes. Query-time work therefore embeds
only the question and performs MaxSim against the stored vectors.

A mapping is accepted only when the best score reaches `0.50` and is at least
`0.06` ahead of the second candidate for a single-Concept question. Otherwise
the result is explicitly `unresolved` or `ambiguous`. The model, score, matched
field, matched text, and alternative candidates are returned for audit.

Example: an English `customer due diligence` question can resolve to a Concept
induced from Chinese `客戶盡職審查` chunks without adding a manual bilingual alias.

## Layer 4: lexical Chunk fallback

If no Term reaches the fuzzy threshold, the question is converted into local
search keywords. English stopwords and intent words are removed. Long Chinese
sequences also produce character bigrams so that unsegmented text and common
orthographic differences can still retrieve evidence.

The system retrieves matching `Chunk` nodes and lets their existing
`MENTIONS_CONCEPT` and `ABOUT_CONCEPT` relationships vote for candidate
Concepts. These candidates are explicitly marked `trusted_mapping: false`; they
help exploration but do not become ontology facts automatically.

## Intent policy

Intent classification is independent of Concept identity. Deterministic keyword
groups currently classify control requirements, comparison, definition, and
procedure questions. Intent only changes Chunk ranking.

For `control_requirements`, ranking caps repeated modal-word counts and boosts
general requirement headings and list introductions such as `何謂`, `以下為`,
and `include the following`. This prevents a long exception clause from
outranking the main requirements list merely because it repeats `規定`.

## Relationship confidence policy

Every exact occurrence remains traceable as `Chunk-[:MENTIONS_TERM]->Term`.
The system derives `MENTIONS_CONCEPT` only when the strongest matched Term
reaches `0.45` of that Concept's strongest evidence weight. Evidence weight is
computed automatically from c-TF-IDF, phrase length, and inverse Chunk
frequency. Lower-specificity matches remain available at Term level but do not
become direct Concept relationships.

## Confidence and failure behavior

- Exact Term: trusted Concept mapping.
- High-threshold fuzzy Term: trusted spelling-variation mapping.
- Above-threshold, sufficiently separated multilingual MaxSim: trusted semantic mapping with score.
- Chunk evidence vote: untrusted candidate Concept.
- No evidence: unresolved; return no invented Concept.

This policy removes the manually maintained alias dictionary while retaining
explicit confidence thresholds and an auditable lexical fallback.
