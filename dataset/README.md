# Dataset Guide

This directory contains the source documents and experimental retrieval data
used during the project.

## Current ingestion corpus

`implement_code/load_pdf_into_neo4j.py` reads every PDF in
`dataset/datasets/`. The current corpus contains nine documents:

1. `CG-1-Ch.pdf`
2. `GS-1.pdf`
3. `Guideline_on_AML-CFT_(for_AIs)_chi_May 2023.pdf`
4. `SA-1-Ch.pdf`
5. `SA-2-Ch.pdf`
6. `SPM-AML-1.pdf`
7. `TM-E-1.pdf`
8. `TM-G-1.pdf`
9. `TM-G-2.pdf`

For every parsed chunk, the loader stores the original filename, one-based PDF
page number, chunk index, section hierarchy, exact source text, and retrieval
context. These fields create the path from a graph result back to a PDF page.

## Other dataset folders

- The workbooks in this directory contain test questions, extracted text, and
  retrieval comparisons from earlier project stages.
- `0328_test/` contains Chroma distance-metric and embedding-model experiments.
- `temp/` contains legacy staging documents and workbooks. It is retained for
  experiment reproducibility but is not read by the current ingestion script.

Generated vector stores and Neo4j database files are not committed. Rebuild
them from the source PDFs and the scripts in `implement_code/`.
