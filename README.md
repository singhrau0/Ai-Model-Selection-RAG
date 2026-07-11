# AI Model Selection RAG

A local-first RAG application for querying the model landscape knowledge base.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

For local Llama, install/run Ollama and pull the configured model, for example
`ollama pull llama3.2`. Cloud providers require the matching key in `.env`.

## Run

```bash
.venv/bin/python generation.py --provider local --retrieval graph_hybrid
.venv/bin/python generation.py --provider gemini --retrieval hybrid
.venv/bin/python generation.py --provider openai --retrieval mmr
```

Retrieval methods:

- `dense`: Chroma similarity search with Hugging Face embeddings
- `sparse`: BM25 keyword retrieval
- `mmr`: Chroma maximum marginal relevance for diverse results
- `hybrid`: dense + BM25 through LangChain `EnsembleRetriever`
- `graph`: dense seeds expanded to neighboring chunks with NetworkX
- `graph_hybrid`: combined graph and hybrid documents

Use `src.evaluation.ndcg_for_ids()` for graded nDCG@k evaluation. Direct dense
and sparse vector creation is available through `Vectorizer.dense_vectors()` and
`Vectorizer.sparse_vectors()`.

`Chunker` provides `fixed_chunking`, `paragraph_chunking`, `overlap_chunking`,
and embedding-based `semantic_chunking`.

## Tests

```bash
.venv/bin/python -m pytest -q
```
