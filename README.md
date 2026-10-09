# SCSE 2026 - Graded Exercise 3 (RAG part)

Retrieval-augmented university IT support agent using **Qwen3-0.6B** (local)
over a single knowledge-base file (`university_it_policies.txt`).

## Goal

When asked:

> What exact phone number should I call for the University IT Service Desk?

the agent must state that the available university IT information does not
provide the number. It must **not** invent a phone number (hallucination) and
must not give irrelevant answers.

The knowledge base intentionally contains no verified Service Desk phone
number (see `check_data.py`).

## Files

| File | Status | Purpose |
|---|---|---|
| `policy_loader.py` | given, do not edit | Loads `[POLICY] ... [END POLICY]` sections as LangChain documents |
| `rag_tool.py` | completed | Embeddings (bge-small-en-v1.5), Chroma vector store, retriever, `policy_search` tool |
| `agent.py` | completed | Local Qwen3-0.6B LLM wrapper, strict anti-hallucination prompt, RAG chain |
| `rag.py` | completed | Main entry point: graded target question + interactive mode |

## Setup

The base model is the one downloaded for the SFT part of the exercise
(`../SCSE_26_GEX3/models/qwen3-0.6b`). Override its location with the
`QWEN_MODEL_PATH` environment variable if needed.

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Run

```powershell
python rag.py
```

The embedding model (`bge-small-en-v1.5`, ~130 MB) is fetched automatically
from ModelScope on first run; keep the VPN off. The Chroma store is cached in
`chroma_db/`.
