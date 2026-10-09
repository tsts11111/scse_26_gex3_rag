"""
RAG retrieval layer.

Builds, from the loaded policy documents:
  1. an embedding model (bge-small-en-v1.5, cached locally via ModelScope),
  2. a persistent Chroma vector store,
  3. a LangChain retriever,
  4. a "policy_search" tool that returns the policy passages for a query.

The documents themselves are produced ONLY by:
    documents = load_policy_documents()
"""

from pathlib import Path

from langchain_core.tools import tool

try:
    from langchain_chroma import Chroma
except ImportError:  # older LangChain layout
    from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
EMBEDDING_DIR = ROOT / "models" / "bge-small-en-v1.5"
CHROMA_DIR = ROOT / "chroma_db"

# ModelScope hosts the same BAAI model; Hugging Face is blocked without a VPN.
_EMBEDDING_MODEL_IDS = (
    "BAAI/bge-small-en-v1.5",
    "AI-ModelScope/bge-small-en-v1.5",
)


def _resolve_embedding_dir() -> str:
    """Ensure the embedding model is available locally, downloading via ModelScope."""
    marker = EMBEDDING_DIR / "config.json"
    if not marker.exists():
        from modelscope import snapshot_download

        last_error = None
        for model_id in _EMBEDDING_MODEL_IDS:
            try:
                snapshot_download(model_id, local_dir=str(EMBEDDING_DIR))
                last_error = None
                break
            except Exception as exc:  # try the next known mirror id
                last_error = exc
        if last_error is not None and not marker.exists():
            raise last_error
    return str(EMBEDDING_DIR)


def _embeddings() -> HuggingFaceEmbeddings:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=_resolve_embedding_dir(),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(documents):
    """Create (or open the cached) Chroma vector store from the policy documents."""
    emb = _embeddings()
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)
    return Chroma.from_documents(
        documents,
        emb,
        persist_directory=str(CHROMA_DIR),
    )


def build_retriever(documents, k: int = 4):
    """Return a retriever over the university IT policy knowledge base."""
    vector_store = build_vector_store(documents)
    return vector_store.as_retriever(search_kwargs={"k": k})


def _format_documents(docs) -> str:
    return "\n\n".join(doc.page_content.strip() for doc in docs)


def build_policy_search_tool(retriever):
    """LangChain tool that searches the policy knowledge base for a query."""

    @tool("policy_search")
    def policy_search(query: str) -> str:
        """Search the official university IT policy knowledge base for information
        relevant to the user's question (VPN, Wi-Fi, printing, account locks,
        storage, phishing, IT support, etc.)."""
        docs = retriever.invoke(query)
        return _format_documents(docs)

    return policy_search


if __name__ == "__main__":
    # Standalone build/test of the retrieval layer.
    from policy_loader import load_policy_documents

    documents = load_policy_documents()
    print(f"Loaded {len(documents)} policy documents")
    retriever = build_retriever(documents)
    question = "What exact phone number should I call for the University IT Service Desk?"
    for doc in retriever.invoke(question):
        print("---", doc.metadata)
        print(doc.page_content[:300])
