# =========================================================
# VECTOR STORE SERVICE (ChromaDB RAG)
# =========================================================
# One ChromaDB collection PER CHAT THREAD, so documents
# uploaded in one conversation never leak into another
# conversation's answers. Uses Chroma's built-in local
# embedding model (all-MiniLM-L6-v2 via onnxruntime), so no
# extra paid embedding API is required.

import chromadb
from services.document_service import chunk_text

# Persistent on-disk client -> survives app restarts, just like chatbot.db
_client = chromadb.PersistentClient(path="./chroma_db")


def _collection_name(thread_id: str) -> str:
    # Chroma collection names must be simple strings; prefixing
    # keeps them clearly namespaced per chat thread.
    return f"thread_{thread_id}"


def _get_collection(thread_id: str):
    return _client.get_or_create_collection(name=_collection_name(thread_id))


def add_document(thread_id: str, file_name: str, text: str) -> int:
    """
    Chunk `text` and store it in the collection belonging to
    `thread_id`, tagged with the source file name.
    Returns the number of chunks stored (0 if there was no text).
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    collection = _get_collection(thread_id)
    ids = [f"{file_name}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name} for _ in chunks]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def query(thread_id: str, question: str, n_results: int = 4) -> str:
    """
    Retrieve the most relevant chunks for `question` from this
    thread's documents and format them for the LLM prompt.
    Returns "" if this thread has no documents yet.
    """
    collection = _get_collection(thread_id)
    if collection.count() == 0:
        return ""

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count()),
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return ""

    formatted = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "uploaded document") if meta else "uploaded document"
        formatted.append(f"[From {source}]\n{doc}")

    return "\n\n".join(formatted)


def has_documents(thread_id: str) -> bool:
    return _get_collection(thread_id).count() > 0


def list_documents(thread_id: str) -> list[str]:
    """Return the unique file names uploaded in this thread."""
    collection = _get_collection(thread_id)
    if collection.count() == 0:
        return []
    data = collection.get()
    sources = {m.get("source") for m in data.get("metadatas", []) if m}
    return sorted(sources)
