"""
Part 5: Knowledge & RAG — in-process vector store backed by Google embeddings.
Replaces ChromaDB (had Windows DLL/upsert hang issues) with a lightweight
numpy cosine-similarity store that runs entirely in-process.
"""
import json
import pickle
from pathlib import Path
from google import genai as google_genai
import numpy as np
from backend.config import get_settings

DATA_DIR = Path(__file__).parent.parent.parent / "data"
STORE_PATH = Path("./vector_store.pkl")
EMBED_MODEL = "models/gemini-embedding-001"


def _get_genai_client():
    settings = get_settings()
    return google_genai.Client(api_key=settings.google_api_key)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Gemini text-embedding model."""
    client = _get_genai_client()
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [list(e.values) for e in result.embeddings]


class _VectorStore:
    """Simple in-process cosine-similarity store."""

    def __init__(self):
        self.docs: list[str] = []
        self.metas: list[dict] = []
        self.ids: list[str] = []
        self._matrix: np.ndarray | None = None  # shape (N, D)

    def __len__(self):
        return len(self.docs)

    def add(self, docs: list[str], embeddings: list[list[float]],
            metadatas: list[dict], ids: list[str]):
        self.docs.extend(docs)
        self.metas.extend(metadatas)
        self.ids.extend(ids)
        arr = np.array(embeddings, dtype=np.float32)
        if self._matrix is None:
            self._matrix = arr
        else:
            self._matrix = np.vstack([self._matrix, arr])

    def query(self, query_embedding: list[float], n: int = 3):
        if self._matrix is None or len(self.docs) == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        norms = np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(q) + 1e-8
        scores = (self._matrix @ q) / norms
        top_idx = np.argsort(scores)[::-1][:n]
        return [(self.docs[i], self.metas[i], float(scores[i])) for i in top_idx]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "_VectorStore":
        with open(path, "rb") as f:
            return pickle.load(f)


_STORE: _VectorStore | None = None


def get_store() -> _VectorStore:
    global _STORE
    if _STORE is None:
        if STORE_PATH.exists():
            _STORE = _VectorStore.load(STORE_PATH)
        else:
            _STORE = _VectorStore()
    return _STORE


def ingest_policies():
    """Load policies.json and FAQs, embed them, and save the store."""
    store = get_store()

    docs, metas, ids = [], [], []

    with open(DATA_DIR / "policies.json", encoding="utf-8") as f:
        for p in json.load(f):
            docs.append(f"{p['title']}\n\n{p['content']}")
            metas.append({"category": p["category"], "source": p["id"]})
            ids.append(p["id"])

    with open(DATA_DIR / "faqs.json", encoding="utf-8") as f:
        for i, faq in enumerate(json.load(f)):
            docs.append(f"Q: {faq['question']}\nA: {faq['answer']}")
            metas.append({"category": "faq", "source": f"faq-{i:03d}"})
            ids.append(f"faq-{i:03d}")

    embeddings = embed(docs)
    store.add(docs, embeddings, metas, ids)
    store.save(STORE_PATH)
    print(f"[RAG] Ingested {len(docs)} documents.")
    return len(docs)


def build_knowledge_base():
    """One-time setup — call on startup if store is empty."""
    store = get_store()
    if len(store) == 0:
        print("[RAG] Knowledge base empty — ingesting documents...")
        ingest_policies()
    else:
        print(f"[RAG] Knowledge base ready: {len(store)} documents.")


def search(query: str, n: int = 3) -> list[tuple[str, dict, float]]:
    """Return top-n (doc, metadata, score) tuples for a query string."""
    store = get_store()
    q_emb = embed([query])[0]
    return store.query(q_emb, n=n)
