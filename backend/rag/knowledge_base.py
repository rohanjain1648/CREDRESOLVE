"""
Part 5: Knowledge & RAG — ChromaDB knowledge base setup
Ingests policies, FAQs, and product documents.
"""
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHROMA_PERSIST = "./chroma_db"

# Use sentence-transformers for free local embeddings
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # supports Hindi
)


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PERSIST)


def get_collection(name: str = "credresolve_kb") -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_policies():
    """Load policies.json and FAQs into ChromaDB."""
    collection = get_collection()

    # Load policies
    with open(DATA_DIR / "policies.json", encoding="utf-8") as f:
        policies = json.load(f)

    docs, metas, ids = [], [], []
    for p in policies:
        docs.append(f"{p['title']}\n\n{p['content']}")
        metas.append({"category": p["category"], "source": p["id"]})
        ids.append(p["id"])

    # Load FAQs
    with open(DATA_DIR / "faqs.json", encoding="utf-8") as f:
        faqs = json.load(f)

    for i, faq in enumerate(faqs):
        docs.append(f"Q: {faq['question']}\nA: {faq['answer']}")
        metas.append({"category": "faq", "source": f"FAQ-{i:03d}"})
        ids.append(f"FAQ-{i:03d}")

    # Upsert (skip if already exists)
    collection.upsert(documents=docs, metadatas=metas, ids=ids)
    print(f"[RAG] Ingested {len(docs)} documents into ChromaDB.")
    return len(docs)


def build_knowledge_base():
    """One-time setup — call this on startup if collection is empty."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        print("[RAG] Knowledge base empty — ingesting documents...")
        ingest_policies()
    else:
        print(f"[RAG] Knowledge base ready: {count} documents.")
