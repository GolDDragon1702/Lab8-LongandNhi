from __future__ import annotations

from typing import Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from src.models import Document


class EmbeddingStore:
    def __init__(
        self,
        persist_path: str | None = None,
    ) -> None:
        self._client = (
            chromadb.PersistentClient(path=persist_path)
            if persist_path
            else chromadb.Client()
        )
        print(f"✅ ChromaDB ready — {len(self.list_collections())} collections")

    # ── internal helpers ──────────────────────────────────────

    def _get_or_create(self, collection_name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── public API ────────────────────────────────────────────

    def add_documents(
        self,
        collection_name: str,
        docs: list[Document],
    ) -> None:
        if not docs:
            return
        col = self._get_or_create(collection_name)
        col.add(
            ids=[doc.id for doc in docs],
            documents=[doc.content for doc in docs],
            embeddings=[doc.embeddings for doc in docs],
            metadatas=[doc.metadata for doc in docs],
        )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        col = self._get_or_create(collection_name)
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )
        return [
            {"content": doc, "score": 1 - dist, "metadata": meta}
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    def search_with_filter(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        if not metadata_filter:
            return self.search(collection_name, query_embedding, top_k)

        col = self._get_or_create(collection_name)
        count = col.count()
        if count == 0:
            return []

        where = (
            metadata_filter
            if len(metadata_filter) == 1
            else {"$and": [{k: v} for k, v in metadata_filter.items()]}
        )
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
        )
        return [
            {"content": doc, "score": 1 - dist, "metadata": meta}
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    def get_collection_size(self, collection_name: str) -> int:
        return self._get_or_create(collection_name).count()

    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        col = self._get_or_create(collection_name)
        hits = col.get(where={"doc_id": doc_id})
        ids = hits.get("ids", [])
        if not ids:
            return False
        col.delete(ids=ids)
        return True

    def delete_collection(self, collection_name: str) -> None:
        self._client.delete_collection(collection_name)
        print(f"🗑️  Deleted collection '{collection_name}'")

    def list_collections(self) -> list[str]:
        return [col.name for col in self._client.list_collections()]


if __name__ == "__main__":
    from src.embeddings import LocalEmbedder
    from src.models import Document

    embedder = LocalEmbedder()
    store = EmbeddingStore()

    COL = "test_col"

    # ── 1. Thêm documents ──
    docs = [
        Document(id="doc_1", content="CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.", metadata={"source": "cdc",        "doc_id": "doc_1"}),
        Document(id="doc_2", content="Definition of summit: the highest point of a mountain.",                            metadata={"source": "dictionary", "doc_id": "doc_2"}),
        Document(id="doc_3", content="Python is a high-level programming language known for its simplicity.",             metadata={"source": "wiki",       "doc_id": "doc_3"}),
    ]
    embeddings = embedder.encode_batch([d.content for d in docs], prompt_name="document")
    store.add_documents(COL, docs, embeddings)
    print(f"\n✅ Added {len(docs)} docs — size: {store.get_collection_size(COL)}")

    # ── 2. Search ──
    query = "how much protein should a female eat"
    q_emb = embedder.encode_batch([query], prompt_name="web_search_query")[0]

    results = store.search(COL, q_emb, top_k=2)
    print(f"\n🔍 Search: '{query}'")
    for r in results:
        print(f"  score={r['score']:.4f} | {r['content'][:80]}...")

    # ── 3. Search with filter ──
    results_filtered = store.search_with_filter(COL, q_emb, top_k=2, metadata_filter={"source": "cdc"})
    print(f"\n🔍 Search with filter (source=cdc):")
    for r in results_filtered:
        print(f"  score={r['score']:.4f} | {r['content'][:80]}...")

    # ── 4. Delete document ──
    deleted = store.delete_document(COL, "doc_1")
    print(f"\n🗑️  Delete doc_1: {'OK' if deleted else 'FAILED'}")
    print(f"   Size after delete: {store.get_collection_size(COL)}")

    # ── 5. Delete không tồn tại ──
    deleted_fake = store.delete_document(COL, "doc_999")
    print(f"\n🗑️  Delete doc_999: {'OK' if deleted_fake else 'Not found — OK'}")

    # ── 6. List collections ──
    print(f"\n📋 Collections: {store.list_collections()}")

    # ── 7. Dọn dẹp ──
    store.delete_collection(COL)