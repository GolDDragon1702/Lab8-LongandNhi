from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer

from configs.embed_configs import EmbedConfig


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(
        self,
        model_name: str | None = None,
        prompt_name: str | None = None,
    ) -> None:
        self.model_name = model_name or EmbedConfig().model_name
        self.prompt_name = prompt_name or EmbedConfig().prompt_name
        self._backend_name = self.model_name
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={"dtype": "auto"},
        )

    def __call__(self, text: str) -> list[float]:
        return self.encode(text)

    def encode(self, text: str, prompt_name: str | None = None) -> list[float]:
        embedding = self.model.encode(
            text,
            prompt_name=prompt_name or self.prompt_name,
            normalize_embeddings=True,
        )
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]

    def encode_batch(self, texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            prompt_name=prompt_name or self.prompt_name,
            normalize_embeddings=True,
        )
        return [
            emb.tolist() if hasattr(emb, "tolist") else [float(v) for v in emb]
            for emb in embeddings
        ]

    def __repr__(self) -> str:
        return f"LocalEmbedder(model={self.model_name!r}, prompt={self.prompt_name!r})"

if __name__ == "__main__":
    """Test the LocalEmbedder with example queries and documents."""
    embedder = LocalEmbedder()
    print(embedder)

    queries = [
        "how much protein should a female eat",
        "summit define",
    ]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Definition of summit: the highest point of a mountain, the highest level.",
    ]

    # Encode
    query_embs = embedder.encode_batch(queries, prompt_name="web_search_query")
    doc_embs = embedder.encode_batch(documents, prompt_name="document")

    # Tính similarity
    import numpy as np
    scores = (np.array(query_embs) @ np.array(doc_embs).T) * 100

    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        for j, doc in enumerate(documents):
            print(f"  Doc {j+1}: {scores[i][j]:.2f}")