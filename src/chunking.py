from __future__ import annotations

import math
import re


from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator="",  # chia theo ký tự, không theo từ
        )

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._splitter.split_text(text)


class SentenceChunker:
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_sentences_per_chunk * 120,
            chunk_overlap=0,
            separators=[". ", "! ", "? ", ".\n", "\n"],
        )

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._splitter.split_text(text)


class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=self.separators,
        )

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # edge case: empty separators list
        if not self.separators:
            return [text]
        return self._splitter.split_text(text)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """Delegate to a fresh splitter with remaining separators."""
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            separators=remaining_separators,
        )
        return splitter.split_text(current_text)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }

        return {
            name: {
                "count": len(chunks),
                "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks,
            }
            for name, chunks in strategies.items()
        }
    
if __name__ == "__main__":
    import time
    st = time.time()
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A fox is a small omnivorous mammal. "
        "Dogs are loyal companions and working animals. "
        "Brown bears live in forests across the northern hemisphere. "
        "Jumping is a physical activity that requires leg strength. "
    )

    # Fixed
    fixed = FixedSizeChunker(chunk_size=100, overlap=20).chunk(text)
    print(f"Fixed:     {len(fixed)} chunks → {[len(c) for c in fixed]}")

    # Sentence
    sentence = SentenceChunker(max_sentences_per_chunk=2).chunk(text)
    print(f"Sentence:  {len(sentence)} chunks → {sentence}")

    # Recursive
    recursive = RecursiveChunker(chunk_size=100).chunk(text)
    print(f"Recursive: {len(recursive)} chunks → {[len(c) for c in recursive]}")

    # Comparator
    stats = ChunkingStrategyComparator().compare(text, chunk_size=100)
    for name, s in stats.items():
        print(f"{name:<15} count={s['count']}  avg_len={s['avg_length']:.1f}")

    # Similarity
    print(f"\nSimilarity([1,0], [1,0]) = {compute_similarity([1.0, 0.0], [1.0, 0.0])}")
    print(f"Similarity([1,0], [0,1]) = {compute_similarity([1.0, 0.0], [0.0, 1.0])}")
    
    print(time.time() - st)