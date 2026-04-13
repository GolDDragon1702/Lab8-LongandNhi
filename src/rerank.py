from __future__ import annotations

from typing import List, Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from configs.rerank_configs import RerankConfig

from utils.logger import get_logger

logger = get_logger()


class TransformerReranker:
    """
    Reranker dùng HuggingFace Transformers (sequence classification).
    """

    def __init__(
        self,
        model_name: str = RerankConfig().model_name,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Initialized TransformerReranker: {self.model_name} on {self.device}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        scores = []

        # batch để tránh OOM
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            queries = [query] * len(batch)
            docs = [c["text"] for c in batch]

            inputs = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

                # model này thường output shape (batch, 1)
                batch_scores = logits.squeeze(-1).cpu().tolist()

            scores.extend(batch_scores)

        # sort
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                "text": c["text"],
                "metadata": c["metadata"],
                "score": float(s),
            }
            for c, s in ranked[:top_k]
        ]

    def __repr__(self) -> str:
        return f"TransformerReranker(model={self.model_name}, device={self.device})"
    
if __name__ == "__main__":
    reranker = TransformerReranker()
    print(reranker)

    query = "SLA ticket P1 là gì?"

    candidates = [
        {
            "text": "SLA cho ticket P1 là 15 phút phản hồi ban đầu.",
            "metadata": {"source": "policy_sla"}
        },
        {
            "text": "Ticket P2 có thời gian phản hồi là 2 giờ.",
            "metadata": {"source": "policy_sla"}
        },
        {
            "text": "Hướng dẫn sử dụng hệ thống ticket nội bộ.",
            "metadata": {"source": "guide"}
        },
        {
            "text": "P1 là mức độ ưu tiên cao nhất trong hệ thống.",
            "metadata": {"source": "policy_priority"}
        },
    ]

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=3
    )

    print("\n🔍 Query:", query)
    print("\n📊 Reranked results:")

    for i, r in enumerate(results):
        print(f"{i+1}. score={r['score']:.4f}")
        print(f"   text: {r['text']}")
        print(f"   source: {r['metadata'].get('source')}")
        print()