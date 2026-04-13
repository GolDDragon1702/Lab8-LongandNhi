from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List
import json
from configs.llm_configs import LLMConfig

class ListOutput(BaseModel):
    output: List[str]
    
class RerankOutput(BaseModel):
    indices: List[int]

class OpenAIGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=LLMConfig().OPENAI_API_KEY,
            temperature=temperature,
        )
        self.model_name = model_name

    # ─────────────────────────────────────────────
    # 1. Grounded QA (Sprint 2)
    # ─────────────────────────────────────────────
    def generate_answer(self, query: str, contexts: List[dict]) -> str:
        if not contexts:
            return "Không đủ dữ liệu"

        context_str = ""
        for i, c in enumerate(contexts):
            doc_id = c["metadata"].get("doc_id", str(i))
            source = c["metadata"].get("source", "unknown")

            context_str += f"[{doc_id}] ({source})\n{c['text']}\n\n"

        system_prompt = (
            "Bạn là hệ thống trả lời dựa trên tài liệu (RAG).\n"
            "QUY TẮC:\n"
            "1. Chỉ dùng thông tin trong CONTEXT\n"
            "2. Mỗi ý phải có citation dạng [id]\n"
            "3. Nếu không đủ thông tin → trả lời: 'Không đủ dữ liệu'\n"
            "4. Không suy đoán\n"
        )

        user_prompt = (
            f"CONTEXT:\n{context_str}\n"
            f"QUESTION: {query}\n\n"
            "Trả lời ngắn gọn, có citation."
        )

        res = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        return res.content

    # ─────────────────────────────────────────────
    # 2. Query Expansion
    # ─────────────────────────────────────────────
    def expand_query(self, query: str) -> List[str]:
        structured_llm = self.llm.with_structured_output(ListOutput)

        result = structured_llm.invoke(
            f"Generate 2-3 alternative search queries for: {query}"
        )

        return [query] + result.output

    # ─────────────────────────────────────────────
    # 3. Query Decomposition
    # ─────────────────────────────────────────────
    def decompose_query(self, query: str) -> List[str]:
        structured_llm = self.llm.with_structured_output(ListOutput)

        result = structured_llm.invoke(
            f"Break this query into 2-3 simpler queries: {query}"
        )

        return result.output

    # ─────────────────────────────────────────────
    # 4. LLM Rerank (fallback)
    # ─────────────────────────────────────────────
    def rerank(self, query: str, candidates: List[dict], top_k: int):
        if not candidates:
            return []

        context = "\n\n".join(
            [f"[{i}] {c['text'][:200]}" for i, c in enumerate(candidates)]
        )

        structured_llm = self.llm.with_structured_output(RerankOutput)

        result = structured_llm.invoke(
f"""
Query: {query}

Candidates:
{context}

Select top {top_k} most relevant indices.
"""
        )

        return [candidates[i] for i in result.indices if i < len(candidates)]

    # ─────────────────────────────────────────────
    # 5. LLM Rerank — đọc full content
    # ─────────────────────────────────────────────
    def llm_rerank(self, query: str, candidates: List[dict], top_k: int) -> List[dict]:
        """
        LLM đọc toàn bộ nội dung từng candidate và chọn những chunk
        thực sự phù hợp nhất với query, không chỉ dựa vào embedding score.

        Khác với rerank() (dùng 200 ký tự đầu), hàm này cho LLM đọc
        đủ nội dung để phán xét chính xác hơn.

        Args:
            query: Câu hỏi gốc
            candidates: List chunk với keys "text" và "metadata"
            top_k: Số chunk muốn giữ lại

        Returns:
            List chunk đã được LLM chọn, sắp xếp theo mức độ phù hợp
        """
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))

        context = "\n\n".join(
            f"[{i}] (source: {c.get('metadata', {}).get('source', '?')})\n{c['text']}"
            for i, c in enumerate(candidates)
        )

        structured_llm = self.llm.with_structured_output(RerankOutput)

        result = structured_llm.invoke(
f"""You are a relevance judge for a RAG retrieval system.

Query: {query}

Candidates (read each carefully):
{context}

Select the indices of the {top_k} most relevant chunks.
Only select chunks that actually help answer the query.
Order them from most to least relevant.
"""
        )

        seen = set()
        reranked = []
        for i in result.indices:
            if i < len(candidates) and i not in seen:
                seen.add(i)
                reranked.append(candidates[i])

        return reranked[:top_k]
    
if __name__ == "__main__":
    generator = OpenAIGenerator()

    query = "SLA ticket P1 là gì?"

    contexts = [
        {
            "text": "SLA cho ticket P1 là 15 phút phản hồi ban đầu.",
            "metadata": {"source": "policy_sla", "doc_id": "doc_1"}
        },
        {
            "text": "Ticket P2 có thời gian phản hồi là 2 giờ.",
            "metadata": {"source": "policy_sla", "doc_id": "doc_2"}
        },
        {
            "text": "P1 là mức độ ưu tiên cao nhất.",
            "metadata": {"source": "priority_doc", "doc_id": "doc_3"}
        },
    ]

    print("\n===== TEST QA =====")
    answer = generator.generate_answer(query, contexts)
    print(answer)

    print("\n===== TEST EXPAND =====")
    expanded = generator.expand_query(query)
    print(expanded)

    print("\n===== TEST DECOMPOSE =====")
    decomposed = generator.decompose_query(query)
    print(decomposed)

    print("\n===== TEST RERANK =====")
    reranked = generator.rerank(query, contexts, top_k=2)
    for r in reranked:
        print("-", r["text"])