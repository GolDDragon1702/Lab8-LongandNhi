"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import json
import sys
from pathlib import Path
this_path = Path("/home/shine/Documents/code/program/VinUni/git/day8/again/Lecture-Day-08-09-10/day08/lab/test.ipynb")
sys.path.append(str(this_path.parent.parent))

from src.embeddings import LocalEmbedder
from src.store import EmbeddingStore
from src.rerank import TransformerReranker
from src.llm import OpenAIGenerator

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================
CHROMA_DB_DIR = this_path.parent / "chroma_db"
COLLECTION_NAME = "rag_lab"

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

embedder = LocalEmbedder()
store = EmbeddingStore(persist_path=CHROMA_DB_DIR)
reranker = TransformerReranker()
generator = OpenAIGenerator(model_name=LLM_MODEL)

# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi
        top_k: Số chunk tối đa trả về

    Returns:
        List các dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata (source, section, effective_date, ...)
          - "score": cosine similarity score
    """

    # 2. Search trong vector store
    results = store.search(
        collection_name=COLLECTION_NAME,
        query_embedding=embedder.encode(query),
        top_k=top_k,
    )

    # 3. Chuẩn hóa output format
    return [
        {
            "text": r["content"],
            "metadata": r["metadata"],
            "score": r["score"],
        }
        for r in results
    ]


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# Dùng cho Sprint 3 Variant hoặc kết hợp Hybrid
# =============================================================================

from typing import List, Dict, Any

_bm25 = None
_bm25_corpus = []
_bm25_metadata = []


def _build_bm25_index():
    global _bm25, _bm25_corpus, _bm25_metadata

    if _bm25 is not None:
        return _bm25

    from rank_bm25 import BM25Okapi

    # 1. Load toàn bộ data từ Chroma
    col = store._get_or_create(COLLECTION_NAME)
    data = col.get(include=["documents", "metadatas"])

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        _bm25 = None
        return None

    # 2. Tokenize đơn giản
    tokenized_corpus = [doc.lower().split() for doc in documents]

    # 3. Build BM25
    _bm25 = BM25Okapi(tokenized_corpus)
    _bm25_corpus = documents
    _bm25_metadata = metadatas

    return _bm25


def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval bằng BM25.
    """

    bm25 = _build_bm25_index()
    if bm25 is None:
        return []

    # 1. Tokenize query
    tokenized_query = query.lower().split()

    # 2. Score
    scores = bm25.get_scores(tokenized_query)

    # 3. Lấy top_k
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # 4. Format output
    results = []
    for i in top_indices:
        results.append({
            "text": _bm25_corpus[i],
            "metadata": _bm25_metadata[i],
            "score": float(scores[i]),  # BM25 score (không normalize)
        })

    return results


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval bằng Reciprocal Rank Fusion (RRF).
    """

    # 1. Lấy kết quả từ 2 nguồn
    dense_results = retrieve_dense(query, top_k)
    sparse_results = retrieve_sparse(query, top_k)

    # 2. Map: doc_key → info
    merged = {}

    def _get_key(item):
        # dùng text làm key (simple, đủ dùng)
        return item["text"]

    # 3. Add dense ranks
    for rank, item in enumerate(dense_results):
        key = _get_key(item)

        if key not in merged:
            merged[key] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "rrf_score": 0.0,
            }

        merged[key]["rrf_score"] += dense_weight * (1 / (60 + rank))

    # 4. Add sparse ranks
    for rank, item in enumerate(sparse_results):
        key = _get_key(item)

        if key not in merged:
            merged[key] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "rrf_score": 0.0,
            }

        merged[key]["rrf_score"] += sparse_weight * (1 / (60 + rank))

    # 5. Sort theo RRF score
    results = sorted(
        merged.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    # 6. Format output
    return [
        {
            "text": r["text"],
            "metadata": r["metadata"],
            "score": r["rrf_score"],
        }
        for r in results[:top_k]
    ]


# =============================================================================
# RETRIEVAL — AGENTIC FILTER (Agent chọn sources → hybrid search)
# =============================================================================

def _get_all_sources() -> List[str]:
    """Lấy danh sách tất cả unique sources trong collection."""
    col = store._get_or_create(COLLECTION_NAME)
    data = col.get(include=["metadatas"])
    sources = list({
        m.get("source", "")
        for m in data.get("metadatas", [])
        if m.get("source")
    })
    return sorted(sources)


def _agent_pick_sources(query: str, sources: List[str], max_pick: int = 3) -> List[str]:
    """LLM đọc danh sách sources và chọn những cái phù hợp nhất với query."""
    if not sources:
        return []

    sources_str = "\n".join(f"- {s}" for s in sources)
    prompt = f"""You are a routing agent for a document retrieval system.

Available document sources:
{sources_str}

User query: "{query}"

Select up to {max_pick} sources most likely to contain the answer.
Respond ONLY with a valid JSON array of source names (exact strings from the list above).
Example: ["source_a", "source_b"]"""

    try:
        res = generator.llm.invoke([{"role": "user", "content": prompt}])
        content = res.content.strip().strip("```json").strip("```").strip()
        picked = json.loads(content)
        if isinstance(picked, list):
            return [s for s in picked if s in sources]
    except Exception:
        pass

    return sources  # fallback: giữ tất cả


def retrieve_filter(
    query: str,
    top_k: int = TOP_K_SEARCH,
    max_sources: int = 3,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Agentic filtered retrieval:
      1. Agent đọc toàn bộ sources trong index → chọn sources phù hợp nhất
      2. Hybrid retrieval (dense + BM25 + RRF) giới hạn trong sources đã chọn

    Args:
        query: Câu hỏi
        top_k: Số chunk trả về
        max_sources: Số source tối đa agent chọn
        verbose: In debug info

    Returns:
        List chunks (cùng format với retrieve_dense / retrieve_hybrid)
    """

    # ── Bước 1: Agent chọn sources ────────────────────────────
    all_sources = _get_all_sources()
    if not all_sources:
        return retrieve_hybrid(query, top_k=top_k)

    selected_sources = _agent_pick_sources(query, all_sources, max_pick=max_sources)
    if not selected_sources:
        selected_sources = all_sources

    if verbose:
        print(f"[filter] All sources ({len(all_sources)}): {all_sources}")
        print(f"[filter] Agent selected: {selected_sources}")

    # ── Bước 2: Dense search có filter theo source ────────────
    dense_filtered: List[Dict[str, Any]] = []
    for source in selected_sources:
        hits = store.search_with_filter(
            collection_name=COLLECTION_NAME,
            query_embedding=embedder.encode(query),
            top_k=top_k,
            metadata_filter={"source": source},
        )
        for h in hits:
            dense_filtered.append({
                "text": h["content"],
                "metadata": h["metadata"],
                "score": h["score"],
            })

    # ── Bước 3: Sparse (BM25) search, lọc lại theo source ────
    sparse_all = retrieve_sparse(query, top_k=top_k * 2)
    sparse_filtered = [
        c for c in sparse_all
        if c["metadata"].get("source") in selected_sources
    ]

    # ── Bước 4: RRF merge ─────────────────────────────────────
    merged: Dict[str, Dict] = {}

    for rank, item in enumerate(dense_filtered):
        k = item["text"]
        if k not in merged:
            merged[k] = {"text": item["text"], "metadata": item["metadata"], "rrf_score": 0.0}
        merged[k]["rrf_score"] += 0.6 * (1 / (60 + rank))

    for rank, item in enumerate(sparse_filtered):
        k = item["text"]
        if k not in merged:
            merged[k] = {"text": item["text"], "metadata": item["metadata"], "rrf_score": 0.0}
        merged[k]["rrf_score"] += 0.4 * (1 / (60 + rank))

    ranked = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

    if verbose:
        print(f"[filter] Candidates after hybrid: {len(ranked)}")

    return [
        {"text": r["text"], "metadata": r["metadata"], "score": r["rrf_score"]}
        for r in ranked[:top_k]
    ]


# =============================================================================
# RERANK (Sprint 3 alternative)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
    method: str = "cross_encoder",
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks.

    Args:
        method: "cross_encoder" (TransformerReranker) hoặc "llm" (LLM đọc full content)
    """
    if not candidates:
        return []

    if method == "llm":
        return generator.llm_rerank(query, candidates, top_k=top_k)

    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Query transformation bằng LLM.
    """

    if strategy == "expansion":
        prompt = f"""
Given the query: "{query}"
Generate 2-3 alternative phrasings or related terms.
Return ONLY a JSON array of strings.
"""

    elif strategy == "decomposition":
        prompt = f"""
Break down this query into 2-3 simpler sub-queries:
"{query}"
Return ONLY a JSON array.
"""

    else:
        return [query]

    try:
        response = generator.llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ])

        content = response.content.strip()
        # Strip markdown code fences if present
        content = content.strip("```json").strip("```").strip()

        queries = json.loads(content)

        if isinstance(queries, list):
            return [query] + queries

    except Exception:
        pass

    return [query]


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # TODO: Tùy chỉnh format nếu muốn (thêm effective_date, department, ...)
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    TODO Sprint 2:
    Đây là prompt baseline. Trong Sprint 3, bạn có thể:
    - Thêm hướng dẫn về format output (JSON, bullet points)
    - Thêm ngôn ngữ phản hồi (tiếng Việt vs tiếng Anh)
    - Điều chỉnh tone phù hợp với use case (CS helpdesk, IT support)
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    TODO Sprint 2:
    Chọn một trong hai:

    Option A — OpenAI (cần OPENAI_API_KEY):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,     # temperature=0 để output ổn định, dễ đánh giá
            max_tokens=512,
        )
        return response.choices[0].message.content

    Option B — Google Gemini (cần GOOGLE_API_KEY):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    Lưu ý: Dùng temperature=0 hoặc thấp để output ổn định cho evaluation.
    """
    res = generator.llm.invoke([
        {"role": "user", "content": prompt},
    ])
    return res.content


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    rerank_method: str = "cross_encoder",  # "cross_encoder" hoặc "llm"
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng

    TODO Sprint 2 — Implement pipeline cơ bản:
    1. Chọn retrieval function dựa theo retrieval_mode
    2. Gọi rerank() nếu use_rerank=True
    3. Truncate về top_k_select chunks
    4. Build context block và grounded prompt
    5. Gọi call_llm() để sinh câu trả lời
    6. Trả về kết quả kèm metadata

    TODO Sprint 3 — Thử các variant:
    - Variant A: đổi retrieval_mode="hybrid"
    - Variant B: bật use_rerank=True
    - Variant C: thêm query transformation trước khi retrieve
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "rerank_method": rerank_method,
    }

    # --- Bước 1: Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    elif retrieval_mode == "filter":
        candidates = retrieve_filter(query, top_k=top_k_search, verbose=verbose)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select, method=rerank_method)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "hybrid", "filter"]  # Thêm "sparse" sau khi implement

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError:
            print("Chưa implement — hoàn thành TODO trong retrieve_dense() và call_llm() trước.")
        except Exception as e:
            print(f"Lỗi: {e}")

    # Uncomment sau khi Sprint 3 hoàn thành:
    # print("\n--- Sprint 3: So sánh strategies ---")
    # compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    # compare_retrieval_strategies("ERR-403-AUTH")

    print("\n\nViệc cần làm Sprint 2:")
    print("  1. Implement retrieve_dense() — query ChromaDB")
    print("  2. Implement call_llm() — gọi OpenAI hoặc Gemini")
    print("  3. Chạy rag_answer() với 3+ test queries")
    print("  4. Verify: output có citation không? Câu không có docs → abstain không?")

    print("\nViệc cần làm Sprint 3:")
    print("  1. Chọn 1 trong 3 variants: hybrid, rerank, hoặc query transformation")
    print("  2. Implement variant đó")
    print("  3. Chạy compare_retrieval_strategies() để thấy sự khác biệt")
    print("  4. Ghi lý do chọn biến đó vào docs/tuning-log.md")