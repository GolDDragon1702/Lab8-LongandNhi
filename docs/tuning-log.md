# Tuning Log — RAG Pipeline (Day 08 Lab)

> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13
**Config:**
```
retrieval_mode  = "dense"
chunk_size      = 400 tokens (~1600 ký tự)
overlap         = 80 tokens (~320 ký tự)
top_k_search    = 10
top_k_select    = 3
use_rerank      = False
llm_model       = "gpt-4o-mini"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.70/5 |
| Answer Relevance | 4.70/5 |
| Context Recall | 5.00/5 |
| Completeness | 3.80/5 |

**Câu hỏi yếu nhất:**

| ID | Category | F | R | Rc | C | Vấn đề |
|----|----------|---|---|----|---|--------|
| q06 | SLA | 5 | 5 | 5 | **2** | Dense rank chunk về cấp quyền tạm thời cao hơn chunk escalation timeline → answer thiếu "auto-escalate sau 10 phút" |
| q07 | Access Control | 5 | 5 | 5 | **3** | Answer mô tả đúng nhưng không nêu tên hiện tại của document là "Access Control SOP" |
| q08 | HR Policy | 5 | 5 | 5 | **3** | Trả lời đúng số ngày nhưng bỏ sót điều kiện "Team Lead phê duyệt" và "sau probation period" |

**Phân tích theo Error Tree:**
- [x] **Retrieval: Dense rank sai chunk** — Context Recall = 5/5 (đúng source) nhưng chunk có thông tin đầy đủ nhất không nằm trong top-3 gửi vào prompt (q06)
- [ ] Indexing: Chunking không phải nguyên nhân — metadata đủ, text preview hợp lý
- [ ] Generation: Prompt đã có grounding rule, model không bịa
- [ ] Token overload: Context 3 chunk × ~400 tokens = ~1200 tokens, còn xa giới hạn

**Kết luận nguyên nhân gốc:** Dense embedding đánh giá similarity theo ngữ nghĩa tổng thể, không phân biệt được khi hai chunk cùng nói về "P1" nhưng một cái nói về cấp quyền, một cái nói về escalation. Cần rerank để đọc kỹ nội dung theo context câu hỏi.

---

## Variant 1 — LLM Rerank (Sprint 3)

**Ngày:** 2026-04-13
**Biến thay đổi:** Thêm `use_rerank=True, rerank_method="llm"` — LLM đọc full content từng chunk và chọn top-3 phù hợp nhất

**Lý do chọn biến này:**
q06 thất bại vì dense lấy đúng document nhưng sai chunk — đây là bài toán reranking, không phải retrieval. Cross-encoder transformer rerank cần load model local (~500MB), chậm khi không có GPU. LLM Rerank dùng GPT-4o-mini đã có sẵn, đọc toàn bộ text (không bị giới hạn 200 ký tự như `rerank()` cũ), phù hợp với corpus tiếng Việt có thuật ngữ kỹ thuật mixed.

**Config thay đổi:**
```
# Chỉ thay đổi duy nhất biến này:
use_rerank   = True
rerank_method = "llm"
# Tất cả tham số khác giữ nguyên như baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.70/5 | 4.70/5 | 0.00 |
| Answer Relevance | 4.70/5 | 4.70/5 | 0.00 |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.80/5 | **4.00/5** | **+0.20** |

**Phân tích per-question:**

| ID | Baseline C | Variant C | Thay đổi | Lý do |
|----|-----------|-----------|----------|-------|
| q01 | 3 | 5 | **+2** | LLM rerank đẩy chunk có cả "15 phút" lẫn "4 giờ" lên top → answer đầy đủ hơn |
| q06 | 2 | 5 | **+3** | LLM chọn đúng chunk escalation timeline thay vì chunk cấp quyền tạm thời |
| q09 | 4 | 3 | **-1** | LLM rerank chọn chunk access control thay vì abstain → answer ít grounded hơn |
| q03 | 5 | 5 | 0 | Không đổi |
| q04 | 3 | 3 | 0 | Không đổi — vấn đề nằm ở generation, không phải rerank |

**Nhận xét:**
- Variant cải thiện rõ ở các câu hỏi đòi hỏi **nhiều thông tin cụ thể trong một câu trả lời** (q01, q06): LLM hiểu câu hỏi yêu cầu cả "phản hồi ban đầu" lẫn "resolution time" nên chọn chunk đầy đủ hơn.
- Câu q09 (Insufficient Context) bị giảm vì LLM rerank cố gắng tìm chunk "liên quan nhất" dù không có thông tin — nên chọn chunk về access control và trả lời sai hướng. Đây là điểm yếu của LLM Rerank với abstain cases.

**Kết luận:** Variant 1 tốt hơn baseline về Completeness (+0.20). Nên dùng LLM Rerank cho query đòi hỏi thông tin đa chiều, cân nhắc thêm bước kiểm tra "insufficient context" trước khi rerank để tránh hallucination trên abstain cases.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   Dense retrieval lấy đúng source (Context Recall = 5/5) nhưng sai chunk — completeness thấp không phải vì retrieval bỏ sót document mà vì chunk có thông tin đầy đủ nhất không nằm trong top-3 sau khi rank bằng embedding.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   Rerank (cụ thể là rerank_method). Chunking và retrieval mode không phải bottleneck vì Context Recall đã 5/5. Rerank là bước quyết định xem chunk nào vào prompt.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   Thêm bước "abstain detection" trước LLM Rerank: nếu dense search trả về score < ngưỡng (ví dụ < 0.3) thì skip rerank và trả về "Không đủ dữ liệu" ngay, tránh trường hợp q09. Ngoài ra thử `retrieve_filter` (agent chọn source trước) kết hợp LLM Rerank để xem có cải thiện thêm không.