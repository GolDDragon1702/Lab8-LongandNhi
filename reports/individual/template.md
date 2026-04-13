# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** ___________
**Vai trò trong nhóm:** ___________
**Ngày nộp:** 2026-04-13
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này?

Trong lab này tôi đảm nhận vai trò chính ở Sprint 2 và Sprint 3. Cụ thể, tôi implement hàm `retrieve_dense()` để query ChromaDB bằng embedding vector, hàm `call_llm()` để gọi GPT-4o-mini sinh câu trả lời có citation, và toàn bộ pipeline `rag_answer()` kết nối các bước lại với nhau. Ở Sprint 3, tôi chọn và implement LLM Rerank — thay vì dùng cross-encoder transformer, tôi cho GPT-4o-mini đọc toàn bộ nội dung từng candidate chunk và chọn top-3 phù hợp nhất với câu hỏi. Phần của tôi kết nối với Sprint 1 (index.py của Retrieval Owner tạo ra ChromaDB) và Sprint 4 (eval.py của Eval Owner đọc output từ `rag_answer()` để chấm điểm).

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Trước lab này tôi nghĩ retrieval là bước quan trọng nhất — nếu lấy đúng document thì answer sẽ tốt. Nhưng kết quả eval cho thấy Context Recall đạt 5/5 ngay từ baseline, tức là đúng source đã được retrieve đủ. Vấn đề thực sự nằm ở **chunk ranking trong top-k**: đúng document nhưng chunk có thông tin chi tiết nhất không nằm trong 3 chunk được gửi vào prompt. Điều này làm tôi hiểu rõ hơn tại sao pipeline RAG thực tế cần tách biệt "search rộng" (top-10) và "select hẹp" (top-3) — hai bước này cần metric đánh giá khác nhau. Retrieval đo bằng Recall, còn selection sau rerank ảnh hưởng trực tiếp đến Completeness của answer.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Điều ngạc nhiên nhất là q06 (escalation P1) bị completeness 2/5 ở baseline dù source đúng. Tôi đọc lại answer và thấy model trả lời về "cấp quyền tạm thời 24 giờ" thay vì "auto-escalate sau 10 phút" — hai chunk cùng document, cùng nói về P1 incident, nhưng embedding score của chunk cấp quyền lại cao hơn. Giả thuyết ban đầu của tôi là do chunking bị cắt sai, nhưng sau khi `list_chunks()` thì thấy chunk hoàn toàn hợp lý. Nguyên nhân thực sự là dense embedding không phân biệt được "P1 + cấp quyền" với "P1 + escalation" khi query là "escalation diễn ra như thế nào". Việc debug theo Error Tree (indexing → retrieval → generation) giúp tôi tìm ra đúng nguyên nhân thay vì sửa lung tung.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** q06 — "Escalation trong sự cố P1 diễn ra như thế nào?"

**Phân tích:**

Baseline trả lời: *"On-call IT Admin có thể cấp quyền tạm thời (tối đa 24 giờ) sau khi được Tech Lead phê duyệt bằng lời..."* — Faithfulness 5/5 (hoàn toàn grounded trong context), nhưng Completeness chỉ 2/5 vì expected answer là "tự động escalate lên Senior Engineer nếu không có phản hồi trong 10 phút". Lỗi nằm ở **retrieval/rerank**: dense search đẩy chunk về cấp quyền tạm thời lên rank cao hơn chunk về escalation timeline do cả hai đều chứa từ "P1" và "incident", nhưng chunk cấp quyền có embedding gần hơn với query về mặt cosine.

Sau khi thêm LLM Rerank, variant trả lời đầy đủ cả 5 bước escalation bao gồm "nếu không có phản hồi trong 10 phút, ticket tự động escalate lên Senior Engineer" — Completeness tăng từ 2 lên 5. LLM đọc được câu hỏi yêu cầu **quy trình** (có thứ tự, có timeline) nên chọn đúng chunk chứa numbered steps thay vì chunk chứa exception case.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Tôi sẽ thêm bước **abstain detection trước LLM Rerank**: nếu tất cả candidates có cosine score < 0.3 thì trả về "Không đủ dữ liệu" ngay, không rerank. Lý do: q09 (ERR-403-AUTH) bị giảm completeness sau LLM Rerank vì model cố tìm chunk "gần nhất" dù không có thông tin — dẫn đến answer sai hướng. Ngưỡng score thấp là tín hiệu rõ ràng nhất cho abstain case và không cần LLM call thêm để quyết định.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*