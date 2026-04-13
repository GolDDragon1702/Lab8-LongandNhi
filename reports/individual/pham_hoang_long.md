# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Phạm Hoàng Long  
**Vai trò trong nhóm:** Làm sprint 1, 4  
**Ngày nộp:** 13/4/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi chịu trách nhiệm với 2 sprint là 1 và 4. Cụ thể, ở Sprint 1, tôi trực tiếp implement logic tiền xử lý và chiến thuật chunking dựa trên heading để duy trì tính toàn vẹn của các điều khoản chính sách. Còn ở sprint 4, tôi thiết lập khung đánh giá (Evaluation Framework) sử dụng LLM-as-judge để tự động hóa việc chấm điểm cho 10 câu hỏi kiểm định dựa trên 4 metric tiêu chuẩn.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu sâu sắc hơn về sự khác biệt giữa "Semantic Similarity" và "Relevance". Trước đây, tôi lầm tưởng rằng chỉ cần vector embedding đủ tốt (Dense Retrieval) là có thể giải quyết mọi bài toán tìm kiếm. Tuy nhiên, khi đối mặt với các truy vấn chứa mã lỗi đặc thù như `ERR-403-AUTH` hay các thuật ngữ chuyên môn như `SLA P1`, tôi nhận ra Dense Retrieval thường bị nhiễu bởi các đoạn text có chung ngữ cảnh nhưng khác biệt về chi tiết kỹ thuật. Việc kết hợp Hybrid Retrieval (kèm BM25) là thực sự cần thiết để "neo" (anchor) kết quả vào các từ khóa chính xác. Ngoài ra, tôi cũng hiểu rõ hơn về quy trình "Evaluation-driven development": thay vì đoán mò xem pipeline có tốt hay không, việc có một scorecard với các chỉ số Faithfulness và Recall giúp tôi đưa ra quyết định tuning dựa trên dữ liệu thực thay vì cảm tính.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều khiến tôi bất ngờ nhất là kết quả của Variant 1 (Hybrid + Rerank) lại thấp hơn đáng kể so với Baseline ở một số câu hỏi kỹ thuật. Ban đầu, tôi kỳ vọng việc thêm Reranker sẽ giúp lọc noise hiệu quả hơn, nhưng thực tế nó lại "bóp nghẹt" các chunk chứa thông tin chính xác. Cụ thể ở câu hỏi về mã lỗi `ERR-403`, Reranker đã hạ thấp điểm số của chunk chứa cách xử lý lỗi vì chunk đó có cấu trúc câu hơi khác so với query, dẫn đến việc model trả lời "Tôi không biết". Khó khăn lớn nhất là việc tinh chỉnh RRF weights và lựa chọn reranker model phù hợp với ngôn ngữ tiếng Việt. Tôi mất rất nhiều thời gian để debug tại sao retrieval mode trả về đúng file nhưng stage reranking lại làm mất đi evidence cần thiết cho generation.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** q09: ERR-403-AUTH là lỗi gì và cách xử lý?

**Phân tích:**
- **Baseline:** Trả lời rất tốt (Faithfulness 5/5, Relevance 5/5). Model tìm thấy chính xác chunk trong `it_helpdesk_faq.txt` và trả lời đầy đủ ý về nguyên nhân (quyền truy cập) và cách xử lý (liên hệ IT hoặc reset quyền).
- **Lỗi:** Mặc dù điểm recall cho thấy document đã được tìm thấy, nhưng trong bản Variant, model lại trả về "Tôi không biết".
- **Nguyên nhân:** Lỗi nằm ở bước **Retrieval/Reranking**. Dưới cơ chế Hybrid, BM25 đã đưa được chunk chứa keyword `ERR-403` vào top 10. Tuy nhiên, mô hình Cross-Encoder dùng để rerank có vẻ không được tối ưu cho các mã lỗi dạng chuỗi ký tự khô khan này, dẫn đến việc nó chấm điểm thấp cho chunk đúng và đẩy các chunk có semantic similarity về "quản lý quyền truy cập" chung chung (nhưng không chứa mã lỗi) lên top 3. Khi đưa vào prompt, context block thiếu mất mã lỗi cụ thể, model bám sát quy tắc "Abstain: Chỉ trả lời từ context" và từ chối trả lời.
- **Variant cải thiện hay không:** Không cải thiện, thậm chí là tệ hơn (Regression). Bài học rút ra là reranker cần được test rất kỹ với các dạng truy vấn Edge-case trước khi áp dụng cho toàn bộ hệ thống.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ thử hai cải tiến cụ thể:
1. **Query Transformation (HyDE):** Tôi sẽ dùng LLM để sinh ra một câu trả lời giả định cho các mã lỗi kỹ thuật trước khi embed. Điều này giúp tăng đáng kể recall cho Dense stage vì embedding của "câu trả lời giả định" sẽ gần với "câu trả lời thật" trong document hơn là câu hỏi ngắn của người dùng.
2. **Multi-stage Reranking:** Thay vì chỉ dùng một model Cross-encoder duy nhất cho top 10, tôi sẽ thử kết hợp thêm logic heuristic dựa trên keyword matching (Exact match check) trước khi đưa vào LLM để đảm bảo các thông tin quan trọng như ID lỗi không bao giờ bị lọc mất.

---