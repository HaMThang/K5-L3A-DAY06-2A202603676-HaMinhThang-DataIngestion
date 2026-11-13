# Báo cáo cá nhân — Day 6

- **Họ và tên:** Hà Minh Thắng
- **MSSV:** 2A202603676
- **Lớp:** L3A
- **Repo bài nộp:** https://github.com/HaMThang/K5-L3A-DAY06-2A202603676-HaMinhThang-DataIngestion
- **Commit chốt bài:** Chưa chốt.

## Dữ liệu và pipeline

- **Corpus:** 31 nguồn (15 bài Arduino Learn + 16 trang Python 3.13 Tutorial), chia thành 292 section theo heading; 105 section có ≥1.500 ký tự. Giấy phép: Arduino CC-BY-SA-4.0 (README), Python PSF License v2 / Zero-Clause BSD cho mã ví dụ. **Cần xác minh:** `LICENSE.md` của Arduino còn nhắc CC BY 4.0, không khớp phần CC-BY-SA bên dưới.
- **Chuẩn hóa:** Tách theo heading (Arduino Markdown cấp 2–6; Python HTML lấy `div.body`/`main`), chuẩn hóa Unicode NFC và khoảng trắng, giữ code block, `page` mã hóa theo heading anchor. Không dùng OCR; ảnh không được mirror đầy đủ.
- **Mô hình:** OpenAI `text-embedding-3-small` (1.536 chiều) và `text-embedding-3-large` (3.072 chiều); `gpt-4o-mini` để sinh câu trả lời. PostgreSQL 16 + pgvector 0.8.7, tìm kiếm exact cosine.
- **Metadata:** Bắt buộc `document_id`, `source`, `page`; bổ sung `section`, `category`, `language`, v.v. Filter `category` chỉ dùng cho Arduino (category của Python để trống).
- **Chi phí toàn run:** small **$0.217176**, large **$0.247341**. Thời gian embedding ingest A+B: **41.90 s** và **46.27 s** (C dùng lại vector của B).

## Cấu hình và benchmark

- **A:** chunk ký tự cố định 500, overlap 50.
- **B:** gom theo paragraph, tối đa 1.000 ký tự.
- **C:** như B, thêm metadata filter (chỉ q03, q14 ở dev và q16 ở test).
- **Bộ câu hỏi:** 20 câu (15 dev / 5 test); 20/20 evidence hợp lệ.
- **Tiêu chí chọn (ghi trước):** Evidence Hit@3 dev → Hit@1 → median thời gian embedding query + search. Không chỉnh theo test. Kết quả: chọn **C** cho cả hai run.

## Kết quả

Số trong cột thời gian là median **embedding query / search / LLM (ms)**; chi phí là ước tính LLM cho split đó. Chunk count không đổi theo embedding model.

| Embedding | Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Median E/S/LLM (ms) | LLM cost (USD) | Filter tuân thủ |
|---|---:|---|---:|---:|---:|---:|---:|---|
| small | A | dev | 3/15 | 5/15 | 1053 | 595.4 / 22.6 / 4045.3 | 0.051266 | — |
| small | B | dev | 9/15 | 12/15 | 611 | 595.4 / 16.1 / 4537.5 | 0.053090 | — |
| small | C | dev | 10/15 | 13/15 | 611 | 595.4 / 15.8 / 3861.4 | 0.055634 | 2/2 |
| small | A | test | 1/5 | 3/5 | 1053 | 762.0 / 24.0 / 3749.3 | 0.015178 | — |
| small | B | test | 1/5 | 4/5 | 611 | 762.0 / 14.2 / 2595.4 | 0.018966 | — |
| small | C | test | 1/5 | 4/5 | 611 | 762.0 / 15.4 / 4060.0 | 0.018630 | 1/1 |
| large | A | dev | 6/15 | 7/15 | 1053 | 493.0 / 25.7 / 3705.0 | 0.052324 | — |
| large | B | dev | 13/15 | 15/15 | 611 | 493.0 / 19.1 / 3479.5 | 0.057288 | — |
| large | C | dev | 13/15 | 15/15 | 611 | 493.0 / 15.9 / 4183.7 | 0.055786 | 2/2 |
| large | A | test | 2/5 | 4/5 | 1053 | 411.2 / 26.2 / 3134.1 | 0.017310 | — |
| large | B | test | 3/5 | 5/5 | 611 | 411.2 / 17.5 / 4189.5 | 0.017718 | — |
| large | C | test | 3/5 | 5/5 | 611 | 411.2 / 15.4 / 2954.5 | 0.018234 | 1/1 |

**Lưu ý:** Các điểm coherence, grounding và metadata chưa được chấm thủ công; không suy ra từ Evidence Hit@3. Tỷ lệ filter là kiểm tra tự động.

## Phân tích và kết luận

- **A–B:** Chunk theo paragraph giảm số chunk (1.053 → 611) và tăng Evidence Hit@3 rõ rệt. Với large: dev 7/15 → 15/15, test 4/5 → 5/5.
- **B–C:** Filter tuân thủ đầy đủ (2/2 dev, 1/1 test). Với large, B và C bằng nhau về độ chính xác; C chỉ giảm median search (19.1 → 15.9 ms ở dev, 17.5 → 15.4 ms ở test). Số câu có filter quá ít nên không kết luận được filter cải thiện retrieval nói chung.
- **Ví dụ lỗi:** q16 (ba dây servo): small C không đưa evidence vào top-3, large C xếp đúng lên đầu. q19 (mục Sweep): large C có evidence trong top-3 nhưng top-1 là section lân cận `#sweep-circuit`, nên Hit@1 trượt dù Evidence Hit@3 đạt.
- **Đánh đổi:** Large C đạt 15/15 dev và 5/5 test; small C đạt 13/15 và 4/5 với chi phí toàn run thấp hơn. Test chỉ có 5 câu nên chênh lệch này không mang tính tổng quát.
- **Hạn chế và bước tiếp:** Test chỉ gồm 5 câu từ một tài liệu servo; thời gian phụ thuộc mạng/API. Cần chấm thủ công 10 mẫu coherence/strategy, 5 câu grounding/strategy và độ đúng metadata; mở rộng test sang nhiều tài liệu/category; xác minh lại license Arduino trước khi chia sẻ corpus.
