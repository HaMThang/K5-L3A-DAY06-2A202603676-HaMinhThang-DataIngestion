# So sánh embedding trên cùng corpus và benchmark

Giữ cùng chunk, câu hỏi và LLM. Điểm dưới đây đo truy xuất evidence chính xác, không phải độ chính xác câu trả lời LLM.

| Run | Embedding | Split | Cấu hình | Hit@1 | Evidence Hit@3 | Query embed ms | Search ms | LLM ms |
|---|---|---|---|---|---|---|---|---|
| mini4o_01_small | text-embedding-3-small | dev | A | 3/15 | 5/15 | 595.4 | 22.6 | 4045.3 |
| mini4o_01_small | text-embedding-3-small | dev | B | 9/15 | 12/15 | 595.4 | 16.1 | 4537.5 |
| mini4o_01_small | text-embedding-3-small | dev | C | 10/15 | 13/15 | 595.4 | 15.8 | 3861.4 |
| mini4o_01_small | text-embedding-3-small | test | A | 1/5 | 3/5 | 762.0 | 24.0 | 3749.3 |
| mini4o_01_small | text-embedding-3-small | test | B | 1/5 | 4/5 | 762.0 | 14.2 | 2595.4 |
| mini4o_01_small | text-embedding-3-small | test | C | 1/5 | 4/5 | 762.0 | 15.4 | 4060.0 |
| mini4o_01_large | text-embedding-3-large | dev | A | 6/15 | 7/15 | 493.0 | 25.7 | 3705.0 |
| mini4o_01_large | text-embedding-3-large | dev | B | 13/15 | 15/15 | 493.0 | 19.1 | 3479.5 |
| mini4o_01_large | text-embedding-3-large | dev | C | 13/15 | 15/15 | 493.0 | 15.9 | 4183.7 |
| mini4o_01_large | text-embedding-3-large | test | A | 2/5 | 4/5 | 411.2 | 26.2 | 3134.1 |
| mini4o_01_large | text-embedding-3-large | test | B | 3/5 | 5/5 | 411.2 | 17.5 | 4189.5 |
| mini4o_01_large | text-embedding-3-large | test | C | 3/5 | 5/5 | 411.2 | 15.4 | 2954.5 |

Cấu hình chọn trên dev:
- mini4o_01_small: C; dev Evidence Hit@3, then Hit@1, then median query embedding + search latency; no test tuning
- mini4o_01_large: C; dev Evidence Hit@3, then Hit@1, then median query embedding + search latency; no test tuning

| Run | Số chiều | Ingest embedding giây (A+B) | Tổng chi phí API ước tính USD |
|---|---|---|---|
| mini4o_01_small | 1536 | 41.90 | 0.217176 |
| mini4o_01_large | 3072 | 46.27 | 0.247341 |

Chi phí ước tính theo token được API trả về; xem frozen_config.json và summary từng split. Không bao gồm lần probe kết nối, request lỗi/retry hoặc chi phí máy local. Thời gian gồm network/CPU khác nhau giữa local và API, không phải benchmark phần cứng đồng nhất.

Nhãn grounding, chất lượng câu trả lời và metadata thủ công vẫn cần người đọc chấm. Kiểm tra citation chỉ xác nhận ID/quote tồn tại, không xác nhận suy luận đúng. Tập test chỉ có 5 câu từ một tài liệu servo; kết luận chỉ áp dụng corpus này.
