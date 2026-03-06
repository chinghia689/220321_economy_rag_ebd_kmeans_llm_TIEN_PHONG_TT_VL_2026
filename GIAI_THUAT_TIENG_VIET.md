## 1. Tổng quan hệ thống
Hệ thống được xây dựng trên tập dữ liệu **27.682 văn bản kinh tế** (*dataset_economy*). Mục tiêu là cải thiện chất lượng truy hồi trong mô hình RAG thông qua cơ chế lọc hai tầng: kiểm tra độ tương đồng và chọn cụm tối ưu bằng Energy Distance.
## 2. Tiền xử lý dữ liệu
### 2.1. Phân đoạn văn bản
Toàn bộ tập dữ liệu được chia nhỏ bằng phương pháp:
* **RecursiveCharacterTextSplitter**
* `chunk_size = 600`
* `chunk_overlap = 80`
Việc phân đoạn giúp bảo toàn ngữ cảnh cục bộ và tăng độ chính xác truy hồi.
### 2.2. Biểu diễn vector
Mỗi đoạn văn được ánh xạ sang không gian vector 768 chiều bằng mô hình:
> `intfloat/multilingual-e5-base`
Do đó:
* Mỗi document chunk → vector kích thước **(1 × 768)**
* Query embedding → vector kích thước **(1 × 768)**
### 2.3. Lưu trữ vector
Các embedding được lưu trong:
* **ChromaDB Vector Store**
* Thước đo tương đồng: **Cosine Similarity**
* Cấu hình: `chroma_cosine`
## 3. Truy hồi ban đầu (Initial Retrieval)
Khi nhận truy vấn từ người dùng, hệ thống thực hiện:
1. Embed truy vấn → vector (1 × 768).
2. Truy hồi **top-30 documents** có cosine similarity cao nhất.
## 4. Kiểm tra chất lượng truy hồi
Sau khi lấy top-30 tài liệu, hệ thống tính:
max(cosine similarity)
Ngưỡng kiểm tra được đặt là:
cosine similarity = 0.5
* Nếu `max similarity ≥ 0.5` → kích hoạt cơ chế lọc nâng cao.
* Nếu `max similarity < 0.5` → Dừng hẳn.
## 5. Cơ chế lọc nâng cao
Khi truy hồi ban đầu không đạt chất lượng mong muốn, hệ thống thực hiện hai bước bổ sung:
### 5.1. K-Means Clustering
Tập 30 tài liệu truy hồi được phân cụm bằng K-Means với:
k = min(7, số lượng documents)
Tức là:
* Số cụm tối đa: **7 cụm**
* Các cụm được đánh số: Cluster 0 → Cluster 6
Mỗi cụm đại diện cho một nhóm ngữ nghĩa khác nhau trong tập truy hồi.
### 5.2. Tính Energy Distance
Đối với từng cụm, hệ thống tính **Energy Distance** giữa:
* Vector truy vấn (1 × 768)
* Tập vector trong cụm
Mục tiêu là đánh giá mức độ khác biệt giữa truy vấn và toàn bộ phân phối vector của cụm, thay vì chỉ so sánh với centroid.
### 5.3. Chọn cụm tối ưu
Ba cụm có **Energy Distance nhỏ nhất** được chọn làm:
> Best Cluster
## 6. Sinh câu trả lời
Toàn bộ tài liệu thuộc Best Cluster được sử dụng làm ngữ cảnh đầu vào cho mô hình ngôn ngữ để sinh câu trả lời cuối cùng.
# Đặc điểm phương pháp
Hệ thống bao gồm:
* Truy hồi ban đầu: Top-30 cosine similarity
* Kiểm tra ngưỡng: 0.4
* Phân cụm tối đa: 7 cụm
* So sánh ở mức phân phối: Energy Distance
* Embedding dimension: 768
* Dataset: 27.682 văn bản
Phương pháp này giúp:
* Giảm nhiễu khi top-40 chứa nhiều chủ đề
* Ổn định hơn với truy vấn có độ tương đồng thấp
* Tăng tính nhất quán ngữ nghĩa trước bước sinh


