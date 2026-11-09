# 🕸️ Graph Attention Networks (GAT) for Classification

> **Nghiên cứu và triển khai mô hình mạng nơ-ron chú ý đồ thị (Graph Attention Networks) cho các bài toán phân loại dữ liệu.**

## 📖 Giới thiệu
Kho lưu trữ mã nguồn và tài liệu nghiên cứu chuyên sâu về cơ chế Attention trên dữ liệu cấu trúc đồ thị. Dự án bao gồm việc hiện thực hoá lớp Graph Attention từ nền tảng, cơ chế lan truyền thông điệp (message passing) giữa các node lân cận với trọng số học được, và thực nghiệm phân loại số / node classification.

## 📁 Cấu trúc thư mục
- `GraphAttention/`: Module triển khai các lớp GAT layer và cơ chế self-attention cho đồ thị.
- `number_classification.py`: Kịch bản huấn luyện và đánh giá mô hình phân loại.
- `Nghien cuu ung dung attention vao do thi.docx`: Tài liệu tổng quan lý thuyết và kết quả thực nghiệm.
- `test_graph.py`, `test_node.py`: Bộ kiểm thử đơn vị cho các toán tử đồ thị.

## 🛠️ Công nghệ sử dụng
- Python, PyTorch / PyTorch Geometric, NetworkX, NumPy
