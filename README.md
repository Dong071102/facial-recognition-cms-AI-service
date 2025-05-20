# Human Face Attendance Service

**Human Face Attendance Service** là một dịch vụ điểm danh thông minh bằng cách **nhận diện khuôn mặt thời gian thực** qua **camera IP hoặc webcam**, sử dụng các công nghệ như:

- 🧠 **MobileFaceNet** – Nhận diện khuôn mặt chính xác, nhẹ.
- 📦 **MTCNN** – Phát hiện khuôn mặt nhanh và chính xác.
- 📹 **OpenCV + WebSocket** – Stream video và xử lý real-time.
- 🔥 **PyTorch, YOLO, base64 encoded image**.

---

## 📁 Cấu trúc thư mục chính

```
.
├── src/
│   ├── server.py                  # WebSocket server nhận diện khuôn mặt
│   ├── face_model.py             # MobileFaceNet model
│   ├── facebank.py               # Load embeddings học sinh
│   ├── DB/database.py            # Kết nối database, lưu điểm danh
│   ├── MTCNN/                    # Module phát hiện khuôn mặt
│   └── utils/                    # Fonts, xử lý phụ
├── evidence_image/               # Thư mục ảnh chụp điểm danh
├── main.py                       # Điểm chạy chính
├── requirements.txt
├── pyproject.toml
├── README.md
```

---

## 🚀 Cài đặt

### ⚙️ Yêu cầu:

- Python 3.11.x
- pip ≥ 21
- CUDA 12+ (nếu dùng GPU)
- Hệ điều hành: Ubuntu / Windows đều chạy tốt

### 1. Tạo môi trường ảo (venv):

```bash
python -m venv venv
source venv/bin/activate     # (Linux/macOS)
venv\Scripts\activate.bat    # (Windows)
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

Hoặc nếu dùng `pyproject.toml`:

```bash
pip install .
```

---

## ▶️ Cách chạy

```bash
python src/server.py
```

Hoặc:

```bash
python main.py
```

Nếu chạy thành công, bạn sẽ thấy log:

```
Khởi động WebSocket server duy nhất trên ws://0.0.0.0:11000
```

---

## 🧠 Chức năng chính

### 📡 WebSocket server (port `11000`)

- Mỗi client (ESP32-CAM hoặc web app) kết nối qua WebSocket theo `ws://<server-ip>:11000/ws/<classroom_id>`
- Server tự động lấy thông tin lớp học, lịch học, camera IP từ database (qua `get_all_classes_in_next_2_hours()`)

### 🤖 Nhận diện khuôn mặt & điểm danh

- Dùng MTCNN để **phát hiện khuôn mặt**
- Dùng MobileFaceNet để **embedding** và so sánh
- Nếu nhận diện được và chưa từng điểm danh → lưu ảnh + gọi `attencace_student(...)` để ghi vào DB
- Ảnh lưu trong `evidence_image/YYYY/MM/DD/<schedule_id>_<student_id>.jpg`

---

## 📤 Dữ liệu gửi về client

- Mỗi frame gửi về dạng `base64` ảnh JPEG (240x320)
- Khi nhận diện được khuôn mặt hợp lệ, client sẽ nhận JSON:

```json
{
  "id": "student123",
  "image": "<base64_image>"
}
```

---

## 🛠️ Tùy chỉnh

### 📌 Thay đổi camera:

Trong `server.py`:

```python
cap = cv2.VideoCapture(0)  # Hoặc đường dẫn IP camera
```

### 🧪 Thử với file video:

```python
cap = cv2.VideoCapture("tv3_sau.mp4")
```

---

## 💾 Cơ sở dữ liệu cần

Đảm bảo bạn đã có:

- Bảng lịch học (`schedule`)
- Bảng lớp học (`class`)
- Bảng học sinh (`student`)
- Bảng điểm danh (`attendance`)
- Bảng ánh xạ IP camera (`camera_ip`)
- Hàm `get_all_classes_in_next_2_hours`, `attencace_student`, `init_attendace`, `get_pytorch_embedding` hoạt động.

---

## 🧑‍💻 Dev & Tác giả

- **Tên**: Vũ Bá Đông  
- 📧 **Email**: [vubadong071102@gmail.com](mailto:vubadong071102@gmail.com)

---

## 📄 License

MIT License
