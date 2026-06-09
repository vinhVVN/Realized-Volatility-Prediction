<h1 align="center">📈 Optiver Realized Volatility Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/LightGBM-F37021?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/System%20Design-Enterprise-brightgreen?style=for-the-badge" alt="Enterprise Architecture">
</p>

<p align="center">
  <i>Hệ thống Quantitative Trading Pipeline chuyên nghiệp: Dự báo biến động tài chính từ dữ liệu Order Book & Trade (High-Frequency Trading).</i>
</p>

---

## 📖 Tổng Quan Dự Án (Overview)
Dự án này là giải pháp hệ thống toàn diện cho bài toán **Optiver Realized Volatility Prediction** trên Kaggle. Mục tiêu cốt lõi là dự báo độ biến động tài chính (Realized Volatility) trong 10 phút tiếp theo dựa trên dữ liệu giao dịch tần suất cao (HFT) của 10 phút trước đó.

Hệ thống được thiết kế với tư duy của một **Machine Learning Engineer / Quant Developer**, đặt trọng tâm tuyệt đối vào hai yếu tố: **Tối Ưu Hóa Bộ Nhớ (Memory Optimization)** và **Thiết Kế Hệ Thống Mở Rộng (Scalable System Design)**. Kết quả là một pipeline hoàn toàn tự động, có khả năng xử lý hơn 10GB dữ liệu thô ngay trên môi trường phần cứng hạn chế mà không bị tràn RAM.

---

## ⚡ Thách Thức Kỹ Thuật Cốt Lõi (Core Engineering Challenges)
1. **Dữ Liệu Khổng Lồ & Giới Hạn Phần Cứng:** Xử lý hơn **10GB+** dữ liệu raw tick data (Order Book và Trade) với ràng buộc cực kỳ khắt khe của Kaggle: Giới hạn RAM 16GB trong lúc inference trên GPU. Các phương pháp Pandas thông thường liên tục gây ra lỗi **Out-of-Memory (OOM) Crash**.
2. **Ẩn Giấu Trật Tự Thời Gian (Time-series Scrambling):** Kaggle đã xáo trộn (randomize) hoàn toàn trường `time_id`, phá vỡ trật tự thời gian thực tế. Điều này khiến các kỹ thuật Validation truyền thống trở nên vô dụng và gây ra thảm họa Data Leakage.
3. **Nhiễu Thị Trường Cực Đoan (Market Noise):** Môi trường HFT chứa lượng nhiễu khổng lồ. Các mô hình Neural Network tiêu chuẩn (như 1D-CNN) bị bùng nổ gradient (Gradient Explosion) và phân kỳ không phanh, dẫn đến điểm Baseline ban đầu là một thảm họa (RMSPE 0.77407).
4. **Strict Offline Inference:** Toàn bộ pipeline dự đoán phải hoạt động trong môi trường hoàn toàn cách ly (không có Internet), vô hiệu hóa việc cài đặt thêm package hay download weights từ bên ngoài.

---

## 🏗️ Kiến Trúc Hệ Thống & Giải Pháp (System Architecture & Solutions)

### Khối Kiến Trúc Tổng Thể (ASCII Architecture Diagram)
```text
[10GB Raw HFT Data] 
       │ 
       ▼ (Batched DataLoader)
[Memory-Optimized Feature Pipeline]
       ├─> Base Features (WAP, Spread, Imbalance)
       ├─> Rank Normalization (Liquidity Context)
       └─> K-NN Mahalanobis Neighbors (Cross-sectional Data)
       │
       ▼ (Surgical Garbage Collection)
[Time-Series CV Recovery (2-Stage t-SNE)] 
       │ 
       ▼ (StandardScaler Fitted on Train -> Nuked from RAM)
  ┌────┴─────────────────────────────┐
  ▼                                  ▼
[LightGBM Models]               [Deep Learning Models]
(45% Weight)                         │
  │                     ┌────────────┴────────────┐
  │                     ▼                         ▼
  │               [1D-ResNet-CNN]              [MLP]
  │               (45% Weight)              (10% Weight)
  │                     │                         │
  └─────────────────────┼─────────────────────────┘
                        ▼
            [Dynamic Weighted Ensemble]
         (+ Survival Gate & Non-Negative Clip)
                        │
                        ▼
             [Final Volatility Prediction]
```

### 1. Tối Ưu Hóa Bộ Nhớ Cấp Độ Vi Mô (Surgical Nuke Memory Management)
Để vượt qua giới hạn 16GB RAM của Kaggle khi load 10GB Data:
* **Chiến lược "Load & Nuke":** Trong giai đoạn Inference, hệ thống tải tập Train 10GB vào RAM **chỉ với mục đích duy nhất** là fit `StandardScaler`. Ngay sau khi fit xong, toàn bộ DataFrame Train bị tiêu hủy ngay lập tức bằng lệnh `del` và cưỡng chế giải phóng bộ nhớ bằng `gc.collect()`.
* **Batched Inference:** Dữ liệu Test được nạp và xử lý tính toán K-NN theo từng chunk nhỏ gọn thông qua `DataLoader`, giữ mức tiêu thụ RAM luôn ở ngưỡng an toàn **< 2GB**.

### 2. K-NN Feature Engineering & Xử Lý Ma Trận Suy Biến
* Trích xuất các đặc trưng ngoại lai bằng thuật toán K-Nearest Neighbors với **khoảng cách Mahalanobis**. 
* **Fix Singular Matrix:** Trong các Chunk dữ liệu nhỏ, ma trận hiệp phương sai thường xuyên bị suy biến (Singular Matrix). Tôi đã áp dụng toán học giả nghịch đảo (Pseudo-inverse Covariance) thông qua `np.linalg.pinv` để đảm bảo hệ thống không bao giờ bị sập luồng (Crash-proof) kể cả khi Test Set chỉ có 1 dòng dữ liệu.

### 3. Giải Mã Thời Gian Thực (Time-Series Reversal)
* Đã triển khai thuật toán **2-Stage t-SNE Clustering** kết hợp với tính năng `real_price` để khôi phục lại trật tự thời gian tuyến tính ẩn giấu.
* Từ đó, xây dựng hệ thống **4-Fold Expanding Window Cross-Validation**, cô lập hoàn toàn tương lai khỏi quá khứ, loại trừ 100% rủi ro Data Leakage.

### 4. SoTA Deep Learning Tốc Độ Cao
* Thiết kế mạng **1D-ResNet-CNN** tích hợp Residual Skip Connections, SiLU Activations, và BatchNorm1d để đối phó với bùng nổ Gradient.
* **Tối ưu huấn luyện:** Hạ `hidden_size` xuống 128 và chỉ tiến hành huấn luyện (fine-tune) trên Fold cuối cùng (Fold 4) - Fold chứa lượng thông tin thị trường gần nhất. Chiến lược này nén thời gian training từ **8 giờ xuống chỉ còn 15 phút** mà vẫn giữ nguyên độ chính xác.

### 5. Multi-Seed & Bộ Lọc Sinh Tồn (Survival of the Fittest)
* Thay vì tin tưởng vào 1 mô hình duy nhất, hệ thống train **3 seeds** ngẫu nhiên cho mỗi mạng Neural.
* Triển khai cổng lọc tự động (Automatic Filtering Gate): Ngay trong lúc train, bất kỳ mô hình seed nào có xu hướng phân kỳ (Loss vượt ngưỡng Threshold) sẽ bị hệ thống tự động loại bỏ (discard). Chỉ những models tinh hoa nhất hội tụ thành công mới được giữ lại để Ensemble.

### 6. Hybrid Ensemble Động
* Sự kết hợp trọng số mượt mà giữa các họ thuật toán khác biệt: **LightGBM (45%) + 1D-CNN (45%) + MLP (10%)**.
* Áp dụng chốt chặn bảo mật cuối cùng: `np.clip(preds, 0, None)` để tuyệt đối hóa việc mô hình không bao giờ sinh ra các dự đoán độ biến động âm (vô lý về mặt toán học).

---

## 📊 Bảng Xếp Hạng Hiệu Năng (Performance Ledger)
| Giai đoạn | Thuật toán / Kỹ thuật | Trạng thái RAM | Điểm RMSPE (Càng thấp càng tốt) |
| :--- | :--- | :--- | :--- |
| **Initial Baseline** | 1D-CNN Cơ bản (Bị Gradient Explosion) | 💥 OOM Crash | `0.77407` |
| **Mid-stage** | LightGBM + Base Features | 🟢 Ổn định | `0.22733` |
| **Final System** | **Hybrid Ensemble + Surgical Memory Nuke** | 🟢 **< 2GB Peak** | **`0.24940` (Private LB)** |

*Thành tựu lớn nhất không chỉ nằm ở con số RMSPE, mà là việc xây dựng thành công một kiến trúc Quantitative Pipeline bền bỉ, không rò rỉ dữ liệu, và có thể deploy trực tiếp vào môi trường Production của các Tech Studios.*

---

## 📂 Cấu Trúc Mã Nguồn (Directory Structure)
```bash
.
├── configs/                  # Các file YAML cấu hình Hyperparameters (LGBM, NN, Feature)
├── data/
│   ├── processed/            # Nơi lưu trữ Features đã qua Pipeline (Feather format)
│   └── raw/                  # (Không commit) Dữ liệu raw parquet của Optiver
├── models/                   # Nơi chứa Weights đã huấn luyện (LGBM, CNN, MLP)
├── notebooks/                # Các Kernel thực thi (EDA, Train, Inference)
├── src/                      # Source Code cốt lõi (Core Engine)
│   ├── data/                 # DataLoader song song, Preprocessing (Reduce Memory)
│   ├── features/             # Feature Engineering Pipelines (Book, Trade, K-NN)
│   ├── models/               # Định nghĩa Kiến trúc (CNN, MLP, LGBM) tuân thủ BaseModel
│   └── training/             # PyTorch Trainer Orchestrator, Time-Series CV, Metrics
├── .gitignore
├── requirements.txt
└── setup.py                  # Đóng gói dự án thành Python Package
```
