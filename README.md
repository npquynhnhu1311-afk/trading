# 📈 HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC ĐỊNH LƯỢNG: EMA KẾT HỢP OBV (CỔ PHIẾU ACB)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Dự án Nghiên cứu & Ứng dụng Tài chính Định lượng (Quantitative Trading Framework)**  
> **Đơn vị thực hiện:** Nhóm 3 - Chuyên đề Chiến lược Giao dịch Thuật toán  
> **Mã cổ phiếu mục tiêu:** Ngân hàng TMCP Á Châu (`ACB` - Sàn HOSE)  
> **Thời gian khảo sát:** 10 năm dữ liệu nhật trình (2014 – 2023)

---

## 🎯 1. TỔNG QUAN ĐỀ TÀI & Ý TƯỞNG CHIẾN LƯỢC

Dự án phát triển và triển khai ứng dụng Web tương tác trên nền tảng **Streamlit** nhằm kiểm định tính hiệu quả của chiến lược giao dịch định lượng kết hợp giữa **Đường Trung bình Động Hàm mũ (EMA)** và **Chỉ số Khối lượng Cân bằng (On-Balance Volume - OBV)** trên dữ liệu lịch sử cổ phiếu **ACB**.

### 💡 Ý Tưởng Chiến Lược:
1. **EMA (Exponential Moving Average):** Đóng vai trò là **bộ lọc xu hướng động**, phản ứng nhanh với biến động giá gần nhất.
2. **OBV (On-Balance Volume) & OBV Slope:** Đóng vai trò là **thước đo dòng tiền thực**, xác nhận xung lực mua/bán từ phía tổ chức và khối lượng tích lũy.
3. **Sự kết hợp EMA + OBV:** Mở vị thế khi và chỉ khi giá vừa phá vỡ ngưỡng xu hướng VÀ dòng tiền xác nhận gia tăng, giúp giảm thiểu tối đa các bẫy tín hiệu giả (**False Breakouts / Whipsaws**).

---

## 🛠️ 2. CÁC NÂNG CẤP KỸ THUẬT QUAN TRỌNG ĐÃ ĐƯỢC CHẤN CHỈNH

Dựa trên phản biện chuyên môn từ Giảng viên hướng dẫn: *"Chiến lược chưa hạch toán đúng vị thế giao dịch (trading position), nên các con số 'tối ưu' được báo cáo có thể không chính xác."*, hệ thống đã hoàn thiện toàn diện các nội dung:

1. **Hạch toán Vị thế Chuẩn Xác (Position Accounting):**
   - Thiết lập mô hình máy trạng thái nhị phân: **$0$ (Cầm Tiền mặt / Flat)** và **$1$ (Nắm giữ Cổ phiếu / Long)**.
   - Chỉ kích hoạt lệnh **MUA** khi trạng thái chuyển từ $0 \rightarrow 1$ (không mua gom lặp lại khi đang ôm lệnh).
   - Chỉ kích hoạt lệnh **BÁN** khi trạng thái chuyển từ $1 \rightarrow 0$ (không báo bán khống khi tài khoản đang cầm tiền mặt).
2. **Triệt tiêu Look-Ahead Bias:**
   - Mọi tín hiệu giao dịch đều được trễ một phiên (`.shift(1)`), đảm bảo chỉ sử dụng dữ liệu phiên $t-1$ đã đóng nến để ra quyết định và khớp lệnh ở phiên $t$.
3. **Chi Phí Giao Dịch Thực Tế & Quản Trị Rủi Ro:**
   - **Phí giao dịch (Fees):** $0.2\%$ cho mỗi chiều mua/bán.
   - **Trượt giá (Slippage):** $0.1\%$ cho mỗi chiều giao dịch.
   - **Cắt lỗ Tuyệt đối (Stop-Loss):** $7.0\%$ cố định tính từ giá vốn mua vào.
4. **Loại Bỏ Hiện Tượng "Sharpe Ảo":**
   - Áp dụng ràng buộc cứng: Số lệnh đã đóng $\ge 5$ (`MIN_TRADES >= 5`) trong hàm mục tiêu tối ưu hóa, loại trừ các trường hợp Sharpe vô hạn (`inf`) do thuật toán "ăn may" đứng ngoài thị trường.

---

## 📊 3. TÓM TẮT KẾT QUẢ KIỂM ĐỊNH THỰC NGHIỆM

Quá trình kiểm định sử dụng phương pháp **Hold-Out Out-of-Sample Validation**:
- **Tập Train (2014 – 2020):** 7 năm dữ liệu dùng để huấn luyện và tìm bộ tham số tối ưu thông qua thuật toán Hyperopt / TPE.
- **Tập Test (2021 – 2023):** 3 năm dữ liệu "chưa từng thấy" dùng để kiểm định tính chống chịu thực tế.

### 📌 Bảng Đối Chiếu Hiệu Suất (Tham số Tối ưu: EMA=36, OBV Slope=20):

| Chiến lược | Tập dữ liệu | Tham số Tối ưu | Sharpe Ratio | Tổng Lợi Nhuận (%) | Max Drawdown (%) | Số Lệnh Đã Đóng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EMA Riêng lẻ** | **Train** | EMA=36 | 1.0797 | +211.97% | -28.69% | 70 |
| **EMA Riêng lẻ** | **Test** | EMA=36 | -0.8601 | -37.69% | -59.19% | 41 |
| **OBV Riêng lẻ** | **Train** | OBV_Slope=19 | 1.1735 | +277.80% | -30.46% | 51 |
| **OBV Riêng lẻ** | **Test** | OBV_Slope=19 | -0.6135 | -38.73% | -62.50% | 38 |
| **EMA + OBV Kết hợp** | **Train** | EMA=36, OBV=20 | **1.1514** | **+236.56%** | **-23.87%** | **64** |
| **EMA + OBV Kết hợp** | **Test** | EMA=36, OBV=20 | **-0.8298** | **-36.47%** | **-60.14%** | **37** |
| *ACB Mua & Nắm giữ* | *Train* | Buy & Hold | - | +359.95% | - | 1 |
| *ACB Mua & Nắm giữ* | *Test* | Buy & Hold | - | +30.34% | - | 1 |

### 🔍 Nhận Định Học Thuật Về Hiện Tượng Overfitting:
- **Trên tập Train (2014 - 2020):** Cổ phiếu ACB nằm trong xu thế Uptrend vĩ mô dài hạn. Chiến lược kết hợp phát huy ưu điểm vượt bậc: đạt Sharpe **1.1514**, lợi nhuận **+236.56%** và khống chế Max Drawdown tốt nhất (**-23.87%**, thấp hơn đáng kể so với EMA riêng -28.69% và OBV riêng -30.46%).
- **Trên tập Test (2021 - 2023):** Thị trường chuyển dịch sang trạng thái biến động cao, sụt giảm mạnh và đi ngang (Sideway) kéo dài trong năm 2022 - 2023. Tín hiệu EMA bị trễ nhịp và OBV bị nhiễu do thanh khoản thất thường, dẫn đến việc bị dính Stop-loss liên tiếp (Lợi nhuận -36.47%).
- **Bài học định lượng:** Hiện tượng suy giảm hiệu suất rõ rệt trên tập Test chứng minh tầm quan trọng của việc kiểm định ngoài mẫu (Out-of-sample) và sự cần thiết của các bộ lọc điều kiện vĩ mô (như xu hướng chỉ số VN-Index).

---

## 📂 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
├── ACB.csv                                      # Tệp dữ liệu lịch sử giá & khối lượng ACB (2014 - 2023)
├── ACB_Strategy_EMA_OBV_Nhom3_Adjusted.ipynb    # Jupyter Notebook gốc đã hiệu chỉnh logic
├── app.py                                       # Ứng dụng Web Dashboard phân tích định lượng (Streamlit)
├── requirements.txt                             # Danh sách thư viện phụ thuộc (Chuẩn Streamlit Cloud)
├── requirement.txt                              # Tệp phụ trợ tương thích
└── README.md                                    # Tài liệu hướng dẫn & Báo cáo kết quả
```

---

## 💻 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY LOCAL

### Yêu Cầu Hệ Thống:
- Đã cài đặt **Python 3.9, 3.10 hoặc 3.11** trên máy tính.

### Các Bước Thực Hiện:
1. **Mở Terminal / PowerShell** tại thư mục dự án:
   ```bash
   # Di chuyển vào thư mục dự án
   cd "tạo app trading chiến lược"
   ```

2. **Khởi tạo môi trường ảo (Khuyên dùng):**
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   .\venv\Scripts\activate
   # Hoặc trên macOS/Linux:
   source venv/bin/activate
   ```

3. **Cài đặt các gói phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Web Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🌐 6. HƯỚNG DẪN ĐẨY LÊN GITHUB & DEPLOY STREAMLIT CLOUD (MIỄN PHÍ)

### BƯỚC 1: Đẩy Dự Án Lên GitHub
1. Đăng nhập vào [GitHub](https://github.com/) và tạo một Repository mới (ví dụ: `acb-ema-obv-trading-strategy`, đặt chế độ **Public**).
2. Mở Terminal tại thư mục dự án và chạy các lệnh:
   ```bash
   git init
   git add .
   git commit -m "feat: Initial commit ACB Quant Trading App with EMA + OBV"
   git branch -M main
   git remote add origin https://github.com/<TEN-TAI-KHOAN-GITHUB>/<TEN-REPO>.git
   git push -u origin main
   ```

### BƯỚC 2: Triển Khai Lên Streamlit Community Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"Create app"** (hoặc **"New app"**).
3. Điền thông tin cấu hình triển khai:
   - **Repository:** Chọn repository vừa đẩy lên (ví dụ: `<TEN-TAI-KHOAN>/acb-ema-obv-trading-strategy`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (tùy chọn):** Có thể đặt tên miền tùy chỉnh (ví dụ: `acb-quant-strategy.streamlit.app`).
4. Nhấn **"Deploy!"**.
5. Streamlit Cloud sẽ tự động đọc tệp `requirements.txt`, cài đặt các gói cần thiết và khởi tạo ứng dụng trực tuyến chỉ sau 1-2 phút!

---

## 🖥️ 7. CÁC TÍNH NĂNG VƯỢT TRỘI TRÊN GIAO DIỆN WEB APP

Ứng dụng được thiết kế theo phong cách **Trading Terminal** chuyên nghiệp với 7 Tab chức năng chuyên biệt:

1. **📊 Phân Tích Kỹ Thuật (EDA):** Đồ thị nến Nhật (Candlestick) tương tác Plotly, đường EMA, khối lượng giao dịch và chỉ báo độ dốc OBV Slope theo màu trực quan.
2. **⚙️ Hạch Toán Vị Thế (Position Log):** Minh bạch toàn bộ sổ lệnh (Trade Log), giá vào, giá ra, lợi nhuận từng giao dịch, thời gian nắm giữ, và nút bấm **Tải xuống tệp CSV**.
3. **🏆 So Sánh Hiệu Suất (Train vs Test):** Bảng ma trận đối chiếu 3 chiến lược và Biểu đồ cột 3 chiều (Sharpe Ratio, Lợi nhuận %, Max Drawdown %) chuẩn hóa theo Cell 23 trong Notebook.
4. **📉 Đường Cong Vốn & Drawdown:** Đồ thị tăng trưởng vốn (Equity Curve) chuẩn hóa mốc 100 và biểu đồ sụt giảm tài khoản (Underwater Drawdown Chart).
5. **🔬 Quét Tối Ưu Hóa (Optimization):** Tính năng Grid Search quét không gian tham số EMA và OBV ngay trên giao diện Web kèm bản đồ nhiệt (Heatmap).
6. **🚨 Tín Hiệu Thực Chiến & Quản Trị Rủi Ro:** Động cơ đánh giá phiên giao dịch gần nhất, phát tín hiệu hành động cụ thể (**MUA**, **BÁN**, **NẮM GIỮ**, hoặc **ĐỨNG NGOÀI**) và hiển thị mức giá cắt lỗ mục tiêu.
7. **📚 Báo Cáo Học Thuật & Chẩn Đoán:** Tài liệu phân tích nguyên nhân học thuật, giải thích hiện tượng chuyển dịch chế độ thị trường và các kiến nghị phát triển mô hình.

---

## 👥 THÀNH VIÊN NHÓM THỰC HIỆN
- **Đề tài:** Ứng dụng Phương pháp Định lượng trong Giao dịch Cổ phiếu ACB
- **Học phần:** Phân Tích Kỹ Thuật & Giao Dịch Thuật Toán
- **Năm hoàn thành:** 2026

*Dự án phục vụ mục đích nghiên cứu học thuật và kiểm định mô hình định lượng. Mọi quyết định giao dịch trên thị trường thực tế cần có thêm các biện pháp quản trị rủi ro chuyên sâu.*
