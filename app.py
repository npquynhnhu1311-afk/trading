"""
================================================================================
HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC ĐỊNH LƯỢNG: EMA KẾT HỢP OBV (CỔ PHIẾU ACB)
================================================================================
Tác giả: Nhóm Nghiên cứu & Định lượng Tài chính
Mục đích:
- Kiểm định hiệu quả chiến lược kết hợp đường trung bình động hàm mũ (EMA)
  và khối lượng cân bằng (On-Balance Volume - OBV) trên cổ phiếu ACB.
- Hạch toán đúng vị thế giao dịch (Position Accounting), loại trừ Look-ahead bias,
  áp dụng chi phí giao dịch, trượt giá và quản trị rủi ro Stop-Loss.
- So sánh chi tiết hiệu suất giữa tập huấn luyện (Train: 2014 - 2020) và
  tập kiểm định độc lập (Test: 2021 - 2023).
================================================================================
"""

import os
import io
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Nhập thư viện chỉ báo kỹ thuật ta (có dự phòng hàm tự xây dựng nếu thiếu)
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# Nhập vectorbt (có dự phòng engine thuần numpy/pandas nếu thiếu)
try:
    import vectorbt as vbt
    HAS_VBT = True
except Exception:
    HAS_VBT = False

# -----------------------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ACB Trading Strategy | EMA + OBV Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến giao diện CSS phong cách Fintech Terminal
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 2px;
    }
    .badge-win {
        color: #059669;
        font-weight: 600;
    }
    .badge-loss {
        color: #DC2626;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CÁC HÀM TÍNH TOÁN CHỈ BÁO KỸ THUẬT (TECHNICAL INDICATORS)
# -----------------------------------------------------------------------------
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Tính đường Trung bình Động Hàm mũ (EMA)."""
    if HAS_TA:
        try:
            return ta.trend.ema_indicator(close=series, window=int(period))
        except Exception:
            pass
    return series.ewm(span=int(period), adjust=False).mean()

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Tính chỉ số Khối lượng Cân bằng (On-Balance Volume - OBV)."""
    if HAS_TA:
        try:
            return ta.volume.on_balance_volume(close=close, volume=volume)
        except Exception:
            pass
    # Thuật toán thuần pandas
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv_diff = pd.Series(direction * volume.values, index=close.index)
    obv_diff.iloc[0] = 0
    return obv_diff.cumsum()

def calculate_obv_slope(obv: pd.Series, period: int) -> pd.Series:
    """Tính độ dốc của OBV qua N phiên: OBV(t) - OBV(t - period)."""
    return obv.diff(int(period))

# -----------------------------------------------------------------------------
# TẠO TÍN HIỆU GIAO DỊCH (SIGNAL GENERATION) - CÓ DỊCH PHIÊN SHIFT(1)
# -----------------------------------------------------------------------------
def generate_strategy_signals(df: pd.DataFrame, ema_period: int, obv_slope_period: int, price_col='Close'):
    """
    Sinh tín hiệu Mua/Bán cho cả 3 chiến lược:
    1. EMA Riêng lẻ: Mua khi Close > EMA, Bán khi Close < EMA
    2. OBV Riêng lẻ: Mua khi OBV Slope > 0, Bán khi OBV Slope < 0
    3. EMA + OBV Kết hợp: Mua khi Close > EMA VÀ OBV Slope > 0, Bán khi Close < EMA

    QUAN TRỌNG: Dùng .shift(1) để chống Look-Ahead Bias. Tín hiệu phiên trước
    sẽ được thực thi tại phiên hiện tại.
    """
    close = df[price_col]
    volume = df['Volume']

    ema = calculate_ema(close, ema_period)
    obv = calculate_obv(close, volume)
    obv_slope = calculate_obv_slope(obv, obv_slope_period)

    # 1. EMA Riêng
    raw_ent_ema = (close > ema)
    raw_ext_ema = (close < ema)
    ent_ema = raw_ent_ema.shift(1, fill_value=False)
    ext_ema = raw_ext_ema.shift(1, fill_value=False)

    # 2. OBV Riêng
    raw_ent_obv = (obv_slope > 0)
    raw_ext_obv = (obv_slope < 0)
    ent_obv = raw_ent_obv.shift(1, fill_value=False)
    ext_obv = raw_ext_obv.shift(1, fill_value=False)

    # 3. EMA + OBV Kết hợp
    raw_ent_comb = (close > ema) & (obv_slope > 0)
    raw_ext_comb = (close < ema)
    ent_comb = raw_ent_comb.shift(1, fill_value=False)
    ext_comb = raw_ext_comb.shift(1, fill_value=False)

    indicators = {
        'ema': ema,
        'obv': obv,
        'obv_slope': obv_slope
    }

    signals = {
        'EMA': (ent_ema, ext_ema),
        'OBV': (ent_obv, ext_obv),
        'Combined': (ent_comb, ext_comb)
    }

    return indicators, signals

# -----------------------------------------------------------------------------
# QUẢN LÝ VỊ THẾ & HẠCH TOÁN (POSITION ACCOUNTING)
# -----------------------------------------------------------------------------
def compute_position_accounting(df: pd.DataFrame, entries: pd.Series, exits: pd.Series):
    """
    Quản lý vị thế:
    - 0: Đang cầm Tiền mặt (Cash)
    - 1: Đang nắm giữ Cổ phiếu (Long)
    - Chỉ sinh lệnh MUA khi chuyển từ 0 sang 1
    - Chỉ sinh lệnh BÁN khi chuyển từ 1 sang 0
    """
    n = len(df)
    position = pd.Series(0, index=df.index, dtype=int)
    buy_orders = pd.Series(False, index=df.index, dtype=bool)
    sell_orders = pd.Series(False, index=df.index, dtype=bool)

    current_pos = 0
    for i in range(n):
        if current_pos == 0 and entries.iloc[i]:
            current_pos = 1
            buy_orders.iloc[i] = True
        elif current_pos == 1 and exits.iloc[i]:
            current_pos = 0
            sell_orders.iloc[i] = True
        position.iloc[i] = current_pos

    return position, buy_orders, sell_orders

# -----------------------------------------------------------------------------
# ENGINE BACKTEST TOÀN DIỆN (ĐỘC LẬP & TƯƠNG THÍCH VECTORBT)
# -----------------------------------------------------------------------------
def run_backtest_simulation(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    initial_cash: float = 100.0,
    fee_rate: float = 0.002,      # 0.2%
    slippage: float = 0.001,      # 0.1%
    sl_stop: float = 0.07,        # Stop-loss 7%
    price_col: str = 'Close'
):
    """
    Mô phỏng Backtest chi tiết, hỗ trợ hạch toán vị thế chuẩn xác,
    có tính chi phí giao dịch, trượt giá, và cơ chế cắt lỗ Stop-loss.
    Nếu có VectorBT, đối chiếu kiểm chứng tương thích; luôn trả về
    kết quả nhật ký giao dịch và chuỗi giá trị danh mục đầy đủ.
    """
    close_prices = df[price_col].values
    low_prices = df['Low'].values if 'Low' in df.columns else close_prices
    dates = df.index
    n = len(df)

    # Nếu có vectorbt, ta có thể chạy đồng thời để lấy chỉ số chuẩn xác tuyệt đối
    vbt_pf = None
    if HAS_VBT:
        try:
            vbt_pf = vbt.Portfolio.from_signals(
                close=df[price_col],
                entries=entries.to_numpy(dtype=bool),
                exits=exits.to_numpy(dtype=bool),
                direction='longonly',
                accumulate=False,
                fees=fee_rate,
                slippage=slippage,
                sl_stop=sl_stop,
                sl_trail=False,
                init_cash=initial_cash,
                freq='D'
            )
        except Exception:
            vbt_pf = None

    # Bộ mô phỏng nội tại để trích xuất Trade Log và Equity Curve
    cash = initial_cash
    shares = 0.0
    equity = np.zeros(n)
    in_pos = False
    entry_price = 0.0
    entry_date = None

    trades = []
    positions = np.zeros(n, dtype=int)

    for i in range(n):
        curr_price = close_prices[i]
        curr_low = low_prices[i]
        curr_date = dates[i]

        # Kiểm tra điều kiện Stop-loss nếu đang giữ lệnh
        hit_stop_loss = False
        if in_pos:
            # Nếu giá chạm ngưỡng cắt lỗ 7%
            loss_ratio = (curr_low - entry_price) / entry_price
            if loss_ratio <= -sl_stop:
                hit_stop_loss = True

        # Quyết định hành động phiên i
        if not in_pos:
            # Điều kiện vào lệnh
            if entries.iloc[i]:
                # Mua cổ phiếu với trượt giá và phí
                exec_price = curr_price * (1.0 + slippage)
                # Toàn bộ tiền mua cổ phiếu trừ phí
                buy_cost_unit = exec_price * (1.0 + fee_rate)
                shares = cash / buy_cost_unit
                cost_total = shares * buy_cost_unit
                cash -= cost_total

                in_pos = True
                entry_price = exec_price
                entry_date = curr_date
        else:
            # Đang nắm giữ, kiểm tra xem có lệnh Bán theo tín hiệu hoặc Stop Loss không
            signal_exit = exits.iloc[i]
            if signal_exit or hit_stop_loss:
                # Giá bán
                if hit_stop_loss:
                    raw_exit_price = entry_price * (1.0 - sl_stop)
                    exit_reason = "Stop Loss (-7%)"
                else:
                    raw_exit_price = curr_price
                    exit_reason = "Signal Exit (EMA)"

                exec_price = raw_exit_price * (1.0 - slippage)
                revenue = shares * exec_price * (1.0 - fee_rate)
                pnl_cash = revenue - (shares * entry_price * (1.0 + fee_rate))
                pnl_pct = (exec_price / entry_price - 1.0) * 100.0

                cash += revenue

                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': curr_date,
                    'Entry Price': entry_price,
                    'Exit Price': exec_price,
                    'Shares': shares,
                    'PnL (Cash)': pnl_cash,
                    'PnL (%)': pnl_pct,
                    'Holding Days': (curr_date - entry_date).days,
                    'Exit Reason': exit_reason
                })

                shares = 0.0
                in_pos = False
                entry_price = 0.0
                entry_date = None

        positions[i] = 1 if in_pos else 0
        current_equity = cash + (shares * curr_price if in_pos else 0.0)
        equity[i] = current_equity

    # Chuyển thành Series Equity
    equity_series = pd.Series(equity, index=df.index)

    # Tính toán các chỉ số thống kê
    trades_df = pd.DataFrame(trades)
    closed_trades_count = len(trades_df)

    start_val = initial_cash
    end_val = equity_series.iloc[-1]
    total_return_pct = (end_val / start_val - 1.0) * 100.0

    # Benchmark Buy & Hold
    bm_start = close_prices[0]
    bm_end = close_prices[-1]
    bm_return_pct = (bm_end / bm_start - 1.0) * 100.0

    # Lợi nhuận hàng ngày
    daily_returns = equity_series.pct_change().dropna()
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    if std_ret > 0:
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    if downside_std > 0:
        sortino_ratio = (mean_ret / downside_std) * np.sqrt(252)
    else:
        sortino_ratio = np.nan

    # Max Drawdown
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown_pct = drawdown.min() * 100.0

    # Win Rate & Profit Factor
    if closed_trades_count > 0:
        winning_trades = trades_df[trades_df['PnL (%)'] > 0]
        losing_trades = trades_df[trades_df['PnL (%)'] <= 0]
        win_rate_pct = (len(winning_trades) / closed_trades_count) * 100.0

        gross_profit = winning_trades['PnL (Cash)'].sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades['PnL (Cash)'].sum()) if len(losing_trades) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0

    # Tỷ lệ nắm giữ vị thế (Exposure)
    exposure_pct = (np.mean(positions)) * 100.0

    # Nếu có vectorbt và hoạt động tốt, ưu tiên đồng bộ Sharpe / Drawdown cho khớp 100% notebook
    if vbt_pf is not None:
        try:
            vbt_stats = vbt_pf.stats()
            vbt_sharpe = float(vbt_pf.sharpe_ratio())
            if np.isfinite(vbt_sharpe):
                sharpe_ratio = vbt_sharpe
            vbt_trades = int(vbt_pf.trades.closed.count())
            if vbt_trades > 0:
                closed_trades_count = vbt_trades
            vbt_ret = float(vbt_pf.total_return()) * 100.0
            if np.isfinite(vbt_ret):
                total_return_pct = vbt_ret
            vbt_mdd = float(vbt_pf.max_drawdown()) * 100.0
            if np.isfinite(vbt_mdd):
                max_drawdown_pct = -abs(vbt_mdd)
            # Equity curve từ vbt
            equity_series = vbt_pf.value()
        except Exception:
            pass

    metrics = {
        'Start Value': start_val,
        'End Value': end_val,
        'Total Return (%)': total_return_pct,
        'Benchmark Return (%)': bm_return_pct,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_drawdown_pct,
        'Win Rate (%)': win_rate_pct,
        'Profit Factor': profit_factor,
        'Total Closed Trades': closed_trades_count,
        'Position Exposure (%)': exposure_pct
    }

    return {
        'metrics': metrics,
        'equity_series': equity_series,
        'drawdown_series': drawdown * 100.0,
        'trades_df': trades_df,
        'positions': pd.Series(positions, index=df.index),
        'vbt_pf': vbt_pf
    }

# -----------------------------------------------------------------------------
# HÀM TẢI DỮ LIỆU CỔ PHIẾU
# -----------------------------------------------------------------------------
@st.cache_data
def load_stock_data(file_path_or_buffer):
    """Đọc dữ liệu CSV, chuẩn hóa ngày tháng và sắp xếp theo thời gian."""
    if isinstance(file_path_or_buffer, str):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_csv(file_path_or_buffer)

    # Tìm cột ngày
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'ngay', 'time', 'datetime']:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    # Chuẩn hóa tên cột
    rename_dict = {}
    for col in df.columns:
        c_low = col.lower()
        if c_low in ['close', 'dongcua', 'dong_cua', 'gia_dong']:
            rename_dict[col] = 'Close'
        elif c_low in ['open', 'mocua', 'mo_cua']:
            rename_dict[col] = 'Open'
        elif c_low in ['high', 'caonhat', 'cao_nhat']:
            rename_dict[col] = 'High'
        elif c_low in ['low', 'thapnhat', 'thap_nhat']:
            rename_dict[col] = 'Low'
        elif c_low in ['volume', 'vol', 'khoiluong', 'khoi_luong']:
            rename_dict[col] = 'Volume'

    df.rename(columns=rename_dict, inplace=True)

    # Đảm bảo có các cột cơ bản
    if 'Close' not in df.columns:
        df['Close'] = df.iloc[:, 0]
    if 'Volume' not in df.columns:
        df['Volume'] = 100000.0
    if 'Open' not in df.columns:
        df['Open'] = df['Close']
    if 'High' not in df.columns:
        df['High'] = df[['Open', 'Close']].max(axis=1)
    if 'Low' not in df.columns:
        df['Low'] = df[['Open', 'Close']].min(axis=1)

    return df

# -----------------------------------------------------------------------------
# GIAO DIỆN CHÍNH STREAMLIT
# -----------------------------------------------------------------------------
def main():
    # Header
    st.markdown('<div class="main-title">📈 Kiểm Định Chiến Lược Giao Dịch Định Lượng: EMA + OBV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Nền tảng Backtesting & Đánh giá Rủi ro Vị thế cho Cổ phiếu Ngân hàng Á Châu (ACB) | <b>Đồ án Nhóm 3 (Hiệu chỉnh)</b></div>', unsafe_allow_html=True)

    # ------------------ SIDEBAR ------------------
    st.sidebar.header("🛠️ Cấu Hình Hệ Thống")

    # 1. Nguồn Dữ liệu
    st.sidebar.subheader("1. Dữ Liệu Đầu Vào")
    data_source = st.sidebar.radio(
        "Nguồn tệp dữ liệu:",
        ["Dữ liệu ACB.csv mặc định", "Tải lên tệp CSV tùy chọn"],
        index=0
    )

    df_raw = None
    if data_source == "Dữ liệu ACB.csv mặc định":
        default_file = "ACB.csv"
        if os.path.exists(default_file):
            df_raw = load_stock_data(default_file)
            st.sidebar.success(f"✅ Đã tải `{default_file}` ({len(df_raw):,} phiên)")
        else:
            st.sidebar.error("❌ Không tìm thấy `ACB.csv` trong thư mục gốc. Vui lòng tải file lên.")
    else:
        uploaded_file = st.sidebar.file_uploader("Chọn tệp CSV (Cột Date, Close, Volume,...):", type=["csv"])
        if uploaded_file is not None:
            df_raw = load_stock_data(uploaded_file)
            st.sidebar.success(f"✅ Đã nạp dữ liệu ({len(df_raw):,} phiên)")

    if df_raw is None:
        st.warning("⚠️ Vui lòng nạp tệp dữ liệu để bắt đầu phân tích.")
        return

    # 2. Phân tách Tập Train & Test
    st.sidebar.subheader("2. Phân Chia Mẫu (Hold-out)")
    min_date = df_raw.index.min().date()
    max_date = df_raw.index.max().date()

    split_mode = st.sidebar.selectbox(
        "Giai đoạn phân chia:",
        [
            "Chuẩn Đồ Án (Train: 2014-2020 | Test: 2021-2023)",
            "Toàn bộ giai đoạn dữ liệu",
            "Tùy chỉnh khoảng thời gian"
        ],
        index=0
    )

    if split_mode == "Chuẩn Đồ Án (Train: 2014-2020 | Test: 2021-2023)":
        train_start = pd.to_datetime('2014-01-01').date()
        train_end = pd.to_datetime('2020-12-31').date()
        test_start = pd.to_datetime('2021-01-01').date()
        test_end = pd.to_datetime('2023-12-31').date()
    elif split_mode == "Toàn bộ giai đoạn dữ liệu":
        train_start = min_date
        train_end = max_date
        test_start = min_date
        test_end = max_date
    else:
        col_s1, col_s2 = st.sidebar.columns(2)
        with col_s1:
            train_start = st.date_input("Train Bắt đầu", min_date)
            test_start = st.date_input("Test Bắt đầu", pd.to_datetime('2021-01-01').date())
        with col_s2:
            train_end = st.date_input("Train Kết thúc", pd.to_datetime('2020-12-31').date())
            test_end = st.date_input("Test Kết thúc", max_date)

    # Cắt lọc tập dữ liệu
    train_df = df_raw.loc[str(train_start):str(train_end)].copy()
    test_df = df_raw.loc[str(test_start):str(test_end)].copy()

    # 3. Tham Số Chiến Lược
    st.sidebar.subheader("3. Bộ Tham Số Giao Dịch")
    param_mode = st.sidebar.selectbox(
        "Lựa chọn bộ tham số:",
        [
            "Tham số Tối ưu từ Train (EMA=36, OBV Slope=20) [Khuyên dùng]",
            "Tham số Mặc định (EMA=20, OBV Slope=3)",
            "Tự điều chỉnh tham số thủ công"
        ],
        index=0
    )

    if param_mode == "Tham số Tối ưu từ Train (EMA=36, OBV Slope=20) [Khuyên dùng]":
        ema_p = 36
        obv_p = 20
        st.sidebar.info("💡 Bộ tham số tìm được bằng Hyperopt/TPE tối đa hóa Sharpe trên tập Train 2014-2020.")
    elif param_mode == "Tham số Mặc định (EMA=20, OBV Slope=3)":
        ema_p = 20
        obv_p = 3
        st.sidebar.info("💡 Bộ tham số kỹ thuật phổ biến theo phân tích cơ bản.")
    else:
        ema_p = st.sidebar.slider("Chu kỳ đường EMA (Phiên):", min_value=5, max_value=80, value=36, step=1)
        obv_p = st.sidebar.slider("Độ dốc OBV Slope (Số phiên):", min_value=1, max_value=30, value=20, step=1)

    # 4. Quản Trị Vốn & Chi Phí
    st.sidebar.subheader("4. Quản Trị Vốn & Chi Phí")
    init_capital = st.sidebar.number_input("Vốn đầu tư ban đầu (VND):", min_value=10.0, value=100_000_000.0, step=10_000_000.0, format="%.0f")
    fee_rate = st.sidebar.number_input("Phí giao dịch mỗi chiều (%):", min_value=0.0, max_value=2.0, value=0.2, step=0.05) / 100.0
    slippage = st.sidebar.number_input("Tỷ lệ trượt giá (%):", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100.0
    stop_loss = st.sidebar.number_input("Mức cắt lỗ Stop Loss (%):", min_value=1.0, max_value=25.0, value=7.0, step=0.5) / 100.0

    # ------------------ TÍNH TOÁN DỮ LIỆU & BACKTEST ------------------
    # Tính toán chỉ báo và tín hiệu trên Train
    train_indicators, train_signals = generate_strategy_signals(train_df, ema_p, obv_p)
    # Tính toán chỉ báo và tín hiệu trên Test
    test_indicators, test_signals = generate_strategy_signals(test_df, ema_p, obv_p)

    # Chạy mô phỏng cho 3 chiến lược trên Train
    train_res_ema = run_backtest_simulation(train_df, train_signals['EMA'][0], train_signals['EMA'][1], init_capital, fee_rate, slippage, stop_loss)
    train_res_obv = run_backtest_simulation(train_df, train_signals['OBV'][0], train_signals['OBV'][1], init_capital, fee_rate, slippage, stop_loss)
    train_res_comb = run_backtest_simulation(train_df, train_signals['Combined'][0], train_signals['Combined'][1], init_capital, fee_rate, slippage, stop_loss)

    # Chạy mô phỏng cho 3 chiến lược trên Test
    test_res_ema = run_backtest_simulation(test_df, test_signals['EMA'][0], test_signals['EMA'][1], init_capital, fee_rate, slippage, stop_loss)
    test_res_obv = run_backtest_simulation(test_df, test_signals['OBV'][0], test_signals['OBV'][1], init_capital, fee_rate, slippage, stop_loss)
    test_res_comb = run_backtest_simulation(test_df, test_signals['Combined'][0], test_signals['Combined'][1], init_capital, fee_rate, slippage, stop_loss)

    # KPI Thẻ Tóm tắt Chiến lược Kết hợp (Combined)
    st.markdown("### 📌 Tóm Tắt Nhanh Chiến Lược Kết Hợp (EMA + OBV)")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        ret_train = train_res_comb['metrics']['Total Return (%)']
        ret_test = test_res_comb['metrics']['Total Return (%)']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Lợi Nhuận Train (2014-20)</div>
            <div class="metric-value {'badge-win' if ret_train > 0 else 'badge-loss'}">{ret_train:+.1f}%</div>
            <div class="metric-sub">Test (2021-23): <b>{ret_test:+.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        sh_train = train_res_comb['metrics']['Sharpe Ratio']
        sh_test = test_res_comb['metrics']['Sharpe Ratio']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sharpe Ratio</div>
            <div class="metric-value">{sh_train:.2f}</div>
            <div class="metric-sub">Tập Test: <b>{sh_test:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        mdd_train = train_res_comb['metrics']['Max Drawdown (%)']
        mdd_test = test_res_comb['metrics']['Max Drawdown (%)']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Max Drawdown</div>
            <div class="metric-value badge-loss">{mdd_train:.1f}%</div>
            <div class="metric-sub">Tập Test: <b class="badge-loss">{mdd_test:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        tr_train = train_res_comb['metrics']['Total Closed Trades']
        tr_test = test_res_comb['metrics']['Total Closed Trades']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Số Lệnh Đã Đóng</div>
            <div class="metric-value">{tr_train} lệnh</div>
            <div class="metric-sub">Tập Test: <b>{tr_test} lệnh</b></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        # Tín hiệu phiên gần nhất
        last_date = df_raw.index[-1].strftime('%d/%m/%Y')
        last_close = df_raw['Close'].iloc[-1]
        last_ema = calculate_ema(df_raw['Close'], ema_p).iloc[-1]
        last_obv_slope = calculate_obv_slope(calculate_obv(df_raw['Close'], df_raw['Volume']), obv_p).iloc[-1]
        is_above_ema = last_close > last_ema
        is_obv_pos = last_obv_slope > 0

        # Lấy trạng thái vị thế phiên cuối từ test_res_comb
        last_pos = test_res_comb['positions'].iloc[-1]
        if last_pos == 1:
            rec_status = "NẮM GIỮ (HOLD)"
            rec_color = "#2563EB"
        elif is_above_ema and is_obv_pos:
            rec_status = "MUA MỚI (BUY)"
            rec_color = "#059669"
        else:
            rec_status = "ĐỨNG NGOÀI (CASH)"
            rec_color = "#64748B"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Khuyến Nghị Phiên {last_date}</div>
            <div class="metric-value" style="color: {rec_color}; font-size: 1.3rem;">{rec_status}</div>
            <div class="metric-sub">Giá: {last_close:,.0f} | EMA: {last_ema:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ------------------ CÁC TAB CHỨC NĂNG ------------------
    tabs = st.tabs([
        "📊 Phân Tích Kỹ Thuật (EDA)",
        "⚙️ Hạch Toán Vị Thế (Position Log)",
        "🏆 So Sánh Hiệu Suất (Train vs Test)",
        "📉 Đường Cong Vốn & Drawdown",
        "🔬 Quét Tối Ưu Hóa (Optimization)",
        "🚨 Tín Hiệu Thực Chiến & Quản Trị",
        "📚 Báo Cáo Học Thuật & Chẩn Đoán"
    ])

    # ------------------ TAB 1: PHÂN TÍCH KỸ THUẬT ------------------
    with tabs[0]:
        st.subheader("📊 Biểu Đồ Giá, Chỉ Báo EMA & OBV Slope")
        plot_period = st.radio(
            "Chọn tập dữ liệu hiển thị:",
            ["Tập Test (2021 - 2023) [Khuyên dùng để quan sát]", "Tập Train (2014 - 2020)", "Toàn bộ lịch sử (2014 - 2023)"],
            horizontal=True
        )

        if "Test" in plot_period:
            chart_df = test_df.copy()
            c_ema = test_indicators['ema']
            c_obv = test_indicators['obv']
            c_obv_slope = test_indicators['obv_slope']
            c_buys = test_signals['Combined'][0]
            c_sells = test_signals['Combined'][1]
        elif "Train" in plot_period:
            chart_df = train_df.copy()
            c_ema = train_indicators['ema']
            c_obv = train_indicators['obv']
            c_obv_slope = train_indicators['obv_slope']
            c_buys = train_signals['Combined'][0]
            c_sells = train_signals['Combined'][1]
        else:
            chart_df = df_raw.copy()
            full_ind, full_sig = generate_strategy_signals(chart_df, ema_p, obv_p)
            c_ema = full_ind['ema']
            c_obv = full_ind['obv']
            c_obv_slope = full_ind['obv_slope']
            c_buys = full_sig['Combined'][0]
            c_sells = full_sig['Combined'][1]

        # Tạo Subplots Plotly
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                f"Giá Cổ Phiếu ACB & Đường EMA ({ema_p}) kèm Điểm Vào/Ra Lệnh Kết Hợp",
                "Khối Lượng Giao Dịch (Volume)",
                f"Độ Dốc Chỉ Báo OBV ({obv_p} Phiên) - [Xanh: > 0 (Tăng áp lực mua) | Đỏ: < 0 (Bán)]"
            ),
            row_heights=[0.55, 0.20, 0.25]
        )

        # 1. Candlestick & EMA
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name="Nến Giá ACB",
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=c_ema,
            line=dict(color="#3B82F6", width=2),
            name=f"EMA ({ema_p})"
        ), row=1, col=1)

        # Đánh dấu tín hiệu Mua/Bán
        buy_points = chart_df[c_buys]
        sell_points = chart_df[c_sells]

        if len(buy_points) > 0:
            fig.add_trace(go.Scatter(
                x=buy_points.index,
                y=buy_points['Low'] * 0.98,
                mode='markers',
                marker=dict(symbol='triangle-up', size=11, color='#059669', line=dict(width=1, color='white')),
                name="Tín hiệu Mua (Buy)"
            ), row=1, col=1)

        if len(sell_points) > 0:
            fig.add_trace(go.Scatter(
                x=sell_points.index,
                y=sell_points['High'] * 1.02,
                mode='markers',
                marker=dict(symbol='triangle-down', size=11, color='#DC2626', line=dict(width=1, color='white')),
                name="Tín hiệu Bán (Sell)"
            ), row=1, col=1)

        # 2. Khối lượng
        vol_colors = np.where(chart_df['Close'] >= chart_df['Open'], '#22C55E', '#EF4444')
        fig.add_trace(go.Bar(
            x=chart_df.index,
            y=chart_df['Volume'],
            marker_color=vol_colors,
            name="Khối lượng (Volume)"
        ), row=2, col=1)

        # 3. OBV Slope
        slope_colors = np.where(c_obv_slope >= 0, '#059669', '#DC2626')
        fig.add_trace(go.Bar(
            x=chart_df.index,
            y=c_obv_slope,
            marker_color=slope_colors,
            name=f"OBV Slope ({obv_p})"
        ), row=3, col=1)

        fig.update_layout(
            height=780,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 Xem Bảng Thống Kê Mô Tả Dữ Liệu"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Tập Train (2014 - 2020):**")
                st.dataframe(train_df[['Open', 'High', 'Low', 'Close', 'Volume']].describe(), use_container_width=True)
            with c2:
                st.write("**Tập Test (2021 - 2023):**")
                st.dataframe(test_df[['Open', 'High', 'Low', 'Close', 'Volume']].describe(), use_container_width=True)

    # ------------------ TAB 2: HẠCH TOÁN VỊ THẾ ------------------
    with tabs[1]:
        st.subheader("⚙️ Hạch Toán Vị Thế Giao Dịch (Position Accounting)")
        st.markdown("""
        > **Điểm Chấn Chỉnh Quan Trọng từ Giảng Viên:**
        > Khi xây dựng thuật toán định lượng, tín hiệu `entries` và `exits` là các điều kiện trạng thái (Conditions).
        > Một tài khoản thực tế **chỉ mua khi chưa có cổ phiếu (chuyển trạng thái từ 0 -> 1)** và **chỉ bán khi đang nắm giữ (1 -> 0)**.
        > Nếu không hạch toán vị thế, các phép đo Sharpe Ratio sẽ bị lệch lạc nghiêm trọng do các tín hiệu lặp lại khi đang ôm lệnh.
        """)

        dataset_choice = st.radio("Chọn tập dữ liệu xem sổ lệnh:", ["Tập Test (2021 - 2023)", "Tập Train (2014 - 2020)"], horizontal=True)
        selected_res = test_res_comb if "Test" in dataset_choice else train_res_comb

        trades_log = selected_res['trades_df']
        pos_series = selected_res['positions']

        c_log1, c_log2, c_log3, c_log4 = st.columns(4)
        c_log1.metric("Tổng số lệnh đã đóng", f"{len(trades_log)} lệnh")
        c_log2.metric("Tỷ lệ thời gian giữ vị thế", f"{selected_res['metrics']['Position Exposure (%)']:.1f}%")
        c_log3.metric("Tỷ lệ thắng (Win Rate)", f"{selected_res['metrics']['Win Rate (%)']:.1f}%")
        c_log4.metric("Profit Factor", f"{selected_res['metrics']['Profit Factor']:.2f}")

        if len(trades_log) > 0:
            st.write("📋 **Nhật ký Chi Tiết Từng Lệnh Giao Dịch (Trade Log):**")
            # Format hiển thị
            display_log = trades_log.copy()
            display_log['Entry Date'] = pd.to_datetime(display_log['Entry Date']).dt.strftime('%d/%m/%Y')
            display_log['Exit Date'] = pd.to_datetime(display_log['Exit Date']).dt.strftime('%d/%m/%Y')
            display_log['Entry Price'] = display_log['Entry Price'].map('{:,.1f}'.format)
            display_log['Exit Price'] = display_log['Exit Price'].map('{:,.1f}'.format)
            display_log['PnL (Cash)'] = display_log['PnL (Cash)'].map('{:+,.0f}'.format)
            display_log['PnL (%)'] = display_log['PnL (%)'].map('{:+.2f}%'.format)

            st.dataframe(display_log, use_container_width=True)

            # Tải về CSV
            csv_buf = io.StringIO()
            trades_log.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Tải xuống Nhật Ký Lệnh (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"ACB_Trades_Log_{dataset_choice.split()[1]}.csv",
                mime="text/csv"
            )
        else:
            st.info("Chiến lược không phát sinh giao dịch nào đóng trong giai đoạn này.")

    # ------------------ TAB 3: SO SÁNH HIỆU SUẤT TOÀN DIỆN ------------------
    with tabs[2]:
        st.subheader("🏆 Ma Trận So Sánh Hiệu Suất: Train vs Test & 3 Chiến Lược")

        # Tạo bảng so sánh chuẩn xác như Cell 23 trong Notebook
        comp_data = [
            {
                'Chiến lược': 'EMA Riêng lẻ',
                'Tập dữ liệu': 'Train (2014-20)',
                'Tham số': f'EMA={ema_p}',
                'Sharpe Ratio': train_res_ema['metrics']['Sharpe Ratio'],
                'Tổng lợi nhuận (%)': train_res_ema['metrics']['Total Return (%)'],
                'Max Drawdown (%)': train_res_ema['metrics']['Max Drawdown (%)'],
                'Win Rate (%)': train_res_ema['metrics']['Win Rate (%)'],
                'Số lệnh đóng': train_res_ema['metrics']['Total Closed Trades']
            },
            {
                'Chiến lược': 'EMA Riêng lẻ',
                'Tập dữ liệu': 'Test (2021-23)',
                'Tham số': f'EMA={ema_p}',
                'Sharpe Ratio': test_res_ema['metrics']['Sharpe Ratio'],
                'Tổng lợi nhuận (%)': test_res_ema['metrics']['Total Return (%)'],
                'Max Drawdown (%)': test_res_ema['metrics']['Max Drawdown (%)'],
                'Win Rate (%)': test_res_ema['metrics']['Win Rate (%)'],
                'Số lệnh đóng': test_res_ema['metrics']['Total Closed Trades']
            },
            {
                'Chiến lược': 'OBV Riêng lẻ',
                'Tập dữ liệu': 'Train (2014-20)',
                'Tham số': f'OBV_Slope={obv_p}',
                'Sharpe Ratio': train_res_obv['metrics']['Sharpe Ratio'],
                'Tổng lợi nhuận (%)': train_res_obv['metrics']['Total Return (%)'],
                'Max Drawdown (%)': train_res_obv['metrics']['Max Drawdown (%)'],
                'Win Rate (%)': train_res_obv['metrics']['Win Rate (%)'],
                'Số lệnh đóng': train_res_obv['metrics']['Total Closed Trades']
            },
            {
                'Chiến lược': 'OBV Riêng lẻ',
                'Tập dữ liệu': 'Test (2021-23)',
                'Tham số': f'OBV_Slope={obv_p}',
                'Sharpe Ratio': test_res_obv['metrics']['Sharpe Ratio'],
                'Tổng lợi nhuận (%)': test_res_obv['metrics']['Total Return (%)'],
                'Max Drawdown (%)': test_res_obv['metrics']['Max Drawdown (%)'],
                'Win Rate (%)': test_res_obv['metrics']['Win Rate (%)'],
                'Số lệnh đóng': test_res_obv['metrics']['Total Closed Trades']
            },
            {
                'Chiến lược': 'EMA + OBV Kết hợp',
                'Tập dữ liệu': 'Train (2014-20)',
                'Tham số': f'EMA={ema_p}, OBV={obv_p}',
                'Sharpe Ratio': train_res_comb['metrics']['Sharpe Ratio'],
                'Tổng lợi nhuận (%)': train_res_comb['metrics']['Total Return (%)'],
                'Max Drawdown (%)': train_res_comb['metrics']['Max Drawdown (%)'],
                'Win Rate (%)': train_res_comb['metrics']['Win Rate (%)'],
                'Số lệnh đóng': train_res_comb['metrics']['Total Closed Trades']
            },
            {
                'Chiến lược': 'EMA + OBV Kết hợp',
                'Tập dữ liệu': 'Test (2021-23)',
                'Tham số': f'EMA={ema_p}, OBV={obv_p}',
                'Sharpe Ratio': test_res_comb['metrics']['Sharpe Ratio'],
                'Tổng lợi nhuận (%)': test_res_comb['metrics']['Total Return (%)'],
                'Max Drawdown (%)': test_res_comb['metrics']['Max Drawdown (%)'],
                'Win Rate (%)': test_res_comb['metrics']['Win Rate (%)'],
                'Số lệnh đóng': test_res_comb['metrics']['Total Closed Trades']
            },
            {
                'Chiến lược': 'Mua & Nắm giữ (ACB Benchmark)',
                'Tập dữ liệu': 'Train (2014-20)',
                'Tham số': 'Buy & Hold',
                'Sharpe Ratio': np.nan,
                'Tổng lợi nhuận (%)': train_res_comb['metrics']['Benchmark Return (%)'],
                'Max Drawdown (%)': np.nan,
                'Win Rate (%)': 100.0,
                'Số lệnh đóng': 1
            },
            {
                'Chiến lược': 'Mua & Nắm giữ (ACB Benchmark)',
                'Tập dữ liệu': 'Test (2021-23)',
                'Tham số': 'Buy & Hold',
                'Sharpe Ratio': np.nan,
                'Tổng lợi nhuận (%)': test_res_comb['metrics']['Benchmark Return (%)'],
                'Max Drawdown (%)': np.nan,
                'Win Rate (%)': 100.0,
                'Số lệnh đóng': 1
            }
        ]
        df_comparison = pd.DataFrame(comp_data)

        # Định dạng hiển thị đẹp mắt
        df_display = df_comparison.copy()
        df_display['Sharpe Ratio'] = df_display['Sharpe Ratio'].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
        df_display['Tổng lợi nhuận (%)'] = df_display['Tổng lợi nhuận (%)'].map(lambda x: f"{x:+.2f}%")
        df_display['Max Drawdown (%)'] = df_display['Max Drawdown (%)'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
        df_display['Win Rate (%)'] = df_display['Win Rate (%)'].map(lambda x: f"{x:.1f}%")

        st.dataframe(df_display, use_container_width=True)

        st.markdown("#### 📊 Trực Quan Hóa So Sánh Train vs Test (Tái Hiện Cell 23 Notebook)")

        # Lọc bỏ benchmark để vẽ đồ thị so sánh 3 chiến lược
        plot_comp_df = df_comparison[df_comparison['Chiến lược'] != 'Mua & Nắm giữ (ACB Benchmark)'].copy()

        # Tạo biểu đồ Bar so sánh Sharpe, Return, Drawdown
        fig_bar = make_subplots(
            rows=3, cols=1,
            subplot_titles=("1. Sharpe Ratio (Thước đo Lợi nhuận điều chỉnh Rủi ro)",
                            "2. Tổng Lợi Nhuận (%)",
                            "3. Mức Sụt Giảm Lớn Nhất - Max Drawdown (%)"),
            vertical_spacing=0.1
        )

        strategies = ['EMA Riêng lẻ', 'OBV Riêng lẻ', 'EMA + OBV Kết hợp']

        # Dữ liệu Train / Test
        train_vals_sharpe = [plot_comp_df[(plot_comp_df['Chiến lược']==s) & (plot_comp_df['Tập dữ liệu']=='Train (2014-20)')]['Sharpe Ratio'].values[0] for s in strategies]
        test_vals_sharpe = [plot_comp_df[(plot_comp_df['Chiến lược']==s) & (plot_comp_df['Tập dữ liệu']=='Test (2021-23)')]['Sharpe Ratio'].values[0] for s in strategies]

        train_vals_ret = [plot_comp_df[(plot_comp_df['Chiến lược']==s) & (plot_comp_df['Tập dữ liệu']=='Train (2014-20)')]['Tổng lợi nhuận (%)'].values[0] for s in strategies]
        test_vals_ret = [plot_comp_df[(plot_comp_df['Chiến lược']==s) & (plot_comp_df['Tập dữ liệu']=='Test (2021-23)')]['Tổng lợi nhuận (%)'].values[0] for s in strategies]

        train_vals_mdd = [plot_comp_df[(plot_comp_df['Chiến lược']==s) & (plot_comp_df['Tập dữ liệu']=='Train (2014-20)')]['Max Drawdown (%)'].values[0] for s in strategies]
        test_vals_mdd = [plot_comp_df[(plot_comp_df['Chiến lược']==s) & (plot_comp_df['Tập dữ liệu']=='Test (2021-23)')]['Max Drawdown (%)'].values[0] for s in strategies]

        # 1. Sharpe
        fig_bar.add_trace(go.Bar(name='Train (2014-20)', x=strategies, y=train_vals_sharpe, marker_color='#2563EB', text=[f"{v:.2f}" for v in train_vals_sharpe], textposition='auto'), row=1, col=1)
        fig_bar.add_trace(go.Bar(name='Test (2021-23)', x=strategies, y=test_vals_sharpe, marker_color='#F59E0B', text=[f"{v:.2f}" for v in test_vals_sharpe], textposition='auto'), row=1, col=1)

        # 2. Return
        fig_bar.add_trace(go.Bar(name='Train (2014-20)', x=strategies, y=train_vals_ret, marker_color='#10B981', text=[f"{v:+.1f}%" for v in train_vals_ret], textposition='auto', showlegend=False), row=2, col=1)
        fig_bar.add_trace(go.Bar(name='Test (2021-23)', x=strategies, y=test_vals_ret, marker_color='#EF4444', text=[f"{v:+.1f}%" for v in test_vals_ret], textposition='auto', showlegend=False), row=2, col=1)

        # 3. Drawdown
        fig_bar.add_trace(go.Bar(name='Train (2014-20)', x=strategies, y=train_vals_mdd, marker_color='#6366F1', text=[f"{v:.1f}%" for v in train_vals_mdd], textposition='auto', showlegend=False), row=3, col=1)
        fig_bar.add_trace(go.Bar(name='Test (2021-23)', x=strategies, y=test_vals_mdd, marker_color='#DC2626', text=[f"{v:.1f}%" for v in test_vals_mdd], textposition='auto', showlegend=False), row=3, col=1)

        fig_bar.update_layout(
            height=780,
            barmode='group',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------ TAB 4: ĐƯỜNG CONG VỐN & DRAWDOWN ------------------
    with tabs[3]:
        st.subheader("📈 Đường Cong Vốn (Equity Curve) & Đồ Thị Sụt Giảm (Underwater Drawdown)")

        eq_choice = st.radio("Chọn giai đoạn hiển thị:", ["Tập Test (2021 - 2023) [Khảo sát sụt giảm]", "Tập Train (2014 - 2020) [Khảo sát tăng trưởng]"], horizontal=True)

        if "Test" in eq_choice:
            cur_df = test_df
            eq_ema = (test_res_ema['equity_series'] / test_res_ema['equity_series'].iloc[0]) * 100.0
            eq_obv = (test_res_obv['equity_series'] / test_res_obv['equity_series'].iloc[0]) * 100.0
            eq_comb = (test_res_comb['equity_series'] / test_res_comb['equity_series'].iloc[0]) * 100.0
            eq_bm = (test_df['Close'] / test_df['Close'].iloc[0]) * 100.0

            dd_comb = test_res_comb['drawdown_series']
            dd_ema = test_res_ema['drawdown_series']
        else:
            cur_df = train_df
            eq_ema = (train_res_ema['equity_series'] / train_res_ema['equity_series'].iloc[0]) * 100.0
            eq_obv = (train_res_obv['equity_series'] / train_res_obv['equity_series'].iloc[0]) * 100.0
            eq_comb = (train_res_comb['equity_series'] / train_res_comb['equity_series'].iloc[0]) * 100.0
            eq_bm = (train_df['Close'] / train_df['Close'].iloc[0]) * 100.0

            dd_comb = train_res_comb['drawdown_series']
            dd_ema = train_res_ema['drawdown_series']

        fig_eq = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Tăng Trưởng Danh Mục (Chuẩn Hóa Mốc 100)", "Mức Sụt Giảm Tài Khoản (Drawdown %)"),
            row_heights=[0.65, 0.35]
        )

        fig_eq.add_trace(go.Scatter(x=cur_df.index, y=eq_comb, mode='lines', line=dict(color='#059669', width=2.5), name='Chiến lược Kết hợp (EMA+OBV)'), row=1, col=1)
        fig_eq.add_trace(go.Scatter(x=cur_df.index, y=eq_ema, mode='lines', line=dict(color='#2563EB', width=1.5, dash='dash'), name='EMA Riêng lẻ'), row=1, col=1)
        fig_eq.add_trace(go.Scatter(x=cur_df.index, y=eq_obv, mode='lines', line=dict(color='#8B5CF6', width=1.5, dash='dot'), name='OBV Riêng lẻ'), row=1, col=1)
        fig_eq.add_trace(go.Scatter(x=cur_df.index, y=eq_bm, mode='lines', line=dict(color='#9CA3AF', width=1.5), name='ACB Mua & Nắm Giữ (Benchmark)'), row=1, col=1)

        # Drawdown
        fig_eq.add_trace(go.Scatter(x=cur_df.index, y=dd_comb, mode='lines', line=dict(color='#DC2626', width=1.5), fill='tozeroy', name='Drawdown (EMA+OBV)'), row=2, col=1)

        fig_eq.update_layout(
            height=650,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ------------------ TAB 5: TỐI ƯU HÓA THAM SỐ ------------------
    with tabs[4]:
        st.subheader("🔬 Quét Tối Ưu Hóa Tham Số Lưới (Grid Search Optimization)")
        st.markdown("""
        Tại đây bạn có thể kiểm tra không gian tham số trên tập Train (2014-2020) để tìm bộ tham số đạt **Sharpe Ratio** cao nhất
        với ràng buộc số giao dịch tối thiểu **`MIN_TRADES >= 5`** nhằm tránh hiện tượng Sharpe ảo.
        """)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            ema_scan_min = st.number_input("EMA Period Tối thiểu", min_value=10, max_value=40, value=20, step=5)
            ema_scan_max = st.number_input("EMA Period Tối đa", min_value=30, max_value=70, value=40, step=5)
            ema_step = st.number_input("Bước nhảy EMA", min_value=1, max_value=10, value=4, step=1)
        with col_opt2:
            obv_scan_min = st.number_input("OBV Slope Tối thiểu", min_value=2, max_value=15, value=5, step=2)
            obv_scan_max = st.number_input("OBV Slope Tối đa", min_value=10, max_value=30, value=20, step=5)
            obv_step = st.number_input("Bước nhảy OBV Slope", min_value=1, max_value=5, value=5, step=1)

        btn_run_opt = st.button("🚀 Bắt Đầu Quét Lưới Tham Số")

        if btn_run_opt:
            ema_range = range(int(ema_scan_min), int(ema_scan_max) + 1, int(ema_step))
            obv_range = range(int(obv_scan_min), int(obv_scan_max) + 1, int(obv_step))

            total_comb = len(ema_range) * len(obv_range)
            st.info(f"Đang tính toán {total_comb} tổ hợp tham số trên tập dữ liệu Train...")

            opt_results = []
            progress_bar = st.progress(0.0)
            step_idx = 0

            for e_p in ema_range:
                for o_p in obv_range:
                    _, s_sig = generate_strategy_signals(train_df, e_p, o_p)
                    res = run_backtest_simulation(train_df, s_sig['Combined'][0], s_sig['Combined'][1], init_capital, fee_rate, slippage, stop_loss)
                    m = res['metrics']
                    opt_results.append({
                        'EMA': e_p,
                        'OBV Slope': o_p,
                        'Sharpe Ratio': m['Sharpe Ratio'],
                        'Total Return (%)': m['Total Return (%)'],
                        'Max Drawdown (%)': m['Max Drawdown (%)'],
                        'Trades': m['Total Closed Trades']
                    })
                    step_idx += 1
                    progress_bar.progress(step_idx / total_comb)

            opt_df = pd.DataFrame(opt_results)
            # Lọc MIN_TRADES >= 5
            valid_opt = opt_df[opt_df['Trades'] >= 5].sort_values(by='Sharpe Ratio', ascending=False)

            st.success("✅ Hoàn tất tối ưu hóa!")

            st.write("🏆 **Top 5 Bộ Tham Số Đạt Sharpe Cao Nhất:**")
            st.dataframe(valid_opt.head(5), use_container_width=True)

            # Vẽ Heatmap
            pivot_sharpe = opt_df.pivot(index='EMA', columns='OBV Slope', values='Sharpe Ratio')
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot_sharpe.values,
                x=[f"OBV={col}" for col in pivot_sharpe.columns],
                y=[f"EMA={idx}" for idx in pivot_sharpe.index],
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio")
            ))
            fig_heat.update_layout(
                title="Bản Đồ Nhiệt Sharpe Ratio theo (EMA, OBV Slope)",
                height=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ------------------ TAB 6: TÍN HIỆU THỰC CHIẾN ------------------
    with tabs[5]:
        st.subheader("🚨 Động Cơ Tín Hiệu Thực Chiến & Quản Lý Rủi Ro")
        st.markdown(f"""
        Phân tích trạng thái tín hiệu tại phiên giao dịch gần nhất của tập dữ liệu: **`{df_raw.index[-1].strftime('%d/%m/%Y')}`**
        """)

        last_idx = df_raw.index[-1]
        c_price = df_raw['Close'].iloc[-1]
        c_ema_val = calculate_ema(df_raw['Close'], ema_p).iloc[-1]
        c_obv_val = calculate_obv(df_raw['Close'], df_raw['Volume'])
        c_slope_val = calculate_obv_slope(c_obv_val, obv_p).iloc[-1]

        cond_ema = c_price > c_ema_val
        cond_obv = c_slope_val > 0

        # Lấy trạng thái từ tập test (phiên cuối)
        curr_in_position = test_res_comb['positions'].iloc[-1] == 1

        rec_c1, rec_c2 = st.columns([1.2, 1])

        with rec_c1:
            st.write("##### 1. Trạng Thái Chỉ Báo Kỹ Thuật Hiện Tại:")
            st.write(f"- **Giá Đóng Cửa ACB:** `{c_price:,.0f} VND`")
            st.write(f"- **Đường EMA ({ema_p}):** `{c_ema_val:,.1f} VND` ➔ {'🟢 Giá nằm TRÊN EMA' if cond_ema else '🔴 Giá nằm DƯỚI EMA'}")
            st.write(f"- **Độ Dốc OBV ({obv_p} phiên):** `{c_slope_val:+,.0f}` ➔ {'🟢 Độ dốc DƯƠNG (Áp lực Mua)' if cond_obv else '🔴 Độ dốc ÂM (Áp lực Bán)'}")
            st.write(f"- **Trạng Thái Vị Thế Tài Khoản:** `{'🟢 Đang NẮM GIỮ Cổ Phiếu ACB' if curr_in_position else '⚪ Đang Giữ TIỀN MẶT (Cash)'}`")

        with rec_c2:
            st.write("##### 2. Quyết Định Hành Động:")
            if curr_in_position:
                if not cond_ema:
                    action_box = "⚠️ **KHUYẾN NGHỊ: BÁN (SELL / EXIT)**\nLý do: Giá đóng cửa đã rơi xuống dưới đường EMA."
                    box_type = "error"
                else:
                    action_box = "💎 **KHUYẾN NGHỊ: TIẾP TỤC NẮM GIỮ (HOLD)**\nLý do: Vị thế Long đang duy trì an toàn trên đường EMA."
                    box_type = "info"
            else:
                if cond_ema and cond_obv:
                    action_box = "🚀 **KHUYẾN NGHỊ: MỞ VỊ THẾ MUA (BUY / ENTRY)**\nLý do: Cả 2 điều kiện kỹ thuật đều thỏa mãn (Close > EMA và OBV Slope > 0)."
                    box_type = "success"
                else:
                    action_box = "⏳ **KHUYẾN NGHỊ: ĐỨNG NGOÀI (STAND ASIDE / CASH)**\nLý do: Chưa đủ điều kiện đồng thuận kỹ thuật để giải ngân."
                    box_type = "warning"

            if box_type == "success":
                st.success(action_box)
            elif box_type == "error":
                st.error(action_box)
            elif box_type == "info":
                st.info(action_box)
            else:
                st.warning(action_box)

            st.write("##### 3. Kế Hoạch Quản Trị Rủi Ro:")
            sl_price = c_price * (1.0 - stop_loss)
            st.write(f"- Ngưỡng Cắt Lỗ Tuyệt Đối: **`{sl_price:,.0f} VND`** (-{stop_loss*100:.1f}%)")
            st.write("- Quy tắc giải ngân: Không mua đuổi khi giá cách quá 5% so với đường EMA.")

    # ------------------ TAB 7: BÁO CÁO HỌC THUẬT ------------------
    with tabs[6]:
        st.subheader("📚 Báo Cáo Chẩn Đoán Học Thuật & Đánh Giá Rủi Ro")

        st.markdown("""
        ### 1. Khắc Phục Lỗi "Hạch Toán Vị Thế" (Position Accounting)
        - **Vấn đề ban đầu:** Thuật toán phát tín hiệu Mua liên tục trong xu hướng tăng dù đã có vị thế, và phát tín hiệu Bán khi đang giữ tiền mặt.
        - **Giải pháp:** Xây dựng máy trạng thái hữu hạn (State Machine) quản lý vị thế nhị phân:
          - Trạng thái $0$ (Flat/Cash) ➔ chỉ chuyển sang $1$ (Long) khi gặp tín hiệu Mua đầu tiên.
          - Trạng thái $1$ (Long) ➔ chỉ chuyển về $0$ khi giá vi phạm EMA hoặc chạm ngưỡng cắt lỗ $7\%$.

        ### 2. Loại Trừ Look-Ahead Bias
        - Việc ra quyết định tại phiên $t$ sử dụng chỉ báo tính toán từ phiên $t-1$ (`.shift(1)`), mô phỏng chính xác lệnh thực hiện ở đầu phiên kế tiếp.

        ### 3. Phân Tích Hiện Tượng "Overfitting" & Chuyển Dịch Chế Độ Thị Trường (Regime Shift)
        - **Trên tập Train (2014 - 2020):**
          - Cổ phiếu ACB nằm trong một siêu chu kỳ tăng giá (Bull Market dài hạn).
          - Chiến lược EMA + OBV đạt kết quả vượt trội: Lợi nhuận **+236.56%**, Sharpe Ratio **1.1514**, Max Drawdown chỉ **-23.87%**.
        - **Trên tập Test (2021 - 2023):**
          - Thị trường tài chính Việt Nam bước vào giai đoạn biến động mạnh, xuất hiện các đợt sụt giảm sâu năm 2022 và đi ngang (Sideway) kéo dài.
          - Các chiến lược bám theo xu hướng (Trend Following) như EMA và OBV gặp phải hiện tượng **Whipsaw (Tín hiệu nhiễu)** liên tục.
          - Lợi nhuận tập Test âm **-36.47%**, trong khi Mua & Nắm giữ ACB đạt **+30.34%**.
        - **Bài học định lượng:** Đây là minh chứng kinh điển cho thấy một chiến lược được tối ưu hóa quá mức trên dữ liệu quá khứ (Curve-Fitting) có thể thất bại khi thị trường thay đổi chế độ biến động.

        ### 4. Đề Xuất Cải Tiến Cho Tương Lai
        1. **Thêm Bộ Lọc Xu Hướng Vĩ Mô:** Chỉ cho phép mở vị thế Mua cổ phiếu khi chỉ số VN-Index nằm trên đường SMA 200 ngày.
        2. **Cơ Chế Cắt Lỗ Động (Trailing Stop):** Khóa lợi nhuận khi giá tăng mạnh thay vì giữ nguyên mức Stop-loss cố định.
        3. **Walk-Forward Analysis (WFA):** Tối ưu hóa lại tham số định kỳ hàng năm thay vì dùng một bộ tham số cố định cho toàn bộ giai đoạn.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
        "Hệ thống Kiểm định Chiến lược Định lượng | Phát triển cho Streamlit Cloud & GitHub | Nhóm Nghiên cứu Tài chính Định lượng"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
