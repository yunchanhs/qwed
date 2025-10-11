import time
import pyupbit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta, date
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import threading
import logging
import math
from logging.handlers import RotatingFileHandler
from collections import defaultdict, deque

# ====== (선택) 실거래 전 모의체결을 원한다면 True ======
DRY_RUN = False  # True면 주문은 로그만, 계좌 변동 없음

# ====== 하락장에서도 조건 충족 시 매수 허용 ======
BEAR_ALLOW_BUYS = True

# === 안전 OHLCV 호출 & DF 검증 ===
def safe_get_ohlcv(ticker, interval="minute5", count=200, max_retries=5, base_sleep=0.7):
    for attempt in range(1, max_retries + 1):
        try:
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
            if df is not None and not df.empty and all(c in df.columns for c in ["open","high","low","close","volume"]):
                return df
            else:
                print(f"[safe_get_ohlcv] 빈 DF/컬럼 부족: {ticker} {interval} (시도 {attempt}/{max_retries})")
        except Exception as e:
            print(f"[safe_get_ohlcv] 예외: {ticker} {interval} (시도 {attempt}/{max_retries}) → {e}")
        time.sleep(base_sleep * attempt)
    return None

def is_valid_df(df, min_len=5):
    return df is not None and not df.empty and len(df) >= min_len and all(
        c in df.columns for c in ["open","high","low","close","volume"]
    )

# ====== API 키 (환경변수 권장) ======
ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4")
SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN")

# ====== 스케줄/전략 파라미터 ======
last_trained_time = {}                   # { "KRW-BTC": datetime }
TRAINING_INTERVAL = timedelta(hours=6)

# (베이스 문턱 — 레짐/분위수/히스테리시스로 가변 적용)
ML_BASE_THRESHOLD = 0.5
ML_SELL_THRESHOLD = 0.3
STOP_LOSS_THRESHOLD = -0.05
COOLDOWN_TIME = timedelta(minutes=30)
SURGE_COOLDOWN_TIME = timedelta(minutes=60)

# === 포지션/현금 배분 설정(베이스 값) ===
MAX_ACTIVE_POSITIONS_BASE = 3
USE_CASH_RATIO_BASE = 0.95
MIN_ORDER_KRW = 6000

# === 상위 코인 풀 동적 기본계수 (동적 N의 베이스) ===
TOP_POOL_MULTIPLIER = 12
TOP_POOL_BASE       = 4

# ====== 상태 ======
entry_prices = {}
highest_prices = {}
recent_trades = {}
recent_surge_tickers = {}
last_top_update = datetime.min

# === ML 출력 분포 저장(가변 문턱용)
ml_hist = defaultdict(lambda: deque(maxlen=300))  # 티커별 최근 ML 출력 300개

# === 스케일-인(분할매수) 계획
pos_plan = {}

# ====== 로깅 ======
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
handler = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)

def safe_sleep(sec):
    try:
        time.sleep(sec)
    except Exception:
        pass

# ====== 상태 저장 (원자 + 락) ======
def load_pickle(filename, default_value):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning(f"[로드 실패] {filename}: {e}")
    return default_value

entry_prices = load_pickle("entry_prices.pkl", {})
recent_trades = load_pickle("recent_trades.pkl", {})
highest_prices = load_pickle("highest_prices.pkl", {})
reserved_profit = load_pickle("reserved_profit.pkl", 0.0)
equity_hwm = load_pickle("equity_hwm.pkl", 0.0)
pnl_today = load_pickle("pnl_today.pkl", 0.0)
try:
    _pday = load_pickle("pnl_day.pkl", datetime.now().date().isoformat())
    pnl_day = datetime.fromisoformat(_pday).date() if isinstance(_pday, str) else datetime.now().date()
except Exception:
    pnl_day = datetime.now().date()
consecutive_losses = load_pickle("consecutive_losses.pkl", 0)

state_lock = threading.Lock()

def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def auto_save_state(interval=300):
    while True:
        try:
            with state_lock:
                atomic_save(entry_prices, "entry_prices.pkl")
                atomic_save(recent_trades, "recent_trades.pkl")
                atomic_save(highest_prices, "highest_prices.pkl")
                atomic_save(reserved_profit, "reserved_profit.pkl")
                atomic_save(equity_hwm, "equity_hwm.pkl")
                atomic_save(pnl_today, "pnl_today.pkl")
                atomic_save(pnl_day.isoformat(), "pnl_day.pkl")
                atomic_save(consecutive_losses, "consecutive_losses.pkl")
            log.info("[백업] 상태 자동 저장 완료")
        except Exception as e:
            log.exception(f"[백업 오류] 상태 저장 실패: {e}")
        time.sleep(interval)

# ====== 모델/지표 ======
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.activation = nn.Sigmoid()  # 0~1

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.fc(x[:, -1, :])
        x = self.activation(x)
        return x

def get_macd_from_df(df):
    df['short_ema'] = df['close'].ewm(span=12, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['short_ema'] - df['long_ema']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def get_rsi_from_df(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def get_adx_from_df(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['+DM'] = df['high'] - df['high'].shift(1)
    df['-DM'] = df['low'].shift(1) - df['low']
    df['+DM'] = df['+DM'].where(df['+DM'] > df['-DM'], 0)
    df['-DM'] = df['-DM'].where(df['-DM'] > df['+DM'], 0)
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['adx'] = df['DX'].rolling(window=period).mean()
    return df

def get_atr_from_df(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['atr'] = df['TR'].rolling(window=period).mean()
    return df

def get_features(ticker, normalize=True):
    df = safe_get_ohlcv(ticker, interval="minute5", count=1000)
    if not is_valid_df(df, min_len=100):
        return pd.DataFrame()
    df = get_macd_from_df(df)
    df = get_rsi_from_df(df)
    df = get_adx_from_df(df)
    df = get_atr_from_df(df)
    df['return'] = df['close'].pct_change()
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df.dropna(inplace=True)
    if normalize and not df.empty:
        scaler = MinMaxScaler()
        cols = ['macd', 'signal', 'rsi', 'adx', 'atr', 'return', 'future_return']
        df[cols] = scaler.fit_transform(df[cols])
    return df

# ====== 자본/리저브/드로우다운 관리 ======
DAILY_MAX_LOSS = 0.02
MAX_CONSECUTIVE_LOSSES = 3
PROFIT_SKIM_TRIGGER = 0.03
PROFIT_SKIM_RATIO   = 0.25
RESERVE_RELEASE_DD  = 0.02

def calc_total_equity():
    try:
        krw = get_balance("KRW") or 0.0
    except Exception:
        krw = 0.0
    equity = float(krw)
    for t in set(entry_prices.keys()):
        try:
            coin = t.split('-')[1]
            bal = get_balance(coin)
            if bal and bal > 1e-10:
                px = pyupbit.get_current_price(t)
                if px:
                    equity += float(bal) * float(px)
        except Exception:
            continue
    return equity

def get_initial_balance_for_backtest():
    eq = calc_total_equity()
    return max(300_000, min(10_000_000, int(eq)))

def reset_daily_if_needed():
    global pnl_today, pnl_day, consecutive_losses
    today = datetime.now().date()
    if pnl_day != today:
        pnl_day = today
        pnl_today = 0.0
        consecutive_losses = 0

def record_trade_pnl(pnl_ratio):
    global pnl_today, consecutive_losses
    pnl_today += pnl_ratio
    if pnl_ratio < 0:
        consecutive_losses += 1
    else:
        consecutive_losses = 0

def update_profit_reserve():
    global equity_hwm, reserved_profit
    eq = calc_total_equity()
    if eq > equity_hwm:
        equity_hwm = eq
    threshold = equity_hwm * (1 + PROFIT_SKIM_TRIGGER)
    if eq >= threshold:
        skim_amount = (eq - equity_hwm) * PROFIT_SKIM_RATIO
        if skim_amount > 0:
            reserved_profit += skim_amount
            equity_hwm = eq
            log.info(f"[RESERVE] Skim +{skim_amount:.0f}원 | reserve={reserved_profit:.0f}, HWM={equity_hwm:.0f}")
    if equity_hwm > 0:
        dd = (equity_hwm - eq) / equity_hwm
        if dd >= RESERVE_RELEASE_DD and reserved_profit > 0:
            release = reserved_profit * 0.5
            reserved_profit -= release
            log.info(f"[RESERVE] DD {dd*100:.2f}% → Release {release:.0f}원 | reserve={reserved_profit:.0f}")
    return eq

def get_dd_stage_params():
    """
    드로우다운 단계별 제어:
      stage0: DD<5%          → 기본
      stage1: DD≥5%          → 현금비중 0.80
      stage2: DD≥10%         → 포지션수 -1
      stage3: DD≥15%         → 신규매수 차단 + 현금비중 0.70
    """
    eq = calc_total_equity()
    dd = 0.0 if equity_hwm <= 0 else (equity_hwm - eq) / equity_hwm
    stage = 0
    use_cash = USE_CASH_RATIO_BASE
    max_pos = MAX_ACTIVE_POSITIONS_BASE
    buy_block = False
    if dd >= 0.15:
        stage = 3; use_cash = 0.70; max_pos = max(1, MAX_ACTIVE_POSITIONS_BASE-2); buy_block = True
    elif dd >= 0.10:
        stage = 2; use_cash = 0.75; max_pos = max(1, MAX_ACTIVE_POSITIONS_BASE-1)
    elif dd >= 0.05:
        stage = 1; use_cash = 0.80
    log.info(f"[DD-MONITOR] DD={dd*100:.2f}% (Stage {stage}) | use_cash={use_cash:.2f}, max_pos={max_pos}, buy_block={buy_block}")
    return dd, stage, use_cash, max_pos, buy_block

# ====== 거래 함수 ======
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

def get_balance(ticker):
    try:
        balance = upbit.get_balance(ticker)
        if balance is None:
            print(f"[경고] {ticker} 잔고 None 반환 → 0으로 처리")
            return 0
        return balance
    except Exception as e:
        print(f"[오류] {ticker} 잔고 조회 실패: {e}")
        return 0

def buy_crypto_currency(ticker, amount):
    if DRY_RUN:
        print(f"[DRY_RUN][BUY] {ticker} {amount:.0f} KRW")
        return {"dry": True, "ticker": ticker, "amount": amount}
    try:
        order = upbit.buy_market_order(ticker, amount)
        return order
    except Exception as e:
        print(f"[{ticker}] 매수 중 에러 발생: {e}")
        return None

def sell_crypto_currency(ticker, amount):
    if DRY_RUN:
        print(f"[DRY_RUN][SELL] {ticker} {amount}")
        return {"dry": True, "ticker": ticker, "amount": amount}
    try:
        order = upbit.sell_market_order(ticker, amount)
        return order
    except Exception as e:
        print(f"[{ticker}] 매도 중 에러 발생: {e}")
        return None

# ====== 데이터셋/학습 ======  (num_workers=0 안정화)
class TradingDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.seq_len][['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values
        y = self.data.iloc[idx + self.seq_len]['future_return']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_transformer_model(ticker, epochs=50):
    log.info(f"모델 학습 시작: {ticker}")
    input_dim = 6; d_model = 32; num_heads = 4; num_layers = 1; output_dim = 1
    model = TransformerModel(input_dim, d_model, num_heads, num_layers, output_dim)
    data = get_features(ticker, normalize=True)
    if data is None or data.empty:
        log.info(f"경고: {ticker} 데이터 비어 있음. 학습 스킵"); return None
    seq_len = 30
    dataset = TradingDataset(data, seq_len)
    if len(dataset) == 0:
        log.info(f"경고: {ticker} 데이터셋 너무 작음. 학습 스킵"); return None
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(-1), y_batch.view(-1))
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            log.info(f"[{ticker}] Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")
    log.info(f"모델 학습 완료: {ticker}")
    return model

# === 백테스트(수수료+슬리피지)
def backtest(ticker, model, initial_balance=1_000_000, fee=0.0005, slip_bp=10):
    data = get_features(ticker)
    if data is None or data.empty:
        return 1.0
    balance = initial_balance; position = 0; entry_price = 0
    highest_price = 0
    slip = slip_bp/10000.0
    for i in range(50, len(data) - 1):
        x_input = torch.tensor(
            data.iloc[i-30:i][['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values,
            dtype=torch.float32
        ).unsqueeze(0)
        ml_signal = model(x_input).item()
        current_price = data.iloc[i]['close']
        if position == 0 and ml_signal > ML_BASE_THRESHOLD:
            fill = current_price * (1 + slip)
            position = balance / fill
            entry_price = fill
            highest_price = entry_price
            balance = 0
        elif position > 0:
            highest_price = max(highest_price, current_price)
            peak_drop = (highest_price - current_price) / highest_price
            unrealized = (current_price - entry_price) / entry_price
            if unrealized < STOP_LOSS_THRESHOLD:
                fill = current_price * (1 - slip)
                balance = position * fill * (1 - fee); position = 0; continue
            if peak_drop > 0.02 and ml_signal < ML_SELL_THRESHOLD:
                fill = current_price * (1 - slip)
                balance = position * fill * (1 - fee); position = 0; continue
    final = balance + (position * data.iloc[-1]['close'])
    return final / initial_balance

# === 레짐 인식: BTC/ETH + 브레드스(상위코인 중 MA20 위 비율) 결합 ===
def get_asset_regime(ticker):
    """
    H1 기준:
      bull: MACD>Signal & RSI>55
      bear: MACD<Signal & RSI<45
      else: neutral
    """
    try:
        df = safe_get_ohlcv(ticker, interval="minute60", count=200)
        if not is_valid_df(df, min_len=100):
            return "neutral"
        mac = get_macd_from_df(df.copy())
        macd, signal = mac['macd'].iloc[-1], mac['signal'].iloc[-1]
        rsi = get_rsi_from_df(df.copy())['rsi'].iloc[-1]
        if (macd > signal) and (rsi > 55): return "bull"
        if (macd < signal) and (rsi < 45): return "bear"
        return "neutral"
    except Exception:
        return "neutral"

def compute_breadth_above_ma20(top_list):
    """
    후보군(top_list) 각각의 H1 MA20 위/아래를 체크해서 비율(0~1)을 반환
    """
    if not top_list:
        log.info("[BREADTH] top_list 비어 있음 → 0.0")
        return 0.0
    count_above = 0
    total = 0
    for t in top_list:
        try:
            df = safe_get_ohlcv(t, interval="minute60", count=60)
            if not is_valid_df(df, min_len=25):
                continue
            ma20 = df['close'].rolling(window=20).mean()
            if pd.notna(ma20.iloc[-1]):
                total += 1
                if df['close'].iloc[-1] > ma20.iloc[-1]:
                    count_above += 1
        except Exception:
            continue
    if total == 0:
        return 0.0
    return count_above / total

def composite_market_regime(top_list):
    """
    보수적 결합:
      - bear: BTC bear 또는 breadth < 0.40
      - bull: BTC bull 그리고 (ETH bull 또는 breadth > 0.60)
      - neutral: 나머지
    """
    btc_reg = get_asset_regime("KRW-BTC")
    eth_reg = get_asset_regime("KRW-ETH")
    breadth = compute_breadth_above_ma20(top_list)  # 0~1
    if btc_reg == "bear" or breadth < 0.40:
        regime = "bear"
    elif (btc_reg == "bull") and (eth_reg == "bull" or breadth > 0.60):
        regime = "bull"
    else:
        regime = "neutral"
    log.info(f"[REGIME] BTC={btc_reg} ETH={eth_reg} breadth={breadth*100:.1f}% → regime={regime}")
    return regime

# === 가변 문턱(분위수) + 히스테리시스
HYST_DELTA = 0.05  # T_sell = T_buy - 0.05

def compute_ml_threshold(ticker, regime):
    base = ML_BASE_THRESHOLD
    # 최근 출력 분위수(적응형)
    hist = ml_hist[ticker]
    if len(hist) >= 60:
        q = np.quantile(hist, 0.75)  # 상위 25% 경계
        base = max(0.35, min(0.65, float(q)))
    # 레짐 보정
    if regime == "bull":
        base -= 0.03
    elif regime == "bear":
        base += 0.05
    base = max(0.35, min(0.70, base))
    t_buy = base
    t_sell = max(0.0, t_buy - HYST_DELTA)
    return t_buy, t_sell

# === 스프레드·체결강도 기반 우선순위 랭킹
def get_spread_bp(ticker):
    try:
        orderbook = pyupbit.get_orderbook(ticker)
        if not orderbook or 'orderbook_units' not in orderbook[0]:
            return None
        unit = orderbook[0]['orderbook_units'][0]
        ask, bid = float(unit['ask_price']), float(unit['bid_price'])
        spread = (ask - bid) / ((ask + bid) / 2)
        return spread * 10000.0  # bp
    except Exception:
        return None

def rank_buy_candidates(held_set, top_list, surge_dict, slots):
    cand = []
    seen = set()
    for t in surge_dict.keys():
        if t not in held_set and t not in seen: cand.append(t); seen.add(t)
    for t in top_list:
        if t not in held_set and t not in seen: cand.append(t); seen.add(t)
    scored = []
    for t in cand:
        sp = get_spread_bp(t)
        sp_score = 0.0 if sp is None else max(-2.0, min(2.0, (15.0 - sp) / 5.0))  # 15bp 기준
        surge_bonus = 0.5 if t in surge_dict else 0.0
        scored.append((t, sp_score + surge_bonus))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:max(0, slots)]]

# === ATR 기반 포지션 사이징 + 하드 리스크 캡
POS_RISK_CAP = 0.0075  # 포지션당 계좌위험 0.75% 상한
def calc_atr_position_budget(remaining_krw, remaining_slots, atr_abs, px, equity, base_risk=0.006):
    if atr_abs is None or atr_abs <= 0 or px <= 0:
        return (remaining_krw * USE_CASH_RATIO_BASE) / max(1, remaining_slots)
    k = 1.5
    est_stop_ratio = (atr_abs * k) / px
    if est_stop_ratio <= 0:
        return (remaining_krw * USE_CASH_RATIO_BASE) / max(1, remaining_slots)
    budget_by_risk = (equity * base_risk) / est_stop_ratio
    budget_hardcap = (equity * POS_RISK_CAP) / est_stop_ratio
    equal_split = (remaining_krw * USE_CASH_RATIO_BASE) / max(1, remaining_slots)
    return max(MIN_ORDER_KRW, min(budget_by_risk, budget_hardcap, 1.5 * equal_split))

# === 부분 익절 + 트레일 강화
PARTIAL_TP1 = 0.08; PARTIAL_TP2 = 0.15
TP1_RATIO   = 0.40; TP2_RATIO   = 0.30
TRAIL_DROP_BULL  = 0.04
TRAIL_DROP_BEAR  = 0.025

def try_partial_take_profit(ticker, change_ratio, coin_balance, now):
    did = False
    if change_ratio >= PARTIAL_TP2 and coin_balance > 0:
        amt = coin_balance * TP2_RATIO
        if sell_crypto_currency(ticker, amt):
            did = True; print(f"[{ticker}] 부분익절2: +{PARTIAL_TP2*100:.0f}% → {TP2_RATIO*100:.0f}% 매도")
    elif change_ratio >= PARTIAL_TP1 and coin_balance > 0:
        amt = coin_balance * TP1_RATIO
        if sell_crypto_currency(ticker, amt):
            did = True; print(f"[{ticker}] 부분익절1: +{PARTIAL_TP1*100:.0f}% → {TP1_RATIO*100:.0f}% 매도")
    if did:
        with state_lock:
            recent_trades[ticker] = now
    return did

def should_sell(ticker, current_price, ml_signal, t_sell, regime):
    if ticker not in entry_prices:
        return False
    entry_price = entry_prices[ticker]
    highest_prices[ticker] = max(highest_prices.get(ticker, entry_price), current_price)
    change_ratio = (current_price - entry_price) / entry_price
    peak_drop = (highest_prices[ticker] - current_price) / highest_prices[ticker]
    weak_ml = (ml_signal < t_sell)

    if change_ratio < -0.05: print(f"[{ticker}] 🚨 -5% 손절 발동"); return True
    if change_ratio >= 0.20: print(f"[{ticker}] 🎯 20% 이상 수익 → 무조건 익절"); return True
    elif change_ratio >= 0.15:
        if weak_ml or ml_signal < 0.6: print(f"[{ticker}] ✅ 15% 수익 + AI 약함 → 익절"); return True
        else: print(f"[{ticker}] ✅ 15% 수익 + AI 강함 → 보유"); return False
    elif change_ratio >= 0.10:
        if weak_ml or ml_signal < 0.5: print(f"[{ticker}] ✅ 10% 수익 + AI 약함 → 익절"); return True

    trail_drop = TRAIL_DROP_BULL if regime == "bull" else TRAIL_DROP_BEAR
    if peak_drop > trail_drop and (weak_ml or ml_signal < 0.5):
        print(f"[{ticker}] 📉 트레일링 스탑 발동! 고점 대비 {peak_drop*100:.2f}%"); return True

    try:
        if change_ratio > 0.05 and ml_signal > 0.6:
            df_m5 = safe_get_ohlcv(ticker, interval="minute5", count=200)
            if is_valid_df(df_m5, min_len=50):
                df_m5 = get_macd_from_df(df_m5)
                macd = df_m5['macd'].iloc[-1]; signal = df_m5['signal'].iloc[-1]
                if macd > signal:
                    print(f"[{ticker}] 📈 추세 지속 (MACD 상승) → 보유"); return False
    except Exception as e:
        print(f"[{ticker}] MACD 계산 오류: {e}")
    try:
        df_m5b = safe_get_ohlcv(ticker, interval="minute5", count=200)
        if is_valid_df(df_m5b, min_len=50):
            df_m5b = get_macd_from_df(df_m5b); df_m5b = get_rsi_from_df(df_m5b)
            rsi = df_m5b['rsi'].iloc[-1]; macd = df_m5b['macd'].iloc[-1]; signal = df_m5b['signal'].iloc[-1]
            if rsi > 80 and (weak_ml or ml_signal < 0.5):
                print(f"[{ticker}] RSI 과매수 + AI 약함 → 매도"); return True
            if macd < signal and (weak_ml or ml_signal < 0.5):
                print(f"[{ticker}] MACD 데드크로스 + AI 약함 → 매도"); return True
    except Exception as e:
        print(f"[{ticker}] RSI/MACD 보조 지표 오류: {e}")
    return False

# ====== 유틸: 보유 정합/보유 티커 계산 ======
def reconcile_positions_from_balance():
    to_drop = []
    for t in list(entry_prices.keys()):
        try:
            coin = t.split('-')[1]
            bal = get_balance(coin)
            if not bal or bal < 1e-8:
                to_drop.append(t)
        except Exception:
            to_drop.append(t)
    if to_drop:
        with state_lock:
            for t in to_drop:
                entry_prices.pop(t, None)
                highest_prices.pop(t, None)
        print(f"[RECONCILE] 실보유 0인 티커 정리: {to_drop}")

def get_held_tickers_from_balance():
    held = set()
    for t in entry_prices.keys():
        try:
            coin = t.split('-')[1]
            bal = get_balance(coin)
            if bal and bal > 1e-8:
                held.add(t)
        except Exception:
            pass
    return held

# === 동적 후보수 계산: 자본·레짐 반영 ===
def compute_top_n(current_top=None):
    """
    - 기준 자본 1.5M를 중심으로 0.8~2.0배 스케일
    - 레짐에 따라 bull +20%, bear -20%
    - 베이스는 (MAX_ACTIVE_POSITIONS_BASE * TOP_POOL_MULTIPLIER + TOP_POOL_BASE)
    - 최종 하드범위: 20 ~ 60
    """
    equity = calc_total_equity()
    scale = min(2.0, max(0.8, equity / 1_500_000))

    try:
        regime = composite_market_regime(current_top or [])
    except Exception:
        regime = "neutral"
    regime_k = 1.2 if regime == "bull" else (0.8 if regime == "bear" else 1.0)

    base = MAX_ACTIVE_POSITIONS_BASE * TOP_POOL_MULTIPLIER + TOP_POOL_BASE
    n = int(base * scale * regime_k)
    n = max(20, min(60, n))
    log.info(f"[TOP-N] equity≈{equity:.0f}, regime={regime}, scale={scale:.2f}, n={n}")
    return n

# === 안전 버전: 인자 없이 호출돼도 동적 N 사용 ===
def get_top_tickers(n=None):
    """
    n이 None/0/음수면 compute_top_n()로 동적 결정.
    정상 값이 들어오면 그 값을 그대로 사용.
    """
    if n is None or n <= 0:
        n = compute_top_n(current_top=[])
    n = max(20, min(60, int(n)))  # 하드 클램프

    tickers = pyupbit.get_tickers(fiat="KRW")
    scores = []
    for ticker in tickers:
        df = safe_get_ohlcv(ticker, interval="day", count=3)
        if not is_valid_df(df, min_len=3):
            continue
        df["value"] = df["close"] * df["volume"]
        avg_value = float(df["value"].mean())
        adjusted_score = np.log1p(avg_value)
        scores.append((ticker, adjusted_score))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in sorted_scores[:n]]

# ====== 자동 저장 쓰레드 시작 ======
save_thread = threading.Thread(target=auto_save_state, daemon=True)
save_thread.start()

# ====== 메인 ======
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    models = {}

    # 초기 설정 — 동적 N 적용 (인자 없이도 동작)
    top_tickers = get_top_tickers()  # 내부에서 compute_top_n() 호출
    log.info(f"[{datetime.now()}] 상위 코인 초기화(N={len(top_tickers)}): {top_tickers}")
    last_top_update = datetime.now()  # ▶ 즉시 재학습 루프 방지

    # 초기 학습 (워크포워드 필터 적용)
    for ticker in top_tickers:
        model = train_transformer_model(ticker)
        if model is None:
            continue
        # 동적 초기잔고로 백테스트/워크포워드 수행
        init_bal = get_initial_balance_for_backtest()
        perf_in = backtest(ticker, model, initial_balance=init_bal)
        # 워크포워드: 최근 20% 구간만 분리 평가
        df_all = get_features(ticker)
        if df_all is None or df_all.empty or len(df_all) < 200:
            perf_wf = 1.0
        else:
            cut = int(len(df_all) * 0.8)
            wf = df_all.iloc[cut:].copy()
            seq_len = 30
            if len(wf) <= seq_len + 1:
                perf_wf = 1.0
            else:
                # 간단 워크포워드: 마지막 구간만 모델 추정 → 가상 체결
                balance = init_bal; position = 0; entry = 0
                for i in range(seq_len, len(wf)-1):
                    X = torch.tensor(wf.iloc[i-seq_len:i][['macd','signal','rsi','adx','atr','return']].values, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        s = model(X).item()
                    px = wf.iloc[i]['close']
                    if position == 0 and s > ML_BASE_THRESHOLD:
                        position = balance / px; entry = px; balance = 0
                    elif position > 0:
                        if (px - entry)/entry < STOP_LOSS_THRESHOLD:
                            balance = position * px; position = 0
                final = balance + (position * wf.iloc[-1]['close'])
                perf_wf = final / init_bal

        if perf_in > 1.05 and perf_wf > 1.02:
            models[ticker] = model
            last_trained_time[ticker] = datetime.now()
            log.info(f"[{ticker}] 모델 추가 (일반:{perf_in:.2f}배 / 워크:{perf_wf:.2f}배)")
        else:
            log.info(f"[{ticker}] 모델 제외 (일반:{perf_in:.2f}배 / 워크:{perf_wf:.2f}배)")

    # 시작 시 1회 유령 보유 정리
    reconcile_positions_from_balance()
    last_reconcile = datetime.min

    recent_surge_tickers = {}

    try:
        while True:
            now = datetime.now()

            # 일일 리셋 & 리저브 갱신
            reset_daily_if_needed()
            eq = update_profit_reserve()

            # DD 단계별 유효 파라미터
            dd, dd_stage, USE_CASH_RATIO_EFF, MAX_ACTIVE_POS_EFF, BUY_BLOCK_DD = get_dd_stage_params()

            # 30분마다 유령 보유 정리
            if (now - last_reconcile) >= timedelta(minutes=30):
                reconcile_positions_from_balance()
                last_reconcile = now

            # 1) 상위 코인 업데이트 (6시간마다) — 동적 N 재계산은 get_top_tickers() 내부 처리
            if (now - last_top_update) >= timedelta(hours=6):
                top_tickers = get_top_tickers()  # 내부에서 compute_top_n([]) 호출
                log.info(f"[{now}] 상위 코인 업데이트(N={len(top_tickers)}): {top_tickers}")
                last_top_update = now  # ▶ 윈도우 갱신
                for ticker in top_tickers:
                    # 즉시 재학습 방지
                    if ticker not in models or (datetime.now() - last_trained_time.get(ticker, datetime.min) > TRAINING_INTERVAL):
                        model = train_transformer_model(ticker)
                        if model is None:
                            continue
                        init_bal = get_initial_balance_for_backtest()
                        perf_in = backtest(ticker, model, initial_balance=init_bal)
                        # 간단 워크포워드
                        df_all = get_features(ticker)
                        if df_all is None or df_all.empty or len(df_all) < 200:
                            perf_wf = 1.0
                        else:
                            cut = int(len(df_all) * 0.8)
                            wf = df_all.iloc[cut:].copy()
                            seq_len = 30
                            if len(wf) <= seq_len + 1:
                                perf_wf = 1.0
                            else:
                                balance = init_bal; position = 0; entry = 0
                                for i in range(seq_len, len(wf)-1):
                                    X = torch.tensor(wf.iloc[i-seq_len:i][['macd','signal','rsi','adx','atr','return']].values, dtype=torch.float32).unsqueeze(0)
                                    with torch.no_grad():
                                        s = model(X).item()
                                    px = wf.iloc[i]['close']
                                    if position == 0 and s > ML_BASE_THRESHOLD:
                                        position = balance / px; entry = px; balance = 0
                                    elif position > 0:
                                        if (px - entry)/entry < STOP_LOSS_THRESHOLD:
                                            balance = position * px; position = 0
                                final = balance + (position * wf.iloc[-1]['close'])
                                perf_wf = final / init_bal

                        if perf_in >= 1.05 and perf_wf >= 1.02:
                            models[ticker] = model
                            last_trained_time[ticker] = datetime.now()
                            log.info(f"[{ticker}] 모델 추가/갱신 (일반:{perf_in:.2f}배 / 워크:{perf_wf:.2f}배)")
                        else:
                            log.info(f"[{ticker}] 모델 제외 (일반:{perf_in:.2f}배 / 워크:{perf_wf:.2f}배)")

            # 2) 급상승 감지
            surge_tickers = detect_surge_tickers(threshold=0.03)
            for ticker in surge_tickers:
                if ticker not in recent_surge_tickers:
                    print(f"[{now}] 급상승 감지: {ticker}")
                    recent_surge_tickers[ticker] = now
                    if ticker not in models:
                        model = train_transformer_model(ticker, epochs=10)
                        if model is None:
                            continue
                        init_bal = get_initial_balance_for_backtest()
                        perf_in = backtest(ticker, model, initial_balance=init_bal)
                        # 간단 워크포워드
                        df_all = get_features(ticker)
                        if df_all is None or df_all.empty or len(df_all) < 200:
                            perf_wf = 1.0
                        else:
                            cut = int(len(df_all) * 0.8)
                            wf = df_all.iloc[cut:].copy()
                            seq_len = 30
                            if len(wf) <= seq_len + 1:
                                perf_wf = 1.0
                            else:
                                balance = init_bal; position = 0; entry = 0
                                for i in range(seq_len, len(wf)-1):
                                    X = torch.tensor(wf.iloc[i-seq_len:i][['macd','signal','rsi','adx','atr','return']].values, dtype=torch.float32).unsqueeze(0)
                                    with torch.no_grad():
                                        s = model(X).item()
                                    px = wf.iloc[i]['close']
                                    if position == 0 and s > ML_BASE_THRESHOLD:
                                        position = balance / px; entry = px; balance = 0
                                    elif position > 0:
                                        if (px - entry)/entry < STOP_LOSS_THRESHOLD:
                                            balance = position * px; position = 0
                                final = balance + (position * wf.iloc[-1]['close'])
                                perf_wf = final / init_bal

                        if perf_in > 1.1 and perf_wf >= 1.02:
                            models[ticker] = model
                            last_trained_time[ticker] = datetime.now()
                            log.info(f"[{ticker}] 급상승 모델 추가 (일반:{perf_in:.2f}배 / 워크:{perf_wf:.2f}배)")
                        else:
                            log.info(f"[{ticker}] 급상승 모델 제외 (일반:{perf_in:.2f}배 / 워크:{perf_wf:.2f}배)")

            # 3) 복합 레짐 계산
            regime = composite_market_regime(top_tickers)  # "bull" | "neutral" | "bear"
            market_block = (regime == "bear")

            # 손실/연속손실 블럭
            risk_block = (pnl_today <= -DAILY_MAX_LOSS) or (consecutive_losses >= MAX_CONSECUTIVE_LOSSES)

            # 보유/대상
            held_tickers = get_held_tickers_from_balance()
            target_tickers = set(top_tickers) | set(recent_surge_tickers.keys()) | held_tickers

            # 유효 슬롯(DD 고려)
            slots_available = max(0, MAX_ACTIVE_POS_EFF - len(held_tickers))

            # 후보 랭킹 (스프레드/급등 가점) — bear라도 랭킹은 산출 (후보 힌트용)
            ranked_candidates = rank_buy_candidates(held_tickers, top_tickers, recent_surge_tickers, slots_available)
            # 기본 블럭(DD/리스크)은 유지. 시장(bear) 블럭은 per-ticker 예외(allow_bear_buy)로 해제 가능
            base_block_new = BUY_BLOCK_DD or risk_block
            buy_allowed_tickers = set([] if base_block_new or slots_available == 0 else ranked_candidates)

            log.info(f"[BUY-SELECTION] held={list(held_tickers)} slots={slots_available} allowed={list(buy_allowed_tickers)} regime={regime} base_block={base_block_new}")

            # 사용 가능 KRW (리저브 차감)
            krw_now = get_balance("KRW") or 0.0
            usable_krw = max(0.0, krw_now - reserved_profit)
            remaining_krw = usable_krw
            remaining_slots = slots_available
            log.info(f"[RESERVE] KRW={krw_now:.0f}, reserve={reserved_profit:.0f}, usable={usable_krw:.0f}, HWM={equity_hwm:.0f}")

            # 4) 매수/매도 루프
            for ticker in target_tickers:
                cooldown_limit = SURGE_COOLDOWN_TIME if ticker in recent_surge_tickers else COOLDOWN_TIME
                last_trade_time = recent_trades.get(ticker, datetime.min)
                if now - last_trade_time < cooldown_limit:
                    continue

                try:
                    if ticker not in models:
                        log.info(f"[{ticker}] 모델 없음 → 스킵")
                        continue

                    # 피처 & 지표
                    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
                    if df is None or df.empty:
                        continue
                    df = get_macd_from_df(df)
                    df = get_rsi_from_df(df)
                    df = get_adx_from_df(df)
                    df = get_atr_from_df(df)
                    df = get_features(ticker, normalize=False)

                    macd = df['macd'].iloc[-1]
                    signal = df['signal'].iloc[-1]
                    rsi = df['rsi'].iloc[-1]
                    adx = df['adx'].iloc[-1]
                    atr_abs = df['atr'].iloc[-1]
                    current_price = df['close'].iloc[-1]

                    # 모델 신호 + 히스토리 저장
                    model = models[ticker]
                    features = get_features(ticker)
                    latest_data = features[['macd','signal','rsi','adx','atr','return']].tail(30)
                    X_latest = torch.tensor(latest_data.values, dtype=torch.float32).unsqueeze(0)
                    model.eval()
                    with torch.no_grad():
                        ml_signal = model(X_latest).item()
                    ml_hist[ticker].append(ml_signal)

                    # 가변 문턱 + 히스테리시스 (복합 레짐 사용)
                    T_buy, T_sell = compute_ml_threshold(ticker, regime)

                    log.info(f"[DEBUG] {ticker} | ML={ml_signal:.4f} T_buy={T_buy:.3f}/T_sell={T_sell:.3f} MACD={macd:.4f}/{signal:.4f} RSI={rsi:.1f} ADX={adx:.1f} ATR={atr_abs:.6f} PX={current_price:.2f}")

                    # === 하락장(bear) 전용 예외 허용 조건 (역추세/돌파/상대강도)
                    allow_bear_buy = False
                    if regime == "bear" and BEAR_ALLOW_BUYS and not base_block_new and remaining_slots > 0:
                        try:
                            # 1) 과매도 반등: RSI<35 & MACD>Signal & ML>0.55
                            if (rsi < 35) and (macd > signal) and (ml_signal > 0.55):
                                allow_bear_buy = True
                            else:
                                # 2) ATR 급증(최근 5봉 대비 20% 이상)
                                atr_now = df['atr'].iloc[-1]
                                atr_prev = df['atr'].iloc[-5] if len(df) > 5 else df['atr'].iloc[-1]
                                if atr_now > 1.2 * atr_prev:
                                    allow_bear_buy = True
                                else:
                                    # 3) BTC 대비 상대강도(직전 수익률 차)
                                    btc = safe_get_ohlcv("KRW-BTC", interval="minute5", count=200)
                                    if is_valid_df(btc, 50):
                                        btc_ret = btc['close'].pct_change().iloc[-1]
                                        alt_ret = df['close'].pct_change().iloc[-1]
                                        if (alt_ret - btc_ret) > 0.01:
                                            allow_bear_buy = True
                        except Exception:
                            allow_bear_buy = False

                    # === 매수 로직 (스케일-인: 60% → 20% → 20%)
                    # 기본 후보/슬롯 조건
                    in_rank_and_slot = (ticker in buy_allowed_tickers and remaining_slots > 0 and not base_block_new)
                    # bear 예외 허용 시 후보가 아니어도 진입 가능
                    can_new_buy = in_rank_and_slot or allow_bear_buy

                    if not can_new_buy:
                        if ticker not in entry_prices:
                            log.info(f"[{ticker}] 신규 매수 스킵: 후보 아님/슬롯0/차단(regime/risk/dd)/bear예외X")
                    else:
                        # 공통 기술 조건(레짐별 약간 보정)
                        ATR_THRESHOLD = 0.015 * current_price  # 가격의 1.5%
                        conds = [
                            ("MACD", macd > signal, f"{macd:.4f} > {signal:.4f}"),
                            ("RSI", rsi < (58 if regime=='bull' else 55), f"{rsi:.1f} < {(58 if regime=='bull' else 55)}"),
                            ("ADX", adx > (18 if regime=='bull' else 20), f"{adx:.1f} > {(18 if regime=='bull' else 20)}"),
                            ("ATR", atr_abs > ATR_THRESHOLD, f"{atr_abs:.6f} > {ATR_THRESHOLD:.6f}"),
                        ]
                        # ML 조건: bear 예외면 기존 T_buy보다 약간 높게 보정(가짜 반등 필터)
                        ml_gate = (ml_signal > (T_buy if regime!='bear' else max(T_buy, 0.55)))
                        conds.append(("ML", ml_gate, f"{ml_signal:.3f} > {max(T_buy, 0.55) if regime=='bear' else T_buy:.3f}"))

                        if all(ok for _, ok, _ in conds):
                            if remaining_krw > MIN_ORDER_KRW:
                                equity = calc_total_equity()
                                per_slot_budget = calc_atr_position_budget(remaining_krw, remaining_slots, atr_abs, current_price, equity)

                                plan = pos_plan.get(ticker)
                                if plan is None:
                                    target = max(MIN_ORDER_KRW, min(per_slot_budget, remaining_krw * USE_CASH_RATIO_EFF))
                                    first_amt = target * 0.6
                                    buy_amt = min(first_amt, remaining_krw * USE_CASH_RATIO_EFF)
                                    if buy_amt >= MIN_ORDER_KRW:
                                        if buy_crypto_currency(ticker, buy_amt):
                                            with state_lock:
                                                entry_prices[ticker] = current_price
                                                highest_prices[ticker] = current_price
                                                recent_trades[ticker] = now
                                            remaining_krw -= buy_amt
                                            remaining_slots -= 1
                                            pos_plan[ticker] = {"target": target, "filled": buy_amt, "tr": [0.2,0.2], "last": now}
                                            print(f"[{ticker}] 1차 매수(스케일-인): {buy_amt:.0f}원 / target≈{target:.0f} | 남은KRW≈{remaining_krw:.0f}, 남은슬롯={remaining_slots}")
                                else:
                                    if plan["tr"]:
                                        tranche = plan["tr"][0]
                                        add_amt = plan["target"] * tranche
                                        buy_amt = max(MIN_ORDER_KRW, min(add_amt, remaining_krw * USE_CASH_RATIO_EFF))
                                        if buy_amt >= MIN_ORDER_KRW:
                                            if buy_crypto_currency(ticker, buy_amt):
                                                with state_lock:
                                                    highest_prices[ticker] = max(highest_prices.get(ticker, current_price), current_price)
                                                    recent_trades[ticker] = now
                                                plan["filled"] += buy_amt
                                                plan["tr"].pop(0)
                                                plan["last"] = now
                                                remaining_krw -= buy_amt
                                                print(f"[{ticker}] 추가 매수(스케일-인): {buy_amt:.0f}원 (잔여 트랜치 {len(plan['tr'])}) | 남은KRW≈{remaining_krw:.0f}")
                            else:
                                print(f"[{ticker}] 매수 불가 (KRW<{MIN_ORDER_KRW})")
                        else:
                            reasons = ", ".join([f"{name}({expr})" for name, ok, expr in conds if not ok])
                            print(f"[{ticker}] 매수 조건 불충족 → {reasons}")

                    # === 매도 로직
                    if ticker in entry_prices:
                        entry_price = entry_prices[ticker]
                        highest_prices[ticker] = max(highest_prices.get(ticker, entry_price), current_price)
                        if entry_price == 0:
                            print(f"[{ticker}] 경고: entry_price=0 → 매도 스킵"); continue

                        change_ratio = (current_price - entry_price) / entry_price
                        will_sell = False
                        try:
                            _Tb, T_sell_eff = compute_ml_threshold(ticker, regime)
                            will_sell = should_sell(ticker, current_price, ml_signal, T_sell_eff, regime)
                        except Exception as e:
                            print(f"[{ticker}] should_sell 오류: {e}")

                        force_liquidate = (change_ratio <= -0.05) or (change_ratio >= 0.20)
                        if will_sell or force_liquidate:
                            try:
                                coin = ticker.split('-')[1]
                                coin_balance = get_balance(coin)
                            except Exception as e:
                                print(f"[{ticker}] 잔고 확인 에러: {e}")
                                coin_balance = 0

                            if coin_balance and coin_balance > 0:
                                # 부분익절 우선 체크
                                if try_partial_take_profit(ticker, change_ratio, coin_balance, now):
                                    coin_balance = get_balance(coin)

                                reason = []
                                if change_ratio <= -0.05: reason.append("강제 손절(-5% 이하)")
                                if change_ratio >= 0.20: reason.append("강제 익절(+20% 이상)")
                                if will_sell and not reason: reason.append("전략 매도(should_sell=True)")
                                print(f"[{ticker}] 매도 실행: {', '.join(reason)} | 수익률: {change_ratio*100:.2f}%")

                                sold = False
                                for attempt in range(2):
                                    try:
                                        order = sell_crypto_currency(ticker, coin_balance)
                                        if order: sold = True; break
                                        else: print(f"[{ticker}] 매도 실패(시도 {attempt+1}) → 재시도"); time.sleep(1.0)
                                    except Exception as e:
                                        print(f"[{ticker}] 매도 주문 에러({attempt+1}): {e}")
                                        with state_lock:
                                            recent_trades[ticker] = now
                                        time.sleep(1.0)

                                if sold:
                                    time.sleep(0.7)
                                    try:
                                        remain = get_balance(coin)
                                    except Exception as e:
                                        print(f"[{ticker}] 매도 후 잔고 확인 실패: {e}")
                                        remain = None

                                    if remain is None or remain < 1e-8:
                                        with state_lock:
                                            entry_prices.pop(ticker, None)
                                            highest_prices.pop(ticker, None)
                                            recent_trades[ticker] = now
                                            pos_plan.pop(ticker, None)
                                        print(f"[{ticker}] ✅ 매도 완료 및 상태 정리")
                                        record_trade_pnl(change_ratio)
                                    else:
                                        print(f"[{ticker}] ⚠️ 잔여 수량 감지({remain}). 다음 루프에서 재처리")
                                else:
                                    with state_lock:
                                        recent_trades[ticker] = now
                                    print(f"[{ticker}] ❌ 매도 실패: 주문 미체결")
                            else:
                                print(f"[{ticker}] 매도 불가: 보유 수량 없음/조회 실패")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러: {e}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")
