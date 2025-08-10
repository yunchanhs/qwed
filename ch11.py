import time
import pyupbit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import threading
import logging
import math
import gc
from logging.handlers import RotatingFileHandler

# === 안전 OHLCV 호출 & DF 검증 ===
def safe_get_ohlcv(ticker, interval="minute5", count=200, max_retries=5, base_sleep=0.7):
    """
    pyupbit.get_ohlcv()를 안전하게 호출:
      - 재시도(backoff)
      - 빈/이상 데이터 필터
      - 예외 처리
    실패 시 None 반환
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
            if df is not None and not df.empty and all(c in df.columns for c in ["open","high","low","close","volume"]):
                return df
            else:
                print(f"[safe_get_ohlcv] 빈 DF 또는 컬럼 부족: {ticker} {interval} (시도 {attempt}/{max_retries})")
        except Exception as e:
            print(f"[safe_get_ohlcv] 예외: {ticker} {interval} (시도 {attempt}/{max_retries}) → {e}")
        time.sleep(base_sleep * attempt)  # 점증 대기
    return None

def is_valid_df(df, min_len=5):
    return df is not None and not df.empty and len(df) >= min_len and all(
        c in df.columns for c in ["open","high","low","close","volume"]
    )


# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 모델 학습 주기 관련 변수
last_trained_time = None  # 마지막 학습 시간
TRAINING_INTERVAL = timedelta(hours=8)  # 6시간마다 재학습

# 매매 전략 관련 임계값
ML_THRESHOLD = 0.5
ML_SELL_THRESHOLD = 0.3  # AI 신호 매도 기준
STOP_LOSS_THRESHOLD = -0.05  # 손절 (-5%)
TAKE_PROFIT_THRESHOLD = 0.1  # 익절 (10%)
COOLDOWN_TIME = timedelta(minutes=30)  # 동일 코인 재거래 쿨다운 시간
SURGE_COOLDOWN_TIME = timedelta(minutes=60) # 급등 코인 쿨다운 시간

# 계좌 정보 저장
entry_prices = {}            # 매수한 가격 저장
highest_prices = {}          # 매수 후 최고 가격 저장
recent_trades = {}           # ✅ 최근 거래 기록 ← 이게 꼭 있어야 해!
recent_surge_tickers = {}    # 최근 급상승 감지 코인 저장

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

def execute_sell_and_cleanup(ticker, amount, now):
    """매도 체결 + 상태 정리(예외/실패 포함)"""
    try:
        if amount is None or amount <= 0 or math.isnan(amount):
            log.warning(f"[{ticker}] 매도 수량 이상: amount={amount}")
            return False

        order = sell_crypto_currency(ticker, amount)
        if not order:
            log.warning(f"[{ticker}] 매도 주문 실패(첫 시도). 2초 후 재시도")
            safe_sleep(2)
            order = sell_crypto_currency(ticker, amount)
            if not order:
                log.error(f"[{ticker}] 매도 주문 2회 실패 → 상태만 정리하고 패스")
                # 실패해도 상태는 정리(루프가 막히지 않게)
                entry_prices.pop(ticker, None)
                highest_prices.pop(ticker, None)
                recent_trades[ticker] = now
                return False

        log.info(f"[{ticker}] 매도 체결: {amount}")
        # 체결 성공 → 상태 정리
        entry_prices.pop(ticker, None)
        highest_prices.pop(ticker, None)
        recent_trades[ticker] = now
        return True

    except Exception as e:
        log.exception(f"[{ticker}] 매도 중 예외: {e}")
        # 예외가 나도 상태 정리 (루프 무한 보유 방지)
        entry_prices.pop(ticker, None)
        highest_prices.pop(ticker, None)
        recent_trades[ticker] = now
        return False

def load_pickle(filename, default_value):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning(f"[로드 실패] {filename}: {e}")
    return default_value

# ✅  로딩 (여기가 최고 위치!)
entry_prices = load_pickle("entry_prices.pkl", {})
recent_trades = load_pickle("recent_trades.pkl", {})
highest_prices = load_pickle("highest_prices.pkl", {})

def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic

def auto_save_state(interval=300):
    while True:
        try:
            atomic_save(entry_prices, "entry_prices.pkl")
            atomic_save(recent_trades, "recent_trades.pkl")
            atomic_save(highest_prices, "highest_prices.pkl")
            log.info("[백업] 상태 자동 저장 완료")
        except Exception as e:
            log.exception(f"[백업 오류] 상태 저장 실패: {e}")
        time.sleep(interval)

def get_top_tickers(n=40):
    """
    최근 3일 평균 거래대금 기준 + 급등 초기 필터 보완
    ① 거래대금 = 거래량 * 종가
    ② 가격 기준으로 비정상 데이터 보정
    ③ 급등 초기 코인도 일부 포함
    """
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
    top_tickers = [ticker for ticker, _ in sorted_scores[:n]]
    return top_tickers

def detect_surge_tickers(threshold=0.03):
    """실시간 급상승 코인 감지 (1분봉 5개)"""
    tickers = pyupbit.get_tickers(fiat="KRW")
    surge_tickers = []
    for ticker in tickers:
        df = safe_get_ohlcv(ticker, interval="minute1", count=5)
        if not is_valid_df(df, min_len=5):
            continue
        try:
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            if price_change >= threshold:
                surge_tickers.append(ticker)
        except Exception as e:
            print(f"[detect_surge_tickers] 계산 오류: {ticker} → {e}")
            continue
    return surge_tickers


def get_ohlcv_cached(ticker, interval="minute60"):
    time.sleep(0.5)  # 요청 간격 조절
    return pyupbit.get_ohlcv(ticker, interval=interval)
    
# 머신러닝 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.activation = nn.Sigmoid()  # 🔁 출력값 0~1로 제한

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.fc(x[:, -1, :])
        x = self.activation(x)  # ✅ Sigmoid 활성화 함수 적용
        return x

# 지표 계산 함수 (생략, 기존 코드 동일)
# get_macd, get_rsi, get_adx, get_atr, get_features

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
        return pd.DataFrame()  # 빈 DF 반환해서 상위 로직이 건너뛰게

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

# 거래 관련 함수 (생략, 기존 코드 동일)
# get_balance, buy_crypto_currency, sell_crypto_currency

# Upbit 객체 전역 선언 (한 번만 생성)
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
    """시장가로 코인 매수"""
    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        order = upbit.buy_market_order(ticker, amount)
        return order
    except Exception as e:
        print(f"[{ticker}] 매수 중 에러 발생: {e}")
        return None

def sell_crypto_currency(ticker, amount):
    """시장가로 코인 매도"""
    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        order = upbit.sell_market_order(ticker, amount)
        return order
    except Exception as e:
        print(f"[{ticker}] 매도 중 에러 발생: {e}")
        return None

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
    print(f"모델 학습 시작: {ticker}")
    input_dim = 6
    d_model = 32
    num_heads = 4
    num_layers = 1
    output_dim = 1

    model = TransformerModel(input_dim, d_model, num_heads, num_layers, output_dim)
    data = get_features(ticker, normalize=True)

    if data is None or data.empty:
        print(f"경고: {ticker}의 데이터가 비어 있음. 모델 학습을 건너뜁니다.")
        return None

    seq_len = 30
    dataset = TradingDataset(data, seq_len)

    if len(dataset) == 0:
        print(f"경고: {ticker}의 데이터셋이 너무 작아서 학습을 진행할 수 없음.")
        return None

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(-1), y_batch.view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    print(f"모델 학습 완료: {ticker}")
    return model

def should_retrain(ticker, last_trained_time_dict, model, min_performance=1.05):
    now = datetime.now()
    last_time = last_trained_time_dict.get(ticker, datetime.min)

    if now - last_time > TRAINING_INTERVAL:
        perf = backtest(ticker, model)
        if perf < min_performance:
            print(f"[{ticker}] 모델 재학습 필요 (성과: {perf:.2f})")
            return True
    return False
    
def get_ml_signal(ticker, model):
    """AI 신호 계산"""
    try:
        features = get_features(ticker)
        latest_data = features[['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].tail(30)
        X_latest = torch.tensor(latest_data.values, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            prediction = model(X_latest).item()
        return prediction
    except Exception as e:
        print(f"[{ticker}] AI 신호 계산 에러: {e}")
        return 0

def should_sell(ticker, current_price, ml_signal):
    if ticker not in entry_prices:
        return False

    entry_price = entry_prices[ticker]
    highest_prices[ticker] = max(highest_prices.get(ticker, entry_price), current_price)

    change_ratio = (current_price - entry_price) / entry_price  # 총 수익률
    peak_drop = (highest_prices[ticker] - current_price) / highest_prices[ticker]  # 최고점 대비 하락률

    # 🚨 1. 손절 조건 (절대 -5% 손실)
    if change_ratio < -0.05:
        print(f"[{ticker}] 🚨 -5% 손절 발동")
        return True

    # ✅ 2. 다단계 익절 조건
    if change_ratio >= 0.2:
        print(f"[{ticker}] 🎯 20% 이상 수익 → 무조건 익절")
        return True
    elif change_ratio >= 0.15:
        if ml_signal < 0.6:
            print(f"[{ticker}] ✅ 15% 수익 + AI 약함 → 익절")
            return True
        else:
            print(f"[{ticker}] ✅ 15% 수익 + AI 강함 → 보유")
            return False
    elif change_ratio >= 0.10:
        if ml_signal < 0.5:
            print(f"[{ticker}] ✅ 10% 수익 + AI 약함 → 익절")
            return True

    # 📉 3. 트레일링 스탑 (고점 대비 2.5% 하락 + AI 약함)
    if peak_drop > 0.025 and ml_signal < 0.5:
        print(f"[{ticker}] 📉 트레일링 스탑 발동! 고점 대비 하락률: {peak_drop*100:.2f}%")
        return True

    # 📈 4. 추세 유지 조건 (수익 + AI 강함 + MACD 상승)
    if change_ratio > 0.05 and ml_signal > 0.6:
        try:
            df_m5 = safe_get_ohlcv(ticker, interval="minute5", count=200)
            if is_valid_df(df_m5, min_len=50):
                df_m5 = get_macd_from_df(df_m5)
                macd = df_m5['macd'].iloc[-1]
                signal = df_m5['signal'].iloc[-1]
                if macd > signal:
                    print(f"[{ticker}] 📈 추세 지속 (MACD 상승) → 보유")
                    return False
        except Exception as e:
            print(f"[{ticker}] MACD 계산 오류: {e}")

    # 🧪 5. 보조 지표: RSI 과매수 + AI 약함 → 매도 고려
    try:
        df_m5b = safe_get_ohlcv(ticker, interval="minute5", count=200)
        if is_valid_df(df_m5b, min_len=50):
            df_m5b = get_macd_from_df(df_m5b)
            df_m5b = get_rsi_from_df(df_m5b)

            rsi = df_m5b['rsi'].iloc[-1]
            macd = df_m5b['macd'].iloc[-1]
            signal = df_m5b['signal'].iloc[-1]

            if rsi > 80 and ml_signal < 0.5:
                print(f"[{ticker}] RSI 과매수 + AI 약함 → 매도")
                return True

            if macd < signal and ml_signal < 0.5:
                print(f"[{ticker}] MACD 데드크로스 + AI 약함 → 매도")
                return True
    except Exception as e:
        print(f"[{ticker}] RSI/MACD 보조 지표 오류: {e}")

    return False

def backtest(ticker, model, initial_balance=1_000_000, fee=0.0005):
    """과거 데이터로 백테스트 실행"""
    data = get_features(ticker)
    balance = initial_balance
    position = 0
    entry_price = 0

    highest_price = 0  # 백테스트용 개별 최고가 추적

    for i in range(50, len(data) - 1):
        # 입력 데이터 준비
        x_input = torch.tensor(
            data.iloc[i-30:i][['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values,
            dtype=torch.float32
        ).unsqueeze(0)

        ml_signal = model(x_input).item()
        current_price = data.iloc[i]['close']

        # 매수 조건
        if position == 0 and ml_signal > ML_THRESHOLD:
            position = balance / current_price
            entry_price = current_price
            highest_price = entry_price  # 매수 시 최고가 초기화
            balance = 0
            # print(f"[{ticker}] 🟢 매수 @ {current_price:.2f}, ML: {ml_signal:.4f}")

        # 매도 조건
        elif position > 0:
            highest_price = max(highest_price, current_price)

            # peak_drop 계산 및 손절/익절 조건 판단
            peak_drop = (highest_price - current_price) / highest_price
            unrealized_profit = (current_price - entry_price) / entry_price

            # 손절 조건 (즉시 매도)
            if unrealized_profit < STOP_LOSS_THRESHOLD:
                balance = position * current_price * (1 - fee)
                position = 0
                # print(f"[{ticker}] 🔻 손절 @ {current_price:.2f}")
                continue

            # 익절 조건 + AI 신호 반영
            if peak_drop > 0.02 and ml_signal < ML_SELL_THRESHOLD:
                balance = position * current_price * (1 - fee)
                position = 0
                # print(f"[{ticker}] ✅ 익절 @ {current_price:.2f}, ML: {ml_signal:.4f}")
                continue

    # 포지션 종료 없이 끝났다면 현재가 기준 정산
    final_value = balance + (position * data.iloc[-1]['close'])
    return final_value / initial_balance

# === [3] 자동 저장 함수 정의 (여기!) ===
def auto_save_state(interval=300):
    while True:
        try:
            with open("entry_prices.pkl", "wb") as f:
                pickle.dump(entry_prices, f)
            with open("recent_trades.pkl", "wb") as f:
                pickle.dump(recent_trades, f)
            with open("highest_prices.pkl", "wb") as f:
                pickle.dump(highest_prices, f)
            print("[백업] 상태 자동 저장 완료")
        except Exception as e:
            print(f"[백업 오류] 상태 저장 실패: {e}")
        time.sleep(interval)

# 자동 저장 쓰레드 실행
save_thread = threading.Thread(target=auto_save_state, daemon=True)
save_thread.start()
    
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    tickers = pyupbit.get_tickers(fiat="KRW")
    models = {}

    # 초기 설정
    top_tickers = get_top_tickers(n=40)
    print(f"거래량 상위 코인: {top_tickers}")

    for ticker in top_tickers:
        model = train_transformer_model(ticker)
        if model is None:
            continue
        performance = backtest(ticker, model)
        if performance > 1.05:
            models[ticker] = model
            print(f"[{ticker}] 모델 유지 (백테스트 성과: {performance:.2f}배)")
        else:
            print(f"[{ticker}] 모델 제외 (백테스트 성과 부족: {performance:.2f}배)")

    recent_surge_tickers = {}

    try:
        while True:
            now = datetime.now()

            # ✅ 1. 상위 코인 업데이트
            if now.hour % 6 == 0 and now.minute == 0:
                top_tickers = get_top_tickers(n=40)
                print(f"[{now}] 상위 코인 업데이트: {top_tickers}")

                for ticker in top_tickers:
                    model = models.get(ticker)
                    if model is None or should_retrain(ticker, recent_trades, model):
                        model = train_transformer_model(ticker)
                        if model is None:
                            continue
                        performance = backtest(ticker, model)
                        if performance >= 1.05:
                            models[ticker] = model
                            print(f"[{ticker}] 모델 추가/갱신 (성과: {performance:.2f}배)")
                        else:
                            print(f"[{ticker}] 모델 제외 (성과 부족: {performance:.2f}배)")

            # ✅ 2. 급상승 감지
            surge_tickers = detect_surge_tickers(threshold=0.03)
            for ticker in surge_tickers:
                if ticker not in recent_surge_tickers:
                    print(f"[{now}] 급상승 감지: {ticker}")
                    recent_surge_tickers[ticker] = now

                    if ticker not in models:
                        model = train_transformer_model(ticker, epochs=10)
                        if model is None:
                            continue
                        performance = backtest(ticker, model)
                        if performance > 1.1:
                            models[ticker] = model
                            print(f"[{ticker}] 급상승 모델 추가 (백테스트 성과: {performance:.2f}배)")
                        else:
                            print(f"[{ticker}] 급상승 모델 제외 (백테스트 성과 부족: {performance:.2f}배)")

            # ✅ 3. 매수/매도 대상 선정
            target_tickers = set(top_tickers) | set(recent_surge_tickers.keys())

            for ticker in target_tickers:
                cooldown_limit = SURGE_COOLDOWN_TIME if ticker in recent_surge_tickers else COOLDOWN_TIME
                last_trade_time = recent_trades.get(ticker, datetime.min)

                if now - last_trade_time < cooldown_limit:
                    continue

                try:
                    if ticker not in models:
                        print(f"[{ticker}] 모델이 존재하지 않아 신호 계산을 건너뜁니다.")
                        continue

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
                    atr = df['atr'].iloc[-1]
                    current_price = df['close'].iloc[-1]

                    ml_signal = get_ml_signal(ticker, models[ticker])

                    print(f"[DEBUG] {ticker} 매수 조건 검사")
                    print(f" - ML 신호: {ml_signal:.4f}")
                    print(f" - MACD: {macd:.4f}, Signal: {signal:.4f}")
                    print(f" - RSI: {rsi:.2f}")
                    print(f" - ADX: {adx:.2f}")
                    print(f" - ATR: {atr:.6f}")
                    print(f" - 현재 가격: {current_price:.2f}")

                    ATR_THRESHOLD = 0.015

                    # === 매수 조건 ===
                    if isinstance(ml_signal, (int, float)) and 0 <= ml_signal <= 1:
                        if ml_signal > ML_THRESHOLD and macd > signal and rsi < 50 and adx > 20 and atr > ATR_THRESHOLD:
                            krw_balance = get_balance("KRW")
                            print(f"[DEBUG] 보유 원화 잔고: {krw_balance:.2f}")
                            if krw_balance > 5000:
                                buy_amount = krw_balance * 0.3
                                buy_result = buy_crypto_currency(ticker, buy_amount)
                                if buy_result:
                                    entry_prices[ticker] = current_price
                                    highest_prices[ticker] = current_price
                                    recent_trades[ticker] = now
                                    print(f"[{ticker}] 매수 완료: {buy_amount:.2f}원, 가격: {current_price:.2f}")
                                else:
                                    print(f"[{ticker}] 매수 요청 실패")
                            else:
                                print(f"[{ticker}] 매수 불가 (원화 부족)")
                        else:
                            print(f"[{ticker}] 매수 조건 불충족")

                    # === 매도 조건 ===
                    if ticker in entry_prices:
                        entry_price = entry_prices[ticker]
                        highest_prices[ticker] = max(highest_prices.get(ticker, entry_price), current_price)

                        if entry_price == 0:
                            print(f"[{ticker}] 경고: entry_price가 0입니다. 매도 판단 건너뜀")
                            continue

                        change_ratio = (current_price - entry_price) / entry_price
                        peak_drop = (highest_prices[ticker] - current_price) / max(highest_prices[ticker], 1e-9)

                        will_sell = False
                        try:
                            will_sell = should_sell(ticker, current_price, ml_signal)
                        except Exception as e:
                            print(f"[{ticker}] should_sell 평가 오류: {e}")

                        force_liquidate = (change_ratio <= -0.05) or (change_ratio >= 0.20)

                        if will_sell or force_liquidate:
                            try:
                                coin = ticker.split('-')[1]
                                coin_balance = get_balance(coin)
                            except Exception as e:
                                print(f"[{ticker}] 잔고 확인 에러: {e}")
                                coin_balance = 0

                            if coin_balance and coin_balance > 0:
                                reason = []
                                if change_ratio <= -0.05:
                                    reason.append("강제 손절(-5% 이하)")
                                if change_ratio >= 0.20:
                                    reason.append("강제 익절(+20% 이상)")
                                if will_sell and not reason:
                                    reason.append("전략 매도(should_sell=True)")

                                print(f"[{ticker}] 매도 실행: {', '.join(reason)} | 수익률: {change_ratio*100:.2f}% | 고점대비하락: {peak_drop*100:.2f}%")

                                sold = False
                                for attempt in range(2):
                                    try:
                                        order = sell_crypto_currency(ticker, coin_balance)
                                        if order:
                                            sold = True
                                            break
                                        else:
                                            print(f"[{ticker}] 매도 주문 실패(시도 {attempt+1}) → 재시도")
                                            time.sleep(1.0)
                                    except Exception as e:
                                        print(f"[{ticker}] 매도 주문 에러(시도 {attempt+1}): {e}")
                                        time.sleep(1.0)

                                if sold:
                                    time.sleep(0.7)
                                    try:
                                        remain = get_balance(coin)
                                    except Exception as e:
                                        print(f"[{ticker}] 매도 후 잔고 확인 실패: {e}")
                                        remain = None

                                    if remain is None or remain < 1e-8:
                                        entry_prices.pop(ticker, None)
                                        highest_prices.pop(ticker, None)
                                        recent_trades[ticker] = now
                                        print(f"[{ticker}] ✅ 매도 완료 및 상태 정리")
                                    else:
                                        print(f"[{ticker}] ⚠️ 매도 후 잔여 수량 감지({remain}). 다음 루프에서 재처리 예정.")
                                else:
                                    print(f"[{ticker}] ❌ 매도 실패: 주문 체결 안됨")
                            else:
                                print(f"[{ticker}] 매도 불가: 보유 수량 없음 또는 조회 실패")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")

