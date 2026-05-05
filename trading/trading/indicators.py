"""
indicators.py
=============
보조지표 계산 함수 모음. 외부 라이브러리 없이 numpy/pandas로만 구현.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- 추세 ----------
def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period).mean()


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52):
    """일목균형표 - 추세, 지지저항, 모멘텀 통합"""
    high, low, close = df['high'], df['low'], df['close']
    tenkan = (high.rolling(conv).max() + low.rolling(conv).min()) / 2
    kijun = (high.rolling(base).max() + low.rolling(base).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(base)
    senkou_b = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2).shift(base)
    chikou = close.shift(-base)
    return tenkan, kijun, senkou_a, senkou_b, chikou


# ---------- 모멘텀 ----------
def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(s: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    r = rsi(s, period)
    rmin = r.rolling(period).min()
    rmax = r.rolling(period).max()
    stoch = 100 * (r - rmin) / (rmax - rmin).replace(0, np.nan)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


# ---------- 변동성 ----------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(s: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid  # 밴드 폭 (스퀴즈 감지용)
    return upper, mid, lower, pct_b, bandwidth


# ---------- 거래량 ----------
def vwap(df: pd.DataFrame) -> pd.Series:
    typ = (df['high'] + df['low'] + df['close']) / 3
    return (typ * df['volume']).cumsum() / df['volume'].cumsum()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()


# ---------- 통합 계산 ----------
def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """모든 보조지표를 추가한 DataFrame 반환."""
    df = df.copy()

    # 추세
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['adx'] = adx(df)
    m, s, h = macd(df['close'])
    df['macd'] = m
    df['macd_sig'] = s
    df['macd_hist'] = h
    tk, kj, sa, sb, ch = ichimoku(df)
    df['tenkan'] = tk
    df['kijun'] = kj
    df['senkou_a'] = sa
    df['senkou_b'] = sb

    # 모멘텀
    df['rsi'] = rsi(df['close'])
    k, d = stoch_rsi(df['close'])
    df['stoch_k'] = k
    df['stoch_d'] = d

    # 변동성
    df['atr'] = atr(df)
    df['atr_ma'] = df['atr'].rolling(50).mean()  # ATR의 평균 (regime 감지용)
    u, mid, lo, pb, bw = bollinger(df['close'])
    df['bb_upper'] = u
    df['bb_mid'] = mid
    df['bb_lower'] = lo
    df['bb_pct'] = pb
    df['bb_width'] = bw

    # 거래량
    df['vwap'] = vwap(df)
    df['obv'] = obv(df)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']

    return df