"""
signals.py
==========
개선된 매매 신호 생성기.
 
주요 개선:
1. Look-ahead bias 제거 (직전 봉 종가까지만 사용)
2. 멀티 타임프레임 정합성 (15M / 1H / 4H)
3. 시장 국면별 동적 가중치 + 임계값
4. Whipsaw 필터 (직전 N봉 반대 신호 회피)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
 
from .indicators import compute_all
from . import regime as _regime
from .regime import detect_regime
 
 
# =====================================================================
# 카테고리별 점수
# =====================================================================
def _score_trend(row: pd.Series) -> float:
    s = 0
    if row['ema20'] > row['ema50'] > row['ema200']:
        s += 1
    elif row['ema20'] < row['ema50'] < row['ema200']:
        s -= 1
    s += 1 if row['close'] > row['ema50'] else -1
    s += 1 if row['macd_hist'] > 0 else -1
    # Ichimoku 구름 위/아래
    cloud_top = max(row['senkou_a'], row['senkou_b']) if pd.notna(row['senkou_a']) else None
    cloud_bot = min(row['senkou_a'], row['senkou_b']) if pd.notna(row['senkou_a']) else None
    if cloud_top is not None:
        if row['close'] > cloud_top:
            s += 1
        elif row['close'] < cloud_bot:
            s -= 1
    return s
 
 
def _score_momentum(row: pd.Series) -> float:
    s = 0
    if row['rsi'] < 30:
        s += 1
    elif row['rsi'] > 70:
        s -= 1
    elif 30 <= row['rsi'] < 50:
        s += 0.5
    elif 50 < row['rsi'] <= 70:
        s -= 0.5
    if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 80:
        s += 1
    elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 20:
        s -= 1
    s += 1 if row['macd'] > row['macd_sig'] else -1
    return s
 
 
def _score_volatility(row: pd.Series) -> float:
    s = 0
    if row['bb_pct'] < 0.2:
        s += 1
    elif row['bb_pct'] > 0.8:
        s -= 1
    if row['adx'] > 25:
        s += 1 if row['close'] > row['ema20'] else -1
    return s
 
 
def _score_volume(row: pd.Series) -> float:
    s = 0
    if row['vol_ratio'] > 1.5:
        s += 1 if row['close'] > row['vwap'] else -1
    s += 1 if row['close'] > row['vwap'] else -1
    return s
 
 
def compute_scores(row: pd.Series) -> dict:
    return {
        'trend': _score_trend(row),
        'momentum': _score_momentum(row),
        'volatility': _score_volatility(row),
        'volume': _score_volume(row),
    }
 
 
def weighted_total(scores: dict, regime: str) -> float:
    w = _regime.REGIME_WEIGHTS[regime]
    return sum(scores[k] * w[k] for k in scores)
 
 
# =====================================================================
# 멀티 타임프레임
# =====================================================================
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """15분봉을 1H, 4H 등으로 리샘플링."""
    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()
 
 
def mtf_trend_alignment(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                         df_4h: pd.DataFrame, ts: pd.Timestamp) -> int:
    """
    멀티 타임프레임 추세 정합성.
    +1: 모든 봉에서 ema20>ema50 (상승), -1: 모두 ema20<ema50 (하락), 0: 혼조
    look-ahead 방지: ts 이전 데이터만 사용.
    """
    def trend(df, ts):
        # searchsorted: O(log N) 인덱스 검색으로 ts 이전의 마지막 봉 찾기
        # 기존 df.loc[df.index < ts]는 매번 O(N) 필터링 → 35,000봉에서 매우 느림
        idx = df.index.searchsorted(ts) - 1
        if idx < 50:
            return 0
        last = df.iloc[idx]
        if last['ema20'] > last['ema50']:
            return 1
        elif last['ema20'] < last['ema50']:
            return -1
        return 0
 
    t15 = trend(df_15m, ts)
    t1h = trend(df_1h, ts)
    t4h = trend(df_4h, ts)
    if t15 == t1h == t4h == 1:
        return 1
    if t15 == t1h == t4h == -1:
        return -1
    return 0
 
 
# =====================================================================
# 메인 신호 생성
# =====================================================================
@dataclass
class Signal:
    timestamp: pd.Timestamp
    signal: str            # 'BUY', 'SELL', 'HOLD'
    price: float           # 진입 기준가 (다음 봉 시가 사용 권장)
    regime: str
    raw_score: float       # 가중치 적용 전
    weighted_score: float  # 가중치 적용 후
    mtf_alignment: int     # -1, 0, 1
    scores: dict           # 카테고리별 점수
    atr: float
    notes: str
 
 
def generate_signal(
    df_base: pd.DataFrame,           # 기본 timeframe (예: 15분봉) - 보조지표 포함
    df_15m: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    idx: int = -1,                   # 어떤 봉 기준으로 신호 낼지 (-1 = 마지막 봉)
    whipsaw_lookback: int = 5,
    require_mtf: bool = True,        # 멀티 타임프레임 정합 필수 여부
    recent_signals: Optional[list] = None,  # whipsaw 검사용 직전 신호들
) -> Signal:
    """
    look-ahead bias 방지 핵심:
    - idx 시점 봉의 '종가'를 기반으로 점수 계산
    - 진입은 idx+1 봉의 시가로 가정 (백테스트 엔진에서 처리)
    """
    last = df_base.iloc[idx]
    ts = df_base.index[idx]
 
    # 시장 국면
    regime = detect_regime(last)
 
    # 카테고리별 점수
    scores = compute_scores(last)
    raw = sum(scores.values())
    weighted = weighted_total(scores, regime)
 
    # MTF 정합성
    mtf = 0
    if df_15m is not None and df_1h is not None and df_4h is not None:
        mtf = mtf_trend_alignment(df_15m, df_1h, df_4h, ts)
 
    # 임계값 (국면별)
    thr = _regime.REGIME_THRESHOLDS[regime]
 
    # 신호 결정
    signal = 'HOLD'
    notes = []
 
    if weighted >= thr['buy']:
        signal = 'BUY'
        notes.append(f"가중점수 {weighted:.1f} ≥ +{thr['buy']} ({regime})")
    elif weighted <= thr['sell']:
        signal = 'SELL'
        notes.append(f"가중점수 {weighted:.1f} ≤ {thr['sell']} ({regime})")
 
    # MTF 필터
    if signal == 'BUY' and require_mtf and mtf != 1:
        signal = 'HOLD'
        notes.append("MTF 정합 실패 (4H/1H/15M 추세 불일치)")
    elif signal == 'SELL' and require_mtf and mtf != -1:
        signal = 'HOLD'
        notes.append("MTF 정합 실패")
 
    # Whipsaw 필터: 직전 N봉 안에 반대 신호가 있었으면 무시
    if recent_signals and signal in ('BUY', 'SELL'):
        opposite = 'SELL' if signal == 'BUY' else 'BUY'
        recent = recent_signals[-whipsaw_lookback:]
        if any(s == opposite for s in recent):
            signal = 'HOLD'
            notes.append(f"Whipsaw 필터: 직전 {whipsaw_lookback}봉 내 반대 신호")
 
    return Signal(
        timestamp=ts,
        signal=signal,
        price=float(last['close']),
        regime=regime,
        raw_score=round(raw, 2),
        weighted_score=round(weighted, 2),
        mtf_alignment=mtf,
        scores=scores,
        atr=float(last['atr']),
        notes=" / ".join(notes) if notes else "관망",
    )
 
 
def prepare_mtf(df_15m_raw: pd.DataFrame):
    """15분봉 원본 데이터로부터 1H, 4H 리샘플링 및 지표 계산."""
    df_15m = compute_all(df_15m_raw)
    df_1h = compute_all(resample_ohlcv(df_15m_raw, '1h'))
    df_4h = compute_all(resample_ohlcv(df_15m_raw, '4h'))
    return df_15m, df_1h, df_4h