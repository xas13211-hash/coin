"""
regime.py
=========
시장 국면(Market Regime) 감지.
국면에 따라 신호 점수의 가중치를 다르게 적용한다.

국면 종류:
- TREND      : 추세장 (ADX > 25)
- RANGE      : 횡보장 (ADX < 20, BB 폭 좁음)
- VOLATILE   : 고변동성 (ATR가 평균 대비 1.5배 이상)
- NORMAL     : 그 외
"""
from __future__ import annotations
import pandas as pd


def detect_regime(row: pd.Series) -> str:
    """단일 봉 row에서 시장 국면 판단."""
    adx_val = row.get('adx', 0)
    atr_val = row.get('atr', 0)
    atr_ma = row.get('atr_ma', atr_val)
    bb_width = row.get('bb_width', 0)

    # 고변동성 우선 (안전 우선)
    if atr_ma and atr_val / atr_ma > 1.5:
        return 'VOLATILE'

    # 추세장
    if adx_val > 25:
        return 'TREND'

    # 횡보장 (ADX 낮고 BB 폭도 좁음)
    if adx_val < 20 and bb_width < 0.04:
        return 'RANGE'

    return 'NORMAL'


# 국면별 카테고리 가중치 (trend, momentum, volatility, volume)
REGIME_WEIGHTS = {
    'TREND':    {'trend': 1.5, 'momentum': 1.0, 'volatility': 0.7, 'volume': 1.2},
    'RANGE':    {'trend': 0.5, 'momentum': 1.2, 'volatility': 1.5, 'volume': 1.0},
    'VOLATILE': {'trend': 0.8, 'momentum': 0.8, 'volatility': 0.8, 'volume': 0.8},
    'NORMAL':   {'trend': 1.0, 'momentum': 1.0, 'volatility': 1.0, 'volume': 1.0},
}

# 국면별 진입 임계값 (변동성 클 땐 더 신중하게)
REGIME_THRESHOLDS = {
    'TREND':    {'buy': 4.5, 'sell': -4.5},
    'RANGE':    {'buy': 5.0, 'sell': -5.0},
    'VOLATILE': {'buy': 6.5, 'sell': -6.5},  # 더 강한 합의 필요
    'NORMAL':   {'buy': 5.0, 'sell': -5.0},
}