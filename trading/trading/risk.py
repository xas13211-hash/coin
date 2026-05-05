"""
risk.py
=======
포지션 사이징 + 트레일링 스톱 + 일일 손실 한도 + 상관관계 조정.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


# =====================================================================
# 1. 포지션 사이징
# =====================================================================
def fixed_fractional_size(
    capital: float,
    risk_pct: float,
    entry: float,
    stop: float,
    fee_pct: float = 0.0004,
) -> float:
    """고정 비율 포지션 사이징. 가장 안전한 기본값."""
    stop_dist_pct = abs(entry - stop) / entry
    effective_risk = stop_dist_pct + fee_pct * 2
    if effective_risk <= 0:
        return 0.0
    risk_amount = capital * risk_pct
    return min(risk_amount / effective_risk, capital)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   fraction: float = 0.25) -> float:
    """
    Kelly Criterion (Fractional).
    win_rate: 0~1, avg_win/avg_loss: 평균 수익/손실 비율 (양수)
    fraction: Kelly 결과의 몇 %만 사용할지 (보수적으로 25%)
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    k = (b * p - q) / b
    return max(0.0, min(k * fraction, 0.05))  # 최대 5%까지만 (방어선)


def kelly_position_size(
    capital: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    entry: float,
    stop: float,
    fee_pct: float = 0.0004,
    kelly_frac: float = 0.25,
) -> float:
    """Kelly 기반 포지션 크기. 백테스트로 측정한 통계가 있을 때 사용."""
    risk_pct = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_frac)
    if risk_pct <= 0:
        return 0.0
    return fixed_fractional_size(capital, risk_pct, entry, stop, fee_pct)


# =====================================================================
# 2. 상관관계 조정 (BTC + ETH 동시 보유 시)
# =====================================================================
def correlation_adjustment(
    df_a: pd.DataFrame, df_b: pd.DataFrame, period: int = 100
) -> float:
    """
    두 자산의 최근 N봉 수익률 상관계수 계산.
    상관계수가 0.8 이상이면 둘 중 하나의 포지션을 50%로 줄여야 함.
    """
    ra = df_a['close'].pct_change().tail(period)
    rb = df_b['close'].pct_change().tail(period)
    if len(ra) < 30 or len(rb) < 30:
        return 0.0
    common = ra.index.intersection(rb.index)
    if len(common) < 30:
        return 0.0
    return float(ra.loc[common].corr(rb.loc[common]))


def size_with_correlation(
    base_size: float, correlation: float, holding_other: bool
) -> float:
    """
    이미 다른 종목 포지션이 있을 때 사이즈를 줄임.
    상관계수 0.9 → 50%, 0.8 → 65%, 0.7 → 80%, 0.5 이하 → 100%
    """
    if not holding_other:
        return base_size
    if correlation >= 0.9:
        return base_size * 0.5
    if correlation >= 0.8:
        return base_size * 0.65
    if correlation >= 0.7:
        return base_size * 0.8
    return base_size


# =====================================================================
# 3. 트레일링 스톱
# =====================================================================
def trailing_stop_atr(
    entry: float,
    current_price: float,
    current_stop: float,
    atr_now: float,
    side: str,                # 'long' or 'short'
    activate_at_r: float = 1.0,    # 손익비 1:1 도달 시 활성화
    trail_atr_mult: float = 1.0,   # 가격 - ATR×1 만큼 따라감
    initial_risk: float = None,    # 진입가-최초손절가 거리
) -> float:
    """
    ATR 기반 트레일링 스톱.
    가격이 충분히 유리해지면 손절선을 자동으로 끌어올림. (한 방향으로만 이동)
    """
    if side == 'long':
        gain = current_price - entry
        if initial_risk and gain >= initial_risk * activate_at_r:
            new_stop = current_price - atr_now * trail_atr_mult
            return max(current_stop, new_stop)  # 한 번 올라간 손절은 절대 내려가지 않음
        return current_stop
    else:  # short
        gain = entry - current_price
        if initial_risk and gain >= initial_risk * activate_at_r:
            new_stop = current_price + atr_now * trail_atr_mult
            return min(current_stop, new_stop)
        return current_stop


# =====================================================================
# 4. 일일 손실 한도 (Circuit Breaker)
# =====================================================================
@dataclass
class DailyRiskTracker:
    """하루 단위 손익을 추적하고 한도 도달 시 거래 차단."""
    capital: float
    daily_loss_limit_pct: float = 0.03   # 자본의 3%
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 2

    current_date: Optional[pd.Timestamp] = None
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    halted: bool = False

    def reset_if_new_day(self, ts: pd.Timestamp):
        d = ts.normalize()
        if self.current_date != d:
            self.current_date = d
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.halted = False
            self.consecutive_losses = 0  # 하루 쉬면 리셋 (실전 트레이딩 룰)

    def can_trade(self, ts: pd.Timestamp) -> tuple[bool, str]:
        self.reset_if_new_day(ts)
        if self.halted:
            return False, "일일 손실 한도 도달"
        if self.daily_trade_count >= self.max_trades_per_day:
            return False, f"일일 매매 횟수 한도 ({self.max_trades_per_day}회)"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"연속 손실 {self.consecutive_losses}회 - 휴식 권고"
        return True, "OK"

    def record_trade(self, ts: pd.Timestamp, pnl: float):
        self.reset_if_new_day(ts)
        self.daily_pnl += pnl
        self.daily_trade_count += 1
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.daily_pnl <= -self.capital * self.daily_loss_limit_pct:
            self.halted = True
