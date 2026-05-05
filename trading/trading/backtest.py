"""
backtest.py
===========
백테스트 엔진. 실전 환경을 최대한 시뮬레이션:
- Look-ahead bias 방지 (신호는 t-1 봉 종가까지만, 진입은 t 봉 시가)
- 슬리피지 모델링 (시장가 체결 가정)
- 거래 수수료 (편도 0.04% × 2)
- 트레일링 스톱
- 일일 손실 한도
- 정확한 성과 지표 (승률, 손익비, MDD, 샤프, 소르티노, 수익팩터)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from .indicators import compute_all
from .signals import generate_signal, prepare_mtf
from .risk import (
    fixed_fractional_size,
    trailing_stop_atr,
    DailyRiskTracker,
)


# =====================================================================
# Trade 기록
# =====================================================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str               # 'long' or 'short'
    entry_price: float
    exit_price: float
    size_krw: float         # 명목 가치
    pnl_krw: float
    pnl_pct: float
    fees_krw: float
    exit_reason: str        # 'stop', 'tp1', 'tp2', 'trailing', 'timeout'
    bars_held: int


# =====================================================================
# 백테스트 엔진
# =====================================================================
@dataclass
class BacktestConfig:
    initial_capital: float = 4_800_000
    risk_per_trade_pct: float = 0.015
    fee_pct: float = 0.0004        # 편도
    slippage_pct: float = 0.0005   # 시장가 슬리피지 추정
    atr_stop_mult: float = 1.5     # 손절폭 = ATR × 1.5
    tp1_r: float = 1.0             # 1차 익절 = 1R
    tp2_r: float = 2.0             # 2차 익절 = 2R
    tp1_close_pct: float = 0.5     # 1차 익절 시 청산 비율
    use_trailing: bool = True
    trail_activate_r: float = 1.0
    trail_atr_mult: float = 1.0
    max_hold_bars: int = 32        # 최대 보유 봉 수 (15M × 32 = 8시간)
    require_mtf: bool = True
    use_daily_limit: bool = True
    daily_loss_limit_pct: float = 0.03
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 2
    warmup_bars: int = 200         # 지표 안정화 대기


def run_backtest(
    df_15m_raw: pd.DataFrame,
    config: BacktestConfig = None,
) -> dict:
    """
    15분봉 데이터로 백테스트.
    return: trades, equity_curve, metrics
    """
    if config is None:
        config = BacktestConfig()

    # 멀티 타임프레임 데이터 준비 (전체 한 번만 계산)
    df_15m, df_1h, df_4h = prepare_mtf(df_15m_raw)

    capital = config.initial_capital
    equity_curve = []
    trades: list[Trade] = []
    open_position: Optional[dict] = None
    recent_signals: list[str] = []

    daily_tracker = DailyRiskTracker(
        capital=capital,
        daily_loss_limit_pct=config.daily_loss_limit_pct,
        max_trades_per_day=config.max_trades_per_day,
        max_consecutive_losses=config.max_consecutive_losses,
    )

    n = len(df_15m)
    start_idx = max(config.warmup_bars, 50)

    for i in range(start_idx, n - 1):  # n-1: 다음 봉 시가 진입을 위해
        ts = df_15m.index[i]
        next_bar = df_15m.iloc[i + 1]
        current = df_15m.iloc[i]

        # ----- 1. 오픈 포지션 관리 (손절/익절/트레일링) -----
        if open_position is not None:
            pos = open_position
            high = next_bar['high']
            low = next_bar['low']
            atr_now = current['atr']

            exit_price = None
            exit_reason = None

            # 손절 우선 체크 (보수적 가정: 손절이 먼저 닿았다고 가정)
            if pos['side'] == 'long':
                if low <= pos['stop']:
                    exit_price = pos['stop']
                    exit_reason = 'stop'
                elif not pos['tp1_hit'] and high >= pos['tp1']:
                    # 1차 익절: 절반 청산
                    half_size = pos['size_krw'] * config.tp1_close_pct
                    realized_pnl = _calc_pnl(pos['entry'], pos['tp1'], half_size, 'long', config)
                    pos['size_krw'] -= half_size
                    pos['realized_pnl'] = pos.get('realized_pnl', 0) + realized_pnl
                    pos['tp1_hit'] = True
                    pos['stop'] = pos['entry']  # 본전 손절로 이동
                elif pos['tp1_hit'] and high >= pos['tp2']:
                    exit_price = pos['tp2']
                    exit_reason = 'tp2'
                elif config.use_trailing and pos['tp1_hit']:
                    new_stop = trailing_stop_atr(
                        pos['entry'], current['close'], pos['stop'],
                        atr_now, 'long',
                        activate_at_r=config.trail_activate_r,
                        trail_atr_mult=config.trail_atr_mult,
                        initial_risk=pos['initial_risk'],
                    )
                    pos['stop'] = new_stop
            else:  # short
                if high >= pos['stop']:
                    exit_price = pos['stop']
                    exit_reason = 'stop'
                elif not pos['tp1_hit'] and low <= pos['tp1']:
                    half_size = pos['size_krw'] * config.tp1_close_pct
                    realized_pnl = _calc_pnl(pos['entry'], pos['tp1'], half_size, 'short', config)
                    pos['size_krw'] -= half_size
                    pos['realized_pnl'] = pos.get('realized_pnl', 0) + realized_pnl
                    pos['tp1_hit'] = True
                    pos['stop'] = pos['entry']
                elif pos['tp1_hit'] and low <= pos['tp2']:
                    exit_price = pos['tp2']
                    exit_reason = 'tp2'
                elif config.use_trailing and pos['tp1_hit']:
                    new_stop = trailing_stop_atr(
                        pos['entry'], current['close'], pos['stop'],
                        atr_now, 'short',
                        activate_at_r=config.trail_activate_r,
                        trail_atr_mult=config.trail_atr_mult,
                        initial_risk=pos['initial_risk'],
                    )
                    pos['stop'] = new_stop

            # 시간 손절
            pos['bars_held'] += 1
            if exit_price is None and pos['bars_held'] >= config.max_hold_bars:
                exit_price = next_bar['open']
                exit_reason = 'timeout'

            # 청산 처리
            if exit_price is not None:
                pnl = _calc_pnl(pos['entry'], exit_price, pos['size_krw'], pos['side'], config)
                total_pnl = pnl + pos.get('realized_pnl', 0)
                fees = (pos['initial_size'] + pos['size_krw']) * config.fee_pct  # 진입+청산
                capital += total_pnl
                trade = Trade(
                    entry_time=pos['entry_time'],
                    exit_time=ts,
                    side=pos['side'],
                    entry_price=pos['entry'],
                    exit_price=exit_price,
                    size_krw=pos['initial_size'],
                    pnl_krw=total_pnl,
                    pnl_pct=total_pnl / pos['initial_size'],
                    fees_krw=fees,
                    exit_reason=exit_reason,
                    bars_held=pos['bars_held'],
                )
                trades.append(trade)
                daily_tracker.record_trade(ts, total_pnl)
                open_position = None

        # ----- 2. 신규 진입 검토 -----
        if open_position is None:
            can_trade, reason = daily_tracker.can_trade(ts)
            if can_trade:
                # 신호 생성 (look-ahead 방지: 현재 봉 종가까지 사용)
                sig = generate_signal(
                    df_base=df_15m,
                    df_15m=df_15m,
                    df_1h=df_1h,
                    df_4h=df_4h,
                    idx=i,
                    require_mtf=config.require_mtf,
                    recent_signals=recent_signals,
                )
                recent_signals.append(sig.signal)
                if len(recent_signals) > 20:
                    recent_signals.pop(0)

                if sig.signal in ('BUY', 'SELL'):
                    side = 'long' if sig.signal == 'BUY' else 'short'
                    # 진입가는 다음 봉 시가 + 슬리피지
                    raw_entry = next_bar['open']
                    if side == 'long':
                        entry = raw_entry * (1 + config.slippage_pct)
                        stop_dist = sig.atr * config.atr_stop_mult
                        stop = entry - stop_dist
                        tp1 = entry + stop_dist * config.tp1_r
                        tp2 = entry + stop_dist * config.tp2_r
                    else:
                        entry = raw_entry * (1 - config.slippage_pct)
                        stop_dist = sig.atr * config.atr_stop_mult
                        stop = entry + stop_dist
                        tp1 = entry - stop_dist * config.tp1_r
                        tp2 = entry - stop_dist * config.tp2_r

                    size = fixed_fractional_size(
                        capital, config.risk_per_trade_pct,
                        entry, stop, config.fee_pct,
                    )
                    if size > 0:
                        open_position = {
                            'entry_time': ts,
                            'side': side,
                            'entry': entry,
                            'stop': stop,
                            'tp1': tp1,
                            'tp2': tp2,
                            'size_krw': size,
                            'initial_size': size,
                            'initial_risk': stop_dist,
                            'tp1_hit': False,
                            'bars_held': 0,
                            'realized_pnl': 0.0,
                        }

        # ----- 3. equity curve 기록 -----
        unrealized = 0.0
        if open_position is not None:
            unrealized = _calc_pnl(
                open_position['entry'], current['close'],
                open_position['size_krw'], open_position['side'], config,
                include_fees=False,
            )
            unrealized += open_position.get('realized_pnl', 0)
        equity_curve.append({
            'time': ts,
            'equity': capital + unrealized,
        })

    equity_df = pd.DataFrame(equity_curve).set_index('time')
    metrics = calculate_metrics(trades, equity_df, config)

    return {
        'trades': trades,
        'equity_curve': equity_df,
        'metrics': metrics,
        'final_capital': capital,
    }


def _calc_pnl(entry, exit_price, size_krw, side, config, include_fees=True):
    """size_krw: 명목 가치. PnL을 원화로 반환."""
    if side == 'long':
        gross_pct = (exit_price - entry) / entry
    else:
        gross_pct = (entry - exit_price) / entry
    pnl = size_krw * gross_pct
    if include_fees:
        # 수수료는 양쪽 (편도 × 2). 슬리피지는 entry/exit 가격에 이미 반영됨
        pnl -= size_krw * config.fee_pct * 2
    return pnl


# =====================================================================
# 성과 지표
# =====================================================================
def calculate_metrics(trades: list[Trade], equity: pd.DataFrame,
                      config: BacktestConfig) -> dict:
    if not trades:
        return {'n_trades': 0, 'note': '거래 없음'}

    pnls = np.array([t.pnl_krw for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_return_pct = (equity['equity'].iloc[-1] - config.initial_capital) / config.initial_capital
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf

    # 일별 수익률로 샤프/소르티노
    daily = equity['equity'].resample('1D').last().pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
    downside = daily[daily < 0]
    sortino = (daily.mean() / downside.std()) * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0

    # MDD
    eq = equity['equity']
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    mdd = drawdown.min()

    # 평균 보유 시간
    avg_bars = np.mean([t.bars_held for t in trades])

    # 청산 사유 분포
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    return {
        'n_trades': len(trades),
        'total_return_pct': total_return_pct,
        'final_capital': equity['equity'].iloc[-1],
        'win_rate': win_rate,
        'avg_win_krw': avg_win,
        'avg_loss_krw': avg_loss,
        'avg_win_pct': wins.mean() / config.initial_capital if len(wins) > 0 else 0,
        'avg_loss_pct': losses.mean() / config.initial_capital if len(losses) > 0 else 0,
        'rr_ratio': rr,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'sortino': sortino,
        'mdd': mdd,
        'avg_bars_held': avg_bars,
        'exit_reasons': reasons,
        'best_trade': pnls.max(),
        'worst_trade': pnls.min(),
    }


def print_report(result: dict):
    """성과 리포트 출력."""
    m = result['metrics']
    if m.get('n_trades', 0) == 0:
        print("거래가 발생하지 않았습니다.")
        return

    print("=" * 60)
    print("📊 백테스트 결과")
    print("=" * 60)
    print(f"총 거래 수      : {m['n_trades']}")
    print(f"최종 자본       : {m['final_capital']:>15,.0f} 원")
    print(f"총 수익률       : {m['total_return_pct']*100:>14.2f} %")
    print(f"승률            : {m['win_rate']*100:>14.2f} %")
    print(f"평균 수익(승)   : {m['avg_win_krw']:>15,.0f} 원")
    print(f"평균 손실(패)   : {m['avg_loss_krw']:>15,.0f} 원")
    print(f"손익비 (R:R)    : {m['rr_ratio']:>14.2f}")
    print(f"수익팩터        : {m['profit_factor']:>14.2f}  (>1.5 양호)")
    print(f"샤프 비율       : {m['sharpe']:>14.2f}  (>1.0 양호)")
    print(f"소르티노 비율   : {m['sortino']:>14.2f}")
    print(f"최대낙폭 (MDD)  : {m['mdd']*100:>14.2f} %")
    print(f"평균 보유 봉수  : {m['avg_bars_held']:>14.1f}")
    print(f"최고 거래       : {m['best_trade']:>15,.0f} 원")
    print(f"최악 거래       : {m['worst_trade']:>15,.0f} 원")
    print(f"청산 사유 분포  : {m['exit_reasons']}")
    print("=" * 60)
