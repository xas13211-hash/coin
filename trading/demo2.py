"""
demo2.py - 임계값 민감도 분석.
"""
import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trading.backtest import run_backtest, BacktestConfig, print_report
from trading import regime as regime_module

# 동일한 데이터 재생성
np.random.seed(7)
n = 60 * 24 * 4
dates = pd.date_range('2026-01-01', periods=n, freq='15min')
trend = np.zeros(n)
for start, length, slope in [
    (500, 800, 0.0008), (1500, 600, -0.0006), (2500, 500, 0.0005),
    (3300, 400, -0.0009), (4000, 700, 0.0007),
]:
    trend[start:start + length] = slope
returns = trend + np.random.normal(0, 0.003, n)
close = 120_000_000 * np.exp(np.cumsum(returns))
high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
open_ = np.r_[close[0], close[:-1]]
volume = np.random.lognormal(10, 0.4, n)
volume[1500:1600] *= 2.5
volume[3300:3400] *= 3.0
df = pd.DataFrame({'open': open_, 'high': high, 'low': low,
                   'close': close, 'volume': volume}, index=dates)


def run_with_threshold(threshold: float, require_mtf: bool = False):
    """임계값을 동적으로 바꿔서 백테스트."""
    # regime별 임계값 monkey-patch
    regime_module.REGIME_THRESHOLDS = {
        'TREND':    {'buy': threshold - 0.5, 'sell': -(threshold - 0.5)},
        'RANGE':    {'buy': threshold, 'sell': -threshold},
        'VOLATILE': {'buy': threshold + 1.5, 'sell': -(threshold + 1.5)},
        'NORMAL':   {'buy': threshold, 'sell': -threshold},
    }
    config = BacktestConfig(
        initial_capital=4_800_000,
        risk_per_trade_pct=0.015,
        require_mtf=require_mtf,
        use_daily_limit=True,
        use_trailing=True,
    )
    return run_backtest(df, config)


# 임계값 4가지 비교
results = {}
for thr, mtf in [(2.5, False), (3.5, False), (3.5, True), (4.5, True)]:
    label = f"thr={thr}, MTF={'ON' if mtf else 'OFF'}"
    print(f"\n>>> 실행 중: {label}")
    results[label] = run_with_threshold(thr, mtf)
    m = results[label]['metrics']
    if m.get('n_trades', 0) > 0:
        print(f"   거래={m['n_trades']}, 수익률={m['total_return_pct']*100:+.2f}%, "
              f"승률={m['win_rate']*100:.1f}%, R:R={m['rr_ratio']:.2f}, "
              f"PF={m['profit_factor']:.2f}, MDD={m['mdd']*100:.1f}%")

# 비교 표
print("\n" + "=" * 90)
print(f"{'설정':<25}{'거래수':>8}{'수익률':>10}{'승률':>10}{'손익비':>10}{'PF':>8}{'MDD':>10}{'샤프':>8}")
print("-" * 90)
for label, r in results.items():
    m = r['metrics']
    if m.get('n_trades', 0) == 0:
        print(f"{label:<25}{'0':>8}{'  - 거래 없음':>50}")
        continue
    print(f"{label:<25}{m['n_trades']:>8}"
          f"{m['total_return_pct']*100:>9.2f}%{m['win_rate']*100:>9.1f}%"
          f"{m['rr_ratio']:>10.2f}{m['profit_factor']:>8.2f}"
          f"{m['mdd']*100:>9.1f}%{m['sharpe']:>8.2f}")

# 시각화: 4개 equity curve 한번에
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
colors = ['blue', 'green', 'orange', 'red']
for (label, r), color in zip(results.items(), colors):
    if r['metrics'].get('n_trades', 0) > 0:
        eq = r['equity_curve']['equity'] / 1e6
        axes[0].plot(eq.index, eq, label=f"{label} (n={r['metrics']['n_trades']})",
                     color=color, linewidth=1.2, alpha=0.85)
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        axes[1].plot(eq.index, dd, color=color, linewidth=1, alpha=0.7, label=label)

axes[0].axhline(4.8, color='gray', linestyle='--', alpha=0.5)
axes[0].set_ylabel('Equity (M KRW)')
axes[0].set_title('Threshold Sensitivity: Equity Curves', fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_ylabel('Drawdown (%)')
axes[1].set_title('Drawdowns')
axes[1].legend(loc='lower left', fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/sensitivity_chart.png', dpi=110, bbox_inches='tight')
print("\n차트 저장: sensitivity_chart.png")