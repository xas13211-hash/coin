"""
demo.py
=======
통합 데모: 가짜 BTC 15분봉 데이터로 백테스트 실행.
"""
import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from trading.backtest import run_backtest, BacktestConfig, print_report


# =====================================================================
# 가짜 BTC 15분봉 데이터 생성 (60일치 = 5760봉)
# =====================================================================
np.random.seed(7)
n = 60 * 24 * 4  # 60일 × 24시간 × 4(15분봉)
dates = pd.date_range('2026-01-01', periods=n, freq='15min')

# 추세가 있는 랜덤워크 (실제 시장 비슷하게)
trend = np.zeros(n)
# 여러 구간에 추세 부여
for start, length, slope in [
    (500, 800, 0.0008),    # 강한 상승
    (1500, 600, -0.0006),  # 하락
    (2500, 500, 0.0005),   # 회복
    (3300, 400, -0.0009),  # 급락
    (4000, 700, 0.0007),   # 재상승
]:
    trend[start:start + length] = slope

noise = np.random.normal(0, 0.003, n)
returns = trend + noise
close = 120_000_000 * np.exp(np.cumsum(returns))

high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
open_ = np.r_[close[0], close[:-1]]
volume = np.random.lognormal(10, 0.4, n)
# 변동성 큰 구간엔 거래량 ↑
volume[1500:1600] *= 2.5
volume[3300:3400] *= 3.0

df = pd.DataFrame({
    'open': open_, 'high': high, 'low': low,
    'close': close, 'volume': volume,
}, index=dates)

print(f"데이터: {len(df)}봉 ({df.index[0]} ~ {df.index[-1]})")
print(f"가격 범위: {df['close'].min()/1e6:.1f}M ~ {df['close'].max()/1e6:.1f}M KRW")
print()

# =====================================================================
# 백테스트 실행 - 두 가지 설정 비교
# =====================================================================

print("【시나리오 1】 보수적 설정 (MTF 필수, 일일 한도 적용)")
print("-" * 60)
config1 = BacktestConfig(
    initial_capital=4_800_000,
    risk_per_trade_pct=0.015,
    require_mtf=True,
    use_daily_limit=True,
    use_trailing=True,
)
result1 = run_backtest(df, config1)
print_report(result1)
print()

print("【시나리오 2】 공격적 설정 (MTF 미적용, 트레일링 OFF)")
print("-" * 60)
config2 = BacktestConfig(
    initial_capital=4_800_000,
    risk_per_trade_pct=0.015,
    require_mtf=False,
    use_daily_limit=True,
    use_trailing=False,
)
result2 = run_backtest(df, config2)
print_report(result2)
print()

# =====================================================================
# 시각화
# =====================================================================
fig = plt.figure(figsize=(15, 11))
gs = GridSpec(4, 1, height_ratios=[2.5, 1.5, 1, 1], hspace=0.4)

# (1) 가격 차트 + 진입/청산 포인트
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['close'] / 1e6, color='black', linewidth=0.6, label='BTC')
for t in result1['trades']:
    color = 'green' if t.pnl_krw > 0 else 'red'
    marker = '^' if t.side == 'long' else 'v'
    ax1.scatter(t.entry_time, t.entry_price / 1e6, marker=marker, color=color,
                s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.scatter(t.exit_time, t.exit_price / 1e6, marker='x', color=color,
                s=40, alpha=0.6)
ax1.set_title('Backtest: Price & Trades (Conservative)',
              fontsize=12, fontweight='bold')
ax1.set_ylabel('Price (M KRW)')
ax1.grid(alpha=0.3)
ax1.legend(loc='upper left')

# (2) Equity Curve 비교
ax2 = fig.add_subplot(gs[1])
ax2.plot(result1['equity_curve'].index, result1['equity_curve']['equity'] / 1e6,
         label=f"Conservative (MTF on) - {result1['metrics']['total_return_pct']*100:+.1f}%",
         color='blue', linewidth=1.2)
ax2.plot(result2['equity_curve'].index, result2['equity_curve']['equity'] / 1e6,
         label=f"Aggressive (MTF off) - {result2['metrics']['total_return_pct']*100:+.1f}%",
         color='orange', linewidth=1.2)
ax2.axhline(4.8, color='gray', linestyle='--', alpha=0.5, label='Initial 4.8M')
ax2.set_ylabel('Equity (M KRW)')
ax2.set_title('Equity Curve Comparison')
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)

# (3) Drawdown
ax3 = fig.add_subplot(gs[2])
for label, result, color in [
    ('Conservative', result1, 'blue'),
    ('Aggressive', result2, 'orange'),
]:
    eq = result['equity_curve']['equity']
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax3.fill_between(eq.index, dd, 0, alpha=0.3, color=color, label=label)
ax3.set_ylabel('Drawdown (%)')
ax3.set_title('Drawdown')
ax3.legend(loc='lower left')
ax3.grid(alpha=0.3)

# (4) PnL 분포
ax4 = fig.add_subplot(gs[3])
pnls1 = [t.pnl_krw / 1000 for t in result1['trades']]
pnls2 = [t.pnl_krw / 1000 for t in result2['trades']]
bins = np.linspace(min(pnls1 + pnls2 + [0]), max(pnls1 + pnls2 + [0]), 30)
ax4.hist(pnls1, bins=bins, alpha=0.5, color='blue', label=f'Conservative (n={len(pnls1)})')
ax4.hist(pnls2, bins=bins, alpha=0.5, color='orange', label=f'Aggressive (n={len(pnls2)})')
ax4.axvline(0, color='black', linewidth=0.5)
ax4.set_xlabel('PnL per trade (k KRW)')
ax4.set_ylabel('Count')
ax4.set_title('Trade PnL Distribution')
ax4.legend()
ax4.grid(alpha=0.3)

plt.savefig('/home/claude/backtest_chart.png', dpi=110, bbox_inches='tight')
print(f"\n차트 저장 완료: backtest_chart.png")