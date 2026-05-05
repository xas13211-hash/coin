"""
run_bithumb_backtest.py
=======================
실전 사용 스크립트: 빗썸에서 데이터 받아 백테스트 실행.

사용법:
    # 1. 단발성 빠른 테스트 (1시간봉 60일치 - 즉시 가능)
    python run_bithumb_backtest.py quick

    # 2. 데이터 누적 (15분봉 - cron으로 매시간 실행)
    python run_bithumb_backtest.py collect

    # 3. 누적된 데이터로 백테스트
    python run_bithumb_backtest.py backtest

권장 워크플로:
    [Day 1]  python run_bithumb_backtest.py quick      # 1h봉으로 즉시 검증
    [Day 1~] crontab에 collect 등록 → 1~2주간 15m봉 누적
    [Day 14] python run_bithumb_backtest.py backtest   # 진짜 단타용 백테스트
"""
import sys
from pathlib import Path

import pandas as pd

from trading.bithumb import (
    fetch_candlestick,
    resample,
    update_history,
    update_multi,
    load_history,
    get_15m_for_backtest,
)
from trading.backtest import run_backtest, BacktestConfig, print_report


CACHE_DIR = './data'


# =====================================================================
# 모드 1: 빠른 검증 (1시간봉으로 즉시)
# =====================================================================
def quick_test():
    """
    1h봉 1500봉 ≈ 62일치를 한 번에 받아 즉시 백테스트.
    단점: 15분봉이 아니라서 진짜 단타 시뮬레이션은 아님 (스윙 단타 수준).
    장점: 데이터 누적 기다릴 필요 없이 즉시 실행 가능.
    """
    print("=" * 60)
    print("모드 1: 빠른 검증 (1시간봉)")
    print("=" * 60)

    print("\n[1/3] BTC 1시간봉 다운로드...")
    df_btc = fetch_candlestick('BTC', 'KRW', '1h')
    print(f"  → {len(df_btc)}봉 ({df_btc.index[0]} ~ {df_btc.index[-1]})")

    print("\n[2/3] ETH 1시간봉 다운로드...")
    df_eth = fetch_candlestick('ETH', 'KRW', '1h')
    print(f"  → {len(df_eth)}봉 ({df_eth.index[0]} ~ {df_eth.index[-1]})")

    print("\n[3/3] BTC 백테스트 실행...")
    # 주의: 백테스트 엔진은 15분봉 기준으로 짜여있어서, 1h봉으로 돌리면
    #       max_hold_bars(기본 32 = 8시간)가 32시간이 됨 → 적절히 조정
    config = BacktestConfig(
        initial_capital=4_800_000,
        risk_per_trade_pct=0.015,
        require_mtf=True,
        use_trailing=True,
        max_hold_bars=8,         # 1h봉 × 8 = 8시간
        warmup_bars=200,
    )
    result = run_backtest(df_btc, config)
    print_report(result)

    return df_btc, df_eth, result


# =====================================================================
# 모드 2: 누적 수집 (cron으로 매시간 실행)
# =====================================================================
def collect():
    """
    BTC, ETH 5분봉을 받아 디스크에 누적.
    cron 예시 (매 30분마다):
        */30 * * * * cd /path && python run_bithumb_backtest.py collect >> data/log.txt 2>&1
    """
    print(f"[{pd.Timestamp.now()}] 빗썸 데이터 수집 시작")
    update_multi(['BTC', 'ETH'], 'KRW', '5m', cache_dir=CACHE_DIR)
    print(f"[{pd.Timestamp.now()}] 완료\n")


# =====================================================================
# 모드 3: 누적된 데이터로 본격 백테스트
# =====================================================================
def real_backtest():
    """누적된 5분봉 → 15분봉 변환 → 백테스트."""
    print("=" * 60)
    print("모드 3: 누적 데이터로 본격 백테스트 (15분봉)")
    print("=" * 60)

    df_btc_5m = load_history('BTC', 'KRW', '5m', CACHE_DIR)
    if df_btc_5m is None:
        print("❌ 캐시 없음. 먼저 'collect' 모드로 데이터를 누적하세요.")
        print("   권장: 최소 1주일 누적 (672+ 15분봉)")
        return

    print(f"누적된 5분봉: {len(df_btc_5m)}봉")
    print(f"기간: {df_btc_5m.index[0]} ~ {df_btc_5m.index[-1]}")

    df_btc_15m = resample(df_btc_5m, '15min')
    print(f"15분봉 변환: {len(df_btc_15m)}봉")

    if len(df_btc_15m) < 300:
        print(f"⚠️  봉 수가 적음 ({len(df_btc_15m)}). 최소 300봉 권장.")
        print("   더 누적된 후 다시 실행하세요.")
        return

    config = BacktestConfig(
        initial_capital=4_800_000,
        risk_per_trade_pct=0.015,
        require_mtf=True,
        use_trailing=True,
        max_hold_bars=32,        # 15m × 32 = 8시간
    )
    result = run_backtest(df_btc_15m, config)
    print_report(result)
    return result


# =====================================================================
# CLI
# =====================================================================
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'quick'

    if mode == 'quick':
        quick_test()
    elif mode == 'collect':
        collect()
    elif mode == 'backtest':
        real_backtest()
    else:
        print(f"알 수 없는 모드: {mode}")
        print("사용법: python run_bithumb_backtest.py [quick|collect|backtest]")