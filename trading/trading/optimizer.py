"""
optimizer.py
============
파라미터 자동 최적화.

핵심:
- Grid Search로 모든 파라미터 조합 탐색
- Train(70%) / Test(30%) 분리로 과최적화(overfitting) 검증
- 복합 점수 (Sharpe + Profit Factor - MDD 페널티)로 평가
- Train에서 우수한 파라미터가 Test에서도 우수해야 진짜 좋은 것

⚠️ 중요: 이 도구는 "과거에 잘 됐던 파라미터"를 찾을 뿐입니다.
미래 수익을 보장하지 않습니다. Walk-forward + 페이퍼 트레이딩으로 추가 검증 필수.
"""
from __future__ import annotations
import itertools
from typing import Callable, Optional
import numpy as np
import pandas as pd

from .backtest import run_backtest, BacktestConfig
from . import regime as regime_module


# =====================================================================
# 기본 파라미터 그리드 (강도별)
# =====================================================================
GRID_FAST = {
    'buy_threshold': [3.5, 4.5, 5.5],
    'risk_per_trade_pct': [0.01, 0.015],
    'atr_stop_mult': [1.5, 2.0],
    'tp2_r': [2.0, 2.5],
    'require_mtf': [True],
    'use_trailing': [True],
}  # 24 조합

GRID_STANDARD = {
    'buy_threshold': [3.5, 4.0, 4.5, 5.0, 5.5],
    'risk_per_trade_pct': [0.01, 0.015, 0.02],
    'atr_stop_mult': [1.0, 1.5, 2.0, 2.5],
    'tp2_r': [2.0, 2.5, 3.0],
    'require_mtf': [True],
    'use_trailing': [True],
}  # 180 조합

GRID_PRECISE = {
    'buy_threshold': [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    'risk_per_trade_pct': [0.01, 0.015, 0.02, 0.025],
    'atr_stop_mult': [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    'tp2_r': [1.5, 2.0, 2.5, 3.0],
    'require_mtf': [True, False],
    'use_trailing': [True],
}  # 1344 조합 - 시간 많이 걸림


PRESETS = {
    'fast': GRID_FAST,
    'standard': GRID_STANDARD,
    'precise': GRID_PRECISE,
}


# =====================================================================
# 복합 점수 계산
# =====================================================================
def composite_score(metrics: dict, min_trades: int = 20) -> float:
    """
    파라미터 조합의 종합 점수.
    높을수록 좋음. 거래 수가 너무 적으면 -999 (통계적으로 무의미).

    구성:
    - 샤프 비율 (메인 지표): 위험조정수익률
    - 수익 팩터 보너스: 1 초과분의 절반 가산
    - MDD 페널티: 15% 초과분에 대해 5배 페널티
    """
    n_trades = metrics.get('n_trades', 0)
    if n_trades < min_trades:
        return -999  # 통계적으로 무의미한 표본

    sharpe = metrics.get('sharpe', 0)
    mdd = abs(metrics.get('mdd', 0))
    pf = metrics.get('profit_factor', 0)
    if not np.isfinite(pf):
        pf = 0

    # MDD 15% 초과분 페널티
    mdd_penalty = max(0, mdd - 0.15) * 5

    # 종합 점수
    score = sharpe + (pf - 1) * 0.5 - mdd_penalty
    return float(score)


# =====================================================================
# 단일 조합 실행
# =====================================================================
def generate_grid(param_grid: dict) -> list[dict]:
    """파라미터 격자 → 모든 조합."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_single(
    df: pd.DataFrame,
    params: dict,
    base_config: BacktestConfig,
    min_trades: int = 20,
) -> dict:
    """파라미터 한 조합으로 백테스트 실행."""
    config = BacktestConfig(
        initial_capital=base_config.initial_capital,
        risk_per_trade_pct=params.get('risk_per_trade_pct', base_config.risk_per_trade_pct),
        fee_pct=base_config.fee_pct,
        slippage_pct=base_config.slippage_pct,
        atr_stop_mult=params.get('atr_stop_mult', base_config.atr_stop_mult),
        tp1_r=base_config.tp1_r,
        tp2_r=params.get('tp2_r', base_config.tp2_r),
        tp1_close_pct=base_config.tp1_close_pct,
        use_trailing=params.get('use_trailing', base_config.use_trailing),
        trail_activate_r=base_config.trail_activate_r,
        trail_atr_mult=base_config.trail_atr_mult,
        max_hold_bars=base_config.max_hold_bars,
        require_mtf=params.get('require_mtf', base_config.require_mtf),
        use_daily_limit=base_config.use_daily_limit,
        daily_loss_limit_pct=base_config.daily_loss_limit_pct,
        max_trades_per_day=base_config.max_trades_per_day,
        max_consecutive_losses=base_config.max_consecutive_losses,
        warmup_bars=base_config.warmup_bars,
    )

    # 임계값을 regime 모듈에 적용
    thr = params.get('buy_threshold', 4.5)
    regime_module.REGIME_THRESHOLDS = {
        'TREND':    {'buy': thr - 0.5, 'sell': -(thr - 0.5)},
        'RANGE':    {'buy': thr, 'sell': -thr},
        'VOLATILE': {'buy': thr + 1.5, 'sell': -(thr + 1.5)},
        'NORMAL':   {'buy': thr, 'sell': -thr},
    }

    try:
        result = run_backtest(df, config)
        score = composite_score(result['metrics'], min_trades=min_trades)
        return {
            'params': params,
            'metrics': result['metrics'],
            'score': score,
            'final_capital': result.get('final_capital'),
        }
    except Exception as e:
        return {
            'params': params,
            'metrics': {},
            'score': -999,
            'error': str(e),
        }


# =====================================================================
# 메인: Train/Test 분리 Grid Search
# =====================================================================
def grid_search(
    df: pd.DataFrame,
    param_grid: dict = None,
    base_config: BacktestConfig = None,
    train_pct: float = 0.7,
    top_n_oos: int = 10,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Train/Test 분리 격자 탐색.

    1) 데이터를 train(70%) / test(30%)로 분리
    2) train에서 모든 파라미터 조합 시도
    3) train 점수 top N개를 test 데이터에 적용 (out-of-sample 검증)
    4) train과 test 점수의 상관관계로 과최적화 정도 판단

    Returns
    -------
    {
        'all_train': 모든 조합의 train 결과 (점수 내림차순)
        'oos_results': top N개의 train + test 결과
        'best_params': test에서 가장 좋은 파라미터
        'train_test_correlation': train↔test 점수 상관계수
                                  > 0.5 = 안정적, < 0 = 과최적화 의심
        'train_period', 'test_period': 분리된 기간
    }
    """
    if param_grid is None:
        param_grid = GRID_STANDARD
    if base_config is None:
        base_config = BacktestConfig()

    # 1. 데이터 분리
    split_idx = int(len(df) * train_pct)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    if len(df_test) < 100:
        raise ValueError(f"Test 데이터 부족 ({len(df_test)}봉). 더 긴 데이터 필요.")

    # 2. 모든 조합 생성
    combos = generate_grid(param_grid)
    n_total = len(combos)

    # train/test 별 최소 거래 수: 데이터 길이에 비례하여 자동 조정
    # (15분봉 기준 약 60봉당 1거래 예상)
    train_min_trades = max(5, min(20, len(df_train) // 60))
    test_min_trades = max(3, min(15, len(df_test) // 60))

    # 3. Train 데이터로 모든 조합 실행
    train_results = []
    for i, params in enumerate(combos):
        result = run_single(df_train, params, base_config, min_trades=train_min_trades)
        train_results.append(result)
        if progress_callback:
            progress_callback(i + 1, n_total, 'train')

    train_results.sort(key=lambda x: x['score'], reverse=True)

    # 4. Top N 조합을 test 데이터에 적용
    top_qualifying = [r for r in train_results if r['score'] > -100][:top_n_oos]
    oos_results = []
    for i, tr in enumerate(top_qualifying):
        oos = run_single(df_test, tr['params'], base_config, min_trades=test_min_trades)
        oos_results.append({
            'params': tr['params'],
            'train_score': tr['score'],
            'train_metrics': tr['metrics'],
            'test_score': oos['score'],
            'test_metrics': oos['metrics'],
        })
        if progress_callback:
            progress_callback(i + 1, len(top_qualifying), 'test')

    # 5. Train-Test 상관관계 (과최적화 지표)
    correlation = None
    if len(oos_results) >= 3:
        ts = [r['train_score'] for r in oos_results]
        ts2 = [r['test_score'] for r in oos_results]
        if np.std(ts) > 0 and np.std(ts2) > 0:
            correlation = float(np.corrcoef(ts, ts2)[0, 1])

    # 6. 베스트 결정 (test 점수 기준)
    qualified = [r for r in oos_results if r['test_score'] > -100]
    qualified.sort(key=lambda x: x['test_score'], reverse=True)
    best_by_test = qualified[0] if qualified else None

    return {
        'all_train': train_results,
        'oos_results': oos_results,
        'best_params': best_by_test['params'] if best_by_test else None,
        'best_oos_metrics': best_by_test['test_metrics'] if best_by_test else None,
        'train_test_correlation': correlation,
        'split_idx': split_idx,
        'train_period': (df_train.index[0], df_train.index[-1]),
        'test_period': (df_test.index[0], df_test.index[-1]),
        'n_combos_tested': n_total,
    }


# =====================================================================
# 결과 진단 (과최적화 경고)
# =====================================================================
def diagnose_overfitting(opt_result: dict) -> dict:
    """최적화 결과에서 과최적화 위험을 진단."""
    corr = opt_result.get('train_test_correlation')
    oos = opt_result.get('oos_results', [])

    diagnosis = {
        'level': 'unknown',
        'message': '',
        'recommendation': '',
    }

    if not oos:
        diagnosis['level'] = 'no_data'
        diagnosis['message'] = '검증 결과가 부족합니다.'
        return diagnosis

    # train-test 점수 비교
    train_scores = [r['train_score'] for r in oos]
    test_scores = [r['test_score'] for r in oos]
    avg_train = np.mean(train_scores)
    avg_test = np.mean(test_scores)
    drop = avg_train - avg_test

    if corr is None:
        diagnosis['level'] = 'unknown'
        diagnosis['message'] = (
            f'⚠️ 표본이 부족하거나 점수 변동이 작아 상관계수를 계산할 수 없습니다. '
            f'더 긴 데이터로 다시 시도하세요.'
        )
        diagnosis['recommendation'] = '최소 1000봉 이상 데이터 권장.'
    elif corr >= 0.5 and drop < 0.5:
        diagnosis['level'] = 'good'
        diagnosis['message'] = (
            f'✅ Train-Test 상관계수 {corr:.2f}, 점수 하락폭 {drop:.2f}. '
            f'파라미터가 안정적으로 작동합니다.'
        )
        diagnosis['recommendation'] = '페이퍼 트레이딩으로 한 번 더 검증 후 소액 실전 가능.'
    elif corr >= 0 and drop < 1.0:
        diagnosis['level'] = 'caution'
        diagnosis['message'] = (
            f'⚠️ Train-Test 상관계수 {corr:.2f}, 점수 하락폭 {drop:.2f}. '
            f'어느 정도 안정적이나 시장 변화에 민감할 수 있습니다.'
        )
        diagnosis['recommendation'] = (
            '페이퍼 트레이딩 최소 2주, 가능하면 다른 기간 데이터로 추가 검증.'
        )
    else:
        diagnosis['level'] = 'overfit'
        diagnosis['message'] = (
            f'❌ Train-Test 상관계수 {corr:.2f}, 점수 하락폭 {drop:.2f}. '
            f'과최적화 의심: train에서만 잘 됐을 가능성 큽니다.'
        )
        diagnosis['recommendation'] = (
            '실전 사용 비추천. 더 많은 데이터를 모으거나 단순한 파라미터를 쓰세요.'
        )

    diagnosis['avg_train_score'] = avg_train
    diagnosis['avg_test_score'] = avg_test
    diagnosis['score_drop'] = drop
    diagnosis['correlation'] = corr
    return diagnosis