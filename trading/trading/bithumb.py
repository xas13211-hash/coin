"""
bithumb.py
==========
빗썸 v2.1.5 Public Candlestick API 데이터 수집 모듈.

중요 제약:
1. 빗썸 API는 15m봉을 직접 지원하지 않음 → 5m봉을 받아서 15m으로 리샘플링
2. limit/from/to 파라미터 없음 → 한 번 호출에 약 200~1500봉만 반환
3. 옛날 데이터를 한 번에 받을 수 없음 → 누적 수집(매시간 cron) 필요

지원 interval: 1m, 3m, 5m, 10m, 30m, 1h, 6h, 12h, 24h
Public API라 인증키 불필요.
Rate limit: Public 약 분당 90회 (요청 간 sleep 권장).
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import requests
import pandas as pd


BITHUMB_BASE = "https://api.bithumb.com/public/candlestick"

# 빗썸이 직접 지원하는 interval
NATIVE_INTERVALS = {'1m', '3m', '5m', '10m', '30m', '1h', '6h', '12h', '24h'}

# 사용자 친화적 alias
INTERVAL_ALIAS = {
    '1d': '24h',
    '1day': '24h',
}


# =====================================================================
# 1. 단일 호출
# =====================================================================
def fetch_candlestick(
    symbol: str = 'BTC',
    payment: str = 'KRW',
    interval: str = '5m',
    timeout: int = 10,
) -> pd.DataFrame:
    """
    빗썸 candlestick API 호출.

    Returns
    -------
    DataFrame (index=timestamp, KST)
    columns: open, high, low, close, volume

    Notes
    -----
    - 빗썸 응답 순서는 [timestamp, open, close, high, low, volume]
      (close가 high보다 먼저 옴 - 주의!)
    - timestamp는 milliseconds (UTC). KST(+9h)로 변환.
    """
    if interval == '15m':
        # 15m은 직접 지원 X → 5m을 받아 리샘플링
        df_5m = fetch_candlestick(symbol, payment, '5m', timeout)
        return resample(df_5m, '15min')

    interval = INTERVAL_ALIAS.get(interval, interval)
    if interval not in NATIVE_INTERVALS:
        raise ValueError(f"빗썸 미지원 interval: {interval}. "
                         f"지원: {sorted(NATIVE_INTERVALS)} 또는 '15m'")

    url = f"{BITHUMB_BASE}/{symbol}_{payment}/{interval}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get('status') != '0000':
        raise RuntimeError(f"Bithumb API error: {payload}")

    data = payload['data']
    if not data:
        raise RuntimeError("빗썸 응답이 비어있음 (심볼/마켓 확인 필요)")

    # 빗썸 컬럼 순서: [ts, open, close, high, low, volume]
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    df = df.set_index('timestamp')

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    # backtest 모듈이 기대하는 컬럼 순서로 정렬
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df.sort_index()


# =====================================================================
# 2. 리샘플링
# =====================================================================
def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV를 다른 시간프레임으로 변환."""
    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()


# =====================================================================
# 3. 캐시 + 누적 수집
# =====================================================================
def update_history(
    symbol: str = 'BTC',
    payment: str = 'KRW',
    interval: str = '5m',
    cache_dir: str = './data',
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """
    최신 데이터를 받아 기존 캐시와 머지.
    매시간(또는 매일) 한 번씩 cron으로 돌리면 자연스럽게 historical이 쌓인다.

    저장 위치: {cache_dir}/{symbol}_{payment}_{interval}.parquet
    """
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol}_{payment}_{interval}.parquet"
    cache_file = cache_dir_p / fname

    new_df = fetch_candlestick(symbol, payment, interval)
    time.sleep(sleep_sec)  # rate limit 보호

    if cache_file.exists():
        old_df = pd.read_parquet(cache_file)
        combined = pd.concat([old_df, new_df])
        # 중복 제거 (같은 시각 봉) - 새 데이터 우선
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = new_df

    combined.to_parquet(cache_file)
    print(f"[{symbol}_{payment}_{interval}] "
          f"누적 {len(combined)}봉 ({combined.index[0]} ~ {combined.index[-1]})")
    return combined


def load_history(
    symbol: str = 'BTC',
    payment: str = 'KRW',
    interval: str = '5m',
    cache_dir: str = './data',
) -> Optional[pd.DataFrame]:
    """저장된 캐시만 읽음. 없으면 None."""
    cache_file = Path(cache_dir) / f"{symbol}_{payment}_{interval}.parquet"
    if not cache_file.exists():
        return None
    return pd.read_parquet(cache_file)


# =====================================================================
# 4. 멀티 심볼 일괄 업데이트
# =====================================================================
def update_multi(
    symbols: list[str] = ['BTC', 'ETH'],
    payment: str = 'KRW',
    interval: str = '5m',
    cache_dir: str = './data',
    sleep_sec: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """BTC, ETH 등 여러 심볼을 순차 업데이트."""
    results = {}
    for sym in symbols:
        try:
            results[sym] = update_history(sym, payment, interval, cache_dir, sleep_sec)
        except Exception as e:
            print(f"[{sym}] 실패: {e}")
            results[sym] = None
        time.sleep(sleep_sec)
    return results


# =====================================================================
# 5. 백테스트용 데이터 준비 (5m → 15m 변환)
# =====================================================================
def get_15m_for_backtest(
    symbol: str = 'BTC',
    payment: str = 'KRW',
    cache_dir: str = './data',
    fetch_if_empty: bool = True,
) -> pd.DataFrame:
    """
    백테스트 엔진이 기대하는 15분봉 DataFrame을 반환.
    캐시에 5분봉이 있으면 그걸 15분봉으로 변환, 없으면 새로 받음.
    """
    df_5m = load_history(symbol, payment, '5m', cache_dir)
    if df_5m is None:
        if not fetch_if_empty:
            raise FileNotFoundError(f"캐시 없음: {symbol}_{payment}_5m. "
                                    f"먼저 update_history()를 실행하세요.")
        df_5m = update_history(symbol, payment, '5m', cache_dir)
    return resample(df_5m, '15min')


# =====================================================================
# 6. CLI 진입점 (cron 등록용)
# =====================================================================
if __name__ == '__main__':
    """
    사용법:
        python -m trading.bithumb              # BTC, ETH 5분봉 업데이트
        python -m trading.bithumb 1h           # 1시간봉 업데이트

    crontab 예시 (매 10분마다 자동 수집):
        */10 * * * * cd /path/to/project && python -m trading.bithumb >> data/log.txt 2>&1
    """
    import sys
    interval_arg = sys.argv[1] if len(sys.argv) > 1 else '5m'
    print(f"=== 빗썸 데이터 업데이트 시작 (interval={interval_arg}) ===")
    update_multi(['BTC', 'ETH'], 'KRW', interval_arg)
    print("=== 완료 ===")