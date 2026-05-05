"""
upbit.py
========
업비트 API 데이터 수집 모듈 (백테스트 전용).

빗썸 대비 장점:
- 15분봉 직접 지원 (리샘플링 불필요)
- to 파라미터로 페이지네이션 → 1년, 2년치 historical 데이터 무제한 확보 가능
- KRW-BTC, KRW-ETH 가격이 빗썸과 거의 동일 (차익거래로 0.05% 이내)

지원 interval: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h, 1d
Public API라 인증 키 불필요.
Rate limit: 분당 ~600회 (요청 간 0.15초 sleep으로 안전)

⚠️ 거래는 빗썸에서, 백테스트 데이터만 업비트에서. 
   이는 알고리즘 트레이더들이 흔히 쓰는 우회법.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Callable

import requests
import pandas as pd


UPBIT_BASE = "https://api.upbit.com/v1"

# interval → (endpoint, unit)
INTERVAL_MAP = {
    '1m':  ('minutes', 1),
    '3m':  ('minutes', 3),
    '5m':  ('minutes', 5),
    '10m': ('minutes', 10),
    '15m': ('minutes', 15),     # 빗썸과 달리 직접 지원!
    '30m': ('minutes', 30),
    '1h':  ('minutes', 60),
    '4h':  ('minutes', 240),
    '1d':  ('days', None),
    '24h': ('days', None),
}

# 봉 1개당 분 수 (시간 추정용)
MIN_PER_BAR = {
    '1m': 1, '3m': 3, '5m': 5, '10m': 10, '15m': 15,
    '30m': 30, '1h': 60, '4h': 240, '1d': 1440, '24h': 1440,
}


def to_market_code(symbol: str) -> str:
    """'BTC' → 'KRW-BTC' 형식 변환."""
    if '-' in symbol:
        return symbol  # 이미 형식 갖춤
    return f"KRW-{symbol}"


# =====================================================================
# 1. 단일 호출 (최대 200봉)
# =====================================================================
def fetch_candles_single(
    symbol: str = 'BTC',
    interval: str = '15m',
    count: int = 200,
    to: Optional[str] = None,
    timeout: int = 10,
) -> pd.DataFrame:
    """
    업비트 API 단일 호출. 최대 200봉 반환.

    Parameters
    ----------
    symbol : 'BTC', 'ETH' 또는 'KRW-BTC' 같은 풀 코드
    interval : 위 INTERVAL_MAP 참조
    count : 1~200
    to : ISO 8601 형식 (예: '2026-04-15T00:00:00Z'). 미지정 시 최신.

    Returns
    -------
    DataFrame (index=KST tz-naive, columns=open/high/low/close/volume)
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"미지원 interval: {interval}. 지원: {list(INTERVAL_MAP)}")

    market = to_market_code(symbol)
    endpoint, unit = INTERVAL_MAP[interval]
    if endpoint == 'minutes':
        url = f"{UPBIT_BASE}/candles/minutes/{unit}"
    else:
        url = f"{UPBIT_BASE}/candles/days"

    params = {'market': market, 'count': min(count, 200)}
    if to:
        params['to'] = to

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(data)
    # KST 시간 사용 (tz-naive로 통일, 빗썸 모듈과 호환)
    df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.rename(columns={
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume',
    })
    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    return df.sort_index()


# =====================================================================
# 2. 페이지네이션으로 N일치 받기
# =====================================================================
def fetch_history(
    symbol: str = 'BTC',
    interval: str = '15m',
    days: int = 365,
    sleep_sec: float = 0.15,
    progress_callback: Optional[Callable] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    페이지네이션으로 N일치 historical 데이터 수집.

    예: fetch_history('BTC', '15m', days=365)
        → BTC 15분봉 약 35,000봉 (1년치) 반환. 약 3분 소요.

    Parameters
    ----------
    days : 며칠치 받을지
    sleep_sec : 호출 간 sleep (rate limit 보호)
    progress_callback : fn(현재봉수, 목표봉수, 상태메시지)
    """
    minutes_per_bar = MIN_PER_BAR.get(interval, 15)
    target_count = (days * 24 * 60) // minutes_per_bar

    all_batches = []
    to_param: Optional[str] = None
    fetched = 0

    while fetched < target_count:
        # 재시도 로직
        batch = None
        for attempt in range(max_retries):
            try:
                batch = fetch_candles_single(symbol, interval, count=200, to=to_param)
                break
            except requests.exceptions.HTTPError as e:
                # 429 (rate limit) 또는 5xx → 백오프 후 재시도
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                if progress_callback:
                    progress_callback(fetched, target_count,
                                      f"⚠️ 재시도 {attempt+1}/{max_retries} ({wait}초 대기)")
                time.sleep(wait)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.0)

        if batch is None or batch.empty:
            break

        all_batches.append(batch)
        fetched += len(batch)

        # 다음 배치: 현재 배치의 가장 오래된 시각보다 이전부터
        oldest_kst = batch.index[0]
        # KST → UTC 변환 (UTC = KST - 9시간)
        oldest_utc = oldest_kst - pd.Timedelta(hours=9)
        to_param = oldest_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

        if progress_callback:
            progress_callback(fetched, target_count,
                              f"수집 중... {fetched}/{target_count}봉")

        # 거래소가 더 이상 데이터를 안 주면 종료
        if len(batch) < 200:
            break

        time.sleep(sleep_sec)

    if not all_batches:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    combined = pd.concat(all_batches)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    # 정확히 N일치로 자르기 (받은 게 더 많을 수도 있음)
    if len(combined) > 0:
        cutoff = combined.index[-1] - pd.Timedelta(days=days)
        combined = combined[combined.index >= cutoff]

    return combined


# =====================================================================
# 3. 캐시 관리 (빗썸 모듈과 동일 인터페이스)
# =====================================================================
def update_history(
    symbol: str = 'BTC',
    interval: str = '15m',
    cache_dir: str = './data',
    days: int = 365,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    1) 신규 가져오면 N일치 일괄 수집 (initial bulk fetch)
    2) 기존 캐시가 있으면 최신 누락분만 추가 수집

    저장: {cache_dir}/upbit_{symbol}_KRW_{interval}.parquet
    """
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)
    fname = f"upbit_{symbol}_KRW_{interval}.parquet"
    cache_file = cache_dir_p / fname

    if cache_file.exists():
        # 누락분만 추가 (마지막 시각 이후부터 현재까지)
        old_df = pd.read_parquet(cache_file)
        last_time = old_df.index[-1]
        gap_minutes = (pd.Timestamp.now() - last_time).total_seconds() / 60
        gap_days = max(1, int(gap_minutes / 60 / 24) + 1)
        print(f"기존 캐시 발견: {len(old_df)}봉. 최근 {gap_days}일 추가 수집...")
        new_df = fetch_history(symbol, interval, days=gap_days,
                               progress_callback=progress_callback)
        combined = pd.concat([old_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        # 신규: N일치 전체 수집
        print(f"신규 수집: {days}일치 ({symbol} {interval})...")
        combined = fetch_history(symbol, interval, days=days,
                                 progress_callback=progress_callback)

    combined.to_parquet(cache_file)
    print(f"[upbit_{symbol}_{interval}] "
          f"{len(combined)}봉 저장 ({combined.index[0]} ~ {combined.index[-1]})")
    return combined


def load_history(
    symbol: str = 'BTC',
    interval: str = '15m',
    cache_dir: str = './data',
) -> Optional[pd.DataFrame]:
    """저장된 캐시만 읽음. 없으면 None."""
    cache_file = Path(cache_dir) / f"upbit_{symbol}_KRW_{interval}.parquet"
    if not cache_file.exists():
        return None
    return pd.read_parquet(cache_file)


def get_15m_for_backtest(
    symbol: str = 'BTC',
    cache_dir: str = './data',
    fetch_if_empty: bool = True,
    days_if_empty: int = 365,
) -> pd.DataFrame:
    """백테스트용 15분봉 DataFrame. 캐시 우선, 없으면 자동 다운로드."""
    df = load_history(symbol, '15m', cache_dir)
    if df is None:
        if not fetch_if_empty:
            raise FileNotFoundError(
                f"캐시 없음: upbit_{symbol}_KRW_15m. "
                f"먼저 update_history()를 실행하세요."
            )
        df = update_history(symbol, '15m', cache_dir, days=days_if_empty)
    return df


# =====================================================================
# 4. CLI 진입점
# =====================================================================
if __name__ == '__main__':
    """
    사용법:
        python -m trading.upbit              # BTC, ETH 15분봉 1년치 수집
        python -m trading.upbit 5m 30        # 5분봉 30일치만
    """
    import sys
    interval = sys.argv[1] if len(sys.argv) > 1 else '15m'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365

    def print_progress(fetched, target, msg):
        if fetched % 1000 == 0 or fetched >= target:
            print(f"  {msg}")

    for sym in ['BTC', 'ETH']:
        print(f"\n=== {sym} {interval} {days}일치 수집 ===")
        try:
            update_history(sym, interval, days=days,
                           progress_callback=print_progress)
        except Exception as e:
            print(f"[{sym}] 실패: {e}")
        time.sleep(1.0)
    print("\n=== 완료 ===")