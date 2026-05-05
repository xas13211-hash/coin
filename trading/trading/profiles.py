"""
profiles.py
===========
심볼별 운영 파라미터 프로파일 관리.

저장 위치: profiles/{symbol}.json
형식:
{
    "buy_threshold": 4.5,
    "risk_pct": 1.5,
    "atr_stop_mult": 1.5,
    "tp2_r": 2.0,
    "require_mtf": true,
    "use_trailing": true,
    "max_trades": 3,
    "daily_loss_limit": 3.0,
    "updated_at": "2026-05-05T10:30:00"
}
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


# 심볼별 권장 기본값 (각자 다름!)
PROFILE_DEFAULTS = {
    'BTC': {
        'buy_threshold': 4.5,
        'risk_pct': 1.5,
        'atr_stop_mult': 1.5,    # BTC는 변동성 적당
        'tp2_r': 2.0,
        'require_mtf': True,
        'use_trailing': True,
        'max_trades': 3,
        'daily_loss_limit': 3.0,
    },
    'ETH': {
        'buy_threshold': 5.0,    # ETH는 더 신중하게 (가짜 신호 많음)
        'risk_pct': 1.0,         # 변동성 크니까 사이즈 ↓
        'atr_stop_mult': 2.0,    # 손절폭 더 넓게 (잡파동에 휘둘리지 않게)
        'tp2_r': 2.5,            # 추세 잘 가면 더 먹게
        'require_mtf': True,
        'use_trailing': True,
        'max_trades': 3,
        'daily_loss_limit': 3.0,
    },
}

# 자본·수수료는 심볼 무관 (전역)
GLOBAL_DEFAULTS = {
    'capital': 4_800_000,
    'fee_pct': 0.04,
}


def get_profile_path(symbol: str, profile_dir: str = './profiles') -> Path:
    p = Path(profile_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{symbol.lower()}.json"


def load_profile(symbol: str, profile_dir: str = './profiles') -> dict:
    """저장된 프로파일 로드. 없으면 기본값 반환."""
    path = get_profile_path(symbol, profile_dir)
    if path.exists():
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return PROFILE_DEFAULTS.get(symbol, PROFILE_DEFAULTS['BTC']).copy()


def save_profile(symbol: str, params: dict, profile_dir: str = './profiles') -> None:
    """프로파일 저장. 비교 가능하게 timestamp 추가."""
    path = get_profile_path(symbol, profile_dir)
    data = {**params, 'updated_at': datetime.now().isoformat()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_global(profile_dir: str = './profiles') -> dict:
    """전역(자본·수수료) 설정."""
    path = Path(profile_dir) / 'global.json'
    if path.exists():
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return GLOBAL_DEFAULTS.copy()


def save_global(params: dict, profile_dir: str = './profiles') -> None:
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    path = Path(profile_dir) / 'global.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def list_profiles(profile_dir: str = './profiles') -> dict:
    """모든 심볼의 프로파일 요약 반환."""
    p = Path(profile_dir)
    if not p.exists():
        return {}
    result = {}
    for f in p.glob('*.json'):
        if f.stem == 'global':
            continue
        try:
            with open(f, encoding='utf-8') as fp:
                result[f.stem.upper()] = json.load(fp)
        except Exception:
            pass
    return result


# =====================================================================
# 최적화 결과 저장/로드 (pickle 사용 - 복잡한 객체 그대로 저장)
# =====================================================================
import pickle


def save_optimization(symbol: str, result: dict, meta: dict = None,
                      opt_dir: str = './optimizations') -> None:
    """
    Auto-Optimize 결과를 디스크에 저장.
    
    Parameters
    ----------
    symbol : 'BTC' or 'ETH'
    result : grid_search()가 반환한 딕셔너리
    meta : 추가 메타데이터 (실행 시각, 데이터 소스, 강도 등)
    """
    p = Path(opt_dir)
    p.mkdir(parents=True, exist_ok=True)
    payload = {
        'result': result,
        'meta': meta or {},
        'saved_at': datetime.now().isoformat(),
    }
    file_path = p / f"{symbol.lower()}.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(payload, f)


def load_optimization(symbol: str, opt_dir: str = './optimizations') -> Optional[dict]:
    """저장된 최적화 결과 로드. 없으면 None."""
    file_path = Path(opt_dir) / f"{symbol.lower()}.pkl"
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def delete_optimization(symbol: str, opt_dir: str = './optimizations') -> bool:
    """저장된 최적화 결과 삭제."""
    file_path = Path(opt_dir) / f"{symbol.lower()}.pkl"
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def list_optimizations(opt_dir: str = './optimizations') -> dict:
    """모든 심볼의 저장된 최적화 결과 메타데이터 반환."""
    p = Path(opt_dir)
    if not p.exists():
        return {}
    result = {}
    for f in p.glob('*.pkl'):
        try:
            with open(f, 'rb') as fp:
                payload = pickle.load(fp)
            result[f.stem.upper()] = {
                'saved_at': payload.get('saved_at'),
                'meta': payload.get('meta', {}),
                'n_combos': payload.get('result', {}).get('n_combos_tested', 0),
            }
        except Exception:
            pass
    return result