"""
dashboard.py
============
Streamlit 통합 대시보드.

실행:
    streamlit run dashboard.py

탭 구성:
- 📊 Live: 현재 BTC/ETH 신호 + 가격 차트
- 🧪 Backtest: 설정 조절 → 백테스트 실행 → 결과 시각화
- 💾 Data: 캐시 상태 확인 + 수동 업데이트
"""
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading.bithumb import (
    fetch_candlestick, resample, update_history,
    load_history, get_15m_for_backtest,
)
from trading import upbit
from trading.indicators import compute_all
from trading.signals import generate_signal, prepare_mtf
from trading.backtest import run_backtest, BacktestConfig
from trading.optimizer import (
    grid_search, diagnose_overfitting,
    GRID_FAST, GRID_STANDARD, GRID_PRECISE,
)
from trading.profiles import (
    load_profile, save_profile, load_global, save_global,
    list_profiles, PROFILE_DEFAULTS,
    save_optimization, load_optimization, delete_optimization,
    list_optimizations,
)
from trading import regime as regime_module


# =====================================================================
# 페이지 설정
# =====================================================================
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 색상 테마
COLORS = {
    'buy': '#22c55e',
    'sell': '#ef4444',
    'hold': '#94a3b8',
    'profit': '#22c55e',
    'loss': '#ef4444',
    'ema20': '#f59e0b',
    'ema50': '#3b82f6',
    'ema200': '#dc2626',
    'bg': '#0f172a',
}


# =====================================================================
# 활성 심볼 + 프로파일 자동 로드 (앱 시작 시 디스크에서 복원)
# =====================================================================
if 'active_symbol' not in st.session_state:
    st.session_state['active_symbol'] = 'BTC'

# 첫 실행 시 디스크에서 모든 심볼 프로파일 + 전역 설정 로드
if 'profiles_loaded' not in st.session_state:
    # 전역
    _glob = load_global()
    for k, v in _glob.items():
        st.session_state[k] = v
    # 모든 심볼
    for sym in ['BTC', 'ETH']:
        prof = load_profile(sym)
        for k, v in prof.items():
            if k == 'updated_at':
                continue
            st.session_state[f"{sym.lower()}_{k}"] = v
    # 활성 심볼의 마지막 최적화 결과 자동 로드
    active_sym = st.session_state.get('active_symbol', 'BTC')
    last_opt = load_optimization(active_sym)
    if last_opt:
        st.session_state['opt_result'] = last_opt['result']
        st.session_state['opt_symbol_used'] = active_sym
        st.session_state['opt_meta_loaded'] = last_opt.get('meta', {})
        st.session_state['opt_saved_at_loaded'] = last_opt.get('saved_at')
    st.session_state['profiles_loaded'] = True


def autosave_profile(symbol: str):
    """현재 session_state 값을 디스크에 자동 저장."""
    sl = symbol.lower()
    params = {
        'buy_threshold': st.session_state.get(f"{sl}_buy_threshold"),
        'risk_pct': st.session_state.get(f"{sl}_risk_pct"),
        'atr_stop_mult': st.session_state.get(f"{sl}_atr_stop_mult"),
        'tp2_r': st.session_state.get(f"{sl}_tp2_r"),
        'require_mtf': st.session_state.get(f"{sl}_require_mtf"),
        'use_trailing': st.session_state.get(f"{sl}_use_trailing"),
        'max_trades': st.session_state.get(f"{sl}_max_trades"),
        'daily_loss_limit': st.session_state.get(f"{sl}_daily_loss_limit"),
    }
    # None 값 (아직 위젯이 안 만들어진 경우) 제외
    if all(v is not None for v in params.values()):
        save_profile(symbol, params)


def autosave_global():
    """자본·수수료 자동 저장."""
    cap = st.session_state.get('capital')
    fee = st.session_state.get('fee_pct')
    if cap is not None and fee is not None:
        save_global({'capital': cap, 'fee_pct': fee})


# =====================================================================
# 사이드바: 심볼별 운영 파라미터 (자동 로드/저장)
# =====================================================================
with st.sidebar:
    st.title("⚙️ 운영 설정")

    # 활성 심볼 선택
    active = st.radio(
        "🪙 활성 심볼", ["BTC", "ETH"],
        horizontal=True,
        key="active_symbol",
        help="BTC와 ETH는 변동성이 달라 별도 파라미터 필요"
    )
    sym_lower = active.lower()

    # 마지막 저장 시각 표시
    prof_path = Path(f"./profiles/{sym_lower}.json")
    if prof_path.exists():
        try:
            import json as _json
            saved = _json.loads(prof_path.read_text(encoding='utf-8'))
            saved_at = saved.get('updated_at', '')
            if saved_at:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(saved_at)
                st.caption(f"💾 {active} 마지막 저장: {dt.strftime('%m/%d %H:%M')} (자동)")
            else:
                st.caption(f"💡 {active} 운영 파라미터 (변경 시 자동 저장)")
        except Exception:
            st.caption(f"💡 {active} 운영 파라미터 (변경 시 자동 저장)")
    else:
        st.caption(f"💡 {active} 운영 파라미터 (변경 시 자동 저장)")

    # 자동 저장 콜백 (활성 심볼용)
    _on_change = lambda: autosave_profile(active)
    _on_change_global = lambda: autosave_global()

    # ----- 전역 설정 (심볼 무관) -----
    st.divider()
    st.subheader("💰 자본·비용 (공통)")
    capital = st.number_input(
        "총 자본 (원)", min_value=100_000, max_value=1_000_000_000,
        step=100_000, format="%d", key="capital",
        on_change=_on_change_global,
    )
    fee_pct = st.number_input(
        "거래 수수료 (편도, %)", 0.0, 1.0, step=0.01, key="fee_pct",
        on_change=_on_change_global,
        help="빗썸 일반 0.25%, 쿠폰 적용 0.04%"
    )

    # ----- 심볼별 설정 -----
    st.divider()
    st.subheader(f"🎯 {active} 진입 조건")
    buy_threshold = st.slider(
        "매수 임계값", 2.0, 8.0, step=0.5,
        key=f"{sym_lower}_buy_threshold",
        on_change=_on_change,
        help="가중점수가 이 값 이상이면 매수"
    )
    risk_pct = st.slider(
        "1회 거래 리스크 (%)", 0.5, 5.0, step=0.1,
        key=f"{sym_lower}_risk_pct",
        on_change=_on_change,
    )
    require_mtf = st.checkbox(
        "MTF 필수", key=f"{sym_lower}_require_mtf",
        on_change=_on_change,
    )

    st.subheader(f"📐 {active} 청산 조건")
    atr_stop_mult = st.slider(
        "ATR 손절 배수", 1.0, 3.0, step=0.1,
        key=f"{sym_lower}_atr_stop_mult",
        on_change=_on_change,
    )
    tp2_r = st.slider(
        "2차 익절 R 배수", 1.0, 4.0, step=0.25,
        key=f"{sym_lower}_tp2_r",
        on_change=_on_change,
    )
    use_trailing = st.checkbox(
        "트레일링 스톱", key=f"{sym_lower}_use_trailing",
        on_change=_on_change,
    )

    st.subheader(f"🛡️ {active} 리스크 한도")
    max_trades = st.slider(
        "일일 최대 매매", 1, 10, step=1,
        key=f"{sym_lower}_max_trades",
        on_change=_on_change,
    )
    daily_loss_limit = st.slider(
        "일일 손실 한도 (%)", 1.0, 10.0, step=0.5,
        key=f"{sym_lower}_daily_loss_limit",
        on_change=_on_change,
    )

    # ----- 프로파일 관리 -----
    st.divider()
    st.subheader("💾 프로파일 관리")

    if st.button(f"↺ {active} 기본값으로 리셋", use_container_width=True,
                 help=f"{active} 권장 기본값으로 되돌리기"):
        for k, v in PROFILE_DEFAULTS[active].items():
            st.session_state[f"{sym_lower}_{k}"] = v
        autosave_profile(active)
        st.rerun()

    # 비교 보기
    with st.expander("🔍 BTC vs ETH 설정 비교"):
        all_profiles = {}
        for s in ['BTC', 'ETH']:
            all_profiles[s] = {
                'buy_threshold': st.session_state.get(f"{s.lower()}_buy_threshold"),
                'risk_pct': st.session_state.get(f"{s.lower()}_risk_pct"),
                'atr_stop_mult': st.session_state.get(f"{s.lower()}_atr_stop_mult"),
                'tp2_r': st.session_state.get(f"{s.lower()}_tp2_r"),
            }
        comp = pd.DataFrame(all_profiles).T
        comp.columns = ['임계값', '리스크%', 'ATR×', 'TP2R']
        st.dataframe(comp, use_container_width=True)

    st.divider()
    with st.container(border=True):
        st.markdown("##### 💡 탭 안내")
        st.caption(
            "📊 **Live**: BTC와 ETH 동시 표시 (각자 프로파일 사용)\n\n"
            "🧪 **Backtest**: 선택 심볼의 사이드바 값으로 검증\n\n"
            "🔧 **Optimize**: 선택 심볼의 최적값 자동 탐색\n\n"
            "💾 **Data**: 데이터 다운로드/관리"
        )

    if st.button("🔄 신호 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# =====================================================================
# 캐시된 데이터 페처 (60초 TTL)
# =====================================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_with_indicators(symbol: str, interval: str = '5m') -> pd.DataFrame:
    """빗썸에서 5분봉 받아서 15분봉 + 지표 계산."""
    df_5m = fetch_candlestick(symbol, 'KRW', interval)
    df_15m = resample(df_5m, '15min')
    return compute_all(df_15m)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_mtf_data(symbol: str):
    """MTF용: 5분→15분/1시간/4시간 모두 준비."""
    df_5m = fetch_candlestick(symbol, 'KRW', '5m')
    df_15m_raw = resample(df_5m, '15min')
    return prepare_mtf(df_15m_raw)


def get_signal_for_symbol(symbol: str, recent_signals: list = None):
    """단일 심볼의 현재 신호 반환."""
    df_15m, df_1h, df_4h = fetch_mtf_data(symbol)
    sig = generate_signal(
        df_base=df_15m, df_15m=df_15m, df_1h=df_1h, df_4h=df_4h,
        idx=-1, require_mtf=require_mtf,
        recent_signals=recent_signals or [],
    )
    return sig, df_15m


# =====================================================================
# 신호 카드 컴포넌트
# =====================================================================
def signal_card(symbol: str, sig, df: pd.DataFrame, profile: dict):
    """신호를 카드 형태로 표시. profile은 해당 심볼의 운영 파라미터."""
    color_map = {'BUY': COLORS['buy'], 'SELL': COLORS['sell'], 'HOLD': COLORS['hold']}
    emoji_map = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}
    color = color_map[sig.signal]

    with st.container(border=True):
        col1, col2 = st.columns([2, 3])

        with col1:
            st.markdown(f"### {emoji_map[sig.signal]} {symbol}")
            st.markdown(
                f"<h2 style='color:{color}; margin:0;'>{sig.signal}</h2>",
                unsafe_allow_html=True,
            )
            st.caption(f"국면: **{sig.regime}**")

        with col2:
            st.metric("현재가", f"₩ {sig.price:,.0f}")
            st.metric(
                "합의 점수", f"{sig.weighted_score:+.2f}",
                delta=f"raw: {sig.raw_score:+.1f}",
                delta_color="off",
            )
            mtf_label = {1: "↑ 상승 정합", -1: "↓ 하락 정합", 0: "혼조"}[sig.mtf_alignment]
            st.caption(f"MTF: {mtf_label}")

        # 카테고리별 점수 바
        st.divider()
        score_cols = st.columns(4)
        for col, (k, v) in zip(score_cols, sig.scores.items()):
            with col:
                emoji = "📈" if v > 0 else "📉" if v < 0 else "➖"
                st.metric(k.capitalize(), f"{emoji} {v:+.1f}")

        # 진입 제안 (신호 있을 때만)
        if sig.signal != 'HOLD':
            st.divider()
            stop_dist = sig.atr * profile['atr_stop_mult']
            entry = sig.price
            tp2_r_local = profile['tp2_r']
            if sig.signal == 'BUY':
                stop = entry - stop_dist
                tp1 = entry + stop_dist
                tp2 = entry + stop_dist * tp2_r_local
            else:
                stop = entry + stop_dist
                tp1 = entry - stop_dist
                tp2 = entry - stop_dist * tp2_r_local

            stop_pct = abs(stop - entry) / entry
            position_size = min(
                (capital * profile['risk_pct'] / 100) / (stop_pct + fee_pct / 100 * 2),
                capital,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("권장 포지션", f"₩ {position_size:,.0f}")
            c2.metric("손절가", f"₩ {stop:,.0f}", f"{(stop-entry)/entry*100:+.2f}%")
            c3.metric("1차 익절 (1R)", f"₩ {tp1:,.0f}", f"{(tp1-entry)/entry*100:+.2f}%")
            c4.metric(f"2차 익절 ({tp2_r_local}R)", f"₩ {tp2:,.0f}",
                      f"{(tp2-entry)/entry*100:+.2f}%")

        st.caption(f"💬 {sig.notes}")


# =====================================================================
# Plotly 차트 — 가격 + 지표
# =====================================================================
def make_price_chart(df: pd.DataFrame, symbol: str, last_n: int = 200):
    """캔들스틱 + EMA + RSI + MACD 통합 차트."""
    df = df.tail(last_n)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol}/KRW (15M)", "RSI(14)", "MACD"),
    )

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Price',
        increasing_line_color=COLORS['buy'],
        decreasing_line_color=COLORS['sell'],
    ), row=1, col=1)

    # EMA
    for col, color, name in [
        ('ema20', COLORS['ema20'], 'EMA20'),
        ('ema50', COLORS['ema50'], 'EMA50'),
        ('ema200', COLORS['ema200'], 'EMA200'),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.2),
                opacity=0.85,
            ), row=1, col=1)

    # 볼린저
    if 'bb_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_upper'],
            line=dict(color='gray', width=0.5), name='BB Upper',
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_lower'],
            line=dict(color='gray', width=0.5), name='BB',
            fill='tonexty', fillcolor='rgba(128,128,128,0.08)',
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi'], name='RSI',
        line=dict(color='#a855f7', width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash', width=0.5), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash', width=0.5), row=2, col=1)

    # MACD
    fig.add_trace(go.Bar(
        x=df.index, y=df['macd_hist'], name='MACD Hist',
        marker_color=[
            COLORS['buy'] if v > 0 else COLORS['sell']
            for v in df['macd_hist']
        ],
        opacity=0.5,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['macd'], name='MACD',
        line=dict(color='#3b82f6', width=1.2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['macd_sig'], name='Signal',
        line=dict(color='#f59e0b', width=1.2),
    ), row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation='h', y=1.04, x=0),
    )
    fig.update_yaxes(title_text="Price (KRW)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


# =====================================================================
# 백테스트 결과 차트
# =====================================================================
def make_equity_chart(equity: pd.DataFrame, initial: float):
    """Equity curve + drawdown."""
    eq = equity['equity']
    dd = (eq - eq.cummax()) / eq.cummax() * 100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Equity Curve", "Drawdown (%)"),
    )

    # Equity
    is_profit = eq.iloc[-1] >= initial
    line_color = COLORS['profit'] if is_profit else COLORS['loss']
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq / 1e6, name='Equity',
        line=dict(color=line_color, width=2),
        fill='tozeroy', fillcolor='rgba(34,197,94,0.05)' if is_profit else 'rgba(239,68,68,0.05)',
    ), row=1, col=1)
    fig.add_hline(
        y=initial / 1e6, line=dict(color='gray', dash='dash', width=1),
        row=1, col=1, annotation_text=f"Initial ₩{initial/1e6:.1f}M",
    )

    # Drawdown
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, name='Drawdown',
        line=dict(color=COLORS['loss'], width=1.5),
        fill='tozeroy', fillcolor='rgba(239,68,68,0.2)',
    ), row=2, col=1)

    fig.update_layout(
        height=500, showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_yaxes(title_text="Capital (M KRW)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    return fig


def make_trades_chart(trades, df_price: pd.DataFrame):
    """가격 위에 진입/청산 마커."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_price.index, y=df_price['close'] / 1e6,
        line=dict(color='#64748b', width=1), name='Price',
    ))
    for t in trades:
        color = COLORS['profit'] if t.pnl_krw > 0 else COLORS['loss']
        marker = '▲' if t.side == 'long' else '▼'
        fig.add_trace(go.Scatter(
            x=[t.entry_time], y=[t.entry_price / 1e6],
            mode='markers', marker=dict(symbol='triangle-up' if t.side == 'long' else 'triangle-down',
                                         size=10, color=color, line=dict(width=1, color='black')),
            showlegend=False, hovertext=f"{marker} {t.side} entry",
        ))
        fig.add_trace(go.Scatter(
            x=[t.exit_time], y=[t.exit_price / 1e6],
            mode='markers', marker=dict(symbol='x', size=8, color=color),
            showlegend=False, hovertext=f"exit ({t.exit_reason}): {t.pnl_krw:+,.0f}",
        ))
    fig.update_layout(
        height=400, margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Price (M KRW)", showlegend=False,
    )
    return fig


# =====================================================================
# 메인 헤더
# =====================================================================
st.title("📈 Crypto Trading Dashboard")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST  ·  Capital: ₩{capital:,}")

tab_live, tab_bt, tab_opt, tab_data = st.tabs([
    "📊 Live Signals", "🧪 Backtest", "🔧 Auto-Optimize", "💾 Data"
])


# =====================================================================
# Tab 1: Live Signals (BTC와 ETH 각자 프로파일 사용)
# =====================================================================
with tab_live:
    st.subheader("🪙 현재 매매 신호")
    st.caption(
        "BTC·ETH 각자의 프로파일 (사이드바에서 활성 심볼 변경하며 편집)을 적용합니다."
    )

    # BTC와 ETH 각자의 프로파일 가져오기
    btc_profile = {
        k: st.session_state.get(f"btc_{k}", v)
        for k, v in PROFILE_DEFAULTS['BTC'].items()
    }
    eth_profile = {
        k: st.session_state.get(f"eth_{k}", v)
        for k, v in PROFILE_DEFAULTS['ETH'].items()
    }

    col_btc, col_eth = st.columns(2)
    sig_btc = sig_eth = None
    df_btc = df_eth = None

    with col_btc:
        try:
            with st.spinner("BTC 데이터 가져오는 중..."):
                # BTC의 임계값을 regime 모듈에 적용
                bt = btc_profile['buy_threshold']
                regime_module.REGIME_THRESHOLDS = {
                    'TREND':    {'buy': bt - 0.5, 'sell': -(bt - 0.5)},
                    'RANGE':    {'buy': bt, 'sell': -bt},
                    'VOLATILE': {'buy': bt + 1.5, 'sell': -(bt + 1.5)},
                    'NORMAL':   {'buy': bt, 'sell': -bt},
                }
                df_15m_btc, df_1h_btc, df_4h_btc = fetch_mtf_data('BTC')
                sig_btc = generate_signal(
                    df_base=df_15m_btc, df_15m=df_15m_btc,
                    df_1h=df_1h_btc, df_4h=df_4h_btc, idx=-1,
                    require_mtf=btc_profile['require_mtf'],
                    recent_signals=[],
                )
                df_btc = df_15m_btc
            signal_card('BTC', sig_btc, df_btc, btc_profile)
        except Exception as e:
            st.error(f"BTC 신호 조회 실패: {e}")

    with col_eth:
        try:
            with st.spinner("ETH 데이터 가져오는 중..."):
                et = eth_profile['buy_threshold']
                regime_module.REGIME_THRESHOLDS = {
                    'TREND':    {'buy': et - 0.5, 'sell': -(et - 0.5)},
                    'RANGE':    {'buy': et, 'sell': -et},
                    'VOLATILE': {'buy': et + 1.5, 'sell': -(et + 1.5)},
                    'NORMAL':   {'buy': et, 'sell': -et},
                }
                df_15m_eth, df_1h_eth, df_4h_eth = fetch_mtf_data('ETH')
                sig_eth = generate_signal(
                    df_base=df_15m_eth, df_15m=df_15m_eth,
                    df_1h=df_1h_eth, df_4h=df_4h_eth, idx=-1,
                    require_mtf=eth_profile['require_mtf'],
                    recent_signals=[],
                )
                df_eth = df_15m_eth
            signal_card('ETH', sig_eth, df_eth, eth_profile)
        except Exception as e:
            st.error(f"ETH 신호 조회 실패: {e}")

    st.divider()

    # 차트
    st.subheader("📊 가격 차트")
    chart_symbol = st.radio(
        "심볼", ["BTC", "ETH"], horizontal=True, label_visibility="collapsed",
    )
    last_n = st.slider("표시 봉 수", 50, 500, 200, 50)
    df_chart = df_btc if chart_symbol == 'BTC' else df_eth
    if df_chart is not None:
        st.plotly_chart(
            make_price_chart(df_chart, chart_symbol, last_n),
            use_container_width=True,
        )


# =====================================================================
# Tab 2: Backtest (사이드바 값을 과거 데이터로 검증)
# =====================================================================
with tab_bt:
    st.subheader("🧪 백테스트")
    st.caption("**사이드바 값**으로 과거 데이터에서 어떤 성과가 났을지 검증합니다.")

    # 1. 검증할 시장 조건 (심볼·데이터)
    st.markdown("##### 🎯 검증할 시장 조건")
    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        bt_symbol = st.selectbox(
            "심볼", ["BTC", "ETH"],
            index=0 if active == 'BTC' else 1,
            help=f"기본은 사이드바 활성 심볼({active})"
        )
    with bt_col2:
        bt_interval = st.selectbox(
            "데이터 소스",
            [
                "💎 업비트 15m (1년치) — 추천",
                "💎 업비트 1h (장기)",
                "🔸 빗썸 1h (즉시, ~60일)",
                "🔸 빗썸 5m → 15m (캐시)",
            ],
        )

    # 2. 선택한 심볼의 프로파일 로드 + 표시
    bt_sym_lower = bt_symbol.lower()
    bt_buy_threshold = st.session_state.get(f"{bt_sym_lower}_buy_threshold", 4.5)
    bt_risk_pct = st.session_state.get(f"{bt_sym_lower}_risk_pct", 1.5)
    bt_atr_mult = st.session_state.get(f"{bt_sym_lower}_atr_stop_mult", 1.5)
    bt_tp2_r = st.session_state.get(f"{bt_sym_lower}_tp2_r", 2.0)
    bt_mtf = st.session_state.get(f"{bt_sym_lower}_require_mtf", True)
    bt_trail = st.session_state.get(f"{bt_sym_lower}_use_trailing", True)
    bt_max_trades = st.session_state.get(f"{bt_sym_lower}_max_trades", 3)
    bt_loss_limit = st.session_state.get(f"{bt_sym_lower}_daily_loss_limit", 3.0)

    with st.container(border=True):
        st.markdown(f"##### 📋 {bt_symbol} 프로파일 (사이드바에서 변경)")
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("매수 임계값", bt_buy_threshold)
        sum_col2.metric("ATR 배수", bt_atr_mult)
        sum_col3.metric("TP2 R", bt_tp2_r)
        sum_col4.metric("리스크/거래", f"{bt_risk_pct}%")
        sum_col5, sum_col6, sum_col7, sum_col8 = st.columns(4)
        sum_col5.metric("MTF", "✅" if bt_mtf else "❌")
        sum_col6.metric("트레일링", "✅" if bt_trail else "❌")
        sum_col7.metric("일일 매매", f"{bt_max_trades}회")
        sum_col8.metric("일일 손실 한도", f"{bt_loss_limit}%")

    run_bt = st.button("▶️ 백테스트 실행", type="primary", use_container_width=True)

    if run_bt:
        try:
            with st.spinner("데이터 가져오는 중..."):
                if bt_interval.startswith('💎 업비트 15m'):
                    # 업비트 15분봉 (캐시 우선, 없으면 1년치 자동 다운로드)
                    df_bt = upbit.load_history(bt_symbol, '15m')
                    if df_bt is None or len(df_bt) < 1000:
                        st.info("업비트 15m 캐시 없음 또는 부족. 1년치 자동 다운로드 (약 3분)...")
                        prog = st.progress(0)
                        status = st.empty()
                        def cb(f, t, m):
                            prog.progress(min(f / max(t, 1), 1.0))
                            status.text(m)
                        df_bt = upbit.update_history(bt_symbol, '15m', days=365,
                                                     progress_callback=cb)
                        prog.empty()
                        status.empty()
                    max_hold = 32
                    warmup = 200
                elif bt_interval.startswith('💎 업비트 1h'):
                    df_bt = upbit.load_history(bt_symbol, '1h')
                    if df_bt is None or len(df_bt) < 1000:
                        st.info("업비트 1h 캐시 없음 또는 부족. 1년치 자동 다운로드...")
                        df_bt = upbit.update_history(bt_symbol, '1h', days=365)
                    max_hold = 8
                    warmup = 200
                elif bt_interval.startswith('🔸 빗썸 1h'):
                    df_bt = fetch_candlestick(bt_symbol, 'KRW', '1h')
                    max_hold = 8
                    warmup = min(200, len(df_bt) // 3)
                else:  # 빗썸 5m → 15m
                    df_bt_5m = load_history(bt_symbol, 'KRW', '5m')
                    if df_bt_5m is None:
                        st.error(f"{bt_symbol} 빗썸 5분봉 캐시 없음. Data 탭에서 먼저 수집하세요.")
                        st.stop()
                    df_bt = resample(df_bt_5m, '15min')
                    max_hold = 32
                    warmup = 200

            if len(df_bt) < 250:
                st.warning(f"봉 수 부족 ({len(df_bt)}). 최소 250봉 필요.")
                st.stop()

            with st.spinner(f"백테스트 실행 중 ({len(df_bt)}봉)..."):
                config = BacktestConfig(
                    initial_capital=capital,
                    risk_per_trade_pct=bt_risk_pct / 100,
                    fee_pct=fee_pct / 100,
                    atr_stop_mult=bt_atr_mult,
                    tp2_r=bt_tp2_r,
                    require_mtf=bt_mtf,
                    use_trailing=bt_trail,
                    max_trades_per_day=bt_max_trades,
                    daily_loss_limit_pct=bt_loss_limit / 100,
                    max_hold_bars=max_hold,
                    warmup_bars=warmup,
                )
                # 선택 심볼의 매수 임계값을 regime 모듈에 적용
                regime_module.REGIME_THRESHOLDS = {
                    'TREND':    {'buy': bt_buy_threshold - 0.5,
                                 'sell': -(bt_buy_threshold - 0.5)},
                    'RANGE':    {'buy': bt_buy_threshold,
                                 'sell': -bt_buy_threshold},
                    'VOLATILE': {'buy': bt_buy_threshold + 1.5,
                                 'sell': -(bt_buy_threshold + 1.5)},
                    'NORMAL':   {'buy': bt_buy_threshold,
                                 'sell': -bt_buy_threshold},
                }
                result = run_backtest(df_bt, config)
                st.session_state['bt_result'] = result
                st.session_state['bt_df'] = df_bt
                st.session_state['bt_symbol'] = bt_symbol

        except Exception as e:
            st.error(f"백테스트 실패: {e}")
            import traceback
            with st.expander("상세 오류"):
                st.code(traceback.format_exc())

    # 결과 표시
    if 'bt_result' in st.session_state:
        result = st.session_state['bt_result']
        m = result['metrics']
        df_bt = st.session_state['bt_df']

        if m.get('n_trades', 0) == 0:
            st.warning("거래가 발생하지 않았습니다. 임계값을 낮추거나 MTF를 끄세요.")
        else:
            # 핵심 메트릭 카드
            st.markdown(f"### 📊 결과 요약 — {st.session_state['bt_symbol']}")

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                "총 수익률", f"{m['total_return_pct']*100:+.2f}%",
                delta=f"₩ {m['final_capital']-capital:+,.0f}",
            )
            mc2.metric("승률", f"{m['win_rate']*100:.1f}%")
            mc3.metric("거래 수", m['n_trades'])
            mc4.metric("MDD", f"{m['mdd']*100:.2f}%")

            mc5, mc6, mc7, mc8 = st.columns(4)
            mc5.metric("손익비 (R:R)", f"{m['rr_ratio']:.2f}")
            mc6.metric("수익 팩터", f"{m['profit_factor']:.2f}",
                       delta="좋음" if m['profit_factor'] > 1.5 else "주의",
                       delta_color="normal" if m['profit_factor'] > 1.5 else "inverse")
            mc7.metric("샤프 비율", f"{m['sharpe']:.2f}")
            mc8.metric("소르티노", f"{m['sortino']:.2f}")

            st.divider()

            # Equity curve
            st.markdown("### 📈 Equity Curve")
            st.plotly_chart(
                make_equity_chart(result['equity_curve'], capital),
                use_container_width=True,
            )

            # 거래 마커가 있는 가격 차트
            st.markdown("### 🎯 거래 진입/청산 위치")
            st.plotly_chart(
                make_trades_chart(result['trades'], df_bt),
                use_container_width=True,
            )

            # 거래 리스트
            st.markdown("### 📋 거래 내역")
            trades_data = []
            for t in result['trades']:
                trades_data.append({
                    '진입시각': t.entry_time.strftime('%Y-%m-%d %H:%M'),
                    '청산시각': t.exit_time.strftime('%Y-%m-%d %H:%M'),
                    '방향': t.side,
                    '진입가': f"{t.entry_price:,.0f}",
                    '청산가': f"{t.exit_price:,.0f}",
                    '손익(원)': f"{t.pnl_krw:+,.0f}",
                    '손익(%)': f"{t.pnl_pct*100:+.2f}%",
                    '청산사유': t.exit_reason,
                    '보유봉': t.bars_held,
                })
            df_trades = pd.DataFrame(trades_data)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)

            # 청산 사유 파이
            with st.expander("📊 청산 사유 분포"):
                reasons = m['exit_reasons']
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(reasons.keys()),
                    values=list(reasons.values()),
                    hole=0.4,
                )])
                fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)


# =====================================================================
# Tab 3: Auto-Optimize (자동 파라미터 최적화)
# =====================================================================
with tab_opt:
    st.subheader("🔧 자동 파라미터 최적화")

    # 저장된 최적화 이력 표시
    saved_opts = list_optimizations()
    if saved_opts:
        with st.container(border=True):
            st.markdown("##### 💾 저장된 최적화 결과")
            opt_hist_cols = st.columns(len(saved_opts) + 1)
            for i, (sym, info) in enumerate(saved_opts.items()):
                with opt_hist_cols[i]:
                    saved_at = info.get('saved_at', '')
                    if saved_at:
                        from datetime import datetime as _dt
                        dt = _dt.fromisoformat(saved_at)
                        time_str = dt.strftime('%m/%d %H:%M')
                    else:
                        time_str = '?'
                    meta = info.get('meta', {})
                    intensity = meta.get('intensity', '').split(' ')[0] if meta.get('intensity') else ''
                    n_bars = meta.get('n_bars', 0)
                    if st.button(
                        f"📂 {sym}\n{time_str}\n{intensity} {n_bars:,}봉",
                        use_container_width=True,
                        key=f"load_opt_{sym}",
                    ):
                        loaded = load_optimization(sym)
                        if loaded:
                            st.session_state['opt_result'] = loaded['result']
                            st.session_state['opt_symbol_used'] = sym
                            st.session_state['opt_meta_loaded'] = loaded.get('meta', {})
                            st.session_state['opt_saved_at_loaded'] = loaded.get('saved_at')
                            st.rerun()
            with opt_hist_cols[-1]:
                if st.button("🗑️ 모두 삭제", use_container_width=True,
                             help="저장된 모든 최적화 결과 삭제"):
                    for sym in list(saved_opts.keys()):
                        delete_optimization(sym)
                    if 'opt_result' in st.session_state:
                        del st.session_state['opt_result']
                    st.rerun()

    # 과최적화 경고
    st.warning(
        "⚠️ **이 도구는 과거에 가장 잘 작동한 파라미터를 찾습니다. "
        "미래 수익을 보장하지 않습니다.** "
        "Train(70%) / Test(30%) 분리 검증으로 과최적화 위험을 최소화하지만, "
        "최종 검증은 페이퍼 트레이딩으로 해야 합니다."
    )

    with st.expander("📚 어떻게 작동하나요? (꼭 읽어보세요)"):
        st.markdown("""
        **1단계 — 데이터 분할**: 전체 데이터를 70% Train + 30% Test로 분리

        **2단계 — Grid Search**: Train 데이터에서 모든 파라미터 조합 백테스트

        **3단계 — Out-of-Sample 검증**: Train 상위 10개를 한 번도 본 적 없는 Test 데이터로 검증

        **4단계 — 진단**:
        - Train↔Test 상관계수 **> 0.5** → ✅ 안정적
        - Train↔Test 상관계수 **0 ~ 0.5** → ⚠️ 주의
        - Train↔Test 상관계수 **< 0** → ❌ 과최적화 (Train에서만 좋았음)

        **결과 해석:**
        - "Train에서 +30%, Test에서 -10%" → 우연히 맞은 것, 실전 사용 금지
        - "Train +10%, Test +8%" → 안정적, 신뢰할 만함
        """)

    st.divider()

    # 설정
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        opt_symbol = st.selectbox(
            "심볼", ["BTC", "ETH"],
            index=0 if active == 'BTC' else 1,
            key="opt_symbol",
            help=f"기본은 사이드바 활성 심볼({active})"
        )
    with opt_col2:
        opt_data_source = st.selectbox(
            "데이터 소스",
            [
                "💎 업비트 15m (1년치) — 추천",
                "💎 업비트 1h",
                "🔸 빗썸 1h",
                "🔸 빗썸 5m → 15m (캐시)",
            ],
            key="opt_data",
        )
    with opt_col3:
        opt_train_pct = st.slider("Train 비율", 0.5, 0.8, 0.7, 0.05, key="opt_train")

    opt_intensity = st.radio(
        "탐색 강도",
        [
            "⚡ 빠름 (24 조합, ~30초)",
            "🎯 표준 (180 조합, ~4분) — 추천",
            "🔬 정밀 (1344 조합, ~25분)",
        ],
        index=1,
        horizontal=False,
    )

    grid_map = {
        "⚡ 빠름 (24 조합, ~30초)": GRID_FAST,
        "🎯 표준 (180 조합, ~4분) — 추천": GRID_STANDARD,
        "🔬 정밀 (1344 조합, ~25분)": GRID_PRECISE,
    }
    selected_grid = grid_map[opt_intensity]

    with st.expander("🔍 탐색할 파라미터 범위 보기"):
        for k, v in selected_grid.items():
            st.write(f"- **{k}**: {v}")

    run_opt = st.button("🚀 최적화 시작", type="primary", use_container_width=True)

    if run_opt:
        try:
            # 데이터 로드
            with st.spinner("데이터 가져오는 중..."):
                if opt_data_source.startswith('💎 업비트 15m'):
                    df_opt = upbit.load_history(opt_symbol, '15m')
                    if df_opt is None or len(df_opt) < 1000:
                        st.info("업비트 15m 캐시 없음. 1년치 다운로드 중 (약 3분)...")
                        prog = st.progress(0)
                        status = st.empty()
                        def cb(f, t, m):
                            prog.progress(min(f / max(t, 1), 1.0))
                            status.text(m)
                        df_opt = upbit.update_history(opt_symbol, '15m', days=365,
                                                     progress_callback=cb)
                        prog.empty()
                        status.empty()
                    max_hold = 32
                elif opt_data_source.startswith('💎 업비트 1h'):
                    df_opt = upbit.load_history(opt_symbol, '1h')
                    if df_opt is None or len(df_opt) < 1000:
                        st.info("업비트 1h 캐시 없음. 1년치 다운로드 중...")
                        df_opt = upbit.update_history(opt_symbol, '1h', days=365)
                    max_hold = 8
                elif opt_data_source.startswith('🔸 빗썸 1h'):
                    df_opt = fetch_candlestick(opt_symbol, 'KRW', '1h')
                    max_hold = 8
                else:  # 빗썸 5m → 15m
                    df_5m = load_history(opt_symbol, 'KRW', '5m')
                    if df_5m is None:
                        st.error(f"{opt_symbol} 빗썸 5분봉 캐시 없음. Data 탭에서 먼저 수집하세요.")
                        st.stop()
                    df_opt = resample(df_5m, '15min')
                    max_hold = 32

            if len(df_opt) < 400:
                st.warning(f"데이터 부족 ({len(df_opt)}봉). 최소 400봉 권장.")
                st.stop()

            st.info(f"데이터: {len(df_opt)}봉 ({df_opt.index[0]} ~ {df_opt.index[-1]})")

            # 진행률 UI
            progress_bar = st.progress(0)
            status_text = st.empty()

            def callback(i, total, phase):
                progress_bar.progress(i / total)
                phase_label = {"train": "🏋️ Train 학습", "test": "🧪 Test 검증"}[phase]
                status_text.text(f"{phase_label}: {i}/{total}")

            opt_sym_lower = opt_symbol.lower()
            opt_max_trades = st.session_state.get(
                f"{opt_sym_lower}_max_trades", 3)
            opt_loss_limit = st.session_state.get(
                f"{opt_sym_lower}_daily_loss_limit", 3.0)

            base_config = BacktestConfig(
                initial_capital=capital,
                fee_pct=fee_pct / 100,
                max_trades_per_day=opt_max_trades,
                daily_loss_limit_pct=opt_loss_limit / 100,
                max_hold_bars=max_hold,
                warmup_bars=min(200, len(df_opt) // 4),
            )

            opt_result = grid_search(
                df_opt,
                param_grid=selected_grid,
                base_config=base_config,
                train_pct=opt_train_pct,
                progress_callback=callback,
            )

            progress_bar.empty()
            status_text.empty()

            # session_state에 저장 (이번 세션용)
            st.session_state['opt_result'] = opt_result
            st.session_state['opt_symbol_used'] = opt_symbol

            # 디스크에 저장 (영구 보관)
            save_optimization(opt_symbol, opt_result, meta={
                'symbol': opt_symbol,
                'data_source': opt_data_source,
                'intensity': opt_intensity,
                'train_pct': opt_train_pct,
                'n_bars': len(df_opt),
                'data_period': f"{df_opt.index[0]} ~ {df_opt.index[-1]}",
            })
            st.success(f"✅ 최적화 완료 + 디스크 저장됨")

        except Exception as e:
            st.error(f"최적화 실패: {e}")
            import traceback
            with st.expander("상세 오류"):
                st.code(traceback.format_exc())

    # 결과 표시
    if 'opt_result' in st.session_state:
        opt_result = st.session_state['opt_result']
        diag = diagnose_overfitting(opt_result)

        st.divider()

        # 어떤 결과인지 헤더 (언제, 무엇으로 돌렸는지)
        sym_used = st.session_state.get('opt_symbol_used', '?')
        meta_loaded = st.session_state.get('opt_meta_loaded')
        saved_at_loaded = st.session_state.get('opt_saved_at_loaded')

        # 디스크에서 최신 메타 가져오기
        if not meta_loaded:
            saved_data = load_optimization(sym_used)
            if saved_data:
                meta_loaded = saved_data.get('meta', {})
                saved_at_loaded = saved_data.get('saved_at')

        with st.container(border=True):
            mh_col1, mh_col2, mh_col3 = st.columns([1, 2, 2])
            with mh_col1:
                st.markdown(f"### 🎯 {sym_used}")
            with mh_col2:
                if saved_at_loaded:
                    from datetime import datetime as _dt
                    dt = _dt.fromisoformat(saved_at_loaded)
                    st.caption(f"**실행 시각**: {dt.strftime('%Y-%m-%d %H:%M')}")
                if meta_loaded:
                    st.caption(f"**데이터**: {meta_loaded.get('data_source', '?')}")
                    st.caption(f"**강도**: {meta_loaded.get('intensity', '?')}")
            with mh_col3:
                if meta_loaded:
                    st.caption(f"**기간**: {meta_loaded.get('data_period', '?')}")
                    st.caption(f"**봉 수**: {meta_loaded.get('n_bars', 0):,}봉")
                    st.caption(f"**Train 비율**: {meta_loaded.get('train_pct', '?')}")

        st.markdown("### 📋 진단 결과")

        # 진단 카드
        level_colors = {
            'good': '#22c55e', 'caution': '#f59e0b',
            'overfit': '#ef4444', 'unknown': '#94a3b8',
            'no_data': '#94a3b8',
        }
        level_emoji = {
            'good': '✅', 'caution': '⚠️',
            'overfit': '❌', 'unknown': '❓', 'no_data': '❓',
        }
        color = level_colors.get(diag['level'], '#94a3b8')

        with st.container(border=True):
            st.markdown(
                f"<div style='border-left:4px solid {color}; padding-left:12px;'>"
                f"<h3 style='margin:0; color:{color};'>"
                f"{level_emoji.get(diag['level'], '?')} {diag['message']}</h3>"
                f"<p style='margin-top:8px;'>📌 <b>권장:</b> {diag['recommendation']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if diag.get('correlation') is not None:
                d_col1, d_col2, d_col3 = st.columns(3)
                d_col1.metric("Train-Test 상관계수",
                              f"{diag['correlation']:.2f}",
                              help=">0.5 안정 / <0 과최적화 의심")
                d_col2.metric("평균 Train 점수", f"{diag['avg_train_score']:.2f}")
                d_col3.metric("평균 Test 점수", f"{diag['avg_test_score']:.2f}",
                              delta=f"{-diag['score_drop']:.2f}",
                              delta_color="inverse")

        # 베스트 파라미터
        if opt_result.get('best_params'):
            st.divider()
            st.markdown("### 🏆 추천 파라미터 (Out-of-Sample 최고)")

            best = opt_result['best_params']
            best_metrics = opt_result.get('best_oos_metrics', {})

            bp_col1, bp_col2 = st.columns(2)

            with bp_col1:
                with st.container(border=True):
                    st.markdown("**📐 파라미터 설정**")
                    st.write(f"- 매수 임계값: **{best['buy_threshold']}**")
                    st.write(f"- 1회 리스크: **{best['risk_per_trade_pct']*100}%**")
                    st.write(f"- ATR 손절 배수: **{best['atr_stop_mult']}**")
                    st.write(f"- 2차 익절 (R): **{best['tp2_r']}**")
                    st.write(f"- MTF 사용: **{best['require_mtf']}**")
                    st.write(f"- 트레일링: **{best['use_trailing']}**")

                    st.divider()
                    # 자동 적용 버튼: 활성 심볼의 프로파일에 적용
                    apply_label = f"✅ {active} 사이드바에 자동 적용"
                    if st.button(apply_label,
                                 type="primary", use_container_width=True,
                                 key="apply_best"):
                        sl = active.lower()
                        st.session_state[f'{sl}_buy_threshold'] = float(best['buy_threshold'])
                        st.session_state[f'{sl}_risk_pct'] = float(best['risk_per_trade_pct'] * 100)
                        st.session_state[f'{sl}_atr_stop_mult'] = float(best['atr_stop_mult'])
                        st.session_state[f'{sl}_tp2_r'] = float(best['tp2_r'])
                        st.session_state[f'{sl}_require_mtf'] = bool(best['require_mtf'])
                        st.session_state[f'{sl}_use_trailing'] = bool(best['use_trailing'])
                        # 자동 저장도 함께
                        save_profile(active, {
                            'buy_threshold': float(best['buy_threshold']),
                            'risk_pct': float(best['risk_per_trade_pct'] * 100),
                            'atr_stop_mult': float(best['atr_stop_mult']),
                            'tp2_r': float(best['tp2_r']),
                            'require_mtf': bool(best['require_mtf']),
                            'use_trailing': bool(best['use_trailing']),
                            'max_trades': st.session_state.get(f'{sl}_max_trades', 3),
                            'daily_loss_limit': st.session_state.get(f'{sl}_daily_loss_limit', 3.0),
                        })
                        st.success(f"✅ {active} 프로파일에 적용 + 디스크 저장됨")
                        st.rerun()

            with bp_col2:
                with st.container(border=True):
                    st.markdown("**📊 Test 데이터 성과 (out-of-sample)**")
                    if best_metrics:
                        st.metric("수익률", f"{best_metrics.get('total_return_pct', 0)*100:+.2f}%")
                        st.metric("승률", f"{best_metrics.get('win_rate', 0)*100:.1f}%")
                        st.metric("거래 수", best_metrics.get('n_trades', 0))
                        st.metric("MDD", f"{best_metrics.get('mdd', 0)*100:.2f}%")
                        st.metric("샤프 비율", f"{best_metrics.get('sharpe', 0):.2f}")

        # Top 10 비교 테이블
        st.divider()
        st.markdown("### 🏅 Top 10 결과 (Train vs Test)")
        st.caption("Train과 Test 결과가 비슷할수록 신뢰할 만합니다. 차이가 크면 과최적화.")

        top10_data = []
        for r in opt_result['oos_results']:
            p = r['params']
            tr = r['train_metrics']
            te = r['test_metrics']
            top10_data.append({
                '임계값': p['buy_threshold'],
                'ATR×': p['atr_stop_mult'],
                'TP2(R)': p['tp2_r'],
                '리스크%': f"{p['risk_per_trade_pct']*100:.1f}",
                'Train점수': f"{r['train_score']:.2f}",
                'Test점수': f"{r['test_score']:.2f}" if r['test_score'] > -100 else "X",
                'Train수익': f"{tr.get('total_return_pct', 0)*100:+.1f}%",
                'Test수익': f"{te.get('total_return_pct', 0)*100:+.1f}%",
                'Train거래': tr.get('n_trades', 0),
                'Test거래': te.get('n_trades', 0),
            })

        st.dataframe(pd.DataFrame(top10_data), use_container_width=True, hide_index=True)

        # Train-Test 산점도
        st.markdown("### 📈 Train vs Test 점수 산점도")
        st.caption("점들이 우상향 직선에 가까이 있을수록 안정적입니다. 흩어져 있으면 과최적화.")

        oos_scatter = opt_result['oos_results']
        train_pts = [r['train_score'] for r in oos_scatter]
        test_pts = [r['test_score'] for r in oos_scatter if r['test_score'] > -100]

        if len(test_pts) >= 3:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=[r['train_score'] for r in oos_scatter if r['test_score'] > -100],
                y=test_pts, mode='markers',
                marker=dict(size=12, color='#3b82f6', line=dict(width=1, color='white')),
                name='파라미터 조합',
            ))
            # 이상적인 y=x 라인
            min_v = min(min(train_pts), min(test_pts)) - 0.5
            max_v = max(max(train_pts), max(test_pts)) + 0.5
            fig_scatter.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode='lines', line=dict(color='gray', dash='dash'),
                name='이상적 (y=x)',
            ))
            fig_scatter.update_layout(
                xaxis_title="Train 점수", yaxis_title="Test 점수",
                height=400, margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# =====================================================================
# Tab 4: Data Management
# =====================================================================
with tab_data:
    st.subheader("💾 데이터 캐시 상태")

    cache_dir = Path('./data')
    if not cache_dir.exists():
        st.info("아직 캐시가 없습니다. 아래 버튼으로 데이터를 수집하세요.")
    else:
        files = list(cache_dir.glob('*.parquet'))
        if not files:
            st.info("캐시 파일이 없습니다.")
        else:
            cache_info = []
            for f in files:
                df = pd.read_parquet(f)
                cache_info.append({
                    '파일': f.name,
                    '봉 수': len(df),
                    '시작': str(df.index[0]),
                    '끝': str(df.index[-1]),
                    '크기 (KB)': f.stat().st_size // 1024,
                })
            st.dataframe(pd.DataFrame(cache_info), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔄 수동 업데이트")

    upd_col1, upd_col2 = st.columns(2)
    with upd_col1:
        upd_symbols = st.multiselect("심볼", ["BTC", "ETH"], default=["BTC", "ETH"])
    with upd_col2:
        upd_interval = st.selectbox(
            "Interval",
            ["5m", "10m", "30m", "1h"],
            help="5m을 누적해두면 15m 백테스트에 사용 가능"
        )

    if st.button("📥 데이터 수집", use_container_width=True):
        for sym in upd_symbols:
            with st.spinner(f"{sym} 수집 중..."):
                try:
                    df = update_history(sym, 'KRW', upd_interval, str(cache_dir))
                    st.success(f"✅ {sym}: {len(df)}봉 누적됨")
                except Exception as e:
                    st.error(f"❌ {sym} 실패: {e}")
        time.sleep(0.5)
        st.rerun()

    st.divider()
    st.subheader("💎 업비트 historical 다운로드 (장기 백테스트용)")
    st.caption(
        "업비트는 페이지네이션 지원이라 1년치 데이터를 한 번에 받을 수 있어요. "
        "백테스트 정확도가 크게 향상됩니다."
    )

    upb_col1, upb_col2, upb_col3 = st.columns(3)
    with upb_col1:
        upb_symbols = st.multiselect("심볼", ["BTC", "ETH"],
                                     default=["BTC", "ETH"], key="upb_syms")
    with upb_col2:
        upb_interval = st.selectbox("Interval",
                                    ["15m", "5m", "1h", "30m"], key="upb_intv")
    with upb_col3:
        upb_days = st.slider("기간 (일)", 30, 730, 365, 30, key="upb_days")

    upb_estimate = upb_days * 24 * 60 // {'1m': 1, '5m': 5, '15m': 15,
                                          '30m': 30, '1h': 60}[upb_interval]
    st.info(f"📊 예상 수집량: 약 **{upb_estimate:,}봉** "
            f"(소요 시간 약 {upb_estimate // 200 * 0.2 / 60:.1f}분)")

    if st.button("💎 업비트 다운로드", use_container_width=True, type="primary"):
        for sym in upb_symbols:
            st.write(f"**{sym}** 수집 중...")
            prog = st.progress(0)
            status = st.empty()
            def make_cb(prog, status):
                def cb(f, t, m):
                    prog.progress(min(f / max(t, 1), 1.0))
                    status.text(m)
                return cb
            try:
                df = upbit.update_history(
                    sym, upb_interval, str(cache_dir), days=upb_days,
                    progress_callback=make_cb(prog, status),
                )
                prog.progress(1.0)
                status.text(f"✅ {len(df)}봉 완료")
                st.success(f"✅ {sym} {upb_interval}: {len(df)}봉 "
                           f"({df.index[0]} ~ {df.index[-1]})")
            except Exception as e:
                st.error(f"❌ {sym} 실패: {e}")
        time.sleep(0.5)
        st.rerun()

    st.divider()
    st.subheader("📅 자동 수집 설정 (cron)")
    st.markdown("""
    데이터를 자동으로 누적하려면 터미널에 입력:
    ```bash
    crontab -e
    ```
    그리고 아래 줄 추가 (30분마다 자동 수집):
    ```
    */30 * * * * cd /your/project/path && python -m trading.bithumb 5m >> data/log.txt 2>&1
    ```
    """)


# =====================================================================
# Footer
# =====================================================================
st.divider()
st.caption(
    "⚠️ 이 도구는 백테스트 및 신호 분석 보조용입니다. "
    "투자 결정은 본인 책임이며, 실거래 전 충분한 검증과 모의투자를 권장합니다."
)