import io
from datetime import date
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mantis_backtest import Config, download_us, download_kr, download_kr_benchmark, add_features, baseline_signals, backtest_one

st.set_page_config(page_title="MANTIS Stock Lab", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 12px; border-radius: 12px;}
.stButton>button {width:100%; min-height: 3rem; font-weight: 700;}
.small-note {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("📈 MANTIS Stock Lab")
st.caption("일봉 데이터 기반 연구용 복합지표 · 투자수익을 보장하지 않습니다")

with st.sidebar:
    st.header("분석 설정")
    market_label = st.radio("시장", ["미국", "한국"], horizontal=True)
    market = "US" if market_label == "미국" else "KR"
    default_tickers = "NVDA,MSFT,AMZN" if market == "US" else "005930,000660,035420"
    tickers_raw = st.text_input("종목 코드 (쉼표로 구분)", default_tickers)
    benchmark = st.text_input("벤치마크", "SPY" if market == "US" else "1001")
    start_date = st.date_input("시작일", date(2016,1,1), min_value=date(2000,1,1))
    end_date = st.date_input("종료일", date.today())
    st.divider()
    fee_bps = st.number_input("수수료 (bps, 편도)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    slip_bps = st.number_input("슬리피지 (bps, 편도)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    max_hold = st.number_input("최대 보유 거래일", min_value=5, max_value=250, value=60, step=5)
    run = st.button("🚀 백테스트 실행", type="primary")

st.info("처음 실행 시 데이터 다운로드에 시간이 걸릴 수 있습니다. 한국 종목은 KRX 데이터, 미국 종목은 Yahoo Finance 데이터를 사용합니다.")

if run:
    tickers = [x.strip().upper() for x in tickers_raw.split(",") if x.strip()]
    if not tickers:
        st.error("최소 한 개의 종목 코드를 입력하세요.")
        st.stop()
    if start_date >= end_date:
        st.error("종료일은 시작일보다 뒤여야 합니다.")
        st.stop()
    cfg = Config(start=str(start_date), end=str(end_date), benchmark=benchmark,
                 max_hold=int(max_hold), fee_bps=float(fee_bps), slippage_bps=float(slip_bps))
    summaries, trades_all, feature_frames, errors = [], [], {}, []
    progress = st.progress(0, text="데이터를 준비하고 있습니다...")
    for idx, ticker in enumerate(tickers):
        try:
            if market == "US":
                raw = download_us(ticker, cfg.start, cfg.end)
                b = download_us(benchmark, cfg.start, cfg.end)["close"]
                features = add_features(raw, b, use_flows=False)
            else:
                raw = download_kr(ticker, cfg.start, cfg.end)
                b = download_kr_benchmark(benchmark, cfg.start, cfg.end)
                features = add_features(raw, b, use_flows=True)
            feature_frames[ticker] = features
            for strategy, signal in baseline_signals(features).items():
                trade_df, stats = backtest_one(features, signal, cfg)
                stats.update({"ticker": ticker, "strategy": strategy})
                summaries.append(stats)
                if not trade_df.empty:
                    trade_df.insert(0, "strategy", strategy)
                    trade_df.insert(0, "ticker", ticker)
                    trades_all.append(trade_df)
        except Exception as exc:
            errors.append(f"{ticker}: {type(exc).__name__}: {exc}")
        progress.progress((idx+1)/len(tickers), text=f"분석 중: {ticker} ({idx+1}/{len(tickers)})")
    progress.empty()
    if not summaries:
        st.error("분석에 성공한 종목이 없습니다.")
        for err in errors: st.write("- " + err)
        st.stop()
    summary = pd.DataFrame(summaries)
    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    st.session_state["mantis_summary"] = summary
    st.session_state["mantis_trades"] = trades
    st.session_state["mantis_features"] = feature_frames
    st.session_state["mantis_errors"] = errors

if "mantis_summary" in st.session_state:
    summary = st.session_state["mantis_summary"]
    trades = st.session_state["mantis_trades"]
    features_map = st.session_state["mantis_features"]
    errors = st.session_state["mantis_errors"]

    st.subheader("전략 성과 비교")
    display = summary.copy()
    for col in ["win_rate", "total_return_compounded", "max_drawdown_trade_equity", "avg_trade"]:
        if col in display: display[col] = display[col].map(lambda v: f"{v:.1%}" if pd.notna(v) and abs(v) != float("inf") else "-" if pd.isna(v) else str(v))
    if "profit_factor" in display:
        display["profit_factor"] = display["profit_factor"].map(lambda v: f"{v:.2f}" if pd.notna(v) and abs(v) != float("inf") else "∞" if v == float("inf") else "-")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("MANTIS 신호 차트")
    selected = st.selectbox("차트 종목", list(features_map.keys()))
    f = features_map[selected].dropna(subset=["close"])
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=f.index, open=f.open, high=f.high, low=f.low, close=f.close, name="가격"))
    fig.add_trace(go.Scatter(x=f.index, y=f.sma20, name="SMA20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=f.index, y=f.sma60, name="SMA60", line=dict(width=1)))
    entries = f[f["mantis_entry"].fillna(False)]
    fig.add_trace(go.Scatter(x=entries.index, y=entries.close, mode="markers", name="MANTIS 매수 후보", marker=dict(symbol="triangle-up", size=11)))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10), legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    score_fig = go.Figure(go.Scatter(x=f.index, y=f.mantis_score, mode="lines", name="MANTIS 점수"))
    score_fig.add_hline(y=70, line_dash="dash", annotation_text="기준점수 70")
    score_fig.update_layout(height=240, yaxis_title="점수", margin=dict(l=10,r=10,t=25,b=10))
    st.plotly_chart(score_fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    mantis = summary[summary.strategy == "MANTIS"]
    c1.metric("MANTIS 거래 수", f"{int(mantis.trades.sum()) if not mantis.empty else 0}회")
    c2.metric("평균 승률", f"{mantis.win_rate.mean():.1%}" if not mantis.empty and mantis.win_rate.notna().any() else "데이터 없음")
    c3.metric("평균 손익비 지표", f"{mantis.profit_factor.replace([float('inf')], pd.NA).mean():.2f}" if not mantis.empty and mantis.profit_factor.notna().any() else "데이터 없음")

    st.subheader("거래 내역")
    if trades.empty:
        st.write("해당 조건에서 발생한 거래가 없습니다.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button("거래 내역 CSV 다운로드", trades.to_csv(index=False).encode("utf-8-sig"), "mantis_trades.csv", "text/csv")
    st.download_button("전략 성과 CSV 다운로드", summary.to_csv(index=False).encode("utf-8-sig"), "mantis_summary.csv", "text/csv")
    if errors:
        with st.expander("일부 종목 분석 오류"):
            for err in errors: st.write("- " + err)
else:
    st.markdown("### 시작하기")
    st.markdown("1. 왼쪽에서 시장과 종목 코드를 확인합니다.\n2. 분석 기간과 벤치마크를 선택합니다.\n3. **백테스트 실행**을 누릅니다.\n4. 전략 비교표, 가격 차트, MANTIS 점수, 거래 내역을 확인합니다.")
    st.caption("MANTIS는 연구용 프로토타입입니다. 과거 성과는 미래 수익을 보장하지 않으며, 이 화면의 결과만으로 매매하지 마세요.")
