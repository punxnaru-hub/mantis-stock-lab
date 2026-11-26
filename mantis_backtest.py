#!/usr/bin/env python3
"""
MANTIS v1.0 — reproducible daily-bar research prototype.

Important:
- This is a research implementation, not a claim of profitability.
- Supply a point-in-time universe to avoid survivorship bias.
- Korean foreign/institutional flows are fetched through pykrx where available.
- US version intentionally does not pretend that public daily volume is investor identity.
- Signals are calculated on close; entries execute at next session open.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from pykrx import stock as krx
except ImportError:
    krx = None


@dataclass
class Config:
    start: str = "2016-01-01"
    end: str = "2025-12-31"
    benchmark: str = "SPY"
    max_hold: int = 60
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    stop_atr: float = 2.5
    initial_cash: float = 10_000_000
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def download_us(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("Install yfinance: pip install yfinance")
    d = yf.download(ticker, start=start, end=end, auto_adjust=False,
                    progress=False, actions=False)
    if d.empty:
        raise ValueError(f"No Yahoo Finance data for {ticker}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)
    required = ["open", "high", "low", "close", "volume"]
    d = d[required].copy()
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d.dropna()


def download_kr(ticker: str, start: str, end: str) -> pd.DataFrame:
    if krx is None:
        raise RuntimeError("Install pykrx: pip install pykrx")
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    raw = krx.get_market_ohlcv_by_date(s, e, ticker)
    if raw.empty:
        raise ValueError(f"No KRX data for {ticker}")
    # pykrx columns are Korean labels.
    rename = {"시가": "open", "고가": "high", "저가": "low",
              "종가": "close", "거래량": "volume"}
    d = raw.rename(columns=rename)[["open", "high", "low", "close", "volume"]].copy()
    d.index = pd.to_datetime(d.index)
    # pykrx returns investor categories as columns; values are net traded value (KRW).
    try:
        flow = krx.get_market_trading_value_by_date(s, e, ticker)
        if flow is not None and not flow.empty:
            # Column labels vary slightly by pykrx version.
            foreign_col = next((c for c in flow.columns if "외국인" in str(c)), None)
            inst_col = next((c for c in flow.columns if "기관합계" in str(c)), None)
            if foreign_col is not None:
                d["foreign_value"] = pd.to_numeric(flow[foreign_col], errors="coerce")
            if inst_col is not None:
                d["institution_value"] = pd.to_numeric(flow[inst_col], errors="coerce")
    except Exception as exc:
        warnings.warn(f"Investor-flow download failed ({ticker}): {exc}")
    return d.sort_index()


def download_kr_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Download KOSPI/KOSDAQ index close using pykrx's index endpoint."""
    if krx is None:
        raise RuntimeError("Install pykrx: pip install pykrx")
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    raw = krx.get_index_ohlcv_by_date(s, e, ticker)
    if raw.empty:
        raise ValueError(f"No KRX index data for benchmark {ticker}")
    close_col = "종가" if "종가" in raw.columns else raw.columns[3]
    result = pd.to_numeric(raw[close_col], errors="coerce")
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def add_features(d: pd.DataFrame, benchmark_close: Optional[pd.Series] = None,
                 use_flows: bool = False) -> pd.DataFrame:
    x = d.copy().sort_index()
    o, h, l, c, v = (x[k].astype(float) for k in ["open", "high", "low", "close", "volume"])
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["sma20"] = c.rolling(20).mean()
    x["sma60"] = c.rolling(60).mean()
    x["sma20_slope"] = x["sma20"].pct_change(5)
    x["sma60_slope"] = x["sma60"].pct_change(10)

    # Accumulation: OBV slope, Chaikin money flow, and up/down volume imbalance.
    direction = np.sign(c.diff()).fillna(0)
    x["obv"] = (direction * v).cumsum()
    x["obv_slope"] = x["obv"].pct_change(20)
    mfm = _safe_div((c-l) - (h-c), (h-l)).clip(-1, 1).fillna(0)
    x["cmf20"] = _safe_div((mfm*v).rolling(20).sum(), v.rolling(20).sum())
    upv = (v * (c > prev)).rolling(20).sum()
    dnv = (v * (c < prev)).rolling(20).sum()
    x["up_down_ratio"] = _safe_div(upv, dnv + 1)

    # Investor flow is normalized by traded value, not raw won.
    traded_value = c * v
    if use_flows:
        for col in ["foreign_value", "institution_value"]:
            if col not in x:
                x[col] = np.nan
            x[col + "_ratio5"] = _safe_div(x[col].rolling(5).sum(), traded_value.rolling(5).sum())
            x[col + "_ratio20"] = _safe_div(x[col].rolling(20).sum(), traded_value.rolling(20).sum())
    else:
        x["foreign_value_ratio5"] = np.nan
        x["foreign_value_ratio20"] = np.nan
        x["institution_value_ratio5"] = np.nan
        x["institution_value_ratio20"] = np.nan

    # Relative strength vs benchmark; align to asset sessions.
    if benchmark_close is not None:
        b = benchmark_close.reindex(x.index).ffill()
        rel = c / b
        x["rs20"] = rel.pct_change(20)
        x["rs60"] = rel.pct_change(60)
    else:
        x["rs20"] = c.pct_change(20)
        x["rs60"] = c.pct_change(60)

    # Volatility contraction and breakout quality.
    x["atr_pct"] = _safe_div(x["atr"], c)
    x["atr_pct_rank"] = x["atr_pct"].rolling(120, min_periods=60).rank(pct=True)
    x["range20"] = _safe_div(c.rolling(20).max() - c.rolling(20).min(), c)
    x["volume_ratio"] = _safe_div(v, v.rolling(20).mean())
    x["close_location"] = _safe_div(c-l, h-l).clip(0, 1)
    x["breakout"] = (c > h.shift(1).rolling(20).max()).astype(int)

    # Standardized, clipped component scores. Rolling z-scores use only past data.
    def zscore(s, window=120):
        mu = s.shift(1).rolling(window, min_periods=40).mean()
        sd = s.shift(1).rolling(window, min_periods=40).std()
        return ((s-mu)/sd.replace(0, np.nan)).clip(-2.5, 2.5).add(2.5).div(5).mul(100)

    accumulation_raw = (
        zscore(x["obv_slope"]) * 0.35 +
        zscore(x["cmf20"]) * 0.35 +
        zscore(np.log1p(x["up_down_ratio"].clip(lower=0))) * 0.30
    )
    trend_raw = (
        zscore(x["sma20_slope"]) * 0.45 +
        zscore(x["sma60_slope"]) * 0.35 +
        zscore(_safe_div(c, x["sma20"]) - 1) * 0.20
    )
    relative_raw = zscore(x["rs20"]) * 0.55 + zscore(x["rs60"]) * 0.45
    compression_raw = (100 - x["atr_pct_rank"] * 100).clip(0, 100)

    if use_flows:
        foreign = zscore(x["foreign_value_ratio5"]) * 0.6 + zscore(x["foreign_value_ratio20"]) * 0.4
        inst = zscore(x["institution_value_ratio5"]) * 0.6 + zscore(x["institution_value_ratio20"]) * 0.4
        flow_raw = (foreign + inst) / 2
        # Missing flow must not be silently interpreted as zero/positive.
        flow_raw = flow_raw.where(x["foreign_value_ratio5"].notna() |
                                  x["institution_value_ratio5"].notna())
    else:
        flow_raw = pd.Series(np.nan, index=x.index)

    # If flow unavailable (US), redistribute its 20% weight to accumulation and relative strength.
    if use_flows:
        x["mantis_score"] = (0.25*accumulation_raw + 0.20*flow_raw +
                             0.20*trend_raw + 0.15*relative_raw +
                             0.10*compression_raw + 0.10*zscore(x["volume_ratio"]))
    else:
        x["mantis_score"] = (0.35*accumulation_raw + 0.25*trend_raw +
                             0.20*relative_raw + 0.10*compression_raw +
                             0.10*zscore(x["volume_ratio"]))

    # Hard gates: score alone cannot trigger a trade.
    x["mantis_entry"] = (
        (x["mantis_score"] >= 70) &
        (c > x["sma20"]) &
        (x["sma20"] > x["sma60"]) &
        (x["volume_ratio"] >= 1.2) &
        (x["close_location"] >= 0.65) &
        (x["breakout"] == 1)
    )
    return x


def baseline_signals(x: pd.DataFrame) -> dict[str, pd.Series]:
    c = x.close
    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100/(1 + _safe_div(gain, loss))
    # MACD(12,26,9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    # Bollinger(20,2): close crosses upper band
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    upper = mid + 2*sd
    # Envelope(20, +5%): close crosses upper envelope
    env = mid * 1.05
    return {
        "MANTIS": x["mantis_entry"].fillna(False),
        "RSI_recovery": ((rsi.shift(1) < 30) & (rsi >= 30)),
        "MACD_cross": ((macd.shift(1) <= sig.shift(1)) & (macd > sig)),
        "Bollinger_break": ((c.shift(1) <= upper.shift(1)) & (c > upper)),
        "Envelope_break": ((c.shift(1) <= env.shift(1)) & (c > env)),
    }


def backtest_one(x: pd.DataFrame, entry_signal: pd.Series, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One ticker, one open position at a time; close signal at t, fill at t+1 open."""
    x = x.dropna(subset=["open", "high", "low", "close", "atr"]).copy()
    sig = entry_signal.reindex(x.index).fillna(False)
    fee = (cfg.fee_bps + cfg.slippage_bps) / 10000
    trades = []
    in_pos = False
    entry_i = None
    entry_px = stop_px = None
    for i in range(len(x)-1):
        row = x.iloc[i]
        nxt = x.iloc[i+1]
        if not in_pos and bool(sig.iloc[i]):
            entry_px = float(nxt.open) * (1 + fee)
            entry_i = i+1
            stop_px = entry_px - cfg.stop_atr * float(row.atr)
            in_pos = True
            continue
        if in_pos:
            held = i - entry_i + 1
            exit_reason = None
            # Stop evaluated against today's low; conservative gap handling.
            if float(row.low) <= stop_px:
                raw_exit = min(float(row.open), stop_px) if float(row.open) < stop_px else stop_px
                exit_reason = "ATR_stop"
            elif float(row.close) < float(row.sma20):
                raw_exit = float(nxt.open)
                exit_reason = "close_below_SMA20"
            elif held >= cfg.max_hold:
                raw_exit = float(nxt.open)
                exit_reason = "max_hold"
            else:
                continue
            exit_px = raw_exit * (1 - fee)
            ret = exit_px / entry_px - 1
            trades.append({"entry_date": x.index[entry_i], "exit_date": x.index[i] if exit_reason=="ATR_stop" else x.index[i+1],
                           "entry": entry_px, "exit": exit_px, "return": ret,
                           "bars": held, "reason": exit_reason})
            in_pos = False
            entry_i = entry_px = stop_px = None
    if in_pos:
        exit_px = float(x.close.iloc[-1]) * (1-fee)
        trades.append({"entry_date": x.index[entry_i], "exit_date": x.index[-1],
                       "entry": entry_px, "exit": exit_px, "return": exit_px/entry_px-1,
                       "bars": len(x)-entry_i, "reason": "end_of_data"})
    t = pd.DataFrame(trades)
    if t.empty:
        return t, {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                   "avg_trade": np.nan, "total_return_compounded": 0.0,
                   "max_drawdown_trade_equity": 0.0}
    eq = (1+t["return"]).cumprod()
    dd = eq/eq.cummax()-1
    wins = t.loc[t["return"] > 0, "return"].sum()
    losses = -t.loc[t["return"] < 0, "return"].sum()
    stats = {
        "trades": len(t),
        "win_rate": float((t["return"] > 0).mean()),
        "profit_factor": float(wins/losses) if losses > 0 else np.inf,
        "avg_trade": float(t["return"].mean()),
        "median_trade": float(t["return"].median()),
        "total_return_compounded": float(eq.iloc[-1]-1),
        "max_drawdown_trade_equity": float(dd.min()),
        "avg_holding_bars": float(t["bars"].mean()),
    }
    return t, stats


def run_symbol(ticker: str, market: str, cfg: Config, benchmark: Optional[str] = None):
    if market == "US":
        d = download_us(ticker, cfg.start, cfg.end)
        b = download_us(benchmark or cfg.benchmark, cfg.start, cfg.end).close
        x = add_features(d, b, use_flows=False)
    elif market == "KR":
        d = download_kr(ticker, cfg.start, cfg.end)
        # KOSPI index ticker 1001; use KOSDAQ 2001 if required.
        b = download_kr_benchmark(benchmark or "1001", cfg.start, cfg.end)
        x = add_features(d, b, use_flows=True)
    else:
        raise ValueError("market must be KR or US")

    strategies = baseline_signals(x)
    rows, all_trades = [], []
    for name, sig in strategies.items():
        trades, stats = backtest_one(x, sig, cfg)
        stats.update({"ticker": ticker, "strategy": name})
        rows.append(stats)
        if not trades.empty:
            trades.insert(0, "strategy", name)
            trades.insert(0, "ticker", ticker)
            all_trades.append(trades)
    return pd.DataFrame(rows), (pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame())


def main():
    p = argparse.ArgumentParser(description="MANTIS v1.0 backtest")
    p.add_argument("--market", choices=["KR", "US"], required=True)
    p.add_argument("--tickers", required=True, help="Comma-separated tickers; KR examples 005930,000660; US examples NVDA,MSFT")
    p.add_argument("--benchmark", default=None, help="US: SPY; KR: 1001 (KOSPI) or 2001 (KOSDAQ)")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--out", default="mantis_results")
    args = p.parse_args()
    cfg = Config(start=args.start, end=args.end)
    summary, trades = [], []
    for ticker in [z.strip() for z in args.tickers.split(",") if z.strip()]:
        try:
            s, t = run_symbol(ticker, args.market, cfg, args.benchmark)
            summary.append(s)
            if not t.empty:
                trades.append(t)
            print(f"Completed {ticker}")
        except Exception as exc:
            print(f"FAILED {ticker}: {exc}")
    if not summary:
        raise SystemExit("No ticker completed. Check symbols, internet, and data packages.")
    out = pd.concat(summary, ignore_index=True)
    out.to_csv(args.out + "_summary.csv", index=False)
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(args.out + "_trades.csv", index=False)
    print("\n=== Summary ===")
    print(out.to_string(index=False))
    print(f"\nSaved {args.out}_summary.csv and {args.out}_trades.csv (if trades exist).")
    print("Caution: per-symbol results are not a portfolio-level, point-in-time-universe backtest.")
    print("Use a delisted-inclusive historical universe before making investment conclusions.")


if __name__ == "__main__":
    main()
