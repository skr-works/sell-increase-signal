import os
import json
import math
import time
import random
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import jpholiday
import gspread
from google.oauth2.service_account import Credentials
from zoneinfo import ZoneInfo
import datetime as dt


JST = ZoneInfo("Asia/Tokyo")

# ダウンロード結果キャッシュ（スプシの銘柄コード -> 日足DF）
_DATA_CACHE: Dict[str, pd.DataFrame] = {}


# -------------------------
# 休場日スキップ（JST）
# -------------------------
def is_skip_day_jst(today: dt.date) -> Tuple[bool, str]:
    if today.weekday() >= 5:
        return True, "土日のためスキップ"
    if jpholiday.is_holiday(today):
        return True, "祝日のためスキップ"
    if (today.month == 12 and today.day == 31) or (today.month == 1 and today.day in (1, 2, 3)):
        return True, "年末年始（12/31〜1/3）のためスキップ"
    return False, ""


# -------------------------
# 設定
# -------------------------
@dataclass(frozen=True)
class Settings:
    atr_periods: Tuple[int, int, int] = (14, 50, 100)
    stop_multipliers: Tuple[int, int] = (2, 3)
    near_atr_factor: float = 0.5
    ret_lookback: int = 20
    vol_short: int = 20
    vol_long: int = 60
    vol_spike_ratio: float = 1.5
    history_years: int = 2

    # レート制限対策（最大1000銘柄想定）
    yf_batch_size: int = 50
    yf_batch_sleep: float = 1.2
    yf_retry_max: int = 3
    yf_retry_base: float = 2.0


# -------------------------
# Secretsのパース
# -------------------------
def _relaxed_json_loads(raw: str) -> Any:
    s = raw
    out_chars: List[str] = []
    in_string = False
    escape = False

    for ch in s:
        if in_string:
            if escape:
                out_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                out_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                out_chars.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out_chars.append("\\n")
                continue
            if ch == "\r":
                out_chars.append("\\r")
                continue
            if ch == "\t":
                out_chars.append("\\t")
                continue
            out_chars.append(ch)
        else:
            if ch == '"':
                out_chars.append(ch)
                in_string = True
                escape = False
                continue
            out_chars.append(ch)

    fixed = "".join(out_chars)
    return json.loads(fixed)


def _brace_balanced_json_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
    buf: List[str] = []
    brace = 0
    in_string = False
    escape = False

    i = start_idx
    while i < len(lines):
        line = lines[i]
        buf.append(line)
        for ch in line:
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                    continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
        if brace == 0 and any("{" in x for x in buf):
            return "\n".join(buf).strip(), i + 1
        i += 1
    return "\n".join(buf).strip(), len(lines)


def _parse_kv_multiline(raw: str) -> Dict[str, Any]:
    lines = raw.splitlines()
    out: Dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        sep_pos = None
        for s in ["=", ":"]:
            p = line.find(s)
            if p != -1:
                sep_pos = p
                break
        if sep_pos is None:
            continue
        key = line[:sep_pos].strip()
        val = line[sep_pos + 1 :].strip()
        if (val == "" and i < len(lines) and lines[i].lstrip().startswith("{")) or val.startswith("{"):
            block_start = (i - 1) if val.startswith("{") else i
            block_text, next_i = _brace_balanced_json_block(lines, block_start)
            try:
                out[key] = _relaxed_json_loads(block_text)
            except Exception:
                out[key] = block_text
            i = next_i
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        low = val.lower()
        if low in ("true", "false"):
            out[key] = (low == "true")
        else:
            try:
                out[key] = float(val) if "." in val else int(val)
            except Exception:
                vv = val.strip()
                if (vv.startswith("{") and vv.endswith("}")) or (vv.startswith("[") and vv.endswith("]")):
                    try:
                        out[key] = _relaxed_json_loads(vv)
                    except Exception:
                        out[key] = val
                else:
                    out[key] = val
    return out


def parse_app_secrets(raw: str) -> Dict[str, Any]:
    try:
        v = json.loads(raw)
        if isinstance(v, dict): return v
    except Exception:
        pass
    try:
        v = _relaxed_json_loads(raw)
        if isinstance(v, dict): return v
    except Exception:
        pass
    v = _parse_kv_multiline(raw)
    if isinstance(v, dict) and v: return v
    raise RuntimeError("APP_SECRETS_JSON を解釈できません")


def load_secrets() -> Tuple[Dict[str, Any], Settings]:
    raw = os.environ.get("APP_SECRETS_JSON", "")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("APP_SECRETS_JSON が未設定です")
    secrets = parse_app_secrets(raw)
    if secrets.get("google_service_account_json") is None:
        raise RuntimeError("google_service_account_json が未設定です")
    settings_in = secrets.get("settings", {}) or {}
    s = Settings(
        atr_periods=tuple(settings_in.get("atr_periods", [14, 50, 100])),
        stop_multipliers=tuple(settings_in.get("stop_multipliers", [2, 3])),
        near_atr_factor=float(settings_in.get("near_atr_factor", 0.5)),
        ret_lookback=int(settings_in.get("ret_lookback", 20)),
        vol_short=int(settings_in.get("vol_short", 20)),
        vol_long=int(settings_in.get("vol_long", 60)),
        vol_spike_ratio=float(settings_in.get("vol_spike_ratio", 1.5)),
        history_years=int(settings_in.get("history_years", 2)),
        yf_batch_size=int(settings_in.get("yf_batch_size", 50)),
        yf_batch_sleep=float(settings_in.get("yf_batch_sleep", 1.2)),
        yf_retry_max=int(settings_in.get("yf_retry_max", 3)),
        yf_retry_base=float(settings_in.get("yf_retry_base", 2.0)),
    )
    return secrets, s


# -------------------------
# ティッカー正規化
# -------------------------
def normalize_yf_ticker(raw_ticker: str) -> str:
    t = (raw_ticker or "").strip()
    if not t: return t
    if t[0].isdigit() and ("." not in t):
        return t + ".T"
    return t


# -------------------------
# 指標計算
# -------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - prev_close).abs()
    lc = (df["Low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr_sma(df: pd.DataFrame, n: int) -> Optional[float]:
    if df is None or len(df) < n + 1: return None
    tr = true_range(df)
    v = tr.rolling(window=n).mean().iloc[-1]
    return None if pd.isna(v) else float(v)


def sma(series: pd.Series, n: int) -> Optional[float]:
    if series is None or len(series) < n: return None
    v = series.rolling(window=n).mean().iloc[-1]
    return None if pd.isna(v) else float(v)


def ret_n(series: pd.Series, n: int) -> Optional[float]:
    if series is None or len(series) < n + 1: return None
    v = float(series.iloc[-1] / series.iloc[-1 - n] - 1.0)
    return v


def vol_annualized(series: pd.Series, n: int) -> Optional[float]:
    if series is None or len(series) < n + 1: return None
    r = series.pct_change().dropna().iloc[-n:]
    if len(r) < n: return None
    return float(r.std(ddof=0) * math.sqrt(252.0))


def stop_proximity(price: Optional[float], stop_price: Optional[float], atr_val: Optional[float], near_factor: float) -> str:
    if price is None or stop_price is None or atr_val is None: return ""
    if price <= stop_price: return "割れ"
    if price <= stop_price + near_factor * atr_val: return "接近"
    return "OK"


def compute_row_outputs(ticker: str, cost: Optional[float], settings: Settings) -> List[Any]:
    df = _DATA_CACHE.get(ticker)
    needed = {"High", "Low", "Close"}
    if df is None or df.empty or not needed.issubset(set(df.columns)):
        return ["取得失敗"] + [""] * 43

    df = df.dropna(subset=["High", "Low", "Close"]).copy()
    if len(df) < 5: return ["取得失敗"] + [""] * 43

    close = df["Close"]
    price = float(close.iloc[-1])
    pnl = (price - cost) if cost is not None else None
    pnl_pct = (price / cost - 1.0) if (cost is not None and cost != 0) else None

    atr14, atr50, atr100 = atr_sma(df, 14), atr_sma(df, 50), atr_sma(df, 100)
    atrp14 = (atr14 / price) if (atr14 and price) else None
    atrp50 = (atr50 / price) if (atr50 and price) else None
    atrp100 = (atr100 / price) if (atr100 and price) else None

    def get_stop(a, m):
        if a is None: return None, None
        dist = a * m
        return float(dist), float(price - dist)

    sd14_2, sp14_2 = get_stop(atr14, 2)
    sd14_3, sp14_3 = get_stop(atr14, 3)
    sd50_2, sp50_2 = get_stop(atr50, 2)
    sd50_3, sp50_3 = get_stop(atr50, 3)
    sd100_2, sp100_2 = get_stop(atr100, 2)
    sd100_3, sp100_3 = get_stop(atr100, 3)

    cg = lambda sp: float(cost - sp) if (cost is not None and sp is not None) else ""
    prox = lambda sp, a: stop_proximity(price, sp, a, settings.near_atr_factor)

    ma50, ma200 = sma(close, 50), sma(close, 200)
    trend = (ma50 > ma200) if (ma50 and ma200) else None
    sell_warn = "なし" if trend else ("強（トレンド崩れ：売り準備）" if trend is False else "")
    sell_msg = "モメンタム維持" if trend else ("モメンタム崩れ" if trend is False else "")

    ret20, v20, v60 = ret_n(close, settings.ret_lookback), vol_annualized(close, settings.vol_short), vol_annualized(close, settings.vol_long)
    vol_spike = (v20 > settings.vol_spike_ratio * v60) if (v20 and v60) else None

    if trend is None or ret20 is None or vol_spike is None:
        add_sig, add_reason = "見送り", "データ不足"
    elif not trend:
        add_sig, add_reason = "買い増し禁止", "トレンド崩れ"
    elif vol_spike:
        add_sig, add_reason = "買い増し禁止", "ボラ急騰"
    elif ret20 > 0:
        add_sig, add_reason = "買い増し", "上昇継続"
    else:
        add_sig, add_reason = "見送り", "直近弱含み"

    return [
        price, pnl or "", pnl_pct or "",
        atr14 or "", atr50 or "", atr100 or "", atrp14 or "", atrp50 or "", atrp100 or "",
        sd14_2 or "", sd14_3 or "", sd50_2 or "", sd50_3 or "", sd100_2 or "", sd100_3 or "",
        sp14_2 or "", sp14_3 or "", sp50_2 or "", sp50_3 or "", sp100_2 or "", sp100_3 or "",
        cg(sp14_2), cg(sp14_3), cg(sp50_2), cg(sp50_3), cg(sp100_2), cg(sp100_3),
        prox(sp14_2, atr14), prox(sp14_3, atr14), prox(sp50_2, atr50), prox(sp50_3, atr50), prox(sp100_2, atr100), prox(sp100_3, atr100),
        sell_warn, sell_msg, ma50 or "", ma200 or "", str(trend).upper() if trend is not None else "",
        ret20 or "", v20 or "", v60 or "", str(vol_spike).upper() if vol_spike is not None else "", add_sig, add_reason
    ]


# -------------------------
# Sheets I/O
# -------------------------
def open_worksheet(secrets: Dict[str, Any]):
    sid = str(secrets.get("spreadsheet_id", "")).strip()
    sname = str(secrets.get("sheet_name", "Holdings")).strip()
    sa_info = secrets.get("google_service_account_json")
    if isinstance(sa_info, str):
        try: sa_info = json.loads(sa_info)
        except: sa_info = _relaxed_json_loads(sa_info)
    creds = Credentials.from_service_account_info(sa_info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    return gspread.authorize(creds).open_by_key(sid).worksheet(sname)


# -------------------------
# yfinance 一括取得（修正ポイント）
# -------------------------
def _download_batch_silent(tickers: List[str], period: str, retry_max: int, retry_base: float) -> Tuple[Optional[pd.DataFrame], bool]:
    rate_limited = False
    for attempt in range(1, retry_max + 1):
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                df = yf.download(
                    tickers=tickers, period=period, interval="1d",
                    auto_adjust=False, progress=False, threads=False,
                    group_by='column' # カラム構造を固定
                )
            if df is not None and not df.empty:
                return df, rate_limited
        except Exception as e:
            if any(x in str(e).lower() for x in ["429", "ratelimit", "too many requests"]):
                rate_limited = True
            if attempt < retry_max:
                time.sleep(retry_base * (2**(attempt-1)) + random.random())
    return None, rate_limited


def _split_multi_ticker_df(df: pd.DataFrame, yf_ticker: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    try:
        # MultiIndexの場合 (Field, Ticker) または (Ticker, Field) を試行
        if isinstance(df.columns, pd.MultiIndex):
            # yfinanceの標準的な一括DL構造 (Field, Ticker) から抽出
            sub = df.loc[:, (slice(None), yf_ticker)]
            sub.columns = sub.columns.get_level_values(0)
            return sub.copy()
        return df.copy()
    except:
        return None


def prefetch_yfinance(raw_tickers: List[str], settings: Settings) -> Dict[str, Any]:
    _DATA_CACHE.clear()
    yf_to_raws: Dict[str, List[str]] = {}
    for r in raw_tickers:
        if r.strip():
            yt = normalize_yf_ticker(r)
            yf_to_raws.setdefault(yt, []).append(r)

    uniq_yf = list(yf_to_raws.keys())
    ok, fail, rl_batches = 0, 0, 0
    bs, period = settings.yf_batch_size, f"{settings.history_years}y"

    for i in range(0, len(uniq_yf), bs):
        batch = uniq_yf[i : i + bs]
        df, rl = _download_batch_silent(batch, period, settings.yf_retry_max, settings.yf_retry_base)
        if rl: rl_batches += 1
        
        if df is None or df.empty:
            fail += sum(len(yf_to_raws[t]) for t in batch)
        else:
            for yt in batch:
                sub = _split_multi_ticker_df(df, yt)
                if sub is not None and not sub.empty and {"High", "Low", "Close"}.issubset(sub.columns):
                    for r in yf_to_raws[yt]:
                        _DATA_CACHE[r] = sub
                        ok += 1
                else:
                    fail += len(yf_to_raws[yt])
        if i + bs < len(uniq_yf):
            time.sleep(settings.yf_batch_sleep + random.random())

    return {"total": len(raw_tickers), "ok": ok, "fail": fail, "rate_limited_batches": rl_batches}


def main():
    today = dt.datetime.now(JST).date()
    skip, reason = is_skip_day_jst(today)
    if skip:
        print(f"[SKIP] {today} {reason}")
        return

    secrets, settings = load_secrets()
    ws = open_worksheet(secrets)
    rows = ws.get("A2:C")
    if not rows: return

    tickers = [r[0].strip() for r in rows if r and r[0].strip()]
    stats = prefetch_yfinance(tickers, settings)
    print(f"DL: total={stats['total']} ok={stats['ok']} fail={stats['fail']} rate_limited_batches={stats['rate_limited_batches']} retries<= {settings.yf_retry_max}")

    outputs = []
    for r in rows:
        ticker = r[0].strip() if r and r[0].strip() else ""
        cost = None
        if ticker and len(r) > 2:
            try: cost = float(r[2])
            except: pass
        outputs.append(compute_row_outputs(ticker, cost, settings) if ticker else [""] * 44)

    ws.update(range_name="D2", values=outputs, value_input_option="USER_ENTERED")
    print(f"SHEET_UPDATE: rows={len(outputs)}")


if __name__ == "__main__":
    main()
