import os
import json
import math
import time
import random
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import re
import tempfile
import urllib.request

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
    yf_batch_size: int = 20
    yf_batch_sleep: float = 1.0
    yf_retry_max: int = 5
    yf_retry_base: float = 3.0


# -------------------------
# Secretsのパース（工程増やさずに頑丈に）
#   1) JSON
#   2) 文字列内の実改行などを救済してJSON
#   3) key=value / key: value の複数行形式（google_service_account_json/settingsは {..} ブロック対応）
# -------------------------
def _relaxed_json_loads(raw: str) -> Any:
    """
    JSON文字列リテラル内に混入した実改行/CR/TABを \\n/\\r/\\t に補正してから json.loads。
    """
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
    """
    lines[start_idx] から始まる JSONブロック（{...}）を、括弧の対応が取れるまで連結して返す。
    返り値: (json_text, next_index)
    """
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
    """
    APP_SECRETS_JSON がJSONでない場合の救済：
    - key=value または key: value を複数行で書いたものを辞書化
    - google_service_account_json / settings が { で始まる場合は、括弧が閉じるまでブロックとして読み込んで JSON として解析
    """
    lines = raw.splitlines()
    out: Dict[str, Any] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue
        if line.startswith("#"):
            continue

        sep_pos = None
        sep = None
        for s in ["=", ":"]:
            p = line.find(s)
            if p != -1:
                sep_pos = p
                sep = s
                break
        if sep_pos is None:
            continue

        key = line[:sep_pos].strip()
        val = line[sep_pos + 1 :].strip()

        if (val == "" and i < len(lines) and lines[i].lstrip().startswith("{")) or val.startswith("{"):
            if val.startswith("{"):
                block_start = i - 1
            else:
                block_start = i

            block_text, next_i = _brace_balanced_json_block(lines, block_start)
            try:
                out[key] = _relaxed_json_loads(block_text) if isinstance(block_text, str) else json.loads(block_text)
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
                if "." in val:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
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
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    try:
        v = _relaxed_json_loads(raw)
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    v = _parse_kv_multiline(raw)
    if isinstance(v, dict) and v:
        return v

    raise RuntimeError("APP_SECRETS_JSON を解釈できません（JSONまたは key=value / key: value 形式にしてください）")


def load_secrets() -> Tuple[Dict[str, Any], Settings]:
    raw = os.environ.get("APP_SECRETS_JSON", "")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("APP_SECRETS_JSON が未設定です（GitHub Secretsに設定してください）")

    secrets = parse_app_secrets(raw)

    sa_val = secrets.get("google_service_account_json", None)
    if sa_val is None:
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
        yf_batch_size=int(settings_in.get("yf_batch_size", 20)),
        yf_batch_sleep=float(settings_in.get("yf_batch_sleep", 1.0)),
        yf_retry_max=int(settings_in.get("yf_retry_max", 5)),
        yf_retry_base=float(settings_in.get("yf_retry_base", 3.0)),
    )
    return secrets, s


# -------------------------
# ティッカー正規化（downloadに渡すものだけ変える）
# -------------------------
def normalize_yf_ticker(raw_ticker: str) -> str:
    t = (raw_ticker or "").strip()
    if not t:
        return t
    # 先頭が数字 & "." を含まない -> 日本株コード扱いで ".T"
    if t[0].isdigit() and ("." not in t):
        return t + ".T"
    return t


# -------------------------
# yfinance → 指標計算
# -------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - prev_close).abs()
    lc = (df["Low"] - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr


def atr_sma(df: pd.DataFrame, n: int) -> Optional[float]:
    if df is None or len(df) < n + 2:
        return None
    tr = true_range(df)
    atr = tr.rolling(window=n).mean()
    v = atr.iloc[-1]
    return None if pd.isna(v) else float(v)


def sma(series: pd.Series, n: int) -> Optional[float]:
    if series is None or len(series) < n:
        return None
    v = series.rolling(window=n).mean().iloc[-1]
    return None if pd.isna(v) else float(v)


def ret_n(series: pd.Series, n: int) -> Optional[float]:
    if series is None or len(series) < n + 1:
        return None
    v = float(series.iloc[-1] / series.iloc[-1 - n] - 1.0)
    return v


def vol_annualized(series: pd.Series, n: int) -> Optional[float]:
    if series is None or len(series) < n + 1:
        return None
    r = series.pct_change().dropna()
    r_n = r.iloc[-n:]
    if len(r_n) < n:
        return None
    v = float(r_n.std(ddof=0) * math.sqrt(252.0))
    return v


def stop_proximity(price: Optional[float], stop_price: Optional[float], atr_val: Optional[float], near_factor: float) -> str:
    if price is None or stop_price is None or atr_val is None:
        return ""
    if price <= stop_price:
        return "割れ"
    if price <= stop_price + near_factor * atr_val:
        return "接近"
    return "OK"


def compute_row_outputs(
    ticker: str,
    cost: Optional[float],
    settings: Settings,
) -> List[Any]:
    df = _DATA_CACHE.get(ticker)

    needed = {"High", "Low", "Close"}
    if df is None or df.empty or not needed.issubset(set(df.columns)):
        return ["取得失敗"] + [""] * (44 - 1)

    df = df.dropna(subset=["High", "Low", "Close"]).copy()
    if len(df) < 5:
        return ["取得失敗"] + [""] * (44 - 1)

    close = df["Close"]
    price = float(close.iloc[-1])

    pnl = (price - cost) if cost is not None else None
    pnl_pct = (price / cost - 1.0) if (cost is not None and cost != 0) else None

    atr14 = atr_sma(df, 14)
    atr50 = atr_sma(df, 50)
    atr100 = atr_sma(df, 100)

    atrp14 = (atr14 / price) if (atr14 is not None and price != 0) else None
    atrp50 = (atr50 / price) if (atr50 is not None and price != 0) else None
    atrp100 = (atr100 / price) if (atr100 is not None and price != 0) else None

    def mul(x: Optional[float], m: int) -> Optional[float]:
        return None if x is None else float(x * m)

    sd_14_2 = mul(atr14, 2)
    sd_14_3 = mul(atr14, 3)
    sd_50_2 = mul(atr50, 2)
    sd_50_3 = mul(atr50, 3)
    sd_100_2 = mul(atr100, 2)
    sd_100_3 = mul(atr100, 3)

    def stop_price(distance: Optional[float]) -> Optional[float]:
        return None if distance is None else float(price - distance)

    sp_14_2 = stop_price(sd_14_2)
    sp_14_3 = stop_price(sd_14_3)
    sp_50_2 = stop_price(sd_50_2)
    sp_50_3 = stop_price(sd_50_3)
    sp_100_2 = stop_price(sd_100_2)
    sp_100_3 = stop_price(sd_100_3)

    def cost_gap(sp: Optional[float]) -> Optional[float]:
        if cost is None or sp is None:
            return None
        return float(cost - sp)

    cg_14_2 = cost_gap(sp_14_2)
    cg_14_3 = cost_gap(sp_14_3)
    cg_50_2 = cost_gap(sp_50_2)
    cg_50_3 = cost_gap(sp_50_3)
    cg_100_2 = cost_gap(sp_100_2)
    cg_100_3 = cost_gap(sp_100_3)

    near = settings.near_atr_factor
    a_14_2 = stop_proximity(price, sp_14_2, atr14, near)
    a_14_3 = stop_proximity(price, sp_14_3, atr14, near)
    a_50_2 = stop_proximity(price, sp_50_2, atr50, near)
    a_50_3 = stop_proximity(price, sp_50_3, atr50, near)
    a_100_2 = stop_proximity(price, sp_100_2, atr100, near)
    a_100_3 = stop_proximity(price, sp_100_3, atr100, near)

    ma50 = sma(close, 50)
    ma200 = sma(close, 200)
    trend = None
    if ma50 is not None and ma200 is not None:
        trend = (ma50 > ma200)

    if trend is None:
        sell_warn = ""
        sell_msg = ""
    elif trend:
        sell_warn = "なし"
        sell_msg = "モメンタム維持：売り警戒なし（MA50>MA200）"
    else:
        sell_warn = "強（トレンド崩れ：売り準備）"
        sell_msg = "モメンタム崩れ：売り局面が近い（MA50<=MA200）。買い増し禁止。撤退（売却）を検討"

    ret20 = ret_n(close, settings.ret_lookback)
    vol20 = vol_annualized(close, settings.vol_short)
    vol60 = vol_annualized(close, settings.vol_long)

    vol_spike = None
    if vol20 is not None and vol60 is not None:
        vol_spike = (vol20 > settings.vol_spike_ratio * vol60)

    if trend is None or ret20 is None or vol_spike is None:
        add_sig = "見送り"
        add_reason = "見送り：データ不足"
    else:
        if not trend:
            add_sig = "買い増し禁止"
            add_reason = "買い増し禁止：トレンド崩れ"
        elif vol_spike:
            add_sig = "買い増し禁止"
            add_reason = "買い増し禁止：ボラ急騰"
        elif ret20 > 0 and (not vol_spike):
            add_sig = "買い増し"
            add_reason = "買い増し：上昇トレンド継続・直近強い・ボラ急騰なし"
        else:
            add_sig = "見送り"
            add_reason = "見送り：直近が弱い"

    out: List[Any] = [
        price,
        pnl if pnl is not None else "",
        pnl_pct if pnl_pct is not None else "",

        atr14 if atr14 is not None else "",
        atr50 if atr50 is not None else "",
        atr100 if atr100 is not None else "",
        atrp14 if atrp14 is not None else "",
        atrp50 if atrp50 is not None else "",
        atrp100 if atrp100 is not None else "",

        sd_14_2 if sd_14_2 is not None else "",
        sd_14_3 if sd_14_3 is not None else "",
        sd_50_2 if sd_50_2 is not None else "",
        sd_50_3 if sd_50_3 is not None else "",
        sd_100_2 if sd_100_2 is not None else "",
        sd_100_3 if sd_100_3 is not None else "",

        sp_14_2 if sp_14_2 is not None else "",
        sp_14_3 if sp_14_3 is not None else "",
        sp_50_2 if sp_50_2 is not None else "",
        sp_50_3 if sp_50_3 is not None else "",
        sp_100_2 if sp_100_2 is not None else "",
        sp_100_3 if sp_100_3 is not None else "",

        cg_14_2 if cg_14_2 is not None else "",
        cg_14_3 if cg_14_3 is not None else "",
        cg_50_2 if cg_50_2 is not None else "",
        cg_50_3 if cg_50_3 is not None else "",
        cg_100_2 if cg_100_2 is not None else "",
        cg_100_3 if cg_100_3 is not None else "",

        a_14_2,
        a_14_3,
        a_50_2,
        a_50_3,
        a_100_2,
        a_100_3,

        sell_warn,
        sell_msg,

        ma50 if ma50 is not None else "",
        ma200 if ma200 is not None else "",
        ("TRUE" if trend else "FALSE") if trend is not None else "",
        ret20 if ret20 is not None else "",
        vol20 if vol20 is not None else "",
        vol60 if vol60 is not None else "",
        ("TRUE" if vol_spike else "FALSE") if vol_spike is not None else "",
        add_sig,
        add_reason,
    ]

    if len(out) != 44:
        raise RuntimeError(f"出力列数が不正: {len(out)}（44が必要）")

    return out


# -------------------------
# Sheets I/O
# -------------------------
def open_worksheet(secrets: Dict[str, Any]):
    spreadsheet_id = str(secrets.get("spreadsheet_id", "")).strip()
    sheet_name = str(secrets.get("sheet_name", "Holdings")).strip()
    if not spreadsheet_id:
        raise RuntimeError("spreadsheet_id が未設定です")

    sa_val = secrets.get("google_service_account_json", None)
    if sa_val is None:
        raise RuntimeError("google_service_account_json が未設定です")

    if isinstance(sa_val, dict):
        sa_info = sa_val
    elif isinstance(sa_val, str):
        try:
            sa_info = json.loads(sa_val)
        except Exception:
            sa_info = _relaxed_json_loads(sa_val)
        if not isinstance(sa_info, dict):
            raise RuntimeError("google_service_account_json の解釈結果がdictではありません")
    else:
        raise RuntimeError("google_service_account_json は文字列またはオブジェクトで指定してください")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)
    return ws


# -------------------------
# yfinance 一括取得（秘匿・レート制限対策）
# -------------------------
def _looks_rate_limited(exc: BaseException) -> bool:
    s = str(exc).lower()
    return ("ratelimit" in s) or ("rate limited" in s) or ("too many requests" in s) or ("429" in s)


def _download_batch_silent(tickers: List[str], period: str, retry_max: int, retry_base: float) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    download中のstdout/stderrを握りつぶす。ログには何も出さない。
    戻り値: (df or None, rate_limited_flag)
    """
    rate_limited = False

    for attempt in range(1, retry_max + 1):
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                df = yf.download(
                    tickers=tickers,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return None, rate_limited
            return df, rate_limited
        except Exception as e:
            if _looks_rate_limited(e):
                rate_limited = True
            if attempt < retry_max:
                wait = retry_base * (2 ** (attempt - 1)) + random.uniform(0.0, 0.7)
                time.sleep(wait)
                continue
            return None, rate_limited

    return None, rate_limited


def _split_multi_ticker_df(df: pd.DataFrame, yf_ticker: str) -> Optional[pd.DataFrame]:
    """
    yf.download 複数ティッカー時の MultiIndex 列を、単一ティッカーの OHLCV DataFrame に戻す。
    形式差を吸収するため、level=1 と level=0 を両方試す。
    """
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        sub = None
        # 典型: (Field, Ticker)
        try:
            sub = df.xs(yf_ticker, axis=1, level=1, drop_level=True).copy()
        except Exception:
            sub = None
        # 逆: (Ticker, Field)
        if sub is None:
            try:
                sub = df.xs(yf_ticker, axis=1, level=0, drop_level=True).copy()
            except Exception:
                sub = None

        if sub is None or sub.empty:
            return None

        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in sub.columns]
        if not keep:
            return None
        return sub[keep].dropna(how="all")

    return df.copy()


def prefetch_yfinance(raw_tickers: List[str], settings: Settings) -> Dict[str, Any]:
    """
    取得結果は _DATA_CACHE に入れる（キーはスプシの銘柄コード）。
    publicログ用に「件数のみ」返す。
    """
    _DATA_CACHE.clear()

    # raw -> yf の対応を作る（スプシはrawのまま）
    yf_to_raws: Dict[str, List[str]] = {}
    for raw in raw_tickers:
        r = (raw or "").strip()
        if not r:
            continue
        yf_t = normalize_yf_ticker(r)
        yf_to_raws.setdefault(yf_t, []).append(r)

    uniq_yf = list(yf_to_raws.keys())
    total = len(uniq_yf)
    ok = 0
    fail = 0
    rate_limited_batches = 0

    period = f"{settings.history_years}y"
    bs = max(1, int(settings.yf_batch_size))
    sleep_s = max(0.0, float(settings.yf_batch_sleep))

    for i in range(0, total, bs):
        batch_yf = uniq_yf[i : i + bs]
        df, rl = _download_batch_silent(batch_yf, period, settings.yf_retry_max, settings.yf_retry_base)
        if rl:
            rate_limited_batches += 1

        if df is None or df.empty:
            # バッチ全滅
            for yf_t in batch_yf:
                fail += len(yf_to_raws.get(yf_t, []))
        else:
            for yf_t in batch_yf:
                sub = _split_multi_ticker_df(df, yf_t) if len(batch_yf) > 1 else df
                if sub is None or sub.empty or not {"High", "Low", "Close"}.issubset(set(sub.columns)):
                    fail += len(yf_to_raws.get(yf_t, []))
                    continue
                # キャッシュは raw で持つ（以降の処理は変えない）
                for raw in yf_to_raws.get(yf_t, []):
                    _DATA_CACHE[raw] = sub
                    ok += 1

        if sleep_s > 0 and (i + bs) < total:
            time.sleep(sleep_s + random.uniform(0.0, 0.4))

    return {
        "total": sum(len(v) for v in yf_to_raws.values()),  # raw件数
        "ok": ok,
        "fail": fail,
        "rate_limited_batches": rate_limited_batches,
    }


# -------------------------
# 追加：最小限のポートフォリオ管理（列は4つだけ追加）
#   AW=時価, AX=ウェイト, AY=含み損益(円), AZ=セクター(33業種)
#   上部(A1:J3, K1:R3)に集計・ストレスを出す
# -------------------------
_JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _extract_code(raw_ticker: str) -> Optional[str]:
    t = (raw_ticker or "").strip()
    if not t:
        return None
    m = re.match(r"^(\d{4,5})", t)
    if not m:
        return None
    return m.group(1).zfill(4)


def _load_jpx_sector_map() -> Dict[str, str]:
    """
    JPXの公開「上場銘柄一覧」相当のExcelから、コード -> 33業種区分 を作る。
    失敗したら空dictを返す（セクターは空欄になる）。
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tf:
            tmp_path = tf.name
        urllib.request.urlretrieve(_JPX_LIST_URL, tmp_path)
        df = pd.read_excel(tmp_path, dtype=str)
        os.remove(tmp_path)

        code_col = None
        sector_col = None

        # 列名はJPX側で揺れるため、包含で拾う
        for c in df.columns:
            cs = str(c)
            if code_col is None and ("コード" in cs):
                code_col = c
            if sector_col is None and ("33" in cs and "業種" in cs):
                sector_col = c

        if code_col is None or sector_col is None:
            return {}

        out: Dict[str, str] = {}
        for _, row in df.iterrows():
            code_raw = row.get(code_col, "")
            sec_raw = row.get(sector_col, "")
            if code_raw is None:
                continue
            code_s = str(code_raw).strip()
            if code_s == "":
                continue
            # "7203.0" みたいなのを救済
            code_s = re.sub(r"\.0+$", "", code_s)
            if not re.match(r"^\d{4,5}$", code_s):
                continue
            code_s = code_s.zfill(4)

            sec_s = "" if sec_raw is None else str(sec_raw).strip()
            if sec_s == "":
                continue

            out[code_s] = sec_s

        return out
    except Exception:
        return {}


def _build_sector_for_raws(raw_tickers: List[str]) -> Dict[str, str]:
    """
    raw_ticker -> 33業種（取れない場合は ""）
    """
    code_to_sector = _load_jpx_sector_map()
    out: Dict[str, str] = {}
    if not code_to_sector:
        for t in raw_tickers:
            out[t] = ""
        return out

    for t in raw_tickers:
        code = _extract_code(t)
        out[t] = code_to_sector.get(code, "") if code else ""
    return out


def _build_returns_matrix(raw_tickers: List[str], min_obs: int = 60) -> pd.DataFrame:
    """
    _DATA_CACHE から Adj Close優先で価格系列を集め、日次リターンDF（列=raw_ticker）を返す。
    """
    series_list: List[pd.Series] = []
    for t in raw_tickers:
        df = _DATA_CACHE.get(t)
        if df is None or df.empty:
            continue
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            continue
        s = df[col].dropna()
        if len(s) < (min_obs + 1):
            continue
        s.name = t
        series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    prices = pd.concat(series_list, axis=1)
    rets = prices.pct_change()
    rets = rets.dropna(how="all")
    # NaNが多い列を落とす（最小観測数）
    ok_cols = [c for c in rets.columns if rets[c].dropna().shape[0] >= min_obs]
    rets = rets[ok_cols]
    # ここでは簡素に共通日だけ使う（行NaNを落とす）
    rets = rets.dropna(how="any")
    return rets


def _fmt_top3_pairs(items: List[Tuple[str, float]], unit: str) -> str:
    """
    code:value を3つまで短く整形
    unit: "yen" or "pct" or "rho"
    """
    out = []
    for k, v in items[:3]:
        if unit == "yen":
            out.append(f"{k}:{int(round(v)):,}")
        elif unit == "pct":
            out.append(f"{k}:{v*100:.1f}%")
        elif unit == "rho":
            out.append(f"{k}:{v:.2f}")
        else:
            out.append(f"{k}:{v}")
    return ", ".join(out)


def _compute_portfolio_and_write_top(ws, rows: List[List[Any]], outputs44: List[List[Any]], extras4: List[List[Any]], sector_map: Dict[str, str]):
    """
    rows: A5:D の取得結果
    outputs44: E5:AV の書き込み予定値
    extras4: AW5:AZ の書き込み予定値（AW/AX/AY/AZ）
    セルA1:J3、K1:R3に集計とストレスを書き込む。
    """
    now_jst = dt.datetime.now(JST)
    now_str = now_jst.strftime("%Y-%m-%d %H:%M:%S")

    # 有効行の抽出
    tickers: List[str] = []
    shares_map: Dict[str, float] = {}
    cost_map: Dict[str, float] = {}
    price_map: Dict[str, float] = {}
    pnl_map: Dict[str, float] = {}
    mv_map: Dict[str, float] = {}

    total_input = 0
    ok_price = 0

    for i, r in enumerate(rows):
        t = (r[0] if len(r) > 0 else "").strip()
        if not t:
            continue
        total_input += 1

        sh = _to_float(r[2] if len(r) > 2 else None)
        cost = _to_float(r[3] if len(r) > 3 else None)
        sh = float(sh) if sh is not None else 0.0
        shares_map[t] = sh
        if cost is not None:
            cost_map[t] = float(cost)

        o = outputs44[i]
        # priceは先頭（取得失敗の場合は文字列）
        price = o[0]
        if isinstance(price, (int, float)) and sh > 0:
            ok_price += 1
            price_map[t] = float(price)
            mv = float(price) * sh
            mv_map[t] = mv
            if cost is not None:
                pnl_map[t] = (float(price) - float(cost)) * sh
        tickers.append(t)

    total_mv = sum(mv_map.values())
    # extras4 のウェイト（AX）を埋め直す（AW/AY/セクターは既に入ってる想定）
    if total_mv > 0:
        for i, r in enumerate(rows):
            t = (r[0] if len(r) > 0 else "").strip()
            if not t:
                continue
            mv = mv_map.get(t, 0.0)
            extras4[i][1] = (mv / total_mv) if mv > 0 else ""

    # 集中度
    weights = [v / total_mv for v in mv_map.values()] if total_mv > 0 else []
    top1 = max(weights) if weights else None
    top5 = sum(sorted(weights, reverse=True)[:5]) if weights else None
    hhi = sum(w * w for w in weights) if weights else None

    # セクター偏り（既知セクターのみ）
    sector_weights: Dict[str, float] = {}
    for t, mv in mv_map.items():
        sec = sector_map.get(t, "")
        if not sec:
            continue
        sector_weights[sec] = sector_weights.get(sec, 0.0) + mv
    max_sector_ratio = None
    if sector_weights and total_mv > 0:
        max_sector_ratio = max(sector_weights.values()) / total_mv

    # リターン行列（VaR/相関）
    rets = _build_returns_matrix(list(mv_map.keys()), min_obs=60)
    var95 = None
    var99 = None
    worst = None
    corr_pair = ("", 0.0)
    var_contrib: List[Tuple[str, float]] = []

    if not rets.empty and total_mv > 0:
        cols = list(rets.columns)
        w = np.array([mv_map.get(c, 0.0) for c in cols], dtype=float)
        if w.sum() > 0:
            w = w / w.sum()
            rp = rets.values @ w
            q05 = float(np.quantile(rp, 0.05))
            q01 = float(np.quantile(rp, 0.01))
            var95 = max(0.0, -q05)
            var99 = max(0.0, -q01)
            worst = float(np.min(rp))

            # 相関上位ペア（最大の正の相関）
            corr = rets.corr().values
            if corr.shape[0] >= 2:
                iu = np.triu_indices(corr.shape[0], k=1)
                vals = corr[iu]
                if vals.size > 0:
                    k = int(np.nanargmax(vals))
                    i0 = int(iu[0][k])
                    j0 = int(iu[1][k])
                    corr_pair = (f"{cols[i0]}-{cols[j0]}", float(vals[k]))

            # VaR寄与（共分散ベースの近似 → VaR95へスケール）
            Sigma = np.cov(rets.values, rowvar=False, ddof=1)
            port_var = float(w.T @ Sigma @ w)
            if port_var > 0:
                port_sigma = math.sqrt(port_var)
                m = Sigma @ w
                comp_sigma = w * m / port_sigma  # volatility contribution (decimal)
                if var95 is not None and port_sigma > 0:
                    factor = var95 / port_sigma
                    comp_var = comp_sigma * factor  # VaR95 contribution (decimal)
                    pairs = list(zip(cols, comp_var))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    var_contrib = pairs[:3]

    # 含み損TOP3（損が大きい=最も負の値）
    loss_items: List[Tuple[str, float]] = []
    for t, pnl in pnl_map.items():
        if pnl < 0:
            loss_items.append((t, pnl))
    loss_items.sort(key=lambda x: x[1])  # もっとも負のものが先
    loss_top3 = loss_items[:3]

    # A1:J3 に書く
    a1j3: List[List[Any]] = [
        ["更新日時(JST)", now_str, "総時価(円)", (total_mv if total_mv > 0 else ""), "含み損益(円)", (sum(pnl_map.values()) if pnl_map else 0.0), "上位1比率", (top1 if top1 is not None else ""), "上位5比率", (top5 if top5 is not None else "")],
        ["VaR95(1日)", (var95 if var95 is not None else ""), "VaR95(円)", ((var95 * total_mv) if (var95 is not None and total_mv > 0) else ""), "VaR99(1日)", (var99 if var99 is not None else ""), "VaR99(円)", ((var99 * total_mv) if (var99 is not None and total_mv > 0) else ""), "カバレッジ", f"{ok_price}/{total_input}"],
        ["HHI(集中度)", (hhi if hhi is not None else ""), "最大セクター比率", (max_sector_ratio if max_sector_ratio is not None else ""), "含み損TOP3", (_fmt_top3_pairs(loss_top3, "yen") if loss_top3 else ""), "VaR寄与TOP3", (_fmt_top3_pairs(var_contrib, "pct") if var_contrib else ""), "相関上位ペア", (f"{corr_pair[0]}:{corr_pair[1]:.2f}" if corr_pair[0] else "")],
    ]
    ws.update(range_name="A1", values=a1j3, value_input_option="USER_ENTERED")

    # ストレス K1:R3（8本）
    # 影響(%)は負の値で表現（損失）
    def _w_sum(keys: List[str]) -> float:
        if total_mv <= 0:
            return 0.0
        return sum(mv_map.get(k, 0.0) for k in keys) / total_mv

    # Top銘柄
    sorted_by_w = sorted(mv_map.items(), key=lambda x: x[1], reverse=True)
    top1_t = [sorted_by_w[0][0]] if sorted_by_w else []
    top3_t = [x[0] for x in sorted_by_w[:3]] if sorted_by_w else []

    # 最大セクターの銘柄群（既知セクターのみ）
    max_sector_keys: List[str] = []
    if sector_weights:
        max_sec = max(sector_weights.items(), key=lambda x: x[1])[0]
        max_sector_keys = [t for t in mv_map.keys() if sector_map.get(t, "") == max_sec]

    # 相関上位ペアの銘柄2つ
    corr_pair_keys: List[str] = []
    if corr_pair[0]:
        a, b = corr_pair[0].split("-", 1)
        corr_pair_keys = [a, b]

    s1 = -0.10
    s2 = -0.20
    s3 = -0.30 * _w_sum(top1_t)
    s4 = -0.20 * _w_sum(top3_t)
    s5 = -0.15 * _w_sum(max_sector_keys)
    s6 = -0.15 * _w_sum(corr_pair_keys)
    s7 = (-(var95) if var95 is not None else "")
    s8 = (worst if worst is not None else "")

    stress_pct = [s1, s2, s3, s4, s5, s6, s7, s8]
    stress_yen = []
    for x in stress_pct:
        if isinstance(x, (int, float)) and total_mv > 0:
            stress_yen.append(x * total_mv)
        else:
            stress_yen.append("")

    k1r3: List[List[Any]] = [
        ["S1 All(-10%)", "S2 All(-20%)", "S3 Top1(-30%)", "S4 Top3(-20%)", "S5 MaxSector(-15%)", "S6 TopCorrPair(-15%)", "S7 VaR95", "S8 WorstDay"],
        stress_pct,
        stress_yen,
    ]
    ws.update(range_name="K1", values=k1r3, value_input_option="USER_ENTERED")


def main():
    today = dt.datetime.now(JST).date()
    skip, reason = is_skip_day_jst(today)
    if skip:
        print(f"[SKIP] {today.isoformat()} JST / {reason}")
        return

    secrets, settings = load_secrets()
    ws = open_worksheet(secrets)

    # A=銘柄コード, B=銘柄名, C=株数, D=取得単価（SBI CSV想定）
    rows = ws.get("A5:D")
    if not rows:
        print("RUN: rows=0")
        return

    tickers: List[str] = []
    for r in rows:
        t = (r[0] if len(r) > 0 else "").strip()
        if t:
            tickers.append(t)

    stats = prefetch_yfinance(tickers, settings)

    # ログは件数のみ
    print(
        f"DL: total={stats['total']} ok={stats['ok']} fail={stats['fail']} rate_limited_batches={stats['rate_limited_batches']} retries<= {settings.yf_retry_max}"
    )

    # セクター（33業種）マップ（取れない場合は空欄）
    sector_map = _build_sector_for_raws(tickers)

    outputs44: List[List[Any]] = []
    extras4: List[List[Any]] = []

    for r in rows:
        ticker = (r[0] if len(r) > 0 else "").strip()
        shares_raw = (r[2] if len(r) > 2 else "")
        cost_raw = (r[3] if len(r) > 3 else "")

        shares = _to_float(shares_raw)
        shares = float(shares) if shares is not None else 0.0

        cost = None
        try:
            v = _to_float(cost_raw)
            cost = float(v) if v is not None else None
        except Exception:
            cost = None

        if not ticker:
            outputs44.append([""] * 44)
            extras4.append(["", "", "", ""])
            continue

        out = compute_row_outputs(ticker, cost, settings)
        outputs44.append(out)

        # 追加4列：AW=時価, AX=ウェイト(後で埋める), AY=含み損益(円), AZ=セクター
        price = out[0]
        if isinstance(price, (int, float)) and shares > 0:
            mv = float(price) * shares
            pnl_yen = ((float(price) - cost) * shares) if (cost is not None) else ""
            extras4.append([mv, "", pnl_yen, sector_map.get(ticker, "")])
        else:
            extras4.append(["", "", "", sector_map.get(ticker, "")])

    # 既存44列（E5開始）
    ws.update(range_name="E5", values=outputs44, value_input_option="USER_ENTERED")
    # 追加4列（AW5開始：時価/ウェイト/含み損益/セクター）
    ws.update(range_name="AW5", values=extras4, value_input_option="USER_ENTERED")

    # 上部ブロック（A1:J3, K1:R3）
    _compute_portfolio_and_write_top(ws, rows, outputs44, extras4, sector_map)

    # ログは件数のみ
    print(f"SHEET_UPDATE: rows={len(outputs44)}")


if __name__ == "__main__":
    main()
