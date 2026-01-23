import os
import json
import math
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

        # その行で閉じたら終了（brace==0 は「外側の {..} が閉じた」）
        # ただし開始前に brace が0のままはあり得ないので、少なくとも1回は { を見ている前提
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
        # コメントっぽい行は無視（勝手な拡張だが害は少ない）
        if line.startswith("#"):
            continue

        # 区切りを探す（= or :）
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

        # 値が空で、次行が { ならブロック扱い
        if (val == "" and i < len(lines) and lines[i].lstrip().startswith("{")) or val.startswith("{"):
            if val.startswith("{"):
                # 同じ行から開始
                block_start = i - 1
            else:
                # 次行から開始
                block_start = i

            block_text, next_i = _brace_balanced_json_block(lines, block_start)
            try:
                out[key] = _relaxed_json_loads(block_text) if isinstance(block_text, str) else json.loads(block_text)
            except Exception:
                # 最後の手段：そのまま文字列
                out[key] = block_text
            i = next_i
            continue

        # クォート剥がし（"..." / '...'）
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        # 可能なら数値/真偽/JSON を解釈（軽く）
        low = val.lower()
        if low in ("true", "false"):
            out[key] = (low == "true")
        else:
            # 数値
            try:
                if "." in val:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            except Exception:
                # JSONっぽい（[ ... ] や { ... }）を解釈
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
    # 1) まず素直に JSON
    try:
        v = json.loads(raw)
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    # 2) 文字列内改行救済して JSON
    try:
        v = _relaxed_json_loads(raw)
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    # 3) key=value / key: value の複数行
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
    )
    return secrets, s


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
    period = f"{settings.history_years}y"

    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

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
        # 文字列として入っている場合も、壊れていても救済して dict にする
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


def main():
    today = dt.datetime.now(JST).date()
    skip, reason = is_skip_day_jst(today)
    if skip:
        print(f"[SKIP] {today.isoformat()} JST / {reason}")
        return

    secrets, settings = load_secrets()
    ws = open_worksheet(secrets)

    rows = ws.get("A2:C")
    if not rows:
        print("A2:C にデータがありません。終了します。")
        return

    outputs: List[List[Any]] = []
    for r in rows:
        ticker = (r[0] if len(r) > 0 else "").strip()
        cost_raw = (r[2] if len(r) > 2 else "")
        cost = None
        try:
            cost = float(cost_raw) if str(cost_raw).strip() != "" else None
        except Exception:
            cost = None

        if not ticker:
            outputs.append([""] * 44)
            continue

        outputs.append(compute_row_outputs(ticker, cost, settings))

    ws.update("D2", outputs, value_input_option="USER_ENTERED")
    print(f"更新完了: {len(outputs)} 行 / 書込範囲 D2:AU{len(outputs)+1}")


if __name__ == "__main__":
    main()
