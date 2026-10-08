import copy
import io
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import re
from datetime import datetime

IST = "Asia/Kolkata"
STATE_FILE = "live_state.json"
MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
PATTERNS = ("Bullish Engulfing", "Bullish Harami", "Tweezer Bottom")

CANDLE_KEYS = ("prev_O", "prev_H", "prev_L", "prev_C", "sig_O", "sig_H", "sig_L", "sig_C")

# Sidebar-la illaadha fixed settings - venumna inge maathunga
SL_PCT = 0.0            # stop loss % below entry (0 = NO stop loss: exit only at target or time exit)
MAX_HOLD_DAYS = 1       # 1 = same day 3:15 PM exit
TREND_LOOKBACK = 0      # downtrend lookback (0 = off) - 0 = exactly your Excel formulas
TWEEZER_TOL_PCT = 0.10
FUTURES_MARGIN_PCT = 10.0   # NFO / MCX futures: only this % of the contract value is blocked as margin (100 = full value)
BROKERAGE_PCT = 0.0     # per side
SLIPPAGE_PCT = 0.0      # per side
FRESH_SECONDS = 120        # a signal is taken only if its candle closed less than this many seconds ago
RUN_CONFIG_FILE = "run_config.json"   # remembers what was running, so it can resume after a restart
REFRESH_SECONDS = 5     # gap between two scan cycles

TIMEFRAMES = {"1 minute": ("ONE_MINUTE", 1), "3 minutes": ("THREE_MINUTE", 3), "5 minutes": ("FIVE_MINUTE", 5),
              "10 minutes": ("TEN_MINUTE", 10), "15 minutes": ("FIFTEEN_MINUTE", 15),
              "30 minutes": ("THIRTY_MINUTE", 30), "1 hour": ("ONE_HOUR", 60)}


# ============================== CONFIG ==============================
@dataclass
class Config:
    capital: float = 100000.0
    max_open: int = 5
    target_pct: float = 1.0
    sl_pct: float = SL_PCT
    max_hold: int = MAX_HOLD_DAYS
    tweezer_tol_pct: float = TWEEZER_TOL_PCT
    trend_lookback: int = TREND_LOOKBACK
    brokerage_pct: float = BROKERAGE_PCT
    slippage_pct: float = SLIPPAGE_PCT
    interval: str = "FIVE_MINUTE"
    tf_min: int = 5
    patterns: tuple = PATTERNS
    position_amount: float = 0.0               # 0 = equal weight (capital / max positions)
    no_limit: bool = False                     # True = no max-positions limit (cash is the only limit)

    @property
    def per_position(self):
        return self.position_amount if self.position_amount > 0 else self.capital / self.max_open


# ============================== TIME ==============================
def now_ist():
    return pd.Timestamp.now(tz=IST).tz_localize(None)


def market_open(now, exchange="NSE"):
    """NSE/BSE/NFO 9:15-15:30 ; MCX 9:00-23:30 (Mon-Fri). Exchange holidays are not handled."""
    start, end = (dtime(9, 0), dtime(23, 30)) if exchange == "MCX" else (dtime(9, 15), dtime(15, 30))
    return now.weekday() < 5 and start <= now.time() <= end


def exit_time(exchange):
    """Time exit / last entry cut-off = 15 minutes before the close."""
    return dtime(23, 15) if exchange == "MCX" else dtime(15, 15)


def session_done(now):
    return now.weekday() < 5 and now.time() > dtime(15, 30)


# ============================== PATTERNS ==============================
def detect_patterns(df, cfg):
    o, l, c = df["open"], df["low"], df["close"]
    po, pl, pc = o.shift(1), l.shift(1), c.shift(1)
    prev_red, cur_green = pc < po, c > o
    sig = pd.DataFrame({
        "Bullish Engulfing": prev_red & cur_green & (o <= pc) & (c >= po),
        "Bullish Harami": prev_red & cur_green & (o > pc) & (c < po),
        "Tweezer Bottom": prev_red & cur_green & ((l - pl).abs() <= pl * cfg.tweezer_tol_pct / 100),
    })
    if cfg.trend_lookback > 0:
        sig = sig.mul(pc < c.shift(cfg.trend_lookback + 1), axis=0).astype(bool)
    return sig.fillna(False)


def classify_candles(r, tol_pct=TWEEZER_TOL_PCT):
    """Your 3 Excel formulas on two candles (prev_* = row 2, sig_* = row 3). B open, D low, E close.
       Engulfing: AND(B2>E2, B3<E3, B3<=E2, E3>=B2)   Harami: AND(B2>E2, B3<E3, B3>E2, E3<B2)
       Tweezer  : AND(E2<B2, E3>B3, ABS(D3-D2)<=0.001*D2)"""
    try:
        po, pl, pc = float(r["prev_O"]), float(r["prev_L"]), float(r["prev_C"])
        o, l, c = float(r["sig_O"]), float(r["sig_L"]), float(r["sig_C"])
    except (KeyError, TypeError, ValueError):
        return None
    if any(math.isnan(x) for x in (po, pl, pc, o, l, c)):
        return None
    red, green = pc < po, c > o
    names = []
    if red and green and o <= pc and c >= po:
        names.append("Bullish Engulfing")
    if red and green and o > pc and c < po:
        names.append("Bullish Harami")
    if red and green and abs(l - pl) <= pl * tol_pct / 100:
        names.append("Tweezer Bottom")
    return " + ".join(names) if names else "No pattern"


# ============================== ANGEL ONE ==============================
def angel_login(api_key, client_id, mpin, totp_secret):
    """Returns (smart, error_message)."""
    try:
        import pyotp
        from SmartApi import SmartConnect
    except ImportError:
        return None, "Install first:  pip install smartapi-python pyotp"
    try:
        smart = SmartConnect(api_key=api_key.strip())
        otp = pyotp.TOTP(totp_secret.replace(" ", "").strip()).now()
        res = smart.generateSession(client_id.strip(), mpin.strip(), otp)
        if not res or not res.get("status"):
            return None, f"Login failed: {(res or {}).get('message', 'unknown error')}"
        return smart, None
    except Exception as ex:
        return None, f"Login error: {ex}"


def _to_ist_naive(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert(IST).dt.tz_localize(None)


def _candles(smart, item, interval, start, end):
    params = {"exchange": item["exchange"], "symboltoken": str(item["token"]), "interval": interval,
              "fromdate": start.strftime("%Y-%m-%d %H:%M"), "todate": end.strftime("%Y-%m-%d %H:%M")}
    for _ in range(4):                            # Angel limits: ~3 calls/sec and ~180/min
        time.sleep(0.7)
        try:
            res = smart.getCandleData(params)
        except Exception:                         # "Access denied ... exceeding access rate" comes as an exception
            res = None
        if res and res.get("status") and res.get("data") is not None:
            if not res["data"]:
                return pd.DataFrame()
            d = pd.DataFrame(res["data"], columns=["date", "open", "high", "low", "close", "volume"])
            d["date"] = _to_ist_naive(d["date"])
            return d
        time.sleep(3)
    return pd.DataFrame()


def fetch_candles(smart, item, interval, now):
    """Last ~5 days of candles at the chosen timeframe (enough for pattern + downtrend lookback)."""
    d = _candles(smart, item, interval, now - pd.Timedelta(days=5), now)
    return d[["date", "open", "high", "low", "close"]] if not d.empty else d


def fetch_daily(smart, item, now):
    d = _candles(smart, item, "ONE_DAY", now - pd.Timedelta(days=90), now)
    return d[["date", "open", "high", "low", "close"]] if not d.empty else d


def fetch_bars(smart, item, since, now):
    """1-minute candles from `since` to now; index = IST time."""
    d = _candles(smart, item, "ONE_MINUTE", since, now)
    return d.set_index("date")[["open", "high", "low", "close"]] if not d.empty else d


def fetch_ltp(smart, item):
    try:
        res = smart.ltpData(item["exchange"], item["tradingsymbol"], str(item["token"]))
        return float(res["data"]["ltp"])
    except Exception:
        return None


LAST_API_ERROR = {"quote": "", "bulk": ""}       # last Angel message, shown in the warnings when prices do not load


def _quote_from(d):
    ltp = float(d["ltp"])
    return {"open": float(d.get("open", ltp)), "high": float(d.get("high", ltp)), "low": float(d.get("low", ltp)),
            "close": float(d.get("close", ltp)), "ltp": ltp}


def fetch_quote(smart, item):
    """Live price of one stock (the day's open / high / low are NOT used - the table builds the timeframe candle)."""
    try:
        res = smart.ltpData(item["exchange"], item["tradingsymbol"], str(item["token"]))
        d = (res or {}).get("data")
        if not d:
            LAST_API_ERROR["quote"] = str((res or {}).get("message") or res)
            return None
        return _quote_from(d)
    except Exception as ex:
        LAST_API_ERROR["quote"] = f"{type(ex).__name__}: {ex}"
        return None


def fetch_quotes_bulk(smart, items):
    """Live prices for ALL stocks in ONE Angel call (getMarketData). Returns {token: quote};
       empty dict if the call is not available / fails (then it falls back to one-by-one)."""
    out = {}
    try:
        by_ex = {}
        for it in items:
            by_ex.setdefault(it["exchange"], []).append(str(it["token"]))
        res = smart.getMarketData("OHLC", by_ex)
        fetched = ((res or {}).get("data") or {}).get("fetched") or []
        if not fetched:
            LAST_API_ERROR["bulk"] = str((res or {}).get("message") or res)[:200]
        for r in fetched:
            try:
                out[str(r["symbolToken"])] = _quote_from(r)
            except (KeyError, TypeError, ValueError):
                continue
    except Exception as ex:
        LAST_API_ERROR["bulk"] = f"{type(ex).__name__}: {ex}"
        return {}
    return out


# ============================== INSTRUMENTS (CSV) ==============================
NIFTY50_FALLBACK = (
    "ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BEL BHARTIARTL CIPLA "
    "COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK "
    "INDUSINDBK INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE "
    "SBIN SHRIRAMFIN SUNPHARMA TATACONSUM TATAMOTORS TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO").split()


@st.cache_data(ttl=86400, show_spinner=False)
def nifty50():
    try:
        import requests
        r = requests.get("https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        syms = pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
        if len(syms) >= 40:
            return syms
    except Exception:
        pass
    return list(NIFTY50_FALLBACK)


MASTER_CACHE = "scrip_master_cache.pkl"


@st.cache_resource(ttl=86400, show_spinner=False)
def load_master():
    """Whole scrip master kept in memory; also cached on disk for the day, so only the first load is slow."""
    if os.path.exists(MASTER_CACHE) and pd.Timestamp(os.path.getmtime(MASTER_CACHE), unit="s").date() == pd.Timestamp.now().date():
        try:
            return pd.read_pickle(MASTER_CACHE)
        except Exception:
            pass
    import requests
    df = pd.DataFrame(requests.get(MASTER_URL, timeout=120).json())
    df.columns = [c.strip().lower() for c in df.columns]
    keep = [c for c in ("exch_seg", "symbol", "name", "token", "lotsize", "instrumenttype", "expiry") if c in df.columns]
    df = df[keep]
    try:
        df.to_pickle(MASTER_CACHE)
    except Exception:
        pass
    return df


def search_item(smart, exchange, text):
    """Fallback when the list can't be loaded: ask Angel's search API for one name."""
    cache = st.session_state.setdefault("search_cache", {})
    key = f"{exchange}:{text}"
    if key not in cache:
        cache[key] = None
        try:
            rows = (smart.searchScrip(exchange, text) or {}).get("data") or []
            rows = [r for r in rows if r["tradingsymbol"].upper() in (text, text + "-EQ")] or rows
            if rows:
                r = rows[0]
                cache[key] = {"symbol": r["tradingsymbol"].replace("-EQ", ""), "tradingsymbol": r["tradingsymbol"],
                              "token": r["symboltoken"], "exchange": exchange, "lot": 1}
        except Exception:
            pass
    return cache[key]


SENSEX30 = set(
    "ADANIPORTS ASIANPAINT AXISBANK BAJFINANCE BAJAJFINSV BHARTIARTL ETERNAL HCLTECH HDFCBANK HINDUNILVR ICICIBANK "
    "INFY ITC KOTAKBANK LT M&M MARUTI NESTLEIND NTPC POWERGRID RELIANCE SBIN SUNPHARMA TATAMOTORS TATASTEEL TCS "
    "TECHM TITAN TRENT ULTRACEMCO".split())
MCX_MAIN = set("GOLD GOLDM SILVER SILVERM CRUDEOIL CRUDEOILM NATURALGAS COPPER ZINC ALUMINIUM LEAD NICKEL".split())
NFO_INDEX = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


def exchange_list(df, exchange):
    """Only the main, real instruments: NSE = Nifty 50, BSE = Sensex 30,
    NFO = index + Nifty 50 futures, MCX = main commodity futures (nearest expiry of each)."""
    d = df[df["exch_seg"] == exchange].copy()
    nm = d["name"].fillna("").str.upper()
    sy = d["symbol"].fillna("").str.upper()
    if exchange == "NSE":
        d = d[sy.str.endswith("-EQ") & nm.isin(set(nifty50()))]
        d["label"] = d["name"]
    elif exchange == "BSE":
        d = d[nm.isin(SENSEX30) | sy.isin(SENSEX30)]
        d["label"] = d["name"].where(nm.isin(SENSEX30), d["symbol"])
    else:
        itype = d["instrumenttype"].fillna("") if "instrumenttype" in d else pd.Series("", index=d.index)
        base = MCX_MAIN if exchange == "MCX" else NFO_INDEX | set(nifty50())
        d = d[itype.str.startswith("FUT") & nm.isin(base)]
        d["exp"] = pd.to_datetime(d["expiry"], format="%d%b%Y", errors="coerce")
        d = d[d["exp"] >= pd.Timestamp.now().normalize()].sort_values("exp")
        d = d.drop_duplicates("name")                  # nearest expiry only
        d["label"] = d["symbol"]
    return d.drop_duplicates("label").sort_values("label")


@st.cache_data(ttl=3600, show_spinner=False)
def exchange_list_cached(exchange):
    return exchange_list(load_master(), exchange)


def no_dim():
    """Streamlit greys out ('white screen') everything that is still being re-run. Switch that effect off."""
    st.markdown("""<style>
    [data-stale="true"], .stale-element, [data-testid="stElementContainer"][data-stale="true"],
    [data-testid="stSidebar"] [data-stale="true"] { opacity: 1 !important; transition: none !important; }
    </style>""", unsafe_allow_html=True)


def to_item(row, exchange):
    try:                                              # futures trade in lots
        lot = max(int(float(row.get("lotsize") or 1)), 1) if exchange in ("NFO", "MCX") else 1
    except ValueError:
        lot = 1
    return {"symbol": row["label"], "tradingsymbol": row["symbol"], "token": row["token"],
            "exchange": exchange, "lot": lot}


# ============================== STATE ==============================
def load_state(path=STATE_FILE):
    s = json.load(open(path)) if os.path.exists(path) else {}
    for k in ("open", "closed", "handled", "log", "signals"):
        s.setdefault(k, [])
    s.pop("waiting", None)                              # old versions had a waiting queue - gone now
    for g in s["signals"]:
        if str(g.get("status", "")).startswith(("WAITING", "EXPIRED")):
            g["status"] = "SKIPPED (positions were full)"
    return s


def save_state(state, path=STATE_FILE):
    with open(path, "w") as f:
        json.dump(state, f, indent=1, default=str)


def add_log(state, now, msg):
    state["log"].append(f"{pd.Timestamp(now):%Y-%m-%d %H:%M:%S}  {msg}")
    state["log"] = state["log"][-200:]



# ============================== PATTERN OUTLOOK ==============================
def predict_patterns(done, forming, ltp, cfg):
    """Last column of the monitoring table: what could form on the NEXT candle (patterns are confirmed only after a candle closes)."""
    lb = cfg.trend_lookback
    if len(done) < lb + 2:
        return "Not enough candles"
    p = done.iloc[-1]
    po, pc, pl = float(p["open"]), float(p["close"]), float(p["low"])
    if not pc < po:
        return "No setup (last candle not red)"
    if lb > 0 and not pc < float(done["close"].iloc[-1 - lb]):
        return "No setup (no downtrend)"
    return (f"WATCH: Engulfing if close > {po:.2f} | Harami if open > {pc:.2f} and close < {po:.2f} | "
            f"Tweezer if low ~ {pl:.2f}")


# ============================== EXIT (live) ==============================
def _check_exit(pos, bars):
    sl, tg = pos.get("stop_loss"), pos["target"]
    for ts, r in bars[bars.index >= pd.Timestamp(pos["last_checked"])].iterrows():
        if sl:                                           # stop loss is optional
            if r["open"] <= sl:
                return float(r["open"]), ts, "SL (gap)"
            if r["low"] <= sl:
                return sl, ts, "SL"
        if r["open"] >= tg:
            return float(r["open"]), ts, "Target (gap)"
        if r["high"] >= tg:
            return tg, ts, "Target"
    return None


def _close(state, cfg, pos, raw_px, when, reason):
    exit_px = raw_px * (1 - cfg.slippage_pct / 100)
    qty = pos["qty"]
    pnl = (exit_px - pos["entry"]) * qty - cfg.brokerage_pct / 100 * (pos["entry"] + exit_px) * qty
    risk = (pos["entry"] - pos["stop_loss"]) * qty if pos.get("stop_loss") else 0
    done = {k: pos[k] for k in ("symbol", "exchange", "pattern", "signal_date", "entry_time",
                                "entry", "stop_loss", "target", "qty")}
    done.update({k: pos[k] for k in CANDLE_KEYS + ("signal_close",) if k in pos})
    done.update({"exit_time": str(when), "exit": round(exit_px, 2), "reason": reason,
                 "pnl": round(pnl, 2), "r_multiple": round(pnl / risk, 2) if risk > 0 else None})
    state["closed"].append(done)
    state["open"] = [p for p in state["open"] if p is not pos]
    add_log(state, when, f"CLOSED {pos['symbol']} @ {exit_px:.2f}  {reason}  P&L {pnl:,.2f}")


def check_exits(state, cfg, now, bars_fn, ignore_hours=False):
    """Target / SL from 1-minute candles, time exit at 15:15. Exited rows leave Open and land in Closed one by one."""
    for pos in list(state["open"]):
        try:
            bars = bars_fn(pos, pd.Timestamp(pos["last_checked"]))
        except Exception as ex:
            add_log(state, now, f"price error {pos['symbol']}: {ex}")
            continue
        if bars is None or bars.empty:
            continue
        hit = _check_exit(pos, bars)
        if hit:
            _close(state, cfg, pos, hit[0], hit[1], hit[2])
            continue
        pos.setdefault("last_price", float(bars["close"].iloc[-1]))      # live price comes from the quote thread
        pos["last_checked"] = str(bars.index[-1])
        held = int(np.busday_count(pos["entry_time"][:10], str(now.date()))) + 1
        if held >= cfg.max_hold and (ignore_hours or now.time() >= exit_time(pos["exchange"])):
            _close(state, cfg, pos, pos["last_price"], now, "Time exit")


# ============================== ENGINE (background thread) ==============================
def margin_factor(exchange):
    return FUTURES_MARGIN_PCT / 100 if exchange in ("NFO", "MCX") else 1.0


def candle_boundary(now, exchange, tf_min):
    """Start time of the candle that is running right now (candles are aligned to the session start)."""
    start = now.normalize() + (pd.Timedelta(hours=9) if exchange == "MCX" else pd.Timedelta(hours=9, minutes=15))
    if now < start:
        return start
    step = pd.Timedelta(minutes=tf_min)
    return start + ((now - start) // step) * step


class Engine:
    """Does all the Angel One work in a background thread, so the page never waits.
       - candles are downloaded ONCE per new candle (not every refresh)
       - live prices + pattern outlook are refreshed continuously
       - entries / exits happen here; the page only displays a snapshot."""
    GRACE = 4            # seconds to wait after a candle closes before asking Angel for it
    EXIT_EVERY = 15      # seconds between target / SL checks
    QUOTE_EVERY = 1.0    # seconds between live price updates (every table that shows a price)

    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.stop_evt = threading.Event()
        self.qthread = None
        self.state = load_state()
        self.smart = self.cfg = None
        self.creds, self.client_id, self.login_time = None, "", 0.0   # kept in memory only (for the daily re-login)
        self.items, self.exchange, self.ignore = [], "NSE", False
        self._clear_runtime()

    def _clear_runtime(self):
        self.cache, self.quotes, self.rows = {}, {}, []
        self.quote_errors, self.cycle_errors = [], []
        self.status, self.last_update, self.last_exit_check = "", None, 0.0

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive() and not self.stop_evt.is_set()

    # ---- login / resume ----
    def set_login(self, smart, creds, client_id):
        self.smart, self.creds, self.client_id, self.login_time = smart, creds, client_id, time.time()

    def _save_run_config(self, running):
        if self.cfg is None:
            return
        try:
            from dataclasses import asdict
            with open(RUN_CONFIG_FILE, "w") as f:
                json.dump({"running": running, "exchange": self.exchange, "ignore": self.ignore,
                           "items": self.items, "cfg": asdict(self.cfg)}, f, default=str)
        except Exception:
            pass

    def try_autoresume(self):
        """After a restart (PC reboot, crash): log in with the ANGEL_* environment variables and carry on."""
        try:
            if not os.path.exists(RUN_CONFIG_FILE):
                return
            rc = json.load(open(RUN_CONFIG_FILE))
            creds = env_creds()
            if not rc.get("running") or not creds:
                return
            smart, err = angel_login(*creds)
            if err:
                self.status = f"Auto-resume login failed: {err}"
                return
            cd = rc["cfg"]
            cd["patterns"] = tuple(cd.get("patterns", PATTERNS))
            self.set_login(smart, creds, creds[1].strip().upper())
            self.start(smart, Config(**cd), rc["items"], rc["exchange"], rc.get("ignore", False))
            with self.lock:
                add_log(self.state, now_ist(), "Auto-resumed after restart")
        except Exception as ex:
            self.status = f"Auto-resume failed: {ex}"

    # ---- control ----
    def start(self, smart, cfg, items, exchange, ignore_hours=False):
        self.stop()
        with self.lock:
            self.smart, self.cfg, self.items = smart, cfg, list(items)[:50]
            self.exchange, self.ignore = exchange, ignore_hours
            self._clear_runtime()
            self.status = "Starting..."
            self.stop_evt = threading.Event()
            self.thread = threading.Thread(target=self._loop, args=(self.stop_evt,), daemon=True)
            self.qthread = threading.Thread(target=self._quote_loop, args=(self.stop_evt,), daemon=True)
            self.thread.start()
            self.qthread.start()
        self._save_run_config(True)

    def stop(self):
        self.stop_evt.set()
        for t in (self.thread, self.qthread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2)
        self.thread = self.qthread = None
        self.status = "Stopped"
        self._save_run_config(False)

    def reset_all(self):
        self.stop()
        with self.lock:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
            self.state = load_state()
            self._clear_runtime()

    def close_now(self, key):
        with self.lock:
            for p in self.state["open"]:
                if f"{p['exchange']}:{p['symbol']}" == key:
                    _close(self.state, self.cfg or Config(), p, p["last_price"], now_ist(), "Manual exit")
                    save_state(self.state)
                    return

    def delete_position(self, key):
        with self.lock:
            self.state["open"] = [p for p in self.state["open"] if f"{p['exchange']}:{p['symbol']}" != key]
            save_state(self.state)

    def snapshot(self):
        with self.lock:
            return {"state": copy.deepcopy(self.state), "rows": list(self.rows),
                    "errors": list(self.quote_errors) + list(self.cycle_errors),
                    "status": self.status, "last_update": self.last_update, "running": self.running,
                    "cfg": self.cfg, "exchange": self.exchange}

    # ---- worker ----
    def _loop(self, evt):
        while not evt.is_set():
            t0 = time.time()
            try:
                self._cycle(evt)
            except Exception as ex:
                with self.lock:
                    self.cycle_errors = [f"engine error: {ex}"]
            evt.wait(max(1.0, 3.0 - (time.time() - t0)))

    def _quote_loop(self, evt):
        """Own thread: live prices every second (ltp, running candle high/low, open positions' price + P&L).
           It never waits for candle downloads or exit checks, so prices stay live all the time."""
        while not evt.is_set():
            t0 = time.time()
            try:
                cfg, now = self.cfg, now_ist()
                live_ok = self.ignore or market_open(now, self.exchange)
                B = candle_boundary(now, self.exchange, cfg.tf_min)
                errs = []
                self._refresh_quotes(self.items, live_ok, errs, evt, B)
                if evt.is_set():
                    return
                with self.lock:
                    self.quote_errors = errs
                self._build_rows(None, B, pd.Timedelta(minutes=cfg.tf_min), live_ok, final=False, persist=False)
            except Exception as ex:
                with self.lock:
                    self.quote_errors = [f"price error: {ex}"]
            evt.wait(max(0.2, self.QUOTE_EVERY - (time.time() - t0)))

    def _candle_due(self, key, B, now, live_ok):
        c = self.cache.get(key)
        if c is None:
            return True
        if c.get("retry_at") is not None:                       # last download failed -> try again after a short wait
            return time.time() >= c["retry_at"]
        if c["b"] == B or not live_ok:
            return False
        return now >= B + pd.Timedelta(seconds=self.GRACE)

    def _refresh_quotes(self, items, live_ok, errors, evt, B):
        got = fetch_quotes_bulk(self.smart, items)              # ONE call for every stock
        for it in items:
            if evt.is_set():
                return
            key = (it["exchange"], it["symbol"])
            q = got.get(str(it["token"]))
            if q is None:                                       # bulk call missing -> one by one
                q = fetch_quote(self.smart, it)
                time.sleep(0.1)
            if not q:
                errors.append(f"{it['symbol']}: no live price ({LAST_API_ERROR['quote'] or LAST_API_ERROR['bulk'] or 'no reply'})")
                continue
            with self.lock:
                self.quotes[key] = q
                c = self.cache.get(key)
                if c:
                    f = c.get("forming")
                    if live_ok and (f is None or f.get("b") != B):      # a new candle started -> begin a fresh one
                        f = c["forming"] = {"open": q["ltp"], "high": q["ltp"], "low": q["ltp"], "b": B}
                    if f:
                        f["high"], f["low"] = max(f["high"], q["ltp"]), min(f["low"], q["ltp"])
        with self.lock:
            for p in self.state["open"]:
                q = self.quotes.get((p["exchange"], p["symbol"]))
                if q:
                    p["last_price"] = q["ltp"]

    def _build_rows(self, errors, B, tf, live_ok, final=True, persist=True):
        cfg = self.cfg
        with self.lock:
            open_keys = {(p["exchange"], p["symbol"]) for p in self.state["open"]}
            rows = []
            for it in self.items:
                key = (it["exchange"], it["symbol"])
                q, c = self.quotes.get(key), self.cache.get(key)
                row = {"symbol": it["symbol"], "open": None, "high": None, "low": None, "ltp": None,
                       "prev_close": None, "change_%": None, "updated": f"{now_ist():%H:%M:%S}",
                       "position": "IN POSITION" if key in open_keys else "-",
                       "pattern_outlook": "Loading candles..."}
                if q:
                    row["ltp"] = q["ltp"]
                if c and c.get("done") is not None and len(c["done"]):
                    d, f = c["done"], c.get("forming")
                    if live_ok and f:                                # running candle of your timeframe
                        o, h, l, pc = f["open"], f["high"], f["low"], float(d["close"].iat[-1])
                    elif len(d) > 1:                                 # market closed -> last closed candle
                        o, h, l = (float(d[k].iat[-1]) for k in ("open", "high", "low"))
                        pc = float(d["close"].iat[-2])
                    else:
                        o = h = l = pc = None
                    if o is not None:
                        row.update({"open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "prev_close": round(pc, 2),
                                    "change_%": round((q["ltp"] / pc - 1) * 100, 2) if q and pc else None})
                if c and c.get("done") is not None:
                    row["pattern_outlook"] = predict_patterns(c["done"], c.get("forming"), q["ltp"] if q else None, cfg)
                rows.append(row)
            self.rows, self.last_update = rows, now_ist()
            if errors is not None:
                self.cycle_errors = list(errors)
            if final:
                self.status = ("Waiting for the next candle (" + f"{(B + tf):%H:%M}" + ")" if live_ok else
                               "Market closed - prices only, no entry / exit")
            if persist:
                save_state(self.state)

    def _cycle(self, evt):
        if self.creds and time.time() - self.login_time > 8 * 3600:       # Angel session is valid ~1 day
            sm, err = angel_login(*self.creds)
            if sm is not None:
                self.smart, self.login_time = sm, time.time()
        cfg, items, smart, exch = self.cfg, self.items, self.smart, self.exchange
        now = now_ist()
        live_ok = self.ignore or market_open(now, exch)
        tf = pd.Timedelta(minutes=cfg.tf_min)
        B = candle_boundary(now, exch, cfg.tf_min)
        errors = []

        # 1) live prices run in their own thread (_quote_loop, every second) - nothing to do here

        # 2) candles - only when a new candle has closed (table keeps updating while they load)
        due = [it for it in items if self._candle_due((it["exchange"], it["symbol"]), B, now, live_ok)]
        for n, it in enumerate(due, 1):
            if evt.is_set():
                return
            self.status = f"Loading candles {n}/{len(due)}"
            try:
                self._load_candles(it, B, tf, live_ok)
            except Exception as ex:
                errors.append(f"{it['symbol']}: {ex}")

        # 3) exits (target from 1-minute candles, time exit)
        if live_ok and time.time() - self.last_exit_check >= self.EXIT_EVERY:
            with self.lock:
                snap = [dict(p) for p in self.state["open"]]
            bars_map = {}
            for p in snap:
                if evt.is_set():
                    return
                self.status = f"Checking exit {p['symbol']}"
                try:
                    bars_map[(p["exchange"], p["symbol"])] = fetch_bars(smart, p, pd.Timestamp(p["last_checked"]), now_ist())
                except Exception as ex:
                    errors.append(f"{p['symbol']}: exit check failed ({ex})")
            with self.lock:
                check_exits(self.state, cfg, now_ist(), lambda it, since: bars_map.get((it["exchange"], it["symbol"])), self.ignore)
                save_state(self.state)
            self.last_exit_check = time.time()

        # 4) final table
        self._build_rows(errors, B, tf, live_ok, final=True)

    def _load_candles(self, it, B, tf, live_ok):
        cfg = self.cfg
        key = (it["exchange"], it["symbol"])
        df = fetch_candles(self.smart, it, cfg.interval, now_ist())
        if df is None or df.empty:
            fails = self.cache.get(key, {}).get("fails", 0) + 1
            with self.lock:
                self.cache[key] = {"b": None if fails < 3 else B, "done": None, "forming": None, "tries": 0,
                                   "fails": fails if fails < 3 else 0,
                                   "retry_at": time.time() + 20 if fails < 3 else None}
            raise RuntimeError(f"no candle data (try {fails})")
        nowf = now_ist()
        mask = (df["date"] + tf) <= nowf
        done, run = df[mask].reset_index(drop=True), df[~mask]
        forming = None if run.empty else {**{k: float(run.iloc[-1][k]) for k in ("open", "high", "low")}, "b": B}
        tries = self.cache.get(key, {}).get("tries", 0) + 1
        lag = live_ok and not done.empty and done["date"].iat[-1] < B - tf and tries < 3   # Angel not published it yet
        with self.lock:
            self.cache[key] = {"b": None if lag else B, "done": done, "forming": forming, "tries": tries if lag else 0}
        if not lag:
            self._try_entry(it, done)

    def _skip(self, rec, now, key, why):
        """(lock held) a pattern appeared but we cannot enter -> show it in Signals + Log, never silent."""
        state = self.state
        state["handled"] = (state["handled"] + [key])[-2000:]
        rec["status"] = f"SKIPPED ({why})"
        state["signals"] = (state["signals"] + [rec])[-200:]
        add_log(state, now, f"SKIPPED {rec['symbol']}: {why}  ({rec['pattern']})")
        save_state(state)

    def _enter(self, it, rec, ctime, price, now):
        """(lock held) size the position and open it at the live price."""
        cfg, state = self.cfg, self.state
        ex, sym = it["exchange"], it["symbol"]
        tf = pd.Timedelta(minutes=cfg.tf_min)
        entry = price * (1 + cfg.slippage_pct / 100)
        lot = int(it.get("lot", 1))
        mf = margin_factor(ex)                          # futures block only a part of the contract value
        cash = (cfg.capital + sum(t["pnl"] for t in state["closed"])
                - sum(p["entry"] * p["qty"] * p.get("mf", 1.0) for p in state["open"]))
        one_lot = entry * lot * mf                      # money needed for the smallest tradable size
        qty = math.floor(min(cfg.per_position, cash) / one_lot) * lot
        if qty < 1:
            rec["status"] = f"SKIPPED (1 lot needs {one_lot:,.0f})"
            add_log(state, now, f"SKIPPED {sym}: 1 lot needs {one_lot:,.0f} (price {entry:.2f} x lot {lot}"
                                f"{f' x margin {mf * 100:g}%' if mf < 1 else ''}) but amount per position is "
                                f"{cfg.per_position:,.0f}, free cash {cash:,.0f}")
        else:
            state["open"].append({
                "symbol": sym, "tradingsymbol": it["tradingsymbol"], "token": it["token"], "exchange": ex,
                "pattern": rec["pattern"], "signal_date": str(ctime), "signal_close": str(ctime + tf),
                "entry_time": str(now.floor("s")),
                "entry": round(entry, 2),
                "stop_loss": round(entry * (1 - cfg.sl_pct / 100), 2) if cfg.sl_pct > 0 else None,
                "target": round(entry * (1 + cfg.target_pct / 100), 2), "qty": int(qty), "mf": mf, "last_price": price,
                "last_checked": str(now.floor("min")), **{k: rec[k] for k in CANDLE_KEYS}})
            rec["status"] = "ENTERED"
            add_log(state, now, f"ENTRY {sym} @ {entry:.2f}  qty {qty}  TGT {entry * (1 + cfg.target_pct / 100):.2f}  "
                                f"({rec['pattern']})")
        state["signals"] = (state["signals"] + [rec])[-200:]
        save_state(state)

    def _try_entry(self, it, done):
        """Last COMPLETED candle shows a pattern -> entry at the LIVE price.
           All positions full (or any other blocker) -> SKIPPED with the reason, nothing is queued.
           When a position exits, the next FRESH signal that appears takes the free slot."""
        cfg, smart = self.cfg, self.smart
        now = now_ist()
        ex, sym = it["exchange"], it["symbol"]
        if not (self.ignore or market_open(now, ex)):
            return
        if len(done) < cfg.trend_lookback + 3:
            return
        sig = detect_patterns(done, cfg)
        names = [p for p in cfg.patterns if bool(sig[p].iat[-1])]
        if not names:
            return
        tf = pd.Timedelta(minutes=cfg.tf_min)
        ctime = done["date"].iat[-1]
        if not self.ignore and now - (ctime + tf) > min(2 * tf, pd.Timedelta(seconds=FRESH_SECONDS)):
            return                                              # old signal, not fresh
        key = f"{ex}:{sym}|{ctime}"
        with self.lock:
            if key in self.state["handled"]:
                return
        pv, sg = done.iloc[-2], done.iloc[-1]                   # the two candles the pattern was found on
        candles = {f"{p}_{k}": round(float(c[n]), 2) for p, c in (("prev", pv), ("sig", sg))
                   for k, n in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"))}
        rec = {"time": str(now.floor("s")), "symbol": sym, "exchange": ex, "pattern": "+".join(names),
               "candle": str(ctime), **candles, "status": "ENTERED"}

        def blocked():
            """(kind, reason) why we cannot enter right now (lock must be held), or None"""
            st_ = self.state
            if not self.ignore and now.time() >= exit_time(ex):
                return "skip", f"no new entries after {exit_time(ex):%H:%M}"
            if any(p["exchange"] == ex and p["symbol"] == sym for p in st_["open"]):
                return "skip", "already in position"
            if len(st_["open"]) >= cfg.max_open:
                return "skip", f"positions full {len(st_['open'])}/{cfg.max_open}"
            return None

        def handle(b):
            self._skip(rec, now, key, b[1])

        with self.lock:
            b = blocked()
            if b:
                handle(b)
                return
        price = fetch_ltp(smart, it) or float(done["close"].iat[-1])
        with self.lock:
            b = blocked()                                       # state may have changed while we fetched the price
            if b:
                handle(b)
                return
            self.state["handled"] = (self.state["handled"] + [key])[-2000:]
            self._enter(it, rec, ctime, price, now)


def env_creds():
    c = tuple(os.getenv(k, "").strip() for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_MPIN", "ANGEL_TOTP_SECRET"))
    return c if all(c) else None


@st.cache_resource
def get_engine():
    eng = Engine()
    threading.Thread(target=eng.try_autoresume, daemon=True).start()      # no waiting for the page
    return eng



# ============================== ACCESS CONTROL ==============================
# User -> Name + Email -> Request Access -> PENDING
# Admin -> changes Google Sheet Status to APPROVED / REJECTED
# Approved users -> existing Angel One login -> paper trading
#
# Streamlit Cloud Secrets required:
#
# [access_control]
# request_secret = "THE_SAME_SECRET_USED_IN_GOOGLE_APPS_SCRIPT"
#
# The Apps Script URL is already configured below.

ACCESS_APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbw5oSPyD4bjUN6bL8YqsWH94kHo_HvRWszcLecpLDm1wZDO7p6F34nmDSwN-w5SUWo63g/exec"


def _get_access_secret():
    try:
        return str(st.secrets["access_control"]["request_secret"]).strip()
    except Exception:
        return ""


def _access_call(action, name="", email=""):
    secret = _get_access_secret()

    if not secret:
        return {
            "status": "ERROR",
            "message": "Access secret is missing. Add [access_control] request_secret in Streamlit Secrets."
        }

    payload = {
        "action": action,
        "name": name.strip(),
        "email": email.strip().lower(),
        "secret": secret,
        "requested_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        response = requests.post(
            ACCESS_APPS_SCRIPT_URL,
            json=payload,
            timeout=15,
        )

        if response.status_code != 200:
            return {
                "status": "ERROR",
                "message": f"Access server returned HTTP {response.status_code}."
            }

        try:
            return response.json()
        except Exception:
            return {
                "status": "ERROR",
                "message": "Invalid response received from access server."
            }

    except requests.RequestException as exc:
        return {
            "status": "ERROR",
            "message": f"Could not contact access server: {exc}"
        }


def _valid_access_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def access_request_page():
    st.title("🔐 Request Access")
    st.caption(
        "Administrator approval is required before using the Live Paper Trading application."
    )

    saved_email = st.session_state.get("access_email", "")
    saved_name = st.session_state.get("access_name", "")

    if saved_email:
        result = _access_call("check_access", email=saved_email)
        status = str(result.get("status", "")).upper()

        if status == "APPROVED":
            st.session_state["access_granted"] = True
            st.rerun()
        elif status == "PENDING":
            st.info("⏳ Your access request is pending admin approval.")
        elif status == "REJECTED":
            st.error("❌ Your access request was rejected by the administrator.")
        elif status == "ERROR":
            st.warning(result.get("message", "Unable to check access status."))

    with st.form("access_request_form"):
        name = st.text_input(
            "Name",
            value=saved_name,
            placeholder="Enter your name",
        )
        email = st.text_input(
            "Email ID",
            value=saved_email,
            placeholder="Enter your email address",
        )
        submitted = st.form_submit_button(
            "Request Access",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    name = name.strip()
    email = email.strip().lower()

    if not name:
        st.error("Please enter your name.")
        return

    if not _valid_access_email(email):
        st.error("Please enter a valid email address.")
        return

    st.session_state["access_name"] = name
    st.session_state["access_email"] = email

    check = _access_call("check_access", email=email)
    status = str(check.get("status", "NEW")).upper()

    if status == "APPROVED":
        st.session_state["access_granted"] = True
        st.rerun()

    if status == "PENDING":
        st.info("⏳ This email already has a pending access request.")
        return

    if status == "REJECTED":
        st.error("❌ This email was rejected by the administrator.")
        return

    if status == "ERROR":
        st.error(check.get("message", "Could not check access status."))
        return

    result = _access_call(
        "request_access",
        name=name,
        email=email,
    )

    result_status = str(result.get("status", "ERROR")).upper()

    if result_status == "APPROVED":
        st.session_state["access_granted"] = True
        st.rerun()
    elif result_status == "PENDING":
        st.success(
            "✅ Access request submitted successfully. "
            "Please wait for administrator approval."
        )
    elif result_status == "REJECTED":
        st.error("❌ Access was rejected by the administrator.")
    else:
        st.error(result.get("message", "Could not submit the access request."))

    st.caption(
        "After approval, refresh this page. The Angel One login will then appear."
    )


def access_gate():
    if st.session_state.get("access_granted", False):
        return True

    email = st.session_state.get("access_email", "").strip().lower()

    if email:
        result = _access_call("check_access", email=email)
        if str(result.get("status", "")).upper() == "APPROVED":
            st.session_state["access_granted"] = True
            return True

    access_request_page()
    return False


# ============================== UI ==============================
def login_page():
    no_dim()
    page = st.empty()
    with page.container():
        st.title("Angel One Login")
        st.caption("Credentials are used only for this session. Nothing is saved to disk. "
                   "Never share your TOTP secret with anyone.")
        with st.form("login"):
            api_key = st.text_input("API Key", value=os.getenv("ANGEL_API_KEY", ""))
            client_id = st.text_input("Client ID", value=os.getenv("ANGEL_CLIENT_ID", ""))
            mpin = st.text_input("MPIN", type="password")
            totp_secret = st.text_input("TOTP Secret Key", type="password")
            go = st.form_submit_button("Connect", type="primary")
    if not go:
        return
    if not all([api_key, client_id, mpin, totp_secret]):
        st.error("Ella 4 fields-um fill pannunga.")
        return
    with st.spinner("Connecting to Angel One..."):
        smart, err = angel_login(api_key, client_id, mpin, totp_secret)
    if err:
        st.error(err)
        return
    page.empty()                                        # login form maranjidum
    st.title("Connected")
    with st.spinner("Angel One stock list load aagudhu (first time konjam neram aagum)..."):
        try:
            load_master()
        except Exception:
            pass
    get_engine().set_login(smart, (api_key.strip(), client_id.strip(), mpin.strip(), totp_secret.strip()),
                           client_id.strip().upper())
    st.session_state.smart = smart
    st.session_state.client_id = client_id.strip().upper()
    st.rerun()


def excel_check_frame(entries):
    """Two rows per entry (previous candle, signal candle) laid out like your sheet: A label, B open, C high,
       D low, E close. F/G/H hold your Excel formulas (written as formulas, Excel calculates them when the
       CSV is opened); I = the pattern the app used. Entries without stored candles are skipped."""
    rows = []
    rec = sorted([e for e in entries if all(k in e for k in CANDLE_KEYS)], key=lambda e: str(e.get("entry_time")))
    for k, e in enumerate(rec):
        r0, r1 = 2 + 2 * k, 3 + 2 * k                      # sheet rows: header is row 1
        rows.append([f"{e['symbol']} PREVIOUS candle", e["prev_O"], e["prev_H"], e["prev_L"], e["prev_C"], "", "", "", ""])
        rows.append([f"{e['symbol']} SIGNAL candle {e.get('signal_date', '')} to {e.get('signal_close', '?')} (entry {e.get('entry_time', '')})",
                     e["sig_O"], e["sig_H"], e["sig_L"], e["sig_C"],
                     f'=IF(AND(B{r0}>E{r0},B{r1}<E{r1},B{r1}<=E{r0},E{r1}>=B{r0}),"BULLISH","")',
                     f'=IF(AND(B{r0}>E{r0},B{r1}<E{r1},B{r1}>E{r0},E{r1}<B{r0}),"BULLISH harami","")',
                     f'=IF(AND(E{r0}<B{r0},E{r1}>B{r1},ABS(D{r1}-D{r0})<=0.001*D{r0}),"TWEEZER BOTTOM","")',
                     e.get("pattern", "")])
    return pd.DataFrame(rows, columns=["candle", "Open", "High", "Low", "Close", "Engulfing (F)", "Harami (G)",
                                       "Tweezer (H)", "App pattern (I)"])


def log_frame(lines, n=100):
    """Log text lines -> table (latest first): time | event | symbol | details."""
    rows = []
    for ln in reversed(lines[-n:]):
        t, _, msg = ln.partition("  ")
        parts = msg.split(" ", 2)
        if parts[0] in ("ENTRY", "CLOSED", "SKIPPED") and len(parts) >= 2:
            ev, sym, rest = parts[0], parts[1].rstrip(":"), (parts[2] if len(parts) > 2 else "")
        else:
            ev, sym, rest = "INFO", "", msg
        rows.append({"time": t, "event": ev, "symbol": sym, "details": rest})
    return pd.DataFrame(rows, columns=["time", "event", "symbol", "details"])


def _frames():
    """One consistent snapshot + the tables built from it (every fragment calls this)."""
    snap = get_engine().snapshot()
    state, cfg = snap["state"], snap["cfg"] or Config()
    open_df, closed_df = pd.DataFrame(state["open"]), pd.DataFrame(state["closed"])
    if not closed_df.empty:
        old_pat = closed_df["pattern"] if "pattern" in closed_df.columns else [None] * len(closed_df)
        closed_df["pattern"] = [classify_candles(r) or p for r, p in zip(closed_df.to_dict("records"), old_pat)]
        order = ["symbol", "exchange", "signal_date", "signal_close", *CANDLE_KEYS, "pattern", "entry_time", "entry",
                 "exit_time", "exit", "reason", "pnl"]
        closed_df = closed_df[[c for c in order if c in closed_df.columns]].rename(
            columns={"signal_date": "signal_candle_start", "signal_close": "signal_candle_close"})
        closed_df["pnl_%"] = ((closed_df["exit"] / closed_df["entry"] - 1) * 100).round(2)
    if not open_df.empty:
        open_df["live_price"] = open_df["last_price"].round(2)
        open_df["PnL"] = ((open_df["last_price"] - open_df["entry"]) * open_df["qty"]).round(2)
        open_df["PnL_%"] = ((open_df["last_price"] / open_df["entry"] - 1) * 100).round(2)
    return snap, state, cfg, open_df, closed_df


def _csv_button(label, df, name, key):
    """Download button under every table - always visible, disabled until the table has data."""
    empty = df is None or df.empty
    st.download_button(label, b"" if empty else df.to_csv(index=False).encode(), f"{name}_{now_ist():%Y%m%d}.csv",
                       "text/csv", key=key, disabled=empty)


# The page is split in fragments: price tables refresh every SECOND, the rest (signals, closed trades, logs, CSV
# buttons) every few seconds - they only change when a candle closes / a trade exits.
def live_top(exchange, tf_label):
    """Status line + metrics + 1. Monitoring (live price every second)."""
    snap, state, cfg, open_df, closed_df = _frames()
    now = now_ist()
    hrs = "9:00-23:30" if exchange == "MCX" else "9:15-15:30"
    mkt = (f"{exchange} market OPEN" if market_open(now, exchange) else
           f"{exchange} market CLOSED (entry/exit run {hrs} IST, Mon-Fri)")
    upd = f"{snap['last_update']:%H:%M:%S}" if snap["last_update"] is not None else "-"
    st.write(f"**{'RUNNING' if snap['running'] else 'STOPPED'}**  |  {mkt}  |  now {now:%H:%M:%S} IST  |  "
             f"prices updated {upd}  |  timeframe: {tf_label or '-'}  |  per position: {cfg.per_position:,.0f}")
    if snap["running"]:
        st.caption(f"Engine: {snap['status']}")
    else:
        st.info("Exchange + stocks select panni, settings kudutthu **Start** click pannunga. **Stop** na nikkum; trades save aagi irukkum.")

    realized = float(closed_df["pnl"].sum()) if not closed_df.empty else 0.0
    unreal = float(open_df["PnL"].sum()) if not open_df.empty else 0.0
    win = (closed_df["pnl"] > 0).mean() * 100 if not closed_df.empty else 0.0
    m = st.columns(5)
    m[0].metric("Open positions", f"{len(open_df)} / {'no limit' if cfg.no_limit else cfg.max_open}")
    m[1].metric("Closed trades", len(closed_df))
    m[2].metric("Realized P&L", f"{realized:,.0f}")
    invested = float((open_df["entry"] * open_df["qty"]).sum()) if not open_df.empty else 0.0
    m[3].metric("Unrealized P&L", f"{unreal:,.0f}", f"{unreal / invested * 100:.2f}%" if invested else None)
    m[4].metric("Win rate (closed)", f"{win:.0f}%")

    st.subheader("1. Monitoring - live data + pattern outlook")
    mon_df = pd.DataFrame(snap["rows"])
    if mon_df.empty:
        st.write("Stocks select panni Start pannina, live data + pattern outlook inge varum.")
    else:
        st.caption(f"ltp = live price (every second) | open / high / low = running {tf_label or 'timeframe'} candle | "
                   "prev_close = previous candle close | change_% = ltp vs previous candle close | "
                   "pattern_outlook changes only when a candle closes")
        if mon_df["ltp"].isna().all() and snap["errors"]:
            st.warning("Live price varala: " + " | ".join(snap["errors"][:3]))
        st.dataframe(mon_df, use_container_width=True, hide_index=True,
                     column_config={"pattern_outlook": st.column_config.TextColumn("pattern_outlook", width="large")})


def live_mid():
    """Monitoring CSV + 2. Signals (changes only when a candle closes)."""
    snap, state, cfg, open_df, closed_df = _frames()
    _csv_button("Download monitoring CSV", pd.DataFrame(snap["rows"]), "monitoring", "dl_monitor")

    st.subheader("2. Signals")
    sig_df = pd.DataFrame(state["signals"])
    if sig_df.empty:
        st.write("Signals illa (candle mudiyum bodhu pattern vandha inge varum).")
    else:
        st.dataframe(sig_df.iloc[::-1], use_container_width=True, hide_index=True)       # latest on top
    _csv_button("Download signals CSV", sig_df, "signals", "dl_signals")
    if snap["errors"]:
        with st.expander(f"{len(snap['errors'])} warning(s)"):
            st.write("\n".join(snap["errors"]))


def live_positions():
    """3. Live positions (live price + P&L every second)."""
    snap, state, cfg, open_df, closed_df = _frames()
    st.subheader("3. Live positions (entry aana, exit-ku wait)")
    cols = [c for c in ["symbol", "exchange", "pattern", "entry_time", "entry", "live_price", "PnL", "PnL_%"] if c in open_df.columns]
    if open_df.empty:
        st.write("Open positions illa.")
    else:
        st.dataframe(open_df[cols], use_container_width=True, hide_index=True)


def live_bottom():
    """Positions CSV + 4. Closed trades + 5. Logs + 6. Excel check (change only on entry / exit)."""
    snap, state, cfg, open_df, closed_df = _frames()
    cols = [c for c in ["symbol", "exchange", "pattern", "entry_time", "entry", "live_price", "PnL", "PnL_%"] if c in open_df.columns]
    _csv_button("Download live positions CSV", open_df[cols] if not open_df.empty else open_df, "live_positions", "dl_open")

    st.subheader("4. Closed trades (exit aanadhu)")
    show = ["symbol", "exchange", "pattern", "entry_time", "entry", "exit_time", "exit", "reason", "pnl", "pnl_%"]
    if closed_df.empty:
        st.write("Closed trades illa.")
    else:
        st.dataframe(closed_df[[c for c in show if c in closed_df.columns]].rename(columns={"pnl": "PnL"}).iloc[::-1],
                     use_container_width=True, hide_index=True)                      # latest exit on top
    _csv_button("Download closed trades CSV", closed_df, "closed_trades", "dl_closed")   # CSV also has the 2 candles' OHLC

    st.subheader("5. Logs")
    log_df = log_frame(state["log"])
    if log_df.empty:
        st.write("No events yet.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True)                  # latest on top
    _csv_button("Download logs CSV", log_df, "logs", "dl_log")

    st.subheader("6. Excel check - every entry's 2 candles + your 3 formulas")
    check_df = excel_check_frame(state["open"] + state["closed"])
    if check_df.empty:
        st.write("Entry aana piragu inge varum (pazhaiya entries-ku candle data illa).")
    else:
        st.caption("CSV-a Excel-la open pannunga. B=Open, C=High, D=Low, E=Close. F, G, H = unga formulas, I = app sonna pattern. "
                   "F/G/H-la theriyardhum I-layum ore pattern irukkanum.")
        st.dataframe(check_df.iloc[:, :5], use_container_width=True, hide_index=True)
    _csv_button("Download Excel check CSV (open + closed entries)", check_df, "excel_check", "dl_check")


def main_page():
    ss = st.session_state
    smart = ss.smart
    engine = get_engine()
    no_dim()

    st.title("Live Candle Paper Trading")
    st.caption("Bullish Engulfing | Bullish Harami | Tweezer Bottom - signal on the candle close of your timeframe, "
               "entry at LIVE price, exit at target % / SL %")

    with st.sidebar:
        st.write(f"Connected: **{ss.client_id}**")
        if st.button("Logout"):
            engine.stop()
            for k in ("smart", "client_id"):
                ss.pop(k, None)
            st.rerun()

        st.header("Controls")
        c1, c2 = st.columns(2)
        start_clicked = c1.button("Start", type="primary", use_container_width=True)
        if c2.button("Stop", use_container_width=True):
            engine.stop()
        status_ph = st.empty()

        st.header("Stocks")
        items = []
        try:
            master = load_master()
        except Exception as ex:
            master = None
            st.warning(f"Stock list load aagala ({ex}). Names type panni add pannunga.")
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX"])
        if master is not None:
            inst = exchange_list_cached(exchange)
            labels = inst["label"].tolist()
            by_label = inst.set_index("label")
            pick_all = st.checkbox(f"Select all {len(labels)}", key=f"all_{exchange}")
            picked = labels if pick_all else st.multiselect(f"{exchange} stocks ({len(labels)}) - select or type to search",
                                                            labels, key=f"stocks_{exchange}")
            typed = st.text_input("Or type names (comma separated)")
            names, miss = list(picked), []
            for t in [x.strip().upper() for x in typed.split(",") if x.strip()]:
                hit = inst[(inst["label"].str.upper() == t) | (inst["symbol"].str.upper() == t)
                           | (inst["name"].str.upper() == t)]
                if hit.empty:
                    miss.append(t)
                elif hit.iloc[0]["label"] not in names:
                    names.append(hit.iloc[0]["label"])
            if miss:
                st.warning("Not found in " + exchange + ": " + ", ".join(miss))
            items = [to_item({**by_label.loc[n].to_dict(), "label": n}, exchange) for n in names]
        else:
            typed = st.text_input("Type names (comma separated)")
            miss = []
            for t in [x.strip().upper() for x in typed.split(",") if x.strip()]:
                it = search_item(smart, exchange, t)
                items.append(it) if it else miss.append(t)
            if miss:
                st.warning("Not found in " + exchange + ": " + ", ".join(miss))
        if len(items) > 50:
            st.caption("Angel API limit kaaranama mudhal 50 stocks mattum monitor aagum.")

        st.header("Settings")
        if engine.running:
            st.caption("Running - stocks / settings maatha Stop pannittu marubadi Start pannunga.")
        capital = st.number_input("Capital", min_value=10000.0, value=None, step=10000.0,
                                  placeholder="Capital enter pannunga")
        tf_label = st.selectbox("Candle timeframe", list(TIMEFRAMES), index=None, placeholder="Select timeframe")
        NO_LIMIT = "Select max (no limit)"
        max_pick = st.selectbox("Max open positions", [NO_LIMIT] + list(range(1, 26)) + [30, 40, 50], index=0)
        no_limit = max_pick == NO_LIMIT
        if no_limit:
            max_open = 10 ** 6
            st.caption("Max select pannala = position limit illa (capital irukkura varaikum entry edukkum).")
            position_amount = st.number_input("Amount per position", min_value=1.0, value=None, step=1000.0,
                                              placeholder="Amount enter pannunga", key="pp_nolimit")
        else:
            max_open = int(max_pick)
            position_amount = st.number_input("Amount per position (equal weight = capital / max positions)",
                                              min_value=1.0, value=float(round(capital / max_open, 2)) if capital else None,
                                              step=1000.0, placeholder="Amount enter pannunga",
                                              key=f"pp_{capital}_{max_open}")
            if capital and position_amount and position_amount * max_open > capital:
                st.warning(f"{max_open} x {position_amount:,.0f} = {position_amount * max_open:,.0f} capital-ai vida athigam.")
        target_pct = st.number_input("Target (%)", min_value=0.1, max_value=20.0, value=None, step=0.1,
                                     placeholder="Target % enter pannunga")
        ignore_hours = False

        st.header("Manage positions")
        open_now = engine.snapshot()["state"]["open"]
        keys = [f"{p['exchange']}:{p['symbol']}" for p in open_now]
        if keys:
            sel = st.selectbox("Open position", keys)
            m1, m2 = st.columns(2)
            if m1.button("Close now", use_container_width=True, help="Exit at last price, saved in Closed trades"):
                engine.close_now(sel)
                st.rerun()
            if m2.button("Delete", use_container_width=True, help="Remove completely, no record"):
                engine.delete_position(sel)
                st.rerun()
        else:
            st.caption("Open positions illa.")

        st.header("Reset")
        if st.button("Reset all trades", disabled=not st.checkbox("Delete ALL open + closed trades")):
            engine.reset_all()
            st.rerun()

    interval, tf_min = TIMEFRAMES[tf_label] if tf_label else ("FIVE_MINUTE", 5)
    cfg = Config(capital=float(capital or 0), max_open=int(max_open), target_pct=float(target_pct or 0),
                 interval=interval, tf_min=tf_min, position_amount=float(position_amount or 0), no_limit=no_limit)

    if start_clicked:
        missing = [n for n, v in (("Capital", capital), ("Candle timeframe", tf_label),
                                  ("Amount per position", position_amount), ("Target %", target_pct)) if not v]
        if missing:
            st.error("Start pannum munnaadi select / enter pannunga: " + ", ".join(missing))
        elif not items:
            st.error("Stocks select pannunga.")
        else:
            engine.start(engine.smart or smart, cfg, items, exchange, ignore_hours)
    status_ph.write("Status: **RUNNING**" if engine.running else "Status: **STOPPED**")

    # the page itself never waits: the engine works in the background, these blocks only display.
    # price tables refresh every 1 s, signals / closed trades / logs / CSV buttons every 3 s
    frag = getattr(st, "fragment", None) or st.experimental_fragment
    fast = 1 if engine.running else None
    slow = 3 if engine.running else None
    frag(run_every=fast)(live_top)(exchange, tf_label)
    frag(run_every=slow)(live_mid)()
    frag(run_every=fast)(live_positions)()
    frag(run_every=slow)(live_bottom)()


# ============================== ENTRY ==============================
st.set_page_config(page_title="Live Paper Trading", layout="wide")

# ACCESS GATE FIRST.
# No Angel One login or engine auto-resume is initialized until access is approved.
if not access_gate():
    st.stop()

# Existing paper-trading startup logic.
_eng = get_engine()

if "smart" not in st.session_state and _eng.running and _eng.smart is not None:
    st.session_state.smart = _eng.smart
    st.session_state.client_id = _eng.client_id or "running"

if "smart" not in st.session_state:
    login_page()
else:
    main_page()
