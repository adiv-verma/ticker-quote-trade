import streamlit as st
import requests, uuid, re, json, time
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI

# -------------------- Secrets (supports [api]/[auth] or top-level) --------------------
_api  = st.secrets.get("api", {})
_auth = st.secrets.get("auth", {})

PUBLIC_API_SECRET = _api.get("PUBLIC_API_SECRET", st.secrets.get("PUBLIC_API_SECRET", ""))
OPENAI_API_KEY    = _api.get("OPENAI_API_KEY",    st.secrets.get("OPENAI_API_KEY", ""))
API               = _api.get("API_BASE",          st.secrets.get("API_BASE", "https://api.public.com"))
ACCOUNT_ID        = _api.get("ACCOUNT_ID",        st.secrets.get("ACCOUNT_ID", None))

APP_PASSWORD      = _auth.get("APP_PASSWORD",     st.secrets.get("APP_PASSWORD", ""))  # <- set this in secrets
MODEL             = "gpt-4o-mini"

st.set_page_config(page_title="Trading Chatbot", page_icon="💬", layout="centered")
st.title("💬 Trading Chatbot")

# -------------------- Password Gate --------------------
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    with st.form("login", clear_on_submit=False):
        pw = st.text_input("Enter password", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if APP_PASSWORD and pw == APP_PASSWORD:
            st.session_state.authed = True
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# -------------------- Clients --------------------
oclient = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Public API helpers --------------------
def get_access_token(secret: str, minutes: int = 15) -> str:
    r = requests.post(
        f"{API}/userapiauthservice/personal/access-tokens",
        json={"validityInMinutes": minutes, "secret": secret},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["accessToken"]

def get_brokerage_account_id(token: str) -> str:
    r = requests.get(
        f"{API}/userapigateway/trading/account",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    r.raise_for_status()
    accts = r.json()["accounts"]
    acct = next(a for a in accts if a.get("accountType") == "BROKERAGE")
    return acct["accountId"]

def list_instruments(token: str, page_size: int = 400) -> List[dict]:
    out, next_page = [], None
    headers = {"Authorization": f"Bearer {token}"}
    while True:
        params = {"pageSize": page_size}
        if next_page: params["pageToken"] = next_page
        r = requests.get(f"{API}/userapigateway/instruments", headers=headers, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("instruments", []))
        next_page = j.get("nextPageToken")
        if not next_page:
            break
    return out

def resolve_symbol(token: str, symbol_or_name: str) -> Optional[str]:
    s = (symbol_or_name or "").strip()
    if not s:
        return None
    if re.fullmatch(r"[A-Za-z\.-]{1,8}", s) and s.upper() == s:
        return s.upper()
    for it in list_instruments(token):
        if it.get("type") != "EQUITY":
            continue
        if it.get("symbol", "").upper() == s.upper():
            return it["symbol"]
        name = (it.get("name") or "").lower()
        if s.lower() in name:
            return it["symbol"]
    return None

def get_equity_quote(token: str, account_id: str, symbol: str) -> Optional[dict]:
    r = requests.post(
        f"{API}/userapigateway/marketdata/{account_id}/quotes",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"instruments": [{"symbol": symbol.upper(), "type": "EQUITY"}]},
        timeout=15,
    )
    r.raise_for_status()
    quotes = r.json().get("quotes", [])
    return quotes[0] if quotes else None

def place_order(token: str, account_id: str, body: dict) -> dict:
    r = requests.post(
        f"{API}/userapigateway/trading/{account_id}/order",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body, timeout=20,
    )
    r.raise_for_status()
    return r.json()

def get_order(token: str, account_id: str, order_id: str) -> Optional[dict]:
    r = requests.get(
        f"{API}/userapigateway/trading/{account_id}/order/{order_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

# -------------------- LLM helpers --------------------
SYSTEM = """You are a trading assistant that interprets user messages into either a QUOTE query or an ORDER intent for equities.
Return STRICT JSON with this shape:

{
  "type": "QUOTE" | "ORDER" | "ASK",
  "question": string | null,
  "missing": string[] | null,
  "intent": {
    "symbol": string | null,
    "side": "BUY" | "SELL" | null,
    "orderType": "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT" | null,
    "quantity": number | null,
    "amount": number | null,
    "limitPrice": number | null,
    "stopPrice": number | null,
    "tif": "DAY" | "GTC" | null
  },
    "summary": string | null
}

Rules:
- If user asks for price (e.g., "price of X", "quote X"), set type="QUOTE".
- For orders, quantity OR amount is required; do NOT assume any default if missing.
- LIMIT/STOP_LIMIT require limitPrice; STOP/STOP_LIMIT require stopPrice.
- If anything mandatory is missing, set type="ASK" with one best follow-up "question" and list "missing".
- JSON only.
"""

def llm_interpret(dialog: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = oclient.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content": SYSTEM}] + dialog,
        temperature=0.1,
    )
    return json.loads(resp.choices[0].message.content)

def llm_extract_target(dialog: List[Dict[str, str]]) -> Optional[str]:
    resp = oclient.chat.completions.create(
        model=MODEL,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":"Extract the target ticker or company name as JSON: {\"query\": string}. If none, put null."},
            *dialog
        ],
        temperature=0.0,
    )
    try:
        return json.loads(resp.choices[0].message.content).get("query")
    except Exception:
        return None

# -------------------- Session state --------------------
if "token" not in st.session_state: st.session_state.token = None
if "account_id" not in st.session_state: st.session_state.account_id = ACCOUNT_ID
if "dialog" not in st.session_state: st.session_state.dialog = []  # LLM history
if "messages" not in st.session_state: st.session_state.messages = []  # chat UI
if "pending_order" not in st.session_state: st.session_state.pending_order = None  # {'order_id','body','intent'}
if "pending_intent" not in st.session_state: st.session_state.pending_intent = None  # dict being filled incrementally
if "gather_mode" not in st.session_state: st.session_state.gather_mode = None       # which field we’re asking for
if "last_status" not in st.session_state: st.session_state.last_status = None

# -------------------- Auth (silent) --------------------
def ensure_auth() -> Optional[str]:
    if st.session_state.token:
        return st.session_state.token
    if not PUBLIC_API_SECRET:
        say("Missing trading API secret in Streamlit secrets.")
        return None
    try:
        st.session_state.token = get_access_token(PUBLIC_API_SECRET)
        return st.session_state.token
    except requests.HTTPError:
        say("Authentication failed.")
        return None

def ensure_account_id() -> Optional[str]:
    if st.session_state.account_id:
        return st.session_state.account_id
    token = ensure_auth()
    if not token:
        return None
    try:
        st.session_state.account_id = get_brokerage_account_id(token)
        return st.session_state.account_id
    except Exception:
        say("Could not resolve brokerage account.")
        return None

# -------------------- Chat rendering helpers --------------------
def say(text: str):
    st.session_state.messages.append({"role":"assistant","content":text})
    with st.chat_message("assistant"):
        st.markdown(text)

def you(text: str):
    st.session_state.messages.append({"role":"user","content":text})
    with st.chat_message("user"):
        st.markdown(text)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------- Parsing helpers for gather_mode answers --------------------
def normalize_side(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s in {"buy","b"}: return "BUY"
    if s in {"sell","s"}: return "SELL"
    return None

def normalize_ordertype(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s in {"m","market"}: return "MARKET"
    if s in {"l","limit"}: return "LIMIT"
    if s in {"stop"}: return "STOP"
    if s in {"stop limit","stop_limit","sl","stoplimit"}: return "STOP_LIMIT"
    return None

def normalize_tif(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s == "day": return "DAY"
    if s == "gtc": return "GTC"
    return None

def try_float(s: str) -> Optional[float]:
    try:
        return float(s.replace("$","").replace(",","").strip())
    except Exception:
        return None

def try_int(s: str) -> Optional[int]:
    try:
        return int(s.replace(",","").strip())
    except Exception:
        return None

def next_missing(intent: dict) -> Optional[str]:
    # required: symbol, side, orderType, tif, and either quantity or amount
    # plus price fields depending on orderType
    if not intent.get("symbol"): return "symbol"
    if not intent.get("side"): return "side"
    if not intent.get("orderType"): return "orderType"
    otype = intent.get("orderType")
    if otype in {"LIMIT","STOP_LIMIT"} and intent.get("limitPrice") is None: return "limitPrice"
    if otype in {"STOP","STOP_LIMIT"} and intent.get("stopPrice")  is None: return "stopPrice"
    if intent.get("quantity") is None and intent.get("amount") is None: return "quantity"
    if not intent.get("tif"): return "tif"
    return None

def prompt_for(field: str) -> str:
    prompts = {
        "symbol": "Which ticker do you want to trade?",
        "side": "Buy or Sell?",
        "orderType": "What order type: Market, Limit, Stop, or Stop Limit?",
        "limitPrice": "What limit price?",
        "stopPrice": "What stop price?",
        "quantity": "How many shares (or say a dollar amount instead)?",
        "tif": "Time in force: DAY or GTC?",
    }
    return prompts.get(field, "Please provide the missing detail.")

def build_order_body(sym: str, intent: dict) -> dict:
    body = {
        "orderId": str(uuid.uuid4()),
        "instrument": {"symbol": sym, "type": "EQUITY"},
        "orderSide": intent["side"],
        "orderType": intent["orderType"],
        "expiration": {"timeInForce": intent["tif"]},
    }
    if intent.get("quantity") is not None: body["quantity"] = str(intent["quantity"])
    if intent.get("amount")   is not None: body["amount"]   = str(intent["amount"])
    if intent.get("limitPrice") is not None: body["limitPrice"] = str(intent["limitPrice"])
    if intent.get("stopPrice")  is not None: body["stopPrice"]  = str(intent["stopPrice"])
    return body

def summarize(sym: str, intent: dict) -> str:
    side = intent.get("side")
    qty  = intent.get("quantity")
    amt  = intent.get("amount")
    otype = intent.get("orderType")
    tif  = intent.get("tif")
    lpx  = intent.get("limitPrice")
    spx  = intent.get("stopPrice")
    parts = [side]
    parts.append(f"{qty} share(s)" if qty is not None else f"${amt}")
    parts.append(f"of {sym}")
    parts.append(f"as {otype}")
    if lpx is not None: parts.append(f"@ {lpx}")
    if spx is not None: parts.append(f"stop {spx}")
    parts.append(f"({tif})")
    return " ".join(parts)

# -------------------- Main Chat Input --------------------
placeholder = "Check quotes or place orders (e.g., 'quote NVDA', 'buy 1 AAPL limit 190 GTC')"
user_text = st.chat_input(placeholder)

# Quick commands
def is_confirm(msg: str) -> bool:
    return bool(re.fullmatch(r"\s*(confirm|yes|y|place|ok)\s*", msg.lower()))
def is_cancel(msg: str) -> bool:
    return bool(re.fullmatch(r"\s*(cancel|no|n|stop|abort)\s*", msg.lower()))

# -------------------- Process input --------------------
if user_text:
    user_text = user_text.strip()
    you(user_text)

    token = st.session_state.token or (get_access_token(PUBLIC_API_SECRET) if PUBLIC_API_SECRET else None)
    st.session_state.token = token
    if token and not st.session_state.account_id:
        try:
            st.session_state.account_id = ACCOUNT_ID or get_brokerage_account_id(token)
        except Exception:
            say("Could not resolve brokerage account.")
            st.stop()

    # 1) If awaiting confirmation of a ready order
    if st.session_state.pending_order and (is_confirm(user_text) or is_cancel(user_text)):
        if is_cancel(user_text):
            st.session_state.pending_order = None
            st.session_state.pending_intent = None
            st.session_state.gather_mode = None
            say("Cancelled. No order was placed.")
            st.stop()
        else:
            po = st.session_state.pending_order
            try:
                place_order(st.session_state.token, st.session_state.account_id, po["body"])
                say(f"Order submitted: **{po['body']['orderId']}**. Checking status…")
                # Only show status when it changes
                st.session_state.last_status = None
                for _ in range(12):
                    s = get_order(st.session_state.token, st.session_state.account_id, po["body"]["orderId"])
                    if s:
                        status_text = s.get("status","UNKNOWN")
                        if status_text != st.session_state.last_status:
                            st.session_state.last_status = status_text
                            say(f"Status: **{status_text}**")
                        if status_text in {"FILLED","REJECTED","CANCELLED","EXPIRED","REPLACED"}:
                            break
                    time.sleep(1.2)
                if st.session_state.last_status is None:
                    say("Order submitted. Current status isn’t available yet.")
            except requests.HTTPError:
                say("Order placement failed.")
            finally:
                st.session_state.pending_order = None
                st.session_state.pending_intent = None
                st.session_state.gather_mode = None
        st.stop()

    # 2) If we are in gather_mode, capture this answer directly (no LLM re-ask)
    if st.session_state.gather_mode and st.session_state.pending_intent is not None:
        field = st.session_state.gather_mode
        val = user_text

        if field == "symbol":
            sym = resolve_symbol(st.session_state.token, val)
            if not sym:
                say("I couldn’t resolve that ticker—try the exact symbol (e.g., NVDA).")
                st.stop()
            st.session_state.pending_intent["symbol"] = sym

        elif field == "side":
            side = normalize_side(val)
            if not side:
                say("Please reply with **Buy** or **Sell**.")
                st.stop()
            st.session_state.pending_intent["side"] = side

        elif field == "orderType":
            ot = normalize_ordertype(val)
            if not ot:
                say("Please reply with one of: **Market, Limit, Stop, Stop Limit**.")
                st.stop()
            st.session_state.pending_intent["orderType"] = ot

        elif field == "limitPrice":
            f = try_float(val)
            if f is None:
                say("Please reply with a valid limit price (number).")
                st.stop()
            st.session_state.pending_intent["limitPrice"] = f

        elif field == "stopPrice":
            f = try_float(val)
            if f is None:
                say("Please reply with a valid stop price (number).")
                st.stop()
            st.session_state.pending_intent["stopPrice"] = f

        elif field == "quantity":
            # Accept either shares or dollar amount like "$500"
            if re.search(r"^\s*\$", val):
                amt = try_float(val)
                if amt is None:
                    say("Please reply with a number like $500 or a share count like 5.")
                    st.stop()
                st.session_state.pending_intent["amount"] = amt
                st.session_state.pending_intent.pop("quantity", None)
            else:
                qty = try_int(val)
                if qty is None:
                    say("Please reply with a whole number of shares (e.g., 1, 10) or $amount.")
                    st.stop()
                st.session_state.pending_intent["quantity"] = qty
                st.session_state.pending_intent.pop("amount", None)

        elif field == "tif":
            t = normalize_tif(val)
            if not t:
                say("Please reply with **DAY** or **GTC**.")
                st.stop()
            st.session_state.pending_intent["tif"] = t

        # See what’s still missing
        need = next_missing(st.session_state.pending_intent)
        if need:
            st.session_state.gather_mode = need
            say(prompt_for(need))
            st.stop()
        else:
            # Build & confirm
            sym = st.session_state.pending_intent["symbol"]
            body = build_order_body(sym, st.session_state.pending_intent)
            st.session_state.pending_order = {"body": body}
            say("Please review: " + summarize(sym, st.session_state.pending_intent))
            say("Type **confirm** to place this order, or **cancel** to abort.")
            st.stop()

    # 3) Otherwise, interpret this new message from scratch
    # Ensure auth first
    if not (st.session_state.token and st.session_state.account_id):
        say("Missing credentials. Check secrets.")
        st.stop()

    # Try to interpret with LLM once
    st.session_state.dialog.append({"role":"user","content":user_text})
    try:
        parsed = llm_interpret(st.session_state.dialog)
    except Exception:
        say("Sorry, I couldn’t interpret that. Try rephrasing.")
        st.stop()

    ptype = parsed.get("type")
    intent = parsed.get("intent") or {}

    if ptype == "QUOTE":
        qstr = llm_extract_target(st.session_state.dialog) or user_text
        sym = resolve_symbol(st.session_state.token, qstr)
        if not sym:
            say(f"I couldn’t find a tradable symbol for “{qstr}”. What ticker did you mean?")
        else:
            try:
                quote = get_equity_quote(st.session_state.token, st.session_state.account_id, sym)
                if not quote:
                    say(f"No quote returned for **{sym}**.")
                else:
                    last = quote.get("last") or quote.get("lastPrice") or quote.get("price") or quote.get("closePrice")
                    bid, ask = quote.get("bid"), quote.get("ask")
                    vol = quote.get("volume")
                    # Keep it one line and clean; avoid funky formatting
                    say(f"**{sym}** ≈ {last}. Bid {bid}, ask {ask}. Volume {vol}.")
            except Exception:
                say("Quote lookup failed.")
        st.stop()

    if ptype == "ASK":
        # Seed pending_intent with what the LLM already parsed, then start gathering
        st.session_state.pending_intent = intent
        need = next_missing(st.session_state.pending_intent)
        st.session_state.gather_mode = need
        say(prompt_for(need))
        st.stop()

    if ptype == "ORDER":
        # If anything is missing, enter gather mode
        st.session_state.pending_intent = intent
        need = next_missing(st.session_state.pending_intent)
        if need:
            st.session_state.gather_mode = need
            say(prompt_for(need))
            st.stop()
        else:
            sym = st.session_state.pending_intent["symbol"]
            body = build_order_body(sym, st.session_state.pending_intent)
            st.session_state.pending_order = {"body": body}
            say("Please review: " + summarize(sym, st.session_state.pending_intent))
            say("Type **confirm** to place this order, or **cancel** to abort.")
            st.stop()
