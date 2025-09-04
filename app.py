import streamlit as st
import requests, uuid, re, json, time
from typing import Optional, Dict, Any, List
from openai import OpenAI

# -------------------- Secrets (supports [api] or top-level) --------------------
_api = st.secrets.get("api", {})
PUBLIC_API_SECRET = _api.get("PUBLIC_API_SECRET", st.secrets.get("PUBLIC_API_SECRET", ""))
OPENAI_API_KEY    = _api.get("OPENAI_API_KEY",    st.secrets.get("OPENAI_API_KEY", ""))
API               = _api.get("API_BASE",          st.secrets.get("API_BASE", "https://api.public.com"))
ACCOUNT_ID        = _api.get("ACCOUNT_ID",        st.secrets.get("ACCOUNT_ID", None))
MODEL             = "gpt-4o-mini"

st.set_page_config(page_title="Trading Chatbot", page_icon="💬", layout="centered")
st.title("💬 Trading Chatbot")

# -------------------- Clients --------------------
oclient = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Public API helpers (same spirit as your code) --------------------
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
    # exact ticker in CAPS (NVDA, BRK.B, etc.)
    if re.fullmatch(r"[A-Za-z\.-]{1,8}", s) and s.upper() == s:
        return s.upper()
    # fallback: scan instruments by name (simple best-effort)
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

# -------------------- LLM prompts (no JSON is shown to the user) --------------------
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
- For orders, quantity OR amount is required; do NOT assume an order type if not specified.
- LIMIT/STOP_LIMIT require limitPrice; STOP/STOP_LIMIT require stopPrice.
- If anything mandatory is missing, set type="ASK" with a single best follow-up "question" and list "missing".
- JSON only. No chatter.
"""

def llm_interpret(dialog: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = oclient.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content": SYSTEM}] + dialog,
        temperature=0.1,
    )
    return json.loads(resp.choices[0].message.content)

def llm_phrase(text: str) -> str:
    resp = oclient.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":"Rewrite briefly and clearly for an end user."},
            {"role":"user","content":text}
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

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
if "dialog" not in st.session_state: st.session_state.dialog = []  # LLM context
if "messages" not in st.session_state: st.session_state.messages = []  # UI chat history
if "pending_order" not in st.session_state: st.session_state.pending_order = None  # {'order_id','body','summary','symbol'}
if "gather_mode" not in st.session_state: st.session_state.gather_mode = None       # which field are we asking for explicitly

# -------------------- Auth helpers (silent; we do not display IDs) --------------------
def ensure_auth() -> Optional[str]:
    if st.session_state.token:
        return st.session_state.token
    if not PUBLIC_API_SECRET:
        with st.chat_message("assistant"):
            st.error("Missing trading API secret in Streamlit secrets.")
        return None
    try:
        st.session_state.token = get_access_token(PUBLIC_API_SECRET)
        return st.session_state.token
    except requests.HTTPError as e:
        with st.chat_message("assistant"):
            st.error(f"Authentication failed.")
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
        with st.chat_message("assistant"):
            st.error("Could not resolve brokerage account.")
        return None

# -------------------- Chat history rendering --------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------- Helpers --------------------
def say(text: str):
    st.session_state.messages.append({"role":"assistant","content":text})
    with st.chat_message("assistant"):
        st.markdown(text)

def you(text: str):
    st.session_state.messages.append({"role":"user","content":text})
    with st.chat_message("user"):
        st.markdown(text)

def is_confirm(msg: str) -> bool:
    return bool(re.fullmatch(r"\s*(confirm|yes|y|place|ok)\s*", msg.lower()))

def is_cancel(msg: str) -> bool:
    return bool(re.fullmatch(r"\s*(cancel|no|n|stop|abort)\s*", msg.lower()))

def first_missing(missing: List[str]) -> Optional[str]:
    return missing[0] if missing else None

def build_order_body(sym: str, intent: dict) -> dict:
    side = intent.get("side")
    otype = intent.get("orderType")
    qty   = intent.get("quantity")
    amt   = intent.get("amount")
    lpx   = intent.get("limitPrice")
    spx   = intent.get("stopPrice")
    tif   = intent.get("tif")  # DO NOT default; we will ask for it if missing
    body = {
        "orderId": str(uuid.uuid4()),
        "instrument": {"symbol": sym, "type": "EQUITY"},
        "orderSide": side,
        "orderType": otype,
        "expiration": {"timeInForce": tif} if tif else {"timeInForce": None},
    }
    if qty is not None: body["quantity"] = str(qty)
    if amt is not None: body["amount"] = str(amt)
    if lpx is not None: body["limitPrice"] = str(lpx)
    if spx is not None: body["stopPrice"] = str(spx)
    return body

# -------------------- Main chat input --------------------
user_text = st.chat_input("Type here (e.g., 'buy 1 share of NVDA at 120 limit, GTC')")

if user_text:
    user_text = user_text.strip()
    you(user_text)

    # Handle pending order confirmations first
    if st.session_state.pending_order and (is_confirm(user_text) or is_cancel(user_text)):
        if is_cancel(user_text):
            st.session_state.pending_order = None
            st.session_state.gather_mode = None
            say("Cancelled. No order was placed.")
            st.stop()
        else:
            po = st.session_state.pending_order
            token = ensure_auth(); account_id = ensure_account_id()
            if token and account_id:
                try:
                    place_order(token, account_id, po["body"])
                    say(f"Order submitted: **{po['body']['orderId']}**. Checking status…")
                    # brief tracking
                    status = None
                    for _ in range(12):
                        s = get_order(token, account_id, po["body"]["orderId"])
                        if s:
                            status = s
                            status_text = s.get("status", "UNKNOWN")
                            rej = s.get("rejectReason")
                            if rej: status_text += f" — {rej}"
                            say(f"Status: **{status_text}**")
                            if s.get("status") in {"FILLED","REJECTED","CANCELLED","EXPIRED","REPLACED"}:
                                break
                        time.sleep(1.2)
                    if not status:
                        say("Order submitted. Current status isn’t available yet.")
                except requests.HTTPError:
                    say("Order placement failed.")
            st.session_state.pending_order = None
            st.session_state.gather_mode = None
        st.stop()

    # Normal processing
    token = ensure_auth(); account_id = ensure_account_id()
    if not (token and account_id):
        st.stop()

    # Keep LLM dialog context like original
    st.session_state.dialog.append({"role":"user","content":user_text})

    try:
        parsed = llm_interpret(st.session_state.dialog)
    except Exception:
        say("Sorry, I couldn’t interpret that. Try rephrasing.")
        st.stop()

    ptype = parsed.get("type")
    intent = parsed.get("intent") or {}
    missing = parsed.get("missing") or []

    if ptype == "QUOTE":
        # Pull the target symbol/company and show a concise quote
        qstr = llm_extract_target(st.session_state.dialog) or user_text
        sym = resolve_symbol(token, qstr)
        if not sym:
            say(f"I couldn’t find a tradable symbol for “{qstr}”. What ticker did you mean?")
        else:
            try:
                quote = get_equity_quote(token, account_id, sym)
                if not quote:
                    say(f"No quote returned for **{sym}**.")
                else:
                    last = quote.get("last") or quote.get("lastPrice") or quote.get("price") or quote.get("closePrice")
                    bid, ask = quote.get("bid"), quote.get("ask")
                    vol = quote.get("volume")
                    say(llm_phrase(f"{sym} is around {last}. Bid {bid}, ask {ask}. Volume {vol}."))
            except Exception as e:
                say(f"Quote lookup failed.")
        st.stop()

    if ptype == "ASK" or missing:
        # Don’t assume anything. Ask for the next most important missing field.
        need = first_missing(missing)
        # Prefer to gather in a sensible order
        priority = ["symbol", "side", "orderType", "quantity", "amount", "limitPrice", "stopPrice", "tif"]
        need = next((f for f in priority if f in missing), need)

        prompts = {
            "symbol": "Which ticker or company do you want to trade?",
            "side": "Buy or Sell?",
            "orderType": "What order type: Market, Limit, Stop, or Stop Limit?",
            "quantity": "How many shares?",
            "amount": "Alternatively, what dollar amount?",
            "limitPrice": "What limit price?",
            "stopPrice": "What stop price?",
            "tif": "Time in force: DAY or GTC?",
        }
        # If user said “buy 1 stock”, do not guess. Ask for ticker first, then orderType, then TIF, etc.
        say(prompts.get(need, parsed.get("question") or "Can you share the missing details?"))
        st.session_state.gather_mode = need
        st.stop()

    # If we’re here, we have enough to build an order — but STILL don’t assume defaults.
    symbol_input = intent.get("symbol")
    sym = resolve_symbol(token, symbol_input) if symbol_input else None
    if not sym:
        say("Which ticker do you want to trade?")
        st.session_state.gather_mode = "symbol"
        st.stop()

    # Check TIF; if missing, ask.
    if not intent.get("tif"):
        say("Time in force for this order: **DAY** or **GTC**?")
        st.session_state.gather_mode = "tif"
        st.stop()

    # If order type demands prices, ensure they’re present.
    otype = intent.get("orderType")
    if otype in {"LIMIT", "STOP_LIMIT"} and intent.get("limitPrice") is None:
        say("What **limit price** do you want?")
        st.session_state.gather_mode = "limitPrice"
        st.stop()
    if otype in {"STOP", "STOP_LIMIT"} and intent.get("stopPrice") is None:
        say("What **stop price** do you want?")
        st.session_state.gather_mode = "stopPrice"
        st.stop()

    # Need either quantity or amount
    if intent.get("quantity") is None and intent.get("amount") is None:
        say("How many shares, or what dollar amount?")
        st.session_state.gather_mode = "quantity"  # we’ll accept either in next turn
        st.stop()

    # Build body WITHOUT assuming defaults (tif is guaranteed above)
    body = build_order_body(sym, intent)

    # Human summary
    side = intent.get("side")
    qty  = intent.get("quantity")
    amt  = intent.get("amount")
    tif  = intent.get("tif")
    lpx  = intent.get("limitPrice")
    spx  = intent.get("stopPrice")
    summary_parts = [f"{side}"]
    summary_parts.append(f"{qty} share(s)" if qty is not None else f"${amt}")
    summary_parts.append(f"of {sym}")
    summary_parts.append(f"as {otype}")
    if lpx is not None: summary_parts.append(f"@ {lpx}")
    if spx is not None: summary_parts.append(f"stop {spx}")
    summary_parts.append(f"({tif})")
    human_summary = " ".join(summary_parts)

    say("Please review: " + llm_phrase(human_summary))
    st.session_state.pending_order = {"body": body}
    say("Type **confirm** to place this order, or **cancel** to abort.")
