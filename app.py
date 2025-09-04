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

# -------------------- Public API helpers (same as your code) --------------------
def get_access_token(secret: str, minutes: int = 15) -> str:
    r = requests.post(
        f"{API}/userapiauthservice/personal/access-tokens",
        json={"validityInMinutes": minutes, "secret": secret},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["accessToken"]

def get_brokerage_account_id(token: str) -> str:
    # Optional: only used if ACCOUNT_ID not provided in secrets
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

def preflight_single_leg(token: str, account_id: str, body: dict) -> dict:
    r = requests.post(
        f"{API}/userapigateway/trading/{account_id}/preflight/single-leg",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body, timeout=20,
    )
    r.raise_for_status()
    return r.json()

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

# -------------------- LLM prompts (same spirit as original) --------------------
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
- If user asks for price (e.g., "price of X", "quote X"), set type="QUOTE" and ignore intent.
- For orders, quantity OR amount is required; at least one must be present.
- LIMIT or STOP_LIMIT require limitPrice; STOP or STOP_LIMIT require stopPrice.
- If anything mandatory is missing, set type="ASK", include a single best "question" to ask next, list "missing".
- Keep "summary" one sentence, user-friendly.
- Do not include any extra keys or commentary. JSON only.
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
    # Used to pull the symbol/company when user did a plain "quote" ask
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

# -------------------- Auth helpers (silent; no account id shown) --------------------
def ensure_auth() -> Optional[str]:
    if st.session_state.token:
        return st.session_state.token
    if not PUBLIC_API_SECRET:
        # show a generic error without leaking anything
        with st.chat_message("assistant"):
            st.error("Missing trading API secret. Please add it to Streamlit secrets.")
        return None
    try:
        st.session_state.token = get_access_token(PUBLIC_API_SECRET)
        return st.session_state.token
    except requests.HTTPError as e:
        with st.chat_message("assistant"):
            st.error(f"Authentication failed: {e}")
        return None

def ensure_account_id() -> Optional[str]:
    if st.session_state.account_id:
        return st.session_state.account_id
    token = ensure_auth()
    if not token:
        return None
    # Try to fetch if not provided; still do not display it.
    try:
        st.session_state.account_id = get_brokerage_account_id(token)
        return st.session_state.account_id
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Could not resolve brokerage account: {e}")
        return None

# -------------------- Chat history rendering --------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["type"] == "text":
            st.markdown(m["content"])
        elif m["type"] == "json":
            st.json(m["content"])

# -------------------- Main chat input --------------------
user_text = st.chat_input("Type a message (e.g., 'Quote NVDA' or 'Buy 5 AAPL limit 190 GTC')")

def add_assistant_text(msg: str):
    st.session_state.messages.append({"role":"assistant","type":"text","content":msg})
    with st.chat_message("assistant"):
        st.markdown(msg)

def add_assistant_json(obj: dict, header: Optional[str] = None):
    st.session_state.messages.append({"role":"assistant","type":"json","content":obj})
    with st.chat_message("assistant"):
        if header:
            st.markdown(header)
        st.json(obj)

def add_user_text(msg: str):
    st.session_state.messages.append({"role":"user","type":"text","content":msg})
    with st.chat_message("user"):
        st.markdown(msg)

# -------------------- Handle confirmation / cancellation quick paths --------------------
def is_confirm(msg: str) -> bool:
    return bool(re.fullmatch(r"\s*(confirm|yes|y|place|ok)\s*", msg.lower()))

def is_cancel(msg: str) -> bool:
    return bool(re.fullmatch(r"\s*(cancel|no|n|stop|abort)\s*", msg.lower()))

if user_text is not None:
    user_text = user_text.strip()
    if not user_text:
        pass
    else:
        add_user_text(user_text)

        # If we have a pending order and user replies confirm/cancel, handle that first.
        if st.session_state.pending_order and (is_confirm(user_text) or is_cancel(user_text)):
            if is_cancel(user_text):
                st.session_state.pending_order = None
                add_assistant_text("Order cancelled. No order was placed.")
            else:
                # Place the pending order
                po = st.session_state.pending_order
                token = ensure_auth(); account_id = ensure_account_id()
                if token and account_id:
                    try:
                        place_order(token, account_id, po["body"])
                        add_assistant_text(f"Order submitted: **{po['order_id']}**. Checking status...")
                        # brief tracking
                        status = None
                        for _ in range(12):
                            s = get_order(token, account_id, po["order_id"])
                            if s:
                                status = s
                                st.session_state.messages.append({"role":"assistant","type":"json","content":s})
                                with st.chat_message("assistant"):
                                    st.json(s)
                                if s.get("status") in {"FILLED","REJECTED","CANCELLED","EXPIRED","REPLACED"}:
                                    break
                            time.sleep(1.2)
                        if not status:
                            add_assistant_text("Order submitted. Current status isn’t available yet.")
                    except requests.HTTPError as e:
                        add_assistant_text("Order placement failed.")
                        st.session_state.messages.append({"role":"assistant","type":"text","content":f"```\n{e.response.text}\n```"})
                        with st.chat_message("assistant"):
                            st.code(e.response.text)
                st.session_state.pending_order = None
            st.stop()

        # Normal chat flow
        # Ensure auth before making any API actions later
        token = ensure_auth(); account_id = ensure_account_id()
        if not (token and account_id):
            st.stop()

        # Update LLM dialog and interpret
        st.session_state.dialog.append({"role":"user","content":user_text})
        try:
            parsed = llm_interpret(st.session_state.dialog)
        except Exception as e:
            add_assistant_text(f"Sorry, I couldn't interpret that: {e}")
            st.stop()

        # Show interpretation for transparency (optional—kept minimal)
        # add_assistant_json(parsed, header="Interpretation:")

        ptype = parsed.get("type")

        if ptype == "QUOTE":
            # Extract target, resolve symbol, fetch quote
            qstr = llm_extract_target(st.session_state.dialog) or user_text
            sym = resolve_symbol(token, qstr)
            if not sym:
                add_assistant_text(f"I couldn’t find a tradable symbol for “{qstr}”. Try the exact ticker.")
            else:
                try:
                    quote = get_equity_quote(token, account_id, sym)
                    if not quote:
                        add_assistant_text(f"No quote returned for **{sym}**.")
                    else:
                        last = quote.get("last") or quote.get("lastPrice") or quote.get("price") or quote.get("closePrice")
                        bid, ask = quote.get("bid"), quote.get("ask")
                        vol = quote.get("volume")
                        add_assistant_text(llm_phrase(f"{sym} is around {last}. Bid {bid}, ask {ask}. Volume {vol}."))
                        add_assistant_json(quote)
                except Exception as e:
                    add_assistant_text(f"Quote lookup failed: {e}")

        elif ptype == "ASK":
            q = parsed.get("question") or "Can you clarify the missing fields?"
            add_assistant_text(q)

        else:  # ORDER (or sufficient info parsed)
            intent = parsed.get("intent") or {}
            missing = parsed.get("missing") or []

            if missing:
                add_assistant_text(parsed.get("question") or f"I still need: {', '.join(missing)}.")
            else:
                symbol_input = intent.get("symbol")
                sym = resolve_symbol(token, symbol_input) if symbol_input else None
                if not sym:
                    add_assistant_text(f"I couldn’t resolve a ticker for “{symbol_input}”. Please provide the symbol (e.g., NVDA).")
                else:
                    side = intent.get("side")
                    otype = intent.get("orderType")
                    qty   = intent.get("quantity")
                    amt   = intent.get("amount")
                    lpx   = intent.get("limitPrice")
                    spx   = intent.get("stopPrice")
                    tif   = (intent.get("tif") or "DAY")

                    order_id = str(uuid.uuid4())
                    body = {
                        "orderId": order_id,
                        "instrument": {"symbol": sym, "type": "EQUITY"},
                        "orderSide": side,
                        "orderType": otype,
                        "expiration": {"timeInForce": tif},
                    }
                    if qty is not None: body["quantity"] = str(qty)
                    if amt is not None: body["amount"] = str(amt)
                    if lpx is not None: body["limitPrice"] = str(lpx)
                    if spx is not None: body["stopPrice"] = str(spx)

                    human_summary = parsed.get("summary") or f"{side} {qty or ('$'+str(amt))} of {sym} as {otype} ({tif})" + (f" @ {lpx}" if lpx else "") + (f" stop {spx}" if spx else "")
                    add_assistant_text("Please review: " + llm_phrase(human_summary))

                    # Preflight
                    try:
                        pre = preflight_single_leg(token, account_id, body)
                        # Short, humanized preflight summary
                        parts = []
                        if pre.get("estimatedCost") is not None:
                            parts.append(f"Estimated cost: {pre['estimatedCost']}.")
                        if pre.get("buyingPowerImpact") is not None:
                            parts.append(f"Buying power impact: {pre['buyingPowerImpact']}.")
                        if pre.get("warnings"):
                            parts.append("Warnings: " + "; ".join(map(str, pre["warnings"])))
                        add_assistant_text(("Preflight OK. " + " ".join(parts)).strip())
                        add_assistant_json(pre)

                        # Stash pending order; ask for chat confirmation
                        st.session_state.pending_order = {
                            "order_id": order_id,
                            "body": body,
                            "summary": human_summary,
                            "symbol": sym,
                        }
                        add_assistant_text("Type **confirm** to place this order, or **cancel** to abort.")
                    except requests.HTTPError as e:
                        add_assistant_text("Preflight failed.")
                        with st.chat_message("assistant"):
                            st.code(e.response.text)
