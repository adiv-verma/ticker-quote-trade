# app.py — Simple Streamlit Trading Assistant (Quotes + Orders)
# ------------------------------------------------------------
# Run:  streamlit run app.py
#
# Secrets (works with either nested or top-level):
# .streamlit/secrets.toml
#
# [api]
# PUBLIC_API_SECRET = "..."
# OPENAI_API_KEY    = "..."
# API_BASE          = "https://api.public.com"   # optional
# ACCOUNT_ID        = "..."                      # optional
#
# # or top-level:
# PUBLIC_API_SECRET = "..."
# OPENAI_API_KEY    = "..."
# API_BASE          = "https://api.public.com"
# ACCOUNT_ID        = "..."

import streamlit as st
import requests, uuid, re, json, time
from typing import Optional, Dict, Any, List
from openai import OpenAI

# ---------- Config / Secrets (supports [api] or top-level) ----------
_api = st.secrets.get("api", {})
PUBLIC_API_SECRET = _api.get("PUBLIC_API_SECRET", st.secrets.get("PUBLIC_API_SECRET", ""))
OPENAI_API_KEY    = _api.get("OPENAI_API_KEY",    st.secrets.get("OPENAI_API_KEY", ""))
API               = _api.get("API_BASE",          st.secrets.get("API_BASE", "https://api.public.com"))
ACCOUNT_ID        = _api.get("ACCOUNT_ID",        st.secrets.get("ACCOUNT_ID", None))
MODEL             = "gpt-4o-mini"

st.set_page_config(page_title="Trading Assistant", page_icon="📈", layout="wide")
st.title("📈 Simple Trading Assistant")
st.caption("Quotes and natural-language orders. Minimal UI, close to your original code.")

# ---------- Clients ----------
oclient = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- API helpers (same behavior as your original) ----------
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

def list_instruments(token: str, page_size: int = 500) -> List[dict]:
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
    """Simple resolution: accept exact ticker in ALL CAPS; else best-effort by name."""
    s = (symbol_or_name or "").strip()
    if not s:
        return None
    if re.fullmatch(r"[A-Za-z\.-]{1,8}", s) and s.upper() == s:
        return s.upper()
    for it in list_instruments(token):
        if it.get("type") != "EQUITY": 
            continue
        if it.get("symbol","").upper() == s.upper(): 
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

# ---------- LLM helpers (same logic, minimal) ----------
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
- For orders, quantity OR amount is required.
- LIMIT/STOP_LIMIT require limitPrice; STOP/STOP_LIMIT require stopPrice.
- If mandatory fields are missing, set type="ASK", add a single best 'question', and list 'missing'.
- JSON only; no commentary.
"""

def llm_interpret(dialog: list[Dict[str, str]]) -> Dict[str, Any]:
    if not oclient:
        return {"type": "ASK", "question": "OpenAI key missing.", "missing": ["OPENAI_API_KEY"], "intent": None, "summary": None}
    resp = oclient.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content": SYSTEM}] + dialog,
        temperature=0.1,
    )
    return json.loads(resp.choices[0].message.content)

def llm_phrase(text: str) -> str:
    if not oclient:
        return text
    resp = oclient.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":"Rewrite briefly and clearly for an end user."},
                  {"role":"user","content":text}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- Session state ----------
if "token" not in st.session_state: st.session_state.token = None
if "account_id" not in st.session_state: st.session_state.account_id = ACCOUNT_ID
if "dialog" not in st.session_state: st.session_state.dialog = []
if "last_order_id" not in st.session_state: st.session_state.last_order_id = None

# ---------- Helpers ----------
def ensure_auth() -> Optional[str]:
    if st.session_state.token:
        return st.session_state.token
    if not PUBLIC_API_SECRET:
        st.error("Missing PUBLIC_API_SECRET in secrets (either top-level or under [api]).")
        return None
    try:
        st.session_state.token = get_access_token(PUBLIC_API_SECRET)
        return st.session_state.token
    except requests.HTTPError as e:
        st.error(f"Auth failed: {e}")
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
    except Exception as e:
        st.error(f"Failed to fetch brokerage account id: {e}")
        return None

# ---------- Layout: two simple columns ----------
col_left, col_right = st.columns(2)

# ===== Quotes =====
with col_left:
    st.subheader("🔎 Quote")
    q = st.text_input("Symbol or Company", placeholder="e.g., NVDA or Nvidia")
    if st.button("Get Quote"):
        token = ensure_auth()
        account_id = ensure_account_id()
        if token and account_id and q:
            sym = resolve_symbol(token, q)
            if not sym:
                st.warning(f"Could not resolve “{q}” to a tradable symbol.")
            else:
                try:
                    qt = get_equity_quote(token, account_id, sym)
                    if not qt:
                        st.info(f"No quote returned for {sym}.")
                    else:
                        last = qt.get("last") or qt.get("lastPrice") or qt.get("price") or qt.get("closePrice")
                        bid, ask = qt.get("bid"), qt.get("ask")
                        vol = qt.get("volume")
                        st.success(llm_phrase(f"{sym} is around {last}. Bid {bid}, ask {ask}. Volume {vol}."))
                        with st.expander("Raw quote JSON"):
                            st.json(qt)
                except Exception as e:
                    st.error(f"Quote lookup failed: {e}")
        else:
            st.warning("Enter a symbol/company and check secrets.")

# ===== Trade (NL) =====
with col_right:
    st.subheader("📝 Trade (Natural-language)")
    ins = st.text_area("Instruction", placeholder="e.g., Buy 10 AAPL @ 190 limit, GTC", height=110)
    parse = st.button("Parse & Preflight")
    if parse:
        if not ins.strip():
            st.warning("Enter an instruction.")
        else:
            st.session_state.dialog.append({"role":"user","content":ins})
            parsed = llm_interpret(st.session_state.dialog)
            st.write("Interpretation:")
            st.json(parsed)

            ptype = parsed.get("type")
            if ptype == "QUOTE":
                if not oclient:
                    st.info("OpenAI key missing for symbol extraction.")
                else:
                    # simple symbol extraction
                    ask_symbol = oclient.chat.completions.create(
                        model=MODEL,
                        response_format={"type":"json_object"},
                        messages=[
                            {"role":"system","content":"Extract the target ticker or company name as JSON: {\"query\": string}. If none, put null."},
                            *st.session_state.dialog
                        ],
                        temperature=0.0,
                    )
                    qstr = json.loads(ask_symbol.choices[0].message.content).get("query")
                    token = ensure_auth(); account_id = ensure_account_id()
                    if token and account_id and qstr:
                        sym = resolve_symbol(token, qstr)
                        if not sym:
                            st.warning(f"Could not resolve “{qstr}”. Try the exact ticker.")
                        else:
                            qt = get_equity_quote(token, account_id, sym)
                            if qt:
                                last = qt.get("last") or qt.get("lastPrice") or qt.get("price") or qt.get("closePrice")
                                bid, ask = qt.get("bid"), qt.get("ask")
                                vol = qt.get("volume")
                                st.success(llm_phrase(f"{sym} is around {last}. Bid {bid}, ask {ask}. Volume {vol}."))
                            else:
                                st.info(f"No quote for {sym}.")
                    else:
                        st.warning("Missing auth or symbol.")
            elif ptype == "ASK":
                st.info(parsed.get("question") or "Need more details.")
            else:
                # ORDER path
                intent = parsed.get("intent") or {}
                missing = parsed.get("missing") or []
                if missing:
                    st.warning("Missing: " + ", ".join(missing))
                else:
                    token = ensure_auth(); account_id = ensure_account_id()
                    if not (token and account_id):
                        st.stop()
                    symbol_input = intent.get("symbol")
                    sym = resolve_symbol(token, symbol_input) if symbol_input else None
                    if not sym:
                        st.warning(f"Couldn’t resolve a ticker for “{symbol_input}”. Provide the symbol (e.g., NVDA).")
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

                        human = parsed.get("summary") or f"{side} {qty or ('$'+str(amt))} of {sym} as {otype} ({tif})" + (f" @ {lpx}" if lpx else "") + (f" stop {spx}" if spx else "")
                        st.info("Review: " + llm_phrase(human))

                        # Preflight
                        try:
                            pre = preflight_single_leg(token, account_id, body)
                            # Simple human summary
                            parts = []
                            if pre.get("estimatedCost") is not None:
                                parts.append(f"Estimated cost: {pre['estimatedCost']}.")
                            if pre.get("buyingPowerImpact") is not None:
                                parts.append(f"Buying power impact: {pre['buyingPowerImpact']}.")
                            if pre.get("warnings"):
                                parts.append("Warnings: " + "; ".join(map(str, pre["warnings"])))
                            st.success("Preflight OK. " + ( " ".join(parts) or "" ))
                            with st.expander("Preflight JSON"):
                                st.json(pre)
                        except requests.HTTPError as e:
                            st.error("Preflight failed.")
                            st.code(e.response.text)
                            st.stop()

                        # Place
                        if st.button(f"Place Order {order_id}"):
                            try:
                                _ = place_order(token, account_id, body)
                                st.session_state.last_order_id = order_id
                                st.success(f"Order submitted: {order_id}")
                            except requests.HTTPError as e:
                                st.error("Order placement failed.")
                                st.code(e.response.text)

# ----- Simple diagnostics -----
st.divider()
st.caption("Diagnostics")
st.write({
    "PUBLIC_API_SECRET?": bool(PUBLIC_API_SECRET),
    "OPENAI_API_KEY?": bool(OPENAI_API_KEY),
    "API": API,
    "ACCOUNT_ID (current)": st.session_state.account_id or "(not set)",
})
