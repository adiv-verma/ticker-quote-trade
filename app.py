# app.py
# Streamlit Trading Assistant (Quotes + Orders) with Live Visuals
# ---------------------------------------------------------------
# To run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Expected secrets in .streamlit/secrets.toml or Streamlit Cloud:
# [api]
# PUBLIC_API_SECRET = "..."
# OPENAI_API_KEY    = "..."
# API_BASE          = "https://api.public.com"   # optional, defaults below
# ACCOUNT_ID        = "..."                      # optional; if omitted, fetched via API
#
# [ui]
# REFRESH_MS = 4000                              # optional, live refresh interval for quotes
# MODEL      = "gpt-4o-mini"                     # optional

import time, uuid, requests, re, json
from typing import Optional, Dict, Any, List
import streamlit as st
from datetime import datetime
from openai import OpenAI

# ---- Config / Secrets ----
API = st.secrets.get("api", {}).get("API_BASE", "https://api.public.com")
PUBLIC_API_SECRET = st.secrets.get("api", {}).get("PUBLIC_API_SECRET", "")
OPENAI_API_KEY    = st.secrets.get("api", {}).get("OPENAI_API_KEY", "")
ACCOUNT_ID        = st.secrets.get("api", {}).get("ACCOUNT_ID")  # optional

MODEL = st.secrets.get("ui", {}).get("MODEL", "gpt-4o-mini")
REFRESH_MS = int(st.secrets.get("ui", {}).get("REFRESH_MS", 4000))

st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
)

# ---- Styles (subtle polish) ----
st.markdown("""
<style>
/* Card look for blocks */
.block { background: #0e1117; border: 1px solid #262730; border-radius: 14px; padding: 16px; }
.small { font-size: 0.90rem; color: #a0a0a0; }
/* Center metrics a bit nicer on wide screens */
[data-testid="stMetricValue"] { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---- OpenAI client ----
oclient = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---- Public API helpers ----
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
    s = symbol_or_name.strip()
    if re.fullmatch(r"[A-Za-z\.-]{1,8}", s) and s.upper() == s:
        return s.upper()
    for it in list_instruments(token):
        if it.get("type") != "EQUITY": continue
        if it.get("symbol","").upper() == s.upper(): return it["symbol"]
        name = (it.get("name") or "").lower()
        if s.lower() in name: return it["symbol"]
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

# ---- LLM Prompts ----
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

def phrase_preflight(symbol: str, pre: dict) -> str:
    est = pre if isinstance(pre, dict) else {}
    parts = []
    if "estimatedCost" in est and est["estimatedCost"] is not None:
        parts.append(f"Estimated cost: {est['estimatedCost']}.")
    if "buyingPowerImpact" in est and est["buyingPowerImpact"] is not None:
        parts.append(f"Buying power impact: {est['buyingPowerImpact']}.")
    if est.get("warnings"):
        parts.append("Warnings: " + "; ".join([str(w) for w in est["warnings"]]))
    msg = f"Preflight for {symbol}: " + (" ".join(parts) or "no details returned.")
    return llm_phrase(msg)

def phrase_final(status: Optional[dict], order_id: str) -> str:
    if not status:
        return llm_phrase(f"Order {order_id} was submitted. Current status isn’t available yet.")
    stt = status.get("status")
    rr = status.get("rejectReason")
    execs = status.get("executions") or []
    base = f"Order {order_id} status: {stt or 'UNKNOWN'}."
    if rr: base += f" Reject reason: {rr}."
    if execs:
        fills = "; ".join([f"{e.get('quantity')} @ {e.get('price')} ({e.get('timestamp')})" for e in execs])
        base += f" Executions: {fills}."
    return llm_phrase(base)

# ---- Session State ----
if "token" not in st.session_state: st.session_state.token = None
if "account_id" not in st.session_state: st.session_state.account_id = ACCOUNT_ID
if "quote_history" not in st.session_state: st.session_state.quote_history = {}  # {SYM: [(ts, price), ...]}
if "last_order_id" not in st.session_state: st.session_state.last_order_id = None
if "dialog" not in st.session_state: st.session_state.dialog = []

# ---- Auth Helper ----
def ensure_auth() -> Optional[str]:
    if st.session_state.token: return st.session_state.token
    if not PUBLIC_API_SECRET:
        st.error("Missing PUBLIC_API_SECRET in secrets. Add it under [api].")
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
    if not token: return None
    try:
        st.session_state.account_id = get_brokerage_account_id(token)
        return st.session_state.account_id
    except Exception as e:
        st.error(f"Failed to fetch brokerage account id: {e}")
        return None

# ---- Header ----
st.title("📈 Trading Assistant")
st.caption("Quotes • Natural-language trading • Preflight • Order tracking")

# ---- Sidebar: Connection / Controls ----
with st.sidebar:
    st.subheader("Connection")
    ok_openai = "✅" if OPENAI_API_KEY else "⚠️"
    ok_api = "✅" if PUBLIC_API_SECRET else "❌"
    st.text(f"OpenAI key: {ok_openai}")
    st.text(f"Public API secret: {ok_api}")

    if st.button("Authenticate / Refresh Token", use_container_width=True):
        st.session_state.token = None
        token = ensure_auth()
        if token:
            st.success("Authenticated")
    if st.button("Resolve Brokerage Account", use_container_width=True):
        acct = ensure_account_id()
        if acct:
            st.success(f"Account: {acct}")

    st.divider()
    st.subheader("Live Refresh")
    st.write("Auto-refresh will keep the Quotes tab live.")
    st.write(f"Interval: {REFRESH_MS} ms")

# ---- Tabs ----
tab_quotes, tab_trade, tab_orders, tab_settings = st.tabs(
    ["📊 Quotes", "📝 Trade (Natural-language)", "📬 Orders", "⚙️ Settings"]
)

# ===================== Quotes Tab =====================
with tab_quotes:
    st.markdown("### Real-time Quotes")
    colq1, colq2 = st.columns([2, 1])
    with colq1:
        q_input = st.text_input("Symbol or Company", placeholder="e.g., NVDA or Nvidia", key="q_input")
    with colq2:
        live = st.toggle("Live auto-refresh", value=True)

    token = ensure_auth()
    account_id = ensure_account_id()

    # Auto-refresh if live
    if live:
        st.autorefresh(interval=REFRESH_MS, key="autorefresh_quotes")

    if st.button("Get Quote", type="primary"):
        if not (token and account_id and q_input):
            st.warning("Enter a symbol or company and ensure you’re authenticated.")
        else:
            sym = resolve_symbol(token, q_input)
            if not sym:
                st.error(f"Could not resolve '{q_input}' to a tradable symbol.")
            else:
                try:
                    quote = get_equity_quote(token, account_id, sym)
                    if not quote:
                        st.info(f"No quote returned for {sym}.")
                    else:
                        last = quote.get("last") or quote.get("lastPrice") or quote.get("price") or quote.get("closePrice")
                        bid, ask = quote.get("bid"), quote.get("ask")
                        vol = quote.get("volume")
                        ts = datetime.utcnow().isoformat()

                        # update rolling per-session history
                        hist = st.session_state.quote_history.setdefault(sym, [])
                        if isinstance(last, (int, float)):
                            hist.append((ts, float(last)))
                            st.session_state.quote_history[sym] = hist[-300:]  # keep last 300 prints

                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"{sym} Last", f"{last}")
                        c2.metric("Bid / Ask", f"{bid} / {ask}")
                        c3.metric("Volume", f"{vol}")

                        # Sparkline of session prices
                        import pandas as pd
                        import plotly.express as px
                        if hist:
                            df = pd.DataFrame(hist, columns=["ts", "price"])
                            fig = px.line(df, x="ts", y="price", title=f"{sym} (session sparkline)")
                            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=260)
                            st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Raw quote JSON"):
                            st.json(quote)
                except Exception as e:
                    st.error(f"Quote lookup failed: {e}")

# ===================== Trade Tab =====================
with tab_trade:
    st.markdown("### Natural-language Trading")
    st.caption("Describe what you want to do (e.g., “Buy 10 shares of Nvidia at a limit of 120 GTC”).")

    nl = st.text_area("Your instruction", height=100, placeholder="e.g., 'Buy $500 of AAPL as a market order, DAY.'")
    colA, colB = st.columns([1,1])
    with colA:
        parse_btn = st.button("Parse Intent", type="primary")
    with colB:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.session_state.dialog = []

    if parse_btn:
        if not nl.strip():
            st.warning("Enter an instruction first.")
        elif not oclient:
            st.error("OpenAI API key missing. Add it to secrets.")
        else:
            st.session_state.dialog.append({"role": "user", "content": nl})
            parsed = llm_interpret(st.session_state.dialog)

            st.markdown("#### Interpretation")
            st.json(parsed)

            ptype = parsed.get("type")
            if ptype == "QUOTE":
                # Ask LLM for the target security string
                if not oclient:
                    st.info("OpenAI key missing for symbol extraction.")
                else:
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
                    if not qstr:
                        st.warning("Please specify a symbol or company.")
                    else:
                        token = ensure_auth(); account_id = ensure_account_id()
                        if token and account_id:
                            sym = resolve_symbol(token, qstr)
                            if not sym:
                                st.error(f"Could not find a tradable symbol for “{qstr}”. Try the exact ticker.")
                            else:
                                quote = get_equity_quote(token, account_id, sym)
                                if not quote:
                                    st.info(f"No quote for {sym}.")
                                else:
                                    last = quote.get("last") or quote.get("lastPrice") or quote.get("price") or quote.get("closePrice")
                                    bid, ask = quote.get("bid"), quote.get("ask")
                                    vol = quote.get("volume")
                                    st.success(llm_phrase(f"{sym} is around {last}. Bid {bid}, ask {ask}. Volume {vol}."))

            elif ptype == "ASK":
                st.info(parsed.get("question") or "Need more details.")
            else:
                # ORDER flow
                intent = parsed.get("intent") or {}
                missing = parsed.get("missing") or []
                if missing:
                    st.warning(f"Missing: {', '.join(missing)}")
                else:
                    token = ensure_auth(); account_id = ensure_account_id()
                    if not (token and account_id):
                        st.stop()

                    symbol_input = intent.get("symbol")
                    sym = resolve_symbol(token, symbol_input) if symbol_input else None
                    if not sym:
                        st.error(f"Couldn’t resolve a ticker for “{symbol_input}”. Provide the symbol (e.g., NVDA).")
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
                        st.markdown(f"##### Review")
                        st.info(llm_phrase("Please review: " + human_summary))

                        # Preflight
                        with st.spinner("Preflighting..."):
                            try:
                                pre = preflight_single_leg(token, account_id, body)
                                st.success(phrase_preflight(sym, pre))
                                with st.expander("Preflight JSON"):
                                    st.json(pre)
                            except requests.HTTPError as e:
                                st.error("Preflight failed.")
                                st.code(e.response.text)
                                st.stop()

                        # Place order
                        if st.button(f"🚀 Place Order {order_id}", type="primary"):
                            try:
                                _ = place_order(token, account_id, body)
                                st.session_state.last_order_id = order_id
                                st.success(f"Order submitted: {order_id}")
                            except requests.HTTPError as e:
                                st.error("Order placement failed.")
                                st.code(e.response.text)

# ===================== Orders Tab =====================
with tab_orders:
    st.markdown("### Track Orders")
    token = ensure_auth(); account_id = ensure_account_id()

    default_oid = st.session_state.last_order_id or ""
    order_id_input = st.text_input("Order ID", value=default_oid, placeholder="UUID from placement response")

    col1, col2 = st.columns([1,1])
    live_track = col1.toggle("Live status (auto-refresh)", value=True, key="orders_live")
    if live_track:
        st.autorefresh(interval=REFRESH_MS, key="autorefresh_orders")

    if col2.button("Fetch Status", type="primary"):
        if not (token and account_id and order_id_input):
            st.warning("Enter an order ID and ensure you’re authenticated.")
        else:
            try:
                status = get_order(token, account_id, order_id_input)
                if not status:
                    st.info("No status available yet.")
                else:
                    st.markdown("#### Current Status")
                    s = status.get("status", "UNKNOWN")
                    color = "green" if s == "FILLED" else ("red" if s in {"REJECTED","CANCELLED","EXPIRED"} else "orange")
                    st.markdown(f"<div class='block'><span class='small'>Order</span><h3 style='margin:4px 0'>{order_id_input}</h3><b style='color:{color}'>{s}</b></div>", unsafe_allow_html=True)
                    st.write(phrase_final(status, order_id_input))
                    with st.expander("Status JSON"):
                        st.json(status)
            except Exception as e:
                st.error(f"Status fetch failed: {e}")

# ===================== Settings Tab =====================
with tab_settings:
    st.markdown("### Settings & Diagnostics")
    st.write("Secrets loaded (keys only, no values):")
    visible = {
        "api_keys_present": {
            "PUBLIC_API_SECRET": bool(PUBLIC_API_SECRET),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        },
        "api_base": API,
        "account_id": st.session_state.account_id or "(not set)",
        "model": MODEL,
        "refresh_ms": REFRESH_MS,
    }
    st.json(visible)

    st.divider()
    st.markdown("#### Troubleshooting")
    st.write("- If quotes don’t update, re-auth in the sidebar.")
    st.write("- If symbol resolution fails, try the exact ticker (e.g., AAPL).")
    st.write("- For orders, ensure required fields: **quantity or amount**, plus **limit/stop** where applicable.")
