import streamlit as st
import requests, uuid, re, json, time
from typing import Optional, Dict, Any, List
from openai import OpenAI

# ---------- Secrets (supports [api] or top-level) ----------
_api = st.secrets.get("api", {})
PUBLIC_API_SECRET = _api.get("PUBLIC_API_SECRET", st.secrets.get("PUBLIC_API_SECRET", ""))
OPENAI_API_KEY    = _api.get("OPENAI_API_KEY",    st.secrets.get("OPENAI_API_KEY", ""))
API               = _api.get("API_BASE",          st.secrets.get("API_BASE", "https://api.public.com"))
ACCOUNT_ID        = _api.get("ACCOUNT_ID",        st.secrets.get("ACCOUNT_ID", None))
MODEL             = "gpt-4o-mini"

st.set_page_config(page_title="Trading Assistant", page_icon="📈", layout="wide")
st.title("📈 Trading Assistant")
st.caption("Quotes and natural-language trading")

# ---------- Clients ----------
oclient = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- API helpers ----------
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

def list_instruments(token: str, page_size: int = 200) -> List[dict]:
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

# ---------- LLM helpers ----------
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
}"""

def llm_interpret(dialog: list[Dict[str, str]]) -> Dict[str, Any]:
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
        messages=[{"role":"system","content":"Rewrite briefly and clearly for an end user."},
                  {"role":"user","content":text}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- Session state ----------
if "token" not in st.session_state: st.session_state.token = None
if "account_id" not in st.session_state: st.session_state.account_id = ACCOUNT_ID
if "dialog" not in st.session_state: st.session_state.dialog = []

def ensure_auth() -> Optional[str]:
    if st.session_state.token:
        return st.session_state.token
    st.session_state.token = get_access_token(PUBLIC_API_SECRET)
    return st.session_state.token

def ensure_account_id() -> Optional[str]:
    if st.session_state.account_id:
        return st.session_state.account_id
    token = ensure_auth()
    st.session_state.account_id = get_brokerage_account_id(token)
    return st.session_state.account_id

# ---------- Layout ----------
col_left, col_right = st.columns(2)

# ===== Quotes =====
with col_left:
    st.subheader("🔎 Quote")
    q = st.text_input("Symbol or Company", placeholder="e.g., NVDA")
    if st.button("Get Quote"):
        token = ensure_auth(); account_id = ensure_account_id()
        sym = resolve_symbol(token, q)
        if not sym:
            st.warning(f"Could not resolve “{q}”.")
        else:
            qt = get_equity_quote(token, account_id, sym)
            if not qt:
                st.info(f"No quote returned for {sym}.")
            else:
                last = qt.get("last") or qt.get("lastPrice") or qt.get("price")
                bid, ask = qt.get("bid"), qt.get("ask")
                vol = qt.get("volume")
                st.success(llm_phrase(f"{sym} is around {last}. Bid {bid}, ask {ask}. Volume {vol}."))
                with st.expander("Raw quote JSON"):
                    st.json(qt)

# ===== Trade =====
with col_right:
    st.subheader("📝 Trade (Natural Language)")
    ins = st.text_area("Instruction", placeholder="e.g., Buy 5 shares of NVDA at 120 limit GTC")
    if st.button("Parse / Preflight"):
        if not ins.strip():
            st.warning("Enter an instruction.")
        else:
            st.session_state.dialog.append({"role":"user","content":ins})
            parsed = llm_interpret(st.session_state.dialog)
            st.write("Interpretation:"); st.json(parsed)

            if parsed["type"] == "ASK":
                st.info(parsed.get("question"))
            elif parsed["type"] == "QUOTE":
                st.info("Looks like a quote request — try the left panel.")
            else:  # ORDER
                intent = parsed["intent"]
                sym = resolve_symbol(ensure_auth(), intent.get("symbol"))
                if not sym:
                    st.warning("Ticker missing or not resolved.")
                else:
                    order_id = str(uuid.uuid4())
                    body = {
                        "orderId": order_id,
                        "instrument": {"symbol": sym, "type": "EQUITY"},
                        "orderSide": intent["side"],
                        "orderType": intent["orderType"],
                        "expiration": {"timeInForce": intent.get("tif") or "DAY"},
                    }
                    if intent.get("quantity"): body["quantity"] = str(intent["quantity"])
                    if intent.get("amount"):   body["amount"] = str(intent["amount"])
                    if intent.get("limitPrice"): body["limitPrice"] = str(intent["limitPrice"])
                    if intent.get("stopPrice"):  body["stopPrice"] = str(intent["stopPrice"])

                    st.info("Review: " + llm_phrase(parsed["summary"]))

                    # Preflight
                    try:
                        pre = preflight_single_leg(st.session_state.token, st.session_state.account_id, body)
                        st.success("Preflight OK.")
                        with st.expander("Preflight JSON"): st.json(pre)
                    except Exception as e:
                        st.error(f"Preflight failed: {e}")
                        st.stop()

                    if st.button("🚀 Place Order"):
                        try:
                            place_order(st.session_state.token, st.session_state.account_id, body)
                            st.success(f"Order submitted: {order_id}")
                            with st.spinner("Checking status..."):
                                for _ in range(8):
                                    s = get_order(st.session_state.token, st.session_state.account_id, order_id)
                                    if s and s.get("status") in {"FILLED","REJECTED","CANCELLED"}:
                                        st.write("Final status:", s)
                                        break
                                    time.sleep(1)
                        except Exception as e:
                            st.error(f"Order failed: {e}")
