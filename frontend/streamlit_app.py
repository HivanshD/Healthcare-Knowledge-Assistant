import os
import requests
import streamlit as st

API_BASE = os.getenv("UI_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="MedAgent AI", layout="wide")

# --------- Styles ---------
def inject_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_css()

# --------- Header ---------
st.markdown(
    '''
    <div class="app-header">
        <div class="brand">
            <div class="brand-title">Healthcare Knowledge Assistant</div>
            <div class="brand-subtitle">Retrieval-augmented assistant</div>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# --------- Sidebar ---------
with st.sidebar:
    st.markdown("### Settings")
    api_base = st.text_input("API Base URL", value=API_BASE)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Ingest Sample Docs"):
            try:
                r = requests.post(f"{api_base}/api/ingest", timeout=90)
                if r.ok:
                    st.success(f"Indexed {r.json().get('files_indexed')} file(s).")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))
    with col_b:
        if st.button("Health Check"):
            try:
                r = requests.get(f"{api_base}/health", timeout=15)
                st.info(r.json())
            except Exception as e:
                st.error(str(e))

# --------- Main ---------
left, right = st.columns([1.1, 1.4], gap="large")

with left:
    st.markdown("#### Query")
    q = st.text_area("", placeholder="Ask a question grounded in your ingested documents...", height=130)
    topk = st.slider("Context chunks", min_value=1, max_value=10, value=5)
    run = st.button("Submit", type="primary")
    clear = st.button("Clear")

    if clear:
        st.session_state.pop("last_result", None)
        st.experimental_rerun()

with right:
    st.markdown("#### Result")
    result_container = st.container()

if run:
    if not q or not q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing"):
            try:
                r = requests.post(f"{api_base}/api/query", json={"query": q.strip(), "top_k": int(topk)}, timeout=120)
                if r.ok:
                    st.session_state["last_result"] = r.json()
                else:
                    st.session_state["last_result"] = {"error": r.text}
            except Exception as e:
                st.session_state["last_result"] = {"error": str(e)}

# --------- Render ---------
with result_container:
    data = st.session_state.get("last_result")
    if not data:
        st.markdown('<div class="placeholder">No results yet. Submit a query to begin.</div>', unsafe_allow_html=True)
    elif "error" in data:
        st.error(data["error"])
    else:
        st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)
        st.markdown("##### Sources")
        if data.get("sources"):
            for s in data["sources"]:
                st.markdown(
                    f'<div class="source-item"><span class="src-name">{s.get("source","unknown")}</span>'
                    f'<span class="src-id">{s.get("chunk_id","")}</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No sources returned.")
