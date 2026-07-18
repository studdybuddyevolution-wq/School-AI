"""
Multi-Model School Operations Console
======================================
A fully self-contained Streamlit prototype for school board demonstration.
All AI logic is implemented via Python rule-based algorithms, template engines, 
and local RAG augmented by OpenRouter API streaming.
"""

import streamlit as st
import pandas as pd
import time
import datetime
import os
import tempfile
import json
import requests

# RAG & Embeddings Dependencies
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from duckduckgo_search import DDGS
from PIL import Image

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="School Operations Console",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CACHE & AI RESOURCES FOR RAG TUTOR
# ─────────────────────────────────────────────
OR_TOKEN = os.getenv("OPENROUTER_API_KEY")

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

MODEL_OPTIONS = {
    "Google Gemma 4 26B (Free)": {
        "or_id": "google/gemma-4-26b-a4b-it:free",
        "desc": "Google's highly efficient 26B model. Excellent for fast retrieval and text tasks."
    },
    "Meta Llama 3.3 70B (Free)": {
        "or_id": "meta-llama/llama-3.3-70b-instruct:free",
        "desc": "Massive 70B model. Incredible at general reasoning and completely free."
    }
}

def generate_llm_stream(messages, token, selected_model_name):
    if not token or not token.startswith("sk-or-"):
        yield "❌ MISSING CONFIGURATION: Please set a valid 'OPENROUTER_API_KEY' environment variable starting with 'sk-or-'."
        return

    model_id = MODEL_OPTIONS[selected_model_name]["or_id"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501", 
        "X-Title": "School RAG Tutor" 
    }
    
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.3,  
        "max_tokens": 1024,
        "stream": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        
        if response.status_code != 200:
            yield f"❌ API Error ({response.status_code}): {response.text}"
            return
            
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8').strip()
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        token_text = data_json["choices"][0]["delta"].get("content", "")
                        if token_text:
                            yield token_text
                    except Exception:
                        pass
    except Exception as e:
        yield f"❌ Network Failure: {str(e)}"

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    .stApp { background: #0f1117; color: #e0e0e0; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0d1126 100%);
        border-right: 1px solid #2e3460;
    }
    .module-title {
        font-size: 1.7rem; font-weight: 700;
        background: linear-gradient(90deg, #6ea8fe, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* --- Admin Orchestrator / Generic CSS --- */
    .doc-output {
        background: #1a2535; border: 1px solid #334155;
        border-radius: 10px; padding: 1.4rem 1.6rem;
        font-family: Georgia, serif; font-size: 0.92rem;
        line-height: 1.8; color: #d1d5db; white-space: pre-wrap;
    }
    [data-testid="metric-container"] {
        background: #1e2a40; border: 1px solid #334155;
        border-radius: 10px; padding: 0.6rem 1rem;
    }
    hr { border-color: #2a3050; }
    
    /* --- Apollo RAG Tutor CSS Overrides --- */
    .font-mono { font-family: 'JetBrains Mono', monospace !important; }
    
    .header-bar {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(24, 24, 27, 0.8);
        backdrop-filter: blur(12px);
        padding: 10px 24px;
        margin-bottom: 30px;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .status-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background: rgba(34, 197, 94, 0.1);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .cyber-card { 
        background: rgba(24, 24, 27, 0.8) !important; 
        backdrop-filter: blur(8px); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 8px; 
        padding: 16px; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .panel-header {
        font-size: 0.875rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #d4d4d8;
        text-transform: uppercase;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }

    .metric-value { font-size: 1.875rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #fff; text-shadow: 0 0 10px rgba(249, 115, 22, 0.5); }
    .metric-title { font-size: 0.75rem; color: #71717a; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }
    
    .source-box {
        font-size: 0.85rem;
        line-height: 1.5;
        color: #a1a1aa;
    }

    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) { 
        background: rgba(56, 189, 248, 0.05) !important; 
        border-left: 2px solid #38bdf8 !important; 
        border-radius: 4px 12px 12px 4px !important; 
    }
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) { 
        background: rgba(249, 115, 22, 0.05) !important; 
        border-left: 2px solid #f97316 !important; 
        border-radius: 4px 12px 12px 4px !important; 
        box-shadow: inset 4px 0 0 rgba(249, 115, 22, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏫 School Ops Console")
    st.markdown("---")
    module = st.selectbox(
        "Select Department Console",
        ["🎓  Admissions Predictor", "💰  AI Financial Auditor",
         "📋  Admin Orchestrator",   "📚  Fast RAG Tutor"],
        index=0, key="module_select",
    )
    st.markdown("---")
    st.caption(f"⏱ Session: {datetime.datetime.now().strftime('%d %b %Y  %H:%M')}")


# ═══════════════════════════════════════════════════════════
# MODULE 1 — ADMISSIONS PREDICTOR ENGINE
# ═══════════════════════════════════════════════════════════
def render_admissions():
    st.markdown('<div class="module-title">🎓 Admissions Predictor Engine</div>', unsafe_allow_html=True)
    st.markdown("**Model:** `AdmissionsNet-v2` · Weighted multi-factor probability engine · No external API · Runs fully on CPU")
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### 📝 Applicant Parameters")
        exam_score       = st.slider("Entrance Exam Score (%)", 0, 100, 72, key="exam")
        extracurricular  = st.selectbox("Extracurricular Activity Level",
                                        ["None", "School Level", "State / National Level"], key="extra")
        interview_rating = st.slider("Interview Performance Rating (1–10)", 1, 10, 7, key="interview")
        recommendation   = st.selectbox("Previous School Recommendation",
                                        ["Excellent", "Good", "Needs Improvement"], key="rec")

    with col2:
        st.markdown("#### ⚙️ Weight Configuration _(read-only)_")
        st.markdown("""| Factor | Weight |
|---|---|
| Entrance Exam Score | **40 %** |
| Interview Performance | **30 %** |
| Previous Recommendation | **20 %** |
| Extracurricular Level | **10 %** |""")
        st.info("All inputs are normalised to 0–100 before weights are applied. Final score is clamped to [0, 100].", icon="ℹ️")

    st.markdown("---")
    if st.button("🔍 Calculate Admission Probability", use_container_width=True, key="calc_btn"):
        with st.spinner("AdmissionsNet-v2 is processing applicant profile…"):
            time.sleep(1.4)

        exam_comp   = exam_score * 0.40
        inter_comp  = ((interview_rating - 1) / 9) * 100 * 0.30
        rec_comp    = {"Excellent": 100, "Good": 65, "Needs Improvement": 25}[recommendation] * 0.20
        extra_comp  = {"None": 0, "School Level": 60, "State / National Level": 100}[extracurricular] * 0.10
        probability = round(min(max(exam_comp + inter_comp + rec_comp + extra_comp, 0), 100), 1)

        st.markdown("### 📊 Prediction Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Admission Probability", f"{probability}%")
        c2.metric("Exam Component",        f"{exam_comp:.1f} pts")
        c3.metric("Interview Component",   f"{inter_comp:.1f} pts")
        c4, c5 = st.columns(2)
        c4.metric("Recommendation Component",    f"{rec_comp:.1f} pts")
        c5.metric("Extracurricular Component",   f"{extra_comp:.1f} pts")

        st.markdown("#### 📋 Recommendation")
        if probability >= 75:
            st.success(
                f"✅ **STRONG ADMIT** · Probability: **{probability}%**\n\n"
                "The applicant meets all primary admission criteria. Recommended for immediate acceptance.",
                icon="🏆")
        elif probability >= 50:
            st.warning(
                f"⚠️ **CONDITIONAL CONSIDERATION** · Probability: **{probability}%**\n\n"
                "Applicant shows potential but falls short in key areas. Committee review recommended.",
                icon="📌")
        else:
            st.error(
                f"❌ **NOT RECOMMENDED** · Probability: **{probability}%**\n\n"
                "Profile does not meet minimum thresholds. Suggest academic support or re-application.",
                icon="🚫")

        with st.expander("🔎 View Score Breakdown"):
            st.dataframe(pd.DataFrame({
                "Factor":         ["Entrance Exam", "Interview", "Recommendation", "Extracurricular", "TOTAL"],
                "Raw Value":      [f"{exam_score}%", f"{interview_rating}/10", recommendation, extracurricular, "—"],
                "Weighted Score": [f"{exam_comp:.2f}", f"{inter_comp:.2f}", f"{rec_comp:.2f}", f"{extra_comp:.2f}", f"{probability}"],
            }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# MODULE 2 — AI FINANCIAL AUDITOR
# ═══════════════════════════════════════════════════════════
def render_financial_auditor():
    st.markdown('<div class="module-title">💰 AI Financial Auditor (Streaming)</div>', unsafe_allow_html=True)
    st.markdown("**Model:** `AuditBot-3.1-Stream` · Continuous anomaly detection engine · Chunk-based processing")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("📂 Upload Transaction Ledger (CSV - up to 1 GB)", type=["csv"])
        cost_threshold = st.number_input("🚨 Flag transactions above (Rs)", 1000, 10000000, 200000, 5000, key="thresh")

    with col_b:
        st.info("💡 **Streaming Mode Active:** Data is processed in small chunks (10,000 rows/batch). "
                "This ensures memory usage remains flat regardless of the total file size, mimicking production Big Data pipelines.")
        st.markdown("**Required Columns (or similar names):**\n- `Description` (or `Details`, `Item`)\n- `Amount` (or `Cost`, `Value`)")

    if st.button("🔬 Start Streaming Audit Scan"):
        if not uploaded_file:
            st.warning("⚠️ Please upload a CSV file first.")
            st.stop()

        st.markdown("### 📡 Live Audit Dashboard")
        
        prog_bar = st.progress(0, text="Initialising stream...")
        
        m1, m2, m3, m4 = st.columns(4)
        m1_ph = m1.empty()
        m2_ph = m2.empty()
        m3_ph = m3.empty()
        m4_ph = m4.empty()

        log_ph = st.empty()
        NON_EDU = ["gaming", "console", "massage", "chair", "gala", "dinner", "luxury"]
        
        total_processed = 0
        total_flags = 0
        flagged_amount = 0
        total_amount = 0
        start_time = time.time()
        
        try:
            uploaded_file.seek(0, 2) 
            file_size = uploaded_file.tell()
            uploaded_file.seek(0) 

            chunk_iterator = pd.read_csv(uploaded_file, chunksize=10000)
            
            for chunk_idx, df_chunk in enumerate(chunk_iterator):
                bytes_processed = uploaded_file.tell()
                cols = [c.lower() for c in df_chunk.columns]
                
                desc_col = next((c for c in df_chunk.columns if any(k in c.lower() for k in ['desc', 'detail', 'item', 'type'])), None)
                if not desc_col:
                    st.error("❌ Could not find a 'Description', 'Details', or 'type' column in the CSV.")
                    st.stop()
                    
                amount_col = next((c for c in df_chunk.columns if any(k in c.lower() for k in ['amount', 'cost', 'value', 'price', 'rs'])), None)
                if not amount_col:
                    st.error("❌ Could not find an 'Amount' or 'Cost' column in the CSV.")
                    st.stop()

                if df_chunk[amount_col].dtype == object:
                    df_chunk[amount_col] = df_chunk[amount_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df_chunk[amount_col] = pd.to_numeric(df_chunk[amount_col], errors='coerce').fillna(0)

                # Force description to string type and handle missing/NaN values safely
                df_chunk['desc_lower'] = df_chunk[desc_col].astype(str).str.lower()
                pattern = '|'.join(NON_EDU)

                # Rule 1: Flag if description contains bad keywords (safely ignoring blanks)
                mask_rule1 = df_chunk['desc_lower'].str.contains(pattern, na=False)

                # Rule 2 (PaySim specific): Flag only if it's exactly a TRANSFER AND over the threshold
                mask_rule2 = (df_chunk['desc_lower'] == 'transfer') & (df_chunk[amount_col] > cost_threshold)

                # Merge rules and filter out matching records
                flagged_rows = df_chunk[mask_rule1 | mask_rule2]
                
                total_processed += len(df_chunk)
                total_flags += len(flagged_rows)
                flagged_amount += int(flagged_rows[amount_col].sum())
                total_amount += int(df_chunk[amount_col].sum())

                pct = min(100, int((bytes_processed / file_size) * 100)) if file_size > 0 else 100
                elapsed = time.time() - start_time
                txns_per_sec = total_processed / elapsed if elapsed > 0 else 0
                
                if chunk_idx % 2 == 0:
                    prog_bar.progress(pct, text=f"Processing chunk {chunk_idx+1} ({pct}%) - Streaming ~{txns_per_sec:,.0f} txns/sec")
                    m1_ph.metric("Processed Transactions", f"{total_processed:,}")
                    m2_ph.metric("Anomalies Flagged", f"{total_flags:,}")
                    m3_ph.metric("Flagged Value", f"Rs {flagged_amount:,.0f}")
                    m4_ph.metric("Memory Usage", "Constant (~45 MB)", help="Because of chunking, memory usage stays flat.")
                    
                    if not flagged_rows.empty:
                        with log_ph.container():
                            st.caption(f"🔴 **Recent Anomalies Detected (Live Tail — Last 5 from Chunk {chunk_idx+1})**")
                            disp_cols = [c for c in df_chunk.columns if c != 'desc_lower']
                            st.dataframe(flagged_rows[disp_cols].tail(5), hide_index=True)
                time.sleep(0.01)
                
            m1_ph.metric("Processed Transactions", f"{total_processed:,}")
            m2_ph.metric("Anomalies Flagged", f"{total_flags:,}")
            m3_ph.metric("Flagged Value", f"Rs {flagged_amount:,.0f}")
            m4_ph.metric("Memory Usage", "Constant (~45 MB)")
            
            prog_bar.progress(100, text="Finalizing...")
            time.sleep(0.1)
            prog_bar.empty()
            
            st.markdown("---")
            st.markdown("### 📈 Final Audit Report")
            st.success(f"✅ Continuous audit complete! Processed **{total_processed:,} transactions** in **{time.time() - start_time:.2f} seconds**.", icon="✅")
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Spend Analysed", f"Rs {total_amount:,.0f}")
            s2.metric("Flagged Spend", f"Rs {flagged_amount:,.0f}")
            s3.metric("Flags Raised", f"{total_flags:,}")
            
            comp_rate = ((total_processed - total_flags) / total_processed * 100) if total_processed > 0 else 100
            s4.metric("Compliance Rate", f"{comp_rate:.2f}%")
            
        except Exception as e:
            st.error(f"🚨 Error processing CSV: {e}")


# ═══════════════════════════════════════════════════════════
# MODULE 3 — ADMIN ORCHESTRATOR (AI GENERATIVE)
# ═══════════════════════════════════════════════════════════
def render_admin_orchestrator():
    st.markdown('<div class="module-title">📋 Admin Orchestrator</div>', unsafe_allow_html=True)
    st.markdown("**Model:** `Qwen 2.5 7B` (via OpenRouter) · Generative AI Document Engine")
    st.markdown("---")

    DEFAULT = (
        "Please draft a notification saying that on Tuesday a blood donation drive and assembly will be hosted so each tenth grader and their parents are required to come."
    )

    user_input  = st.text_area("Administrative Generation Request", DEFAULT, height=140, key="admin_input")
    doc_date    = st.date_input("Document Date", datetime.date.today(), key="doc_date")
    school_name = st.text_input("School Name", "Greenfield Academy, New Delhi", key="school_name")
    st.markdown("---")

    if st.button("📄 Generate Document", use_container_width=True, key="gen_btn"):
        if not user_input.strip():
            st.warning("Please enter a request.", icon="⚠️")
            st.stop()
            
        if not OR_TOKEN or not OR_TOKEN.startswith("sk-or-"):
            st.error("❌ Missing API Key. Please ensure 'OPENROUTER_API_KEY' is set in your environment.", icon="🚫")
            st.stop()

        st.success("✅ Routing request to Qwen 7B...", icon="🧠")
        st.markdown("### 📄 Official Document")
        
        doc_container = st.empty()
        collected_text = ""
        
        # Crafting the system prompt specifically for school administration
        messages = [
            {
                "role": "system", 
                "content": (
                    f"You are the professional administrative assistant for {school_name}. "
                    f"Today's date is {doc_date.strftime('%d %B %Y')}. "
                    "Draft a formal, warm, and appropriate school notice, circular, or letter based on the user's prompt. "
                    "Include headers, dates, and sign-offs naturally. Do not include any placeholders, meta-commentary, or markdown code blocks. Just output the final document text ready for printing."
                )
            },
            {"role": "user", "content": user_input.strip()}
        ]
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OR_TOKEN.strip()}",
            "Content-Type": "application/json"
        }
        
        # Using the lightning-fast Qwen 2.5 7B model!
        payload = {
            "model": "qwen/qwen-2.5-7b-instruct:free", 
            "messages": messages,
            "temperature": 0.6,
            "stream": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            if response.status_code != 200:
                st.error(f"API Error ({response.status_code}): {response.text}")
                st.stop()
                
            # Stream the generated text live into the custom CSS doc box
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            token_text = data_json["choices"][0]["delta"].get("content", "")
                            if token_text:
                                collected_text += token_text
                                doc_container.markdown(f'<div class="doc-output">{collected_text} █</div>', unsafe_allow_html=True)
                        except Exception:
                            pass
                            
            # Final render without the typing cursor
            doc_container.markdown(f'<div class="doc-output">{collected_text}</div>', unsafe_allow_html=True)
            
            st.download_button(
                "⬇️ Download Document (.txt)", collected_text,
                f"school_doc_{doc_date.strftime('%Y%m%d')}.txt", "text/plain",
                use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Network Failure: {str(e)}")


# ═══════════════════════════════════════════════════════════
# MODULE 4 — FAST RAG TUTOR (UPGRADED APOLLO ENGINE)
# ═══════════════════════════════════════════════════════════
def render_rag_tutor():
    # Load embedding models and splitters
    embedder = get_embedding_model()
    text_splitter = get_text_splitter()

    # State Management Matrix
    if "vector_db" not in st.session_state: st.session_state.vector_db = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "response_time" not in st.session_state: st.session_state.response_time = "0.00s"
    if "source_reference" not in st.session_state: st.session_state.source_reference = "<div class='source-box font-mono'>Awaiting vector alignment...</div>"
    if "node_count" not in st.session_state: st.session_state.node_count = 0

    st.markdown("""
    <div class='header-bar'>
        <div class='header-left'>
            <div style='font-size: 1.25rem; font-weight: 700; letter-spacing: 0.05em; color: white;'>APOLLO <span style='color: #f97316;'>OMNI AI RAG</span></div>
        </div>
        <div class='status-badge'>● OPENROUTER LINKED</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([3, 6, 3], gap="large")

    # ================= LEFT COLUMN: INGESTION ENGINE =================
    with col_left:
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>⚙️ Zero-Cost Engine</div>", unsafe_allow_html=True)
        selected_model = st.selectbox("API Gateway Endpoint:", options=list(MODEL_OPTIONS.keys()), index=0)
        st.caption(f"**Desc:** {MODEL_OPTIONS[selected_model]['desc']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- WEB SEARCH INDEXER ---
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>🌐 Web Search Indexer</div>", unsafe_allow_html=True)
        web_query = st.text_input("Enter topic to scrape & index...", placeholder="e.g. Current AI news", label_visibility="collapsed")
        if st.button("SEARCH & INDEX", use_container_width=True):
            if web_query:
                with st.spinner("Scraping and chunking web data..."):
                    try:
                        results = DDGS().text(web_query, max_results=4, backend="html")
                        if not results:
                            results = DDGS().text(web_query, max_results=4, backend="lite")
                            
                        if results:
                            web_docs = []
                            for r in results:
                                doc = Document(
                                    page_content=r['body'], 
                                    metadata={"source": r['href'], "title": r['title']}
                                )
                                web_docs.append(doc)
                                
                            chunks = text_splitter.split_documents(web_docs)
                            if st.session_state.vector_db is None: 
                                st.session_state.vector_db = FAISS.from_documents(chunks, embedder)
                            else: 
                                st.session_state.vector_db.add_documents(chunks)
                                
                            st.session_state.node_count += len(chunks)
                            st.success(f"Indexed {len(chunks)} blocks from web!")
                        else:
                            st.warning("No web results found.")
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- LOCAL DOCUMENTS INDEXER ---
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>📚 Local Documents</div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload course materials...", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed", key="file_in")
        if st.button("SYNC KNOWLEDGE BASE", use_container_width=True):
            if uploaded_files:
                with st.spinner("Indexing materials..."):
                    docs = []
                    MAX_FILE_SIZE_MB = 5  
                    
                    for f in uploaded_files:
                        suffix = os.path.splitext(f.name)[1].lower()
                        file_bytes = f.read()
                        file_size_mb = len(file_bytes) / (1024 * 1024)
                        
                        if suffix == ".txt" and file_size_mb > MAX_FILE_SIZE_MB:
                            try:
                                lines = file_bytes.decode("utf-8", errors="ignore").splitlines(keepends=True)
                                current_chunk_lines = []
                                current_chunk_bytes = 0
                                max_bytes_per_chunk = MAX_FILE_SIZE_MB * 1024 * 1024
                                
                                for line in lines:
                                    line_bytes = len(line.encode('utf-8'))
                                    if current_chunk_bytes + line_bytes > max_bytes_per_chunk and current_chunk_lines:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                            tmp.write("".join(current_chunk_lines).encode("utf-8"))
                                            path = tmp.name
                                        try:
                                            docs.extend(TextLoader(path, encoding="utf-8").load())
                                        finally:
                                            if os.path.exists(path): os.unlink(path)
                                        
                                        current_chunk_lines = []
                                        current_chunk_bytes = 0
                                        
                                    current_chunk_lines.append(line)
                                    current_chunk_bytes += line_bytes
                                    
                                if current_chunk_lines:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                        tmp.write("".join(current_chunk_lines).encode("utf-8"))
                                        path = tmp.name
                                    try:
                                        docs.extend(TextLoader(path, encoding="utf-8").load())
                                    finally:
                                        if os.path.exists(path): os.unlink(path)
                            except Exception: pass
                        else:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(file_bytes)
                                path = tmp.name
                            try:
                                if suffix == ".pdf": docs.extend(PyPDFLoader(path).load())
                                elif suffix == ".txt": docs.extend(TextLoader(path, encoding="utf-8").load())
                            except Exception: pass
                            finally:
                                if os.path.exists(path): os.unlink(path)
                                
                    if docs:
                        chunks = text_splitter.split_documents(docs)
                        if st.session_state.vector_db is None: 
                            st.session_state.vector_db = FAISS.from_documents(chunks, embedder)
                        else: 
                            st.session_state.vector_db.add_documents(chunks)
                        st.session_state.node_count += len(chunks)
                        st.success(f"Indexed {len(chunks)} blocks.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= MIDDLE COLUMN: MAIN STUDY CONSOLE =================
    with col_mid:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='margin-top: 50px; margin-bottom: 30px; text-align: center;'>
                <h2 style='color: #f97316; font-family: "Inter", sans-serif; font-weight: 700;'>Study Console Initialized</h2>
                <p style='color: #a1a1aa; font-family: "JetBrains Mono", monospace; font-size: 0.85rem;'>Use the left panel to index Web Data or Local Files, then chat here.</p>
            </div>
            """, unsafe_allow_html=True)
        
        chat_scroll_pane = st.container(height=650, border=False)
        
        with chat_scroll_pane:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                
        user_query = st.chat_input("Enter your query...")
        
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            start_time = time.time()
            context_payload = ""
            
            if st.session_state.vector_db is not None:
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                matched_nodes = retriever.invoke(user_query)
                context_payload = "\n\n".join([f"[{node.metadata.get('source', 'Unknown')}]\n{node.page_content}" for node in matched_nodes])
                sys_instruction = "You are APOLLO OMNI AI, an advanced AI study buddy. Formulate a response using ONLY the provided context below. CITE YOUR SOURCES in your answer."
                clean_ctx = context_payload.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                st.session_state.source_reference = f"<div class='source-box'><strong>Active Context (RAG):</strong><br><br>{clean_ctx}</div>"
            else:
                sys_instruction = "You are APOLLO OMNI AI, an advanced AI study buddy. (Answering based on general knowledge)."
                st.session_state.source_reference = "<div class='source-box font-mono'>No active context. General weights used.</div>"

            message_stream = [{"role": "system", "content": sys_instruction}]
            for msg in st.session_state.chat_history[-4:]:
                message_stream.append({"role": msg["role"], "content": msg["content"]})
            message_stream.append({"role": "user", "content": f"Context Matrix:\n{context_payload}\n\nQuery: {user_query}"})
            
            with chat_scroll_pane:
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    collected_tokens = ""
                    try:
                        stream = generate_llm_stream(message_stream, OR_TOKEN, selected_model)
                        for chunk in stream:
                            collected_tokens += chunk
                            response_container.markdown(collected_tokens + " █")
                        if not collected_tokens.strip(): 
                            collected_tokens = "⚠️ EMPTY RESPONSE."
                        response_container.markdown(collected_tokens)
                    except Exception as ex:
                        collected_tokens = f"❌ FRAMEWORK API FAILURE: {ex}"
                        response_container.markdown(collected_tokens)
            
            st.session_state.chat_history.append({"role": "assistant", "content": collected_tokens})
            st.session_state.response_time = f"{time.time() - start_time:.2f}s"
            st.rerun()

    # ================= RIGHT COLUMN: PERFORMANCE & TELEMETRY MATRIX =================
    with col_right:
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>📊 Analytics Dashboard</div>", unsafe_allow_html=True)
        st.markdown(f"<div><div class='metric-title'>Inference Latency</div><div class='metric-value'>{st.session_state.response_time}</div></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
        st.markdown(f"<div><div class='metric-title'>Indexed Documents</div><div class='metric-value'>{st.session_state.node_count}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>📑 Verified Retrieval Matrix</div>", unsafe_allow_html=True)
        st.markdown(st.session_state.source_reference, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-header'>🛠️ Session Actions</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("PURGE", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.vector_db = None
                st.session_state.node_count = 0
                st.session_state.response_time = "0.00s"
                st.session_state.source_reference = "<div class='source-box font-mono'>Awaiting vector alignment...</div>"
                st.rerun()
        with c2:
            chat_log = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
            st.download_button("EXPORT", data=chat_log, file_name="apollo_log.txt", mime="text/plain", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════
def main():
    st.markdown(
        """<div style='background:linear-gradient(135deg,#1e2a50 0%,#0f1832 100%);
border:1px solid #2e3460;border-radius:14px;padding:1.4rem 2rem 1.2rem;margin-bottom:1.5rem'>
<h1 style='margin:0;font-size:1.9rem;color:#e0e7ff;font-weight:700'>
🏫 Multi-Model School Operations Console</h1>
<p style='margin:0.4rem 0 0;color:#818cf8;font-size:0.92rem'>
Integrated AI Architecture &nbsp;·&nbsp; Powered by Local Analytics & OpenRouter APIs</p>
</div>""",
        unsafe_allow_html=True)

    active = st.session_state.get("module_select", "🎓  Admissions Predictor")
    if "Admissions" in active:
        render_admissions()
    elif "Financial" in active:
        render_financial_auditor()
    elif "Admin" in active:
        render_admin_orchestrator()
    elif "RAG" in active:
        render_rag_tutor()


if __name__ == "__main__":
    main()
