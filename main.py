"""
============================================================
main.py — Streamlit Frontend for Faculty Information RAG System
============================================================
This is the entry point for the application. It provides:
1. A premium dark-themed UI with glassmorphism cards
2. Chat-style interface for asking questions about faculty
3. Sidebar with system status, model selection, and controls
4. Session-based chat history with styled message bubbles
5. Example queries as clickable chips for quick starts

Run with: streamlit run main.py
============================================================
"""

import time
import streamlit as st
from datetime import datetime

# ── Page Configuration (must be first Streamlit call) ──────
st.set_page_config(
    page_title="Faculty RAG System — UIET Panjab University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import project modules (after page config) ────────────
from embeddings import get_or_create_vectorstore
from rag_chain import build_rag_chain, query_chain, check_ollama_status, get_available_models


# ============================================================
# CUSTOM CSS — Premium Dark Theme with Glassmorphism
# ============================================================
def inject_custom_css():
    """Inject custom CSS for a premium, modern dark UI."""
    st.markdown("""
    <style>
    /* ── Import Google Font ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ──────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main Container ─────────────────────────────────── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* ── Header Gradient Title ──────────────────────────── */
    .header-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15) 0%, rgba(99, 179, 237, 0.1) 50%, rgba(236, 72, 153, 0.08) 100%);
        border-radius: 20px;
        border: 1px solid rgba(108, 99, 255, 0.2);
        backdrop-filter: blur(10px);
    }

    .header-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #63B3ED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    .header-badge {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.3rem 1rem;
        background: rgba(108, 99, 255, 0.15);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 20px;
        font-size: 0.75rem;
        color: #A5B4FC;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* ── Glass Card ──────────────────────────────────────── */
    .glass-card {
        background: rgba(26, 29, 41, 0.7);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(108, 99, 255, 0.35);
        box-shadow: 0 4px 24px rgba(108, 99, 255, 0.08);
    }

    /* ── Chat Message Bubbles ───────────────────────────── */
    .user-message {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.2), rgba(108, 99, 255, 0.1));
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 16px 16px 4px 16px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #E2E8F0;
        font-size: 0.95rem;
    }

    .assistant-message {
        background: rgba(30, 34, 48, 0.8);
        border: 1px solid rgba(99, 179, 237, 0.15);
        border-radius: 16px 16px 16px 4px;
        padding: 1.2rem 1.4rem;
        margin: 0.5rem 0;
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    .message-timestamp {
        font-size: 0.7rem;
        color: #64748B;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }

    .user-label { color: #A5B4FC; }
    .assistant-label { color: #63B3ED; }

    /* ── Source Documents Expander ───────────────────────── */
    .source-chip {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        margin: 0.15rem;
        background: rgba(108, 99, 255, 0.1);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 8px;
        font-size: 0.75rem;
        color: #A5B4FC;
    }

    /* ── Sidebar Styles ─────────────────────────────────── */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-online {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ADE80;
    }

    .status-offline {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #F87171;
    }

    .stat-card {
        background: rgba(26, 29, 41, 0.5);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6C63FF;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* ── Example Query Chips ────────────────────────────── */
    .example-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }

    /* ── Input Area Enhancement ──────────────────────────── */
    .stTextInput > div > div > input {
        background: rgba(26, 29, 41, 0.8) !important;
        border: 1px solid rgba(108, 99, 255, 0.25) !important;
        border-radius: 12px !important;
        color: #E2E8F0 !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.95rem !important;
        transition: border-color 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2) !important;
    }

    /* ── Button Enhancement ──────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #5B54E0) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.3px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #5B54E0, #4A43CF) !important;
        box-shadow: 0 4px 16px rgba(108, 99, 255, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Spinner ─────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #6C63FF !important;
    }

    /* ── Divider ─────────────────────────────────────────── */
    hr {
        border-color: rgba(108, 99, 255, 0.1) !important;
    }

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #2D3348; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6C63FF; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize all session state variables on first run."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "vs_status" not in st.session_state:
        st.session_state.vs_status = None
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama3"
    
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False


# ============================================================
# VECTOR STORE LOADING (cached across reruns)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_vectorstore_cached(_force_rebuild=False):
    """
    Load or create the vector store (cached so it persists across reruns).
    The underscore prefix on _force_rebuild prevents Streamlit from hashing it.
    """
    return get_or_create_vectorstore(force_rebuild=_force_rebuild)


# ============================================================
# SIDEBAR — System Status Panel
# ============================================================
def render_sidebar():
    """Render the sidebar with system status, model selection, and controls."""
    with st.sidebar:
        # ── Sidebar Header ──────────────────────────────────
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem;">
            <span style="font-size: 2rem;">⚙️</span>
            <h3 style="margin: 0.3rem 0 0; font-weight: 700; color: #E2E8F0;">System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ── Ollama Connection Status ────────────────────────
        st.markdown("##### 🔌 Ollama Connection")
        ollama_status = check_ollama_status()
        
        if ollama_status["running"]:
            st.markdown(
                '<div class="status-indicator status-online">🟢 Connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-indicator status-offline">🔴 Disconnected</div>',
                unsafe_allow_html=True,
            )
            st.error(ollama_status["error"])
            st.info("Start Ollama with: `ollama serve`")
        
        st.markdown("")
        
        # ── Model Selection ─────────────────────────────────
        st.markdown("##### 🤖 Model Selection")
        available_models = get_available_models()
        
        # Find current model index in available list
        current_model = st.session_state.selected_model
        default_index = 0
        for i, m in enumerate(available_models):
            if current_model in m:
                default_index = i
                break
        
        selected = st.selectbox(
            "Choose LLM model:",
            options=available_models,
            index=default_index,
            key="model_selector",
            label_visibility="collapsed",
        )
        
        # Rebuild chain if model changed
        if selected != st.session_state.selected_model:
            st.session_state.selected_model = selected
            st.session_state.rag_chain = None  # Force chain rebuild
            st.rerun()
        
        st.markdown("")
        
        # ── Vector Store Status ─────────────────────────────
        st.markdown("##### 📦 Vector Store")
        
        if st.session_state.vs_status:
            status = st.session_state.vs_status
            source_label = "✅ Loaded from disk" if status["source"] == "loaded" else "🆕 Freshly built"
            st.markdown(
                f'<div class="status-indicator status-online">{source_label}</div>',
                unsafe_allow_html=True,
            )
            
            # Stats cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{status['num_chunks']}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                num_pdfs = status.get('num_pdfs', '—')
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{num_pdfs}</div>
                    <div class="stat-label">PDFs</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="status-indicator status-offline">⏳ Not loaded</div>',
                unsafe_allow_html=True,
            )
        
        st.markdown("")
        
        # ── Rebuild Button ──────────────────────────────────
        st.markdown("##### 🔄 Controls")
        if st.button("🔨 Rebuild Vector Store", use_container_width=True):
            # Clear cached vector store and rebuild
            load_vectorstore_cached.clear()
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.vs_status = None
            st.session_state.system_ready = False
            st.rerun()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # ── About Section ───────────────────────────────────
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem; opacity: 0.6;">
            <p style="font-size: 0.75rem; color: #64748B; margin: 0;">
                Faculty RAG System v1.0<br>
                UIET • Panjab University<br>
                Powered by LangChain & Ollama
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# HEADER — Title and Description
# ============================================================
def render_header():
    """Render the main header section with gradient title."""
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🎓 Faculty Information RAG System</div>
        <div class="header-subtitle">
            Ask questions about UIET Panjab University faculty — powered by local AI
        </div>
        <div class="header-badge">
            🔒 100% LOCAL • NO DATA LEAVES YOUR MACHINE
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# EXAMPLE QUERIES — Clickable Quick Start Chips
# ============================================================
EXAMPLE_QUERIES = [
    "Who are the faculty members in the CSE department?",
    "What is the qualification of Dr. Parminder Kaur?",
    "List the research areas of ECE faculty",
    "Which faculty members have a PhD?",
    "Tell me about the IT department faculty",
]


def render_example_queries():
    """Render example query buttons that auto-fill the input."""
    st.markdown(
        '<p style="color: #64748B; font-size: 0.85rem; margin-bottom: 0.5rem;">'
        '💡 Try an example query:</p>',
        unsafe_allow_html=True,
    )
    
    cols = st.columns(len(EXAMPLE_QUERIES))
    for i, query in enumerate(EXAMPLE_QUERIES):
        with cols[i % len(EXAMPLE_QUERIES)]:
            # Truncate display text for chips
            display = query if len(query) <= 35 else query[:32] + "..."
            if st.button(display, key=f"example_{i}", use_container_width=True):
                return query
    return None


# ============================================================
# CHAT HISTORY — Styled Message Display
# ============================================================
def render_chat_history():
    """Render all previous chat messages in styled bubbles."""
    if not st.session_state.chat_history:
        return
    
    st.markdown("---")
    
    for entry in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class="user-message">
            <div class="message-label user-label">👤 You</div>
            {entry['question']}
            <div class="message-timestamp">{entry['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-label assistant-label">🤖 Assistant</div>
            {entry['answer']}
            <div class="message-timestamp">{entry['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Source documents (collapsible)
        if entry.get("sources"):
            with st.expander("📄 Source Documents", expanded=False):
                for src in entry["sources"]:
                    st.markdown(
                        f'<span class="source-chip">📑 {src}</span>',
                        unsafe_allow_html=True,
                    )


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main application entry point."""
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # ── System Initialization ───────────────────────────────
    # Load/create vector store on first run
    if not st.session_state.system_ready:
        with st.status("🚀 Initializing system...", expanded=True) as status_container:
            try:
                # Step 1: Load or create vector store
                st.write("📦 Loading vector store...")
                vectorstore, vs_status = load_vectorstore_cached()
                st.session_state.vectorstore = vectorstore
                st.session_state.vs_status = vs_status
                
                if vs_status["source"] == "loaded":
                    st.write(f"✅ Vector store loaded ({vs_status['num_chunks']} chunks)")
                else:
                    st.write(f"✅ Vector store created ({vs_status['num_chunks']} chunks from {vs_status['num_pdfs']} PDFs)")
                
                # Step 2: Build RAG chain
                st.write(f"🔗 Building RAG chain with {st.session_state.selected_model}...")
                chain = build_rag_chain(vectorstore, st.session_state.selected_model)
                st.session_state.rag_chain = chain
                st.write("✅ RAG chain ready")
                
                st.session_state.system_ready = True
                status_container.update(label="✅ System ready!", state="complete", expanded=False)
                
            except Exception as e:
                status_container.update(label="❌ Initialization failed", state="error")
                st.error(f"Failed to initialize: {str(e)}")
                st.info(
                    "**Troubleshooting tips:**\n"
                    "1. Make sure Ollama is running: `ollama serve`\n"
                    "2. Pull the embedding model: `ollama pull nomic-embed-text`\n"
                    "3. Pull the LLM model: `ollama pull llama3`\n"
                    "4. Place PDF files in the `data/pdfs/` directory if web scraping fails"
                )
                return
    
    # ── Rebuild RAG chain if model changed ──────────────────
    if st.session_state.rag_chain is None and st.session_state.vectorstore is not None:
        with st.spinner(f"Switching to {st.session_state.selected_model}..."):
            chain = build_rag_chain(
                st.session_state.vectorstore,
                st.session_state.selected_model,
            )
            st.session_state.rag_chain = chain
    
    # ── Example Queries ─────────────────────────────────────
    example_query = render_example_queries()
    
    st.markdown("")
    
    # ── Question Input ──────────────────────────────────────
    question = st.text_input(
        "Ask a question about UIET faculty:",
        placeholder="e.g., What are the research interests of the CSE department faculty?",
        key="question_input",
        label_visibility="collapsed",
    )
    
    # Use example query if clicked
    if example_query:
        question = example_query
    
    # ── Submit Button ───────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit = st.button("🔍 Ask Question", use_container_width=True)
    
    # ── Process Question ────────────────────────────────────
    if submit and question and question.strip():
        if st.session_state.rag_chain is None:
            st.warning("System not ready. Please wait for initialization.")
            return
        
        # Show processing animation
        with st.spinner("🧠 Thinking..."):
            start_time = time.time()
            result = query_chain(st.session_state.rag_chain, question.strip())
            elapsed = time.time() - start_time
        
        # Display the answer in a styled card
        st.markdown(f"""
        <div class="glass-card">
            <div class="message-label assistant-label">🤖 Answer</div>
            <div style="color: #E2E8F0; line-height: 1.7; font-size: 0.95rem;">
                {result['answer']}
            </div>
            <div class="message-timestamp">
                Generated in {elapsed:.1f}s using {st.session_state.selected_model}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show source documents
        if result["source_documents"]:
            with st.expander("📄 View Source Documents", expanded=False):
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i}:** `{source}` (Page {page})")
                    st.markdown(
                        f'<div style="background: rgba(26,29,41,0.5); padding: 0.8rem; '
                        f'border-radius: 8px; font-size: 0.85rem; color: #94A3B8; '
                        f'border-left: 3px solid #6C63FF; margin-bottom: 0.5rem;">'
                        f'{doc.page_content[:300]}...</div>',
                        unsafe_allow_html=True,
                    )
        
        # Save to chat history
        source_names = list(set(
            doc.metadata.get("source", "Unknown")
            for doc in result["source_documents"]
        ))
        
        st.session_state.chat_history.append({
            "question": question.strip(),
            "answer": result["answer"],
            "sources": source_names,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "model": st.session_state.selected_model,
        })
    
    elif submit and (not question or not question.strip()):
        st.warning("Please enter a question first!")
    
    # ── Chat History ────────────────────────────────────────
    render_chat_history()


# ── Application Entry Point ────────────────────────────────
if __name__ == "__main__":
    main()
