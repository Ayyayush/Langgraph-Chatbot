# =========================================================
# Thinkr.ai - AGENTIC AI RESEARCH ASSISTANT (STREAMLIT FRONTEND)
# =========================================================

import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads, generate_research_report
from langchain_core.messages import HumanMessage
from services import vector_store, document_service
import uuid

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Thinkr.ai - Agentic AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    st.session_state['message_history'] = []
    st.session_state['research_report'] = None
    add_thread(new_id)
    st.rerun()

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    if not state:
        return []

    messages = state.values.get("messages", [])

    temp = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        temp.append({
            "role": role,
            "content": msg.content
        })

    return temp

# =========================================================
# SESSION SETUP
# =========================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "research_report" not in st.session_state:
    st.session_state["research_report"] = None

add_thread(st.session_state["thread_id"])

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🧠 Thinkr.ai")
st.sidebar.caption("Agentic AI Research Assistant")

# New Chat Button
if st.sidebar.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
    reset_chat()

st.sidebar.divider()

# ---------------------------------------------------------
# RESEARCH TOOLS: PDF upload (scoped to the current thread)
# ---------------------------------------------------------
st.sidebar.markdown("### 📄 Research Tools")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF to research",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_pdf is not None:
    already_added = uploaded_pdf.name in vector_store.list_documents(st.session_state["thread_id"])
    if not already_added:
        with st.sidebar.status(f"Processing {uploaded_pdf.name}...", expanded=False):
            try:
                text = document_service.extract_text_from_pdf(uploaded_pdf)
                if not text:
                    st.sidebar.warning(
                        "No extractable text found in this PDF "
                        "(it may be a scanned/image-only document)."
                    )
                else:
                    chunk_count = vector_store.add_document(
                        st.session_state["thread_id"], uploaded_pdf.name, text
                    )
                    st.sidebar.success(f"Indexed {uploaded_pdf.name} ({chunk_count} chunks).")
            except Exception as e:
                st.sidebar.error(f"Could not process PDF: {e}")

# Show which documents are attached to this thread
docs_in_thread = vector_store.list_documents(st.session_state["thread_id"])
if docs_in_thread:
    st.sidebar.caption("📎 Documents in this chat:")
    for doc_name in docs_in_thread:
        st.sidebar.caption(f"• {doc_name}")

# Research mode override
mode_choice = st.sidebar.selectbox(
    "Research mode",
    options=["Auto", "Normal Chat", "Web Search", "Document Search", "Combined Research"],
    help="Auto lets the assistant decide. Otherwise it forces every message in this "
         "turn to use the chosen mode.",
)
MODE_MAP = {
    "Auto": "auto",
    "Normal Chat": "normal_chat",
    "Web Search": "web_search",
    "Document Search": "document_search",
    "Combined Research": "combined_research",
}

# Generate research report button
if st.sidebar.button("📊 Generate Research Report", use_container_width=True):
    with st.spinner("Building research report..."):
        st.session_state["research_report"] = generate_research_report(
            st.session_state["thread_id"]
        )

st.sidebar.divider()
st.sidebar.markdown("### 💬 Conversations")

# Chat list (with current highlight)
for i, thread_id in enumerate(st.session_state["chat_threads"][::-1]):

    is_current = thread_id == st.session_state["thread_id"]
    label = f"🟢 {thread_id[:8]}..." if is_current else f"⚪ {thread_id[:8]}..."

    if st.sidebar.button(
        label,
        key=f"thread_{i}_{thread_id}",
        use_container_width=True
    ):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        st.session_state["research_report"] = None
        st.rerun()

# =========================================================
# MAIN UI
# =========================================================

st.markdown("""
<div style="
    padding: 15px;
    border-radius: 12px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    text-align: center;
">
    <h1 style="
        font-size: 38px;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    ">
        🧠 Thinkr.ai
    </h1>
    <p style="color: #94a3b8; margin-top: 0;">Agentic AI Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# Research report (if one has been generated for this thread)
if st.session_state["research_report"]:
    with st.expander("📊 Research Report", expanded=True):
        st.markdown(st.session_state["research_report"])

# Chat container
chat_container = st.container()

with chat_container:
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# =========================================================
# INPUT + STREAMING
# =========================================================

user_input = st.chat_input("Ask anything, or ask about your uploaded document...", key="main_chat_input")

if user_input:

    # USER MESSAGE
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "mode_override": MODE_MAP[mode_choice],
        }
    }

    # AI STREAMING
    full_response = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()

        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response)

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": full_response
    })
