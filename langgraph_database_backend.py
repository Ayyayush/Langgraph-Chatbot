# =========================================================
# Thinkr.ai BACKEND (LANGGRAPH + GROQ + SQLITE)
# =========================================================
# This file upgrades the original single-node chatbot into a
# small agentic research workflow:
#
#   START -> planner -> researcher -> generator -> END
#                  \_______________/
#                (skipped for normal chat)
#
#   planner    : decides what kind of help the query needs
#   researcher : gathers web / document context if needed
#   generator  : writes the final answer (this is the node
#                whose LLM call gets streamed to the UI)
#
# The SQLite persistence (SqliteSaver) and thread management
# from the original project are untouched, so existing chat
# history keeps working exactly as before.

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

from services import web_search, vector_store

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

# =========================================================
# LLM (GROQ)
# =========================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # fast + good
    temperature=0
)

# =========================================================
# RESEARCH MODES
# =========================================================
# "auto"     -> planner decides using simple keyword rules
# any other  -> user (via the UI) forces this exact mode
NORMAL_CHAT = "normal_chat"
WEB_SEARCH = "web_search"
DOCUMENT_SEARCH = "document_search"
COMBINED_RESEARCH = "combined_research"

WEB_KEYWORDS = [
    "latest", "recent", "current", "today", "news", "2025", "2026",
    "search the web", "look up", "what is happening", "update on",
]
DOC_KEYWORDS = [
    "document", "pdf", "uploaded", "paper", "summarize", "summary",
    "methodology", "conclusion", "findings", "this file", "the file",
    "attached",
]


def _classify_query(query: str, has_documents: bool) -> str:
    """Very small, transparent heuristic - no extra LLM call needed
    just to decide *what kind* of help a message needs."""
    q = query.lower()
    wants_web = any(k in q for k in WEB_KEYWORDS)
    wants_doc = has_documents and any(k in q for k in DOC_KEYWORDS)

    if wants_web and wants_doc:
        return COMBINED_RESEARCH
    if wants_web:
        return WEB_SEARCH
    if wants_doc:
        return DOCUMENT_SEARCH
    return NORMAL_CHAT


# =========================================================
# STATE
# =========================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: str
    web_results: str
    doc_context: str


# =========================================================
# NODE 1: PLANNER
# =========================================================
def planner_node(state: ChatState, config: RunnableConfig):
    last_message = state["messages"][-1].content
    thread_id = config["configurable"]["thread_id"]
    mode_override = config["configurable"].get("mode_override", "auto")

    if mode_override != "auto":
        mode = mode_override
    else:
        mode = _classify_query(last_message, vector_store.has_documents(thread_id))

    return {"mode": mode}


# =========================================================
# NODE 2: RESEARCHER
# =========================================================
def researcher_node(state: ChatState, config: RunnableConfig):
    query = state["messages"][-1].content
    thread_id = config["configurable"]["thread_id"]
    mode = state["mode"]

    web_results = ""
    doc_context = ""

    if mode in (WEB_SEARCH, COMBINED_RESEARCH):
        web_results = web_search.search(query)

    if mode in (DOCUMENT_SEARCH, COMBINED_RESEARCH):
        doc_context = vector_store.query(thread_id, query)

    return {"web_results": web_results, "doc_context": doc_context}


def _route_after_planner(state: ChatState):
    return "researcher" if state["mode"] != NORMAL_CHAT else "generator"


# =========================================================
# NODE 3: GENERATOR (final answer)
# =========================================================
def generator_node(state: ChatState):
    messages = list(state["messages"])
    context_blocks = []

    if state.get("web_results"):
        context_blocks.append(f"WEB SEARCH RESULTS:\n{state['web_results']}")
    if state.get("doc_context"):
        context_blocks.append(f"RELEVANT DOCUMENT EXCERPTS:\n{state['doc_context']}")

    if context_blocks:
        instructions = (
            "You are Thinkr.ai, a research assistant. Use the research context "
            "below to answer the user's latest question as accurately as possible. "
            "Mention sources (links or document names) where relevant. If the "
            "context doesn't fully answer the question, say so honestly.\n\n"
            + "\n\n".join(context_blocks)
        )
        messages = [SystemMessage(content=instructions)] + messages

    response = llm.invoke(messages)

    # Reset per-turn research context so it doesn't leak into the next question
    return {"messages": [response], "web_results": "", "doc_context": ""}


# =========================================================
# SQLITE CHECKPOINTER
# =========================================================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# =========================================================
# GRAPH
# =========================================================
graph = StateGraph(ChatState)

graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("generator", generator_node)

graph.add_edge(START, "planner")
graph.add_conditional_edges(
    "planner",
    _route_after_planner,
    {"researcher": "researcher", "generator": "generator"},
)
graph.add_edge("researcher", "generator")
graph.add_edge("generator", END)

chatbot = graph.compile(checkpointer=checkpointer)

# =========================================================
# RETRIEVE THREADS
# =========================================================
def retrieve_all_threads():
    threads = set()

    try:
        checkpoints = checkpointer.list(None)
    except Exception:
        return []

    for checkpoint in checkpoints:
        config = getattr(checkpoint, "config", None)

        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")

            if thread_id:
                threads.add(thread_id)

    return sorted(list(threads))


# =========================================================
# RESEARCH REPORT GENERATION
# =========================================================
REPORT_PROMPT = """You are Thinkr.ai. Using the conversation below (which already
contains the user's questions and your researched answers), write a structured
research report on the main topic discussed.

Use exactly this Markdown structure:

# Research Topic
# Executive Summary
# Key Findings
# Detailed Analysis
# Evidence / Sources
# Conclusion

Be concise but thorough. If sources/links were mentioned in the conversation,
list them under "Evidence / Sources". If a section has nothing to add, say so
briefly instead of leaving it empty.

CONVERSATION:
{conversation}
"""


def generate_research_report(thread_id: str) -> str:
    """
    Build a structured research report from everything discussed so far
    in this thread (reuses the same persisted conversation - no separate
    storage needed).
    """
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state else []

    if not messages:
        return "There's no conversation yet to build a report from."

    transcript_lines = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        transcript_lines.append(f"{role}: {msg.content}")
    transcript = "\n\n".join(transcript_lines)

    prompt = REPORT_PROMPT.format(conversation=transcript)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
