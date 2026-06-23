"""
main.py  LangGraph RAG agent with proper @tool + ToolNode
=========================================================
Agent flow (recursion_limit=3, so at most 3 tool calls total):

  START
    -> check_intent        classifies clinical vs non-clinical
    -> agent_node          LLM decides which tool to call and with what args
    -> tool_node           executes the chosen tool (search_chunks or get_adjacent_chunks)
    -> agent_node          LLM sees tool result, decides to call another tool or stop
    -> format_response     synthesises all tool results into final markdown answer
  END

The LLM autonomously decides:
  - call 1: always search_chunks (it knows this from the system prompt)
  - call 2 or 3: get_adjacent_chunks if it needs more context, or stop early

recursion_limit=3 is passed at graph.invoke() time, capping total agent<->tool cycles.
"""

import os
import json
import logging
from typing import TypedDict, Literal, Annotated
import operator

import psycopg2
import psycopg2.pool
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger("pmtpt.graph")


# ============================================================
# Database connection pool
# ============================================================

RAW_DB_URL = os.getenv("DATABASE_URL", "")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info("Initialising connection pool ...")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=5, dsn=RAW_DB_URL, connect_timeout=5
        )
        logger.info("Connection pool ready.")
    return _pool


# ============================================================
# Embedding model  shared with ingest.py, loaded once per process
# ============================================================

_EMBED_MODEL: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info("Loading embedding model ...")
        _EMBED_MODEL = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME", None),
        )
        logger.info("Embedding model ready.")
    return _EMBED_MODEL

def _embed_query(text: str) -> list[float]:
    return _get_embed_model().encode(text, show_progress_bar=False).tolist()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    request_timeout=30,
)

_retry_decorator = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


# ============================================================
# Agent state
# ============================================================

class AgentState(TypedDict):
    user_question:      str
    is_clinical:        bool
    messages:           Annotated[list, operator.add]   # full message history for the agent loop
    formatted_response: str
    error:              str


# ============================================================
# Tool 1: search_chunks
# The LLM will always call this first.
# It embeds the query, runs cosine similarity via pgvector, returns top-k chunks.
# ============================================================

@tool
def search_chunks(query: str, top_k: int = 5) -> str:
    """
    Search clinical PDF documents by semantic similarity.
    Always call this first for any clinical question.
    Returns the top matching chunks with their source document, page number, and similarity score.

    Args:
        query:  the clinical question or search phrase to look up
        top_k:  number of chunks to return (default 5, max 10)
    """
    top_k = min(top_k, 10)
    query_vec = _embed_query(query)
    vec_literal = "[" + ",".join(str(v) for v in query_vec) + "]"

    sql = """
        SELECT
            chunk_id::text,
            document_id::text,
            chunk_index,
            page_number,
            chunk_text,
            filename,
            COALESCE(category, 'general') AS category,
            ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS similarity
        FROM chunks_with_document
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    pool = _get_pool()
    conn = pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute(sql, (vec_literal, vec_literal, top_k))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        cur.close()

        if not rows:
            return "No relevant chunks found in the indexed documents."

        chunks = [dict(zip(cols, row)) for row in rows]

        result_lines = []
        for c in chunks:
            result_lines.append(
                f"[chunk_id: {c['chunk_id']} | {c['filename']} | page {c['page_number']} "
                f"| similarity: {c['similarity']}]\n{c['chunk_text']}"
            )

        return "\n\n===\n\n".join(result_lines)

    except Exception as e:
        logger.exception("search_chunks failed: %s", e)
        return f"Search failed: {e}"
    finally:
        pool.putconn(conn)


# ============================================================
# Tool 2: get_adjacent_chunks
# The LLM calls this only when a retrieved chunk cuts off mid-sentence
# or when it needs surrounding context to complete an answer.
# ============================================================

@tool
def get_adjacent_chunks(chunk_id: str, direction: str = "both") -> str:
    """
    Fetch the neighbouring chunks (previous, next, or both) for a specific chunk.
    Call this only when a retrieved chunk appears to cut off mid-sentence or
    the answer clearly continues beyond what was returned by search_chunks.

    Args:
        chunk_id:   the UUID of the chunk you want to expand context around
        direction:  'prev' to get the chunk before it,
                    'next' to get the chunk after it,
                    'both' to get both neighbours (default)
    """
    if direction not in ("prev", "next", "both"):
        direction = "both"

    pool = _get_pool()
    conn = pool.getconn()
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT document_id, chunk_index FROM document_chunks WHERE id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            return f"Chunk {chunk_id} not found."

        document_id, chunk_index = str(row[0]), row[1]

        if direction == "prev":
            indices = list(range(max(0, chunk_index - 2), chunk_index))
        elif direction == "next":
            indices = list(range(chunk_index + 1, chunk_index + 3))
        else:
            indices = (
                list(range(max(0, chunk_index - 2), chunk_index)) +
                list(range(chunk_index + 1, chunk_index + 3))
            )

        if not indices:
            return "No adjacent chunks available (already at boundary)."

        placeholders = ",".join(["%s"] * len(indices))
        cur.execute(
            f"""
            SELECT
                ck.id::text      AS chunk_id,
                ck.chunk_index,
                ck.page_number,
                ck.chunk_text,
                d.filename,
                COALESCE(dc.label, 'general') AS category
            FROM document_chunks ck
            JOIN clinical_documents d   ON d.id  = ck.document_id
            LEFT JOIN document_categories dc ON dc.id = d.category_id
            WHERE ck.document_id = %s
              AND ck.chunk_index IN ({placeholders})
            ORDER BY ck.chunk_index
            """,
            (document_id, *indices),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        cur.close()

        if not rows:
            return "No adjacent chunks found."

        chunks = [dict(zip(cols, row)) for row in rows]
        result_lines = []
        for c in chunks:
            result_lines.append(
                f"[chunk_id: {c['chunk_id']} | {c['filename']} | page {c['page_number']}]\n{c['chunk_text']}"
            )

        return "\n\n===\n\n".join(result_lines)

    except Exception as e:
        logger.exception("get_adjacent_chunks failed: %s", e)
        return f"Adjacent chunk lookup failed: {e}"
    finally:
        pool.putconn(conn)


# ============================================================
# Register tools with LLM and ToolNode
# ============================================================

TOOLS = [search_chunks, get_adjacent_chunks]
llm_with_tools = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


# ============================================================
# System prompt  tells the LLM its role and tool usage rules
# ============================================================

SYSTEM_PROMPT = """You are an expert Clinical AI Assistant for Tuberculosis Preventive Treatment (TPT) in India.
You have access to a database of indexed clinical PDF documents including treatment protocols,
lab reports, patient summaries, policy documents, and research papers.

You have two tools:

1. search_chunks  call this FIRST for every clinical question.
   It searches the PDF database semantically and returns the most relevant text chunks.

2. get_adjacent_chunks  call this ONLY if a retrieved chunk cuts off mid-sentence
   or you need surrounding context to complete the answer. Pass the chunk_id from
   the search result and the direction (prev, next, or both).

Rules:
- Always call search_chunks first.
- Only call get_adjacent_chunks if the search results are clearly incomplete.
- Do not call any tool more than once with the same arguments.
- After gathering enough context, stop calling tools and write your final answer.
- Base your answer only on the retrieved chunks. Do not hallucinate.
- Cite the source filename and page number for every key fact.
- Format the final answer in clear Markdown."""


# ============================================================
# Graph nodes
# ============================================================

def check_intent(state: AgentState) -> AgentState:
    """
    Classifies the question as clinical or non-clinical.
    Non-clinical questions get a friendly reply and skip the tool loop entirely.
    """
    question = state.get("user_question", "").strip()
    if not question:
        return {**state, "error": "Empty question received.", "formatted_response": "Please ask a clinical question."}

    prompt = f"""Classify the following message as CLINICAL or NON_CLINICAL.

CLINICAL means: anything about patient treatment, TB protocols, lab results, clinical guidelines,
research, policies, medications, dosages, diagnoses, or anything answerable from clinical PDF documents.
NON_CLINICAL means: greetings, jokes, or general knowledge unrelated to clinical documents.

Reply with exactly one word: CLINICAL or NON_CLINICAL

Message: {question}
Classification:"""

    try:
        @_retry_decorator
        def _call():
            return llm.invoke(prompt).content.strip().upper()

        classification = _call()
        is_clinical = classification.startswith("CLINICAL")
        logger.info("Intent classification: %s", classification)

        if not is_clinical:
            @_retry_decorator
            def _friendly():
                p = f"""You are a friendly Clinical Document AI Assistant.
The user sent a non-clinical message. Respond warmly in 2-3 sentences.
Mention that you can help search clinical PDFs including treatment protocols, lab reports, and research papers.
Message: {question}"""
                return llm.invoke(p).content.strip()

            return {
                **state,
                "is_clinical": False,
                "formatted_response": _friendly(),
                "messages": [],
            }

        return {
            **state,
            "is_clinical": True,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
        }

    except Exception as e:
        logger.exception("check_intent failed: %s", e)
        return {
            **state,
            "is_clinical": True,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
        }


def route_after_intent(state: AgentState) -> Literal["agent_node", "format_response"]:
    return "agent_node" if state.get("is_clinical", True) else "format_response"


def agent_node(state: AgentState) -> AgentState:
    """
    The core agent loop node.
    The LLM receives the full message history (including prior tool results)
    and decides to either call a tool or produce a final answer.
    """
    messages = state.get("messages", [])

    @_retry_decorator
    def _call():
        return llm_with_tools.invoke(messages)

    try:
        response = _call()
        logger.info(
            "Agent response: tool_calls=%d",
            len(response.tool_calls) if hasattr(response, "tool_calls") else 0,
        )
        return {**state, "messages": [response]}
    except Exception as e:
        logger.exception("agent_node failed: %s", e)
        return {**state, "error": str(e)}


def route_after_agent(state: AgentState) -> Literal["tool_node", "format_response"]:
    """
    If the last message from the LLM contains tool_calls, execute them.
    Otherwise the LLM is done reasoning and we move to format_response.
    """
    if state.get("error"):
        return "format_response"

    messages = state.get("messages", [])
    if not messages:
        return "format_response"

    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"

    return "format_response"


def format_response(state: AgentState) -> AgentState:
    """
    Synthesises the full message history (including all tool results)
    into a final clean Markdown clinical response.

    If the agent already produced a plain-text final answer (no more tool calls),
    we use it directly. Otherwise we ask the LLM to summarise what it collected.
    """
    if state.get("error"):
        return {**state, "formatted_response": "Something went wrong. Please try again."}

    if state.get("formatted_response"):
        return state

    messages = state.get("messages", [])
    if not messages:
        return {**state, "formatted_response": "I could not find relevant information in the clinical documents."}

    last = messages[-1]

    # If the last message is a plain AIMessage (no tool calls), it is the final answer
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", []):
        return {**state, "formatted_response": last.content.strip()}

    # Fallback: ask the LLM to write the final answer based on the conversation so far
    summary_messages = messages + [
        HumanMessage(content=(
            "Based on all the document chunks you retrieved above, "
            "write the final clinical answer in clear Markdown. "
            "Cite the source filename and page number for key facts. "
            "Do not call any more tools."
        ))
    ]

    @_retry_decorator
    def _summarise():
        return llm.invoke(summary_messages).content.strip()

    try:
        answer = _summarise()
        return {**state, "formatted_response": answer}
    except Exception as e:
        logger.exception("format_response failed: %s", e)
        return {**state, "formatted_response": "Something went wrong while formatting the response."}


# ============================================================
# Graph assembly
# recursion_limit=3 is enforced at invoke() time in app.py,
# capping the agent<->tool_node loop to at most 3 cycles.
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("check_intent",    check_intent)
workflow.add_node("agent_node",      agent_node)
workflow.add_node("tool_node",       tool_node)
workflow.add_node("format_response", format_response)

workflow.add_edge(START, "check_intent")

workflow.add_conditional_edges(
    "check_intent", route_after_intent,
    {"agent_node": "agent_node", "format_response": "format_response"}
)

workflow.add_conditional_edges(
    "agent_node", route_after_agent,
    {"tool_node": "tool_node", "format_response": "format_response"}
)

# After tool_node executes, go back to agent_node so the LLM can see the result
# and decide whether to call another tool or stop.
# recursion_limit=3 breaks this loop after 3 agent<->tool cycles.
workflow.add_edge("tool_node", "agent_node")

workflow.add_edge("format_response", END)

graph = workflow.compile()
