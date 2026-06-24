"""
main.py  LangGraph RAG agent — Groq SDK direct tool calling
============================================================
Bypasses LangChain bind_tools/ToolNode entirely.
Tools are defined as plain dicts and sent directly to the Groq API,
exactly like the InferenceClient pattern — model-agnostic and reliable.

Agent flow (max 3 tool call cycles):

  START
    -> check_intent       classifies clinical vs non-clinical
    -> retrieve           embeds query, fetches top-5 chunks from pgvector
    -> generate           Groq API call with tools; LLM decides to call a tool or answer
    -> execute_tool       runs whichever tool the LLM picked, appends result to messages
    -> generate           LLM sees tool result, decides again (loop, max 3 cycles)
    -> format_response    extracts final answer from message history
  END
"""

import os
import json
import logging
from typing import Any, Literal, TypedDict

import psycopg2
import psycopg2.pool
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langgraph.graph import StateGraph, START, END

load_dotenv()
logger = logging.getLogger("pmtpt.graph")


# ============================================================
# Groq client  direct SDK, no LangChain wrapper
# ============================================================

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_retry_decorator = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


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
# Embedding model
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
# Tool definitions as plain dicts sent directly to Groq API.
# No LangChain @tool, no bind_tools — just the raw OpenAI-compatible
# function schema that Groq expects.
# ============================================================

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "Search clinical PDF documents by semantic similarity. "
                "Always call this first for any clinical question. "
                "Returns the top matching chunks with source document, page number, and similarity score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The clinical question or search phrase to look up.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5, max 10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_adjacent_chunks",
            "description": (
                "Fetch the neighbouring chunks (previous, next, or both) for a specific chunk. "
                "Call this only when a retrieved chunk appears to cut off mid-sentence or "
                "the answer clearly continues beyond what search_chunks returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The UUID of the chunk to expand context around.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["prev", "next", "both"],
                        "description": "Which neighbours to fetch (default: both).",
                        "default": "both",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
]


# ============================================================
# Tool executor  plain Python functions, no LangChain ToolNode.
# Each function returns a plain string that goes back to the LLM
# as a tool message.
# ============================================================

def _exec_search_chunks(query: str, top_k: int = 5) -> str:
    top_k = min(int(top_k), 10)
    query_vec   = _embed_query(query)
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
        parts  = []
        for c in chunks:
            parts.append(
                f"[chunk_id: {c['chunk_id']} | {c['filename']} | "
                f"page {c['page_number']} | similarity: {c['similarity']}]\n{c['chunk_text']}"
            )
        return "\n\n===\n\n".join(parts)

    except Exception as e:
        logger.exception("search_chunks failed: %s", e)
        return f"Search failed: {e}"
    finally:
        pool.putconn(conn)


def _exec_get_adjacent_chunks(chunk_id: str, direction: str = "both") -> str:
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
            JOIN clinical_documents d    ON d.id  = ck.document_id
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

        parts = []
        for c in [dict(zip(cols, r)) for r in rows]:
            parts.append(
                f"[chunk_id: {c['chunk_id']} | {c['filename']} | page {c['page_number']}]\n{c['chunk_text']}"
            )
        return "\n\n===\n\n".join(parts)

    except Exception as e:
        logger.exception("get_adjacent_chunks failed: %s", e)
        return f"Adjacent chunk lookup failed: {e}"
    finally:
        pool.putconn(conn)


def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Route a tool call to the correct executor function."""
    try:
        if name == "search_chunks":
            return _exec_search_chunks(
                query=arguments["query"],
                top_k=int(arguments.get("top_k", 5)),
            )
        elif name == "get_adjacent_chunks":
            return _exec_get_adjacent_chunks(
                chunk_id=arguments["chunk_id"],
                direction=arguments.get("direction", "both"),
            )
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================
# Agent state
# ============================================================

MAX_TOOL_CYCLES = 3

class AgentState(TypedDict):
    user_question:      str
    is_clinical:        bool
    messages:           list[dict[str, Any]]
    tool_call_count:    int
    formatted_response: str
    error:              str


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are AskDoc, a helpful AI assistant that answers questions based on uploaded PDF documents.
You have access to a database of indexed PDF documents.

You have two tools:

1. search_chunks — call this FIRST for every question.
   It searches the PDF database semantically and returns the most relevant text chunks.

2. get_adjacent_chunks — call this ONLY if a retrieved chunk cuts off mid-sentence
   or you need surrounding context to complete the answer. Pass the chunk_id from
   the search result and the direction (prev, next, or both).

Rules:
- Always call search_chunks first.
- Only call get_adjacent_chunks if the search results are clearly incomplete.
- Do not call any tool more than once with the same arguments.
- After gathering enough context, stop calling tools and write your final answer.
- Base your answer only on the retrieved chunks. Do not hallucinate.
- Format the final answer in clear, readable Markdown.
- Never include references, citations, source filenames, page numbers, chunk IDs, or internal retrieval details in your answer."""


# ============================================================
# Graph nodes
# ============================================================

def check_intent(state: AgentState) -> AgentState:
    """Prepare messages for the generate node. All questions are answered."""
    question = state.get("user_question", "").strip()
    if not question:
        return {**state, "error": "Empty question.", "formatted_response": "Please ask a question."}

    return {
        **state,
        "is_clinical":     True,
        "tool_call_count": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
    }


def route_after_intent(state: AgentState) -> Literal["generate", "done"]:
    return "generate" if state.get("is_clinical", True) else "done"


def generate(state: AgentState) -> AgentState:
    """
    Core agent node. Sends messages + tool definitions directly to Groq API.
    No LangChain bind_tools — tools are passed as raw dicts.
    The LLM either returns a tool_call or a plain text answer.
    """
    messages = state.get("messages", [])

    @_retry_decorator
    def _call():
        return _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=2048,
        )

    try:
        response = _call()
        msg      = response.choices[0].message

        if msg.tool_calls:
            updated_messages = list(messages) + [{
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }]
            logger.info("generate: LLM requested %d tool call(s).", len(msg.tool_calls))
        else:
            updated_messages = list(messages) + [{
                "role":    "assistant",
                "content": msg.content or "",
            }]
            logger.info("generate: LLM produced final answer.")

        return {**state, "messages": updated_messages}

    except Exception as e:
        logger.exception("generate failed: %s", e)
        return {**state, "error": str(e)}


def route_after_generate(state: AgentState) -> Literal["execute_tool", "format_response"]:
    """
    If the last assistant message has tool_calls AND we're under the cap, execute them.
    Otherwise go straight to format_response.
    """
    if state.get("error"):
        return "format_response"

    messages = state.get("messages", [])
    if not messages:
        return "format_response"

    last         = messages[-1]
    has_tools    = bool(last.get("tool_calls"))
    under_limit  = state.get("tool_call_count", 0) < MAX_TOOL_CYCLES

    if has_tools and under_limit:
        return "execute_tool"
    return "format_response"


def execute_tool(state: AgentState) -> AgentState:
    """
    Executes all tool calls from the last assistant message.
    Appends one tool message per call back into the message history.
    No LangChain ToolNode — plain Python function dispatch.
    """
    messages = list(state.get("messages", []))
    last     = messages[-1]

    for tc in last.get("tool_calls", []):
        name      = tc["function"]["name"]
        arguments = tc["function"]["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        logger.info("execute_tool: calling %s with %s", name, arguments)
        result = _execute_tool(name, arguments)

        messages.append({
            "role":         "tool",
            "tool_call_id": tc["id"],
            "name":         name,
            "content":      result,
        })

    return {
        **state,
        "messages":        messages,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
    }


def format_response(state: AgentState) -> AgentState:
    """
    Extracts the final answer from the message history.
    If the last assistant message is plain text — use it directly.
    If the loop ended on a tool message — ask the LLM to synthesise a final answer.
    """
    if state.get("error"):
        return {**state, "formatted_response": "Something went wrong. Please try again."}

    if state.get("formatted_response"):
        return state

    messages = state.get("messages", [])
    if not messages:
        return {**state, "formatted_response": "No relevant information found in the uploaded documents."}

    last = messages[-1]

    if last.get("role") == "assistant" and not last.get("tool_calls"):
        return {**state, "formatted_response": (last.get("content") or "").strip()}

    summary_messages = list(messages) + [{
        "role":    "user",
        "content": (
            "Based on all the document chunks you retrieved above, "
            "write the final answer in clear, readable Markdown. "
            "Do not call any more tools. Do not mention chunk IDs, filenames, page numbers, or any internal identifiers."
        ),
    }]

    @_retry_decorator
    def _summarise():
        r = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=summary_messages,
            temperature=0,
            max_tokens=2048,
        )
        return r.choices[0].message.content.strip()

    try:
        return {**state, "formatted_response": _summarise()}
    except Exception as e:
        logger.exception("format_response failed: %s", e)
        return {**state, "formatted_response": "Something went wrong while formatting the response."}


# ============================================================
# Graph assembly
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("check_intent",    check_intent)
workflow.add_node("generate",        generate)
workflow.add_node("execute_tool",    execute_tool)
workflow.add_node("format_response", format_response)

workflow.add_edge(START, "check_intent")

workflow.add_conditional_edges(
    "check_intent", route_after_intent,
    {"generate": "generate", "done": "format_response"}
)

workflow.add_conditional_edges(
    "generate", route_after_generate,
    {"execute_tool": "execute_tool", "format_response": "format_response"}
)

workflow.add_edge("execute_tool",    "generate")
workflow.add_edge("format_response", END)

graph = workflow.compile()
