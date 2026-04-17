from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from models import Embedding
from retriever import hybrid_retrieve, is_nepali, is_mostly_english
from utils import get_logger

logger = get_logger()


_RETRIEVAL_SYSTEM = """You are an expert assistant for the Nepal Investigation Commission Report \
(जाँचबुझ आयोगको प्रतिवेदन) about the 2082 BS Bhadra 23-24 protests.

Rules:
- Read the provided excerpts carefully. If an excerpt contains the answer, USE IT.
- Always cite page numbers when referencing excerpts.
- Answer in the same language as the question.
- Only say you lack information if NONE of the excerpts are relevant.
- Be direct. Do not hedge if the answer is present."""


class ChatState(TypedDict):
    query: str
    nepali_query: str
    top_k: int
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    context: Optional[str]


class ChatContext(TypedDict):
    db: Any
    llm: Any
    model: Any


def _build_context(rows: list[Embedding]) -> str:
    parts = []
    for row in rows:
        source = row.embedding_source or {}
        page = source.get("page", "?")
        section = source.get("section_title") or source.get("chapter_title", "")
        header = f"[पृष्ठ {page}" + (f" — {section}" if section else "") + "]"
        parts.append(f"{header}\n{row.text}")
    return "\n\n".join(parts)


async def node_translate_and_retrieve(
    state: ChatState, runtime: Runtime[ChatContext]
) -> ChatState:
    """
    Single LLM call that:
      1. Translates the query to Nepali (for retrieval)
      2. Returns a preliminary answer if context not yet known

    Since we need the Nepali query BEFORE retrieval, we do a lightweight
    translation-only call first, then retrieve, then answer.

    Translation is cheap — we ask the LLM for just the Nepali query string.
    """
    llm = runtime.context["llm"]
    db = runtime.context["db"]
    model = runtime.context["model"]
    query = state["query"]

    if is_nepali(query) or is_mostly_english:
        nepali_query = query
    else:
        translate_messages = [
            SystemMessage(
                content=(
                    "Translate the following search query to Nepali (Devanagari script). "
                    "Output ONLY the Nepali translation — no explanation, no punctuation changes. "
                    "Transliterate proper nouns (names of people, places) into Devanagari."
                )
            ),
            HumanMessage(content=query),
        ]
        result = await llm.ainvoke(translate_messages)
        nepali_query = StrOutputParser().invoke(result).strip()
        logger.info(f"Translated query: '{query}' → '{nepali_query}'")

    state["nepali_query"] = nepali_query

    rows = await hybrid_retrieve(
        query=nepali_query,
        db=db,
        model=model,
        top_k=state["top_k"],
    )

    rows = [r for r in rows if len(r.text.strip()) > 80] or rows

    if not rows:
        state["retrieved_chunks"] = []
        state["context"] = None
        return state

    state["retrieved_chunks"] = [
        {"text": r.text, "source": r.embedding_source} for r in rows
    ]
    state["context"] = _build_context(rows)
    return state


async def node_llm(state: ChatState, runtime: Runtime[ChatContext]) -> ChatState:
    """Answer using retrieved context."""
    llm = runtime.context["llm"]

    messages = [
        SystemMessage(content=_RETRIEVAL_SYSTEM),
        HumanMessage(
            content=f"Report excerpts:\n{state['context']}\n\nQuestion: {state['query']}"
        ),
    ]
    result = await llm.ainvoke(messages)
    state["answer"] = StrOutputParser().invoke(result).strip()
    return state


async def node_no_context(state: ChatState) -> ChatState:
    state["answer"] = "माफ गर्नुहोस्, तपाईंको प्रश्नसँग सम्बन्धित जानकारी फेला परेन।"
    return state


def _route(state: ChatState) -> str:
    return "node_llm" if state.get("context") else "node_no_context"


_workflow = StateGraph(ChatState, ChatContext)
_workflow.add_node("node_translate_and_retrieve", node_translate_and_retrieve)
_workflow.add_node("node_llm", node_llm)
_workflow.add_node("node_no_context", node_no_context)

_workflow.add_edge(START, "node_translate_and_retrieve")
_workflow.add_conditional_edges(
    "node_translate_and_retrieve",
    _route,
    {"node_llm": "node_llm", "node_no_context": "node_no_context"},
)
_workflow.add_edge("node_llm", END)
_workflow.add_edge("node_no_context", END)

chatbot = _workflow.compile()


def make_initial_state(query: str, top_k: int = 8) -> ChatState:
    return ChatState(
        query=query,
        nepali_query="",
        top_k=top_k,
        retrieved_chunks=[],
        answer="",
        context=None,
    )
