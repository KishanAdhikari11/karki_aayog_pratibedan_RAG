from typing import Optional, Any
import re

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select, text
from models import Embedding
from utils.util import get_logger, generate_embeddings
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from dataclasses import dataclass


logger = get_logger()


NEPALI_TO_ARABIC = str.maketrans("०१२३४५६७८९", "0123456789")
ARABIC_TO_NEPALI = str.maketrans("0123456789", "०१२३४५६७८९")


def normalize_numerals(text: str) -> str:
    """Return text with BOTH Nepali and Arabic numeral forms added."""
    arabic = text.translate(NEPALI_TO_ARABIC)
    nepali = text.translate(ARABIC_TO_NEPALI)
    return f"{text} {arabic} {nepali}"


@dataclass
class ChatState:
    query: str
    top_k: Optional[int]
    embedding_source: Optional[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    context: Optional[str]


@dataclass
class ChatContext:
    db: Any
    llm: Any
    model: Any


def build_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", {})
        page = source.get("page_no", "")
        section = source.get("section_id", "")
        label = (
            f"[Chunk {i} | Page {page}"
            + (f" | Section {section}" if section else "")
            + "]"
        )
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


async def hybrid_search(
    query: str,
    db,
    model,
    top_k: int = 20,
) -> list[Any]:
    query_embedding = await generate_embeddings([query], model)
    if not query_embedding:
        raise ValueError("Failed to generate query embedding")
    query_embedding = query_embedding[0]

    vector_stmt = (
        select(Embedding)
        .order_by(Embedding.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    vector_result = await db.execute(vector_stmt)
    vector_rows: list[Embedding] = vector_result.scalars().all()

    normalized_query = normalize_numerals(query)
    keyword_stmt = (
        select(Embedding)
        .where(
            text(
                "to_tsvector('simple', embeddings.text) "
                "@@ plainto_tsquery('simple', :q)"
            )
        )
        .limit(top_k)
        .params(q=normalized_query)
    )
    try:
        keyword_result = await db.execute(keyword_stmt)
        keyword_rows: list[Embedding] = keyword_result.scalars().all()
    except Exception as e:
        logger.warning(f"Keyword search failed, falling back to vector only: {e}")
        keyword_rows = []

    K = 60  # RRF constant
    scores: dict[int, float] = {}
    id_to_row: dict[int, Embedding] = {}

    for rank, row in enumerate(vector_rows):
        scores[row.id] = scores.get(row.id, 0) + 1 / (K + rank + 1)
        id_to_row[row.id] = row

    for rank, row in enumerate(keyword_rows):
        scores[row.id] = scores.get(row.id, 0) + 1 / (K + rank + 1)
        id_to_row[row.id] = row

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [id_to_row[i] for i in sorted_ids[:top_k]]


def rerank(query: str, rows: list[Any], top_k: int) -> list[Any]:
    """
    Lightweight reranker: boost chunks whose text contains query tokens.
    Replace with a real cross-encoder if you need higher accuracy.
    """
    query_tokens = set(re.findall(r"\w+", normalize_numerals(query).lower()))

    def score(row):
        chunk_tokens = set(re.findall(r"\w+", normalize_numerals(row.text).lower()))
        overlap = len(query_tokens & chunk_tokens)
        return overlap

    return sorted(rows, key=score, reverse=True)[:top_k]


async def no_context(state: ChatState) -> ChatState:
    state.answer = "Sorry, I couldn't find any relevant information in the report to answer your query."
    return state


async def with_context(state: ChatState, runtime: Runtime[ChatContext]) -> ChatState:
    db = runtime.context.db
    model = runtime.context.model
    if not db or not model:
        raise ValueError("Database session or model not found in runtime context")

    top_k = state.top_k or 8
    if top_k is None:
        raise ValueError

    candidate_k = max(top_k * 3, 25)
    rows = await hybrid_search(state.query, db, model, top_k=candidate_k)

    if not rows:
        state.context = None
        state.retrieved_chunks = []
        return state

    rows = rerank(state.query, rows, top_k=top_k)

    state.retrieved_chunks = [
        {"text": row.text, "source": row.embedding_source} for row in rows
    ]
    state.context = build_context(state.retrieved_chunks)
    return state


async def node_llm(state: ChatState, runtime: Runtime[ChatContext]) -> ChatState:

    SYSTEM_PROMPT = ChatPromptTemplate.from_template(
        """You are the official AI Assistant for the Karki Investigation Commission Report (जाँचबुझ आयोगको प्रतिवेदन).

## YOUR ROLE
Answer questions **strictly based on the context chunks below**. Each chunk is labelled with its Page number and Section so you can cite sources precisely.

## RULES
1. **Language:** Reply in the same language the user uses (Nepali or English).
2. **Always cite:** Every factual claim must include (Page X, Section Y).
3. **Numbers & dates:** The report uses Nepali numerals (२३, २४). Bhadra 23 = September 8, Bhadra 24 = September 9.
4. **Piece together facts:** If a number (deaths, injuries) is spread across multiple chunks, combine them and state the total clearly.
5. **Be explicit about gaps:** If a specific figure is truly absent from all chunks, say exactly which pages were checked and what related information WAS found.
6. **Never hallucinate:** Do not invent numbers, names, or dates not present in the context.
7. **Structured answers:** For casualty/event questions, use a clear format:
   - Date / Location / What happened / Number affected / Source (Page, Section)

## CONTEXT (from the report):
{context}

## USER QUESTION:
{query}

## ANSWER:"""
    )
    llm = runtime.context.llm
    if not llm:
        raise ValueError("LLM chat model is required")

    prompt = ChatPromptTemplate.from_messages(["system", SYSTEM_PROMPT])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    result = await chain.ainvoke({"context": state.context, "query": state.query})
    state.answer = result
    return state


def route_from_context(state: ChatState) -> str:
    return "node_llm" if state.context else "no_context"


workflow = StateGraph(state_schema=ChatState, context_schema=ChatContext)
workflow.add_node("with_context", with_context)
workflow.add_node("node_llm", node_llm)
workflow.add_node("no_context", no_context)

workflow.add_edge(START, "with_context")
workflow.add_conditional_edges(
    "with_context",
    route_from_context,
    {"node_llm": "node_llm", "no_context": "no_context"},
)
workflow.add_edge("node_llm", END)
workflow.add_edge("no_context", END)

chatbot = workflow.compile()
