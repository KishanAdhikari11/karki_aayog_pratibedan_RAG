from typing import Optional, TypedDict, Any, Sequence

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select
from models import Embedding
from utils import get_logger
from langgraph.runtime import Runtime
from utils import generate_embeddings
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START,END, StateGraph


logger = get_logger()


class ChatState(TypedDict):
    prompt: ChatPromptTemplate
    query: str
    top_k: Optional[int]
    embedding_source: Optional[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    context: Optional[str]


class ChatContext(TypedDict):
   db : Any
   llm : Any
   model :Any


async def no_context(state: ChatState) -> ChatContext:
    state["answer"] = (
        "Sorry, I couldn't find any relevant information to answer your query."
    )
    return state


async def with_context(state: ChatState, runtime: Runtime[ChatContext]) -> ChatContext:
    db = runtime.context.get("db")
    model = runtime.context.get("model")
    if not db or not model:
        raise ValueError("Database session or model not found in runtime context")
    embeddings: Optional[Sequence[Embedding]] = None
    query_embedding = await generate_embeddings(
        [state["query"]], model
    )
    if not query_embedding:
        raise ValueError("Embedding of query is required")
    query_embedding = query_embedding[0]
    stmt = (
        select(Embedding)
        .order_by(Embedding.embedding.cosine_distance(query_embedding))
        .limit(state.get("top_k", 5))
    )
    result = await db.execute(stmt)
    embeddings = result.scalars().all()
    if not embeddings:
        state["context"] = None
        state["retrieved_chunks"] = []
        return state
    state["retrieved_chunks"] = [
        {"text": emb.text, "source": emb.embedding_source} for emb in embeddings
    ]
    text_context = "\n".join([embedding.text for embedding in embeddings])
    state["context"] = text_context
    return state

async def node_llm(state: ChatState,runtime: Runtime[ChatContext]):
    llm=runtime.context.get("llm")
    if not llm:
        raise ValueError("LLM chat model is required")
    
    prompt=state["prompt"]
    output_parser=StrOutputParser()
    chain= prompt | llm | output_parser
    result = await chain.ainvoke({"context": state["context"], "query": state["query"]})
    state["answer"] = result
    return state

workflow=StateGraph(state_schema=ChatState,context_schema=ChatContext)
workflow.add_node("node_llm",node_llm)
workflow.add_node("with_context",with_context)
workflow.add_node("no_context",no_context)

workflow.add_edge(START,"with_context")

def route_from_node(state:ChatState) ->str:
    return "node_llm" if state["context"] else "no_context"

workflow.add_conditional_edges(
    "with_context",
    route_from_node,
    {"node_llm": "node_llm", "no_context": "no_context"}    
)

workflow.add_edge("node_llm",END)
workflow.add_edge("no_context",END)

chatbot=workflow.compile()