from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from chatbot_workflow import ChatContext, chatbot, make_initial_state
from database import get_db
from schemas import ErrorResponseSchema
from utils import get_logger

logger = get_logger()

router = APIRouter(
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponseSchema,
            "description": "Rate limit exceeded",
        },
    },
)


def _get_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    return model


def _get_llm(request: Request):
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(status_code=500, detail="LLM not loaded")
    return llm


@router.post("/chat")
async def chat(
    request: Request,
    query: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    top_k: int = 7,
):

    model = _get_model(request)
    llm = _get_llm(request)

    logger.info(f"Received query: {query[:200]}{'...' if len(query) > 200 else ''}")

    # === FORCE NO THINKING MODE FOR QWEN ===
    # This is the most reliable way that works across most Qwen3 / Qwen2.5 deployments
    no_think_query = query.strip() + "\n\n/no_think"

    # Optional: Add a very strict instruction at the beginning
    enhanced_query = (
        "Answer directly and concisely. "
        "Do not think step by step. "
        "Do not use <think> tags or any internal reasoning. "
        "Go straight to the final answer.\n\n"
        + no_think_query
    )

    # Create initial state with the modified query (graph remains untouched)
    initial_state = make_initial_state(
        query=enhanced_query,   # <-- only this part is changed
        top_k=top_k
    )

    runtime_context: ChatContext = {
        "db": db,
        "llm": llm,
        "model": model,
    }

    try:
        logger.info("Starting chatbot graph invocation...")
        final_state = await chatbot.ainvoke(
            input=initial_state,
            context=runtime_context,
        )
        logger.info("Chatbot graph completed successfully.")
    except Exception as e:
        logger.error(f"Chat pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Chat pipeline error. Please try again."
        )

    return {
        "answer": final_state.get("answer", "Something went wrong."),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "query": query,                    # return original query to client
        "used_no_think": True
    }