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
    top_k: int =7,
):

    model = _get_model(request)
    llm = _get_llm(request)

    initial_state = make_initial_state(query=query, top_k=top_k)

    runtime_context: ChatContext = {
        "db": db,
        "llm": llm,
        "model": model,
    }

    try:
        final_state = await chatbot.ainvoke(
            input=initial_state,
            context=runtime_context,
        )
    except Exception as e:
        logger.error(f"Chat pipeline failed: {e}")
        raise HTTPException(status_code=500, detail="Chat pipeline error")

    return {
        "answer": final_state.get("answer", "Something went wrong."),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "query": query,
    }
