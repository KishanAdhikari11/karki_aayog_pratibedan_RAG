from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from utils.util import timer

from chatbot_workflow import ChatState, ChatContext, chatbot
from database import get_db
from schemas import ErrorResponseSchema
from langchain.chat_models import init_chat_model


router = APIRouter(
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponseSchema,
            "description": "Rate limit exceeded",
        },
    },
)


@router.post("/chat")
@timer
async def chat(query: str, db: Annotated[AsyncSession, Depends(get_db)]):

    initial_state = ChatState(
        query=query,
        top_k=20,
        embedding_source=None,
        retrieved_chunks=[],
        answer="",
        context=None,
    )

    runtime_context = ChatContext(
        db=db,
        llm=init_chat_model(
            "gemini-3.1-flash-lite-preview", model_provider="google_genai"
        ),
        model=None,
    )

    try:
        from main import app

        runtime_context.model = getattr(app.state, "model", None)
    except Exception:
        pass

    if not runtime_context.model:
        return {"error": "Embedding model not loaded. Please restart the server."}

    final_state = await chatbot.ainvoke(
        input=initial_state,
        context=runtime_context,
    )

    return {
        "answer": final_state.get("answer", "Sorry, something went wrong."),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
    }
