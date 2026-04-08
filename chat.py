from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from dependencies import ChatState, ChatContext, chatbot
from database import get_db
from schemas import ErrorResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model


router = APIRouter(
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponseSchema,
            "description": "Rate limit exceed",
        },
    },
)


@router.post("/chat")
async def chat(query: str, db: Annotated[AsyncSession, Depends(get_db)]):
    
    initial_state: ChatState = {
        "prompt": ChatPromptTemplate.from_template(
       """You are a precise assistant. Answer ONLY using the provided context.
If the context doesn't contain enough info, say "I don't have sufficient information."
Be concise and accurate. Cite the page/source when possible.

Context:
{context}

Question: {query}
Answer:"""
        ),
        "query": query,
        "top_k": 8,
        "embedding_source": None,
        "retrieved_chunks": [],
        "answer": "",
        "context": None,
    }

    runtime_context: ChatContext = {
        "db": db,
        "llm": init_chat_model("gemini-3-flash-preview", model_provider="google_genai"),
        "model": None,   
    }

    try:
        from main import app
        runtime_context["model"] = getattr(app.state, "model", None)
    except Exception:
        pass  # fallback - model will be None

    if not runtime_context.get("model"):
        return {"error": "Embedding model not loaded. Please restart the server."}

    final_state = await chatbot.ainvoke(
        input=initial_state,
        context=runtime_context
    )

    return {
        "answer": final_state.get("answer", "Sorry, something went wrong."),
        "retrieved_chunks": final_state.get("retrieved_chunks", [])
    }