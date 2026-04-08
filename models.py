from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.sqltypes import String, Integer
from sqlalchemy.ext.asyncio import AsyncAttrs
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from typing import Any


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    embedding_source: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)

    def __repr__(self):
        return f"<Embedding: {self.text}>"
