import re
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Embedding
from utils import generate_embeddings, get_logger

logger = get_logger()

_RRF_K = 60
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def is_nepali(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


def is_mostly_english(text: str) -> bool:
    nepali_chars = len(re.findall(r"[\u0900-\u097F]", text))
    return nepali_chars / max(len(text), 1) < 0.2


def _tokenize(text: str) -> list[str]:
    return text.split()


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank)


async def _fetch_neighbours(
    db: AsyncSession,
    chunks: list[Embedding],
    window: int = 1,
) -> list[Embedding]:
    """
    Simpler neighbour fetch using JSONB text comparison.
    """
    seen_ids = {c.id for c in chunks}
    extra: list[Embedding] = []

    for chunk in chunks:
        source = chunk.embedding_source or {}
        page = source.get("page")
        chunk_index = source.get("chunk_index")
        if page is None or chunk_index is None:
            continue

        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbour_index = chunk_index + offset
            if neighbour_index < 0:
                continue

            stmt = select(Embedding).where(
                and_(
                    Embedding.embedding_source["page"].astext == str(page),
                    Embedding.embedding_source["chunk_index"].astext
                    == str(neighbour_index),
                )
            )
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            if row and row.id not in seen_ids:
                seen_ids.add(row.id)
                extra.append(row)

    return extra


def _merge_consecutive(chunks: list[Embedding]) -> list[Embedding]:
    """
    Sort chunks by (page, chunk_index) and merge consecutive ones from the
    same page into a single synthetic Embedding with combined text.
    This gives the LLM a coherent narrative instead of fragments.
    """

    def sort_key(c: Embedding):
        s = c.embedding_source or {}
        return (s.get("page", 0), s.get("chunk_index", 0))

    sorted_chunks = sorted(chunks, key=sort_key)

    merged: list[Embedding] = []
    i = 0
    while i < len(sorted_chunks):
        group = [sorted_chunks[i]]
        while i + 1 < len(sorted_chunks):
            curr = sorted_chunks[i].embedding_source or {}
            nxt = sorted_chunks[i + 1].embedding_source or {}
            same_page = curr.get("page") == nxt.get("page")
            consecutive = (
                curr.get("chunk_index") is not None
                and nxt.get("chunk_index") is not None
                and nxt["chunk_index"] == curr["chunk_index"] + 1
            )
            if same_page and consecutive:
                group.append(sorted_chunks[i + 1])
                i += 1
            else:
                break
        i += 1

        if len(group) == 1:
            merged.append(group[0])
        else:
            combined_text = " ".join(c.text for c in group)
            synthetic = Embedding(
                id=group[0].id,
                text=combined_text,
                embedding=group[0].embedding,
                embedding_source=group[0].embedding_source,
            )
            merged.append(synthetic)

    return merged


async def hybrid_retrieve(
    query: str,
    db: AsyncSession,
    model: Any,
    top_k: int = 8,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    expand_neighbours: bool = True,
    neighbour_window: int = 1,
) -> list[Embedding]:
    candidate_limit = top_k * 4

    query_embedding = generate_embeddings([query], model)
    query_vec = query_embedding[0]

    stmt = (
        select(Embedding)
        .order_by(Embedding.embedding.cosine_distance(query_vec))
        .limit(candidate_limit)
    )
    result = await db.execute(stmt)
    candidates: list[Embedding] = list(result.scalars().all())

    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    corpus = [_tokenize(c.text) for c in candidates]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    vector_ranks = {c.id: i for i, c in enumerate(candidates)}
    bm25_order = sorted(
        range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True
    )
    bm25_ranks = {candidates[i].id: rank for rank, i in enumerate(bm25_order)}

    fused: dict[int, float] = {}
    for c in candidates:
        fused[c.id] = vector_weight * _rrf_score(
            vector_ranks[c.id]
        ) + bm25_weight * _rrf_score(bm25_ranks[c.id])

    top_chunks = sorted(candidates, key=lambda c: fused[c.id], reverse=True)[:top_k]

    if expand_neighbours:
        neighbours = await _fetch_neighbours(db, top_chunks, window=neighbour_window)
        all_chunks = top_chunks + neighbours
        return _merge_consecutive(all_chunks)

    return top_chunks
