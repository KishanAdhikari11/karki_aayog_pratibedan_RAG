# Karki Aayog Pratibedan RAG
 An AI-powered question answering system for the Karki Investigation Commission Report (जाँचबुझ आयोगको प्रतिवेदन) — a 907-page Nepali government document investigating the 2025 Gen Z protests.

---



## Architecture

### Ingestion Pipeline

```
PDF → JSON (structured by page) → Text Cleaning → Chunking → Embedding → PostgreSQL (pgvector)
```

- **PDF Extraction**: The source document is a 900-page Preeti-encoded Nepali PDF, extracted and converted to structured JSON with page numbers and TOC metadata
- **TOC Metadata**: Each chunk is enriched with structured metadata — report title, भाग (part), परिच्छेद (chapter), and section ID/title — mapped from a separate TOC file
- **Text Splitting**: `RecursiveCharacterTextSplitter` with Nepali-aware separators (`।`, `। `) and chunk size of 1000 tokens
- **Embedding Model**: `intfloat/multilingual-e5-large` via FastEmbed — chosen for strong Devanagari script support
- **Storage**: PostgreSQL with `pgvector` extension for vector similarity search
- **Batch Ingestion**: Chunks are embedded and inserted in batches of 50 pages to manage memory efficiently

### Retrieval Pipeline (Hybrid Search)

```
User Query → Query Embedding + Keyword Expansion → Vector Search + Full-Text Search → RRF Fusion → Reranking → LLM
```


#### 1. Vector Search
Cosine similarity search via `pgvector` using the multilingual-e5-large embedding model.

#### 2. Full-Text Search
PostgreSQL `tsvector` + `plainto_tsquery` with `'simple'` dictionary for language-agnostic keyword matching on the `text` column.

#### 3. Reciprocal Rank Fusion (RRF)
Results from both searches are merged using RRF scoring:
```
score(d) = Σ 1 / (K + rank(d))    where K = 60
```
This balances semantic relevance (vector) with exact keyword matches (full-text) without requiring score normalization.

#### 4. Lightweight Reranker
After RRF fusion, chunks are reranked by token overlap between the query and chunk text (with numeral normalization applied). Acts as a fast cross-encoder substitute — can be swapped for a real cross-encoder model for higher accuracy.

### Generation Pipeline (LangGraph)

```
ChatState → with_context node → route_from_context → node_llm / no_context → Answer
```

Built with **LangGraph** using a typed state graph:

- **`with_context`**: Runs hybrid search + reranking, populates `retrieved_chunks` and `context`
- **`route_from_context`**: Routes to LLM if context found, otherwise returns a graceful no-context response
- **`node_llm`**: Invokes Gemini 2.5 Flash with a structured system prompt enforcing citation, language matching, and no hallucination
- **`ChatContext`**: Dataclass injected at runtime containing the DB session, LLM, and embedding model



## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + SQLAlchemy (async) |
| Database | PostgreSQL + pgvector |
| Embeddings | FastEmbed (`intfloat/multilingual-e5-large`) |
| Vector Search | pgvector cosine distance |
| Full-Text Search | PostgreSQL tsvector |
| Orchestration | LangGraph |
| LLM | Gemini 2.5 Flash (via `init_chat_model`) |
| Package Manager | uv |

---

## Setup

```bash
# Clone and install
git clone https://github.com/KishanAdhikari11/karki_aayog_pratibedan_RAG
cd karki_aayog_pratibedan_RAG
uv sync

# Run
uv run uvicorn main:app --reload 
```
or Run via Makefile
```bash
make dev 
```

### Environment Variables

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
GOOGLE_API_KEY=your_gemini_api_key
```

### Ingest the Document

```bash
curl -X POST "http://localhost:8000/internal/ingest-json?json_file=document/data.json&toc_file=document/toc.json"
```

 Ingestion runs once and takes ~45 minutes on CPU for the full 900-page document. Data persists in PostgreSQL after ingestion.

---

## API

### `POST /chat`

```bash
curl -X POST "http://localhost:8000/chat?query=what does report says about TOB?"
```

```json
{
  "answer": "...",
  "retrieved_chunks": [...]
}
```

---

## Limitations

- Table data retrieval is limited — tables are embedded as plain text
- Images and diagrams in the PDF are not processed
- Reranker is token-overlap based — a cross-encoder would improve accuracy
- Embedding generation requires CPU time; GPU recommended for production

---
