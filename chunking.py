import re
from typing import Optional


def clean_ocr_noise(text: str) -> str:
    """Clean common OCR artifacts from Nepali text."""
    text = re.sub(r"  +", " ", text) #remove one or more space
    
    text = text.replace("|", "।") 
    # Remove trailing page numbers like \n३ or \n3
    text = re.sub(r"\n\d+\s*$", "", text.strip())
    # Remove standalone digits on their own line (page number artifacts)
    text = re.sub(r"^\d+\s*\n", "", text)
    # Normalize unicode whitespace
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text.strip()


def split_nepali_sentences(text: str) -> list[str]:
    """
    Split Nepali text into sentences on purna viram (।).
    Handles cases where OCR may have dropped or misplaced markers.
    """
    parts = re.split(r"(।)", text)

    sentences = []
    current = ""
    for part in parts:
        current += part
        if part == "।":
            s = current.strip()
            if len(s) > 15:  # skip tiny fragments / noise
                sentences.append(s)
            current = ""

    # Catch trailing text without a purna viram (common at page end)
    if current.strip() and len(current.strip()) > 15:
        sentences.append(current.strip())

    return sentences


def chunk_page(
    page_no: int,
    content: str,
    sentences_per_chunk: int = 4,
    overlap: int = 1,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Chunk a single page's content into overlapping sentence windows.

    Args:
        page_no: Page number from source document
        content: Raw OCR text of the page
        sentences_per_chunk: How many sentences per chunk (4 works well for
                             long Nepali legal sentences ~150-250 words)
        overlap: Sentences shared between consecutive chunks (prevents
                 context loss at chunk boundaries)
        metadata: TOC metadata dict to attach to each chunk's source field

    Returns:
        List of chunk dicts ready for embedding and DB storage
    """
    content = clean_ocr_noise(content)
    sentences = split_nepali_sentences(content)

    if not sentences:
        return []

    chunks = []
    i = 0
    chunk_index = 0

    while i < len(sentences):
        window = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(window)

        chunk = {
            "page_no": page_no,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "sentence_count": len(window),
            "source": {
                "page": page_no,
                "chunk_index": chunk_index,
                **(metadata or {}),
            },
        }
        chunks.append(chunk)
        chunk_index += 1
        i += sentences_per_chunk - overlap

    return chunks


def chunk_all_pages(
    pages: list[dict],
    toc_processor=None,
    sentences_per_chunk: int = 4,
    overlap: int = 1,
) -> list[dict]:
    
    all_chunks = []

    for page in pages:
        page_no = page.get("page_no","")
        content = page.get("content", "")

        if not content or not content.strip():
            continue

        metadata = {}
        if toc_processor is not None:
            metadata = toc_processor.get_metadata_for_page(page_no)

        page_chunks = chunk_page(
            page_no=page_no,
            content=content,
            sentences_per_chunk=sentences_per_chunk,
            overlap=overlap,
            metadata=metadata,
        )
        all_chunks.extend(page_chunks)

    return all_chunks
