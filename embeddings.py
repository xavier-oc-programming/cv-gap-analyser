"""
embeddings.py — embedding and Pinecone operations.

No classes. All state is module-level (lazy-loaded globals).
Pinecone fails gracefully — never raises on missing credentials.

Requires PINECONE_API_KEY set as environment variable.
Free tier at pinecone.io — starter plan, no credit card required.
Index: cv-gap-analyser-jobs, dimension: 384, metric: cosine
"""
import os
from datetime import datetime, timezone

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    MAX_TEXT_LENGTH,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    TOP_K_SIMILAR,
    STRONG_MATCH_THRESHOLD,
    GOOD_MATCH_THRESHOLD,
    PARTIAL_MATCH_THRESHOLD,
)

_embed_model = None
_pinecone_index = None


def get_embed_model():
    """Lazy-load all-MiniLM-L6-v2 on first call. Cached globally."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def embed_text(text: str) -> list[float]:
    """
    Embed text using all-MiniLM-L6-v2.
    Returns 384-dimensional vector.
    Truncates to MAX_TEXT_LENGTH before embedding.
    """
    text = text[:MAX_TEXT_LENGTH]
    model = get_embed_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def _match_label(score: float) -> str:
    if score >= STRONG_MATCH_THRESHOLD:
        return "Strong match"
    elif score >= GOOD_MATCH_THRESHOLD:
        return "Good match"
    elif score >= PARTIAL_MATCH_THRESHOLD:
        return "Partial match"
    return "Weak match"


def get_pinecone_index():
    """
    Lazy-load Pinecone client and index on first call.
    Returns None if PINECONE_API_KEY not set — callers must handle None.
    """
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"[embeddings] Pinecone init failed: {e}")
        return None

    return _pinecone_index


def store_job_description(
    job_id: str,
    title: str,
    company: str,
    text: str,
    source_url: str = None,
) -> dict:
    """
    Embed and store a job description in Pinecone.

    Args:
        job_id: unique identifier (slug e.g. 'accenture-mlops-2026')
        title: job title
        company: company name
        text: full job description text
        source_url: original URL if fetched from web (optional)

    Pinecone metadata:
      {
        "title": str,
        "company": str,
        "text_preview": first 300 chars,
        "word_count": int,
        "source_url": str or None,
        "stored_at": ISO timestamp
      }

    Returns: {"job_id": str, "stored": bool}
    Silently returns {"job_id": job_id, "stored": False} if Pinecone
    not configured — never raises on missing credentials.
    """
    index = get_pinecone_index()
    if index is None:
        return {"job_id": job_id, "stored": False}

    try:
        vector = embed_text(text)
        metadata = {
            "title": title,
            "company": company,
            "text_preview": text[:300],
            "word_count": len(text.split()),
            "source_url": source_url or "",
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        index.upsert(vectors=[{"id": job_id, "values": vector, "metadata": metadata}])
        return {"job_id": job_id, "stored": True}
    except Exception as e:
        print(f"[embeddings] store_job_description failed: {e}")
        return {"job_id": job_id, "stored": False}


def find_similar_jobs(cv_text: str, top_k: int = TOP_K_SIMILAR) -> list[dict]:
    """
    Find the most semantically similar job descriptions to a CV.

    Returns list of dicts:
      [
        {
          "job_id": str,
          "title": str,
          "company": str,
          "similarity_score": float,
          "match_label": str,
          "text_preview": str,
          "source_url": str or None,
          "stored_at": str
        }
      ]
    Returns empty list if Pinecone not configured.
    """
    index = get_pinecone_index()
    if index is None:
        return []

    try:
        vector = embed_text(cv_text)
        results = index.query(vector=vector, top_k=top_k, include_metadata=True)
        output = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            score = round(float(match.get("score", 0)), 4)
            output.append({
                "job_id": match["id"],
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "similarity_score": score,
                "match_label": _match_label(score),
                "text_preview": meta.get("text_preview", ""),
                "source_url": meta.get("source_url") or None,
                "stored_at": meta.get("stored_at", ""),
            })
        return output
    except Exception as e:
        print(f"[embeddings] find_similar_jobs failed: {e}")
        return []
