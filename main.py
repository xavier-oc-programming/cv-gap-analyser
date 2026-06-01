"""
main.py — FastAPI application for CV Gap Analyser.

Routes:
  GET  /                      — welcome message
  GET  /health                — health check
  POST /match/text            — CV text + JD text
  POST /match/url             — CV text + JD URL
  POST /match/pdf             — CV PDF + JD PDF
  POST /match/cv-text-jd-pdf  — CV text + JD PDF
  POST /match/cv-text-jd-url  — CV text form field + JD URL form field
  POST /fetch-url             — preview URL extraction
  GET  /api/similar-jobs      — find similar jobs by CV text
  GET  /api/job-library       — list built-in library
  POST /api/add-job           — add job by text
  POST /api/add-job-url       — add job by URL
  GET  /api/skills            — extract skills from text
"""
import os
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import embeddings as emb
import scorer
import url_fetcher
import pdf_extractor
from skill_extractor import extract_skills
from config import JOB_LIBRARY_DIR, UPLOAD_DIR


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MatchTextRequest(BaseModel):
    cv_text: str = Field(..., min_length=100, description="Full CV text")
    jd_text: str = Field(..., min_length=50, description="Job description text")
    jd_title: Optional[str] = Field(None, description="Job title (optional)")
    jd_company: Optional[str] = Field(None, description="Company name (optional)")
    store_job: bool = Field(False, description="Store this JD in Pinecone library")


class MatchURLRequest(BaseModel):
    cv_text: str = Field(..., min_length=100, description="Full CV text")
    jd_url: str = Field(..., description="Job posting URL")
    jd_title: Optional[str] = Field(None)
    jd_company: Optional[str] = Field(None)
    store_job: bool = Field(False)


class MatchResponse(BaseModel):
    match: dict
    rouge: dict
    skills: dict
    similar_jobs: list
    recommendations: list[str]
    summary: str
    jd_source_url: Optional[str] = None
    processing_time_ms: int


class URLFetchResponse(BaseModel):
    text: str
    url: str
    word_count: int
    extraction_method: str
    truncated: bool


class HealthResponse(BaseModel):
    status: str
    embedding_model_loaded: bool
    pinecone_connected: bool
    jobs_indexed: int


class FetchURLRequest(BaseModel):
    url: str


class AddJobRequest(BaseModel):
    job_id: str
    title: str
    company: str
    text: str
    source_url: Optional[str] = None


class AddJobURLRequest(BaseModel):
    job_id: str
    title: str
    company: str
    url: str


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_state = {
    "model_loaded": False,
    "pinecone_connected": False,
    "jobs_indexed": 0,
}


def _load_job_library():
    """Seed Pinecone with built-in job library if index is empty."""
    index = emb.get_pinecone_index()
    if index is None:
        print("[startup] Pinecone not configured — skipping job library indexing.")
        return

    try:
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        _state["jobs_indexed"] = total
        _state["pinecone_connected"] = True

        if total > 0:
            print(f"[startup] Pinecone index already has {total} vectors — skipping seed.")
            return

        print("[startup] Seeding Pinecone with job library...")
        lib_dir = JOB_LIBRARY_DIR
        job_files = [
            ("data-science-mlops-engineer", "Data Science / MLOps Engineer", "Global Consulting Firm (Madrid)", "data-science-mlops-engineer.txt"),
            ("ml-engineer-aws", "Machine Learning Engineer", "Technology Company", "ml-engineer-aws.txt"),
            ("data-engineer-azure", "Data Engineer", "Financial Services Firm", "data-engineer-azure.txt"),
            ("ai-engineer-llm", "AI Engineer — LLMs and GenAI", "AI Product Company", "ai-engineer-llm.txt"),
            ("backend-python-engineer", "Senior Python Backend Engineer", "SaaS Company", "backend-python-engineer.txt"),
        ]
        count = 0
        for job_id, title, company, filename in job_files:
            path = lib_dir / filename
            if path.exists():
                text = path.read_text(encoding="utf-8")
                result = emb.store_job_description(job_id, title, company, text)
                if result["stored"]:
                    count += 1
                    print(f"[startup] Indexed: {title}")
        _state["jobs_indexed"] = count
        print(f"[startup] Job library seeded: {count} jobs indexed.")
    except Exception as e:
        print(f"[startup] Job library seeding failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)

    def _background_init():
        print("[startup] Loading embedding model...")
        emb.get_embed_model()
        _state["model_loaded"] = True
        print("[startup] Embedding model loaded.")
        _load_job_library()

    thread = threading.Thread(target=_background_init, daemon=True)
    thread.start()

    print("[startup] CV Gap Analyser starting. Model loading in background.")
    yield
    print("[shutdown] CV Gap Analyser shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CV Gap Analyser",
    description=(
        "Semantic CV vs job description analysis. Identifies missing skills, "
        "computes match score, measures keyword overlap with ROUGE, and searches "
        "a job description library via Pinecone. Accepts text, PDF, or URL for "
        "the job description."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analysis_response(result: dict, t0: float) -> MatchResponse:
    elapsed_ms = int((time.time() - t0) * 1000)
    return MatchResponse(
        match=result["match"],
        rouge=result["rouge"],
        skills=result["skills"],
        similar_jobs=result["similar_jobs"],
        recommendations=result["recommendations"],
        summary=result["summary"],
        jd_source_url=result.get("jd_source_url"),
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def root():
    html_path = Path("templates/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return JSONResponse(content={
        "message": "CV Gap Analyser API",
        "docs": "/docs",
        "health": "/health",
    })


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    index = emb.get_pinecone_index()
    connected = index is not None
    jobs = _state.get("jobs_indexed", 0)
    if connected and jobs == 0:
        try:
            stats = index.describe_index_stats()
            jobs = stats.get("total_vector_count", 0)
            _state["jobs_indexed"] = jobs
        except Exception:
            pass
    return HealthResponse(
        status="ok",
        embedding_model_loaded=_state["model_loaded"],
        pinecone_connected=connected,
        jobs_indexed=jobs,
    )


@app.post("/match/text", response_model=MatchResponse, tags=["Match"])
def match_text(req: MatchTextRequest):
    t0 = time.time()
    result = scorer.full_analysis(
        cv_text=req.cv_text,
        jd_text=req.jd_text,
        jd_title=req.jd_title,
        jd_company=req.jd_company,
    )
    if req.store_job and emb.get_pinecone_index() is not None:
        if req.jd_title and req.jd_company:
            import uuid
            job_id = f"custom-{uuid.uuid4().hex[:8]}"
            emb.store_job_description(job_id, req.jd_title, req.jd_company, req.jd_text)
    return _analysis_response(result, t0)


@app.post("/match/url", response_model=MatchResponse, tags=["Match"])
def match_url(req: MatchURLRequest):
    t0 = time.time()
    try:
        fetched = url_fetcher.fetch_job_from_url(req.jd_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = scorer.full_analysis(
        cv_text=req.cv_text,
        jd_text=fetched["text"],
        jd_title=req.jd_title,
        jd_company=req.jd_company,
        jd_source_url=req.jd_url,
    )

    if req.store_job and req.jd_title and req.jd_company:
        import uuid
        job_id = f"url-{uuid.uuid4().hex[:8]}"
        emb.store_job_description(
            job_id, req.jd_title, req.jd_company,
            fetched["text"], source_url=req.jd_url
        )

    return _analysis_response(result, t0)


@app.post("/match/pdf", response_model=MatchResponse, tags=["Match"])
async def match_pdf(
    cv_file: UploadFile = File(...),
    jd_file: UploadFile = File(...),
):
    t0 = time.time()
    try:
        cv_bytes = await cv_file.read()
        jd_bytes = await jd_file.read()
        cv_data = pdf_extractor.extract_text_from_bytes(cv_bytes)
        jd_data = pdf_extractor.extract_text_from_bytes(jd_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = scorer.full_analysis(cv_text=cv_data["text"], jd_text=jd_data["text"])
    return _analysis_response(result, t0)


@app.post("/match/cv-text-jd-pdf", response_model=MatchResponse, tags=["Match"])
async def match_cv_text_jd_pdf(
    cv_text: str = Form(...),
    jd_file: UploadFile = File(...),
):
    t0 = time.time()
    if len(cv_text) < 100:
        raise HTTPException(status_code=422, detail="cv_text must be at least 100 characters.")
    try:
        jd_bytes = await jd_file.read()
        jd_data = pdf_extractor.extract_text_from_bytes(jd_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = scorer.full_analysis(cv_text=cv_text, jd_text=jd_data["text"])
    return _analysis_response(result, t0)


@app.post("/match/cv-text-jd-url", response_model=MatchResponse, tags=["Match"])
async def match_cv_text_jd_url(
    cv_text: str = Form(...),
    jd_url: str = Form(...),
):
    t0 = time.time()
    if len(cv_text) < 100:
        raise HTTPException(status_code=422, detail="cv_text must be at least 100 characters.")
    try:
        fetched = url_fetcher.fetch_job_from_url(jd_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = scorer.full_analysis(
        cv_text=cv_text,
        jd_text=fetched["text"],
        jd_source_url=jd_url,
    )
    return _analysis_response(result, t0)


@app.post("/fetch-url", response_model=URLFetchResponse, tags=["Utilities"])
def fetch_url(req: FetchURLRequest):
    try:
        result = url_fetcher.fetch_job_from_url(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return URLFetchResponse(**result)


@app.post("/api/preview-pdf", tags=["Utilities"])
async def preview_pdf(file: UploadFile = File(...)):
    """Extract and return a text preview from a PDF upload."""
    try:
        pdf_bytes = await file.read()
        data = pdf_extractor.extract_text_from_bytes(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "text": data["text"],
        "word_count": data["word_count"],
        "page_count": data["page_count"],
        "truncated": data["truncated"],
    }


@app.post("/match/pdf-cv-url-jd", response_model=MatchResponse, tags=["Match"])
async def match_pdf_cv_url_jd(
    cv_file: UploadFile = File(...),
    jd_url: str = Form(...),
):
    """CV uploaded as PDF + job description fetched from URL."""
    t0 = time.time()
    try:
        cv_bytes = await cv_file.read()
        cv_data = pdf_extractor.extract_text_from_bytes(cv_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(cv_data["text"]) < 100:
        raise HTTPException(status_code=422, detail="CV PDF contains too little text.")

    try:
        fetched = url_fetcher.fetch_job_from_url(jd_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = scorer.full_analysis(
        cv_text=cv_data["text"],
        jd_text=fetched["text"],
        jd_source_url=jd_url,
    )
    return _analysis_response(result, t0)


@app.post("/match/pdf-cv-text-jd", response_model=MatchResponse, tags=["Match"])
async def match_pdf_cv_text_jd(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(...),
):
    """CV uploaded as PDF + job description pasted as text."""
    t0 = time.time()
    try:
        cv_bytes = await cv_file.read()
        cv_data = pdf_extractor.extract_text_from_bytes(cv_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(cv_data["text"]) < 100:
        raise HTTPException(status_code=422, detail="CV PDF contains too little text.")
    if len(jd_text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Job description text is too short.")

    result = scorer.full_analysis(cv_text=cv_data["text"], jd_text=jd_text)
    return _analysis_response(result, t0)


@app.get("/api/similar-jobs", tags=["Job Library"])
def similar_jobs(cv_text: str):
    return emb.find_similar_jobs(cv_text)


@app.get("/api/job-library", tags=["Job Library"])
def job_library():
    lib_dir = JOB_LIBRARY_DIR
    entries = [
        {"job_id": "data-science-mlops-engineer", "title": "Data Science / MLOps Engineer", "company": "Global Consulting Firm (Madrid)", "filename": "data-science-mlops-engineer.txt"},
        {"job_id": "ml-engineer-aws", "title": "Machine Learning Engineer", "company": "Technology Company", "filename": "ml-engineer-aws.txt"},
        {"job_id": "data-engineer-azure", "title": "Data Engineer", "company": "Financial Services Firm", "filename": "data-engineer-azure.txt"},
        {"job_id": "ai-engineer-llm", "title": "AI Engineer — LLMs and GenAI", "company": "AI Product Company", "filename": "ai-engineer-llm.txt"},
        {"job_id": "backend-python-engineer", "title": "Senior Python Backend Engineer", "company": "SaaS Company", "filename": "backend-python-engineer.txt"},
    ]
    result = []
    for e in entries:
        path = lib_dir / e["filename"]
        word_count = 0
        if path.exists():
            text = path.read_text(encoding="utf-8")
            word_count = len(text.split())
        result.append({
            "job_id": e["job_id"],
            "title": e["title"],
            "company": e["company"],
            "word_count": word_count,
        })
    return result


@app.post("/api/add-job", tags=["Job Library"])
def add_job(req: AddJobRequest):
    return emb.store_job_description(
        req.job_id, req.title, req.company, req.text, req.source_url
    )


@app.get("/api/job-library-text/{job_id}", tags=["Job Library"])
def job_library_text(job_id: str):
    """Return the full text of a built-in library job description."""
    lib_dir = JOB_LIBRARY_DIR
    filemap = {
        "data-science-mlops-engineer": "data-science-mlops-engineer.txt",
        "ml-engineer-aws": "ml-engineer-aws.txt",
        "data-engineer-azure": "data-engineer-azure.txt",
        "ai-engineer-llm": "ai-engineer-llm.txt",
        "backend-python-engineer": "backend-python-engineer.txt",
    }
    filename = filemap.get(job_id)
    if not filename:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not in library.")
    path = lib_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Library file not found: {filename}")
    text = path.read_text(encoding="utf-8")
    return {"job_id": job_id, "text": text, "word_count": len(text.split())}


@app.post("/api/add-job-url", tags=["Job Library"])
def add_job_url(req: AddJobURLRequest):
    try:
        fetched = url_fetcher.fetch_job_from_url(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = emb.store_job_description(
        req.job_id, req.title, req.company,
        fetched["text"], source_url=req.url
    )
    return {
        "job_id": result["job_id"],
        "stored": result["stored"],
        "word_count": fetched["word_count"],
        "extraction_method": fetched["extraction_method"],
    }


@app.get("/api/skills", tags=["Utilities"])
def skills(text: str):
    return extract_skills(text)


# ---------------------------------------------------------------------------
# Demo frontend
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=HTMLResponse, tags=["Frontend"])
def demo():
    html_path = Path("templates/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Demo frontend not found.</h1>", status_code=404)
