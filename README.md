---
title: CV Gap Analyser
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# cv-gap-analyser

Semantic CV vs job description analysis. Identifies which skills the role requires that the CV does not demonstrate — and produces specific, actionable recommendations to close those gaps. Match score (0–100), skill gap list, ROUGE keyword overlap for ATS alignment, and a searchable job description library via Pinecone. Accepts the job description as text, PDF, or URL.

Built to solve a real problem: preparing a job application and needing a systematic way to identify what to evidence.

**Live demo → [xavier-oc-machinelearn-cv-gap-analyser.hf.space](https://xavier-oc-machinelearn-cv-gap-analyser.hf.space)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**API docs → [/docs](https://xavier-oc-machinelearn-cv-gap-analyser.hf.space/docs)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**Notebook → notebook.ipynb**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-sentence--transformers-yellow)
![Pinecone](https://img.shields.io/badge/Pinecone-vector--db-green)
![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-orange)
![trafilatura](https://img.shields.io/badge/trafilatura-URL--extraction-lightgrey)
![FastAPI](https://img.shields.io/badge/FastAPI-REST--API-teal)
![Docker](https://img.shields.io/badge/Docker-container-blue)

---

## 0. Prerequisites

- Python 3.11+
- Pinecone account (free tier at [pinecone.io](https://pinecone.io) — starter plan, no credit card required)
- `PINECONE_API_KEY` in `.env` (copy `.env.example` and fill in your key)

The application works without Pinecone — all Pinecone-dependent features (job library search, similar roles) return empty results gracefully. The core analysis (match score, ROUGE, skill extraction) works without any API keys.

## 1. Quick start

```bash
git clone https://github.com/xavier-oc-programming/cv-gap-analyser
cd cv-gap-analyser

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your PINECONE_API_KEY

python -m uvicorn main:app --reload
# Open http://localhost:8000
```

API docs at `http://localhost:8000/docs`.

Run tests:

```bash
pytest tests/ -v
```

## 2. Project structure

```
cv-gap-analyser/
├── config.py                 — all constants (thresholds, model names, paths)
├── embeddings.py             — sentence-transformers + Pinecone operations
├── skill_extractor.py        — keyword-only skill extraction (~150 terms)
├── scorer.py                 — match score, ROUGE, full_analysis()
├── pdf_extractor.py          — PyMuPDF text extraction from PDF bytes
├── url_fetcher.py            — LinkedIn guest API + trafilatura + BeautifulSoup
├── main.py                   — FastAPI app (15 routes)
├── Dockerfile                — python:3.11-slim, deployed to HuggingFace Spaces
├── startup.txt               — gunicorn startup command
├── notebook.ipynb            — full walkthrough with examples
├── README.md
├── requirements.txt
├── portfolio.yaml
├── .gitignore
├── .github/workflows/ci.yml  — GitHub Actions CI
├── job_library/              — five built-in job descriptions (committed)
│   ├── README.md
│   ├── data-science-mlops-engineer.txt
│   ├── ml-engineer-aws.txt
│   ├── data-engineer-azure.txt
│   ├── ai-engineer-llm.txt
│   └── backend-python-engineer.txt
├── templates/index.html      — demo frontend (inline CSS/JS, PDF viewer)
├── tests/test_api.py         — pytest API tests
└── uploads/                  — runtime only, gitignored
```

## 3. How it works

**Semantic matching** — CV and job description are each embedded as 384-dimensional vectors using `all-MiniLM-L6-v2`. Cosine similarity between the vectors produces the match score (0–1, displayed as 0–100). This captures conceptual alignment regardless of exact word choice: "deploying models to production cloud infrastructure" matches "MLOps and cloud-native deployment" because the vectors cluster in the same region of embedding space.

**Skill extraction** — case-insensitive whole-word keyword matching against a curated list of ~150 ML/data/software technologies covering AWS, Azure, GCP, LLM tooling, data engineering, and MLOps. An initial spaCy noun phrase pass was removed after testing — it consistently extracted job description boilerplate, company names, benefit descriptions, and non-English phrases as false-positive skills. The keyword list is deterministic and precise.

**ROUGE keyword overlap** — ROUGE treats the JD as reference and the CV as hypothesis. Higher ROUGE-L means more CV language appears verbatim in the JD. This matters for ATS: a CV can score high semantically (same concepts, different words) but fail automated screening that looks for exact keyword presence. ROUGE-L below 0.3 alongside high semantic similarity is a signal to mirror more of the JD terminology.

**Pinecone job library** — five built-in job descriptions are stored as Pinecone vectors at startup. Every CV analysis queries the library for the most semantically similar roles. New roles can be added via API.

## 4. URL extraction

The most practical input mode: paste a job posting URL and the system fetches and extracts the text.

**LinkedIn** — LinkedIn job pages require authentication and load content via JavaScript. LinkedIn exposes an unauthenticated guest API endpoint (`/jobs-guest/jobs/api/jobPosting/{job_id}`) that returns server-rendered job posting HTML without login. The URL fetcher extracts the job ID from any LinkedIn jobs URL format and hits this endpoint directly. The extraction method badge shows `linkedin_guest_api` when this path is used.

**trafilatura** — for non-LinkedIn URLs, trafilatura is the primary extractor. It was built for academic web crawling — it identifies the main content area of a web page and extracts just that text, discarding navigation, footers, sidebars, and cookie banners. It handles most company career pages cleanly without custom CSS selectors.

**Failure case** — pages that load content via JavaScript after the initial page load. The **BeautifulSoup fallback** handles these: it strips all HTML tags from the raw response body and returns noisier but usable text.

The `/fetch-url` endpoint lets you verify extraction quality before running the full analysis. The response includes an `extraction_method` field (`linkedin_guest_api`, `trafilatura`, or `beautifulsoup_fallback`).

## 5. Job library

Five built-in job descriptions in `job_library/`:

| Job ID | Title | Company |
|--------|-------|---------|
| `data-science-mlops-engineer` | Data Science / MLOps Engineer | Global Consulting Firm (Madrid) |
| `ml-engineer-aws` | Machine Learning Engineer | Technology Company |
| `data-engineer-azure` | Data Engineer | Financial Services Firm |
| `ai-engineer-llm` | AI Engineer — LLMs and GenAI | AI Product Company |
| `backend-python-engineer` | Senior Python Backend Engineer | SaaS Company |

**Add a job by text:**
```bash
curl -X POST http://localhost:8000/api/add-job \
  -H "Content-Type: application/json" \
  -d '{"job_id": "acme-ml-2026", "title": "ML Engineer", "company": "Acme", "text": "..."}'
```

**Add a job by URL:**
```bash
curl -X POST http://localhost:8000/api/add-job-url \
  -H "Content-Type: application/json" \
  -d '{"job_id": "acme-ml-2026", "title": "ML Engineer", "company": "Acme", "url": "https://..."}'
```

## 6. Results

Tested against the Data Science / MLOps Engineer target role. Xavier's CV scores **Strong match (76/100)** with Good skill coverage. Matched skills include Amazon Bedrock, Amazon SageMaker, AWS, Azure, CI/CD, Docker, FastAPI, GitHub Actions, LangChain, MLflow, MLOps, Pinecone, Python, RAG, and sentence-transformers. Missing skills flagged: Bedrock Guardrails, BigQuery, GCP, Kubernetes — consistent with the actual gap between the CV and the target role at the time of analysis.

## 7. API Reference

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Demo frontend |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API docs (Swagger) |
| POST | `/match/text` | CV text + JD text |
| POST | `/match/url` | CV text + JD URL |
| POST | `/match/pdf` | CV PDF + JD PDF |
| POST | `/match/cv-text-jd-pdf` | CV text + JD PDF upload |
| POST | `/match/cv-text-jd-url` | CV text + JD URL (form fields) |
| POST | `/match/pdf-cv-url-jd` | CV PDF upload + JD URL |
| POST | `/match/pdf-cv-text-jd` | CV PDF upload + JD text |
| POST | `/fetch-url` | Preview URL extraction |
| POST | `/api/preview-pdf` | Extract text preview from PDF |
| GET | `/api/similar-jobs` | Find similar jobs by CV text |
| GET | `/api/job-library` | List built-in library |
| GET | `/api/job-library-text/{id}` | Get full text of library job |
| POST | `/api/add-job` | Add job by text |
| POST | `/api/add-job-url` | Add job by URL |
| GET | `/api/skills` | Extract skills from text |

**Example — text match:**
```bash
curl -X POST http://localhost:8000/match/text \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Python ML engineer with SageMaker, Bedrock, and FastAPI experience...",
    "jd_text": "Looking for ML Engineer with Python, AWS, MLOps...",
    "store_job": false
  }'
```

**Example — URL match:**
```bash
curl -X POST http://localhost:8000/match/url \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Python ML engineer...",
    "jd_url": "https://www.linkedin.com/jobs/view/1234567890",
    "jd_title": "ML Engineer",
    "jd_company": "Company"
  }'
```

## 8. Deployment — HuggingFace Spaces

The app is deployed as a Docker Space on HuggingFace Spaces (free tier).

```bash
# Add HF remote (one-time)
git remote add hf https://huggingface.co/spaces/xavier-oc-machinelearn/cv-gap-analyser

# Deploy
git push hf main
```

Add `PINECONE_API_KEY` as a Repository Secret in the Space settings UI — HuggingFace injects it as an environment variable at runtime.

The Space URL: `https://xavier-oc-machinelearn-cv-gap-analyser.hf.space`

Docker build runs on every push: `pip install -r requirements.txt` then `uvicorn main:app --host 0.0.0.0 --port 8000`.

## 9. CI/CD

GitHub Actions runs on every push to `main`:

1. `actions/checkout@v4`
2. `actions/setup-python@v5` — Python 3.11
3. `pip install -r requirements.txt`
4. `pytest tests/ -v` (with `PINECONE_API_KEY=""` — handled gracefully)

## 10. Design decisions

**Why semantic embeddings over keyword matching.** Keyword matching counts shared words. It misses synonyms, paraphrases, and domain-equivalent descriptions. A CV written in plain language about work that exactly matches a job description may score zero on keyword overlap. Cosine similarity between sentence embeddings captures shared meaning regardless of specific word choice. The practical implication: a CV describing "deploying models to production cloud infrastructure" correctly scores high against a role requiring "MLOps and cloud-native deployment".

**Why all-MiniLM-L6-v2 over larger models.** Larger embedding models (e.g., `all-mpnet-base-v2`, `e5-large`) produce marginally better similarity estimates but require significantly more memory and inference time. For CV-JD matching on professional text, the quality difference is negligible. MiniLM-L6-v2 loads in under 2 seconds on CPU, embeds a 500-word document in milliseconds, and runs comfortably on HuggingFace Spaces free CPU tier. The 384-dimensional output is small enough for Pinecone's free tier without dimension reduction.

**Why ROUGE alongside semantic similarity (ATS framing).** The two metrics answer different questions. Semantic similarity asks: do these documents cover the same concepts? ROUGE asks: does the CV use the same words as the JD? Both matter. A CV with high semantic similarity but low ROUGE-L is at risk of failing ATS screening — automated keyword-matching filters that operate before a human reads the CV. The ROUGE score surfaces this specific failure mode and motivates a concrete recommendation: mirror the JD's exact terminology where possible.

**Why trafilatura over BeautifulSoup or a headless browser.** BeautifulSoup strips all HTML tags — it returns navigation, footers, cookie banners, and job description body indiscriminately. A headless browser handles JavaScript-rendered pages correctly but is significantly heavier, slower, and harder to deploy. trafilatura was purpose-built for main content extraction and handles most career pages cleanly. BeautifulSoup remains as a fallback for pages where trafilatura returns too little content. LinkedIn required a third approach — the unauthenticated guest API — because both trafilatura and BeautifulSoup only see the login wall.

**Why keyword-only skill extraction.** The initial implementation used spaCy noun phrase extraction as a second pass to catch technologies not in the keyword list. In testing, it consistently extracted job description boilerplate (benefit descriptions, company names, location phrases), non-English phrases from Spanish job postings, and multi-word fragments that were not skills. The noise ratio was too high to be useful. The keyword list (~150 terms) is deterministic, produces zero false positives, and covers the real technology landscape for ML/AI/data roles. Missing niche tools is a documented limitation — better than surfacing garbage.

**Why Pinecone for the job library.** The job library could be implemented as in-memory cosine similarity. For five documents that would be trivially fast. Pinecone is used for two reasons: (1) it makes the library extensible — new jobs added via the API are immediately searchable without restarting the server; (2) it demonstrates a practical vector database integration pattern relevant to RAG pipelines and production ML systems. The free tier handles the library at zero cost.

**Why this project was built.** I built an early version of this tool while preparing my application for a Data Science / MLOps Engineer role at Accenture. The gap analysis identified Amazon Bedrock, SageMaker, and LLMOps as skills present in the job description but without strong portfolio evidence in my CV at the time. The projects built in response — including the SageMaker MLOps pipeline, Bedrock benchmarking projects, and the LLM document summarizer with LLMOps monitoring — are now in the portfolio. The tool identified the gaps. The work closed them.

## Limitations

Real limitations discovered during development, documented honestly.

**Skill extraction is keyword-only.** The keyword list (~150 terms) is precise but cannot catch tools not on the list. New or niche technologies will be missed entirely.

**LinkedIn URL extraction is fragile.** The unauthenticated guest API endpoint works for most direct job posting URLs but is undocumented and may break without notice. HTML structure varies across job types.

**Non-English job descriptions.** The keyword list is English-only. Spanish, French, or other-language JDs return few matched skills even when the underlying technologies are identical.

**HuggingFace Spaces free tier constraints.** The app sleeps after inactivity, causing a cold start delay on the first request. The shared CPU tier can be slow under concurrent load.

**Match scores are not calibrated to hiring outcomes.** A score of 72/100 does not mean a 72% chance of interview. The thresholds reflect linguistic similarity — not actual job fit, recruiter judgement, or cultural match.

## 11. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | >=2.6 | Semantic embeddings (all-MiniLM-L6-v2) |
| `pinecone` | >=3.0 | Vector database for job library |
| `spacy` | >=3.7 | NLP dependency (model no longer used for skill extraction) |
| `trafilatura` | >=1.8 | URL content extraction |
| `rouge-score` | >=0.1.2 | ROUGE keyword overlap metrics |
| `pymupdf` | >=1.23 | PDF text extraction |
| `beautifulsoup4` | >=4.12 | HTML fallback extraction |
| `requests` | >=2.31 | HTTP client for URL fetching |
| `fastapi` | >=0.110 | REST API framework |
| `uvicorn` | >=0.27 | ASGI server (development) |
| `gunicorn` | >=21.0 | WSGI server (production) |
| `pydantic` | >=2.0 | Request/response validation |
| `numpy` | >=1.24,<2.0 | Numerical operations |
| `pandas` | >=2.0 | Data manipulation |
| `jupyter` | >=1.0 | Notebook environment |
| `pytest` | >=7.0 | API tests |
| `httpx` | >=0.27 | Async HTTP client for TestClient |
| `python-multipart` | >=0.0.9 | File upload support (FastAPI) |
| `python-dotenv` | >=1.0 | Load `.env` for local development |
