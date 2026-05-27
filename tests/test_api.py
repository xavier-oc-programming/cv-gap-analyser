"""
tests/test_api.py — pytest tests for CV Gap Analyser API.

Run with: pytest tests/ -v
PINECONE_API_KEY can be empty — embeddings.py handles this gracefully.
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE_CV = """
Python developer with experience in machine learning and data science.
Skilled in TensorFlow, scikit-learn, pandas, and NumPy. Deployed models
to AWS Lambda and Azure App Service. Experience with Docker, CI/CD pipelines,
and REST APIs using FastAPI and Flask. Knowledge of NLP, deep learning,
and MLOps patterns including model versioning and experiment tracking with
MLflow. Built RAG pipelines using LangChain and ChromaDB. Amazon Bedrock
used across multiple projects for zero-shot classification benchmarking.
"""

SAMPLE_JD = """
We are looking for a Machine Learning Engineer with strong Python skills.
Requirements: TensorFlow or PyTorch, AWS or Azure cloud deployment, Docker,
FastAPI, MLflow, CI/CD, REST APIs. Experience with NLP and deep learning
preferred. Knowledge of MLOps practices and Amazon Bedrock required.
LangChain and RAG pipeline experience a plus.
"""


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_match_text():
    response = client.post("/match/text", json={
        "cv_text": SAMPLE_CV,
        "jd_text": SAMPLE_JD,
        "store_job": False
    })
    assert response.status_code == 200
    data = response.json()
    assert "match" in data
    assert "skills" in data
    assert "recommendations" in data
    assert "summary" in data
    assert 0 <= data["match"]["match_score_pct"] <= 100
    assert data["match"]["match_label"] in [
        "Strong match", "Good match", "Partial match", "Weak match"
    ]


def test_match_cv_too_short():
    response = client.post("/match/text", json={
        "cv_text": "Too short.",
        "jd_text": SAMPLE_JD,
        "store_job": False
    })
    assert response.status_code == 422


def test_fetch_url_invalid():
    response = client.post("/fetch-url",
        json={"url": "https://this-url-does-not-exist-xyz.com"})
    assert response.status_code == 400


def test_skills_endpoint():
    response = client.get(f"/api/skills?text={SAMPLE_CV[:300]}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_job_library():
    response = client.get("/api/job-library")
    assert response.status_code == 200
    jobs = response.json()
    assert isinstance(jobs, list)
    assert len(jobs) >= 5


def test_similar_jobs():
    response = client.get(f"/api/similar-jobs?cv_text={SAMPLE_CV[:200]}")
    assert response.status_code == 200


def test_job_library_text():
    response = client.get("/api/job-library-text/data-science-mlops-engineer")
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert len(data["text"]) > 100


def test_job_library_text_not_found():
    response = client.get("/api/job-library-text/nonexistent-job")
    assert response.status_code == 404
