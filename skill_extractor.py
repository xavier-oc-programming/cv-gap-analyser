"""
skill_extractor.py — extract skills and technologies from text.

No classes. Two public functions: extract_skills(), compare_skills().

spaCy model (en_core_web_sm) is loaded lazily and cached globally.
It must be downloaded before use: python -m spacy download en_core_web_sm
In Docker it is downloaded at build time to avoid runtime download on first request.
"""
import re

from config import SPACY_MODEL

# Comprehensive technology and skill keyword list.
# Used for case-insensitive whole-word matching against CV and JD text.
TECH_KEYWORDS = [
    # Languages
    "Python", "R", "SQL", "Java", "Scala", "JavaScript", "TypeScript",
    "Go", "Rust", "C++", "C#", "Bash", "Shell",
    # ML frameworks
    "TensorFlow", "Keras", "PyTorch", "scikit-learn", "XGBoost", "LightGBM",
    "CatBoost", "Hugging Face", "HuggingFace",
    # Data libraries
    "pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "SciPy",
    # Cloud
    "AWS", "Azure", "GCP", "Google Cloud",
    "Lambda", "SageMaker", "Bedrock", "S3", "EC2", "ECS", "EKS",
    "Azure Functions", "Azure ML", "Azure Data Factory", "Databricks",
    # Orchestration / infra
    "Docker", "Kubernetes", "Terraform", "Ansible", "Helm",
    # ML ops
    "MLflow", "Kubeflow", "Airflow", "dbt", "Spark", "PySpark", "Hadoop",
    "MLOps", "LLMOps", "DevOps",
    # APIs / frameworks
    "FastAPI", "Flask", "Django", "REST API", "GraphQL", "gRPC",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Pinecone", "ChromaDB", "Weaviate", "Qdrant",
    # LLM / AI
    "LangChain", "RAG", "LLM", "GPT", "BERT", "Transformer", "HuggingFace",
    "Foundation Models", "Prompt Engineering", "Vector Database",
    "Bedrock Guardrails", "Llama", "Mistral", "Claude",
    # CI/CD / dev tools
    "Git", "GitHub", "CI/CD", "GitHub Actions", "Jenkins", "GitLab CI",
    # BI / analytics
    "Power BI", "Tableau", "Looker", "Metabase",
    # Explainability
    "SHAP", "LIME",
    # Data engineering
    "Data Pipeline", "ETL", "Data Warehouse", "Data Lake",
    "Apache Kafka", "Flink",
    # ML concepts
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Statistics", "Linear Algebra", "Probability", "A/B Testing",
    "Feature Engineering", "Hyperparameter Tuning", "Transfer Learning",
    # Process
    "Agile", "Scrum",
    # Libraries used in this project (self-referential but useful)
    "trafilatura", "spaCy", "sentence-transformers",
    # Azure specifics
    "Azure App Service", "Azure DevOps",
    # Misc
    "ROUGE", "PyMuPDF", "BeautifulSoup",
]

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def extract_skills(text: str) -> list[str]:
    """
    Extract skill and technology mentions from text.

    Two-pass approach:
    1. Keyword matching: scan for TECH_KEYWORDS (case-insensitive,
       whole-word match using regex \\b boundaries)
    2. spaCy noun phrases: extract noun chunks that look like
       technical terms (not common stopwords, length > 2 chars)

    Deduplicate, normalise case, return sorted list.
    """
    found = set()

    # Pass 1: keyword matching
    for kw in TECH_KEYWORDS:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found.add(kw)

    # Pass 2: spaCy noun phrases
    nlp = _get_nlp()
    doc = nlp(text[:5000])  # limit to avoid slow processing on huge texts
    stopwords = nlp.Defaults.stop_words
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase) > 2 and phrase.lower() not in stopwords:
            # Only include if it looks technical (contains capital or digit)
            if re.search(r'[A-Z0-9]', phrase):
                found.add(phrase)

    return sorted(found)


def compare_skills(cv_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compare skills from CV against job description.

    Returns:
        {
          "matched": list[str],
          "missing": list[str],
          "additional": list[str],
          "match_rate": float,
          "coverage_label": "Excellent" / "Good" / "Partial" / "Low"
        }

    coverage thresholds: >=0.8 Excellent, >=0.6 Good, >=0.4 Partial, <0.4 Low
    """
    cv_lower = {s.lower() for s in cv_skills}
    jd_lower = {s.lower() for s in jd_skills}

    matched = sorted([s for s in jd_skills if s.lower() in cv_lower])
    missing = sorted([s for s in jd_skills if s.lower() not in cv_lower])
    additional = sorted([s for s in cv_skills if s.lower() not in jd_lower])

    match_rate = len(matched) / len(jd_skills) if jd_skills else 0.0

    if match_rate >= 0.8:
        coverage_label = "Excellent"
    elif match_rate >= 0.6:
        coverage_label = "Good"
    elif match_rate >= 0.4:
        coverage_label = "Partial"
    else:
        coverage_label = "Low"

    return {
        "matched": matched,
        "missing": missing,
        "additional": additional,
        "match_rate": round(match_rate, 4),
        "coverage_label": coverage_label,
    }
