"""
skill_extractor.py — extract skills and technologies from text.

No classes. Two public functions: extract_skills(), compare_skills().

Extraction is keyword-only: case-insensitive whole-word matching against a
curated list of ~150 technology and skill terms. spaCy noun phrase extraction
was removed — it consistently extracted job description boilerplate, company
names, and non-English phrases as false-positive skills.
"""
import re

from config import SPACY_MODEL

TECH_KEYWORDS = [
    # --- Languages ---
    "Python", "R", "SQL", "Java", "Scala", "JavaScript", "TypeScript",
    "Go", "Rust", "C++", "C#", "Bash", "Shell", "YAML",

    # --- ML / DL frameworks ---
    "TensorFlow", "Keras", "PyTorch", "scikit-learn", "XGBoost", "LightGBM",
    "CatBoost", "HuggingFace", "Hugging Face", "JAX", "MXNet",

    # --- Data libraries ---
    "pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "SciPy", "Statsmodels",

    # --- AWS ---
    "AWS", "Amazon Web Services",
    "SageMaker", "Amazon SageMaker",
    "Bedrock", "Amazon Bedrock",
    "Bedrock Guardrails",
    "Lambda", "AWS Lambda",
    "S3", "EC2", "ECS", "EKS",
    "AWS Glue", "AWS Step Functions",
    "CloudFormation", "CloudWatch",
    "Rekognition", "Comprehend", "Textract",

    # --- Azure ---
    "Azure", "Microsoft Azure",
    "Azure ML", "Azure Machine Learning",
    "Azure Data Factory",
    "Azure App Service",
    "Azure DevOps",
    "Azure Synapse",
    "Azure Databricks",
    "Azure Functions",
    "Azure Blob Storage",
    "Azure Cognitive Services",
    "Azure OpenAI",

    # --- GCP ---
    "GCP", "Google Cloud", "Google Cloud Platform",
    "BigQuery",
    "Vertex AI",
    "Cloud Run",
    "Cloud Functions",
    "Cloud Build",
    "Cloud Composer",
    "Dataflow",
    "Dataproc",
    "GCS", "Cloud Storage",
    "GKE",
    "Pub/Sub",
    "Apigee",
    "Looker Studio",

    # --- Infra / DevOps ---
    "Docker", "Kubernetes", "Terraform", "Ansible", "Helm",
    "CI/CD", "GitHub Actions", "GitLab CI", "Jenkins", "CircleCI",
    "Git", "GitHub", "GitLab", "Bitbucket",
    "Linux", "Nginx",

    # --- MLOps / LLMOps ---
    "MLflow", "Kubeflow", "MLOps", "LLMOps", "DevOps",
    "Model Registry", "Feature Store", "Weights and Biases", "W&B",

    # --- Data engineering ---
    "Airflow", "Apache Airflow",
    "Spark", "PySpark", "Apache Spark",
    "Hadoop", "Hive",
    "Kafka", "Apache Kafka",
    "Flink", "Apache Flink",
    "dbt", "ETL", "ELT",
    "Data Pipeline", "Data Warehouse", "Data Lake",
    "Databricks", "Snowflake", "Redshift",

    # --- Databases ---
    "PostgreSQL", "MySQL", "SQLite",
    "MongoDB", "Cassandra", "DynamoDB",
    "Redis", "Memcached",
    "Elasticsearch", "OpenSearch",
    "Pinecone", "ChromaDB", "Weaviate", "Qdrant", "Milvus",

    # --- Frameworks / APIs ---
    "FastAPI", "Flask", "Django", "FastAPI",
    "REST API", "GraphQL", "gRPC",
    "Microservices", "Serverless",

    # --- LLM / GenAI ---
    "LLM", "LLMs",
    "RAG",
    "LangChain", "LlamaIndex",
    "GPT", "GPT-4", "GPT-3",
    "BERT", "Transformer", "Transformers",
    "Foundation Models",
    "Prompt Engineering",
    "Vector Database", "Vector Search",
    "Claude", "Llama", "Mistral", "Gemini",
    "OpenAI", "Anthropic",
    "Fine-tuning", "RLHF", "LoRA",

    # --- NLP / CV ---
    "NLP", "Natural Language Processing",
    "Computer Vision",
    "spaCy", "NLTK",
    "OCR", "CNNs", "RNNs",
    "Stable Diffusion", "Diffusion Models",
    "Sentence Transformers", "sentence-transformers",

    # --- ML concepts ---
    "Machine Learning", "Deep Learning",
    "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
    "Feature Engineering", "Hyperparameter Tuning", "Transfer Learning",
    "A/B Testing", "Experiment Tracking",
    "Time Series", "Forecasting",
    "Clustering", "Classification", "Regression",
    "Statistics", "Linear Algebra", "Probability",

    # --- Explainability ---
    "SHAP", "LIME",

    # --- BI / Analytics ---
    "Power BI", "Tableau", "Looker", "Metabase", "Grafana", "Kibana",
    "Google Analytics",

    # --- Process ---
    "Agile", "Scrum", "Kanban",

    # --- Misc ---
    "trafilatura", "PyMuPDF", "BeautifulSoup",
    "pytest", "unittest",
    "ROUGE", "BLEU",
]


def extract_skills(text: str) -> list[str]:
    """
    Extract skill and technology mentions from text using keyword matching.

    Case-insensitive whole-word match (regex \\b boundaries) against
    TECH_KEYWORDS. Returns deduplicated, sorted list using the canonical
    keyword capitalisation.

    spaCy noun phrase extraction was removed: it produced too many false
    positives (boilerplate, non-English phrases, company names) to be useful.
    The keyword list is deterministic, precise, and covers the real technology
    landscape for ML/AI/data roles.
    """
    found: dict[str, str] = {}  # lowercase key → canonical keyword

    for kw in TECH_KEYWORDS:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found[kw.lower()] = kw

    return sorted(found.values())


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
