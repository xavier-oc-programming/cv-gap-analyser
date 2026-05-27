"""
config.py — single source of truth for all constants.
No classes — all constants are module-level.
"""
from pathlib import Path

# Embeddings
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384

# Pinecone
# Requires PINECONE_API_KEY set as environment variable
# Free tier at pinecone.io — starter plan, no credit card required
# Index: cv-gap-analyser-jobs, dimension: 384, metric: cosine
PINECONE_INDEX_NAME = 'cv-gap-analyser-jobs'
PINECONE_METRIC = 'cosine'
TOP_K_SIMILAR = 5

# Match scoring thresholds
# cosine similarity thresholds for all-MiniLM-L6-v2 on professional text.
# Above 0.75 = strong semantic alignment
# 0.60-0.75 = good match, some gaps
# 0.45-0.60 = partial match, significant gaps
# Below 0.45 = weak match
STRONG_MATCH_THRESHOLD = 0.75
GOOD_MATCH_THRESHOLD = 0.60
PARTIAL_MATCH_THRESHOLD = 0.45

# Skill extraction
SPACY_MODEL = 'en_core_web_sm'

# Text limits
MAX_PDF_SIZE_MB = 10
MAX_TEXT_LENGTH = 8000
# URL-fetched content may include residual boilerplate even after
# trafilatura extraction — a tighter limit reduces noise in embeddings.
MAX_URL_TEXT_LENGTH = 6000

# Paths
UPLOAD_DIR = Path('uploads')
JOB_LIBRARY_DIR = Path('job_library')

# Azure
AZURE_APP_NAME = 'cv-gap-analyser'
AZURE_RESOURCE_GROUP = 'cv-gap-analyser-rg'
AZURE_LOCATION = 'westeurope'
