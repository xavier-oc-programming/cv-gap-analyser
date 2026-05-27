# Job Library

Five realistic job descriptions committed to this repository.

These files are loaded into Pinecone at application startup. The index is seeded
once — the startup routine checks the existing vector count and skips indexing
if the library is already present.

Each file represents a realistic role in the data/ML/AI engineering space.
The `data-science-mlops-engineer.txt` description mirrors the target Accenture
role and should produce a strong match against an AI engineering CV.

## Adding jobs

Via the API:

- `POST /api/add-job` — submit text directly
- `POST /api/add-job-url` — fetch and index from a career page URL

## Files

| File | Title | Company |
|------|-------|---------|
| `data-science-mlops-engineer.txt` | Data Science / MLOps Engineer | Global Consulting Firm (Madrid) |
| `ml-engineer-aws.txt` | Machine Learning Engineer | Technology Company |
| `data-engineer-azure.txt` | Data Engineer | Financial Services Firm |
| `ai-engineer-llm.txt` | AI Engineer — LLMs and GenAI | AI Product Company |
| `backend-python-engineer.txt` | Senior Python Backend Engineer | SaaS Company |
