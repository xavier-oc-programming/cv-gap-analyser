"""
scorer.py — match scoring and full analysis.

No classes. Three public functions:
  compute_match_score()
  compute_rouge_overlap()
  full_analysis()
"""
from config import (
    STRONG_MATCH_THRESHOLD,
    GOOD_MATCH_THRESHOLD,
    PARTIAL_MATCH_THRESHOLD,
)
from embeddings import embed_text, find_similar_jobs
from skill_extractor import extract_skills, compare_skills


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _match_label(score: float) -> str:
    if score >= STRONG_MATCH_THRESHOLD:
        return "Strong match"
    elif score >= GOOD_MATCH_THRESHOLD:
        return "Good match"
    elif score >= PARTIAL_MATCH_THRESHOLD:
        return "Partial match"
    return "Weak match"


def compute_match_score(cv_text: str, jd_text: str) -> dict:
    """
    Cosine similarity between CV and JD embeddings.

    Returns:
        {
          "similarity_score": float,
          "match_label": str,
          "match_score_pct": int
        }
    """
    cv_vec = embed_text(cv_text)
    jd_vec = embed_text(jd_text)
    score = round(_cosine(cv_vec, jd_vec), 4)
    label = _match_label(score)
    pct = min(100, max(0, int(round(score * 100))))
    return {
        "similarity_score": score,
        "match_label": label,
        "match_score_pct": pct,
    }


def compute_rouge_overlap(cv_text: str, jd_text: str) -> dict:
    """
    ROUGE scores treating JD as reference, CV as hypothesis.
    Higher = more shared terminology = better ATS keyword alignment.

    Returns:
        {
          "rouge1": float,
          "rouge2": float,
          "rougeL": float,
          "interpretation": str
        }
    """
    from rouge_score import rouge_scorer as rs_module

    scorer = rs_module.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(jd_text, cv_text)

    r1 = round(scores['rouge1'].fmeasure, 4)
    r2 = round(scores['rouge2'].fmeasure, 4)
    rL = round(scores['rougeL'].fmeasure, 4)
    pct = int(round(rL * 100))

    if rL > 0.3:
        note = "Good keyword alignment."
    else:
        note = "Consider mirroring more of the job description terminology."

    interpretation = (
        f"Your CV shares {pct}% of key phrases with this job description "
        f"(ROUGE-L: {rL:.2f}). {note}"
    )

    return {
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rL,
        "interpretation": interpretation,
    }


def full_analysis(
    cv_text: str,
    jd_text: str,
    jd_title: str = None,
    jd_company: str = None,
    jd_source_url: str = None,
) -> dict:
    """
    Complete match analysis between CV and job description.

    Steps:
    1. compute_match_score()
    2. compute_rouge_overlap()
    3. extract_skills() on both texts
    4. compare_skills()
    5. find_similar_jobs() from Pinecone
    6. Generate recommendations from missing skills
    7. Generate plain-English summary

    Returns:
        {
          "match": dict,
          "rouge": dict,
          "skills": dict,
          "similar_jobs": list,
          "recommendations": list[str],
          "summary": str,
          "jd_source_url": str or None
        }
    """
    match = compute_match_score(cv_text, jd_text)
    rouge = compute_rouge_overlap(cv_text, jd_text)

    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(jd_text)
    skills = compare_skills(cv_skills, jd_skills)

    similar_jobs = find_similar_jobs(cv_text)

    recommendations = [
        f"Add evidence of {skill} to your CV — it appears in the job "
        "description but is not currently demonstrated."
        for skill in skills["missing"]
    ]

    missing = skills["missing"]
    top3 = ", ".join(missing[:3]) if missing else "none identified"
    n_matched = len(skills["matched"])
    n_required = len(jd_skills)

    summary = (
        f"Your CV is a {match['match_label']} for this role "
        f"({match['match_score_pct']}/100). "
        f"You demonstrate {n_matched} of {n_required} required skills. "
        f"The strongest gaps are: {top3}. "
        f"{rouge['interpretation']}"
    )

    return {
        "match": match,
        "rouge": rouge,
        "skills": skills,
        "similar_jobs": similar_jobs,
        "recommendations": recommendations,
        "summary": summary,
        "jd_source_url": jd_source_url,
    }
