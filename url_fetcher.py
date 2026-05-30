"""
url_fetcher.py — fetch and extract clean text from job posting URLs.

No classes. Single public function: fetch_job_from_url().

LinkedIn handling:
LinkedIn job pages require authentication and load content via JavaScript,
making them inaccessible to standard scrapers. LinkedIn exposes an
unauthenticated guest API endpoint:
  https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}
This returns a server-rendered HTML fragment for the job posting — no login
required. We extract the job ID from any LinkedIn jobs URL and hit this
endpoint directly.

Supported LinkedIn URL formats:
  /jobs/view/1234567890/
  /jobs/view/1234567890/?...
  ?currentJobId=1234567890
  ?jobId=1234567890
"""
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import trafilatura

from config import MAX_URL_TEXT_LENGTH

LINKEDIN_GUEST_API = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# trafilatura was built for academic web crawling — it identifies
# the main content area of a web page (article, job description,
# product listing) and extracts just that text. It handles most
# company career pages and job boards without custom CSS selectors.
# Failure case: pages that load content via JavaScript after the
# initial page load — trafilatura only sees the initial HTML.
# BeautifulSoup fallback handles these cases with noisier output.


def _extract_linkedin_job_id(url: str) -> str | None:
    """Extract LinkedIn job ID from any LinkedIn jobs URL format."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    # Query param: ?currentJobId=... or ?jobId=...
    for param in ("currentJobId", "jobId"):
        if param in qs:
            return qs[param][0]

    # Path: /jobs/view/1234567890/
    match = re.search(r'/jobs/view/(\d+)', parsed.path)
    if match:
        return match.group(1)

    return None


def _fetch_linkedin(url: str) -> tuple:
    """
    Fetch a LinkedIn job posting via the unauthenticated guest API.
    Targets the job description div directly to avoid LinkedIn boilerplate.
    """
    job_id = _extract_linkedin_job_id(url)
    if not job_id:
        raise ValueError(
            "Could not extract a LinkedIn job ID from this URL. "
            "Use a direct job posting URL — e.g. linkedin.com/jobs/view/1234567890 "
            "or copy the currentJobId from the URL bar."
        )

    guest_url = LINKEDIN_GUEST_API.format(job_id=job_id)
    try:
        response = requests.get(guest_url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            raise ValueError(
                f"LinkedIn guest API returned {response.status_code} for job ID {job_id}. "
                "The job posting may have been removed or the ID is invalid."
            )
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not reach LinkedIn: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Try to extract just the job description section — avoids boilerplate
    # LinkedIn guest API uses these containers for the actual JD content
    description_selectors = [
        "div.show-more-less-html__markup",
        "div.description__text",
        "section.show-more-less-html",
        "div.jobs-description-content__text",
        "div[class*='description']",
    ]
    description_el = None
    for selector in description_selectors:
        description_el = soup.select_one(selector)
        if description_el:
            break

    if description_el:
        # Also grab the job title and company for context
        title_el = soup.select_one("h2.top-card-layout__title, h1.top-card-layout__title, h2[class*='title']")
        company_el = soup.select_one("a.topcard__org-name-link, span.topcard__flavor")
        header = ""
        if title_el:
            header += title_el.get_text(strip=True) + "\n"
        if company_el:
            header += company_el.get_text(strip=True) + "\n\n"
        text = header + description_el.get_text(separator="\n", strip=True)
    else:
        # Fallback: strip all boilerplate sections and take remaining text
        for tag in soup(["script", "style", "nav", "footer", "header", "button"]):
            tag.decompose()

        # Remove known LinkedIn boilerplate sections
        for selector in ["div.top-card-layout__cta", "section.similar-jobs",
                         "div.apply-button", "footer"]:
            for el in soup.select(selector):
                el.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Strip the most common LinkedIn login/signup boilerplate lines
        boilerplate_phrases = [
            "Join or sign in", "New to LinkedIn", "Join now", "Sign in",
            "Click on the link", "emailed a one-time link", "primary email address",
            "By clicking Continue", "User Agreement", "Privacy Policy", "Cookie Policy",
            "See who", "has hired for this role", "applicants",
        ]
        lines = text.splitlines()
        lines = [l for l in lines if not any(p.lower() in l.lower() for p in boilerplate_phrases)]
        text = "\n".join(lines)

    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    text = re.sub(r' {2,}', ' ', text)

    if len(text.strip()) < 100:
        raise ValueError(
            f"LinkedIn job ID {job_id} returned too little content. "
            "The posting may be expired or region-restricted."
        )

    return text, "linkedin_guest_api"


def fetch_job_from_url(url: str) -> dict:
    """
    Fetch and extract clean text from a job posting URL.

    Three-pass approach:
    1. LinkedIn: if the URL is a LinkedIn jobs URL, use the unauthenticated
       guest API endpoint to bypass the login wall. Extracts job ID from URL,
       fetches https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{id},
       parses the returned HTML fragment with BeautifulSoup.
    2. Primary (non-LinkedIn): trafilatura — identifies and extracts the main
       content area of the page, discarding navigation, footers, sidebars.
    3. Fallback: BeautifulSoup — strips all HTML tags from the raw response.
       Noisier but works on pages trafilatura can't parse.

    Args:
        url: job posting URL

    Returns:
        {
          "text": str,
          "url": str,
          "word_count": int,
          "extraction_method": str,
          "truncated": bool
        }

    Raises:
        ValueError if URL is unreachable or yields fewer than 50 chars.
    """
    text = None
    extraction_method = None

    # Pass 1: LinkedIn-specific handling
    if "linkedin.com" in url:
        text, extraction_method = _fetch_linkedin(url)

    # Pass 2: trafilatura
    if not text:
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded) if downloaded else None
            if text and len(text.strip()) >= 100:
                extraction_method = "trafilatura"
            else:
                text = None
        except Exception:
            text = None

    # Pass 3: BeautifulSoup fallback
    if not text:
        try:
            response = requests.get(url, timeout=10, headers=HEADERS)
            if response.status_code != 200:
                raise ValueError(
                    f"URL returned status {response.status_code}: {url}"
                )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Could not reach URL: {url}. Error: {e}")

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        extraction_method = "beautifulsoup_fallback"

    if not text or len(text.strip()) < 50:
        raise ValueError(
            f"Could not extract meaningful text from URL: {url}. "
            "The page may require JavaScript or authentication."
        )

    original_length = len(text)
    text = text[:MAX_URL_TEXT_LENGTH]
    truncated = len(text) < original_length

    return {
        "text": text,
        "url": url,
        "word_count": len(text.split()),
        "extraction_method": extraction_method,
        "truncated": truncated,
    }
