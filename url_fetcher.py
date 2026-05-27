"""
url_fetcher.py — fetch and extract clean text from job posting URLs.

No classes. Single public function: fetch_job_from_url().
"""
import requests
from bs4 import BeautifulSoup
import trafilatura

from config import MAX_URL_TEXT_LENGTH

# trafilatura was built for academic web crawling — it identifies
# the main content area of a web page (article, job description,
# product listing) and extracts just that text. It handles most
# company career pages and job boards (LinkedIn, Indeed, Glassdoor)
# without custom CSS selectors or site-specific rules.
# Failure case: pages that load content via JavaScript after the
# initial page load — trafilatura only sees the initial HTML.
# BeautifulSoup fallback handles these cases with noisier output.


def fetch_job_from_url(url: str) -> dict:
    """
    Fetch and extract clean text from a job posting URL.

    Two-pass approach:
    1. Primary: trafilatura — fetches URL, extracts main content,
       discards navigation, footers, sidebars, and boilerplate.
       Handles most company career pages and job boards cleanly.
    2. Fallback: BeautifulSoup — if trafilatura returns None or
       fewer than 100 characters (page likely JavaScript-rendered
       or trafilatura failed), fall back to BeautifulSoup stripping
       all HTML tags from the raw response. Noisier but gets something.

    Args:
        url: job posting URL

    Returns:
        {
          "text": str — extracted job description text,
          "url": str — the original URL,
          "word_count": int,
          "extraction_method": "trafilatura" or "beautifulsoup_fallback",
          "truncated": bool
        }

    Raises:
        ValueError if URL is unreachable (non-200 response or timeout).
        ValueError if extracted text is fewer than 50 characters after
        both extraction attempts (likely a JavaScript-only page).
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded) if downloaded else None
    except Exception:
        text = None

    extraction_method = 'trafilatura'

    if not text or len(text.strip()) < 100:
        # Fallback: BeautifulSoup
        try:
            response = requests.get(
                url, timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            if response.status_code != 200:
                raise ValueError(
                    f"URL returned status {response.status_code}: {url}"
                )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Could not reach URL: {url}. Error: {e}")

        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        extraction_method = 'beautifulsoup_fallback'

    if not text or len(text.strip()) < 50:
        raise ValueError(
            f"Could not extract meaningful text from URL: {url}. "
            "The page may require JavaScript to render content."
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
