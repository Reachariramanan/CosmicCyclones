# drik_scraper.py

import time
import random
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from drik_parser import extract_cards, extract_planetary_table, extract_tamil_panchang

# Thread-local storage to reuse one Session per thread for speed
thread_local = threading.local()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def get_session():
    """Returns a reused requests session with a retry mechanism for the thread."""
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)
        
        # Automatically retries on specific status codes often used for rate limiting
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            raise_on_status=False
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        thread_local.session = session
            
    return thread_local.session

def _safe_get(session, url, timeout=15, max_retries=4):
    """
    GET with per-request retry for low-level TCP resets (ConnectionResetError,
    ConnectionError) that bypass the urllib3 Retry adapter.
    Returns (response | None).
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            return resp
        except (ConnectionResetError, requests.exceptions.ConnectionError) as e:
            if attempt == max_retries:
                raise
            # Recreate session so the broken socket is discarded
            thread_local.session = None
            session = get_session()
            sleep_s = (1.5 * (2 ** (attempt - 1))) + random.uniform(0.3, 1.0)
            time.sleep(sleep_s)
    return None  # unreachable, but satisfies type checkers


def fetch_drik_data(geoname_id, date_value, time_value):
    """Fetch all 3 DrikPanchang pages in parallel for maximum speed."""

    url_english   = f"https://www.drikpanchang.com/panchang/day-panchang.html?geoname-id={geoname_id}&date={date_value}&time={time_value}"
    url_tamil     = f"https://www.drikpanchang.com/tamil/tamil-day-panchangam.html?geoname-id={geoname_id}&date={date_value}&time={time_value}"
    url_planetary = f"https://www.drikpanchang.com/planet/position/planetary-positions-sidereal.html?geoname-id={geoname_id}&date={date_value}&time={time_value}"

    result = {"english_panchang": None, "tamil_panchang": None, "planetary_positions": None}

    # Small human-like jitter before firing
    time.sleep(random.uniform(0.5, 1.2))

    def _fetch(url):
        # Each parallel sub-request gets its own session (thread-local)
        session = get_session()
        return _safe_get(session, url)

    tasks = {
        "english":   url_english,
        "tamil":     url_tamil,
        "planetary": url_planetary,
    }

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch, url): key for key, url in tasks.items()}
            for future in as_completed(futures):
                key  = futures[future]
                resp = future.result()   # raises on unrecoverable error
                if resp is None or resp.status_code != 200:
                    continue
                if key == "english":
                    result["english_panchang"]  = extract_cards(resp.text, date_value)
                elif key == "tamil":
                    result["tamil_panchang"]    = extract_tamil_panchang(resp.text, date_value)
                else:
                    result["planetary_positions"] = extract_planetary_table(resp.text, date_value)

    except Exception as e:
        print(f"Connection error for {geoname_id} on {date_value}: {e}")

    return result