import csv, re, time, json, sys
from pathlib import Path
import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sources.csv"
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Doc_GPT crawler (educational RAG project)"}

def log(msg):
    print(msg, flush=True)

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")  # requires lxml installed
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in ["header", "footer", "nav", ".nhsuk-c-cookie-banner", ".nhsuk-global-alert"]:
        for el in soup.select(sel):
            el.decompose()
    mains = soup.select("main, article, #maincontent, .nhsuk-width-container")
    text = " ".join([m.get_text(' ', strip=True) for m in mains]) or soup.get_text(' ', strip=True)
    return re.sub(r"\s+", " ", text).strip()

def fetch(url: str) -> str:
    with httpx.Client(follow_redirects=True, headers=HEADERS, timeout=30) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text

def sha16(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def main():
    if not SRC.exists():
        sys.exit(f"[fatal] missing {SRC}")
    raw = SRC.read_text(encoding="utf-8", errors="ignore")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
        SRC.write_text(raw, encoding="utf-8")
    reader = csv.DictReader(raw.splitlines())
    total = saved = 0
    for row in reader:
        url = (row.get("url") or "").strip()
        title = (row.get("title") or "").strip()
        if not url.startswith("http"):
            log(f"[skip] bad row: {row}")
            continue
        total += 1
        try:
            log(f"[fetch] {url}")
            text = clean_html(fetch(url))
            if len(text) < 200:
                log(f"[skip] too short: {url}")
                continue
            out = RAW / f"{sha16(url)}.json"
            out.write_text(json.dumps(
                {"url": url, "title": title, "text": text},
                ensure_ascii=False, indent=2
            ), encoding="utf-8")
            log(f"[saved] {url} -> {out.name} ({len(text)} chars)")
            saved += 1
            time.sleep(0.3)
        except Exception as e:
            log(f"[error] {url}: {e!r}")
    log(f"[done] rows={total}, saved={saved}, out_dir={RAW}")

if __name__ == "__main__":
    main()
