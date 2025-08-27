# -*- coding: utf-8 -*-
"""
Doc_GPT crawler
- Reads data/sources.csv
- Fetches each URL with retries and timeouts
- Extracts clean article text (readability -> BeautifulSoup fallback)
- Writes JSON files to data/raw/<sha16(url)>.json
- Skips existing/unchanged pages unless --overwrite
- CLI filters: --max, --only-domain, --only-topic, --delay

CSV columns supported:
    url (required), domain (optional), topic (optional), title (optional)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx
from bs4 import BeautifulSoup

try:
    # Optional but preferred
    from readability import Document as ReadabilityDocument  # type: ignore
    HAVE_READABILITY = True
except Exception:
    HAVE_READABILITY = False

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sources.csv"
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADERS = {
    "User-Agent": "Doc_GPT crawler (educational RAG project; contact: local-user)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}


# ---------- utilities ----------

def log(msg: str) -> None:
    print(msg, flush=True)


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Row:
    url: str
    domain: Optional[str] = None
    topic: Optional[str] = None
    title: Optional[str] = None


def read_sources_csv(path: Path) -> list[Row]:
    if not path.exists():
        sys.exit(f"[fatal] missing {path}")

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
        path.write_text(raw, encoding="utf-8")

    rows: list[Row] = []
    reader = csv.DictReader(raw.splitlines())
    for r in reader:
        url = (r.get("url") or "").strip()
        if not url.startswith("http"):
            log(f"[skip] bad row (no http URL): {r}")
            continue
        rows.append(
            Row(
                url=url,
                domain=(r.get("domain") or "").strip() or None,
                topic=(r.get("topic") or "").strip() or None,
                title=(r.get("title") or "").strip() or None,
            )
        )
    return rows


# ---------- fetching ----------

def fetch_html(url: str, headers: Dict[str, str], timeout_s: float = 20.0, retries: int = 3, backoff: float = 0.8) -> str:
    last_exc: Optional[Exception] = None
    with httpx.Client(follow_redirects=True, headers=headers, timeout=timeout_s) as client:
        for attempt in range(1, retries + 1):
            try:
                r = client.get(url)
                r.raise_for_status()
                # basic content-type guard
                ctype = r.headers.get("content-type", "")
                if "text/html" not in ctype and "xml" not in ctype:
                    log(f"[warn] non-HTML content-type for {url}: {ctype}")
                return r.text
            except Exception as e:
                last_exc = e
                sleep_for = backoff * attempt
                log(f"[retry {attempt}/{retries}] {url}: {e!r} -> sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


# ---------- extraction ----------

NHS_STRIP_SELECTORS = [
    "header", "footer", "nav",
    ".nhsuk-c-cookie-banner", ".nhsuk-global-alert",
    ".nhsuk-feedback-banner", ".nhsuk-cookie-banner",
]
CDC_STRIP_SELECTORS = [
    "header", "footer", "nav",
    ".cmp-cookie", ".usa-banner", ".usa-alert",
]

def soup_clean_text(soup: BeautifulSoup, domain: Optional[str]) -> str:
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # domain-specific junk
    if domain == "nhs":
        for sel in NHS_STRIP_SELECTORS:
            for el in soup.select(sel):
                el.decompose()
    if domain == "cdc":
        for sel in CDC_STRIP_SELECTORS:
            for el in soup.select(sel):
                el.decompose()

    # try likely main containers
    mains = soup.select("main, article, #maincontent, .nhsuk-width-container, .cdc-main, .content, .container")
    text = " ".join(m.get_text(" ", strip=True) for m in mains) or soup.get_text(" ", strip=True)
    return norm_ws(text)


def extract_with_readability(url: str, html: str, domain: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Returns (text, title) using readability if available, else ("", None)
    """
    if not HAVE_READABILITY:
        return "", None
    try:
        doc = ReadabilityDocument(html, url=url)
        title = norm_ws(doc.short_title() or "") or None
        content_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(content_html, "lxml")
        text = norm_ws(soup.get_text(" ", strip=True))
        return text, title
    except Exception:
        return "", None


def extract_text_and_title(url: str, html: str, domain: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (text, title, canonical_url?)
    """
    soup_full = BeautifulSoup(html, "lxml")

    # canonical URL if present
    canonical = None
    link = soup_full.find("link", rel=lambda x: x and "canonical" in x)
    if link and link.get("href"):
        canonical = link["href"].strip()

    # try readability first
    text_rd, title_rd = extract_with_readability(url, html, domain)
    if len(text_rd) >= 500:
        return text_rd, title_rd, canonical

    # fallback: full-page soup with domain-specific cleaning
    text_bs = soup_clean_text(soup_full, domain)

    # title fallback
    title_tag = soup_full.find("title")
    title_bs = norm_ws(title_tag.get_text(" ", strip=True)) if title_tag else None

    # choose best
    text = text_rd if len(text_rd) > len(text_bs) else text_bs
    title = title_rd or title_bs
    return text, title, canonical


# ---------- writer ----------

def write_json(out_path: Path, payload: Dict) -> None:
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- main ----------

def run(args: argparse.Namespace) -> None:
    rows = read_sources_csv(SRC)

    # filters
    if args.only_domain:
        rows = [r for r in rows if (r.domain or "").lower() == args.only_domain.lower()]
    if args.only_topic:
        rows = [r for r in rows if (r.topic or "").lower() == args.only_topic.lower()]
    if args.max and args.max > 0:
        rows = rows[: args.max]

    total = saved = skipped = 0

    for row in rows:
        total += 1
        url = row.url
        out = RAW / f"{sha16(url)}.json"
        if out.exists() and not args.overwrite:
            log(f"[skip] exists: {out.name} (use --overwrite to refresh)")
            skipped += 1
            continue

        try:
            log(f"[fetch] {url}")
            html = fetch_html(url, DEFAULT_HEADERS, timeout_s=args.timeout, retries=args.retries, backoff=args.backoff)
            text, title_auto, canonical = extract_text_and_title(url, html, row.domain)
            if len(text) < 200:
                log(f"[skip] too short: {url} (len={len(text)})")
                skipped += 1
                continue

            sha_text = sha16(text)
            # if file exists, check if changed
            if out.exists() and not args.overwrite:
                try:
                    prev = json.loads(out.read_text(encoding="utf-8"))
                    if prev.get("sha_text") == sha_text:
                        log(f"[skip] unchanged: {url}")
                        skipped += 1
                        continue
                except Exception:
                    pass

            payload = {
                "url": url,
                "canonical_url": canonical,
                "domain": row.domain,
                "topic": row.topic,
                "title": row.title or title_auto,
                "text": text,
                "sha_text": sha_text,
                "fetched_at": now_iso(),
                "source_row": {
                    "title": row.title,
                    "domain": row.domain,
                    "topic": row.topic,
                },
            }
            write_json(out, payload)
            log(f"[saved] {url} -> {out.name} (chars={len(text)})")
            saved += 1

            if args.delay > 0:
                time.sleep(args.delay)

        except Exception as e:
            log(f"[error] {url}: {e!r}")

    log(f"[done] rows={total}, saved={saved}, skipped={skipped}, out_dir={RAW}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Doc_GPT crawler")
    p.add_argument("--max", type=int, default=0, help="Max number of rows to process (0 = all)")
    p.add_argument("--only-domain", type=str, default=None, help="Only crawl rows with this domain (e.g., nhs, cdc)")
    p.add_argument("--only-topic", type=str, default=None, help="Only crawl rows with this topic")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files")
    p.add_argument("--delay", type=float, default=0.3, help="Delay between saves (seconds)")
    p.add_argument("--timeout", type=float, default=25.0, help="Per-request timeout seconds")
    p.add_argument("--retries", type=int, default=3, help="HTTP retries on failure")
    p.add_argument("--backoff", type=float, default=0.8, help="Backoff multiplier between retries")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
