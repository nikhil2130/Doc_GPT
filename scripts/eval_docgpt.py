# -*- coding: utf-8 -*-
"""
Doc_GPT quick evaluation harness.

What it does:
- Loads eval items from data/eval.jsonl (one JSON per line)
- Calls your running API at /ask for each query
- Saves raw results to data/eval_runs/<timestamp>/*.json
- Computes simple metrics:
    * citation@1 domain match (does top citation come from an expected domain?)
    * citation_any domain match (does ANY citation match an expected domain?)
    * red-flag detection hit rate (if red flag expected)
    * latency stats
    * answer length stats
- Writes a summary report to data/eval_report.json and prints a concise table

Usage (PowerShell):
    cd D:\projects\Doc_GPT
    .\.venv\Scripts\Activate.ps1
    python .\scripts\eval_docgpt.py --host 127.0.0.1 --port 8000 --k 6
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
import datetime as dt

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVAL_FILE = DATA / "eval.jsonl"
RUNS_DIR = DATA / "eval_runs"
REPORT_FILE = DATA / "eval_report.json"

def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url or "").netloc.lower()
        # strip port
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # drop leading www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def load_items(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        raise SystemExit(f"[fatal] eval file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                items.append(j)
            except Exception as e:
                print(f"[warn] bad line: {e}")
    if not items:
        raise SystemExit("[fatal] empty eval set")
    return items

def ask(base_url: str, question: str, k: int, timeout: float = 40.0) -> Tuple[Dict[str, Any], float]:
    payload = {"query": question, "k": int(k)}
    st = perf_counter()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{base_url}/ask", json=payload)
            r.raise_for_status()
            js = r.json()
    except Exception as e:
        return ({"_error": repr(e)}, float("nan"))
    dur = perf_counter() - st
    return (js, dur)

def compute_metrics(items: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    c_at1 = 0
    c_any = 0
    rf_hit = 0
    rf_total = 0
    latencies: List[float] = []
    lengths: List[int] = []

    rows: List[Dict[str, Any]] = []

    for it, res in zip(items, results):
        q = it.get("query", "")
        expect_domains = [d.lower() for d in it.get("expect_domains", [])]
        expect_rf = it.get("expect_redflag", "none")  # "none" | "maybe" | "emergency"
        item_id = it.get("id")

        row: Dict[str, Any] = {
            "id": item_id, "query": q, "expect_domains": expect_domains,
            "expect_redflag": expect_rf,
            "citation@1": False, "citation_any": False, "redflag_hit": None,
        }

        if "_error" in res:
            row["error"] = res["_error"]
            rows.append(row)
            continue

        # latency & length
        if isinstance(res.get("_latency"), (int, float)):
            latencies.append(res["_latency"])
        if isinstance(res.get("answer"), str):
            lengths.append(len(res["answer"]))

        # citations
        cites = res.get("citations") or []
        got_domains = [domain_of(c.get("url", "")) for c in cites]
        got_domains = [d for d in got_domains if d]
        at1_match = (len(got_domains) > 0 and any(got_domains[0].endswith(ed) for ed in expect_domains))
        any_match = any(any(gd.endswith(ed) for ed in expect_domains) for gd in got_domains)

        row["got_domains"] = got_domains
        row["citation@1"] = bool(at1_match)
        row["citation_any"] = bool(any_match)
        c_at1 += 1 if at1_match else 0
        c_any += 1 if any_match else 0

        # red-flag metric
        rf = (res.get("red_flag") or {})
        severity = (rf.get("severity") or "none").lower()
        if expect_rf != "none":
            rf_total += 1
            # count as hit if we flagged when we expected ("maybe" or "emergency")
            rf_hit += 1 if severity in ("urgent", "emergency") else 0
        row["redflag_observed"] = severity
        row["redflag_hit"] = None if expect_rf == "none" else (severity in ("urgent", "emergency"))

        rows.append(row)

    metrics = {
        "n": n,
        "citation_at1_rate": (c_at1 / n) if n else 0.0,
        "citation_any_rate": (c_any / n) if n else 0.0,
        "redflag_hit_rate": (rf_hit / rf_total) if rf_total else None,
        "latency_ms_avg": (sum(latencies) / len(latencies) * 1000.0) if latencies else None,
        "latency_ms_p95": None,
        "answer_len_avg": (sum(lengths) / len(lengths)) if lengths else None,
        "rows": rows,
    }
    if latencies:
        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(0.95 * (len(sorted_lat)-1))]
        metrics["latency_ms_p95"] = p95 * 1000.0
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    items = load_items(EVAL_FILE)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, it in enumerate(items, start=1):
        q = it.get("query", "")
        print(f"[{idx}/{len(items)}] {q}")
        js, dur = ask(base_url, q, k=args.k)
        if isinstance(js, dict):
            js["_latency"] = dur
        results.append(js)
        # save raw
        with (out_dir / f"{idx:03d}.json").open("w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)

    metrics = compute_metrics(items, results)
    # write report
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # pretty print
    print("\n=== Doc_GPT Eval Summary ===")
    print(f"Items:              {metrics['n']}")
    print(f"Citation@1 rate:    {metrics['citation_at1_rate']:.3f}")
    print(f"Citation ANY rate:  {metrics['citation_any_rate']:.3f}")
    if metrics.get("redflag_hit_rate") is not None:
        print(f"Red-flag hit rate:  {metrics['redflag_hit_rate']:.3f}")
    if metrics.get("latency_ms_avg") is not None:
        print(f"Avg latency:        {metrics['latency_ms_avg']:.1f} ms   (p95 {metrics['latency_ms_p95']:.1f} ms)")
    if metrics.get("answer_len_avg") is not None:
        print(f"Avg answer length:  {metrics['answer_len_avg']:.1f} chars")

    print(f"\nRaw run folder: {out_dir}")
    print(f"Report:        {REPORT_FILE}")

if __name__ == "__main__":
    main()
