"""
Quick smoke-test evaluator for /ask:
- Sends 5 common primary-care prompts
- Prints citation coverage metrics
Usage:
  python scripts/eval_docgpt.py --host 127.0.0.1 --port 8000 --k 6
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import httpx

def ask(base: str, q: str, k: int = 6) -> dict:
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{base}/ask", json={"query": q, "k": k})
        r.raise_for_status()
        return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    ap.add_argument("--k", default=6, type=int)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"status         :", end=" ")

    try:
        r = httpx.get(f"{base}/healthz", timeout=10.0)
        r.raise_for_status()
        j = r.json()
        print("ok"); 
        for k in ("backend","mode","alpha","model","openai_base_url","count","dim","dir","bm25"):
            if k in j: print(f"{k:15}: {j[k]}")
    except Exception as e:
        print("UNAVAILABLE", e)
        return

    prompts = [
        "I have a sore throat and a mild fever. What can I do at home?",
        "I have chest pain and shortness of breath. What should I do?",
        "I'm wheezing and my chest feels tight, should I go to the hospital?",
        "I think I have flu with fever and aches. Any home care tips?",
        "I have vomiting and can't keep fluids down, when should I worry about dehydration?",
    ]

    outdir = Path("data/eval_runs") / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    hit_any = 0
    hit_at1 = 0

    for i, q in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {q}")
        try:
            res = ask(base, q, k=args.k)
        except Exception as e:
            print("   ERROR:", e)
            continue

        # write each raw result
        (outdir / f"{i:03d}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

        cits = res.get("citations", [])
        if cits:
            hit_any += 1
            # very rough @1 proxy: if the first retrieved citation's URL also appears in the answer text
            url0 = cits[0].get("url","")
            if url0 and url0.lower() in (res.get("answer","").lower()):
                hit_at1 += 1

    N = len(prompts)
    report = {
        "items": N,
        "citation_at1_rate": round(hit_at1 / N, 3),
        "citation_any_rate": round(hit_any / N, 3),
    }
    print("\n=== Doc_GPT Eval Summary ===")
    print(f"Items:              {report['items']}")
    print(f"Citation@1 rate:    {report['citation_at1_rate']:.3f}")
    print(f"Citation ANY rate:  {report['citation_any_rate']:.3f}")

    Path("data/eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nRaw run folder:", str(outdir))
    print("Report:       ", "data/eval_report.json")

if __name__ == "__main__":
    main()
