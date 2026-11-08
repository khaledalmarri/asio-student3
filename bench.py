#!/usr/bin/env python3
"""ASIO Student 3 - serving benchmark (Step 3).

Sends streaming requests to an OpenAI-compatible server (vLLM) at several
concurrency levels and records, per level:
  TTFT  - time to first token (prefill + queueing)
  TPOT  - time per output token after the first (decode speed per request)
  throughput - output tokens per second across all requests
  ok / failed request counts (a run with failures is not a valid result)

Prompts are random token IDs of an exact length, so every request has the
same prompt size and the prefix cache cannot make repeats look fast.
Standard library only: runs with any Python 3.8+.
"""
import argparse, json, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from datetime import datetime, timezone


def one_request(url, model, prompt_tokens, output_tokens, timeout):
    body = {
        "model": model,
        "prompt": [random.randint(1000, 30000) for _ in range(prompt_tokens)],
        "max_tokens": output_tokens,
        "ignore_eos": True,          # always generate exactly output_tokens
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    usage_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
    except Exception as e:  # count it, never hide it
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    if first is None:
        return {"ok": False, "error": "no tokens received"}
    n = usage_tokens or chunks
    return {
        "ok": True,
        "ttft_ms": (first - t0) * 1000,
        "tpot_ms": (last - first) * 1000 / (n - 1) if n > 1 else None,
        "e2e_ms": (last - t0) * 1000,
        "out_tokens": n,
    }


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 2)


def _client_process(job):
    """One client process: `threads` concurrent streams, `n` requests in total."""
    url, model, prompt_tokens, output_tokens, timeout, threads, n = job
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(lambda _: one_request(url, model, prompt_tokens, output_tokens, timeout), range(n)))


def run_level(args, concurrency):
    n_requests = args.requests or max(4 * concurrency, 16)
    # Split the load across several client processes so one Python process (GIL)
    # does not become the bottleneck at high concurrency.
    procs = max(1, min(args.procs, concurrency))
    jobs = [(args.url, args.model, args.prompt_tokens, args.output_tokens, args.timeout,
             concurrency // procs + (1 if i < concurrency % procs else 0),
             n_requests // procs + (1 if i < n_requests % procs else 0)) for i in range(procs)]
    if procs == 1:
        start = time.perf_counter()
        results = _client_process(jobs[0])
        wall = time.perf_counter() - start
    else:
        with Pool(procs) as pool:          # processes start here, outside the timed region
            pool.map(abs, range(procs))    # make sure every worker is up
            start = time.perf_counter()
            results = [r for part in pool.map(_client_process, jobs, chunksize=1) for r in part]
            wall = time.perf_counter() - start
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    return {
        "concurrency": concurrency,
        "client_procs": procs,
        "requests": n_requests,
        "ok": len(ok),
        "failed": len(bad),
        "first_error": bad[0]["error"] if bad else None,
        "wall_s": round(wall, 3),
        "throughput_tok_s": round(sum(r["out_tokens"] for r in ok) / wall, 1),
        "ttft_p50_ms": pct([r["ttft_ms"] for r in ok], 50),
        "ttft_p95_ms": pct([r["ttft_ms"] for r in ok], 95),
        "tpot_p50_ms": pct([r["tpot_ms"] for r in ok], 50),
        "tpot_p95_ms": pct([r["tpot_ms"] for r in ok], 95),
        "e2e_p95_ms": pct([r["e2e_ms"] for r in ok], 95),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8401")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="config name, e.g. qwen0.5b-tp2")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--requests", type=int, default=0, help="requests per level (default 4x concurrency, min 16)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--procs", type=int, default=1, help="client processes to split the load across (use 4+ above ~15k tok/s)")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    print(f"warming up {args.url} ...")
    for _ in range(2):
        r = one_request(args.url, args.model, args.prompt_tokens, args.output_tokens, args.timeout)
        if not r["ok"]:
            raise SystemExit(f"warm-up failed: {r['error']}")

    print(f"{'conc':>5} {'ok/fail':>9} {'tok/s':>9} {'TTFT p50':>9} {'TTFT p95':>9} {'TPOT p50':>9} {'TPOT p95':>9}")
    for c in args.concurrency:
        row = run_level(args, c)
        print(f"{c:>5} {row['ok']:>4}/{row['failed']:<4} {row['throughput_tok_s']:>9} "
              f"{row['ttft_p50_ms']!s:>9} {row['ttft_p95_ms']!s:>9} {row['tpot_p50_ms']!s:>9} {row['tpot_p95_ms']!s:>9}")
        if row["failed"]:
            print(f"      first error: {row['first_error']}")
        record = {"label": args.label, "model": args.model, "prompt_tokens": args.prompt_tokens,
                  "output_tokens": args.output_tokens,
                  "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
    print(f"appended {len(args.concurrency)} rows to {args.out}")


if __name__ == "__main__":
    main()
