"""Stress test the RAG pipeline while capturing machine + process metrics.

What it does:
  1. Seeds the vector store with sample documents (so retrieval is realistic).
  2. Starts a background sampler that records system CPU/RAM and the CPU/RAM of
     the Python (this process) and Ollama processes once per interval.
  3. Fires N queries at a configurable concurrency against src.rag.answer().
  4. Writes a report to the Downloads folder:
       - rag_stress_report_<ts>.json   (summary + config + env)
       - rag_stress_queries_<ts>.csv    (per-query latency + success)
       - rag_stress_resources_<ts>.csv  (time series of resource samples)
       - rag_stress_latency_<ts>.png    (latency distribution + per-query)
       - rag_stress_resources_<ts>.png  (CPU/RAM over time)
       - rag_stress_throughput_<ts>.png (completed queries over time)

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\stress_test.py --total 30 --concurrency 4

Flags:
    --total N        total number of queries to issue   (default 30)
    --concurrency K  number of concurrent workers        (default 4)
    --interval S     resource sampling interval seconds   (default 0.5)
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import NUM_CTX, NUM_PREDICT, OLLAMA_MODEL, TOP_K
from src.loaders import load_file
from src.rag import answer
from src.vectorstore import add_documents, collection_count, reset_vectorstore

# A pool of varied questions so the model isn't just answering an identical
# cached prompt every time.
QUESTIONS = [
    "Which department does Carol work in?",
    "Who is based in Austin?",
    "List everyone in the Engineering department.",
    "What city is Alice in?",
    "How many people work in Sales?",
    "Which employees live in Denver?",
    "What is Bob's department?",
    "Summarize the people and their cities.",
]


def downloads_dir() -> Path:
    """Resolve the user's Downloads folder on Windows (fallback to ~/Downloads)."""
    candidate = Path.home() / "Downloads"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def seed_corpus() -> int:
    """Reset and seed the vector store with a small sample dataset."""
    reset_vectorstore()
    sample = ROOT / "data" / "uploads" / "_stress_employees.csv"
    pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank"],
            "department": ["Engineering", "Sales", "Engineering", "Sales", "Engineering", "Support"],
            "city": ["Seattle", "Austin", "Denver", "Austin", "Seattle", "Denver"],
            "years": [5, 3, 7, 2, 4, 6],
        }
    ).to_csv(sample, index=False)
    added = add_documents(load_file(sample))
    sample.unlink(missing_ok=True)
    return added


def find_ollama_procs() -> list[psutil.Process]:
    """Return running Ollama processes (server + runner)."""
    procs = []
    for p in psutil.process_iter(["name"]):
        try:
            if p.info["name"] and "ollama" in p.info["name"].lower():
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


class ResourceSampler(threading.Thread):
    """Background thread that samples system + process metrics at an interval."""

    def __init__(self, interval: float, start_time: float):
        super().__init__(daemon=True)
        self.interval = interval
        self.start_time = start_time
        self.samples: list[dict] = []
        # NOTE: don't name this `_stop` — that shadows threading.Thread._stop().
        self._stop_evt = threading.Event()
        self.proc = psutil.Process()
        self.ollama = find_ollama_procs()
        # Prime cpu_percent so the first real reading isn't 0.0.
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)
        for o in self.ollama:
            try:
                o.cpu_percent(interval=None)
            except psutil.Error:
                pass
        self.ncpu = psutil.cpu_count(logical=True) or 1

    def _ollama_cpu_mem(self) -> tuple[float, float]:
        cpu = 0.0
        mem = 0.0
        for o in list(self.ollama):
            try:
                cpu += o.cpu_percent(interval=None)
                mem += o.memory_info().rss / (1024 ** 2)  # MB
            except psutil.Error:
                continue
        return cpu, mem

    def run(self) -> None:
        while not self._stop_evt.is_set():
            vm = psutil.virtual_memory()
            o_cpu, o_mem = self._ollama_cpu_mem()
            self.samples.append(
                {
                    "t": round(time.perf_counter() - self.start_time, 3),
                    "sys_cpu_pct": psutil.cpu_percent(interval=None),
                    "sys_mem_pct": vm.percent,
                    "sys_mem_used_mb": round(vm.used / (1024 ** 2), 1),
                    "proc_cpu_pct": round(self.proc.cpu_percent(interval=None) / self.ncpu, 1),
                    "proc_mem_mb": round(self.proc.memory_info().rss / (1024 ** 2), 1),
                    "ollama_cpu_pct": round(o_cpu / self.ncpu, 1),
                    "ollama_mem_mb": round(o_mem, 1),
                }
            )
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()


def run_query(idx: int, start_time: float) -> dict:
    """Issue one query and record its latency and outcome."""
    q = QUESTIONS[idx % len(QUESTIONS)]
    t0 = time.perf_counter()
    ok = True
    err = ""
    try:
        res = answer(q)
        ans_len = len(res.answer)
    except Exception as exc:  # capture failures instead of crashing the run
        ok = False
        err = str(exc)
        ans_len = 0
    latency = time.perf_counter() - t0
    return {
        "idx": idx,
        "question": q,
        "start_s": round(t0 - start_time, 3),
        "end_s": round(time.perf_counter() - start_time, 3),
        "latency_s": round(latency, 3),
        "ok": ok,
        "answer_chars": ans_len,
        "error": err,
    }


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def make_graphs(out_dir: Path, ts: str, queries: list[dict], samples: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")  # headless backend; no display needed
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    latencies = [q["latency_s"] for q in queries if q["ok"]]

    # 1) Latency: histogram + per-query scatter.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if latencies:
        ax1.hist(latencies, bins=min(15, max(5, len(latencies) // 2)), color="#2563EB", edgecolor="white")
        ax1.axvline(statistics.mean(latencies), color="#EF4444", linestyle="--", label=f"mean {statistics.mean(latencies):.2f}s")
        ax1.axvline(percentile(latencies, 95), color="#F59E0B", linestyle="--", label=f"p95 {percentile(latencies, 95):.2f}s")
        ax1.legend()
    ax1.set_title("Query latency distribution")
    ax1.set_xlabel("latency (s)")
    ax1.set_ylabel("count")

    order = sorted([q for q in queries if q["ok"]], key=lambda x: x["idx"])
    ax2.plot([q["idx"] for q in order], [q["latency_s"] for q in order], marker="o", color="#2563EB", ms=4)
    ax2.set_title("Latency per query (in order issued)")
    ax2.set_xlabel("query #")
    ax2.set_ylabel("latency (s)")
    fig.tight_layout()
    p = out_dir / f"rag_stress_latency_{ts}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 2) Resource time series: CPU and memory.
    if samples:
        t = [s["t"] for s in samples]
        fig, (axc, axm) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axc.plot(t, [s["sys_cpu_pct"] for s in samples], label="system CPU %", color="#2563EB")
        axc.plot(t, [s["ollama_cpu_pct"] for s in samples], label="ollama CPU %", color="#10B981")
        axc.plot(t, [s["proc_cpu_pct"] for s in samples], label="harness CPU %", color="#F59E0B")
        axc.set_ylabel("CPU %")
        axc.set_title("CPU utilization during stress test")
        axc.legend(loc="upper right")

        axm.plot(t, [s["sys_mem_used_mb"] for s in samples], label="system mem used (MB)", color="#2563EB")
        axm.plot(t, [s["ollama_mem_mb"] for s in samples], label="ollama mem (MB)", color="#10B981")
        axm.plot(t, [s["proc_mem_mb"] for s in samples], label="harness mem (MB)", color="#F59E0B")
        axm.set_ylabel("memory (MB)")
        axm.set_xlabel("elapsed (s)")
        axm.set_title("Memory usage during stress test")
        axm.legend(loc="upper right")
        fig.tight_layout()
        p = out_dir / f"rag_stress_resources_{ts}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    # 3) Throughput: cumulative completed queries over time.
    done = sorted([q["end_s"] for q in queries if q["ok"]])
    if done:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(done, range(1, len(done) + 1), drawstyle="steps-post", color="#2563EB")
        ax.set_title("Cumulative completed queries over time")
        ax.set_xlabel("elapsed (s)")
        ax.set_ylabel("queries completed")
        fig.tight_layout()
        p = out_dir / f"rag_stress_throughput_{ts}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG pipeline stress test with metrics capture.")
    ap.add_argument("--total", type=int, default=30, help="total queries to issue")
    ap.add_argument("--concurrency", type=int, default=4, help="concurrent workers")
    ap.add_argument("--interval", type=float, default=0.5, help="resource sampling interval (s)")
    args = ap.parse_args()

    out_dir = downloads_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Seeding corpus... ", end="", flush=True)
    chunks = seed_corpus()
    print(f"{chunks} chunks ({collection_count()} in store)")

    # Warm up once so the model is loaded; the timed run measures steady state.
    print("Warming up model (one query)... ", end="", flush=True)
    _ = answer(QUESTIONS[0])
    print("done")

    print(
        f"Running: total={args.total} concurrency={args.concurrency} "
        f"model={OLLAMA_MODEL} top_k={TOP_K} num_predict={NUM_PREDICT} num_ctx={NUM_CTX}"
    )

    start = time.perf_counter()
    sampler = ResourceSampler(args.interval, start)
    sampler.start()

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_query, i, start) for i in range(args.total)]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            results.append(r)
            status = "ok" if r["ok"] else "FAIL"
            print(f"  [{i}/{args.total}] q#{r['idx']} {status} {r['latency_s']:.2f}s")

    wall = time.perf_counter() - start
    sampler.stop()
    sampler.join(timeout=2)

    results.sort(key=lambda x: x["idx"])
    latencies = [r["latency_s"] for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    succeeded = len(latencies)

    summary = {
        "timestamp": ts,
        "config": {
            "total_queries": args.total,
            "concurrency": args.concurrency,
            "sampling_interval_s": args.interval,
            "model": OLLAMA_MODEL,
            "top_k": TOP_K,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "python": platform.python_version(),
        },
        "results": {
            "succeeded": succeeded,
            "failed": len(failures),
            "wall_time_s": round(wall, 2),
            "throughput_qps": round(succeeded / wall, 3) if wall > 0 else 0,
            "latency_s": {
                "min": round(min(latencies), 3) if latencies else None,
                "mean": round(statistics.mean(latencies), 3) if latencies else None,
                "median": round(statistics.median(latencies), 3) if latencies else None,
                "p95": round(percentile(latencies, 95), 3) if latencies else None,
                "p99": round(percentile(latencies, 99), 3) if latencies else None,
                "max": round(max(latencies), 3) if latencies else None,
            },
        },
        "peak_resources": {
            "sys_cpu_pct": max((s["sys_cpu_pct"] for s in sampler.samples), default=None),
            "sys_mem_pct": max((s["sys_mem_pct"] for s in sampler.samples), default=None),
            "ollama_mem_mb": max((s["ollama_mem_mb"] for s in sampler.samples), default=None),
            "harness_mem_mb": max((s["proc_mem_mb"] for s in sampler.samples), default=None),
        },
        "errors": [f["error"] for f in failures][:10],
    }

    # Write report files.
    json_path = out_dir / f"rag_stress_report_{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    q_csv = out_dir / f"rag_stress_queries_{ts}.csv"
    pd.DataFrame(results).to_csv(q_csv, index=False)

    r_csv = out_dir / f"rag_stress_resources_{ts}.csv"
    pd.DataFrame(sampler.samples).to_csv(r_csv, index=False)

    graphs = make_graphs(out_dir, ts, results, sampler.samples)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary["results"], indent=2))
    print("\nReports written to:", out_dir)
    for p in [json_path, q_csv, r_csv, *graphs]:
        print("  -", p.name)
    print("\nSTRESS_TEST_DONE")


if __name__ == "__main__":
    main()
