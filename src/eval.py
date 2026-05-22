"""
Evaluation harness for FiQA hybrid retrieval.

Metrics reported:
  - Recall@10, MRR (overall and stratified)
  - p50/p95 latency: cold (first 20 queries) and warm (after 100-query warmup)
  - Peak RAM after index load (psutil RSS, or tracemalloc fallback)

Ablation: hybrid alpha sweep over [0.3, 0.5, 0.7].

Outputs: results/bench.json  (quality + latency)
         results/failure_candidates.json  (raw data for failures.md)
"""

import json
import os
import pickle
import sys
import time
import tracemalloc

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

DATA_DIR = os.path.join(ROOT, "data", "fiqa")
RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Metric helpers (imported by tests/test_eval_metrics.py)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list, relevant_ids: set) -> float:
    """Reciprocal rank of the first relevant hit (0 if none found)."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _measure_ram_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)


def load_indexes():
    tracemalloc.start()
    ram_before = _measure_ram_mb()

    with open(os.path.join(DATA_DIR, "doc_ids.json"), encoding="utf-8") as f:
        doc_ids = json.load(f)
    with open(os.path.join(DATA_DIR, "dev_queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(DATA_DIR, "dev_qrels.json"), encoding="utf-8") as f:
        qrels = json.load(f)

    print("Loading BM25 index...")
    with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "rb") as f:
        bm = pickle.load(f)

    print("Loading dense embeddings + model...")
    dn = DenseRetriever(model_name="all-MiniLM-L6-v2", use_model=True)
    dn.load_index(DATA_DIR, bm.docs)

    ram_after = _measure_ram_mb()
    tracemalloc.stop()
    peak_ram_mb = ram_after - ram_before

    print(f"Indexes loaded. RAM delta: {peak_ram_mb:.0f} MB  (total RSS: {ram_after:.0f} MB)")
    return bm, dn, doc_ids, queries, qrels, ram_after


# ---------------------------------------------------------------------------
# Per-retriever evaluation
# ---------------------------------------------------------------------------

def _pct(values: list, p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def _mean(values: list) -> float:
    return float(np.mean(values)) if values else 0.0


def eval_retriever(retriever, queries: dict, qrels: dict, doc_ids: list,
                   docs: list, top_k: int = 10, label: str = ""):
    """
    Run full evaluation over all dev queries.

    Returns (summary_dict, per_query_dict, cold_times, warm_times).

    Cold  = first 20 queries (no warmup).
    Warm  = queries 121+ (after 100-query warmup window of queries 21-120).
    """
    doc_lens = [len(d.split()) for d in docs]
    len_threshold = float(np.percentile(doc_lens, 90))
    long_doc_ids = {did for did, ln in zip(doc_ids, doc_lens) if ln >= len_threshold}
    id_to_text = {did: docs[i] for i, did in enumerate(doc_ids)}
    id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    query_ids = list(queries.keys())
    per_query = {}
    cold_times, warm_times = [], []

    print(f"  Evaluating {label} over {len(query_ids)} queries...")
    for i, qid in enumerate(query_ids):
        query_text = queries[qid]
        relevant = set(qrels.get(qid, {}).keys())

        t0 = time.perf_counter()
        hits = retriever.search(query_text, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_doc_ids = [doc_ids[idx] for idx, _ in hits]

        r10 = recall_at_k(retrieved_doc_ids, relevant, top_k)
        rr = mrr(retrieved_doc_ids, relevant)
        query_len = len(query_text.split())
        has_long_gold = bool(relevant & long_doc_ids)

        per_query[qid] = {
            "recall_at_10": r10,
            "mrr": rr,
            "latency_ms": elapsed_ms,
            "query_len": query_len,
            "has_long_gold": has_long_gold,
            "retrieved_doc_ids": retrieved_doc_ids,
            "relevant": list(relevant),
        }

        if i < 20:
            cold_times.append(elapsed_ms)
        elif i < 120:
            pass  # warmup window — intentionally not measured
        else:
            warm_times.append(elapsed_ms)

    all_lat = [v["latency_ms"] for v in per_query.values()]
    all_r10 = [v["recall_at_10"] for v in per_query.values()]
    all_mrr = [v["mrr"] for v in per_query.values()]

    short_r10 = [v["recall_at_10"] for v in per_query.values() if v["query_len"] < 5]
    med_r10 = [v["recall_at_10"] for v in per_query.values() if 5 <= v["query_len"] <= 15]
    long_r10 = [v["recall_at_10"] for v in per_query.values() if v["query_len"] > 15]
    long_doc_r10 = [v["recall_at_10"] for v in per_query.values() if v["has_long_gold"]]
    short_doc_r10 = [v["recall_at_10"] for v in per_query.values() if not v["has_long_gold"]]

    summary = {
        "recall_at_10": _mean(all_r10),
        "mrr": _mean(all_mrr),
        "p50_ms": _pct(all_lat, 50),
        "p95_ms": _pct(all_lat, 95),
        "cold_p50_ms": _pct(cold_times, 50),
        "cold_p95_ms": _pct(cold_times, 95),
        "warm_p50_ms": _pct(warm_times, 50),
        "warm_p95_ms": _pct(warm_times, 95),
        "n_queries": len(per_query),
        "stratified": {
            "query_short":  {"n": len(short_r10),    "recall_at_10": _mean(short_r10)},
            "query_medium": {"n": len(med_r10),      "recall_at_10": _mean(med_r10)},
            "query_long":   {"n": len(long_r10),     "recall_at_10": _mean(long_r10)},
            "long_docs":    {"n": len(long_doc_r10), "recall_at_10": _mean(long_doc_r10)},
            "short_docs":   {"n": len(short_doc_r10),"recall_at_10": _mean(short_doc_r10)},
        },
    }
    return summary, per_query, cold_times, warm_times


# ---------------------------------------------------------------------------
# Ablation: hybrid alpha sweep
# ---------------------------------------------------------------------------

def ablation_alpha_sweep(bm, dn, queries, qrels, doc_ids, docs,
                         alphas=(0.3, 0.5, 0.7), top_k=10):
    results = {}
    for alpha in alphas:
        hy = HybridRetriever(bm, dn, alpha=alpha)
        key = f"alpha_{alpha}"
        print(f"  Ablation hybrid {key}...")
        summary, _, _, _ = eval_retriever(
            hy, queries, qrels, doc_ids, docs, top_k=top_k, label=key
        )
        results[key] = {
            "alpha": alpha,
            "recall_at_10": summary["recall_at_10"],
            "mrr": summary["mrr"],
            "p50_ms": summary["p50_ms"],
            "p95_ms": summary["p95_ms"],
        }
    return results


# ---------------------------------------------------------------------------
# Failure candidate collection
# ---------------------------------------------------------------------------

def collect_failures(per_query: dict, queries: dict, doc_ids: list,
                     docs: list, qrels: dict, n: int = 10) -> list:
    """Return up to n queries where recall@10 == 0, with retrieval detail."""
    id_to_text = {did: docs[i] for i, did in enumerate(doc_ids)}
    id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    failed = [
        (qid, data) for qid, data in per_query.items()
        if data["recall_at_10"] == 0.0
    ]
    # Sort by query length (interesting variety)
    failed.sort(key=lambda x: x[1]["query_len"])
    candidates = []

    for qid, data in failed[:n]:
        relevant = set(data["relevant"])
        retrieved = data["retrieved_doc_ids"]

        top5 = []
        for rank, did in enumerate(retrieved[:5], start=1):
            top5.append({
                "rank": rank,
                "doc_id": did,
                "text_snippet": id_to_text.get(did, "")[:300],
            })

        gold_entries = []
        for gold_did in sorted(relevant):
            gold_rank = next(
                (r + 1 for r, d in enumerate(retrieved) if d == gold_did), -1
            )
            gold_entries.append({
                "doc_id": gold_did,
                "rank_in_top10": gold_rank,
                "text_snippet": id_to_text.get(gold_did, "")[:300],
            })

        candidates.append({
            "query_id": qid,
            "query_text": queries.get(qid, ""),
            "top5_retrieved": top5,
            "gold_passages": gold_entries,
        })

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bm, dn, doc_ids, queries, qrels, peak_ram_mb = load_indexes()
    docs = bm.docs
    top_k = 10

    print("\n--- BM25 ---")
    bm25_summary, _, _, _ = eval_retriever(
        bm, queries, qrels, doc_ids, docs, top_k=top_k, label="BM25"
    )
    bm25_summary["peak_ram_mb"] = peak_ram_mb

    print("\n--- Dense (all-MiniLM-L6-v2) ---")
    dense_summary, _, _, _ = eval_retriever(
        dn, queries, qrels, doc_ids, docs, top_k=top_k, label="Dense"
    )
    dense_summary["peak_ram_mb"] = peak_ram_mb

    print("\n--- Hybrid (alpha=0.5) ---")
    hy = HybridRetriever(bm, dn, alpha=0.5)
    hybrid_summary, hybrid_per_query, _, _ = eval_retriever(
        hy, queries, qrels, doc_ids, docs, top_k=top_k, label="Hybrid"
    )
    hybrid_summary["peak_ram_mb"] = peak_ram_mb

    print("\n--- Ablation: alpha sweep ---")
    ablation = ablation_alpha_sweep(bm, dn, queries, qrels, doc_ids, docs)

    bench = {
        "bm25": bm25_summary,
        "dense": dense_summary,
        "hybrid": hybrid_summary,
        "ablation": ablation,
    }

    bench_path = os.path.join(RESULTS_DIR, "bench.json")
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2)
    print(f"\nWrote {bench_path}")

    print("\n--- Collecting failure candidates ---")
    failures = collect_failures(hybrid_per_query, queries, doc_ids, docs, qrels, n=10)
    fail_path = os.path.join(RESULTS_DIR, "failure_candidates.json")
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    print(f"Wrote {fail_path}  ({len(failures)} failure candidates)")

    _print_summary(bench)


def _print_summary(bench: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'Retriever':<12} {'R@10':>6} {'MRR':>6} {'p50ms':>7} {'p95ms':>7}")
    print("-" * 60)
    for name in ("bm25", "dense", "hybrid"):
        s = bench[name]
        print(
            f"{name:<12} {s['recall_at_10']:>6.3f} {s['mrr']:>6.3f}"
            f" {s['p50_ms']:>7.1f} {s['p95_ms']:>7.1f}"
        )
    print("\nAblation (hybrid alpha sweep):")
    for key, v in bench["ablation"].items():
        print(f"  {key}: R@10={v['recall_at_10']:.3f}  MRR={v['mrr']:.3f}  p95={v['p95_ms']:.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()
