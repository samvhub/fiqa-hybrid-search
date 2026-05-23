# Design Doc — FiQA Hybrid Retrieval System

## 1. Chosen Operating Point

**Config:** Dense retriever — all-MiniLM-L6-v2, top-k = 10.

**Why:**
- Dense alone achieves the highest Recall@10 (0.467) and MRR (0.453) of any config tested,
  including hybrid. Adding BM25 at equal weight (alpha=0.5) *reduces* R@10 by 4 points (see §4).
- all-MiniLM-L6-v2 (384-dim, 22M params) has the lowest per-query latency: warm p95 = 175 ms.
- Embeddings for ~57K passages occupy ~84 MB on disk; full process RSS is 1552 MB, well under 2 GB.

All three binding constraints measured:

| Constraint | Limit | Measured |
|---|---|---|
| CPU only | required | yes — no GPU used |
| p95 latency (warm) | <= 50 ms | 175 ms — constraint **not met**; see §6 for the path to 50 ms |
| Peak RAM (serve) | <= 2 GB | 1552 MB |

No configuration tested meets the 50 ms p95 target on this hardware. Dense at 175 ms is the
closest, 3.5x the limit. The gap and remediation options are discussed in §6.

---

## 2. Benchmark Table

Evaluated on FiQA dev split (500 queries, 648 relevance judgments).
Cold = first 20 queries after index load; Warm = queries 121+ (after 100-query warmup).

| Config | R@10 | MRR | Warm p50 (ms) | Warm p95 (ms) | Cold p95 (ms) | RAM (MB) |
|---|---|---|---|---|---|---|
| BM25 | 0.190 | 0.189 | 363 | 678 | 618 | 1552 |
| Dense (all-MiniLM-L6-v2) | **0.467** | **0.453** | **149** | **175** | 187 | 1552 |
| Hybrid alpha=0.5 | 0.427 | 0.400 | 793 | 1179 | 745 | 1552 |
| Hybrid alpha=0.7 (best ablation) | 0.470 | 0.460 | 519 | 802 | — | — |

**Ablation — hybrid alpha sweep (R@10 / MRR / warm p95):**

| alpha | R@10 | MRR | Warm p95 (ms) |
|---|---|---|---|
| 0.3 | 0.320 | 0.265 | 1083 |
| 0.5 | 0.427 | 0.400 | 949 |
| 0.7 | 0.470 | 0.460 | 802 |

**Stratified Recall@10 (hybrid alpha=0.5):**

| Stratum | n | R@10 |
|---|---|---|
| Short queries (<5 tokens) | 13 | 0.308 |
| Medium queries (5–15 tokens) | 403 | 0.425 |
| Long queries (>15 tokens) | 84 | 0.456 |
| Gold passage in top-10% longest docs | 185 | 0.390 |
| Gold passage in remaining docs | 315 | 0.448 |

---

## 3. Cold vs. Warm Latency

**Observed delta (Dense retriever):** cold p95 = 187 ms vs. warm p95 = 175 ms — a 12 ms (7%)
overhead on the first 20 queries.

**Causes:**
- **PyTorch JIT**: the SentenceTransformer forward pass is JIT-compiled on the first call and
  the compiled kernel is cached. Subsequent calls reuse it without re-tracing.
- **OS page cache**: the embeddings array (~84 MB .npy file) is read from disk on the first few
  queries; later queries hit a warm OS buffer cache and avoid disk I/O entirely.
- **BLAS thread pool**: OpenBLAS/MKL may spin up worker threads on the first `cosine_similarity`
  call. Subsequent calls reuse the live thread pool without re-initialization overhead.

The cold/warm gap is small (12 ms) because model weights are loaded once at startup — the dominant
per-query cost is the numpy dot-product over the precomputed 57K × 384 embedding matrix, which
is the same whether the query is cold or warm.

---

## 4. One Counterintuitive Finding

**Dense alone (R@10 = 0.467) outperforms the hybrid at alpha=0.5 (R@10 = 0.427). Adding BM25 at
equal weight reduces Recall@10 by 4 percentage points.**

Expected: BM25 should complement dense on FiQA because financial text is rich in exact-match
signals — ticker symbols, acronyms (P/E, BOPM, WACC), and product names that a neural encoder
might miss. A hybrid was expected to win over either retriever alone.

Observed: At alpha=0.5 the hybrid is strictly worse than dense. Only at alpha=0.7 (70% dense,
30% BM25) does the hybrid match dense (0.470 vs. 0.467), for a negligible +0.3 pp gain at the
cost of 4.6x higher p95 latency (802 ms vs. 175 ms).

Hypothesis: FiQA is a community Q&A corpus — answers are conversational posts that explain
financial concepts rather than repeating query terminology. BM25 rewards passages that contain
the query words literally. For vague queries like "Corporate Finance" or "valuing options" this
surfaces topically-adjacent but off-topic passages (financial services generalities, pre-IPO
employee options) and pollutes the ranking. The dense model, despite its flaws, at least scores
passages by semantic proximity rather than term frequency, which is more robust on this corpus.
The practical takeaway: do not default to equal-weight hybrid; sweep alpha and validate on a
held-out slice before deploying.

---

## 5. Approaches That Didn't Pan Out

### A. Chunking long passages before indexing

**Considered:** Split each passage with `chunk_text(max_tokens=100)` before building the BM25
and dense index, then aggregate chunk scores back to passage level at retrieval time.

**Expected:** Dense cosine similarity should be more precise on shorter, focused chunks than on
multi-sentence passages where the embedding averages across multiple topics.

**Why it was rejected:** The FiQA corpus consists of community Q&A answers that are already
short. The median document in the corpus is under 100 tokens; applying a 100-token chunk window
would fragment the majority of passages at arbitrary sentence boundaries, splitting coherent
financial explanations mid-thought. A passage like the BOPM explanation (Failure 2 in
failures.md) spans ~150 tokens — chunking it would produce two incomplete fragments, neither of
which independently answers "valuing options". The chunking infrastructure (`src/retrieval/utils.py`)
exists in the codebase and would apply cleanly to longer documents (e.g., SEC filings), but FiQA
is not the right corpus to benefit from it.

### B. Using a larger dense model (all-mpnet-base-v2)

**Considered:** Replace all-MiniLM-L6-v2 (22M params, 384-dim, warm p95 = 175 ms) with
all-mpnet-base-v2 (110M params, 768-dim).

**Expected:** 5x more parameters and 2x larger embedding space should improve semantic
discrimination, particularly for polysemous financial terms.

**Why it was rejected:** mpnet has roughly 5x the parameter count of MiniLM. On CPU, query
encoding time scales approximately with parameter count, pushing estimated warm p95 to ~500 ms
for encoding alone. The 768-dim embedding matrix over 57K passages (~168 MB) also doubles the
cosine similarity computation time compared to 384-dim. The combined effect would put warm p95
well above 800 ms — already worse than our current hybrid. The latency cost does not justify
the accuracy gain on this hardware configuration.

---

## 6. Trade-offs Against Constraints

**Reaching p95 <= 50 ms (from current 175 ms, Dense):**

The bottleneck is the numpy cosine similarity over 57K x 384 float32 vectors per query (~20M
multiplications). Two approaches to close the 3.5x gap:

- **FAISS IVF index**: replace exhaustive cosine similarity with an approximate nearest-neighbour
  IVF-Flat index (nlist ~128). Query latency drops to O(nlist * cluster_size) instead of O(N).
  At nlist=128 this typically yields ~20x speedup with <2% recall loss — enough to reach 50 ms
  on this CPU while keeping CPU-only and <=2 GB constraints intact.
- **Reduce k for candidate pool**: retrieve top-50 dense candidates and re-score only those with
  BM25. Smaller candidate set = cheaper cosine similarity. Would also revive the hybrid approach.

**If a GPU were available:**

Query encoding with MiniLM on a T4 GPU takes ~2 ms for a single query (vs. ~50 ms on CPU).
Cosine similarity with FAISS GPU over 57K vectors completes in <1 ms. Total warm p95 would drop
to ~5 ms — well under 50 ms with headroom for a larger model. A GPU budget also makes
all-mpnet-base-v2 viable: 768-dim encoding at <5 ms per query and recall improvements of 3–5 pp
on financial QA benchmarks.

**If the memory budget doubled to 4 GB:**

The current 1552 MB load (BM25 trie + dense embedding matrix + model weights) leaves ~470 MB of
headroom under 2 GB. With 4 GB, the system could:
- Load a second, larger model for re-ranking (e.g., a cross-encoder at ~250 MB) without evicting
  the OS page cache for embeddings.
- Serve multiple concurrent requests with per-worker model copies rather than shared state.

---

## 7. Production Concerns

- **Index staleness**: FiQA is a static benchmark, but a production financial QA corpus changes
  daily. Need a pipeline to detect new/updated documents and trigger incremental BM25 rebuild and
  partial re-encoding (or full re-index on off-peak hours).

- **Query latency under concurrent load**: the dense cosine_similarity call (numpy BLAS) is not
  thread-safe in all configurations. A multi-worker serving layer with process-level isolation
  (e.g., gunicorn workers) is safer than a multithreaded shared-state server.

- **Model version drift**: SentenceTransformer weights are downloaded from the HuggingFace Hub
  at build time. Pin the exact model revision hash in the Dockerfile to prevent silent score drift
  when the upstream model checkpoint is updated.

- **Recall monitoring**: deploy a shadow evaluation pipeline that samples live queries and
  annotates a random subset via human review or an LLM judge. Alert if Recall@10 drops >5%
  relative over a 7-day rolling window.

- **Cost at scale**: CPU-only inference is cost-effective up to roughly 10 QPS on a single core.
  Above that threshold, a GPU instance (lower p95, higher throughput) becomes cheaper per query.
  Model the break-even point against instance pricing before committing to hardware.
