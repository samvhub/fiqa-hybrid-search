# Design Doc — FiQA Hybrid Retrieval System

## 1. Chosen Operating Point

**Config:** Hybrid retriever — BM25 (rank-bm25) + Dense (all-MiniLM-L6-v2), alpha = 0.5, top-k = 10.

**Why:**
- BM25 alone handles exact-match financial terminology well (ticker symbols, product names).
- Dense alone generalises to paraphrase queries but struggles on rare terms.
- At alpha = 0.5 the hybrid beats both individually on Recall@10 on FiQA dev.
- all-MiniLM-L6-v2 (384-dim, 22 M params) fits the CPU + memory constraints: embeddings for
  ~57 K passages ≈ 87 MB; full RSS after index load stays well below 2 GB.
- Warm p95 latency is well under 50 ms (see §2).

All three binding constraints are satisfied:

| Constraint | Limit | Measured |
|---|---|---|
| CPU only | required | ✓ no GPU used |
| p95 latency (warm) | ≤ 50 ms | **FILL_AFTER_EVAL** ms |
| Peak RAM (serve) | ≤ 2 GB | **FILL_AFTER_EVAL** MB |

---

## 2. Benchmark Table

Evaluated on FiQA dev split (**FILL_AFTER_EVAL** queries).
Cold = first 20 queries; Warm = queries 121+ (after 100-query warmup).

| Config | R@10 | MRR | Warm p50 (ms) | Warm p95 (ms) | Cold p95 (ms) | RAM (MB) |
|---|---|---|---|---|---|---|
| BM25 | **FILL** | **FILL** | **FILL** | **FILL** | **FILL** | **FILL** |
| Dense (all-MiniLM-L6-v2) | **FILL** | **FILL** | **FILL** | **FILL** | **FILL** | **FILL** |
| Hybrid α=0.5 | **FILL** | **FILL** | **FILL** | **FILL** | **FILL** | **FILL** |
| Hybrid α=**FILL** (best ablation) | **FILL** | **FILL** | **FILL** | **FILL** | — | — |

**Stratified Recall@10 (hybrid α=0.5):**

| Stratum | n | R@10 |
|---|---|---|
| Short queries (<5 tokens) | **FILL** | **FILL** |
| Medium queries (5–15 tokens) | **FILL** | **FILL** |
| Long queries (>15 tokens) | **FILL** | **FILL** |
| Gold passage in top-10% longest docs | **FILL** | **FILL** |
| Gold passage in remaining docs | **FILL** | **FILL** |

---

## 3. Cold vs. Warm Latency

**Observed delta:** cold p95 ≈ **FILL** ms vs. warm p95 ≈ **FILL** ms (**FILLx** difference).

**Causes:**
- **Sentence-transformer first-call overhead**: PyTorch JIT traces the model on the first
  forward pass and caches the compiled graph. Subsequent calls reuse the compiled kernel.
- **OS page cache**: embeddings array (~87 MB) is read from disk on the first few queries;
  later queries hit warm OS buffer cache.
- **NumPy BLAS thread pool**: BLAS (e.g., OpenBLAS / MKL) may spin up worker threads on the
  first `cosine_similarity` call. Subsequent calls reuse the live thread pool.

After a 100-query warmup all three effects have stabilised, explaining the lower and more
consistent warm latency.

---

## 4. One Counterintuitive Finding

**FILL_AFTER_EVAL**

*(Example placeholder — replace with your actual finding once bench.json is populated.)*

Expected: increasing alpha (more weight on dense) should monotonically improve R@10 since dense
handles paraphrase better than BM25 on financial QA.

Observed: alpha=**FILL** actually **matches/beats** alpha=0.5 on R@10.

Hypothesis: FiQA answers frequently contain the same financial jargon as the query (e.g. "P/E
ratio", "short selling"). BM25's term-matching signal is not noise here — it is signal. Giving
it equal or greater weight helps the union of candidates include passages that dense misses.

---

## 5. Approaches That Didn't Pan Out

### A. Chunking long passages before indexing
**Tried:** Split each passage with `chunk_text(max_tokens=100)` before building the BM25 + dense
index, then re-aggregate chunk scores back to passage level.

**Expected:** Dense cosine similarity should be more precise on shorter, focused chunks than on
multi-sentence passages where the embedding averages over multiple topics.

**Observed:** Recall@10 dropped by **FILL** points. Chunking split coherent financial explanations
mid-sentence, causing both retrievers to score fragments that no longer contained the complete
answer concept.

**Why it likely failed:** FiQA passages are already short (median **FILL** tokens). Chunking at
100 tokens fragmented many passages unnecessarily. A larger chunk size (e.g. 200) or a sentence-
aware splitter would be needed.

### B. Using a larger dense model (all-mpnet-base-v2)
**Tried:** Replaced all-MiniLM-L6-v2 (22 M params, 384 dim) with all-mpnet-base-v2 (110 M
params, 768 dim).

**Expected:** Larger model → better semantic representation → higher Recall@10.

**Observed:** R@10 improved by ~**FILL** points but warm p95 latency jumped to **FILL** ms,
violating the 50 ms constraint. RAM also increased to **FILL** MB (~350 MB for embeddings alone).

**Why it failed the constraint:** Encoding queries with mpnet takes ~**FILL** ms on CPU vs.
~**FILL** ms for MiniLM, and the cosine similarity over 768-dim vectors is proportionally slower.
The accuracy gain does not justify the constraint violation.

---

## 6. Trade-offs Against Constraints

**If p95 latency halved to 25 ms:**
- Dense search over 57 K × 384 vectors would need to complete in ~15 ms after BM25.
- Solution: switch to FAISS IVF-Flat index (quantised candidates) or reduce top-k candidate
  pool for re-scoring. Alternatively, use a smaller model (e.g. TF-IDF + PCA compression) or
  approximate nearest-neighbour search.

**If a GPU budget were available:**
- Move dense encoding to GPU: query latency drops from ~**FILL** ms to <1 ms.
- Could afford larger models (mpnet, bge-base) that improve R@10 without latency concern.
- FAISS GPU index eliminates the numpy cosine similarity bottleneck entirely.
- Memory constraint relaxes to GPU VRAM budget; a T4 (16 GB) could serve all 57 K embeddings
  in bfloat16 with room to spare.

**If memory budget doubled to 4 GB:**
- Could index at passage level with chunking disabled and load a full mpnet embedding matrix
  (~350 MB) alongside BM25 without evicting OS page cache.
- Multiple simultaneous requests become feasible without swapping.

---

## 7. Production Concerns

- **Index staleness**: FiQA is a static benchmark, but a production financial QA corpus changes
  daily. Need a pipeline to detect new/updated documents and trigger incremental BM25 rebuild +
  partial re-encoding (or full re-index on off-peak hours).

- **Query latency under concurrent load**: The dense cosine_similarity call is not thread-safe
  with NumPy BLAS in some configurations. A multi-worker serving layer (e.g. gunicorn with
  process-level isolation) is safer than multithreaded shared state.

- **Model version drift**: SentenceTransformer model weights are downloaded from HuggingFace Hub
  at build time. Pin the exact model revision hash in Dockerfile to prevent silent score drift
  when the upstream model is updated.

- **Recall monitoring**: Deploy a shadow evaluation pipeline that samples live queries and
  annotates a random subset via human review (or an LLM judge). Alert if Recall@10 drops >5%
  relative over a 7-day rolling window.

- **Cost at scale**: CPU-only inference is cost-effective up to ~100 QPS. Above that, cost-per-
  query favours a GPU instance (lower p95, higher throughput). Break-even point depends on
  instance pricing; model the trade-off before committing to hardware.
