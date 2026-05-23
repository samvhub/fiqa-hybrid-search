# Retrieval Failure Cases

All cases are from the hybrid retriever (alpha=0.5) on the FiQA dev split.
A failure is defined as Recall@10 = 0 — the gold passage is not in the top 10.

---

## Failure 1 — "Corporate Finance" (qid 4709)

**Top 5 retrieved:**

| Rank | Doc ID | Snippet |
|---|---|---|
| 1 | 583646 | "Financial Services provides or facilitates access to capital... TD Ameritrade, Goldman Sachs, Bank of America..." |
| 2 | 502614 | "I ended up buying: Financial Modeling and Valuation... Mergers, Acquisitions, Divestitures..." |
| 3 | 463599 | "Corporate Finance is essentially divided into capital... Investopedia has a layman overview of most Financial Services jobs..." |
| 4 | 38560 | "You need to go back to finance 101 about the concept of corporation and shareholders..." |
| 5 | 479752 | "The term financialization is used all over the place..." |

**Gold passage (not in top 10):**

Doc 297385: *"Your company wants to raise $25,000,000 for a new project, but flotation costs are incurred by issuing securities... The company's target D/E ratio is 50% (or .50). For every $0.50 of debt..."*

**Diagnosis:** The query is a two-word category label with zero specificity. BM25 surfaces passages containing "corporate" and "finance"; dense finds semantically related financial services passages. Neither retriever can infer that the specific sub-topic needed is *flotation cost calculation with a D/E ratio constraint*. With 57K community Q&A passages about finance, any financially adjacent passage scores comparably. The gold passage never uses the phrase "corporate finance" — it uses domain jargon (flotation costs, D/E ratio, underwriting) that is semantically distant from the generic query embedding.

**Fix to try:** HyDE (Hypothetical Document Embeddings) — prompt an LLM to generate a hypothetical answer to "Corporate Finance", embed that generated answer, and use its vector as the query. The generated text would likely include specific sub-topic terminology (flotation costs, WACC, capital structure), shifting the query representation toward the gold passage and away from generic financial-services passages.

---

## Failure 2 — "valuing options" (qid 5027)

**Top 5 retrieved:**

| Rank | Doc ID | Snippet |
|---|---|---|
| 1 | 39345 | "I can only add that it may be valuable if the company is bought, they may buy the options." |
| 2 | 50735 | "You could also look at your growth in online subscribers as a metric for valuing your company..." |
| 3 | 525213 | "An option, by definition, is a guess about the future value of the stock..." |
| 4 | 57800 | "The options — by themselves — are pretty meaningless in terms of determining their value..." |
| 5 | 114304 | "There's a primer on valuing community banks by oddball Stocks..." |

**Gold passage (not in top 10):**

Doc 409190: *"Below I will try to explain two most common Binomial Option Pricing Models (BOPM)... BOPM splits time to expiry into N equal sub-periods and assumes that in each period the underlying security price may rise or fall by a known proportion..."*

**Diagnosis:** "Options" is polysemous: employee stock options (pre-IPO startup context), financial derivatives (calls/puts valued via BOPM/Black-Scholes), and general choices. The dense encoder maps "valuing options" into a region shared by *valuing something* (subscriber counts, community banks) and *options as pre-IPO grants* — not financial derivatives. The gold passage uses technical vocabulary — "BOPM", "sub-periods", "underlying security price" — that is lexically distant from the short query. BM25 also fails: the query "valuing options" has no term overlap with "Binomial Option Pricing Model".

**Fix to try:** Add a cross-encoder re-ranker over the top-50 dense candidates. A cross-encoder evaluates query and passage jointly and can recognize that a passage explaining how to calculate option value answers "valuing options" even without lexical overlap. Alternatively, query-time synonym expansion ("options pricing", "Black-Scholes", "BOPM") would give BM25 a chance to surface the gold passage.

---

## Failure 3 — "Solicitation of a Security" (qid 10117)

**Top 5 retrieved:**

| Rank | Doc ID | Snippet |
|---|---|---|
| 1 | 511480 | "**Social Security number** In the United States, a Social Security number (SSN) is a nine-digit number issued to U.S. citizens..." |
| 2 | 557115 | "SUBREDDIT RULES This security forum is oriented towards private white hat security professionals. NO ADVERTISING..." |
| 3 | 380972 | "Most answers to this question only address the issue of providing personal information to a scammer..." |
| 4 | 26799 | "many participants in the Social Security system will experience negative rates of return..." |
| 5 | 239780 | "SS is not an investment. It is a Tax... it has always been a pay as you go plan, just like medicare..." |

**Gold passage (not in top 10):**

Doc 508922: *"ASSUMING THIS IS A QUESTION OF U.S. SECURITIES LAWS You didn't explain whether you're related to the mother and son... this really wouldn't qualify as a solicited sale. It wasn't advertised publicly for sale..."*

**Diagnosis:** "Security" in the query activates the wrong word sense in both retrievers. BM25 matches passages with high-frequency co-occurrences of "security" — Social Security (SSN, tax, benefits) and cybersecurity — which vastly outnumber passages about financial instruments. The dense encoder, pre-trained on general text, represents "security" as a centroid weighted toward these dominant senses. The gold passage uses "securities" (plural) and SEC-specific legal concepts ("solicited sale", U.S. Securities Laws), but the embedding for "Solicitation of a Security" sits far from that passage in the embedding space.

**Fix to try:** Replace all-MiniLM-L6-v2 with a financial-domain encoder (FinBERT, or a model fine-tuned on SEC filings and financial QA). Domain-adapted encoders disambiguate "security" toward the financial instrument sense by seeing it predominantly in that context during fine-tuning. An alternative is adding "financial instrument" or "SEC" as a query prefix to steer the embedding without retraining.
