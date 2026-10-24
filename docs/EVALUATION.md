# Evaluation

A frozen set of 30 questions measures whether the agent picks the right tools,
gets the facts right, and backs its citations. It is a small, author-written
regression set on synthetic data, not a benchmark of real-world accuracy.

## The set: [`eval/cases-v1.json`](../eval/cases-v1.json)

| Category | Cases | What it checks |
|---|---:|---|
| Listings | 5 | Filters, cheapest-first, and an empty result that must not invent listings |
| Compare | 3 | Price per sq ft, the lowest-value pick, and mixing a rental with a sale |
| Market | 4 | Trends, and saying when there are too few sales to trust |
| Mortgage | 5 | Loan-to-value caps by buyer type, affordability, and the 25-year maximum term |
| Rules | 7 | Rent increases, notice, eviction, payment, maintenance, re-letting, registration |
| Multi-tool | 3 | Two tools in one question |
| Out of scope | 3 | Unanswerable or unknown questions; no tools or invented figures |

Expected values were computed from the committed data, legislation and
mortgage rules. `tests/test_evaluation.py` re-derives them from the tools and pins
the file's SHA-256. The set was committed (`f80b4bb`) before the first run. A
wrong case is fixed by adding `cases-v2.json`, not by editing v1.

## Scoring ([`evaluation.py`](../src/estate_agent/evaluation.py))

A case passes only if all four dimensions pass:

| Dimension | Pass condition |
|---|---|
| Tool choice | Every required tool ran successfully; no forbidden tool was called |
| Correctness | Every expected fact is in the answer (numbers match when rounded, as in the grounding check); no forbidden phrasing such as "landlord must notify" |
| Citations | Required listing/law IDs are cited; every cited ID came from a tool; every quotation of 12+ characters appears word for word in the tool results |
| Grounding | The agent's own grounding check found no unsupported numbers |

## Results

Run with `python scripts/run_eval.py`. Each run is saved to `eval/runs/` with every
answer, tool call, tool result, grounding report and timing. CI re-scores saved
runs, so a scorer change that would alter a recorded result fails the build.

### Run 1: 24 October 2026, `gemma4:cloud` via Ollama, temperature 0 ([record](../eval/runs/2026-10-24-gemma4-cloud.json))

**27 / 30 cases pass.** Median 3.1 s per question.

| Dimension | Passed |
|---|---:|
| Tool choice | 30 / 30 |
| Correctness | 30 / 30 |
| Citations | 27 / 30 |
| Grounding | 30 / 30 |

| Category | Passed |
|---|---:|
| Listings | 5 / 5 |
| Compare | 3 / 3 |
| Market | 4 / 4 |
| Mortgage | 4 / 5 |
| Rules | 5 / 7 |
| Multi-tool | 3 / 3 |
| Out of scope | 3 / 3 |

**Failures, unchanged from the first run:**

- **R7, retrieval miss.** "Does a tenancy contract have to be registered?" The
  answer is Article 4 of Law 26/2007 ("will be registered with RERA"), but
  `search_rules` returned Articles 18, 21 and 2 and Decree 43 Article 2. The
  agent said the documents did not cover it rather than inventing a rule, which
  is the right behaviour given what it saw, but the search itself missed. This
  is the most important finding: embedding search alone can miss a literal term
  such as "registered".
- **R1, shortened quotation.** The answer quotes Article 13 as "Landlord and Tenant
  may... amend any of its terms…", leaving out words without marking the gap
  clearly. The rest of the answer, including all five rent-increase tiers, is
  correct and word for word.
- **G4, quoted field name.** The answer puts "within debt burden cap" in quotation
  marks: a tool field name (`within_debt_burden_cap`) presented as a quote. All
  figures are correct.

Two cases (G4, G5) needed one revision after the grounding check flagged a draft;
both final answers passed it.

## What to do next

1. **Hybrid retrieval for `search_rules`:** combine embeddings with keyword (BM25)
   scores so literal terms like "registered" are not missed (R7).
2. **Quotation rule in the prompt:** quote only exact text, and never put tool
   field names in quotation marks (R1, G4). Possibly accept "…" in the scorer when
   each part is verbatim, as a v2 scoring rule reported alongside v1.
3. Rerun the set as a new recorded run after each change and compare.

## Limits

- 30 author-written questions on synthetic data; passing them says little about
  real listings or questions phrased very differently.
- Phrase checks (for example "either party", "rough") can miss a correct answer
  worded another way; each failure above was read by hand.
- One run per model. At temperature 0 answers are stable but not guaranteed
  identical between runs.
