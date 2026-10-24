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

### Changes after run 1 (commit `06ba204`)

1. **Hybrid retrieval for `search_rules`:** results are the top sections by
   meaning plus the strongest BM25 keyword match when meaning search missed it
   (words cut to a 5-letter prefix, so "registration" matches "registered").
   An even rank fusion of the two rankings was tried first and lost the right
   article for R4, so keyword matches are added, never re-ranked in. This choice
   was compared on run 1's eight law-search queries plus six rephrasings, so it is
   partly tuned on the evaluation itself.
2. **Quotation rule in the prompt:** quote only word-for-word text, never shorten
   a quote with "...", never put tool field names in quotation marks.

The cases and scorer were not changed.

### Run 2: 24 October 2026, same model and settings, code `06ba204` ([record](../eval/runs/2026-10-24-gemma4-cloud-06ba204.json))

**27 / 30 cases pass, the same total as run 1, but different cases.** Median 2.2 s
per question; no revisions.

| Dimension | Run 1 | Run 2 |
|---|---:|---:|
| Tool choice | 30 / 30 | 30 / 30 |
| Correctness | 30 / 30 | 28 / 30 |
| Citations | 27 / 30 | 29 / 30 |
| Grounding | 30 / 30 | 30 / 30 |
| **All four** | **27 / 30** | **27 / 30** |

`python scripts/compare_runs.py` shows the changes:

| Case | Run 1 → Run 2 | What happened |
|---|---|---|
| R7 | FAIL → PASS | The keyword match found Article 4 (it ranked fifth by meaning alone) and the answer quotes it exactly: contracts "will be registered with RERA" |
| R1 | FAIL → PASS | No shortened quotations |
| G4 | FAIL → PASS | No quoted field names |
| G1 | PASS → FAIL | The answer gives the AED 8,004 payment but drops the AED 360,000 minimum down payment the question asked about |
| C3 | PASS → FAIL | The table shows one listing is for rent and one for sale, but the answer no longer warns that their prices are not comparable |
| M2 | PASS → FAIL | Figures are right, but the model put numbers in quotation marks ("2138.9" AED), apparently over-applying the new quotation rule. The scorer then paired the wrong quotation marks and flagged the words *between* two quotes; that is a scorer bug (below) |

**Reading the result:** the two fixes did what they were meant to (R7, R1, G4), and
the law questions are now 7/7. The new failures are about completeness and style,
not wrong facts or invented citations; tool choice and grounding stay at 30/30.
At temperature 0 the same model still varies between runs, and one run per change
cannot separate the prompt's effect on G1 and C3 from run-to-run variation.

## What to do next

1. **Fix the scorer's quotation pairing** (M2): pair quotation marks in order
   instead of letting a match start at a closing mark. Release it as a new scorer
   version and re-score both runs, reported next to the original scores; the
   recorded v1 summaries stay as they are.
2. **Measure variation:** run the set several times per configuration and report
   the spread, not a single score.
3. **Completeness:** ask the model to include the key figures a tool returns that
   the question asks for (G1), and the tools' warnings (C3). Check this does not
   just teach to the test by adding new cases (cases-v2) the prompt was not tuned on.

## Limits

- 30 author-written questions on synthetic data; passing them says little about
  real listings or questions phrased very differently.
- Phrase checks (for example "either party", "rough") can miss a correct answer
  worded another way; each failure above was read by hand.
- One run per model. At temperature 0 answers are stable but not guaranteed
  identical between runs.
