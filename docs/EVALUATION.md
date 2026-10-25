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

### Scorer 1.1 (commit `e3d7bde`)

Scorer 1.0 skipped quotations shorter than 12 characters and restarted its search
at their closing mark, so in M2 (run 2) the words *between* two short quotes
("2138.9" AED. This represents a "5.3") were checked as if they were a quote.
Scorer 1.1 pairs quotation marks in order. Nothing else changed.

Recorded runs keep the scorer they were scored with, and CI reproduces those
original results. [`eval/rescored-latest.json`](../eval/rescored-latest.json)
lists every run re-scored with the latest scorer, next to its original score:

| Run | Original score | Scorer 1.1 | Changed |
|---|---:|---:|---|
| Run 1 | 27 / 30 (1.0) | 27 / 30 | none |
| Run 2 | 27 / 30 (1.0) | 28 / 30 | M2 FAIL → PASS |

### Variation: four runs of the same agent code

Runs 3 to 5 repeated the set three times with commit `e3d7bde`, whose agent code
is identical to run 2's (only evaluation tooling changed). All four are scored
with scorer 1.1 ([report](../eval/variation-agent-06ba204.json)):

| | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---:|---:|---:|---:|
| **All four dimensions** | 28 | 29 | 29 | 29 |
| Tool choice | 30 | 30 | 30 | 30 |
| Correctness | 28 | 29 | 29 | 29 |
| Citations | 30 | 30 | 30 | 30 |
| Grounding | 30 | 30 | 30 | 30 |

- **28 cases passed in all four runs.**
- **C3 failed in all four:** the comparison table shows rent vs. sale, but the
  answer never says the prices are not comparable. It passed in run 1 with the
  earlier prompt, so this is most likely an effect of the prompt change, not chance.
- **G1 failed once in four** (run 2, the dropped down payment), so that was run-to-run
  variation, not a regression.
- At temperature 0 the wording still changes: only 6 of 30 answers were identical
  word for word across the four runs. The tool calls were identical in 29 of 30;
  in G4 the model sometimes passed the default `buyer: "expat"` explicitly, with
  the same result.

**Current estimate for this configuration: 28 to 29 of 30**, with one consistent
failure (C3).

## Passing on tool warnings, checked on a held-out set

### The held-out set: [`eval/cases-v2.json`](../eval/cases-v2.json)

C3 kept failing because the agent left out the comparison tool's warning. A prompt
fix tuned and judged on C3 alone would just be teaching to the test, so before
changing the prompt a second set was written and committed (`9d81aaa`): 20 new
questions on listings, areas and law articles that v1 does not use. Five of them
(W1–W5) check that a tool's warning reaches the answer: a rent-vs-sale comparison,
two areas with too few sales, a down payment under the cap, and a debt burden over
the cap. The rest cover listings, market, mortgage, law, multi-tool and
out-of-scope questions, as in v1.

| Category | Cases |
|---|---:|
| Warnings | 5 |
| Listings | 4 |
| Market | 1 |
| Mortgage | 2 |
| Rules | 4 |
| Multi-tool | 2 |
| Out of scope | 2 |

### The change (commit `27079b6`)

One system-prompt rule: *if a tool result contains a warning or a check that
fails (for example that prices are not directly comparable, that there are too few
sales to rely on, or that a down payment or debt burden is outside the limits),
tell the user plainly, even if they did not ask.*

### Results: two runs before and two after, scorer 1.1

| Set | Before (prompt at `9d81aaa`) | After (prompt at `27079b6`) |
|---|---|---|
| **v2, held out** | **18, 18 / 20** | **19, 18 / 20** |
| v1 | 28, 29, 29, 29 / 30 (runs 2–5) | **29, 30 / 30** |

Tool choice and grounding were full marks in every run.
Reports: [v2 before](../eval/variation-v2-before-warnings.json),
[v2 after](../eval/variation-v2-after-warnings.json),
[v1 after](../eval/variation-v1-after-warnings.json).

| Case | Before | After | Reading |
|---|---|---|---|
| W1 (v2), rent vs sale | FAIL, FAIL | PASS, PASS | Now says the two "are not directly comparable as one is for rent and the other is for sale" |
| C3 (v1), rent vs sale | FAIL in 4/4 | PASS, PASS | Now opens with the warning |
| W2, W3, W5 (v2) | PASS, PASS | PASS, PASS | These warnings were already passed on before the change |
| W4 (v2), down payment under cap | PASS, PASS | PASS, FAIL | **False fail.** The failing answer says the AED 200,000 down payment "is outside the limits" and gives the AED 400,000 minimum, which is correct; the frozen phrase list just lacks "outside" |
| N7 (v2), off-plan down payment | FAIL, FAIL | FAIL, FAIL | **Case asks too much.** Every answer gives the correct AED 600,000; the case also expects the 50% cap, which the question did not ask for |
| G4 (v1), debt burden | PASS in 4/4 | FAIL, PASS | The model again put a field name in quotes ("within_debt_burden_cap") in one run, despite the quotation rule |

**Conclusion:** the rule fixed the comparison warning on the held-out question as
well as the one it was written for, with no new real failures in four runs.
Counting W4's false fail as a pass, the held-out set is 19/20 in both runs after
the change, against 18/20 before; the remaining miss (N7) is a case-design issue.
v1 reached 30/30 for the first time.

## What to do next

1. **cases-v3 for the case-design issues:** accept "outside the limits" in W4 and
   drop the unasked-for 50% from N7. Add new questions rather than editing v2, and
   report v2 scores unchanged.
2. **Field names in quotes (G4):** the prompt rule alone does not stop it every
   time. A cheaper fix is in the tools: give the agent plain-language warnings
   ("within the 50% debt burden limit") instead of boolean field names.
3. Keep running every change at least twice on both sets.

## Limits

- 50 author-written questions on synthetic data; passing them says little about
  real listings or questions phrased very differently.
- Phrase checks (for example "either party", "rough") can miss a correct answer
  worded another way (W4 above); each failure was read by hand.
- Two to four runs per configuration is a small sample; it shows which cases are
  stable, not precise rates.
- cases-v2 was written by the same person who wrote the prompt change, so it is
  held out from the change, not independent of the author.
