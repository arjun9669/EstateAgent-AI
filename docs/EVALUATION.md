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

### Results: two runs before and two after, scorer 1.1 (re-scored with 1.2 in the next section)

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

## Plain-language mortgage checks, checked on a second held-out set

### What was still wrong

- **Field names in answers.** Answers sometimes showed the mortgage tool's raw
  field names, such as `within_debt_burden_cap` or `dld_transfer_fee`. Quoted, that
  failed v1 G4; unquoted, the scorer did not notice at all. Counted over every
  answer at the warnings-rule code (`27079b6`): **3 of 128** answers showed one.
- **Two v2 cases asked for the wrong thing** (see above): W4's phrase list lacked
  "outside", and N7 expected a 50% figure its question did not ask for.

### The held-out set: [`eval/cases-v3.json`](../eval/cases-v3.json)

Written and committed (`e570f60`) before the tool change and before any run: 14
new questions. Seven are mortgage checks: down payments under the cap for an
expatriate and a national, debt burden over and within the limit, off-plan, and
second homes. The rest cover listings, comparison, market, law and out-of-scope.
The two v2 case-design issues are corrected with new questions, not by editing
v2: the down-payment cases accept "outside", and separate off-plan questions ask
for the amount (D5) and the percentage (D6).

### The change (commit `9e71263`)

`mortgage_calculator`, the tool the agent sees, now:

- names its figures in plain English ("monthly payment (AED)", "DLD transfer fee
  (4% of price)");
- states each check as a sentence, for example *"The monthly payment is 59.3% of
  monthly income, above the 50% debt burden limit. The limit covers all debt
  repayments; this counts only this mortgage."*

The maths moved unchanged to `calculate_mortgage`, which keeps code-style keys for
code and tests. No prompt change.

### Scorer 1.2

After the change, two v1 answers put the new plain-English labels in quotation
marks ("DLD transfer fee (4% of price)"). Those quotes are word for word from the
tool result, but scorer 1.1 only searched the result's values, not its field
names, so it failed them. **Scorer 1.2** also searches field names. Recorded
runs keep their original scores; every run re-scored with 1.2 is in
[`eval/rescored-latest.json`](../eval/rescored-latest.json).

One side effect: a quoted raw field name such as "within_debt_burden_cap" now
passes the citation check, since it is word for word in the tool result. That is
a style problem rather than a faithfulness one, so it is now counted separately as
"answers showing code-style field names" ([`field_name_leaks`](../src/estate_agent/evaluation.py)),
reported next to the scores and not part of pass/fail.

### Results: two runs of each set before and after, scorer 1.2

| Set | Before (`27079b6` / `e570f60`) | After (`9e71263`) |
|---|---|---|
| v3, held out (14) | 14, 14 | **14, 14** |
| v2, held out (20) | 19, 18 | **20, 20** |
| v1 (30) | 30, 30 | **30, 30** |
| Answers showing code-style field names | **3 / 128** | **0 / 128** |

Reports: [v3 before](../eval/variation-v3-before-mortgage-checks.json),
[v3 after](../eval/variation-v3-after-mortgage-checks.json),
[v2 after](../eval/variation-v2-after-mortgage-checks.json),
[v1 after](../eval/variation-v1-after-mortgage-checks.json).

**Reading the result:**

- The corrected v3 cases passed before the change too, which confirms that v2's
  W4 and N7 failures were case-design issues, not agent errors.
- The clear effect of the change is the field-name count: 3 of 128 answers before,
  none after.
- v2 rising to 20/20 is partly a wording effect. The new check sentences say
  "below the minimum" and "banks can lend at most 50%", which match the phrases
  v2's frozen W4 and N7 cases look for. It does not show the agent became more
  correct; v3, written to avoid those phrase problems, is 14/14 both before and after.
- Tool choice and grounding were full marks in every run.

## Model comparison

Same agent code, prompt, tools and question sets; only the model changes
(scorer 1.2, report: [`eval/model-comparison.json`](../eval/model-comparison.json),
script: `scripts/compare_models.py`).

| Model | Where it runs | v1 (30) | v2 (20) | v3 (14) | All 64 | Median time per question | Answers showing field names |
|---|---|---|---|---|---:|---:|---:|
| `gemma4:cloud` (31B) | Ollama cloud, free plan | 30, 30 | 20, 20 | 14, 14 | **100%** | **1.6 s** | 0 / 128 |
| `qwen3:4b` | Locally, Apple M1, 8 GB | 8 | 4 | 3 | **23%** | **140 s** (max 619 s) | 52 / 64 |

gemma4 has two runs per set; qwen3:4b one, which took 3 h 37 min. Groq's
`llama-3.3-70b-versatile` was planned but could not be tested: the Groq account
returned "organization_restricted" on every request.

Per dimension for qwen3:4b (of 64): tool choice **63**, correctness **56**,
grounding 42, citations **17**.

**Reading the qwen3:4b answers:**

- **It picks the right tools and usually finds the right facts.** Tool choice
  failed once and the expected facts were present in 56 of 64 answers, so the tools
  and prompts work with a much smaller model.
- **It writes its reasoning into the answer.** Answers often start "Okay, let me
  process this…" and walk through the tool output by its field names, even with
  thinking mode turned off. That is the 52 field-name answers.
- **It treats a revision request as a chat message.** In 23 answers it echoes the
  agent's instructions or the grounding check's feedback ("They said my previous
  answer had figures… not in the tool results"). Those echoes are quoted text that no
  tool returned, which is why citations pass only 17 times: 46 of the 47 citation
  failures are quotes that are not word for word from a tool.
- **The grounding check works on the weaker model.** It caught figures qwen worked
  out itself, for example a 7.3% total-price change when the tool reported only the
  8.0% change in price per square foot.

**Takeaway:** the agent design carries over to a small local model for choosing tools
and finding facts, but a 4B model on an 8 GB Apple M1 is too slow (about 150× slower)
and too loose with its answer format for this use. Running gemma4 on Ollama's cloud
is the right default; a local model would need a stricter final-answer step (for
example, a separate call that only rewrites the checked facts) rather than more prompting.

## What to do next

1. If the Groq account is unblocked, add `llama-3.3-70b-versatile` to the model
   comparison (`LLM_PROVIDER=groq`, then `scripts/compare_models.py`).
2. Follow-up questions (conversation memory), with a held-out set of multi-turn
   questions written before the change.
3. For local models: a final-answer step that rewrites only checked facts, measured
   on qwen3:4b against this run.

## Limits

- 64 author-written questions on synthetic data; passing them says little about
  real listings or questions phrased very differently.
- Phrase checks (for example "either party", "rough") can miss a correct answer
  worded another way, or reward a tool's wording (v2 above); each failure was read
  by hand.
- Two to four runs per configuration is a small sample; it shows which cases are
  stable, not precise rates.
- The held-out sets were written by the same person who made the changes, so they
  are held out from each change, not independent of the author.
- The scorer was corrected twice after seeing results (1.1, 1.2). Each time the
  original scores were kept and the re-scored ones shown alongside.
