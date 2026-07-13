# Approach_1 — full-run results

Full pipeline run on an RTX PRO 6000 (Blackwell) GPU: substring rule, NLI,
Qwen2.5-14B-Instruct-AWQ judge (self-consistency), cross-lingual check, and a
retrieval branch grounded in the **latest Bengali Wikipedia** — stacked by a
per-branch nested-CV meta-model. Numbers are **honest out-of-fold** on the 299
labeled samples (`ensemble.py`).

## Headline (honest nested-CV OOF)

| Branch | Baseline | +judge (3 sig) | +cross-lingual (4) | **+retrieval (5)** |
|---|---|---|---|---|
| **Overall macro-F1** | 0.6823 (LB 0.666) | 0.7647 | 0.7787 | **0.7925** |
| Context | 0.894 | 0.901 | 0.901 | **0.901** |
| No-context | **0.345** | 0.649 | 0.670 | **0.675** |

The no-context branch (the whole opportunity, and where the private LB's C1 weight
concentrates) went **0.345 → 0.675** — nearly doubled. Overall **+0.11 over baseline**.

## Signals & how the meta-model weights them

| Signal | Context F1 | No-context F1 | no-ctx weight |
|---|---|---|---|
| substring (rule) | **0.894** | — | 0 |
| judge (Qwen-14B, self-consistency) | 0.801 | 0.649 | 1.09 |
| cross-lingual (answer-in-English + agree) | — | 0.615 | 1.09 |
| **retrieval** (bge-m3 → wiki → Qwen grounds) | — | **0.656** | **1.55** |
| NLI (mDeBERTa-xnli) | 0.635 | — | 0 |

- **Context** is solved by `substring` (2.57) + `judge` (2.07); NLI down-weighted
  (0.92), cross-lingual/retrieval ~0. NLI *underperformed* substring (0.635 vs 0.894)
  — a short extractive answer is a poor NLI hypothesis.
- **No-context**: `retrieval` gets the **largest** weight (1.55), with judge and
  cross-lingual both contributing (1.09 each). All three are *complementary* — the
  three attack the closed-book problem from different angles (parametric, cross-lingual,
  external-knowledge).

**Why retrieval matters:** in the cross-lingual pass, **~44% of no-context questions
abstained** — Qwen-14B was UNKNOWN in English too. Those deep-C1 (Bangladesh-specific)
facts are beyond any parametric model; only the wiki-grounded retrieval branch can
reach them, which is why it earns the top no-context weight.

## Corpus (retrieval branch)

- Source: **`dumps.wikimedia.org/bnwiki/latest`** (dump `20260701`, current) — NOT a
  stale pre-cleaned HF snapshot. Raw wikitext → `wikiextractor` (Python 3.10) →
  **438,788 articles** → chunked to **335,628 passages**.
- Retriever: **BAAI/bge-m3** dense (normalized CLS), top-5 per no-context question.
- Grounding: Qwen-14B judges whether the response follows from the retrieved passages
  (self-consistency, 5 votes).

## Caveat

The combined "overall" line re-thresholds on the OOF (mildly optimistic). Stricter
per-branch nested-CV OOF: context 0.870 / no-context 0.653. Treat ~0.76–0.79 as the
honest range. 299 samples → real variance; trust the *direction*, confirm on the LB.

## Files

- `signal_judge.json`, `signal_crosslingual.json`, `signal_retrieval.json` — per-row
  P(faithful). **Reuse to re-run the ensemble without a GPU.**
- `submission_ensemble.csv` — final 5-signal submission (1475 hallucinated / 1041 faithful).

## Reproduce

```bash
# GPU: three signals (Qwen-14B via vLLM; bge-m3 for retrieval)
python kaggle_judge.py            # -> signal_judge.json
python crosslingual.py            # -> signal_crosslingual.json
# retrieval: fetch latest bnwiki dump, extract (py3.10 wikiextractor), then:
python prep_wiki.py               # extracted/**/wiki_* -> passages.jsonl
python retrieve.py                # bge-m3 dense -> retrieved_evidence.json
python ground.py                  # Qwen grounds vs evidence -> signal_retrieval.json
# CPU: substring + NLI, then stitch all five
python nli_grounding.py && python ensemble.py
```

## Ablation: query expansion (tried — neutral)

`retrieve2.py` = dual-query retrieval (embed the **question** *and* the **response**,
union, top-6) + token-safe grounding (`ground.py` now truncates prompts to fit
Qwen's 4096 limit — top-6 Bengali evidence otherwise overflows and vLLM hard-errors).

Result: the retrieval *signal* nudged up (samples 0.6564 → 0.6635; no-context
per-branch OOF 0.6527 → 0.6618), but the **pooled ensemble was flat within noise**
(overall 0.7925 vs 0.7821 — threshold/pooling wiggle on 299 rows). bge-m3 on the
question alone already retrieves well. **Kept v1 (single-query, top-5) as canonical.**

## Next levers (untried)
- Rerank with `bge-reranker-v2-m3` (on the box) before grounding — precision bump.
- Passage-level chunking beyond the first 3 chunks/article; larger top-k with the
  token-safe grounding now that overflow is handled.
- C1 probe set to track the band the LB actually rewards.

---

## Update: 7-signal ensemble (+ Approach_0 judge & self-verify) — LB 0.765

Approach_0's Kaggle judge run (Qwen2.5-14B-**GPTQ**, greedy, different prompts
than Approach_1's self-consistency judge; LB **0.748** standalone) exports two
extra signals. `import_a0_signals.py` converts them (plus the substring rule)
into ensemble format; `ensemble.py` picks them up with zero code changes.

| | 5 signals | **7 signals** |
|---|---|---|
| Overall OOF macro-F1 | 0.7925 | **0.8059** |
| Context | 0.9009 | **0.9087** |
| No-context | 0.6748 | **0.7078** |
| **Public LB** | **0.752** | **0.765** |

- The two judges are *complementary*, not redundant: `a0judge` earns weight
  1.67 in the context branch (alongside substring 2.33 and judge 1.36) and
  0.78 in no-context (alongside retrieval 1.46 and cross-lingual 1.02).
  Different prompts + different quantization = decorrelated errors.
- `a0selfv` (answer-then-compare) adds a small 0.26 no-context weight.
- Both LB checks landed on the local→LB line (−0.03/−0.04 from OOF), so the
  sample split keeps being a trustworthy compass.

Raw Approach_0 verdicts live in `Approach_0/results/signals_{samples,test}.json`.

### Next levers (updated)
- Reranker (`bge-reranker-v2-m3`) before grounding — precision bump on retrieval.
- Second judge model (Gemma-2-27B / TituLLM) as an 8th signal — the a0judge
  gain shows judge diversity pays.
- C1 probe set — still our only visibility into the band the private LB weights.

---

## Update: 8th signal — Qwen2.5-32B judge (single greedy pass)

Meta-model experiments (`experiments_meta.py`, 5-seed x 5-fold) showed the
7-signal stacker is saturated: honest seed-averaged OOF is **0.7946 +-0.0089**
(the 0.8059 previously quoted was one favorable seed), and abstain indicators,
BanglaBERT, branch pruning, and C sweeps all fail to beat it. Only new signal
strength moves it, so: `kaggle_judge32.py` = Qwen2.5-**32B**-Instruct-GPTQ-Int4
on T4x2, one YES/NO per row (~30 min total), stitched by `stitch32.py`.

| | 7 signals | **8 signals** |
|---|---|---|
| OOF macro-F1 (5-seed mean) | 0.7946 +-0.0089 | **0.8032 +-0.0032** |
| No-context | 0.6942 | **0.7169** |
| Context | 0.9026 | 0.8941 |

- judge32 is the strongest standalone no-context signal (0.6618) and takes the
  top meta-weights (ctx 1.53 / noctx 1.33) on arrival — parametric scale
  directly buys C1-adjacent knowledge.
- Seed variance drops 3x: the ensemble got more stable, not just better.
- `submission_final.csv` = the 8-signal predictions (LB pending).

### LB check: 8-signal scored 0.759 (7-signal remains 0.765)

Public LB came back **0.759** — below the 7-signal's 0.765 despite the better
honest OOF (0.8032 vs 0.7946). Read: a +0.009 OOF gain on 299 rows does not
reliably transfer through public-LB split noise (~1.2k rows); these two
ensembles are **statistically indistinguishable** with the data we have. Do
not chase public-LB deltas this small in either direction.

**Final-selection recommendation (Phase 1 deadline):** select BOTH
`submission_ensemble7.csv` (best public, 0.765) and `submission_final.csv`
(best honest OOF, lowest seed variance) as the two finals — hedges the
public/private split noise instead of betting on it.

**Where a real (step-change) gain must come from** — the no-context branch is
still ~0.72 OOF and small signal tweaks now drown in validation noise:
1. Retrieval upgrades (reranker, deeper chunking, larger top-k) — needs the
   bnwiki index box.
2. 32B judge + self-consistency + retrieval-grounded prompts (~90 min Kaggle
   run) — merge the two strongest ideas.

---

## Error analysis: the miss profile is one-sided (drives the next runs)

On the 167 no-context sample rows, the 23 rows that **every** signal gets
wrong-or-abstains-on are **all faithful** (label=1) — not a single hallucinated
row evades all signals. Our judges are systematically over-skeptical: true but
obscure (deep-C1) answers get called hallucinated. 29% of faithful no-context
rows are unanimously miscalled; that one-sided bias IS the gap to the top of
the LB. Context branch is near-solved (substring ∧ judge32 both wrong: 3/132).
Judge-judge agreement is 0.74–0.86 — another generic judge adds nothing.

Two kernels launched to attack exactly this:

1. **`kaggle_judge32v2.py`** — anti-skepticism 32B judging: three-way
   YES/NO/UNSURE (UNSURE→0.5 instead of a wrong NO), logprob-soft outputs,
   few-shot exemplars that include true-but-obscure local facts, self-verify.
   → `signal_j32lp/j32fs/j32sv`
2. **`kaggle_wiki_retrieve.py`** — high-recall evidence: fresh HF bnwiki,
   6 chunks/article, dual query (q and q+response) top-10 union,
   bge-reranker-v2-m3 → top-5 with scores → `retrieved_evidence.json`;
   a 32B grounding kernel follows on its output.

Stitching stays honest: 5-seed × 5-fold OOF via `stitch32.py`/`experiments_meta.py`;
no submission unless decisively better than 0.8032.

---

## Evidence grounding pays off: LB 0.765 → 0.803 (rank 8)

`signal_ret32` (32B judges answers against reranked bnwiki evidence, UNSURE-safe
three-way verdicts) is the **strongest no-context signal to date**: standalone
0.7530 (prev best 0.6877), 0.76 accuracy while decisive on all rows, rescues
7/23 previously-hopeless faithful rows. Stitched:

| config | 5-seed OOF | noctx | Public LB |
|---|---|---|---|
| base8 | 0.8032 | 0.7169 | 0.759 |
| **base8 + j32sv + ret32 (submitted)** | **0.8327 ± 0.008** | **0.7775** | **0.803** |

- ret32 takes the largest noctx meta-weight ever (1.88); j32sv adds 1.05.
- Adding j32fs/j32lp on top *hurts* (0.827/0.820) — signal count is again
  saturating; prune, don't hoard.
- OOF→LB transfer held exactly (−0.03): the honest harness is predictive.
- Ops note: Bengali ≈1+ token/char in Qwen vocab — grounding prompts need
  max_model_len 8192; vLLM 0.24 dropped `truncate_prompt_tokens` from
  SamplingParams.

### Next levers
- **bn.wiktionary corpus** for word-meaning/grammar questions (~⅓ of the
  remaining hopeless rows; Wikipedia is the wrong book for them).
- Fresh bnwiki dump (HF snapshot is 2023-11) once the pipeline is trusted.
- Phase-2 packaging: chain wiki-index → grounding → judges → stitch into one
  offline notebook; evidence + weights as Kaggle datasets.

---

## Phase-2 package verified: offline single notebook, LB 0.800

`Phase_2/` now contains the full solution package. The portable 7-signal
pipeline (drops the three workstation-only signals; OOF 0.8311 vs 0.8327)
ran **fully offline** in a Kaggle kernel with all artifacts pinned:
**4.96 h on T4 x2**, 97.7% agreement with the 10-signal submission,
**public LB 0.800** vs 0.803. Recommended Phase-1 finals: the 10-signal
`submission_ret32.csv` (0.803) + the package output (0.800, exactly
reproducible by the organizers' rerun).

Ops note for the paper: the vLLM wheel set upgrades torch/CUDA libs and must
install AFTER the image-native encoder stages (libnvrtc mismatch otherwise).

---

## Wiktionary-grounded dictionary signal: LB 0.803 → 0.814 (+0.011)

The over-skeptical no-context branch was killing true glosses on **dictionary-lookup
rows** (`X এর ভাবার্থ / শাব্দিক অর্থ / অর্থ কী?`) — 155 no-context test rows, of which
the submitted ensemble called only ~38 faithful vs a true rate ~41%. Wikipedia is the
wrong book; **bn.wiktionary** is the right one.

Pipeline (`kaggle_wikt_ground.py`, kernel `bengali-wikt-ground`, T4x2, internet on):
fresh bnwiktionary dump → 70,202 headword→gloss entries (Kaggle dataset
`bengali-wiktionary-glosses`) → bge-m3 dense retrieve the headword's glosses →
Qwen-32B judges the candidate answer against them (three-way YES/NO/UNSURE, logprob-soft)
→ `signal_wikt.json`. **Gate: trust the verdict only on an exact normalized-headword
match, else abstain (0.5).**

Why OOF can't score it: the labeled bucket is 27/299 rows (10 dict-covered) — a perfect
fix moves 299-row macro-F1 <0.005. Validated per-row instead:
- labeled exact-covered accuracy **9/10**;
- test exact-covered = **101/155**; vs submitted: 52 agree, **43 rescues (0→1)**, 6 flips;
- rescues spot-check as genuine recoveries (`ঢেঁড়স`→worthless person, `টিফিন`→afternoon
  snack); correctly separates the right gloss of `নিজের ঢাক নিজে পেটা` (→1) from the
  wrong-gloss version in the labeled set (→0).

Submitted **rescue-only** (43 dict-confirmed 0→1, no flips; `submission_wikt_rescue.csv`):
**public LB 0.814**. Complementary to the teammate `ভাবার্থ`-override 0.815 (different rows,
single-word `শাব্দিক অর্থ`) — the two should stack.

### Remaining headroom on this bucket
- 54/155 test rows not exact-covered (paraphrase headwords, idioms absent from wiktionary);
  add an idiom (বাগধারা) dictionary + fuzzy/multi-sense retrieval.
- Held 6 flips (1→0): 2-3 are correct (`যমে ধরা`, `লাভের গাঁতি` were wrongly faithful) —
  a `confflip` submission could recover them.

---

## Wiktionary idiom-grounding: OOF 0.8327 -> 0.8460 (clears submit gate)

The 16 remaining hopeless rows are a *corpus* problem, not a prompt problem
(the tiered-fallback idea was falsified: the meta-model already carries j32sv).
~⅓ of them plus **14% of the no-context TEST set (162/1155)** are Bengali
word/idiom-meaning questions that Wikipedia provably cannot answer (retrieval
returns grammar/physics/fruit articles). Fix = the *right book*: **bn.wiktionary**.

Pipeline (all on the VPS RTX PRO 6000; 32B grounding ~1 min):
- `build_wikt.py` — parse `bnwiktionary-latest` (CPU, 7 s) -> `wikt_passages.jsonl`,
  **73,444 entries** (15,182 idioms incl. 7,723 mined from the বাগধারা appendix
  table + 58,262 words).
- `wikt_retrieve.py` — bge-m3 dense retrieval, top-5 dictionary entries per no-ctx
  query. Dense retrieval recovers variant spellings exact-lookup misses
  (e.g. উজানের কই -> `উজানের কই/কৈ: সহজলভ্য বস্তু`).
- `wikt_ground32.py` — Qwen2.5-32B-GPTQ (VPS, util 0.23, `gptq_marlin`,
  `enforce_eager`) judges the given meaning vs the retrieved definition,
  UNSURE-safe soft verdict. Gated to word/idiom queries -> `signal_ret32wikt.json`.

Standalone accuracy splits sharply by question type:
- **Idiom (ভাবার্থ): 12/13 = 0.92** — correctly passes true idioms AND rejects
  false ones; all **75 idiom test rows grounded decisively**.
- Word-meaning (শাব্দিক অর্থ): 7/14 = 0.50 — wrecked by dataset **label noise**
  (চরণদাস/আঠা answers exactly match the dictionary yet are labeled hallucinated),
  polysemy (any-sense matching), and factual leakage. Not used.

So the vehicle is a **targeted idiom-only override**, not a meta-model feature
(too sparse to weight) and not word-meaning (too noisy):

| config | 5-seed OOF | hopeless rescued |
|---|---|---|
| baseline base8+j32sv+ret32 (LB 0.803) | 0.8327 | 0.8/16 |
| **+ idiom-override** | **0.8460 ± 0.008** | 3.8/16 |
| prune(−nli,−crosslingual) + idiom-override | 0.8507 ± 0.007 | 4.0/16 |

`submission_wikt_override.csv` = 0.803 base with the idiom rule applied
(flipped 26/75 idiom test predictions, net +18 faithful). Clears the 0.845 gate;
awaiting LB confirmation. Byproduct: the dictionary **surfaced dataset label
errors** on tricky word-meaning rows — a real ceiling on OOF for that slice.

---

## Self-answer self-consistency reasoning judge: LB 0.814 → 0.828 (+0.014)

Error decomposition of the no-context branch: the mass is **factual world-knowledge**
(factual bucket 65 labeled rows @ 0.723 OOF acc, whoq 38 @ 0.842) plus reasoning/negation
MCQs ("কোনটি নয়", analogies). Every prior judge emits a bare one-word verdict — no
reasoning, anchored on the given answer. Added `signal_sa` (`kaggle_selfanswer.py`, kernel
`bengali-selfanswer`, T4x2): 32B **independently derives** the answer with chain-of-thought,
sampled K=6 at temp 0.7, each chain votes YES/NO on the candidate; P = mean YES.

Standalone it is *not* better (noctx 0.731, whoq 0.675 — closed-book knowledge is the wall,
CoT can't invent facts). **But stacked it gives the biggest single-signal jump since ret32:**
- 10-sig OOF 0.8319 → +sa 0.8447 (+0.0128); noctx 0.784 → 0.808.
- Combined 11-sig + wikt rescue: **OOF 0.8512**, noctx 0.820, **factual bucket 0.723 → 0.785**.
- Public LB **0.828** (predicted 0.825–0.827; OOF→LB transfer −0.023, in line with the −0.03 rule).

Lesson: a decisive, differently-calibrated reasoning signal helps the stacker on borderline
rows even when weaker alone. Knowledge remains the ceiling — next lever is **retrieval** for
the factual mass (cross-lingual enwiki + fresh full bnwiki), not more closed-book prompting.
