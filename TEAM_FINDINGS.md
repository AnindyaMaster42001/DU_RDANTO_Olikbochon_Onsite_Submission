# Findings & Strategy Handoff — Bengali Hallucination Challenge

_Last updated: **2026-07-09**. Team DU_RDANTO. Current best: **LB 0.831, rank 30/180** (on the top-30 cutoff)._

Decision brief for the team taking over. Supersedes the 2026-07-07 version. Every claim
below was verified locally on the 299 labeled rows (honest OOF) or against a live
leaderboard / rules pull. **Read the "Lever ledger" — it records what is measured-dead so
you don't burn days re-running it.**

---

## TL;DR

1. **We are #30/180 at LB 0.831** — the last qualifying spot. Cushion is razor-thin
   (#29 also 0.831; #31 = 0.829). Deadline **2026-07-19 18:00 BST**.
2. **The win came from a ground-truth lever, not a model.** `submission_contrastive.csv`
   = the 0.828 stack + **12 certified label flips** derived from TRAIN labels on
   identical prompts (see "The contrastive lever"). It is the only new signal since 07-07.
3. **The real metric is binary F1 on the HALLUCINATED class (label 0)**, weighted
   heaviest on the C1 Bangladesh-specific band. All 12 flips are `1->0`, so they help the
   real metric *directly* and **transfer to the private LB** (they fix genuinely wrong
   labels, not public-LB noise).
4. **Every signal/model lever we tried beyond the stack is exhausted or dead** (contrastive
   done, fuzzy=0, math-solve/enwiki/bigger-model all measured-dead — see ledger). The
   reproducible ceiling on this hardware is **~0.83–0.85**.
5. **The only untried legal lever for a real jump is FINE-TUNING** (explicitly allowed;
   deferred so far). That is the honest answer to "how did the 0.88+ teams get higher."
6. **Action needed now:** select the 2 finals on kaggle.com (website-only — no CLI).
   See "Finals selection."

---

## The contrastive lever (the win) — READ THIS

**File:** `Approach_1/contrastive_analysis.py` (audit) + `Approach_1/build_contrastive.py` (builds the 0.831 submission).

**Idea (100% legal — uses TRAIN/sample labels only, never test labels):**
Many test prompts are standard Bangla exam questions that also appear *verbatim* among
the 299 labeled samples. When a **faithful** (label==1) sample shares a test row's prompt,
that sample's response is the **gold** answer. If a test row's candidate answer differs
from the *unique* gold, the candidate is a **certified hallucination** — no matter what
the stack predicted.

- 163 of 299 samples are faithful golds. **18** test rows have a candidate that differs
  from a unique gold. After hand-audit we shipped **12** and excluded 6.
- All 12 are `1->0` (the stack wrongly called them faithful). They are authors, places,
  dates, grammar/phonetics, and idioms — e.g. `প্রান্তিক হ্রদ` is in **বান্দরবান**, not
  রাঙ্গামাটি; `জান্নাতাবাদ` was named by **হুমায়ুন**, not জাহাঙ্গীর.

**Why it transfers to private LB:** the flips are ground-truth-certain (a known-correct
answer contradicts the candidate), so they help both the public macro-F1 and the private
class-0 F1. This is categorically different from public-LB threshold tuning.

**The 6 exclusions (do not re-flip these — they are traps):**

| id | Reason excluded |
|---|---|
| 891 | `১৮৩২` == `আঠারশ বত্রিশ সালে` — same year, numeral vs words. **Candidate is correct.** |
| 2373 | `২রা নভেম্বর ১৮৮৬` == `২ নভেম্বর ১৮৮৬` — same date, surface form. **Candidate is correct.** |
| 5 | `প্রখর রোদ` is a valid paraphrase of `খটখটে রোদ` — not a unique-gold mismatch. |
| 98 | `নিয়মের ব্যতিক্রমে সিদ্ধ` is the correct standard gloss — candidate arguably *more* correct. |
| 2015, 2028 | "which is NOT" MCQs — gold uniqueness can't be confirmed without the choice set. |

**The exact-prompt lever is fully mined at 12. Do not lower the bar:**
- Stronger normalization (Bengali digit→ascii, thousands-comma strip, word↔number) surfaces
  the *same* 18 — no new safe flips. `multi_gold` = 0.
- **Fuzzy/near-duplicate prompt matching (jaccard 0.70–0.88) yields 0 safe flips.** Every
  near-dup is a semantic-drift trap: `id1364` train asks *where* Mujib was born, test asks
  *when* → candidate `১৭ মার্চ ১৯২০` is **correct**; `id991` train is *Pakistan's* first
  cadet college, test is *Bangladesh's*. Run `contrastive_analysis.py` to see the full list.
  **Below exact-prompt match, shared tokens ≠ same question. Do not use fuzzy contrastive.**

---

## Lever ledger — what worked, what is DEAD (with evidence)

| Lever | Verdict | Evidence |
|---|---|---|
| **11-signal stack + wikt rescue** | ✅ base | `submission_sa_wikt.csv`, LB 0.828, OOF 0.8512 |
| **Contrastive ground-truth flips** | ✅ **+0.003 (shipped)** | `submission_contrastive.csv`, LB 0.831, 12 certified flips |
| Fuzzy contrastive | ❌ 0 safe flips | semantic drift; see `contrastive_analysis.py` audit |
| **Math/quant self-solver** | ❌ DEAD | Qwen-32B makes *systematic* arithmetic errors (sum 1..100 → called 4999 faithful; 30x=1 → called −0.5 faithful). K=8 self-consistency doesn't rescue. Gate refused (net 0). `kaggle_mathsolve.py` + `validate_mathsolve.py`. |
| **Cross-lingual enwiki retrieval** | ❌ DEAD | 93% coverage achieved, but stacked: macro-F1 0.8545→0.8543, F1-hall 0.8410→0.8413, net rescue **+0** (3 fixed/3 broke), standalone 0.70. Overlaps ret32; where it diverges it's 70% noise. `kaggle_enwiki.py` + `validate_enwiki.py`. **Keep `signal_enwiki.json` for the paper's negative-result section.** |
| **Bigger/better open model (Gemma-2-27B)** | ❌ DEAD (hardware) | dtype trilemma on T4: fp16→gemma2 rejects; bf16→T4 sm75 rejects; fp32→GPTQ rejects. GPTQ ∩ gemma2-fp16-ban ∩ T4-bf16-ban = ∅. 72B doesn't fit 2×T4; Aya-32B has no Bengali; Bengali-native ≤9B too weak. `kaggle_gemma27.py` (moot, kept). |
| Idiom-blend (broad `ret32wikt` override) | ❌ DEAD | flips CORRECT rows to hallucinated on non-idiom (chem/math/grammar) rows. Use idiom-only, where it adds 0 over 0.828. |
| Threshold re-tuning (macro→class-0) | ❌ wash | class-0 F1 ≈ 0.838 either way (trades precision for recall). |
| takitajwar17 "answer key" dataset | ❌ DUD | 66.7% vs our 84.0% on context-truth rows. Do NOT blend. |
| **Fine-tuning** | ⬜ **UNTRIED** | Explicitly allowed (external Bengali data + the sample set). The only lever with a plausible path past ~0.85. High variance, uncertain transfer, needs real OOF discipline. |

---

## Rules that matter (verified)

- **Two phases.** Phase 1 = Kaggle LB. **Top 30 (private LB)** → invited to submit a
  Phase-2 package → organizers run it on a held-out fold → top 15 → in-person final at IUT.
- **Final-rank weights:** Private LB 20% · **Phase-2 held-out 50%** · Presentation 10% ·
  Paper 10% · Novelty 10%. ~80% of the score is Phase-2 + paper + novelty, **not** public LB.
- **Real metric:** binary F1 on class 0 (hallucinated), heaviest on the **C1 band**.
  Public LB shows macro-F1 (softer, different). **Report class-0 F1 internally, not macro.**
- **Phase-2 compute (hard):** offline, open-weight only, **<9h** on **P100 / 2×T4**,
  **<50 GB** weights, no internet at inference. → hardware ceiling ≈ **32B dense 4-bit**;
  **runtime <9h is the binding constraint.** Fine-tuning is allowed.
- **Deadline:** 2026-07-19 18:00 BST. 4 submissions/day. Select up to **2 finals**.

---

## Finals selection (do this on kaggle.com — no CLI exists for it)

Advancement = top-30 on the **private** LB, scored on your **2 selected** submissions
(best of the two counts). Select:

1. **`submission_contrastive.csv`** (LB 0.831) — best public *and* best on the real
   class-0 F1 metric (all 12 flips add true class-0 positives). Primary final.
2. **`submission_sa_wikt.csv`** (LB 0.828, highest pure OOF 0.8512) — best-of-2 insurance
   against the public→private shuffle.

---

## Phase-2 package status (the 50% weight — highest remaining EV)

- **`Phase_2/` reproduces LB 0.800 offline** (portable 7-signal notebook, ~5h on T4×2).
  This is the guaranteed-reproducible fallback final.
- **Gap:** the 0.828/0.831 result relies on signals NOT yet folded into the offline
  package (`signal_sa` + wiktionary rescue + the 12 contrastive flips). Before Phase-2:
  1. Fold `signal_sa` + wiktionary rescue + `build_contrastive.py` into the offline package.
  2. Budget check: self-answer stage alone took ~5.5h — trim (K=6→3, tokens 640→320) to fit <9h.
  3. One end-to-end offline verify run, then this becomes the packaged final.
- **Paper** (`Phase_2/paper/`, 4-page ACL): ⚠️ teammate names are placeholders — fill in.
  The enwiki + bigger-model negative results are a strong novelty story ("we measured the
  knowledge wall; cross-lingual grounding does not break it").

---

## File map (what we added this round)

| Path | What it is |
|---|---|
| `Approach_1/contrastive_analysis.py` | The winning-lever audit (exact + fuzzy). Read-only. Run it first. |
| `Approach_1/build_contrastive.py` | Builds `submission_contrastive.csv` (0.831) as a 12-flip delta on the 0.828 base. |
| `Approach_1/validate_enwiki.py` | Honest OOF gate that proved enwiki DEAD. |
| `Approach_1/validate_mathsolve.py` | Gate that proved the math-solver DEAD. |
| `Approach_1/validate_signal.py` | Generalized OOF gate (reproduces the 0.828 baseline for any new signal). |
| `Approach_1/kaggle_enwiki.py`, `kaggle_mathsolve.py`, `kaggle_gemma27.py` | Kaggle kernels for the three dead offensive levers (kept for the record / paper). |
| `Approach_1/results/signal_enwiki.json`, `signal_mathsolve.json` | Signal outputs (evidence for the negative results). |

**Reproduce the win:** `cd Approach_1 && PYTHONPATH=. python3 build_contrastive.py`
**Gate any NEW signal before spending an LB submission:** add it to `validate_signal.py`'s
stack and require it to lift noctx/class-0 F1 by > ~0.02 OR net-rescue ≥ 4 with no OOF regression.

---

## Recommended next steps (priority order)

1. **Lock the 2 finals now** (above) — protects the thin cushion, guarantees the private-LB shot.
2. **Harden the Phase-2 package** (50% of final score) — fold in `signal_sa` + wikt + the
   12 flips, trim to <9h, one offline verify. Highest EV remaining.
3. **If you want a real jump: fine-tuning.** It's the only untried lever with headroom.
   Be disciplined — honest OOF on the 299, expect high variance, and only submit on a clean
   positive margin. Do NOT chase public-LB noise (gaps of 0.001–0.003 are smaller than the
   public→private shuffle).
4. **Do not re-run the dead levers** (enwiki, gemma/bigger-model, math-solver, fuzzy
   contrastive, takitajwar17 blend). The evidence is in the ledger.
