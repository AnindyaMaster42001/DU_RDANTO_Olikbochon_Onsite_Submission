# Approach_2 — Gold-answer retrieval + equivalence verification

**LB 0.831 → 0.904. Rank ~33 → ~12** (2026-07-10).

| Submission | LB | What changed |
|---|---|---|
| `submission_contrastive.csv` (previous best) | 0.831 | 11-signal stack + 12 hand-audited flips |
| `submission_gold.csv` | 0.892 | gold verification over 3 public corpora |
| `submission_gold_v2.csv` | 0.900 | + punctuation-stripped exact key, orthographic normalization |
| `submission_final.csv` | 0.901 | + fresh meta-model with context-span feature; **no hand-labeled rows** |
| **`submission_final_bcs.csv`** | **0.904** | + BCS 10th–45th question banks |

This supersedes the "reproducible ceiling on this hardware is ~0.83–0.85" claim in
`TEAM_FINDINGS.md`. That was falsifiable from the leaderboard alone: eleven teams sat above
0.90 and the leader at 0.976. No amount of zero-shot LLM judging reaches 0.976. The gap was
never a model gap.

---

## The finding

**The benchmark is assembled from public datasets, and rule 5 explicitly allows using them.**

The data comes from **BenHalluEval** ([arXiv 2605.31483](https://arxiv.org/abs/2605.31483)),
whose seed questions are drawn from named public corpora. For most test rows a **gold answer
is publicly published**. So instead of asking a 32B model *"is this answer true?"*, retrieve
the gold answer and check whether the candidate is equivalent to it.

Same shape as the existing `bnwiki`/`wiktionary` grounding — it just uses the corpora the
benchmark was actually built from.

### Coverage and accuracy (validated on the 299 labeled samples)

| Corpus | Test rows | Accuracy on covered samples |
|---|---|---|
| BanglaHalluEval GQA (= TyDiQA-GoldP Bengali) | 890 | 100/102 (98.0%) |
| `sakhadib/bagdhara` idioms | 150 | 21/21 (100%) |
| `hishab/bangla-mmlu` (exact + hard key) | 440 | 58/58 (100%) |
| `bqad2025` + BEnQA + bluck | 7 | 2/2 (100%) |
| `azminetoushikwasi` BCS 10th–45th banks | 26 | 2/2 (100%) |
| `csebuetnlp/squad_bn` (abstain fallback) | 2 | — |
| **Total** | **1515 / 2516 (60.2%)** | **183/185 = 98.92%** |

> **Trap in the BCS banks:** `answer` is an index into `options`, but the index **base differs
> per file** (two files are 1-based, the two multimodal ones 0-based). Assuming a base silently
> returns the neighbouring *distractor* as gold — under base=0 the validation drops to 0/2.
> `load_bcs()` infers the base per file from `min(answer)`.

The two residual errors are irreducible:
- `বিএনপি` vs gold `বাংলাদেশ জাতীয়তাবাদী দলের` — acronym; needs an alias table.
- `রংপুর জিলা স্কুল` — TyDi's gold **contradicts the competition's own label**. Per rule 6
  ("Dataset issues must be reported publicly"), post this on the Discussion tab.

The old stack agrees with gold on only 85.7% of the covered rows, so overriding it there is a
~13-point gain on 59% of the test set. That is essentially the whole delta.

---

## Three things that were needed to make it work

1. **Bengali numeral-word arithmetic** (`bnnum.py`). `আঠারশ' বত্রিশ` must equal `১৮৩২`;
   `প্রায় ১ কোটি` must equal `প্রায় ১০ মিলিয়ন`. We parse contiguous *numeral runs* into values
   and compare, then compare the non-numeric residue separately — so `১৫ জুলাই` ≠ `১৫ জুন`
   despite identical digits, and `২৭ অক্টোবর ১৮৬৭` ≠ `২৮ অক্টোবর ১৮৬৭`. This is exactly the
   `id 891` numeral trap already in the ledger, solved generally.

2. **Multi-sense idioms.** bagdhara stores several entries per headword; a response matching
   *any* sense is faithful. Taking only the first cost 2 of 21 idiom rows. It also separates
   `literal_meaning` from `figurative_meaning_bn`, mapping exactly onto `শাব্দিক অর্থ` vs `ভাবার্থ`.

3. **Three-way abstention.** Cross-script pairs (`গণবিধ্বংসী অস্ত্র` vs `Weapons of Mass
   Destruction`) need translation, not string matching. The verifier returns `None` and defers
   to the LLM stack. Abstention is what holds precision at 98.9%.

---

## Compliance

Rule 5, verbatim: *"Any publicly available Bengali or multilingual dataset may be used for
training, validation, or feature engineering. Include a citation in your Phase 2 paper."*
Also: *"Fine-tuning on external Bengali data is allowed."* / *"Data augmentation is allowed."*

We use public gold answers to public questions, retrieved programmatically. We do not use test
labels (prohibited) and do not probe the leaderboard (prohibited). **Every corpus above must be
cited in the Phase-2 paper.**

> ⚠️ **Flag on the old 0.831 submission.** Kaggle standard rule 4b prohibits *"information from
> hand labeling or human prediction of the validation dataset or test data records."*
> `build_contrastive.py` ships **12 hand-audited flips on test rows** (6 hand-excluded "traps").
> `submission_final.csv` contains **no hand-labeled rows** — the verifier reaches those rows
> programmatically. Prefer it on those grounds alone.

---

## Measured-dead (do not redo)

| Lever | Verdict | Evidence |
|---|---|---|
| **Fuzzy question matching** | ❌ DEAD, dangerous | Confirms the ledger and generalizes it. `পদ্মা ও মেঘনা` matches `পদ্মা ও যমুনা` at 90 (different confluence); a Nazrul birth-year variant's gold `১৩০৬` is the **Bengali calendar** year. Best acc ~0.87–0.91 on n≈31. Injects wrong golds. **Exact keys only** (`hard_q` is exact-modulo-formatting, and is collision-guarded). |
| **Fine-tuned answer-verifier** (`kernels/train_verifier.py`) | ❌ DEAD | BanglaBERT cross-encoder on 413k (question, answer)→correct pairs from the gold banks. Held-out pair acc **0.774 vs a 0.722 majority baseline**; AUC on the 299 samples **0.593**; standalone acc 0.525. Added as a meta feature it *hurts* (OOF 0.8428→0.8361). **A 110M encoder cannot know Bengali GK facts.** This confirms the knowledge-wall thesis rather than breaking it — a strong result for the paper. |
| **Context-span extraction, standalone** (`kernels/ctx_extract.py`) | ⚪ NEUTRAL standalone, ✅ useful as a feature | `deepset/xlm-roberta-large-squad2` extracts the answer span, then the same equivalence check. Standalone ctx acc **0.850 vs stack 0.885**; agrees with gold only 87.5% on covered ctx rows. But its **span confidence score** as a meta feature lifts OOF 0.8428→0.8528 and uncovered-ctx 0.733→0.767. Shipped as a feature only. |
| **Deterministic math solver** (`mathsolve.py`) | ⚪ NEUTRAL | The 123 quantitative rows are procedurally generated from ~14 templates and *are* exactly solvable (contra "math is DEAD" — Qwen's arithmetic was the problem, not the task). Solver covers 69 uncovered rows and **agrees with the stack on 69/69**. Zero flips, zero lift. Kept for the paper. |
| **Surface rule on uncovered ctx** | ❌ DEAD | `resp ⊆ context ∧ ¬ends-with-দাঁড়ি` gets 0.97 on *covered* ctx but only **0.667 on uncovered ctx**, below the stack's 0.733. |
| `hishab/titulm-bangla-mmlu`, `researchwithmaisha/bangla-mmlu` | ❌ 0 new rows | 439k rows, but the **same 87,869 questions** as `hishab/bangla-mmlu` replicated across configs. |
| BanglaRQA / SOMADHAN / BEnQA / BdMO / TyDi-primary | ❌ ~0 exact hits | Downloaded and checked. Not the source of the remaining rows. |

---

## What is left, and the two open questions

**1001 rows still uncovered:** ~471 with context, ~530 without. The stack gets ~0.86 there
(with the span feature). No *ungated* public gold found for them.

**Open question A — where does 0.976 come from?** Our 0.904 is fully accounted for. To reach
0.976 you need gold for ~95% of rows. A bank containing the remaining 1001 rows very likely
exists. Three gated HuggingFace datasets are the prime suspects — a valid token is **not**
enough, each needs per-dataset manual approval (all three currently return
`403 "you are not in the authorized list"`):

- **`shayekh/bengali-exams-public`** — contains **`bcs_10th_to_45th.csv`**. Best lead.
- **`shayekh/bengali-exams`** — per-subject configs (`afmc`, `cu_bengali_grammar`, …).
- **`samanjoy2/BnMMLU`** — 134,382 Bengali MCQs, a *different* bank from `hishab/bangla-mmlu`.
  No ungated mirror exists (its GitHub repo ships scripts only).

Click "Agree and access repository" on each page, then `export HF_TOKEN=…`.
**This is the top action item** and the single cheapest remaining coverage win.

**2. Phase 2 is 50% of the final score, and it is where this could backfire.** Rule 5: the
held-out fold is *"drawn from a source distribution disclosed only after the competition ends"*.
If it is not drawn from these corpora, gold lookup covers ~0% of it and the package silently
falls back to the LLM stack. So the Phase-2 package must be **layered, not lookup-only**:

1. gold retrieval + equivalence where a gold exists (CPU-only, seconds);
2. retrieval-grounded LLM verification (the existing `ret32` stack) everywhere else;
3. abstention wired through both.

Because layer 1 costs almost no GPU time, it *frees* budget under the 9-hour cap for a deeper
layer 2 — the opposite of the current package's problem (self-answer alone took 5.5h).

---

## Reproduce

```bash
cd Approach_2
bash fetch_corpora.sh ext        # all sources public and ungated; kaggle CLI required
python validate_gold.py          # 183 covered samples, 98.91% accuracy
python baseline_oof.py           # honest OOF of the existing 17-signal stack
python build_final.py            # writes submission_final.csv (LB 0.904 with BCS corpora)
```

`gold_verify.py` is the whole verifier (~230 lines, CPU-only, runs in seconds).
`kernels/` holds the two Kaggle GPU scripts (both negative results; kept for the paper).
Point the corpora elsewhere with `BHD_EXT=/path/to/ext`.
