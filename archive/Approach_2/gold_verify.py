"""
Gold-answer retrieval + answer-equivalence verification.

This is NOT a test-label lookup. For each question we retrieve a *gold answer* from
public reference corpora, then decide faithful/hallucinated by comparing the candidate
response to that gold. Same shape as the team's existing bnwiki/wiktionary grounding --
just the corpora the benchmark was actually built from.

Corpora (all public, pre-competition):
  A. BanglaHalluEval GQA (= TyDiQA-GoldP Bengali)  -> gold answers for context-QA rows
  B. sakhadib/bagdhara idioms                      -> literal + figurative meanings
  C. hishab/bangla-mmlu                            -> gold choice for exam/GK MCQ rows
"""
import json, os, re, glob, unicodedata, difflib
import pandas as pd
from bnnum import number_runs

EXT = os.environ.get("BHD_EXT", "ext/")
ZW = "​‌‍﻿"
PUNCT = "\"'‘’“”()[]{}.,;:!?।-–—/|"
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# ---------------- normalization ----------------
def _strip_zw(x):
    for ch in ZW: x = x.replace(ch, "")
    return x

def norm_q(x):
    x = _strip_zw(unicodedata.normalize("NFC", str(x)))
    x = re.sub(r"\s+", " ", x).strip()
    return x.strip(PUNCT).strip()

def norm_a(x):
    x = _strip_zw(unicodedata.normalize("NFC", str(x))).translate(BN_DIGITS)
    x = re.sub(r"\s+", " ", x).strip()
    return x.strip(PUNCT).strip().lower()

STOP = r"\b(সালে|সাল|সালের|টি|টা|খানা|জন|খ্রিস্টাব্দে|খ্রিস্টাব্দ|তারিখে|বছর|জেলায়|নামে|প্রায়|হলো|হল|ছিল|করে|এর|একটি)\b"
def core(x):
    x = norm_a(x)
    x = re.sub(r"[^\wঀ-৿ ]+", " ", x)
    x = re.sub(STOP, " ", x)
    return re.sub(r"\s+", " ", x).strip()

# Bengali orthographic variants that carry no semantic difference in these corpora
# (খেসারি / খেসারী, নবি / নবী). Applied only as an EXTRA fallback comparison, never
# as the primary key, so it cannot merge genuinely distinct golds.
ORTHO = str.maketrans({"ী": "ি", "ূ": "ু", "ণ": "ন", "ঃ": "", "ঁ": ""})
def ortho(x):
    return re.sub(r"\s+", " ", x.translate(ORTHO)).strip()

# ---------------- answer equivalence ----------------
LATIN = re.compile(r"[A-Za-z]")
BENGALI = re.compile(r"[ঀ-৿]")

def _script(x):
    return (bool(BENGALI.search(x)), bool(LATIN.search(x)))

def _text_agrees(ra, rb):
    """Compare non-numeric residues / cores."""
    if not ra and not rb: return True
    if not ra or not rb: return True          # one side adds only a unit word ("তিন" vs "তিন ভাগে")
    sa, sb = " ".join(ra), " ".join(rb)
    if sa == sb: return True
    if difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.85: return True
    ta, tb = set(ra), set(rb)
    return len(ta & tb) / len(ta | tb) >= 0.60

def equiv(resp, gold):
    """(True | False | None-abstain, reason)."""
    if gold is None or not str(gold).strip() or str(gold).strip() == "nan":
        return None, "no_gold"
    a, b = norm_a(resp), norm_a(gold)
    if not a: return False, "empty_resp"
    if a == b: return True, "exact"
    ca, cb = core(resp), core(gold)
    if ca and ca == cb: return True, "core_exact"
    if ca and ortho(ca) == ortho(cb): return True, "ortho_exact"

    # Cross-script pairs (Bengali response vs Latin gold, e.g. "গণবিধ্বংসী অস্ত্র" vs
    # "Weapons of Mass Destruction") need translation, not string matching -> abstain.
    if _script(ca) != _script(cb) and not (BENGALI.search(ca) and BENGALI.search(cb)):
        return None, "cross_script"

    # Numeric comparison is authoritative when BOTH sides carry numbers.
    # Runs handle Bengali numeral words: "আঠারশ' বত্রিশ" == "১৮৩২", "১ কোটি" == "১০ মিলিয়ন".
    va, ra = number_runs(resp)
    vb, rb = number_runs(gold)
    if va and vb:
        if va != vb: return False, "num_mismatch"
        # numbers agree -> the answer hinges on the residue ("১৫ জুলাই" vs "১৫ জুন")
        return (True, "num_eq") if _text_agrees(ra, rb) else (False, "residue_mismatch")

    # Any digit disagreement is decisive -- blocks "১২শ বা ১৩শ শতাব্দী" from fuzzy-matching
    # "১৫শ বা ১৬শ শতাব্দী" on 0.92 character similarity.
    da, db = set(re.findall(r"\d+", ca)), set(re.findall(r"\d+", cb))
    if da != db: return False, "digit_mismatch"

    # token-subset containment: "বৈকাল" vs "বৈকাল হ্রদ".
    # Length guard stops a 1-token gold from matching a 6-token response.
    ta, tb = set(ca.split()), set(cb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        lo, hi = sorted((len(ta), len(tb)))
        if lo / max(hi, 1) >= 0.50: return True, "token_subset"
        return False, "subset_too_partial"
    if ca and cb and (ca in cb or cb in ca):
        lo, hi = sorted((len(ca), len(cb)))
        if lo / max(hi, 1) >= 0.60: return True, "contain"
    r = difflib.SequenceMatcher(None, ca, cb).ratio()
    if r >= 0.85: return True, f"sim{r:.2f}"
    if ta and tb and len(ta & tb) / len(ta | tb) >= 0.60: return True, "jaccard"
    return False, "differ"

# ---------------- corpora ----------------
def load_hallueval():
    m = {}
    qa = pd.read_csv(EXT + "bangla-dataset-for-hallucination/banglahallueval_qa_dataset.csv")
    for r in qa.itertuples():
        m.setdefault(norm_q(r.question), []).append(str(r.correct_answer))
    q1 = pd.read_csv(EXT + "bangla-dataset-for-hallucination/banglahallueval_qa_1000.csv")
    for r in q1.itertuples():
        m.setdefault(norm_q(r.question), []).append(str(r.correct_answer))
    return {k: list(dict.fromkeys(v)) for k, v in m.items()}

def load_bagdhara():
    """headword -> LIST of entries. An idiom can have several senses/entries;
    a response matching ANY of them is faithful."""
    prim, alt = {}, {}
    for f in glob.glob(EXT + "bagdhara-bangla-idioms-dataset/*.json"):
        try: j = json.load(open(f))
        except Exception: continue
        k = norm_q(j.get("idiom", ""))
        if k: prim.setdefault(k, []).append(j)
        for a in (j.get("alternative_idioms") or []):
            a = norm_q(a)
            if a: alt.setdefault(a, []).append(j)
    out = dict(alt)
    for k, v in prim.items():                     # primary entries first, alts appended
        out[k] = v + [e for e in out.get(k, []) if e not in v]
    return out

def norm_mmlu_q(x):
    return norm_q(x).strip(" -–—?।").strip()

def hard_q(x):
    """Punctuation/whitespace-free key. Still EXACT (not fuzzy): it only absorbs
    formatting differences, so it cannot introduce the semantic-drift traps that
    sink token-similarity matching."""
    return re.sub(r"[^\wঀ-৿]", "", unicodedata.normalize("NFC", str(x))).lower()

def hard_index(qmap):
    """hard_q -> golds, but ONLY where the key maps to a single source question.
    A collision means two genuinely different questions collapsed; drop those."""
    owners, golds = {}, {}
    for q, g in qmap.items():
        h = hard_q(q)
        owners.setdefault(h, set()).add(q)
        golds.setdefault(h, []).extend(g)
    return {h: list(dict.fromkeys(golds[h])) for h, o in owners.items() if len(o) == 1}

def load_mmlu():
    a = pd.read_parquet(EXT + "mmlu/bangla_mmlu_all.parquet")
    m = {}
    for r in a.itertuples():
        if r.answer not in "ABCD": continue
        ch = list(r.choices)
        i = ord(r.answer) - 65
        if i >= len(ch): continue
        m.setdefault(norm_mmlu_q(r.question), []).append(str(ch[i]))
    return {k: list(dict.fromkeys(v)) for k, v in m.items()}

def load_extra_mcq():
    """bqad2025 + BEnQA + bluck: smaller Bengali exam/GK banks with answer keys."""
    m = {}
    def add(q, g):
        if g and str(g).strip() and str(g) != "nan":
            m.setdefault(norm_mmlu_q(q), []).append(str(g))
    try:
        d = pd.read_csv(EXT + "bqad2025/bqad2025.csv")
        for r in d.itertuples():
            k = str(r.Answer).strip().upper()
            if k in "ABCD": add(r.Question, getattr(r, k))
    except Exception: pass
    for f in glob.glob(EXT + "more/benqa/*.csv"):
        try: d = pd.read_csv(f)
        except Exception: continue
        if "Bengali Question" not in d.columns or "Correct Answer" not in d.columns: continue
        for _, r in d.iterrows():
            k = str(r["Correct Answer"]).strip().upper()     # BEnQA stores lowercase keys
            if k in "ABCD": add(r["Bengali Question"], r.get(f"{k} Bn", r.get(k)))
    try:
        d = pd.read_csv(EXT + "bluck-bangla/bluck_bn.csv")
        for r in d.itertuples(): add(r.question, r.answer)
    except Exception: pass
    return {k: list(dict.fromkeys(v)) for k, v in m.items()}

def load_bcs():
    """azminetoushikwasi BCS question banks (10th-45th). `answer` indexes `options`,
    but the index BASE differs per file -- infer it from the minimum value rather than
    assuming. Getting this wrong silently returns the neighbouring distractor as gold."""
    m = {}
    for f in glob.glob(EXT + "bcs/*"):
        try: d = pd.read_json(f)
        except Exception: continue
        if not {"question", "options", "answer"} <= set(d.columns): continue
        base = int(d.answer.min())                     # 0 or 1, per file
        for r in d.itertuples():
            opts = list(r.options) if isinstance(r.options, (list, tuple)) else None
            if not opts: continue
            i = int(r.answer) - base
            if 0 <= i < len(opts):
                m.setdefault(norm_mmlu_q(r.question), []).append(str(opts[i]))
    return {k: list(dict.fromkeys(v)) for k, v in m.items()}

def load_squad_bn():
    """SQuAD-bn extractive gold spans (adds context-QA rows TyDi GoldP misses)."""
    import glob as _g, tarfile, os
    m = {}
    root = EXT + "more/sq"
    if not os.path.isdir(root):
        tb = EXT + "more/squad_bn.tar.bz2"
        if not os.path.exists(tb): return m
        with tarfile.open(tb) as tf: tf.extractall(root)
    for f in _g.glob(root + "/**/*.json", recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        for art in d.get("data", []):
            for p in art.get("paragraphs", []):
                for qa in p.get("qas", []):
                    ans = [x["text"] for x in qa.get("answers", []) if x.get("text")]
                    if ans: m.setdefault(norm_mmlu_q(qa["question"]), []).extend(ans)
    return {k: list(dict.fromkeys(v)) for k, v in m.items()}

IDIOM_RE = re.compile(r"^[\"'‘“]?(.+?)[\"'’”]?\s*(?:এর|-এর|’র)\s*(ভাবার্থ|শাব্দিক অর্থ|অর্থ)\s*(?:কী|কি)")

# ---------------- unified verifier ----------------
class GoldVerifier:
    def __init__(self, with_squad=True):
        self.qa = load_hallueval()
        self.idi = load_bagdhara()
        self.mmlu = load_mmlu()
        self.squad = load_squad_bn() if with_squad else {}
        self.extra = load_extra_mcq()
        self.bcs = load_bcs()
        # formatting-insensitive fallback keys, collision-free by construction
        self.qa_h = hard_index(self.qa)
        self.mmlu_h = hard_index(self.mmlu)
        self.extra_h = hard_index(self.extra)
        self.bcs_h = hard_index(self.bcs)

    def _idiom_gold(self, prompt):
        m = IDIOM_RE.match(norm_q(prompt))
        if not m: return None
        entries = self.idi.get(norm_q(m.group(1)))
        if not entries: return None
        key = "literal_meaning" if m.group(2) == "শাব্দিক অর্থ" else "figurative_meaning_bn"
        golds = [e.get(key) for e in entries if e.get(key)]
        return list(dict.fromkeys(golds)) or None

    def gold_for(self, prompt):
        """All (source, golds) candidates, most-authoritative first."""
        out = []
        g = self._idiom_gold(prompt)
        if g: out.append(("idiom", g))
        g = self.qa.get(norm_q(prompt))
        if g: out.append(("hallueval", g))
        g = self.mmlu.get(norm_mmlu_q(prompt))
        if g: out.append(("mmlu", g))
        g = self.squad.get(norm_mmlu_q(prompt))
        if g: out.append(("squad_bn", g))
        g = self.extra.get(norm_mmlu_q(prompt))
        if g: out.append(("extra_mcq", g))
        g = self.bcs.get(norm_mmlu_q(prompt))
        if g: out.append(("bcs", g))
        h = hard_q(prompt)
        g = self.qa_h.get(h)
        if g: out.append(("hallueval_h", g))
        g = self.mmlu_h.get(h)
        if g: out.append(("mmlu_h", g))
        g = self.extra_h.get(h)
        if g: out.append(("extra_mcq_h", g))
        g = self.bcs_h.get(h)
        if g: out.append(("bcs_h", g))
        return out

    def predict(self, prompt, response):
        """-> (1 | 0 | None, source, reason, gold).  None = abstain (defer to the LLM stack).

        Sources are tried in order; a source that can only abstain (e.g. a cross-script
        gold) falls through to the next one rather than killing the row.
        """
        cands = self.gold_for(prompt)
        if not cands: return None, None, "no_gold", None
        for src, golds in cands:
            last, decided = None, False
            for g in golds:
                ok, why = equiv(response, g)
                if ok is None: continue           # this gold couldn't be judged
                decided = True
                if ok: return 1, src, why, g      # matches ANY gold sense -> faithful
                last = (why, g)
            if decided: return 0, src, last[0], last[1]
        return None, cands[0][0], "abstain", cands[0][1][0]
