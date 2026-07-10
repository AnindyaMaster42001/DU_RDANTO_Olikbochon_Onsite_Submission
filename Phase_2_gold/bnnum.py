"""Bengali numeral-word -> integer. Returns None if not fully parseable."""
import re, unicodedata

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

UNITS = {
    "শূন্য": 0, "এক": 1, "দুই": 2, "দু": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "পাচ": 5,
    "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10,
    "এগারো": 11, "এগার": 11, "বারো": 12, "বার": 12, "তেরো": 13, "তের": 13,
    "চৌদ্দ": 14, "পনেরো": 15, "পনের": 15, "ষোলো": 16, "ষোল": 16,
    "সতেরো": 17, "সতের": 17, "আঠারো": 18, "আঠার": 18, "উনিশ": 19, "ঊনিশ": 19,
    "বিশ": 20, "কুড়ি": 20, "একুশ": 21, "বাইশ": 22, "তেইশ": 23, "চব্বিশ": 24,
    "পঁচিশ": 25, "পচিশ": 25, "ছাব্বিশ": 26, "সাতাশ": 27, "আটাশ": 28, "ঊনত্রিশ": 29, "উনত্রিশ": 29,
    "ত্রিশ": 30, "একত্রিশ": 31, "বত্রিশ": 32, "তেত্রিশ": 33, "চৌত্রিশ": 34,
    "পঁয়ত্রিশ": 35, "ছত্রিশ": 36, "সাঁইত্রিশ": 37, "আটত্রিশ": 38, "ঊনচল্লিশ": 39,
    "চল্লিশ": 40, "একচল্লিশ": 41, "বিয়াল্লিশ": 42, "তেতাল্লিশ": 43, "চুয়াল্লিশ": 44,
    "পঁয়তাল্লিশ": 45, "ছেচল্লিশ": 46, "সাতচল্লিশ": 47, "আটচল্লিশ": 48, "ঊনপঞ্চাশ": 49,
    "পঞ্চাশ": 50, "একান্ন": 51, "বায়ান্ন": 52, "তিপ্পান্ন": 53, "চুয়ান্ন": 54,
    "পঞ্চান্ন": 55, "ছাপ্পান্ন": 56, "সাতান্ন": 57, "আটান্ন": 58, "ঊনষাট": 59,
    "ষাট": 60, "একষট্টি": 61, "বাষট্টি": 62, "তেষট্টি": 63, "চৌষট্টি": 64,
    "পঁয়ষট্টি": 65, "ছেষট্টি": 66, "সাতষট্টি": 67, "আটষট্টি": 68, "ঊনসত্তর": 69,
    "সত্তর": 70, "একাত্তর": 71, "বাহাত্তর": 72, "তিয়াত্তর": 73, "চুয়াত্তর": 74,
    "পঁচাত্তর": 75, "ছিয়াত্তর": 76, "সাতাত্তর": 77, "আটাত্তর": 78, "ঊনআশি": 79,
    "আশি": 80, "একাশি": 81, "বিরাশি": 82, "তিরাশি": 83, "চুরাশি": 84,
    "পঁচাশি": 85, "ছিয়াশি": 86, "সাতাশি": 87, "অষ্টাশি": 88, "ঊননব্বই": 89,
    "নব্বই": 90, "একানব্বই": 91, "বিরানব্বই": 92, "তিরানব্বই": 93, "চুরানব্বই": 94,
    "পঁচানব্বই": 95, "ছিয়ানব্বই": 96, "সাতানব্বই": 97, "আটানব্বই": 98, "নিরানব্বই": 99,
}
MULT = {"শ": 100, "শো": 100, "শত": 100, "হাজার": 1000, "সহস্র": 1000,
        "লক্ষ": 100000, "লাখ": 100000, "কোটি": 10000000,
        # cross-script scale words appear in gold answers ("প্রায় ১০ মিলিয়ন")
        "মিলিয়ন": 1000000, "million": 1000000, "বিলিয়ন": 1000000000,
        "billion": 1000000000, "ট্রিলিয়ন": 1000000000000, "thousand": 1000}
SKIP = {"প্রায়", "সালে", "সাল", "জন", "টি", "টা", "খ্রিস্টাব্দে", "খ্রিস্টাব্দ",
        "সালের", "মধ্যে", "এর", "ও", "আর", "প্রায়ই", "about", "approximately",
        # abbreviation variants of "AD/CE" that differ only orthographically
        "খৃঃ", "খ্রিঃ", "খ্রি", "খৃ", "খ্রীঃ", "খ্রীষ্টাব্দ", "খ্রিষ্টাব্দ", "ad", "ce", "সন"}

def _clean(s):
    s = unicodedata.normalize("NFC", str(s)).translate(BN_DIGITS)
    s = re.sub(r"[^\wঀ-৿\s.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(t.strip(".ঃ") or t for t in s.split())

def _word_val(w):
    """Handle compounds like আঠারশ (18*100) written as one token."""
    if w in UNITS: return ("unit", UNITS[w])
    if w in MULT: return ("mult", MULT[w])
    for suf, m in (("শত", 100), ("শো", 100), ("শ", 100)):
        if w.endswith(suf) and len(w) > len(suf):
            stem = w[: -len(suf)]
            if stem in UNITS: return ("compound", UNITS[stem] * m)
    return None

def parse_bn_number(text):
    """Parse a Bengali number expression to int, or None if unparseable/absent."""
    s = _clean(text)
    if not s: return None
    toks = [t for t in s.split() if t not in SKIP]
    if not toks: return None
    total, cur, seen = 0, 0, False
    for t in toks:
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            v = float(t)
            cur = cur + v if cur else v
            seen = True
            continue
        wv = _word_val(t)
        if wv is None:
            return None                      # unknown token -> refuse
        kind, v = wv
        if kind == "mult":
            if not seen and cur == 0: cur = 1
            if v >= 1000:
                total += (cur or 1) * v; cur = 0
            else:
                cur = (cur or 1) * v
            seen = True
        else:
            cur += v; seen = True
    if not seen: return None
    val = total + cur
    return int(val) if float(val).is_integer() else val

def _is_numeral_tok(t):
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", t)) or _word_val(t) is not None

def number_runs(text):
    """Split into (numeric values in order, non-numeric residue tokens).

    A 'run' is a maximal contiguous stretch of numeral tokens, parsed as one value.
    So '২৭ অক্টোবর ১৮৬৭' -> ([27, 1867], ['অক্টোবর'])
    and 'আঠারশ বত্রিশ সালে' -> ([1832], [])   (সালে is a SKIP word)
    """
    toks = [t for t in _clean(text).split() if t not in SKIP]
    values, residue, run = [], [], []
    def flush():
        if run:
            v = parse_bn_number(" ".join(run))
            if v is not None: values.append(float(v))
            else: residue.extend(run)
            run.clear()
    for t in toks:
        if _is_numeral_tok(t): run.append(t)
        else: flush(); residue.append(t)
    flush()
    return values, residue

def numeric_forms(text):
    """All numeric readings of `text`: literal digits found, plus word-parse."""
    s = _clean(text)
    out = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)}
    p = parse_bn_number(text)
    if p is not None: out.add(float(p))
    return out

def has_number_words(text):
    return any(_word_val(t) is not None for t in _clean(text).split())

if __name__ == "__main__":
    for t in ["আঠারশ' বত্রিশ সালে", "১৮৩২", "দুই লক্ষ তেরো হাজার দুই শত জন",
              "প্রায় ১ কোটি", "১০০০০০০০", "২৬ মার্চ", "তিন", "৩", "১৯ টি", "৯ দিন"]:
        print(f"{t:34s} -> parse={parse_bn_number(t)}  forms={numeric_forms(t)}")
