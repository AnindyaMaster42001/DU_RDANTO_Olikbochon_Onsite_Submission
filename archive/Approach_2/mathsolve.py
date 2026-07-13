"""
Deterministic solver for the procedurally-generated Bengali quantitative rows.

The team's ledger marks math as DEAD because Qwen-32B makes systematic arithmetic
errors. That is a property of the *solver*, not the *problem*: these rows come from a
small set of templates with closed-form answers. We compute the answer exactly and
compare. Abstains (returns None) on any prompt that does not match a known template.
"""
import re, unicodedata

BN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
N = r"([\d,]+(?:\.\d+)?)"          # digit group, thousands-comma tolerant

def _t(s):
    return unicodedata.normalize("NFC", str(s)).translate(BN)

def _f(x):
    return float(str(x).replace(",", "").strip())

def _nums(s):
    return [_f(x) for x in re.findall(r"[\d,]+(?:\.\d+)?", _t(s))]

# Each rule: (compiled regex, lambda over captured groups -> answer)
RULES = [
    # X টাকা, ratio a:b:c among three partners, second partner's share
    (rf"{N}\s*টাকা\s*তিন\s*ব্যবসায়িক\s*অংশীদারের.*?{N}\s*:\s*{N}\s*:\s*{N}\s*অনুপাতে.*?দ্বিতীয়\s*অংশীদার",
     lambda g: _f(g[0]) * _f(g[2]) / (_f(g[1]) + _f(g[2]) + _f(g[3]))),

    # sugar:water = a:b, total T litres, how much WATER (second term)
    (rf"মিশ্রণে\s*চিনি\s*ও\s*পানির\s*অনুপাত\s*{N}\s*:\s*{N}.*?মোট\s*মিশ্রণ\s*{N}\s*লিটার.*?পানি",
     lambda g: _f(g[2]) * _f(g[1]) / (_f(g[0]) + _f(g[1]))),

    # two buses, same direction, speeds u and v, after t hours -> |u-v|*t
    (rf"একই\s*দিকে\s*দুইটি\s*বাস.*?গতিবেগ\s*ঘণ্টায়\s*{N}\s*কিমি.*?ঘণ্টায়\s*{N}\s*কিমি.*?{N}\s*ঘণ্টা\s*পর",
     lambda g: abs(_f(g[0]) - _f(g[1])) * _f(g[2])),

    # price +a% then -b% on the new price, start P
    (rf"প্রথম\s*দফায়\s*{N}\s*%\s*বেড়ে.*?দ্বিতীয়\s*দফায়\s*{N}\s*%\s*কমে.*?শুরুর\s*দাম\s*{N}\s*টাকা",
     lambda g: _f(g[2]) * (1 + _f(g[0]) / 100) * (1 - _f(g[1]) / 100)),

    # initial price P, +a%, then b% discount on the increased price
    (rf"প্রাথমিক\s*মূল্য\s*{N}\s*টাকা.*?{N}\s*%\s*বৃদ্ধি.*?{N}\s*%\s*ছাড়",
     lambda g: _f(g[0]) * (1 + _f(g[1]) / 100) * (1 - _f(g[2]) / 100)),

    # simple interest: principal P, rate r%, t years -> P*r*t/100
    (rf"{N}\s*টাকা\s*মূলধনে\s*বার্ষিক\s*{N}\s*%\s*সরল\s*সুদের\s*হারে\s*{N}\s*বছরে\s*মোট\s*সুদ",
     lambda g: _f(g[0]) * _f(g[1]) * _f(g[2]) / 100),
    (rf"{N}\s*টাকা\s*{N}\s*বছরের\s*জন্য\s*বার্ষিক\s*{N}\s*%\s*সরল\s*সুদে",
     lambda g: _f(g[0]) * _f(g[1]) * _f(g[2]) / 100),

    # work rate: a days and b days alone -> together ab/(a+b)
    (rf"ক\s*একাই\s*{N}\s*দিনে.*?খ\s*একাই\s*তা\s*{N}\s*দিনে.*?একত্রে",
     lambda g: (_f(g[0]) * _f(g[1]) / (_f(g[0]) + _f(g[1]))) if (_f(g[0]) + _f(g[1])) else None),

    # mother:daughter = a:b, sum S -> daughter = S*b/(a+b)
    (rf"মা\s*ও\s*মেয়ের\s*বয়সের\s*অনুপাত\s*{N}\s*:\s*{N}.*?সমষ্টি\s*{N}\s*বছর.*?মেয়ের\s*বয়স",
     lambda g: _f(g[2]) * _f(g[1]) / (_f(g[0]) + _f(g[1]))),

    # two brothers a:b, sum S -> YOUNGER = S*min(a,b)/(a+b)
    (rf"দুই\s*ভাইয়ের\s*বয়সের\s*অনুপাত\s*{N}\s*:\s*{N}.*?সমষ্টি\s*{N}\s*বছর.*?ছোট\s*ভাইয়ের",
     lambda g: _f(g[2]) * min(_f(g[0]), _f(g[1])) / (_f(g[0]) + _f(g[1]))),

    # cows:goats = a:b, total T -> goats (second) = T*b/(a+b)
    (rf"গরু\s*ও\s*ছাগলের\s*সংখ্যার\s*অনুপাত\s*{N}\s*:\s*{N}.*?মোট\s*পশুর\s*সংখ্যা\s*{N}.*?ছাগলের",
     lambda g: _f(g[2]) * _f(g[1]) / (_f(g[0]) + _f(g[1]))),

    # rui:katla = a:b, total T -> katla (second)
    (rf"রুই\s*ও\s*কাতলা\s*মাছের\s*সংখ্যার\s*অনুপাত\s*{N}\s*:\s*{N}.*?মোট\s*মাছের\s*সংখ্যা\s*{N}.*?কাতলা",
     lambda g: _f(g[2]) * _f(g[1]) / (_f(g[0]) + _f(g[1]))),

    # profit / loss on cost price -- both phrasings
    (rf"ক্রয়মূল্য\s*{N}\s*টাকা.*?{N}\s*%\s*লাভে", lambda g: _f(g[0]) * (1 + _f(g[1]) / 100)),
    (rf"ক্রয়মূল্য\s*{N}\s*টাকা.*?{N}\s*%\s*ক্ষতিতে", lambda g: _f(g[0]) * (1 - _f(g[1]) / 100)),
    (rf"{N}\s*টাকায়\s*কেনা\s*হয়েছিল.*?{N}\s*%\s*লাভে", lambda g: _f(g[0]) * (1 + _f(g[1]) / 100)),
    (rf"{N}\s*টাকায়\s*কেনা\s*হয়েছিল.*?{N}\s*%\s*ক্ষতিতে", lambda g: _f(g[0]) * (1 - _f(g[1]) / 100)),
]
RULES = [(re.compile(p, re.S), f) for p, f in RULES]

def solve(prompt):
    """Exact answer for a templated quantitative prompt, else None."""
    p = _t(prompt)
    for rx, fn in RULES:
        m = rx.search(p)
        if not m: continue
        try:
            v = fn(m.groups())
        except ZeroDivisionError:
            return None
        if v is None or v != v: return None
        return float(v)
    return None

def verify(prompt, response, rtol=1e-6):
    """1 faithful / 0 hallucinated / None abstain."""
    truth = solve(prompt)
    if truth is None: return None, None
    cand = _nums(response)
    if not cand: return None, truth
    # accept if ANY number in the response equals the exact answer
    for c in cand:
        if abs(c - truth) <= rtol * max(1.0, abs(truth)):
            return 1, truth
    return 0, truth
