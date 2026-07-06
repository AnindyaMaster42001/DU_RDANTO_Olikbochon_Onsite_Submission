"""Parse bnwiktionary XML -> wikt_passages.jsonl (CPU, stdlib only).

Entry structure (verified):  =={{ভাষা|bn}}== / ===ভাবার্থ=== / {{বাগধারা|bn}} / # meaning
Word entries use ===অর্থ=== or plain '# def' lines under =={{ভাষা|bn}}==.
Emits {"t": headword, "x": "headword: meaning", "kind": "idiom"|"word"}.
"""
import html
import json
import re
import sys
import time

SRC = "/tmp/claude-1000/-media-ninadgns-ssd-Workspace-IUTDatathon/889200ed-1f80-4805-89d9-0f6e8316e525/scratchpad/wikt.xml"
OUT = "/tmp/claude-1000/-media-ninadgns-ssd-Workspace-IUTDatathon/889200ed-1f80-4805-89d9-0f6e8316e525/scratchpad/wikt_passages.jsonl"

t0 = time.time()
data = open(SRC, encoding="utf-8").read()
print(f"read {len(data)/1e6:.0f} MB in {time.time()-t0:.1f}s", flush=True)

page_re = re.compile(r"<page>(.*?)</page>", re.S)
title_re = re.compile(r"<title>(.*?)</title>", re.S)
text_re = re.compile(r'<text\b[^>]*>(.*?)</text>', re.S)


def clean(s):
    s = html.unescape(s)
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)            # templates
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)  # [[a|b]] -> b
    s = re.sub(r"</?[a-zA-Z][^>]*>", " ", s)          # html/ref tags
    s = re.sub(r"'{2,}", "", s)                       # bold/italic
    s = re.sub(r"^[#*:;]+", "", s).strip()            # list/indent markers
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def meanings_for_bn(text):
    """Return (list_of_meaning_strings, is_idiom) for the Bengali section."""
    # isolate the {{ভাষা|bn}} language block (until next top-level == lang ==)
    m = re.search(r"==\s*\{\{ভাষা\|bn\}\}\s*==(.*?)(?=\n==\s*\{\{ভাষা|\Z)", text, re.S)
    block = m.group(1) if m else text
    is_idiom = ("বাগধারা" in block) or ("ভাবার্থ" in block) or ("প্রবাদ" in block)
    out = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("#") and not line.startswith("#:") and not line.startswith("#*"):
            c = clean(line)
            if len(c) >= 2:
                out.append(c)
    return out, is_idiom


def parse_table_idioms(text):
    """Appendix/list pages are wikitables: '|-' rows, '| headword' then '| meaning'.
    Yields (headword, meaning)."""
    text = html.unescape(text)
    rows = re.split(r"\n\|-", text)
    for r in rows:
        cells = [c.strip() for c in re.split(r"\n\s*\|", r) if c.strip()
                 and not c.strip().startswith(("-", "+", "}", "!"))]
        # drop a leading table-open cell like '{|' fragment
        cells = [c for c in cells if not c.startswith("{|")]
        if len(cells) < 2:
            continue
        hw = clean(cells[0])
        hw = re.sub(r"[০-৯0-9]+$", "", hw).strip()          # strip trailing superscript idx
        meaning = clean(" ; ".join(cells[1:]))
        if 1 <= len(hw) <= 40 and len(meaning) >= 2:
            yield hw, meaning


n_pages = n_entries = n_idiom = n_word = n_table = 0
seen = set()
with open(OUT, "w", encoding="utf-8") as w:
    for pm in page_re.finditer(data):
        block = pm.group(1)
        n_pages += 1
        tm, xm = title_re.search(block), text_re.search(block)
        if not tm or not xm:
            continue
        title = html.unescape(tm.group(1)).strip()
        text = xm.group(1)

        # (1) appendix/list pages -> mine the idiom table
        if ("বাগধারা" in title and "তালিকা" in title) or title.startswith("প্রবাদ"):
            for hw, meaning in parse_table_idioms(text):
                if hw in seen:
                    continue
                seen.add(hw)
                n_entries += 1
                n_idiom += 1
                n_table += 1
                w.write(json.dumps({"t": hw, "x": f"{hw}: {meaning}", "kind": "idiom"},
                                   ensure_ascii=False) + "\n")
            continue

        if ":" in title:  # skip other namespaces
            continue
        if "{{ভাষা|bn}}" not in text and "ভাবার্থ" not in text:
            continue
        meanings, is_idiom = meanings_for_bn(text)
        if not meanings:
            continue
        if title in seen:
            continue
        seen.add(title)
        meaning = "; ".join(meanings[:4])
        kind = "idiom" if is_idiom else "word"
        n_entries += 1
        n_idiom += kind == "idiom"
        n_word += kind == "word"
        w.write(json.dumps({"t": title, "x": f"{title}: {meaning}", "kind": kind},
                           ensure_ascii=False) + "\n")

print(f"pages={n_pages}  entries={n_entries}  idiom={n_idiom} "
      f"(of which {n_table} from appendix tables)  word={n_word}", flush=True)
print(f"TOTAL parse time: {time.time()-t0:.1f}s", flush=True)
