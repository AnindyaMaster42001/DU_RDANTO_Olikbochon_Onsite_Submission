"""Parametrized submission builder — reproduces the 0.828 stack and variants.

0.828 = 11-sig per-branch nested-CV logistic stacker (refit on all 299) + wikt
rescue-only (dict-confirmed 0->1 flips on exact-headword matches). This script
makes that recipe explicit and lets us add ret32wikt (idiom override) cleanly.

Usage:
  python3 build_sub.py base        -> submission_base11.csv        (stack only)
  python3 build_sub.py sawikt      -> submission_sa_wikt_repro.csv (stack + wikt rescue)  [== 0.828]
  python3 build_sub.py blend       -> submission_blend.csv (stack+ret32wikt override + wikt rescue)
Prints OOF macro-F1 / F1-hall for the labeled 299 in each case.
"""
import sys, json, os, numpy as np
import common as C
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

STACK=["a0judge","a0selfv","crosslingual","judge","nli","retrieval","substring","judge32","j32sv","ret32","sa"]
S=C.load_samples(); T=C.load_test(); Y=np.array([s["label"] for s in S])
CTX_S=np.array([C.has_context(s) for s in S]); CTX_T=np.array([C.has_context(r) for r in T])
IDS=[r["id"] for r in T]

def sig(n):
    for p in (f"results/signal_{n}.json", f"signal_{n}.json"):
        if os.path.exists(p): return json.load(open(p,encoding="utf-8"))
    raise FileNotFoundError(n)
SIG={n:sig(n) for n in STACK}
def scol(n): return np.asarray(SIG[n]["samples"],float)
def tcol(n): return np.array([SIG[n]["test"].get(i,0.5) for i in IDS],float)
def smat(names): return np.column_stack([scol(n) for n in names])
def tmat(names): return np.column_stack([tcol(n) for n in names])
def bt(y,p):
    ts=np.arange(.3,.71,.02); return float(ts[int(np.argmax([C.macro_f1(y.tolist(),(p>=t).astype(int).tolist()) for t in ts]))])

def oof_hard(names):
    acc=np.zeros(len(Y))
    for seed in range(5):
        pr=np.zeros(len(Y),int)
        for mask in (CTX_S,~CTX_S):
            X,y=smat(names)[mask],Y[mask]; o=np.zeros(len(y)); th=[]
            for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
                c=LogisticRegression(max_iter=1000).fit(X[tr],y[tr]); o[va]=c.predict_proba(X[va])[:,1]
                th.append(bt(y[tr],c.predict_proba(X[tr])[:,1]))
            pr[mask]=(o>=np.median(th)).astype(int)
        acc+=pr
    return (acc/5>=.5).astype(int)

def refit_test(names):
    """Per-branch refit on all 299, threshold=median of per-branch fold thresholds; predict test."""
    lab=np.zeros(len(T),int)
    for ms,mt in ((CTX_S,CTX_T),(~CTX_S,~CTX_T)):
        X,y=smat(names)[ms],Y[ms]; th=[]
        for tr,_ in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y):
            c=LogisticRegression(max_iter=1000).fit(X[tr],y[tr]); th.append(bt(y[tr],c.predict_proba(X[tr])[:,1]))
        thr=float(np.median(th))
        clf=LogisticRegression(max_iter=1000).fit(X,y)
        lab[mt]=(clf.predict_proba(tmat(names)[mt])[:,1]>=thr).astype(int)
    return lab

def wikt_rescue(labels):
    """rescue-only: on exact-headword test rows where wikt says faithful (>0.5) and base said 0, flip to 1."""
    w=sig("wikt"); n=0
    for k,i in enumerate(IDS):
        v=w["test"].get(i,0.5)
        if v>0.5 and labels[k]==0: labels[k]=1; n+=1
    print(f"  wikt rescue: {n} flips 0->1")
    return labels

def report(names,tag):
    h=oof_hard(names)
    print(f"[OOF] {tag:<22} macroF1={C.macro_f1(Y.tolist(),h.tolist()):.4f}  F1hall={C.f1_on_class(Y.tolist(),h.tolist(),0)[0]:.4f}")

mode=sys.argv[1] if len(sys.argv)>1 else "sawikt"
if mode=="base":
    report(STACK,"11-sig stack"); lab=refit_test(STACK); C.write_submission("submission_base11.csv",IDS,lab)
elif mode=="sawikt":
    report(STACK,"11-sig stack"); lab=wikt_rescue(refit_test(STACK)); C.write_submission("submission_sa_wikt_repro.csv",IDS,lab)
elif mode=="blend":
    report(STACK,"11-sig stack"); report(STACK+["ret32wikt"],"+ret32wikt")
    lab=wikt_rescue(refit_test(STACK+["ret32wikt"])); C.write_submission("submission_blend.csv",IDS,lab)
