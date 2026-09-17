"""Adjacency-leakage check: contiguous file-number blocks as CV folds vs stratified folds.

If consecutive Train files come from the same recording run, stratified K-fold puts near-duplicates
on both sides of the split and overestimates. Block CV breaks that adjacency.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Rail Corrugation"))
from code.pipeline import NormalReference, aggregate, side_relative_rows  # noqa: E402
from code.train import decode_side, make_detector  # noqa: E402
from common.metrics import rail_macro_f1  # noqa: E402

W = ROOT / "weights/Rail Corrugation"
lab = pd.read_csv(ROOT / "data/Rail_Corrugation/Train_Labels.csv")
# order rows by the file NUMBER, not lexicographically
lab["num"] = lab.filename.str.extract(r"(\d+)").astype(int)
order = lab.sort_values("num").index.to_numpy()
y = lab.label.to_numpy()
ct_v = joblib.load(W / "train_channel_tables.joblib")
CT, V = [c for c, _ in ct_v], [v for _, v in ct_v]
n = len(y)

# how clustered are the fault labels in file number?
num_by_label = {c: lab.sort_values("num")[lab.sort_values("num").label == c].num.to_numpy() for c in ["Side I", "Side II"]}
print("Side I file numbers:", num_by_label["Side I"].tolist())
print("Side II file numbers:", num_by_label["Side II"].tolist())
# speed similarity of adjacent fault files
sp = pd.Series(V, index=lab.index)
print("adjacent-number speed diffs, Side I:", np.round(np.abs(np.diff([sp[i] for i in lab.sort_values('num').index if lab.label[i]=='Side I'])), 2).tolist())

K = 5
blocks = np.array_split(order, K)
print("label counts per contiguous block:")
for b, idx in enumerate(blocks):
    print(f"  block {b}: {pd.Series(y[idx]).value_counts().to_dict()}")

oof = np.zeros((n, 2))
for b, te in enumerate(blocks):
    tr = np.setdiff1d(order, te)
    ref = NormalReference().fit([CT[i] for i in tr], [V[i] for i in tr], [y[i] for i in tr])
    feats = {i: aggregate(ref.excess(CT[i], V[i]), V[i]) for i in np.concatenate([tr, te])}
    rows, ys = [], []
    for i in tr:
        r1, r2 = side_relative_rows(feats[i]); rows += [r1, r2]; ys += [int(y[i] == "Side I"), int(y[i] == "Side II")]
    R = pd.DataFrame(rows).fillna(-9)
    det = make_detector(b).fit(R, ys)
    for i in te:
        r1, r2 = side_relative_rows(feats[i])
        oof[i] = det.predict_proba(pd.DataFrame([r1, r2]).fillna(-9)[R.columns])[:, 1]
    print(f"block {b} done", flush=True)

best = max(((rail_macro_f1(y, decode_side(oof[:, 0], oof[:, 1], t))["macro_f1"], t) for t in np.linspace(0.1, 0.9, 33)))
pred = decode_side(oof[:, 0], oof[:, 1], best[1])
rep = rail_macro_f1(y, pred)
print(f"\nBLOCK-CV macro F1 = {best[0]:.4f} (tau={best[1]:.2f})  vs stratified 0.8336±0.0139")
print("per-class:", {k: round(v, 3) for k, v in rep["per_class"].items()})
print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")))
