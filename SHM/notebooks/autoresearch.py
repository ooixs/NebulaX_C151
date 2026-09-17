"""SHM improvement phase: residual diagnosis, then candidate refinements under 8-fold CV × 3 seeds."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SHM.code.pipeline import damage_sum, fit_C, load_series, mape  # noqa: E402

W = ROOT / "weights/SHM"; D = ROOT / "data/SHM"
lab = pd.read_csv(D / "Train_Labels.csv"); Dt = lab.damage.to_numpy()
cyc = joblib.load(W / "train_cycles.joblib")
series_stats = pd.DataFrame([dict(mean=x.mean(), std=x.std(), mx=x.max(), mn=x.min(), skew=pd.Series(x).skew())
                             for x in (load_series(D / "Train" / f) for f in lab.filename)])
S5 = np.array([damage_sum(c, 5.0) for c in cyc]); C5 = fit_C(S5, Dt); r = Dt / (S5 / C5)
print("=== residual diagnosis (ratio true/pred, m=5) ===")
print("ratio quantiles:", np.percentile(r, [0, 10, 50, 90, 100]).round(3))
diag = series_stats.assign(damage=Dt, ratio=r, logS=np.log(S5), n_cyc=[len(c) for c in cyc],
                           max_range=[c[:, 0].max() for c in cyc], mean_of_cycles=[np.average(c[:, 1], weights=c[:, 2]) for c in cyc])
print(diag.corr()["ratio"].round(3).sort_values())
print("worst 6 files:\n", diag.assign(f=lab.filename).sort_values("ratio")[["f", "damage", "ratio", "mean", "std", "mx"]].head(3).round(3).to_string(),
      "\n", diag.assign(f=lab.filename).sort_values("ratio")[["f", "damage", "ratio", "mean", "std", "mx"]].tail(3).round(3).to_string())


def cv(pred_fn, name, seeds=3):
    """pred_fn(tr_idx, te_idx) -> predictions for te_idx. Returns mean 1-MAPE."""
    sc = []
    for s in range(seeds):
        oof = np.zeros(len(Dt))
        for tr, te in KFold(8, shuffle=True, random_state=s).split(Dt):
            oof[te] = pred_fn(tr, te)
        sc.append(1 - mape(Dt, oof))
    print(f"{name:60s} CV 1-MAPE = {np.mean(sc):.4f} ± {np.std(sc):.4f}", flush=True)
    return float(np.mean(sc))


def miner(m, **kw):
    S = np.array([damage_sum(c, m, **kw) for c in cyc])
    return lambda tr, te: S[te] / fit_C(S[tr], Dt[tr])


res = {}
res["m=5 baseline"] = cv(miner(5.0), "0. baseline m=5, half-cycle residue")
# 1. fine m grid chosen per fold
Sgrid = {m: np.array([damage_sum(c, m) for c in cyc]) for m in np.round(np.arange(4.90, 5.101, 0.01), 2)}
def fine_m(tr, te):
    best = min(Sgrid, key=lambda m: mape(Dt[tr], Sgrid[m][tr] / fit_C(Sgrid[m][tr], Dt[tr])))
    return Sgrid[best][te] / fit_C(Sgrid[best][tr], Dt[tr])
res["fine m"] = cv(fine_m, "1. m chosen per fold on a 0.01 grid (4.90-5.10)")
# 2. bilinear S-N: slope m1 above knee amplitude a_k, m2 below (Eurocode-style m and m+2), C shared
def bilinear_S(m1, m2, ak):
    out = []
    for c in cyc:
        a = c[:, 0] / 2; n = c[:, 2]
        hi = a >= ak
        out.append(np.sum(n[hi] * a[hi] ** m1) + np.sum(n[~hi] * a[~hi] ** m2 * ak ** (m1 - m2)))
    return np.array(out)
SB = {(m2, ak): bilinear_S(5.0, m2, ak) for m2 in (3.0, 7.0, 9.0) for ak in (1.0, 2.0, 4.0, 8.0)}
def bilinear(tr, te):
    key = min(SB, key=lambda k: mape(Dt[tr], SB[k][tr] / fit_C(SB[k][tr], Dt[tr])))
    return SB[key][te] / fit_C(SB[key][tr], Dt[tr])
res["bilinear"] = cv(bilinear, "2. bilinear S-N (m1=5 above knee, m2 in {3,7,9}, knee in {1,2,4,8})")
# 3. mean-stress: Gerber / SWT-like exponent on (1 - mean/su)
def ms_S(su, expo):
    out = []
    for c in cyc:
        a = c[:, 0] / 2; mn = c[:, 1]; n = c[:, 2]
        aeq = a / np.clip(1 - np.sign(mn) * (np.abs(mn) / su) ** expo, 0.05, None)
        out.append(np.sum(n * aeq ** 5.0))
    return np.array(out)
SM = {(su, e): ms_S(su, e) for su in (60, 100, 200, 400, 800) for e in (1.0, 2.0)}
def meanstress(tr, te):
    key = min(SM, key=lambda k: mape(Dt[tr], SM[k][tr] / fit_C(SM[k][tr], Dt[tr])))
    return SM[key][te] / fit_C(SM[key][tr], Dt[tr])
res["mean-stress"] = cv(meanstress, "3. Goodman(exp 1)/Gerber(exp 2) mean-stress, su in {60..800}")
# 4. residue: full cycles for the largest-range residue only ("close the biggest loop")
def S_bigres():
    out = []
    for c in cyc:
        a = c[:, 0] / 2; n = c[:, 2].copy()
        half = n == 0.5
        if half.any():
            j = np.flatnonzero(half)[np.argmax(a[half])]; n[j] = 1.0
        out.append(np.sum(n * a ** 5.0))
    return np.array(out)
S4 = S_bigres(); res["big residue full"] = cv(lambda tr, te: S4[te] / fit_C(S4[tr], Dt[tr]), "4. largest residue half-cycle counted as full")
# 5. ridge residual model on log ratio with rainflow features, weights 1/D
from sklearn.linear_model import Ridge
feat = np.column_stack([np.log(S5), np.log([damage_sum(c, 3.0) for c in cyc]), np.log([damage_sum(c, 7.0) for c in cyc]),
                        series_stats["mean"], series_stats["std"], series_stats["mx"], np.log([len(c) for c in cyc])])
def ridge_resid(tr, te):
    C = fit_C(S5[tr], Dt[tr]); base = S5 / C
    mu, sd = feat[tr].mean(0), feat[tr].std(0) + 1e-9; Z = (feat - mu) / sd
    rm = Ridge(alpha=10.0).fit(Z[tr], np.log(Dt[tr] / base[tr]), sample_weight=1 / Dt[tr])
    return base[te] * np.exp(rm.predict(Z[te]))
res["ridge residual"] = cv(ridge_resid, "5. m=5 + ridge residual correction (7 features, alpha=10)")
# 6. C fitted by weighted median vs per-fold m and C jointly on a 0.05 grid (3..7) - sanity that m=5 is global
Sg2 = {m: np.array([damage_sum(c, m) for c in cyc]) for m in np.round(np.arange(4.5, 5.51, 0.05), 2)}
def joint(tr, te):
    best = min(Sg2, key=lambda m: mape(Dt[tr], Sg2[m][tr] / fit_C(Sg2[m][tr], Dt[tr])))
    return Sg2[best][te] / fit_C(Sg2[best][tr], Dt[tr])
res["joint m 0.05"] = cv(joint, "6. m on 0.05 grid (4.5-5.5) per fold")
joblib.dump(res, W / "autoresearch_results.joblib")
print("\nsummary:"); [print(f"  {k:20s} {v:.4f}") for k, v in res.items()]
