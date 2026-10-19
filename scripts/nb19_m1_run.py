"""NB19 module M1 runner: classical geothermometer baseline vs ML.

Produces, in 06_results/nb19/:
  m1_geothermometer_metrics.csv     - each geothermometer, own valid subset
  m1_subset_definitions.csv         - subset sizes / state coverage
  m1_ml_vs_geothermometer.csv       - ML re-run on identical subsets, 3 protocols
  m1_paired_tests.csv               - paired Wilcoxon, ML vs best geothermometer
  m1_per_state.csv                  - per-state RMSE, best geothermometer vs ML

ML pipeline is identical to NB09: median imputation fitted on train, one-hot for
categoricals, 10-seed random split / GroupKFold(5) / leave-one-state-out on
states with >= 30 samples in the subset.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
import nb19_geothermometers as G

ROOT = Path(__file__).resolve().parents[1]          # PhD_Research_Operating_System
DATA = ROOT / "03_data_processed" / "geothermal_canonical_v2_0_0.csv"
OUT = ROOT / "06_results" / "nb19"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "Temperature(C)"
GROUP = "State"
MIN_STATE_N = 30
SEEDS = list(range(10))

df = pd.read_csv(DATA)
print(f"dataset: {df.shape[0]} samples, {df[GROUP].nunique()} states")

# ---------------------------------------------------------------- geothermometers
gt = G.compute_all(df)
y = df[TARGET]

rows = []
for name, spec in G.GEOTHERMOMETERS.items():
    m = G.metrics(y, gt[name])
    rows.append(dict(geothermometer=name, reference=spec["ref"],
                     ions=" + ".join(spec["needs"]), **m))
gt_metrics = pd.DataFrame(rows).sort_values("rmse")
gt_metrics.to_csv(OUT / "m1_geothermometer_metrics.csv", index=False)
print("\n=== classical geothermometers (each on its own computable subset) ===")
print(gt_metrics[["geothermometer", "ions", "n", "rmse", "mae", "bias", "r2"]]
      .to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

# ---------------------------------------------------------------- subsets
SUBSETS = {
    "silica": ["SiO2"],
    "na_k": ["Na", "K"],
    "all_ions": ["SiO2", "Na", "K", "Ca", "Mg"],
}
sub_idx = {}
sub_rows = []
for sname, cols in SUBSETS.items():
    mask = df[cols].notna().all(axis=1)
    idx = df.index[mask]
    sub_idx[sname] = idx
    counts = df.loc[idx, GROUP].value_counts()
    elig = counts[counts >= MIN_STATE_N]
    sub_rows.append(dict(subset=sname, required_ions=" + ".join(cols), n=len(idx),
                         n_states=int(counts.size), n_eligible_states=int(elig.size),
                         eligible_states=", ".join(sorted(elig.index))))
sub_def = pd.DataFrame(sub_rows)
sub_def.to_csv(OUT / "m1_subset_definitions.csv", index=False)
print("\n=== comparison subsets ===")
print(sub_def[["subset", "required_ions", "n", "n_eligible_states"]].to_string(index=False))


# ---------------------------------------------------------------- ML pipeline (NB09)
def model_factory():
    return {
        "catboost": CatBoostRegressor(iterations=700, learning_rate=0.04, depth=6,
                                      random_seed=42, verbose=0),
        "xgboost": XGBRegressor(n_estimators=550, learning_rate=0.04, max_depth=6,
                                subsample=0.9, colsample_bytree=0.9,
                                random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=450, learning_rate=0.05,
                                                       random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
        "dummy_median": DummyRegressor(strategy="median"),
    }


def prepare(train_df, test_df, cat_cols, num_cols):
    tr = pd.get_dummies(train_df, columns=cat_cols, drop_first=False)
    te = pd.get_dummies(test_df, columns=cat_cols, drop_first=False)
    te = te.reindex(columns=tr.columns, fill_value=0)
    if num_cols:
        imp = SimpleImputer(strategy="median")
        tr[num_cols] = imp.fit_transform(tr[num_cols])
        te[num_cols] = imp.transform(te[num_cols])
    return tr, te


def run_protocols(sub):
    """Return (metric rows, oof prediction frame) for one subset."""
    d = df.loc[sub].copy()
    X_raw = d.drop(columns=[TARGET, GROUP])
    y_all = d[TARGET]
    groups = d[GROUP]
    cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    metric_rows = []
    # per-sample out-of-fold predictions, one column per (protocol, model)
    oof = pd.DataFrame(index=d.index)

    # --- random split, 10 seeds (average prediction over the seeds where held out)
    acc = {m: pd.DataFrame(index=d.index, columns=SEEDS, dtype=float) for m in model_factory()}
    for seed in SEEDS:
        Xtr_r, Xte_r, ytr, yte = train_test_split(X_raw, y_all, test_size=0.2, random_state=seed)
        Xtr, Xte = prepare(Xtr_r, Xte_r, cat_cols, num_cols)
        for name, mdl in model_factory().items():
            mdl.fit(Xtr, ytr)
            p = mdl.predict(Xte)
            acc[name].loc[Xte_r.index, seed] = p
            metric_rows.append(dict(protocol="random_split", model=name, seed=seed,
                                    holdout="NA", n_test=len(yte),
                                    rmse=float(np.sqrt(mean_squared_error(yte, p))),
                                    mae=float(mean_absolute_error(yte, p)),
                                    r2=float(r2_score(yte, p))))
    for name in acc:
        oof[f"random_split|{name}"] = acc[name].mean(axis=1)

    # --- GroupKFold by state (every sample predicted exactly once)
    gkf = GroupKFold(n_splits=5)
    gk = {m: pd.Series(index=d.index, dtype=float) for m in model_factory()}
    for fold, (tr_i, te_i) in enumerate(gkf.split(X_raw, y_all, groups), start=1):
        Xtr_r, Xte_r = X_raw.iloc[tr_i], X_raw.iloc[te_i]
        ytr, yte = y_all.iloc[tr_i], y_all.iloc[te_i]
        Xtr, Xte = prepare(Xtr_r, Xte_r, cat_cols, num_cols)
        for name, mdl in model_factory().items():
            mdl.fit(Xtr, ytr)
            p = mdl.predict(Xte)
            gk[name].loc[Xte_r.index] = p
            metric_rows.append(dict(protocol="groupkfold", model=name, seed=42,
                                    holdout=f"fold_{fold}", n_test=len(yte),
                                    rmse=float(np.sqrt(mean_squared_error(yte, p))),
                                    mae=float(mean_absolute_error(yte, p)),
                                    r2=float(r2_score(yte, p))))
    for name in gk:
        oof[f"groupkfold|{name}"] = gk[name]

    # --- spatial holdout, leave-one-state-out on states with >= MIN_STATE_N
    counts = groups.value_counts()
    elig = counts[counts >= MIN_STATE_N].index.tolist()
    sp = {m: pd.Series(index=d.index, dtype=float) for m in model_factory()}
    for st in elig:
        te_i = groups.index[groups == st]
        tr_i = groups.index[groups != st]
        Xtr, Xte = prepare(X_raw.loc[tr_i], X_raw.loc[te_i], cat_cols, num_cols)
        ytr, yte = y_all.loc[tr_i], y_all.loc[te_i]
        for name, mdl in model_factory().items():
            mdl.fit(Xtr, ytr)
            p = mdl.predict(Xte)
            sp[name].loc[te_i] = p
            metric_rows.append(dict(protocol="spatial_holdout", model=name, seed=42,
                                    holdout=str(st), n_test=len(yte),
                                    rmse=float(np.sqrt(mean_squared_error(yte, p))),
                                    mae=float(mean_absolute_error(yte, p)),
                                    r2=float(r2_score(yte, p))))
    for name in sp:
        oof[f"spatial_holdout|{name}"] = sp[name]

    return pd.DataFrame(metric_rows), oof


all_metrics, all_oof = [], {}
raw_cache = OUT / "m1_ml_raw_runs.csv"
use_cache = raw_cache.exists() and all((OUT / f"m1_oof_{s}.csv").exists() for s in SUBSETS)
if use_cache:
    print("\n>>> reusing cached ML runs (delete m1_ml_raw_runs.csv to force re-run)")
    ml_raw = pd.read_csv(raw_cache)
    for sname in sub_idx:
        all_oof[sname] = pd.read_csv(OUT / f"m1_oof_{sname}.csv", index_col=0)
else:
    for sname, idx in sub_idx.items():
        print(f"\n>>> ML re-run on subset '{sname}' (n={len(idx)}) ...", flush=True)
        mrows, oof = run_protocols(idx)
        mrows["subset"] = sname
        all_metrics.append(mrows)
        all_oof[sname] = oof
        oof.to_csv(OUT / f"m1_oof_{sname}.csv")
    ml_raw = pd.concat(all_metrics, ignore_index=True)
    ml_raw.to_csv(raw_cache, index=False)

# aggregate: mean over seeds / folds / states, matching NB09 convention
ml_summary = (ml_raw.groupby(["subset", "protocol", "model"])[["rmse", "mae", "r2"]]
              .mean().reset_index())

# geothermometer metrics restricted to each subset, for the same rows
comp_rows = []
for sname, idx in sub_idx.items():
    for gname in G.GEOTHERMOMETERS:
        m = G.metrics(y.loc[idx], gt.loc[idx, gname])
        if m["n"] == 0:
            continue
        comp_rows.append(dict(subset=sname, protocol="none_required",
                              model=f"geothermometer:{gname}", **{k: m[k] for k in ("rmse", "mae", "r2")}))
comp = pd.concat([ml_summary, pd.DataFrame(comp_rows)], ignore_index=True)
comp.to_csv(OUT / "m1_ml_vs_geothermometer.csv", index=False)

print("\n=== ML vs classical, identical subsets ===")
for sname in sub_idx:
    print(f"\n--- subset: {sname} (n={len(sub_idx[sname])}) ---")
    blk = comp[comp.subset == sname]
    ml = blk[blk.protocol != "none_required"].pivot_table(index="model", columns="protocol", values="rmse")
    print("ML RMSE (C):")
    print(ml.to_string(float_format=lambda v: f"{v:7.2f}"))
    g = blk[blk.protocol == "none_required"][["model", "rmse", "r2"]].sort_values("rmse")
    print("Classical RMSE (C):")
    print(g.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

# ---------------------------------------------------------------- paired tests
test_rows = []
for sname, idx in sub_idx.items():
    oof = all_oof[sname]
    # best geothermometer on this subset by RMSE
    gsub = {gn: G.metrics(y.loc[idx], gt.loc[idx, gn]) for gn in G.GEOTHERMOMETERS}
    gsub = {k: v for k, v in gsub.items() if v["n"] > 0}
    best_g = min(gsub, key=lambda k: gsub[k]["rmse"])
    g_est = gt.loc[idx, best_g]
    for col in oof.columns:
        proto, model = col.split("|")
        if model == "dummy_median":
            continue
        pair = pd.DataFrame({"y": y.loc[idx], "ml": oof[col], "geo": g_est}).dropna()
        if len(pair) < 20:
            continue
        e_ml = (pair["ml"] - pair["y"]).abs()
        e_gt = (pair["geo"] - pair["y"]).abs()
        stat, p = wilcoxon(e_ml, e_gt)
        test_rows.append(dict(
            subset=sname, protocol=proto, ml_model=model,
            best_geothermometer=best_g, n_paired=len(pair),
            ml_rmse=float(np.sqrt(((pair["ml"] - pair["y"]) ** 2).mean())),
            gt_rmse=float(np.sqrt(((pair["geo"] - pair["y"]) ** 2).mean())),
            ml_mae=float(e_ml.mean()), gt_mae=float(e_gt.mean()),
            median_abs_err_diff=float((e_ml - e_gt).median()),
            wilcoxon_stat=float(stat), p_value=float(p),
            verdict=("ML better" if e_ml.mean() < e_gt.mean() else "geothermometer better")
                    + (" (p<0.05)" if p < 0.05 else " (n.s.)")))
paired = pd.DataFrame(test_rows)
paired.to_csv(OUT / "m1_paired_tests.csv", index=False)
print("\n=== paired Wilcoxon: ML vs best geothermometer (same samples) ===")
print(paired[["subset", "protocol", "ml_model", "best_geothermometer", "n_paired",
              "ml_rmse", "gt_rmse", "p_value", "verdict"]]
      .to_string(index=False, float_format=lambda v: f"{v:9.3g}"))

# ---------------------------------------------------------------- per-state
ps_rows = []
for sname, idx in sub_idx.items():
    oof = all_oof[sname]
    gsub = {gn: G.metrics(y.loc[idx], gt.loc[idx, gn]) for gn in G.GEOTHERMOMETERS}
    gsub = {k: v for k, v in gsub.items() if v["n"] > 0}
    best_g = min(gsub, key=lambda k: gsub[k]["rmse"])
    for st, g in df.loc[idx].groupby(GROUP):
        if len(g) < MIN_STATE_N:
            continue
        r = dict(subset=sname, state=st, n=len(g), best_geothermometer=best_g,
                 gt_rmse=G.metrics(y.loc[g.index], gt.loc[g.index, best_g])["rmse"])
        for col in ["spatial_holdout|catboost", "groupkfold|catboost", "random_split|catboost"]:
            if col in oof:
                r[col.replace("|", "_")] = G.metrics(y.loc[g.index], oof.loc[g.index, col])["rmse"]
        ps_rows.append(r)
per_state = pd.DataFrame(ps_rows)
per_state.to_csv(OUT / "m1_per_state.csv", index=False)
print("\n=== per-state RMSE (subset=all_ions) ===")
print(per_state[per_state.subset == "all_ions"].to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

print(f"\nwrote outputs to {OUT}")
