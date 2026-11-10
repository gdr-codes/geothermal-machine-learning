r"""nb19 M13 — how much of the random-split score is looking the well up?

THE OBJECTION THIS TESTS
Paper 1's central result is a 9.26 degC gap between random-split and spatial-holdout RMSE,
attributed to "regional geochemical distribution shift". But the canonical dataset's 2,684
samples sit at only 885 distinct (Latitude, Longitude) pairs:

    2,084 of 2,684 samples (77.6%) are at a coordinate shared with another sample
    the largest single coordinates carry 127, 124, 94, 66 and 65 samples
    within a repeated coordinate the temperature spread is a MEDIAN of 1.41 degC,
        against a dataset-wide standard deviation of 33.73 degC
    under a random 80/20 split, a mean of 76.0% of TEST rows have their exact
        coordinate present in TRAIN (10 seeds, range 71.8-79.7%)

So for three test rows in four, a training row exists at the identical location with what is
very nearly the same answer. A tree can score well by recognising the well rather than by
learning geochemistry. If that is doing the work, then the random-split end of the protocol gap
is inflated and part of the gap is ordinary near-duplicate leakage rather than distribution
shift — a much more mundane mechanism than the one the paper names.

WHAT THIS ADDS: a rung on the ladder
    random_split      80/20 by row      locations shared across the split   (published)
    location_grouped  5-fold by (lat,lon)  NO location in both sides        (NEW)
    groupkfold        5-fold by State                                       (published)
    spatial_holdout   leave-one-state-out                                   (published)

location_grouped removes duplicate-well leakage while leaving regional information intact, so it
separates the two mechanisms that random-vs-spatial confounds.

VALIDATION BEFORE CONCLUSION
The random_split arm must reproduce NB12's published means for all four feature families and all
three models — twelve numbers — or the replication is wrong and nothing else here can be
trusted. The script exits non-zero if it fails.

Protocol, models, hyperparameters, seeds and preprocessing are NB12's, copied from
nb19_m10_depth_ablation.py which already validated them. No notebook is touched.

Output: 06_results/nb19/m13_location_leakage.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor

ROS = Path(__file__).resolve().parents[1]
DATA = ROS / "03_data_processed/geothermal_canonical_v2_0_0.csv"
NB12 = (ROS / "github_org_bootstrap/phd-geothermal-ml/manual_bootstrap/step_by_step_notebooks"
            / "12_ablation_feature_families_no_scripts/outputs/summary/tables"
            / "all_runs_feature_family_ablation.csv")
OUT = ROS / "06_results/nb19/m13_location_leakage.csv"

TARGET = "Temperature(C)"
TOL = 0.01
SEEDS = list(range(10))

# Only the two families reproducible from the canonical CSV alone. NB12's ratios_only and
# combined_all include 12 ENGINEERED geothermometer proxies that are not columns in the dataset,
# so reproducing them would mean re-implementing NB12's feature engineering and risking a
# silently different definition. These two are enough for the question: physical_only is the
# pure look-up-the-well case, chemical_only is the one that matters — whether geochemistry still
# predicts temperature once you cannot recognise the location.
FAMILIES = {
    "physical_only": ["Latitude", "Longitude", "Depth(ft)"],
    "chemical_only": None,        # the 17 dissolved species, filled from the data below
}


def models():
    """COPIED VERBATIM from nb19_m10_depth_ablation.py, which reproduces NB12 exactly.

    A first version of this file carried subsample/colsample 0.9 and 400 -> 500 ExtraTrees,
    written from memory rather than copied. GradientBoosting still matched to four decimals
    because its parameters happened to be right, while XGBoost and voting drifted by 0.02-0.10
    degC — small enough to look like version noise and be waved through. The validation guard
    caught it. Do not retype these; copy them.
    """
    return {
        "gradient_boosting": GradientBoostingRegressor(n_estimators=350, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
        "xgboost": XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7,
                                subsample=0.85, colsample_bytree=0.85,
                                random_state=42, n_jobs=24),
        "voting": VotingRegressor(estimators=[
            ("gbm", GradientBoostingRegressor(n_estimators=350, learning_rate=0.05,
                                              max_depth=3, random_state=42)),
            ("xgb", XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7,
                                 subsample=0.85, colsample_bytree=0.85,
                                 random_state=42, n_jobs=24)),
            ("et", ExtraTreesRegressor(n_estimators=400, max_depth=None, min_samples_leaf=1,
                                       n_jobs=-1, random_state=42)),
        ]),
    }


def fit_score(model, Xtr, ytr, Xte, yte):
    """M10's prepare(): impute in place on DataFrames so column names reach the estimator."""
    num = list(Xtr.columns)
    imp = SimpleImputer(strategy="median")
    a, b = Xtr.copy(), Xte.copy()
    a[num] = imp.fit_transform(a[num])
    b[num] = imp.transform(b[num])
    p = model.fit(a, ytr).predict(b)
    return float(np.sqrt(mean_squared_error(yte, p))), float(r2_score(yte, p))


def main():
    d = pd.read_csv(DATA)
    num = [c for c in d.select_dtypes("number").columns if c != TARGET]
    physical = ["Latitude", "Longitude", "Depth(ft)"]
    chemical = [c for c in num if c not in physical]
    if len(chemical) != 17:
        print(f"expected 17 dissolved species, found {len(chemical)}: {chemical}")
        return 1
    FAMILIES["chemical_only"] = chemical

    loc = list(zip(d.Latitude, d.Longitude))
    codes = pd.factorize(pd.Series(loc))[0]
    print(f"{len(d)} samples at {len(set(loc))} distinct locations")

    rows = []
    for fam, cols in FAMILIES.items():
        X, y = d[cols], d[TARGET]
        for mname, proto in [(k, v) for k, v in models().items()]:
            # ---- random split, 10 seeds: the arm that must reproduce ----
            for s in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
                r, r2 = fit_score(models()[mname], Xtr, ytr, Xte, yte)
                rows.append(dict(family=fam, model=mname, protocol="random_split",
                                 fold=s, rmse=r, r2=r2))
            # ---- location-grouped 5-fold: the new arm ----
            gkf = GroupKFold(n_splits=5)
            for i, (tr, te) in enumerate(gkf.split(X, y, groups=codes)):
                r, r2 = fit_score(models()[mname], X.iloc[tr], y.iloc[tr],
                                  X.iloc[te], y.iloc[te])
                rows.append(dict(family=fam, model=mname, protocol="location_grouped",
                                 fold=i, rmse=r, r2=r2))
            print(f"   {fam:14s} {mname:18s} done")

    out = pd.DataFrame(rows)

    # ── VALIDATION: the random arm must match NB12 ──────────────────────────
    pub = (pd.read_csv(NB12).query("protocol == 'random_split'")
           .groupby(["feature_family", "model_name"]).rmse.mean())
    got = out.query("protocol == 'random_split'").groupby(["family", "model"]).rmse.mean()
    print("\nVALIDATION — random split against NB12's published means")
    bad = 0
    for (fam, mod), v in got.items():
        p = pub.get((fam, mod))
        ok = p is not None and abs(v - p) < TOL
        bad += (not ok)
        print(f"   {fam:14s} {mod:18s} {v:8.4f} vs {p:8.4f}  {'MATCH' if ok else 'MISMATCH'}")
    if bad:
        print(f"\n{bad} mismatch(es) — the replication is wrong; reporting nothing else.")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\nHOW MUCH OF THE RANDOM-SPLIT SCORE SURVIVES REMOVING DUPLICATE LOCATIONS?")
    piv = out.groupby(["family", "model", "protocol"]).rmse.mean().unstack()
    piv["leak_degC"] = piv["location_grouped"] - piv["random_split"]
    piv["leak_pct"] = piv["leak_degC"] / piv["random_split"] * 100
    print(piv.round(3).to_string())

    print("\nCOVERAGE_NOTE: this isolates duplicate-COORDINATE leakage only. Samples at "
          "different coordinates within the same field may still be near-duplicates, so the "
          "figures below are a LOWER bound on location leakage. It does not re-run NB09-NB11 "
          "and makes no claim about their published values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
