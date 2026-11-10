r"""nb19 M11 — the baselines and preprocessing alternatives the papers do not yet have.

FOUR QUESTIONS, ALL UNDER SPATIAL HOLDOUT (the deployment-relevant protocol)

1. COORDINATE-ONLY SPATIAL BASELINE. The standard null model for a spatial prediction problem is
   "predict from nearby observations". The papers' KNN baseline uses all features, which is a
   different thing. If a k-nearest-neighbour or inverse-distance model on latitude and longitude
   alone competes with the tree ensembles under geographic extrapolation, that must be reported.

2. NATIVE MISSING-VALUE HANDLING. XGBoost routes NaN down a learned default branch. The published
   pipeline median-imputes first and discards that. With Depth 83.5% missing and SiO2 66% missing,
   imputation is doing a great deal of silent work.

3. NEAREST-STATE MEAN. Under leave-one-state-out, a defensible trivial predictor is the mean of
   the geographically nearest training state, rather than a global constant.

4. ALTERNATIVE IMPUTATION. Median is the crudest choice; kNN imputation is the obvious comparison.

VALIDATION
The median-dummy arm must reproduce the published 29.42 degC. If it does not, the fold definitions
differ from NB09's and nothing else here is comparable.

The XGBoost arms differ from one another ONLY in preprocessing, so they are internally comparable.
They are not calibrated against NB09's published XGBoost value, whose hyperparameters are set in
that notebook and are not reused here.

Output: 06_results/nb19/m11_spatial_baselines.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

ROS = Path(__file__).resolve().parents[1]
DATA = ROS / "03_data_processed/geothermal_canonical_v2_0_0.csv"
OUT = ROS / "06_results/nb19/m11_spatial_baselines.csv"

TARGET, GROUP = "Temperature(C)", "State"
PUBLISHED_DUMMY = 29.42
TOL = 0.01


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def xgb():
    return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, subsample=0.85,
                        colsample_bytree=0.85, random_state=42, n_jobs=24)


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET]
    groups = df[GROUP]
    counts = groups.value_counts()
    states = sorted(counts[counts >= 30].index.tolist())

    feats = [c for c in df.columns if c not in (TARGET, GROUP)
             and pd.api.types.is_numeric_dtype(df[c])]
    coords = ["Latitude", "Longitude"]
    print(f"{len(states)} held-out states, {len(feats)} numeric features, {len(df)} rows\n")

    rows = []
    for st in states:
        te = groups[groups == st].index
        tr = groups[groups != st].index
        ytr, yte = y.loc[tr], y.loc[te]

        def add(name, pred):
            rows.append(dict(method=name, state=st, n_test=len(te), rmse=rmse(yte, pred)))

        add("dummy_median", DummyRegressor(strategy="median").fit(df.loc[tr, coords], ytr)
            .predict(df.loc[te, coords]))

        # 1 — coordinate-only spatial baselines
        for k in (1, 5, 10):
            knn = KNeighborsRegressor(n_neighbors=k).fit(df.loc[tr, coords], ytr)
            add(f"coord_knn_k{k}", knn.predict(df.loc[te, coords]))
        idw = KNeighborsRegressor(n_neighbors=10, weights="distance").fit(df.loc[tr, coords], ytr)
        add("coord_idw_k10", idw.predict(df.loc[te, coords]))

        # 3 — nearest-state mean, by centroid distance
        cent = df.loc[tr].groupby(GROUP)[coords].mean()
        here = df.loc[te, coords].mean()
        nearest = ((cent - here) ** 2).sum(axis=1).idxmin()
        add("nearest_state_mean",
            np.full(len(te), y.loc[tr][groups.loc[tr] == nearest].mean()))

        # 2 and 4 — the same model under three preprocessing regimes
        Xtr_raw, Xte_raw = df.loc[tr, feats], df.loc[te, feats]

        med = SimpleImputer(strategy="median").fit(Xtr_raw)
        add("xgb_median_impute", xgb().fit(med.transform(Xtr_raw), ytr)
            .predict(med.transform(Xte_raw)))

        add("xgb_native_nan", xgb().fit(Xtr_raw, ytr).predict(Xte_raw))

        knni = KNNImputer(n_neighbors=5).fit(Xtr_raw)
        add("xgb_knn_impute", xgb().fit(knni.transform(Xtr_raw), ytr)
            .predict(knni.transform(Xte_raw)))

        print(f"  {st:12s} done")

    res = pd.DataFrame(rows)
    summ = res.groupby("method").rmse.mean().sort_values()

    got = summ["dummy_median"]
    print(f"\nVALIDATION  dummy_median {got:.4f} vs published {PUBLISHED_DUMMY} "
          f"-> {'MATCH' if abs(got - PUBLISHED_DUMMY) < TOL else 'MISMATCH'}")
    if abs(got - PUBLISHED_DUMMY) >= TOL:
        print("fold definitions differ from NB09 — results not comparable. Nothing written.")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"written: {OUT}\n")
    print("MEAN SPATIAL-HOLDOUT RMSE, best first (degC)")
    for k, v in summ.items():
        print(f"   {k:22s} {v:8.3f}")

    print("\nCOVERAGE_NOTE: spatial holdout only; one model family (XGBoost) for the "
          "preprocessing arms, which differ from each other ONLY in preprocessing and are not "
          "calibrated against NB09's published XGBoost figure. Coordinate baselines use no "
          "geochemistry at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
