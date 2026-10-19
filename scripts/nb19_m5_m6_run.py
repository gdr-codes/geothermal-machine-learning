"""NB19 modules M5 + M6.

M5  Mechanistic test of the spatial-collapse explanation (Papers 1, 3).
    The papers attribute the random->spatial gap to spatial autocorrelation
    (Tobler's first law) but never test it. Here we:
      - compute Moran's I for temperature and for model residuals, with a
        permutation test, using row-standardised k-nearest-neighbour weights
        on great-circle distance;
      - regress per-state spatial-holdout RMSE on candidate drivers (training
        support, geographic distance to the training centroid, within-state
        temperature spread, geochemical Mahalanobis distance, target offset)
        to explain the ~6x state-level RMSE spread.

M6  Spatial-block cross-validation (Papers 1, 3).
    Reviewers object that state boundaries are administrative, not geological.
    We repeat the holdout with geology-agnostic contiguous spatial blocks
    (k-means on lat/lon) and check that the protocol gap replicates.

Outputs to 06_results/nb19/.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_data_processed" / "geothermal_canonical_v2_0_0.csv"
OUT = ROOT / "06_results" / "nb19"
OUT.mkdir(parents=True, exist_ok=True)

TARGET, GROUP = "Temperature(C)", "State"
MIN_N, SEEDS = 30, list(range(10))
EARTH_R_KM = 6371.0

df = pd.read_csv(DATA)
X_raw = df.drop(columns=[TARGET, GROUP])
y_all = df[TARGET]
groups = df[GROUP]
cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in X_raw.columns if c not in cat_cols]


def prepare(tr_raw, te_raw):
    tr = pd.get_dummies(tr_raw, columns=cat_cols, drop_first=False)
    te = pd.get_dummies(te_raw, columns=cat_cols, drop_first=False)
    te = te.reindex(columns=tr.columns, fill_value=0)
    imp = SimpleImputer(strategy="median")
    tr[num_cols] = imp.fit_transform(tr[num_cols])
    te[num_cols] = imp.transform(te[num_cols])
    return tr.astype(float), te.astype(float)


def haversine_matrix(lat, lon):
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_weights(D, k=8):
    """Row-standardised binary k-nearest-neighbour spatial weights."""
    n = D.shape[0]
    W = np.zeros((n, n))
    order = np.argsort(D, axis=1)
    for i in range(n):
        nb = [j for j in order[i] if j != i][:k]
        W[i, nb] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def morans_i(x, W, n_perm=999, rng=None):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.all():
        W = W[np.ix_(ok, ok)]
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = W / rs
        x = x[ok]
    n = len(x)
    z = x - x.mean()
    denom = float((z ** 2).sum())
    if denom == 0:
        return dict(I=np.nan, expected_I=np.nan, p_perm=np.nan, n=n)
    num = float(z @ W @ z)
    S0 = float(W.sum())
    I = (n / S0) * (num / denom)
    rng = rng or np.random.default_rng(42)
    perm = np.empty(n_perm)
    for b in range(n_perm):
        zp = rng.permutation(z)
        perm[b] = (n / S0) * float(zp @ W @ zp) / denom
    p = float((np.abs(perm) >= abs(I)).mean())
    return dict(I=float(I), expected_I=float(-1.0 / (n - 1)), p_perm=p, n=n,
                perm_mean=float(perm.mean()), perm_sd=float(perm.std(ddof=1)))


# =============================================================== M5
print("=== M5: spatial autocorrelation (Moran's I, kNN k=8, 999 permutations) ===")
D = haversine_matrix(df["Latitude"], df["Longitude"])
W = knn_weights(D, k=8)

# out-of-fold residuals under each protocol (CatBoost, the spatial champion)
def oof_predictions():
    preds = {}
    # random split (mean over seeds where held out)
    acc = pd.DataFrame(index=df.index, columns=SEEDS, dtype=float)
    for seed in SEEDS:
        tr_r, te_r, ytr, yte = train_test_split(X_raw, y_all, test_size=0.2, random_state=seed)
        Xtr, Xte = prepare(tr_r, te_r)
        m = CatBoostRegressor(iterations=700, learning_rate=0.04, depth=6,
                              random_seed=42, verbose=0).fit(Xtr, ytr)
        acc.loc[te_r.index, seed] = m.predict(Xte)
    preds["random_split"] = acc.mean(axis=1)
    # spatial holdout (states with >= MIN_N)
    sp = pd.Series(index=df.index, dtype=float)
    counts = groups.value_counts()
    for st in counts[counts >= MIN_N].index:
        te_i = groups.index[groups == st]
        tr_i = groups.index[groups != st]
        Xtr, Xte = prepare(X_raw.loc[tr_i], X_raw.loc[te_i])
        m = CatBoostRegressor(iterations=700, learning_rate=0.04, depth=6,
                              random_seed=42, verbose=0).fit(Xtr, y_all.loc[tr_i])
        sp.loc[te_i] = m.predict(Xte)
    preds["spatial_holdout"] = sp
    return preds


oof_path = OUT / "m5_oof_predictions.csv"
if oof_path.exists():
    print("  reusing cached OOF predictions")
    oof = pd.read_csv(oof_path, index_col=0)
else:
    oof = pd.DataFrame(oof_predictions())
    oof.to_csv(oof_path)

mi_rows = [dict(variable="Temperature(C)", **morans_i(y_all.to_numpy(), W))]
for proto in oof.columns:
    resid = (y_all - oof[proto]).to_numpy()
    mi_rows.append(dict(variable=f"residual_{proto}", **morans_i(resid, W)))
mi = pd.DataFrame(mi_rows)
mi.to_csv(OUT / "m5_morans_i.csv", index=False)
print(mi[["variable", "n", "I", "expected_I", "p_perm"]]
      .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

# ---- per-state drivers of spatial holdout RMSE
print("\n=== M5: what drives the state-level RMSE spread? ===")
Xnum = df[num_cols].copy()
Xnum = Xnum.fillna(Xnum.median())
cov = EmpiricalCovariance().fit(Xnum.values)
train_centroid = df[["Latitude", "Longitude"]].mean().values

rows = []
counts = groups.value_counts()
for st in counts[counts >= MIN_N].index:
    m = groups == st
    idx = df.index[m]
    if "spatial_holdout" not in oof or oof.loc[idx, "spatial_holdout"].isna().all():
        continue
    yt = y_all.loc[idx]
    yp = oof.loc[idx, "spatial_holdout"]
    st_ll = df.loc[idx, ["Latitude", "Longitude"]].mean().values
    # geographic distance from this state's centroid to the centroid of all OTHER data
    others = df.index.difference(idx)
    other_centroid = df.loc[others, ["Latitude", "Longitude"]].mean().values
    dist_km = float(haversine_matrix([st_ll[0], other_centroid[0]],
                                     [st_ll[1], other_centroid[1]])[0, 1])
    maha = float(np.sqrt(cov.mahalanobis(Xnum.loc[idx].mean().values.reshape(1, -1))[0]))
    rows.append(dict(
        state=st, n=int(m.sum()), n_train=int(len(df) - m.sum()),
        rmse=float(np.sqrt(mean_squared_error(yt, yp))),
        r2=float(r2_score(yt, yp)),
        train_support_frac=float(m.sum() / len(df)),
        dist_to_train_centroid_km=dist_km,
        temp_sd=float(yt.std(ddof=1)),
        temp_mean_offset=float(yt.mean() - y_all.mean()),
        geochem_mahalanobis=maha,
    ))
ps = pd.DataFrame(rows).sort_values("rmse")
ps.to_csv(OUT / "m5_per_state_drivers.csv", index=False)
print(ps.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

drivers = ["temp_sd", "temp_mean_offset", "dist_to_train_centroid_km",
           "geochem_mahalanobis", "n", "train_support_frac"]
cor_rows = []
for d in drivers:
    rho, p = spearmanr(ps[d], ps["rmse"])
    cor_rows.append(dict(driver=d, spearman_rho=float(rho), p_value=float(p), n_states=len(ps)))
cors = pd.DataFrame(cor_rows).sort_values("spearman_rho", key=abs, ascending=False)
cors.to_csv(OUT / "m5_driver_correlations.csv", index=False)
print("\nSpearman correlation of each candidate driver with per-state spatial RMSE:")
print(cors.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

if len(ps) >= 5:
    Xd = sm.add_constant(ps[["temp_sd", "dist_to_train_centroid_km"]])
    ols = sm.OLS(ps["rmse"], Xd).fit()
    with open(OUT / "m5_ols_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(str(ols.summary()))
    print(f"\nOLS rmse ~ temp_sd + dist_to_train_centroid: R2={ols.rsquared:.3f}, "
          f"n={int(ols.nobs)}")
    print("  coefficients:", {k: round(v, 4) for k, v in ols.params.items()})
    print("  p-values    :", {k: round(v, 4) for k, v in ols.pvalues.items()})

# =============================================================== M6
print("\n=== M6: spatial-block CV (geology-agnostic blocks vs administrative states) ===")
MODELS = {
    "catboost": lambda: CatBoostRegressor(iterations=700, learning_rate=0.04, depth=6,
                                          random_seed=42, verbose=0),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=450, learning_rate=0.05,
                                                           random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
}

rows = []
coords = df[["Latitude", "Longitude"]].values
for K in [5, 10]:
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(coords)
    blocks = pd.Series(km.labels_, index=df.index)
    sizes = blocks.value_counts()
    for b in sizes[sizes >= MIN_N].index:
        te_i = blocks.index[blocks == b]
        tr_i = blocks.index[blocks != b]
        Xtr, Xte = prepare(X_raw.loc[tr_i], X_raw.loc[te_i])
        ytr, yte = y_all.loc[tr_i], y_all.loc[te_i]
        for mname, factory in MODELS.items():
            m = factory().fit(Xtr, ytr)
            p = m.predict(Xte)
            rows.append(dict(protocol=f"spatial_block_k{K}", model=mname, block=int(b),
                             n_test=len(yte),
                             rmse=float(np.sqrt(mean_squared_error(yte, p))),
                             r2=float(r2_score(yte, p))))
    print(f"  k={K}: {int((sizes >= MIN_N).sum())} eligible blocks of {K}", flush=True)

# reference: administrative-state holdout and random split, same models
counts = groups.value_counts()
for st in counts[counts >= MIN_N].index:
    te_i = groups.index[groups == st]
    tr_i = groups.index[groups != st]
    Xtr, Xte = prepare(X_raw.loc[tr_i], X_raw.loc[te_i])
    ytr, yte = y_all.loc[tr_i], y_all.loc[te_i]
    for mname, factory in MODELS.items():
        p = factory().fit(Xtr, ytr).predict(Xte)
        rows.append(dict(protocol="spatial_holdout_state", model=mname, block=st,
                         n_test=len(yte), rmse=float(np.sqrt(mean_squared_error(yte, p))),
                         r2=float(r2_score(yte, p))))
for seed in SEEDS:
    tr_r, te_r, ytr, yte = train_test_split(X_raw, y_all, test_size=0.2, random_state=seed)
    Xtr, Xte = prepare(tr_r, te_r)
    for mname, factory in MODELS.items():
        p = factory().fit(Xtr, ytr).predict(Xte)
        rows.append(dict(protocol="random_split", model=mname, block=f"seed_{seed}",
                         n_test=len(yte), rmse=float(np.sqrt(mean_squared_error(yte, p))),
                         r2=float(r2_score(yte, p))))

m6 = pd.DataFrame(rows)
m6.to_csv(OUT / "m6_spatial_block_raw.csv", index=False)
piv = m6.pivot_table(index="model", columns="protocol", values="rmse", aggfunc="mean")
piv.to_csv(OUT / "m6_spatial_block_summary.csv")
print("\nmean RMSE (C) by protocol:")
print(piv.to_string(float_format=lambda v: f"{v:8.2f}"))
# NOTE: filter(like="spatial") ALSO matches spatial_holdout_state, which averages the
# block variants together with the state holdout and understates the block gap. That bug
# produced the incorrect 11.88/11.99/18.25 figures reported on 2026-07-26. Select the
# block columns explicitly and report the two gaps separately.
block_cols = [c for c in piv.columns if c.startswith("spatial_block")]
block_gap = piv[block_cols].mean(axis=1) - piv["random_split"]
state_gap = piv["spatial_holdout_state"] - piv["random_split"]
print("\nrandom -> BLOCK gap (C), averaged over block resolutions only:")
print(block_gap.to_string(float_format=lambda v: f"{v:8.2f}"))
print("\nrandom -> STATE holdout gap (C), for comparison:")
print(state_gap.to_string(float_format=lambda v: f"{v:8.2f}"))
print(f"\nwrote outputs to {OUT}")
