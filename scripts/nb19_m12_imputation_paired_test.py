r"""nb19 M12 — is the imputation penalty real, or six noisy states?

M11 found that XGBoost scores 28.957 degC under spatial holdout when NaN is left for the model to
route, against 31.394 degC when the published pipeline median-imputes first — a 2.44 degC penalty,
and the difference between beating the trivial baseline and not. That was six state means with no
test attached, which is exactly the kind of claim this project has been burned by before.

Two tests, on identical fits:
  per state   6 paired states. The smallest attainable two-sided p at n=6 is 0.03125, so a null
              result here is uninformative rather than negative.
  per sample  2,644 paired held-out samples, the same procedure used against the geothermometers
              and against the median dummy in M9. This is the one that can actually resolve it.

Only preprocessing differs between the two arms: same model, same hyperparameters, same folds,
same seed. Validated against M11's stored means before any test is reported.

Output: 06_results/nb19/m12_imputation_paired.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

ROS = Path(__file__).resolve().parents[1]
DATA = ROS / "03_data_processed/geothermal_canonical_v2_0_0.csv"
M11 = ROS / "06_results/nb19/m11_spatial_baselines.csv"
OUT = ROS / "06_results/nb19/m12_imputation_paired.csv"

TARGET, GROUP = "Temperature(C)", "State"
TOL = 0.01


def xgb():
    return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, subsample=0.85,
                        colsample_bytree=0.85, random_state=42, n_jobs=24)


def main():
    df = pd.read_csv(DATA)
    y, groups = df[TARGET], df[GROUP]
    counts = groups.value_counts()
    states = sorted(counts[counts >= 30].index.tolist())
    feats = [c for c in df.columns if c not in (TARGET, GROUP)
             and pd.api.types.is_numeric_dtype(df[c])]

    rec = []
    for st in states:
        te = groups[groups == st].index
        tr = groups[groups != st].index
        Xtr, Xte = df.loc[tr, feats], df.loc[te, feats]
        ytr, yte = y.loc[tr], y.loc[te]

        imp = SimpleImputer(strategy="median").fit(Xtr)
        p_med = xgb().fit(imp.transform(Xtr), ytr).predict(imp.transform(Xte))
        p_nat = xgb().fit(Xtr, ytr).predict(Xte)

        for i, idx in enumerate(te):
            rec.append(dict(state=st, y=yte.loc[idx],
                            pred_median_impute=p_med[i], pred_native_nan=p_nat[i]))
        print(f"  {st:12s} n={len(te):5d}  median-impute "
              f"{np.sqrt(mean_squared_error(yte, p_med)):7.3f}   native-NaN "
              f"{np.sqrt(mean_squared_error(yte, p_nat)):7.3f}")

    d = pd.DataFrame(rec)
    per_state = (d.groupby("state")
                 .apply(lambda g: pd.Series({
                     "n": len(g),
                     "rmse_median": np.sqrt(np.mean((g.y - g.pred_median_impute) ** 2)),
                     "rmse_native": np.sqrt(np.mean((g.y - g.pred_native_nan) ** 2))}),
                        include_groups=False)
                 .reset_index())

    m11 = pd.read_csv(M11).groupby("method").rmse.mean()
    got_med, got_nat = per_state.rmse_median.mean(), per_state.rmse_native.mean()
    ok = (abs(got_med - m11["xgb_median_impute"]) < TOL
          and abs(got_nat - m11["xgb_native_nan"]) < TOL)
    print(f"\nVALIDATION vs M11   median-impute {got_med:.4f} / {m11['xgb_median_impute']:.4f}"
          f"   native-NaN {got_nat:.4f} / {m11['xgb_native_nan']:.4f}  "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print("does not reproduce M11 — nothing written, no test reported.")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    ws = wilcoxon(per_state.rmse_native, per_state.rmse_median)
    e_nat = (d.y - d.pred_native_nan).abs()
    e_med = (d.y - d.pred_median_impute).abs()
    wp = wilcoxon(e_nat, e_med)
    wins = int((e_nat < e_med).sum())

    print(f"\nPER STATE   n=6   floor on two-sided p = {2 / 2 ** 6:.5f}")
    print(per_state.round(3).to_string(index=False))
    print(f"   native-NaN better in {(per_state.rmse_native < per_state.rmse_median).sum()}/6"
          f"   Wilcoxon p = {ws.pvalue:.4f}")

    print(f"\nPER SAMPLE  n={len(d):,}")
    print(f"   mean |err| native-NaN      {e_nat.mean():.3f} degC")
    print(f"   mean |err| median-impute   {e_med.mean():.3f} degC")
    print(f"   native-NaN better on       {wins:,}/{len(d):,} ({wins / len(d) * 100:.1f}%)")
    print(f"   Wilcoxon p                 {wp.pvalue:.3e}")

    print("\nCOVERAGE_NOTE: one model family and one hyperparameter set; the arms differ ONLY in "
          "preprocessing. Says nothing about whether native NaN would help CatBoost, "
          "VotingRegressor, or the tuned configurations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
