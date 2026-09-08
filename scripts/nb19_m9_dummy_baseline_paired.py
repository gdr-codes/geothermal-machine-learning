r"""nb19 M9 -- is the spatial-holdout champion distinguishable from a median predictor?

WHY THIS EXISTS
Paper 1 Table 2 reports DummyRegressor(median) at 29.42 degC under spatial holdout against
CatBoost at 27.52. Table 3's Wilcoxon tests cover random_split and groupkfold only -- there
has never been a spatial-holdout significance test against the dummy. An external reviewer
(2026-08) read Table 2 and concluded the models "are not distinguishable from a median
predictor". This script settles it from the primary artefacts.

THE ANSWER DEPENDS ENTIRELY ON THE UNIT OF ANALYSIS, WHICH IS THE FINDING
  per-state, unweighted (n=6) : 4/6 states, mean skill -3.1%, Wilcoxon p = 0.5625
  sample-pooled RMSE          : dummy 40.47 vs CatBoost 32.59  (skill +19.5%)
  per-sample paired (n=2644)  : p ~ 3e-58, model wins 62.1% of samples
Equal-weighting six states lets Oregon and Idaho -- 96 of 2,644 held-out samples -- carry
one third of the average. Paper 4 already flags this convention split; Paper 1 does not.

IDENTITY, ESTABLISHED NOT ASSUMED
The dummy is deterministic (median of y_train), so it is rebuilt per sample and validated
against the six published per-state RMSEs. nb19's m5 CatBoost OOF column is validated
against NB09's published per-state CatBoost RMSEs. Both must match or the script exits
non-zero WITHOUT emitting results.

OUTPUTS (06_results/nb19/)
  m9_per_state_vs_dummy.csv   per-state RMSE, skill, win flag
  m9_paired_summary.csv       the three units of analysis, one row each

COVERAGE_NOTE
Covers spatial holdout only, CatBoost only, and the median dummy only. Per-sample testing
is limited to CatBoost because m5_oof_predictions.csv is the only stored per-sample
prediction set; every other model has aggregate per-fold RMSE and nothing finer, so a
per-sample paired test for Gradient Boosting would require a re-fit and is NOT done here.
It does NOT re-fit any model. If NB09 is ever re-run, the identity checks here will fail
loudly, which is the intended behaviour.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROS = Path(__file__).resolve().parents[1]
DATA = ROS / "03_data_processed/geothermal_canonical_v2_0_0.csv"
NB09 = (ROS / "github_org_bootstrap/phd-geothermal-ml/manual_bootstrap/step_by_step_notebooks"
            / "09_tree_family_benchmark_no_scripts/outputs/summary/tables")
OUT = ROS / "06_results/nb19"

TARGET, GROUP = "Temperature(C)", "State"
TOL = 0.005
failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {detail}")
    if not ok:
        failures.append(label)


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET].to_numpy(float)
    states = df[GROUP].to_numpy()
    counts = pd.Series(states).value_counts()
    eligible = counts[counts >= 30].index.tolist()

    print("IDENTITY CHECKS")
    pub_dummy = (pd.read_csv(NB09 / "dummy_median_all_protocols.csv")
                 .query("protocol == 'spatial_holdout'").set_index("fold")["rmse"])
    dummy = np.full(len(df), np.nan)
    for st in eligible:
        te = states == st
        dummy[te] = np.median(y[~te])
        rmse = float(np.sqrt(np.mean((y[te] - dummy[te]) ** 2)))
        check(f"dummy {st:<11}", abs(rmse - float(pub_dummy.loc[st])) < 1e-6,
              f"{rmse:.4f} vs {float(pub_dummy.loc[st]):.4f}")

    oof = pd.read_csv(ROS / "06_results/nb19/m5_oof_predictions.csv")
    model = oof["spatial_holdout"].to_numpy(float)
    pub_runs = pd.read_csv(NB09 / "all_model_protocol_runs.csv")
    pub_cat = (pub_runs.query("protocol == 'spatial_holdout' and model_name == 'catboost'")
               .set_index("holdout_group")["rmse"])
    for st in eligible:
        te = (states == st) & ~np.isnan(model)
        rmse = float(np.sqrt(np.mean((y[te] - model[te]) ** 2)))
        check(f"catboost {st:<11}", abs(rmse - float(pub_cat.loc[st])) < TOL,
              f"{rmse:.4f} vs {float(pub_cat.loc[st]):.4f}")

    if failures:
        print(f"\nIDENTITY NOT ESTABLISHED -- {len(failures)} failed. No outputs written.")
        return 1

    rows = []
    for st in eligible:
        te = states == st
        dr = float(np.sqrt(np.mean((y[te] - dummy[te]) ** 2)))
        mr = float(np.sqrt(np.mean((y[te] - model[te]) ** 2)))
        rows.append(dict(state=st, n_test=int(te.sum()), dummy_rmse=dr,
                         catboost_rmse=mr, skill_pct=(1 - mr / dr) * 100,
                         model_wins=bool(mr < dr)))
    per_state = pd.DataFrame(rows).sort_values("n_test", ascending=False)

    mask = np.isin(states, eligible) & ~np.isnan(model)
    em, ed = np.abs(y[mask] - model[mask]), np.abs(y[mask] - dummy[mask])
    n = int(mask.sum())
    w_sample = wilcoxon(em, ed)
    w_state = wilcoxon(per_state.catboost_rmse.to_numpy(), per_state.dummy_rmse.to_numpy())
    cnt = per_state.n_test.to_numpy(float)

    def pooled(r):
        return float(np.sqrt((cnt * r ** 2).sum() / cnt.sum()))

    summary = pd.DataFrame([
        dict(unit="per_state_unweighted", n=len(per_state),
             dummy=per_state.dummy_rmse.mean(), model=per_state.catboost_rmse.mean(),
             skill_pct=per_state.skill_pct.mean(),
             wins=f"{int(per_state.model_wins.sum())}/{len(per_state)}",
             p_value=float(w_state.pvalue)),
        dict(unit="sample_pooled_rmse", n=n,
             dummy=pooled(per_state.dummy_rmse.to_numpy()),
             model=pooled(per_state.catboost_rmse.to_numpy()),
             skill_pct=(1 - pooled(per_state.catboost_rmse.to_numpy())
                        / pooled(per_state.dummy_rmse.to_numpy())) * 100,
             wins=f"{int(cnt[per_state.model_wins].sum())}/{int(cnt.sum())}",
             p_value=float("nan")),
        dict(unit="per_sample_paired", n=n, dummy=float(ed.mean()), model=float(em.mean()),
             skill_pct=(1 - em.mean() / ed.mean()) * 100,
             wins=f"{int((em < ed).sum())}/{n}", p_value=float(w_sample.pvalue)),
    ])

    OUT.mkdir(parents=True, exist_ok=True)
    per_state.to_csv(OUT / "m9_per_state_vs_dummy.csv", index=False)
    summary.to_csv(OUT / "m9_paired_summary.csv", index=False)

    print("\nPER STATE")
    print(per_state.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print("\nTHREE UNITS OF ANALYSIS")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:10.4g}"))
    print(f"\nmedian |err|: catboost {np.median(em):.3f} vs dummy {np.median(ed):.3f}"
          "   <- the model's gain is in the tail, not the typical case")
    print(f"\nwritten: {OUT / 'm9_per_state_vs_dummy.csv'}")
    print(f"written: {OUT / 'm9_paired_summary.csv'}")

    # ---- manuscript binding -------------------------------------------------
    # A verifier that never opens the document is not checking the document.
    # These assert the manuscripts quote what was just recomputed, so the numbers
    # cannot drift silently once written.
    print("\nMANUSCRIPT BINDING")
    pooled_d = pooled(per_state.dummy_rmse.to_numpy())
    pooled_m = pooled(per_state.catboost_rmse.to_numpy())
    expected = {
        f"{em.mean():.2f}": "CatBoost mean |err|",
        f"{ed.mean():.2f}": "dummy mean |err|",
        f"{int((em < ed).sum()):,}": "samples where model wins",
        f"{pooled_d:.2f}": "pooled dummy RMSE",
        f"{pooled_m:.2f}": "pooled CatBoost RMSE",
    }
    p1 = (ROS.parent / "Paper1_Journal_Manuscript_REVISED_July2026.md").read_text(encoding="utf8")
    for token, what in expected.items():
        check(f"Paper 1 quotes {what} = {token}", token in p1)

    p6 = (ROS.parent / "Paper6_Journal_Manuscript_REVISED_July2026.md").read_text(encoding="utf8")
    check("Paper 6 Table 5 ranks within protocol",
          "Ranks run within protocol" in p6)
    check("Paper 6 Table 5 no longer ranks a random config above every spatial one",
          "| 5 | VotingRegressor + split conformal \\| spatial" not in p6)

    # mutation: a token that is deliberately wrong must NOT be found, otherwise the
    # check above would pass against any document at all.
    sentinel = f"{em.mean() + 11.11:.2f}"
    check("mutation - fabricated value absent from Paper 1", sentinel not in p1,
          f"(sentinel {sentinel})")

    print("\nCOVERAGE_NOTE: spatial holdout only; CatBoost vs median dummy; consumes stored "
          "OOF predictions and re-fits nothing. The manuscript binding checks that the "
          "computed values APPEAR in P1/P6; it does not check the surrounding prose.")
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED")
        return 1
    print(f"\nALL CHECKS PASSED ({len(expected) + 3} bindings + 12 identity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
