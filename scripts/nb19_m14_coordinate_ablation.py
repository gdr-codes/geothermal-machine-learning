"""M14 - Is the protocol gap a coordinate-extrapolation artefact?

Latitude and Longitude are model features. Tree ensembles cannot extrapolate: a test point
outside the training range of a feature is served by the nearest training leaf. Holding out a
whole state puts every test row outside the training range of BOTH coordinates, so part of the
published protocol gap may be a property of the feature set rather than of the geology.

This reproduces NB09's pipeline exactly (same models, same hyperparameters, same protocols,
same eligibility rule) and then repeats it with the two coordinate columns removed.

Produces every number quoted in Paper 1 Section 4.1's coordinate-ablation paragraph.

COVERAGE_NOTE
  - Hyperparameters are held at NB09's values, which were selected WITH coordinates present.
    A coordinate-free retuning could shift the magnitudes. This bounds, not resolves.
  - The residual no-coordinate gap still contains duplicate-well leakage (M13): 2,684 samples
    sit at 885 coordinates, so removing lat/long does not remove well-level repetition.
  - Four models, not seven: the two gradient-boosting variants and the two forest variants
    cover the inductive biases; adding the rest changes runtime, not the conclusion.
  - Spatial holdout is a single deterministic run per state, as in NB09. No interval is
    claimed on any ratio here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '03_data_processed' / 'geothermal_canonical_v2_0_0.csv'
OUT = Path(__file__).parent / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

TARGET, GROUP = 'Temperature(C)', 'State'
COORDS = ['Latitude', 'Longitude']
MIN_STATE_N = 30          # NB09's eligibility rule
TEST_SIZE = 0.2
SEEDS = range(10)


def models():
    """NB09's model_factory(), restricted to four families."""
    return {
        'catboost': CatBoostRegressor(iterations=700, learning_rate=0.04, depth=6,
                                      random_seed=42, verbose=0),
        'xgboost': XGBRegressor(n_estimators=550, learning_rate=0.04, max_depth=6,
                                subsample=0.9, colsample_bytree=0.9,
                                random_state=42, n_jobs=-1),
        'extra_trees': ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    }


def prep(train_df, test_df, cols):
    """NB09's prepare_design_matrix: median imputation fit on TRAIN only."""
    imp = SimpleImputer(strategy='median')
    tr = pd.DataFrame(imp.fit_transform(train_df[cols]), columns=cols, index=train_df.index)
    te = pd.DataFrame(imp.transform(test_df[cols]), columns=cols, index=test_df.index)
    return tr, te


def aggregate(frame, protocol):
    """Combine repeated runs into one number - and the right way depends on the protocol.

    spatial_holdout: the six held-out states PARTITION the held-out data, so the runs are
    disjoint pieces of one evaluation. RMSE combines in quadrature, weighted by n.

    random_split: the ten seeds are OVERLAPPING resamples of the same rows, not a partition.
    Quadrature pooling would treat them as disjoint and inflate the result (Jensen), so the
    arithmetic mean across seeds is correct. This is also what NB09 reports, which is why the
    published CatBoost random-split value is 20.00 rather than 20.03.
    """
    if protocol == 'spatial_holdout':
        return float(np.sqrt((frame.n * frame.rmse ** 2).sum() / frame.n.sum()))
    return float(frame.rmse.mean())


def run(df, y, groups, cols, label):
    rows = []
    for seed in SEEDS:
        tr, te, ytr, yte = train_test_split(df, y, test_size=TEST_SIZE, random_state=seed)
        A, B = prep(tr, te, cols)
        for name, mdl in models().items():
            mdl.fit(A, ytr)
            rows.append(dict(featset=label, protocol='random_split', model=name,
                             grp=f'seed_{seed}', n=len(B),
                             rmse=float(np.sqrt(mean_squared_error(yte, mdl.predict(B))))))

    eligible = groups.value_counts()[lambda s: s >= MIN_STATE_N].index.tolist()
    for g in eligible:
        tr, te = df[groups != g], df[groups == g]
        A, B = prep(tr, te, cols)
        for name, mdl in models().items():
            mdl.fit(A, y[tr.index])
            rows.append(dict(featset=label, protocol='spatial_holdout', model=name,
                             grp=str(g), n=len(B),
                             rmse=float(np.sqrt(mean_squared_error(y[te.index], mdl.predict(B))))))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA)
    if len(df) != 2684:
        sys.exit(f'FAIL: expected the pinned 2,684-row canonical dataset, got {len(df)}')
    y, groups = df[TARGET], df[GROUP]

    allcols = [c for c in df.columns if c not in (TARGET, GROUP)]
    nocoord = [c for c in allcols if c not in COORDS]

    res = pd.concat([run(df, y, groups, allcols, 'with_coords'),
                     run(df, y, groups, nocoord, 'no_coords')], ignore_index=True)
    res.to_csv(OUT / 'm14_coordinate_ablation_runs.csv', index=False)

    summary = []
    for fs in ('with_coords', 'no_coords'):
        cell = {}
        for proto in ('random_split', 'spatial_holdout'):
            for mdl in sorted(res.model.unique()):
                cell[(mdl, proto)] = aggregate(
                    res[(res.featset == fs) & (res.model == mdl) & (res.protocol == proto)],
                    proto)
        mdls = sorted(res.model.unique())
        spread_r = max(cell[(m, 'random_split')] for m in mdls) - \
            min(cell[(m, 'random_split')] for m in mdls)
        spread_s = max(cell[(m, 'spatial_holdout')] for m in mdls) - \
            min(cell[(m, 'spatial_holdout')] for m in mdls)
        shifts = [cell[(m, 'spatial_holdout')] - cell[(m, 'random_split')] for m in mdls]
        ratios = [cell[(m, 'spatial_holdout')] / cell[(m, 'random_split')] for m in mdls]
        summary.append(dict(
            featset=fs,
            model_spread_random=round(spread_r, 2),
            model_spread_spatial=round(spread_s, 2),
            protocol_shift_mean=round(float(np.mean(shifts)), 2),
            gap_ratio_min=round(min(ratios), 2), gap_ratio_max=round(max(ratios), 2),
            protocol_over_model=round(float(np.mean(shifts)) / max(spread_r, spread_s), 1),
            catboost_random=round(cell[('catboost', 'random_split')], 2),
            catboost_spatial=round(cell[('catboost', 'spatial_holdout')], 2)))
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / 'm14_coordinate_ablation_summary.csv', index=False)

    print(sm.to_string(index=False))
    print()
    w, n = sm[sm.featset == 'with_coords'].iloc[0], sm[sm.featset == 'no_coords'].iloc[0]
    claims = {
        'P1 4.1 catboost random with coords = 20.00': w.catboost_random == 20.00,
        'P1 4.1 catboost random no coords  = 23.60': n.catboost_random == 23.60,
        'P1 4.1 catboost spatial with coords= 32.59': w.catboost_spatial == 32.59,
        'P1 4.1 catboost spatial no coords = 31.48': n.catboost_spatial == 31.48,
        'P1 4.1 gap with coords  1.60-2.00x': (w.gap_ratio_min, w.gap_ratio_max) == (1.60, 2.00),
        'P1 4.1 gap no coords    1.33-1.39x': (n.gap_ratio_min, n.gap_ratio_max) == (1.33, 1.39),
        'P1 4.1 protocol shift no coords = 8.53': n.protocol_shift_mean == 8.53,
        'P1 4.1 model spread spatial with = 4.90': w.model_spread_spatial == 4.90,
        'P1 4.1 model spread spatial no   = 1.38': n.model_spread_spatial == 1.38,
        'P1 4.1 protocol/model with coords = 2.9x': w.protocol_over_model == 2.9,
        'P1 4.1 protocol/model no coords   = 6.2x': n.protocol_over_model == 6.2,
    }
    bad = [k for k, v in claims.items() if not v]
    for k, v in claims.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print(f'\n{len(claims) - len(bad)} PASS   {len(bad)} FAIL')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
