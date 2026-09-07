# Geothermal Machine Learning — Analysis Notebooks

Jupyter notebooks accompanying a series of studies on predicting geothermal reservoir temperature
from water chemistry, and on how much the choice of validation protocol changes the answer.

**Author:** Oluwasogo Bolaji Alonge · Department of Chemical Engineering, University of North
Dakota · <bolaji.alonge@und.edu> · ORCID [0009-0001-8685-3957](https://orcid.org/0009-0001-8685-3957)

## Scope

This repository contains the **analysis notebooks only**. Result tables, intermediate outputs and
model artefacts are not published here.

| Notebook | Subject |
|---|---|
| `09_tree_family_benchmark` | Tree-family benchmark across three validation protocols |
| `10_tree_family_final_recommendation` | Hyperparameter tuning and configuration selection |
| `11_other_ensemble` | Heterogeneous ensembles: voting, stacking, bagging, boosting |
| `12_ablation_feature_families` | Feature-family ablation |
| `13_uncertainty_conformal` | Conformal prediction intervals |
| `14_facies_cluster_modeling` | Unsupervised facies discovery and cluster-conditioned models |
| `15_external_transfer` | External transfer evaluation |
| `15_shap_interpretability` | SHAP feature attribution |
| `16_validation_diagnostics` | Validation diagnostics and significance testing |
| `17_comprehensive_enhancements` | Robustness and diagnostic extensions |
| `18_submission_qa` | Submission quality assurance |

## Data

The training data is the **Argonne Geothermal Geochemical Database v2.00**, publicly available
through the U.S. DOE Geothermal Data Repository at <https://gdr.openei.org/>. It is not
redistributed here. The canonical analysis file is 2,684 samples across 11 U.S. states, after the
cleaning described in the associated manuscripts.

## Environment

Python 3.12, scikit-learn 1.8.0, XGBoost 3.2.0. Results are insensitive to the scikit-learn
version: re-running the tree-family benchmark under 1.5.1 reproduces every model and protocol RMSE
identically to eight decimal places.

## Licence

MIT (`LICENSE`). The underlying data is governed by the terms of the DOE Geothermal Data
Repository.
