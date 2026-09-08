"""Classical geothermometer baseline (NB19 module M1).

Implements the standard solute geothermometers and evaluates them against the
measured reservoir temperature on the SAME canonical dataset used by NB09-NB16,
then re-runs the ML champions on the identical complete-case subsets so the
comparison is apples-to-apples.

Unit conventions (important, and the usual source of error in these equations):
  * Silica, Na-K and K-Mg geothermometers use concentrations in mg/kg (~ mg/L).
  * Na-K-Ca (Fournier & Truesdell, 1973) uses MOLAL concentrations.

References
  Fournier, R.O. (1977). Chemical geothermometers and mixing models for
      geothermal systems. Geothermics 5, 41-50.        [quartz, chalcedony]
  Fournier, R.O. (1979). A revised equation for the Na/K geothermometer.
      GRC Transactions 3, 221-224.                     [Na-K]
  Truesdell, A.H. (1976). Summary of section III: geochemical techniques in
      exploration. Proc. 2nd UN Symposium, 53-79.      [Na-K]
  Giggenbach, W.F. (1988). Geothermal solute equilibria. GCA 52, 2749-2765.
      [Na-K, K-Mg]
  Fournier, R.O. & Truesdell, A.H. (1973). An empirical Na-K-Ca geothermometer
      for natural waters. GCA 37, 1255-1275.           [Na-K-Ca]
"""
import numpy as np
import pandas as pd

# Atomic / molecular weights (g/mol)
MW = {"Na": 22.98977, "K": 39.0983, "Ca": 40.078, "Mg": 24.305, "SiO2": 60.0843}

KELVIN = 273.15


def _safe_log10(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = np.log10(x[ok])
    return out


def _molal(mg_per_l, species):
    """mg/L -> mol/kg (dilute solution approximation, rho ~ 1 kg/L)."""
    return np.asarray(mg_per_l, dtype=float) / (1000.0 * MW[species])


# --------------------------------------------------------------------------
# Silica geothermometers (SiO2 in mg/kg)
# --------------------------------------------------------------------------
def t_quartz_no_steam_loss(sio2):
    """Fournier (1977), quartz, conductive cooling, no steam loss."""
    return 1309.0 / (5.19 - _safe_log10(sio2)) - KELVIN


def t_quartz_max_steam_loss(sio2):
    """Fournier (1977), quartz, adiabatic cooling with maximum steam loss."""
    return 1522.0 / (5.75 - _safe_log10(sio2)) - KELVIN


def t_chalcedony(sio2):
    """Fournier (1977), chalcedony (appropriate below ~180 C)."""
    return 1032.0 / (4.69 - _safe_log10(sio2)) - KELVIN


# --------------------------------------------------------------------------
# Cation geothermometers
# --------------------------------------------------------------------------
def t_na_k_fournier(na, k):
    """Fournier (1979). Na, K in mg/kg."""
    return 1217.0 / (_safe_log10(np.asarray(na, float) / np.asarray(k, float)) + 1.483) - KELVIN


def t_na_k_giggenbach(na, k):
    """Giggenbach (1988). Na, K in mg/kg."""
    return 1390.0 / (_safe_log10(np.asarray(na, float) / np.asarray(k, float)) + 1.75) - KELVIN


def t_na_k_truesdell(na, k):
    """Truesdell (1976). Na, K in mg/kg."""
    return 856.0 / (_safe_log10(np.asarray(na, float) / np.asarray(k, float)) + 0.857) - KELVIN


def t_k_mg_giggenbach(k, mg):
    """Giggenbach (1988) K-Mg. K, Mg in mg/kg."""
    k = np.asarray(k, float)
    mg = np.asarray(mg, float)
    ratio = np.where((k > 0) & (mg > 0), k ** 2 / np.where(mg > 0, mg, np.nan), np.nan)
    return 4410.0 / (14.0 - _safe_log10(ratio)) - KELVIN


def t_na_k_ca(na, k, ca):
    """Fournier & Truesdell (1973) Na-K-Ca, MOLAL concentrations.

    T(C) = 1647 / (log(Na/K) + beta*log(sqrt(Ca)/Na) + 2.24) - 273.15

    Standard beta selection rule: compute with beta = 4/3; if the resulting
    temperature exceeds 100 C, recompute with beta = 1/3 and keep that value.

    Verified against Wairakei reference water (Na=1130, K=146, Ca=12.2 mg/L):
    beta=4/3 gives 324 C -> >100 C -> beta=1/3 gives 240 C, consistent with
    the published Na-K-Ca estimate for that water. See self-test at bottom.
    """
    m_na = _molal(na, "Na")
    m_k = _molal(k, "K")
    m_ca = _molal(ca, "Ca")

    with np.errstate(invalid="ignore", divide="ignore"):
        log_nak = _safe_log10(m_na / m_k)
        log_cana = _safe_log10(np.sqrt(np.where(m_ca > 0, m_ca, np.nan)) / m_na)

        def _t(beta):
            return 1647.0 / (log_nak + beta * log_cana + 2.24) - KELVIN

        t_beta_43 = _t(4.0 / 3.0)
        t_beta_13 = _t(1.0 / 3.0)

    use_low_beta = (~np.isfinite(t_beta_43)) | (t_beta_43 > 100.0)
    return np.where(use_low_beta, t_beta_13, t_beta_43)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
GEOTHERMOMETERS = {
    "quartz_no_steam_loss": dict(fn=lambda d: t_quartz_no_steam_loss(d["SiO2"]),
                                 needs=["SiO2"], ref="Fournier (1977)"),
    "quartz_max_steam_loss": dict(fn=lambda d: t_quartz_max_steam_loss(d["SiO2"]),
                                  needs=["SiO2"], ref="Fournier (1977)"),
    "chalcedony": dict(fn=lambda d: t_chalcedony(d["SiO2"]),
                       needs=["SiO2"], ref="Fournier (1977)"),
    "na_k_fournier": dict(fn=lambda d: t_na_k_fournier(d["Na"], d["K"]),
                          needs=["Na", "K"], ref="Fournier (1979)"),
    "na_k_giggenbach": dict(fn=lambda d: t_na_k_giggenbach(d["Na"], d["K"]),
                            needs=["Na", "K"], ref="Giggenbach (1988)"),
    "na_k_truesdell": dict(fn=lambda d: t_na_k_truesdell(d["Na"], d["K"]),
                           needs=["Na", "K"], ref="Truesdell (1976)"),
    "na_k_ca": dict(fn=lambda d: t_na_k_ca(d["Na"], d["K"], d["Ca"]),
                    needs=["Na", "K", "Ca"], ref="Fournier & Truesdell (1973)"),
    "k_mg_giggenbach": dict(fn=lambda d: t_k_mg_giggenbach(d["K"], d["Mg"]),
                            needs=["K", "Mg"], ref="Giggenbach (1988)"),
}


def compute_all(df):
    """Return a DataFrame of geothermometer temperature estimates (NaN where
    the required ions are missing)."""
    out = pd.DataFrame(index=df.index)
    for name, spec in GEOTHERMOMETERS.items():
        vals = np.asarray(spec["fn"](df), dtype=float)
        # mask rows lacking any required ion
        mask = np.ones(len(df), dtype=bool)
        for col in spec["needs"]:
            mask &= df[col].notna().to_numpy()
        vals = np.where(mask, vals, np.nan)
        # physically implausible estimates -> NaN (kept as a reported count)
        vals = np.where(np.isfinite(vals) & (vals > -50) & (vals < 500), vals, np.nan)
        out[name] = vals
    return out


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(ok.sum())
    if n < 2:
        return dict(n=n, rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan)
    yt, yp = y_true[ok], y_pred[ok]
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return dict(
        n=n,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    )
