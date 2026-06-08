"""Step 0: Monte Carlo validation of AR(1) estimators."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from loguru import logger

from estimators import ols_ar1, andrews_mu_ar1, kilian_bab_ar1, ou_mle_ar1

PHI_GRID = [0.70, 0.80, 0.90, 0.95, 0.99]
T_GRID = [15, 20, 25]
N_SIM = 2000
SEED = 42


def _run_cell(args: tuple) -> dict:
    phi_true, T, n_sim, seed = args
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(max(1 - phi_true ** 2, 1e-6))
    phi_ols_list, phi_mu_list, phi_mle_list = [], [], []
    # Skip Kilian BAB in full N_SIM for speed; use Andrews MU
    for _ in range(n_sim):
        eps = rng.normal(0, sigma, T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = phi_true * y[t - 1] + eps[t]
        y = y - y.mean()
        phi_ols_list.append(ols_ar1(y))
        phi_mu_list.append(andrews_mu_ar1(y))
        phi_mle_list.append(ou_mle_ar1(y)[0])

    results = []
    for name, lst in [("OLS", phi_ols_list), ("Andrews_MU", phi_mu_list), ("OU_MLE", phi_mle_list)]:
        arr = np.array(lst)
        bias = float(np.mean(arr) - phi_true)
        rmse = float(np.sqrt(np.mean((arr - phi_true) ** 2)))
        ci_width = float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5))
        results.append({
            "phi_true": phi_true,
            "T": T,
            "estimator": name,
            "bias": round(bias, 5),
            "rmse": round(rmse, 5),
            "ci_width_95": round(ci_width, 5),
            "mean_est": round(float(np.mean(arr)), 5),
        })

    # Kilian BAB with reduced reps (expensive)
    rng2 = np.random.default_rng(seed + 1)
    phi_kil_list = []
    n_kil = min(n_sim, 200)
    sigma2 = np.sqrt(max(1 - phi_true ** 2, 1e-6))
    for _ in range(n_kil):
        eps = rng2.normal(0, sigma2, T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = phi_true * y[t - 1] + eps[t]
        y = y - y.mean()
        phi_kil_list.append(kilian_bab_ar1(y, B1=99, B2=49, seed=int(rng2.integers(0, 9999))))
    arr_k = np.array(phi_kil_list)
    bias_k = float(np.mean(arr_k) - phi_true)
    rmse_k = float(np.sqrt(np.mean((arr_k - phi_true) ** 2)))
    ci_k = float(np.percentile(arr_k, 97.5) - np.percentile(arr_k, 2.5))
    results.append({
        "phi_true": phi_true,
        "T": T,
        "estimator": "Kilian_BAB",
        "bias": round(bias_k, 5),
        "rmse": round(rmse_k, 5),
        "ci_width_95": round(ci_k, 5),
        "mean_est": round(float(np.mean(arr_k)), 5),
        "n_sim_kilian": n_kil,
    })

    return {"phi_true": phi_true, "T": T, "results": results}


def run_step0(n_sim: int = N_SIM, num_workers: int = 4) -> dict:
    logger.info(f"Step 0: Monte Carlo with N_SIM={n_sim}, workers={num_workers}")
    cells = [(phi, T, n_sim, SEED + i * 100) for i, (phi, T) in enumerate(
        [(phi, T) for phi in PHI_GRID for T in T_GRID]
    )]

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_run_cell, c): c for c in cells}
        for fut in as_completed(futures):
            try:
                cell_out = fut.result()
                all_results.extend(cell_out["results"])
                logger.info(f"  MC cell done: phi={cell_out['phi_true']}, T={cell_out['T']}")
            except Exception as e:
                logger.error(f"MC cell failed: {e}")

    # Select recommended estimator: lowest |bias| at phi in {0.90,0.95,0.99}, T=20
    key_phis = {0.90, 0.95, 0.99}
    est_bias: dict[str, list[float]] = {}
    for r in all_results:
        if r["phi_true"] in key_phis and r["T"] == 20:
            est_bias.setdefault(r["estimator"], []).append(abs(r["bias"]))
    est_max_bias = {e: max(v) for e, v in est_bias.items()}
    recommended = min(est_max_bias, key=lambda e: est_max_bias[e]) if est_max_bias else "Andrews_MU"

    bias_ok = est_max_bias.get(recommended, 1.0) < 0.05

    # Check unit-root boundary
    unit_root_bias = {
        e: next((abs(r["bias"]) for r in all_results if r["estimator"] == e
                 and r["phi_true"] == 0.99 and r["T"] == 20), None)
        for e in est_max_bias
    }
    _ols_bias = next((r['bias'] for r in all_results if r['estimator'] == 'OLS' and r['phi_true'] == 0.90 and r['T'] == 20), None)
    _rec_bias = est_max_bias.get(recommended)
    notes_parts = [
        f"OLS bias at phi=0.90,T=20: {_ols_bias:.3f}" if _ols_bias is not None else "OLS bias at phi=0.90,T=20: N/A",
        f"Recommended estimator: {recommended} (max|bias|={_rec_bias:.3f})" if _rec_bias is not None else f"Recommended estimator: {recommended}",
    ]
    if unit_root_bias.get("Andrews_MU") is not None:
        notes_parts.append(
            f"Andrews MU |bias| at phi=0.99,T=20: {unit_root_bias['Andrews_MU']:.3f} "
            f"(exact theory valid through phi=1)"
        )

    output = {
        "results": all_results,
        "recommended_estimator": recommended,
        "bias_ok": bias_ok,
        "estimator_max_bias_at_key_phis": {e: round(v, 4) for e, v in est_max_bias.items()},
        "notes": "; ".join(notes_parts),
    }
    logger.info(f"Step 0 complete. Recommended: {recommended}, bias_ok={bias_ok}")
    return output
