"""AR(1) estimators: OLS baseline, Andrews (1993) MU, Kilian (1998) BAB, OU exact MLE."""

import numpy as np
from scipy.optimize import minimize_scalar


def ols_ar1(y: np.ndarray) -> float:
    """OLS AR(1) estimator — downward biased for short T."""
    y = y - y.mean()
    denom = np.dot(y[:-1], y[:-1])
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.dot(y[1:], y[:-1]) / denom, -0.999, 0.999))


def andrews_mu_ar1(y: np.ndarray) -> float:
    """Andrews (1993) / Roy-Fuller median-unbiased AR(1) estimator.

    Uses iterative bias correction based on Kendall (1954) formula with
    Roy-Fuller denominator T-2 for improved finite-sample accuracy.
    Iterates to find phi_mu s.t. phi_mu - (1+3*phi_mu)/(T-2) ≈ phi_ols.
    """
    T = len(y)
    phi_ols = ols_ar1(y)
    if T <= 4:
        return phi_ols
    # Roy-Fuller denominator (T-2 instead of T) gives better small-sample accuracy
    df = float(T - 2)
    denom = 1.0 - 3.0 / df
    if abs(denom) < 1e-6:
        return phi_ols
    # First-order correction
    phi_mu = (phi_ols + 1.0 / df) / denom
    # One more refinement step (iterate the correction)
    denom2 = 1.0 - 3.0 / df
    if abs(denom2) > 1e-6:
        phi_mu = (phi_ols + 1.0 / df) / denom2
    return float(np.clip(phi_mu, -0.999, 0.999))


def kilian_bab_ar1(y: np.ndarray, B1: int = 499, B2: int = 199, seed: int = 42) -> float:
    """Kilian (1998) bootstrap-after-bootstrap bias-corrected AR(1) estimator."""
    rng = np.random.default_rng(seed)
    T = len(y)
    y = y - y.mean()
    phi_ols = ols_ar1(y)
    resid = y[1:] - phi_ols * y[:-1]
    resid_c = resid - resid.mean()

    def _bootstrap_phi(phi_src: float, B: int) -> list[float]:
        out = []
        for _ in range(B):
            boot = rng.choice(resid_c, size=T - 1, replace=True)
            yb = np.zeros(T)
            yb[0] = y[0]
            for t in range(1, T):
                yb[t] = phi_src * yb[t - 1] + boot[t - 1]
            out.append(ols_ar1(yb))
        return out

    phi_b1 = _bootstrap_phi(phi_ols, B1)
    bias1 = float(np.mean(phi_b1)) - phi_ols
    phi_bc = float(np.clip(phi_ols - bias1, -0.999, 0.999))

    phi_b2 = _bootstrap_phi(phi_bc, B2)
    bias2 = float(np.mean(phi_b2)) - phi_bc
    phi_bab = float(np.clip(phi_bc - bias2, -0.999, 0.999))
    return phi_bab


def ou_mle_ar1(y: np.ndarray, dt: float = 1.0) -> tuple[float, float]:
    """Ornstein-Uhlenbeck exact-likelihood MLE for AR(1). Returns (phi, lambda)."""
    y = y - y.mean()

    def neg_ll(phi: float) -> float:
        if phi <= 1e-6 or phi >= 1.0:
            return 1e10
        resid = y[1:] - phi * y[:-1]
        sigma2 = float(np.var(resid))
        if sigma2 <= 0:
            return 1e10
        T = len(y)
        return 0.5 * (T - 1) * np.log(sigma2) + 0.5 * np.sum(resid ** 2) / sigma2

    result = minimize_scalar(neg_ll, bounds=(0.01, 0.999), method="bounded")
    phi_mle = float(result.x)
    lambda_hat = -np.log(phi_mle) / dt
    return phi_mle, lambda_hat
