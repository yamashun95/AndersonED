# fit_bethe.py
"""Discrete‑bath fitting on Matsubara axis (Bethe lattice example).

This module exposes two public helpers:

* ``fit_bethe_bath(nbath, beta, nw_fit, *, W=1.0, weight=None, v0=0.05)`` –
  returns (eps_p, V_p, chi2) for the optimal bath.
* ``bethe_green(z, W=1.0)`` – analytic local Green function on any complex
  frequency grid (retarded branch).

When run as a script (``python -m fit_bethe`` or ``python fit_bethe.py``) it
reproduces the demo shown in chat: performs a fit with default parameters and
plots the resulting Im G(ω) against the analytic curve.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "bethe_green",
    "fit_bethe_bath",
]

###############################################################################
# 1.  Analytic Bethe‐lattice Green function                                   #
###############################################################################


def bethe_green(z: np.ndarray | complex, W: float = 1.0) -> np.ndarray | complex:
    """Local Green function of the Bethe lattice (half‑bandwidth *W*).

    Parameters
    ----------
    z
        Complex frequency grid (may be scalar or ndarray).
    W
        Half‑bandwidth of the semicircular DOS (default 1.0).

    Notes
    -----
    Retarded branch (``Im sqrt(z**2 - W**2) >= 0``) is enforced so that
    ``Im G(ω+i0⁺) <= 0`` for real ω inside the band.
    """
    root = np.lib.scimath.sqrt(z * z - W * W)
    if np.isscalar(root):
        if root.imag < 0:  # ensure Im root >= 0
            root = -root
    else:
        mask = root.imag < 0
        root[mask] = -root[mask]
    return 2.0 * (z - root) / W**2


###############################################################################
# 2.  Discrete‑bath model                                                     #
###############################################################################


# helper: unpack optimisation variables → physical parameters
def _unpack_params(
    params: np.ndarray, nb: int, W: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform optimisation variables into (|V|², ε) with constraints."""
    V2 = np.exp(params[:nb])  # positive by construction
    eps = np.tanh(params[nb:]) * W  # confined to (−W, W)
    return V2, np.sort(eps)  # sort to avoid permutation degeneracy


def _G_discrete(params: np.ndarray, iwn: np.ndarray, nb: int, W: float) -> np.ndarray:
    """Green function for the given set of bath parameters (εₚ, Vₚ)."""
    V2, eps = _unpack_params(params, nb, W)
    Delta = (V2[:, None] / (iwn[None, :] - eps[:, None])).sum(axis=0)
    return 1.0 / (iwn - Delta)  # impurity level ε_d = 0


def _residual(
    params: np.ndarray,
    iwn: np.ndarray,
    G_target: np.ndarray,
    nb: int,
    W: float,
    weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return concatenated (Re, Im) residual vector."""
    G_model = _G_discrete(params, iwn, nb, W)
    diff = G_model - G_target
    if weight is not None:
        diff *= weight
    return np.concatenate((diff.real, diff.imag))


###############################################################################
# 3.  Public fitting routine                                                  #
###############################################################################


def fit_bethe_bath(
    nbath: int,
    beta: float,
    nw_fit: int,
    *,
    W: float = 1.0,
    weight: Optional[np.ndarray] | float = None,
    v0: float = 0.05,
    max_iter: int = 20_000,
    tol: float = 1e-12,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a discrete bath (Nb poles) to Bethe‑lattice G(iωₙ).

    Returns
    -------
    eps_p : ndarray shape (Nb,)
        Fitted pole positions (ascending order).
    V_p   : ndarray shape (Nb,)
        Coupling strengths (positive).
    chi2  : float
        Final least‑squares cost (∑|Re|²+|Im|²)/2.
    """
    # Matsubara grid
    iwn = 1j * np.pi * (2 * np.arange(nw_fit) + 1) / beta
    G_target = bethe_green(iwn, W)

    # optional frequency weighting factor
    if isinstance(weight, (float, int)):
        w_arr = None if weight == 1.0 else weight * np.ones_like(iwn)
    else:
        w_arr = weight  # user‑supplied ndarray

    # initial guess — symmetric placement, uniform |V|² = v0
    rng = np.random.default_rng(seed)
    logV2_init = np.log(np.full(nbath, v0))
    eps_init = np.linspace(-0.8 * W, 0.8 * W, nbath)
    x_init = np.arctanh(eps_init / W)
    params0 = np.concatenate((logV2_init, x_init))

    res = least_squares(
        _residual,
        params0,
        args=(iwn, G_target, nbath, W, w_arr),
        method="trf",
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=max_iter,
    )

    V2_opt, eps_opt = _unpack_params(res.x, nbath, W)
    return eps_opt, np.sqrt(V2_opt), res.cost


###############################################################################
# 4.  Demo / CLI                                                              #
###############################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # demo parameters (same as chat example)
    eps_p, V_p, chi2 = fit_bethe_bath(nbath=8, beta=50.0, nw_fit=200)

    print("Fit finished — χ² = {:.3e}".format(chi2))
    print("ε_p  =", np.round(eps_p, 6))
    print("|V_p| =", np.round(V_p, 6))

    # compare on real axis
    W = 1.0
    omega = np.linspace(-1.5 * W, 1.5 * W, 800)
    delta = 1e-3
    G_true = bethe_green(omega + 1j * delta, W)
    Delta_fit = (
        V_p[:, None] ** 2 / (omega[None, :] + 1j * delta - eps_p[:, None])
    ).sum(axis=0)
    G_fit = 1.0 / (omega + 1j * delta - Delta_fit)

    fig, ax = plt.subplots()
    ax.plot(omega, G_true.imag, label="Im G (analytic)")
    ax.plot(omega, G_fit.imag, "--", label=f"Im G (fit, Nb={len(eps_p)})")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"Im $G(\omega)$")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    plt.show()
