"""
Exact diagonalisation routines for the single-impurity Anderson model (SIAM).

Notation
--------
d  : impurity orbital (index 0,1 = ↑,↓)
c_p: bath orbital p   (index 2+2p, 3+2p = ↑,↓)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
from typing import Tuple

# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------


def _jw_sign(state: int, i: int, j: int) -> int:
    """Jordan–Wigner sign picked up when exchanging sites i (< j) and j."""
    mask = ((1 << j) - 1) ^ ((1 << (i + 1)) - 1)
    return -1 if bin(state & mask).count("1") & 1 else 1


def _add_hopping(H: np.ndarray, state: int, src: int, dst: int, t: float) -> None:
    """Add hopping term  t c†_dst c_src  (Fermionic sign込み)."""
    if (state >> src) & 1 and not (state >> dst) & 1:
        sign = _jw_sign(state, dst, src)
        new = state ^ (1 << src) ^ (1 << dst)
        H[state, new] += sign * t
        H[new, state] += sign * t  # Hermitian


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_hamiltonian(
    U: float, eps_d: float, eps_p: np.ndarray, V_p: np.ndarray
) -> np.ndarray:
    """
    Construct the full many-body Hamiltonian in the Fock basis.

    Parameters
    ----------
    U, eps_d : float
        On-site Coulomb & energy of the impurity orbital.
    eps_p, V_p : (Nb,) array
        Bath level energies and hybridisations.

    Returns
    -------
    H : (2**L, 2**L) ndarray
        Dense Hamiltonian matrix, where L = 2(Nb+1).
    """
    Nb = len(eps_p)
    L = 2 * (Nb + 1)
    dim = 1 << L
    H = np.zeros((dim, dim), dtype=np.float64)

    for s in range(dim):
        nd_up = (s >> 0) & 1
        nd_dn = (s >> 1) & 1
        H[s, s] += eps_d * (nd_up + nd_dn) + U * nd_up * nd_dn

        # bath on-site terms
        for p, ep in enumerate(eps_p):
            iu, idn = 2 + 2 * p, 3 + 2 * p
            H[s, s] += ep * (((s >> iu) & 1) + ((s >> idn) & 1))

        # hopping d <-> bath
        for p, V in enumerate(V_p):
            for d_idx, c_idx in [(0, 2 + 2 * p), (1, 3 + 2 * p)]:
                _add_hopping(H, s, d_idx, c_idx, V)

    return H


def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dense exact diagonalisation (wrapper around `scipy.linalg.eigh`)."""
    return la.eigh(H)


def _d_up_matrix(dim: int) -> np.ndarray:
    """Create d†_↑ in the computational basis."""
    op = np.zeros((dim, dim))
    for s in range(dim):
        if not (s & 1):
            op[s | 1, s] = 1  # sign = +1 for orbital 0
    return op


def green_matsubara(
    E: np.ndarray,
    Umat: np.ndarray,
    beta: float,
    nw: int,
) -> np.ndarray:
    """
    Matsubara Green function G(iω_n).

    Returns
    -------
    Giw : (nw,) complex ndarray
    """
    dim = len(E)
    d_up = _d_up_matrix(dim)
    M = Umat.T @ d_up @ Umat
    Vmn2 = np.abs(M) ** 2

    bol = np.exp(-beta * E)
    Z = bol.sum()
    Em, En = np.meshgrid(E, E, indexing="ij")
    pref = (bol[:, None] + bol[None, :]) / Z
    iwn = 1j * np.pi * (2 * np.arange(nw) + 1) / beta
    return np.array([(pref * Vmn2 / (w + Em - En)).sum() for w in iwn])


def green_realaxis(
    E: np.ndarray,
    Umat: np.ndarray,
    omega: np.ndarray,
    eta: float = 1e-2,
    beta: float | None = None,
) -> np.ndarray:
    """
    Zero-temperature (beta=None) or finite-T real-axis Green function.
    """
    dim = len(E)
    d_up = _d_up_matrix(dim)
    M = Umat.T @ d_up @ Umat
    Vmn2 = np.abs(M) ** 2

    if beta is None:
        # T = 0 → ground state only
        gs = 0
        pref = 2 * np.abs(M[gs]) ** 2  #  ⟨GS|d|n⟩⟨n|d†|GS⟩ + h.c.
    else:
        bol = np.exp(-beta * E)
        Z = bol.sum()
        Em, En = np.meshgrid(E, E, indexing="ij")
        pref = (bol[:, None] + bol[None, :]) / Z

    Em, En = np.meshgrid(E, E, indexing="ij")
    G = np.empty_like(omega, dtype=np.complex128)
    for i, w in enumerate(omega):
        z = w + 1j * eta
        G[i] = (pref * Vmn2 / (z + Em - En)).sum()
    return G


def spectral_function(G: np.ndarray) -> np.ndarray:
    """A(ω) = −(1/π) Im G(ω)."""
    return -G.imag / np.pi
