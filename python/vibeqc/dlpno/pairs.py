"""DLPNO pair classification -- classify occupied orbital pairs.

For each occupied orbital pair (i, j) we estimate the pair correlation
energy using the dipole approximation for distant pairs and a
semi-local MP2-like estimate for closer pairs.  Pairs are classified as:

* **strong** -- full PNO treatment with tight truncation threshold
* **weak**   -- PNO treatment with coarser truncation
* **distant** -- dipole-approximation energy only, no PNO construction

References
----------
* Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013), Sec.II.C
* Riplinger, Pinski, Becker, Valeev, Neese, J. Chem. Phys. 144,
  024109 (2016), Sec.II.B
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class PairClass(enum.IntEnum):
    """Pair classification label."""

    STRONG = 0
    WEAK = 1
    DISTANT = 2


@dataclass
class PairInfo:
    """Information about one occupied orbital pair (i, j).

    Attributes
    ----------
    i, j : int
        Occupied orbital indices (0-based).
    r_i, r_j : ndarray
        Orbital centroid positions (Bohr), shape (3,).
    dist : float
        Distance between centroids (Bohr).
    e_pair_est : float
        Estimated pair correlation energy (Ha).  Used to classify as
        strong/weak/distant.
    pair_class : PairClass
        Classification after comparison with thresholds.
    dipole_i, dipole_j : ndarray
        Orbital dipole moments (optional, for improved distant-pair estimate).
    """

    i: int
    j: int
    r_i: np.ndarray
    r_j: np.ndarray
    dist: float
    e_pair_est: float = 0.0
    pair_class: PairClass = PairClass.DISTANT
    dipole_i: Optional[np.ndarray] = None
    dipole_j: Optional[np.ndarray] = None


@dataclass
class PairClassification:
    """Result of classifying all occupied orbital pairs.

    Attributes
    ----------
    n_occ : int
        Number of correlated occupied orbitals.
    strong_pairs : list of PairInfo
        Strong pairs (full PNO treatment).
    weak_pairs : list of PairInfo
        Weak pairs (coarse PNO treatment).
    distant_pairs : list of PairInfo
        Distant pairs (dipole approximation only).
    all_pairs : list of PairInfo
        All pairs, sorted by estimated energy.
    """

    n_occ: int
    strong_pairs: list[PairInfo] = field(default_factory=list)
    weak_pairs: list[PairInfo] = field(default_factory=list)
    distant_pairs: list[PairInfo] = field(default_factory=list)
    all_pairs: list[PairInfo] = field(default_factory=list)

    @property
    def n_strong(self) -> int:
        return len(self.strong_pairs)

    @property
    def n_weak(self) -> int:
        return len(self.weak_pairs)

    @property
    def n_distant(self) -> int:
        return len(self.distant_pairs)

    @property
    def n_total(self) -> int:
        return len(self.all_pairs)


# ---------------------------------------------------------------------------
# Orbital centroids and pair-distance estimation
# ---------------------------------------------------------------------------


def compute_orbital_centroids(
    C_occ: np.ndarray,
    coords: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Compute orbital centroids from occupied MO coefficients.

    The centroid of orbital i is the expectation value of the position
    operator computed with the AO overlap distribution:
        r_i = S_{muν} C_{mui} S_{muν} C_{νi} . R_mu

    Parameters
    ----------
    C_occ : ndarray, shape (nbf, nocc)
        Occupied MO coefficients.
    coords : ndarray, shape (nbf, 3)
        Cartesian coordinates of each basis-function center (Bohr).
    S : ndarray, shape (nbf, nbf)
        AO overlap matrix.

    Returns
    -------
    centroids : ndarray, shape (nocc, 3)
        Orbital centroids in Bohr.
    """
    nocc = C_occ.shape[1]
    CS = C_occ.T @ S  # (nocc, nbf)
    centroids = np.zeros((nocc, 3))
    for i in range(nocc):
        w = CS[i, :] * C_occ[:, i]  # Mulliken weights for orbital i
        w_sum = w.sum()
        if w_sum > 1e-14:
            centroids[i, :] = (w[:, None] * coords).sum(axis=0) / w_sum
    return centroids


def compute_pair_distances(centroids: np.ndarray) -> np.ndarray:
    """Compute the distance matrix between all orbital centroids.

    Parameters
    ----------
    centroids : ndarray, shape (nocc, 3)

    Returns
    -------
    dist : ndarray, shape (nocc, nocc)
        dist[i, j] = ||r_i - r_j|| in Bohr.
    """
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


# ---------------------------------------------------------------------------
# Pair energy estimation -- dipole approximation
# ---------------------------------------------------------------------------


def estimate_pair_energy_dipole(
    dist_ij: float,
    r_i: np.ndarray,
    r_j: np.ndarray,
) -> float:
    """Estimate pair correlation energy from the dipole-dipole
    asymptotic form.

    For well-separated pairs, the leading dispersion term scales as
    R^{-6}.  We use the simple form:
        E_pair ≈ -A / R^6
    with A calibrated from the orbital extent.

    Parameters
    ----------
    dist_ij : float
        Distance between orbital centroids (Bohr).
    r_i, r_j : ndarray
        Orbital centroids (only the norm of their difference is used).

    Returns
    -------
    e_est : float
        Estimated pair energy in Hartree (negative = attractive).
    """
    if dist_ij < 1e-10:
        return -1e10  # degenerate -- should be handled separately
    # Leading dispersion coefficient A ≈ r_orb^6, but here we keep it
    # simple: all distant pairs share a flat falloff.
    return -1.0 / (dist_ij**6)


def estimate_pair_energy_overlap(
    C_occ: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Estimate pair energy from the orbital overlap.

    The differential overlap integral:
        O_{ij} = S_{muν} |C_{mui} C_{νj} S_{muν}|
    is a proxy for the exchange (K) contribution.  We use it as a cheap
    pre-screen for closely-spaced orbitals.

    Parameters
    ----------
    C_occ : ndarray, shape (nbf, nocc)
    S : ndarray, shape (nbf, nbf)
    i, j : int
        Orbital indices.

    Returns
    -------
    overlap_metric : float
        Differential-overlap measure (dimensionless).
    """
    ci = C_occ[:, i]
    cj = C_occ[:, j]
    SCj = S @ cj
    return float(np.abs(ci * SCj).sum())


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------


def classify_pairs(
    C_occ: np.ndarray,
    S: np.ndarray,
    coords: np.ndarray,
    n_frozen: int = 0,
    *,
    tcut_pairs: float = 1e-4,
    tcut_pairs_weak: float = 1e-4,
) -> PairClassification:
    """Classify all occupied orbital pairs as strong, weak, or distant."""
    nocc = C_occ.shape[1]
    active_orbs = list(range(n_frozen, nocc))
    n_active = len(active_orbs)

    if n_active < 2:
        return PairClassification(n_occ=n_active)

    # 1. Orbital centroids
    centroids = compute_orbital_centroids(C_occ, coords, S)

    # 2. Precompute SC = S @ C_occ so per-pair overlap estimate is O(nbf).
    SC = S @ C_occ

    # 3. All pairs with centroids and overlap metrics
    pairs: list[PairInfo] = []
    for idx_i, i in enumerate(active_orbs):
        ri = centroids[i]
        ci = C_occ[:, i]
        for idx_j in range(idx_i + 1, n_active):
            j = active_orbs[idx_j]
            rj = centroids[j]
            dist = float(np.linalg.norm(ri - rj))

            # Precomputed SC: overlap = |ci · SC[:,j]|
            overlap = float(np.abs(ci * SC[:, j]).sum())

            # Simple heuristic: use overlap for close pairs, dipole
            # asymptotics for distant ones.
            if dist < 3.0:
                e_est = -overlap * overlap * 0.5
            elif dist < 10.0:
                e_dip = estimate_pair_energy_dipole(dist, ri, rj)
                e_ov = -overlap * overlap * 0.5
                w = (dist - 3.0) / 7.0
                e_est = (1.0 - w) * e_ov + w * e_dip
            else:
                e_est = estimate_pair_energy_dipole(dist, ri, rj)
                if e_est > -1e-10:
                    e_est = -1e-10

            pairs.append(
                PairInfo(i=i, j=j, r_i=ri, r_j=rj, dist=dist, e_pair_est=e_est)
            )

    # 3. Classify by energy threshold
    strong: list[PairInfo] = []
    weak: list[PairInfo] = []
    distant: list[PairInfo] = []

    for p in pairs:
        if p.e_pair_est < -abs(tcut_pairs):
            strong.append(p)
            p.pair_class = PairClass.STRONG
        elif p.e_pair_est < -abs(tcut_pairs_weak):
            weak.append(p)
            p.pair_class = PairClass.WEAK
        else:
            distant.append(p)
            p.pair_class = PairClass.DISTANT

    # Sort by energy (most negative first)
    all_sorted = sorted(pairs, key=lambda x: x.e_pair_est)

    return PairClassification(
        n_occ=n_active,
        strong_pairs=strong,
        weak_pairs=weak,
        distant_pairs=distant,
        all_pairs=all_sorted,
    )
