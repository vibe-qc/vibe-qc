"""Well-tempered metadynamics layered on the general MD driver.

Metadynamics accelerates the sampling of rare events by depositing a
history-dependent Gaussian bias on a small set of **collective variables**
(CVs) -- here interatomic distances and angles. The accumulating bias pushes
the system out of free-energy minima it has already visited, so it explores
new states and, in the **well-tempered** variant, the bias converges to a
known fraction of the underlying free energy:

    F(s) = -((T + ΔT) / ΔT) . V_bias(s) + const
         = -(g / (g - 1)) . V_bias(s) + const         (g = bias factor)

This is a thin layer over :func:`vibeqc.md.run_md`: the deposited bias is
added to the energy+gradient the driver integrates (so the bias force acts
on the *same* step), and hills are deposited on a fixed stride through the
driver's per-step callback. Like the MD driver it is method-agnostic -- the
underlying potential is any ``force_fn(coords) -> (energy, gradient)``.

References
==========
Metadynamics:
    Laio & Parrinello, "Escaping free-energy minima", Proc. Natl. Acad. Sci.
    USA 99, 12562 (2002). doi:10.1073/pnas.202427399.
Well-tempered metadynamics:
    Barducci, Bussi & Parrinello, "Well-tempered metadynamics: A smoothly
    converging and tunable free-energy method", Phys. Rev. Lett. 100,
    020603 (2008). doi:10.1103/PhysRevLett.100.020603.

The driver structure (velocity-Verlet for the ions, a CV value + nuclear
bias-gradient evaluation, a periodic hill-deposition test, then the
free-energy read-out from the accumulated bias) mirrors the reference MSINDO
``metadynamics.f`` driver (algorithm only -- read for structure, not copied):
the bias deposition at ``metadynamics.f:296`` (META_BIAS_UPDATE), the bias
force at ``metadynamics.f:308`` (CV_FORCES), and the collective-coordinate +
nuclear-gradient coupling at ``metadynamics.f:327,331``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .md import MDTrajectory, Thermostat, run_md

__all__ = [
    "CollectiveVariable",
    "DistanceCV",
    "AngleCV",
    "WellTemperedBias",
    "MetadynamicsResult",
    "run_metadynamics",
]

_KB_J_PER_K: float = 1.380649e-23
_HARTREE_TO_J: float = 4.3597447222071e-18
_KB_HARTREE_PER_K: float = _KB_J_PER_K / _HARTREE_TO_J  # ≈ 3.166811e-6 Ha/K


# ----------------------------------------------------------------------
# Collective variables
# ----------------------------------------------------------------------


class CollectiveVariable:
    """Base class for a collective variable s(R).

    A CV maps the Cartesian geometry (``coords``, ``(n_atoms, 3)`` in bohr) to
    a scalar and supplies its Cartesian gradient ds/dR (same shape as
    ``coords``). Subclasses implement :meth:`value` and :meth:`gradient`.
    """

    def value(self, coords: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DistanceCV(CollectiveVariable):
    """Interatomic distance ``|R_i - R_j|`` (bohr) between atoms ``i`` and ``j``."""

    def __init__(self, i: int, j: int):
        if i == j:
            raise ValueError("DistanceCV needs two distinct atoms")
        self.i = int(i)
        self.j = int(j)

    def value(self, coords: np.ndarray) -> float:
        d = np.asarray(coords)[self.i] - np.asarray(coords)[self.j]
        return float(np.linalg.norm(d))

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=float)
        d = c[self.i] - c[self.j]
        r = float(np.linalg.norm(d))
        g = np.zeros_like(c)
        if r < 1e-12:
            return g
        u = d / r
        g[self.i] = u
        g[self.j] = -u
        return g


class AngleCV(CollectiveVariable):
    """Bond angle (radians) at vertex ``j`` for the triple ``i-j-k``."""

    def __init__(self, i: int, j: int, k: int):
        if len({i, j, k}) != 3:
            raise ValueError("AngleCV needs three distinct atoms")
        self.i = int(i)
        self.j = int(j)
        self.k = int(k)

    def value(self, coords: np.ndarray) -> float:
        c = np.asarray(coords, dtype=float)
        u = c[self.i] - c[self.j]
        v = c[self.k] - c[self.j]
        cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.arccos(np.clip(cos, -1.0, 1.0)))

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=float)
        u = c[self.i] - c[self.j]
        v = c[self.k] - c[self.j]
        ru = float(np.linalg.norm(u))
        rv = float(np.linalg.norm(v))
        g = np.zeros_like(c)
        if ru < 1e-12 or rv < 1e-12:
            return g
        pu = u / ru
        pv = v / rv
        cos = float(np.clip(np.dot(pu, pv), -1.0, 1.0))
        sin = max(np.sqrt(1.0 - cos * cos), 1e-9)
        # dth/dR_i = (costh.p̂ - q̂)/(|u| sinth); symmetric for k; j closes the sum.
        g_i = (cos * pu - pv) / (ru * sin)
        g_k = (cos * pv - pu) / (rv * sin)
        g[self.i] = g_i
        g[self.k] = g_k
        g[self.j] = -(g_i + g_k)
        return g


# ----------------------------------------------------------------------
# Well-tempered Gaussian bias
# ----------------------------------------------------------------------


class WellTemperedBias:
    """History-dependent Gaussian bias on a set of collective variables.

    Hills are Gaussians in CV space of width ``sigma`` (per CV). In the
    well-tempered scheme each new hill's height is scaled down by the bias
    already present at its centre,

        h_k = w0 . exp(-V_bias(s_k) / (k_B ΔT)),   ΔT = (g - 1) T,

    so the bias converges and the free energy follows from
    ``F(s) = -g/(g-1) . V_bias(s)`` (Barducci-Bussi-Parrinello 2008).
    """

    def __init__(
        self,
        cvs: Sequence[CollectiveVariable],
        *,
        sigma: Union[float, Sequence[float]],
        hill_height: float,
        bias_factor: float,
        temperature_K: float,
    ):
        self.cvs = list(cvs)
        n_cv = len(self.cvs)
        if n_cv == 0:
            raise ValueError("metadynamics needs at least one collective variable")
        sig = np.atleast_1d(np.asarray(sigma, dtype=float))
        if sig.size == 1:
            sig = np.full(n_cv, float(sig[0]))
        if sig.size != n_cv or np.any(sig <= 0):
            raise ValueError("sigma must be one positive width per CV")
        if bias_factor <= 1.0:
            raise ValueError("bias_factor (g) must be > 1 for well-tempered metadynamics")
        self.sigma = sig
        self.hill_height = float(hill_height)
        self.bias_factor = float(bias_factor)
        self.temperature_K = float(temperature_K)
        self.delta_T = (self.bias_factor - 1.0) * self.temperature_K
        self._kT_bias = _KB_HARTREE_PER_K * self.delta_T
        self.centers: List[np.ndarray] = []   # deposited hill centres (n_cv,)
        self.heights: List[float] = []        # deposited hill heights

    @property
    def n_cv(self) -> int:
        return len(self.cvs)

    @property
    def n_hills(self) -> int:
        return len(self.centers)

    def cv_values(self, coords: np.ndarray) -> np.ndarray:
        """Vector of CV values s(R), shape ``(n_cv,)``."""
        return np.array([cv.value(coords) for cv in self.cvs], dtype=float)

    def potential(self, s: np.ndarray) -> float:
        """Bias potential V_bias(s) at one CV point ``s`` (shape ``(n_cv,)``)."""
        if not self.centers:
            return 0.0
        c = np.asarray(self.centers)           # (H, n_cv)
        h = np.asarray(self.heights)           # (H,)
        d = (np.asarray(s)[None, :] - c) / self.sigma[None, :]
        return float(np.sum(h * np.exp(-0.5 * np.sum(d * d, axis=1))))

    def force_on_cv(self, s: np.ndarray) -> np.ndarray:
        """Bias force on the CVs, ``-dV_bias/ds`` (shape ``(n_cv,)``)."""
        if not self.centers:
            return np.zeros(self.n_cv)
        c = np.asarray(self.centers)
        h = np.asarray(self.heights)
        diff = np.asarray(s)[None, :] - c       # (H, n_cv)
        d = diff / self.sigma[None, :]
        g = h * np.exp(-0.5 * np.sum(d * d, axis=1))   # (H,)
        # dV/ds_a = S_k g_k . (-(s_a - c_ka)/s_a^2); force = -dV/ds.
        dV = -np.sum(g[:, None] * diff / (self.sigma[None, :] ** 2), axis=0)
        return -dV

    def atomic_gradient(self, coords: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Cartesian gradient of the bias, ``dV_bias/dR`` (shape ``(n_atoms, 3)``).

        Chain rule ``dV/dR = S_a (dV/ds_a)(ds_a/dR)``; the nuclear bias *force*
        is the negative of this. Added to the method gradient so the MD driver
        integrates the biased surface.
        """
        dV_ds = -self.force_on_cv(s)            # dV/ds
        grad = np.zeros_like(np.asarray(coords, dtype=float))
        for alpha, cv in enumerate(self.cvs):
            grad += dV_ds[alpha] * cv.gradient(coords)
        return grad

    def deposit(self, s: np.ndarray) -> float:
        """Deposit a well-tempered hill at CV point ``s``; return its height."""
        height = self.hill_height * np.exp(-self.potential(s) / self._kT_bias)
        self.centers.append(np.asarray(s, dtype=float).copy())
        self.heights.append(float(height))
        return float(height)

    def potential_on_grid(self, grid: np.ndarray) -> np.ndarray:
        """Bias potential over a grid of CV points.

        ``grid`` is ``(M,)`` for one CV or ``(M, n_cv)`` for several; returns
        ``(M,)`` bias values.
        """
        g = np.asarray(grid, dtype=float)
        if g.ndim == 1:
            g = g[:, None]
        return np.array([self.potential(row) for row in g], dtype=float)

    def free_energy_on_grid(self, grid: np.ndarray) -> np.ndarray:
        """Reconstructed free energy ``F(s) = -g/(g-1).V_bias(s)`` on ``grid``,
        shifted so its minimum is zero."""
        gamma = self.bias_factor
        f = -(gamma / (gamma - 1.0)) * self.potential_on_grid(grid)
        return f - np.min(f)


# ----------------------------------------------------------------------
# Result + driver
# ----------------------------------------------------------------------


@dataclass
class MetadynamicsResult:
    """Outcome of a :func:`run_metadynamics` run.

    ``trajectory`` is the (biased) :class:`vibeqc.md.MDTrajectory`;
    ``cv_trajectory`` is ``(F, n_cv)`` CV values per recorded frame; ``bias``
    is the accumulated :class:`WellTemperedBias` (hills + reconstruction).
    """

    trajectory: MDTrajectory
    cv_trajectory: np.ndarray
    bias: WellTemperedBias

    @property
    def n_hills(self) -> int:
        return self.bias.n_hills

    def free_energy(self, grid: np.ndarray) -> np.ndarray:
        """Reconstructed free-energy profile on a grid of CV points (min 0)."""
        return self.bias.free_energy_on_grid(grid)

    def bias_potential(self, grid: np.ndarray) -> np.ndarray:
        """Accumulated bias potential on a grid of CV points."""
        return self.bias.potential_on_grid(grid)


def run_metadynamics(
    force_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    coords: np.ndarray,
    collective_variables: Sequence[CollectiveVariable],
    *,
    atomic_numbers: Optional[Sequence[int]] = None,
    masses_amu: Optional[np.ndarray] = None,
    bias_factor: float = 10.0,
    hill_height: float = 1e-3,
    hill_sigma: Union[float, Sequence[float]] = 0.1,
    deposition_stride: int = 20,
    timestep_fs: float = 0.5,
    n_steps: int = 1000,
    temperature_K: float = 300.0,
    thermostat: Union[str, Thermostat] = "nose_hoover",
    thermostat_tau_fs: float = 50.0,
    velocities: Optional[np.ndarray] = None,
    remove_com: bool = True,
    n_dof: Optional[int] = None,
    seed: Optional[int] = None,
    record_stride: int = 1,
) -> MetadynamicsResult:
    """Well-tempered metadynamics on ``collective_variables``.

    Layers a history-dependent Gaussian bias (deposited every
    ``deposition_stride`` steps, height ``hill_height``, width ``hill_sigma``
    per CV, well-tempered with ``bias_factor`` g) on top of an NVT MD run of
    ``force_fn`` (the same energy+gradient callable :func:`vibeqc.md.run_md`
    takes). Metadynamics needs a thermostat -- ``thermostat`` defaults to
    Nosé-Hoover; the well-tempered free energy is reconstructed at
    ``temperature_K``.

    Returns a :class:`MetadynamicsResult`; reconstruct the free energy with
    ``result.free_energy(grid)``.

    Energies in ``trajectory.potential_energy`` include the bias (the driver
    integrates the biased surface), so the total energy is *not* conserved --
    that is expected for metadynamics.
    """
    bias = WellTemperedBias(
        collective_variables, sigma=hill_sigma, hill_height=hill_height,
        bias_factor=bias_factor, temperature_K=temperature_K,
    )

    def biased_force_fn(c: np.ndarray) -> Tuple[float, np.ndarray]:
        e, g = force_fn(c)
        s = bias.cv_values(c)
        return e + bias.potential(s), g + bias.atomic_gradient(c, s)

    def _deposit(step: int, info: dict) -> None:
        if step % deposition_stride == 0:
            bias.deposit(bias.cv_values(info["coords"]))

    # Deposit the first hill at the starting geometry so the bias is non-empty
    # from step 1 (matches the reference's initial deposition).
    bias.deposit(bias.cv_values(np.asarray(coords, dtype=float)))

    traj = run_md(
        biased_force_fn, coords, atomic_numbers=atomic_numbers,
        masses_amu=masses_amu, timestep_fs=timestep_fs, n_steps=n_steps,
        temperature_K=temperature_K, thermostat=thermostat,
        thermostat_tau_fs=thermostat_tau_fs, velocities=velocities,
        remove_com=remove_com, n_dof=n_dof, seed=seed,
        record_stride=record_stride, callback=_deposit,
    )

    cv_traj = np.array(
        [bias.cv_values(p) for p in traj.positions], dtype=float
    )
    return MetadynamicsResult(trajectory=traj, cv_trajectory=cv_traj, bias=bias)
