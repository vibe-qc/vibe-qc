"""Interaction-range convergence scan -- the real-space dual of a k-mesh study.

A cyclic cluster of size ``(N1,N2,N3)`` has a geometrically dual
Γ-centred ``(N1,N2,N3)`` character mesh, and a real-space interaction radius
``R_c`` corresponds to a uniform reciprocal spacing ``Δk = pi/R_c``
(:func:`~vibeqc.periodic.ccm.wigner_seitz.kspacing_for_interaction_range`). For
one already specified block-circulant finite Hamiltonian, real-supercell and
character representations are related by the discrete Fourier transform. This
geometric/representation duality does not identify the union-and-weight Γ-CCM
construction with a neutral GDF control or with χ-CCM.

This module scans a property (energy/atom by default) against the interaction
radius ``R_c``, deriving the minimal cluster for each radius automatically
(:func:`~vibeqc.periodic.ccm.wigner_seitz.nrep_for_interaction_range`). It is the
CCM analogue of a "k-point convergence" plot.

Reference: Peintinger & Bredow, *J. Comput. Chem.* **35**, 839 (2014),
doi:10.1002/jcc.23550 (Sec. convergence with cluster size, Tables 2-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .system import CCMSystem
from .wigner_seitz import nrep_for_interaction_range

__all__ = [
    "InteractionRangePoint",
    "InteractionRangeScan",
    "ccm_interaction_range_scan",
]


@dataclass
class InteractionRangePoint:
    """One point of an interaction-range scan."""

    radius: float          # requested interaction radius R_c (bohr)
    nrep: tuple            # derived cluster mesh
    n_atoms: int
    r_in: float            # achieved WS inscribed-sphere radius l₁/2 (bohr)
    kspacing: float        # equivalent uniform k-spacing Δk = pi/r_in (bohr⁻¹)
    energy_per_atom: float
    value: float           # the scanned property (== energy_per_atom by default)
    converged: bool        # did the underlying SCF converge?

    @property
    def kmesh_equivalent(self) -> tuple:
        """The Γ-centred Monkhorst-Pack k-mesh this cyclic cluster is dual to."""
        return self.nrep


@dataclass
class InteractionRangeScan:
    """Result of :func:`ccm_interaction_range_scan` -- a convergence staircase."""

    points: List[InteractionRangePoint] = field(default_factory=list)
    label: str = "energy_per_atom"

    # -- convenience views ------------------------------------------------------
    @property
    def radii(self) -> np.ndarray:
        return np.array([p.radius for p in self.points], dtype=float)

    @property
    def nreps(self) -> List[tuple]:
        return [p.nrep for p in self.points]

    @property
    def n_atoms(self) -> np.ndarray:
        return np.array([p.n_atoms for p in self.points], dtype=int)

    @property
    def energies_per_atom(self) -> np.ndarray:
        return np.array([p.energy_per_atom for p in self.points], dtype=float)

    @property
    def values(self) -> np.ndarray:
        return np.array([p.value for p in self.points], dtype=float)

    @property
    def deltas(self) -> np.ndarray:
        """Successive change in the scanned value (``nan`` for the first point)."""
        v = self.values
        d = np.full_like(v, np.nan)
        if v.size > 1:
            d[1:] = np.diff(v)
        return d

    def converged_radius(self, tol: float) -> Optional[float]:
        """Smallest radius after which ``|Δ value| < tol`` for all later points.

        Returns the radius of the first point whose value (and every value
        beyond it) is within ``tol`` of the final value, or ``None`` if the scan
        never settles to that tolerance.
        """
        v = self.values
        if v.size == 0:
            return None
        vfinal = v[-1]
        for i in range(v.size):
            if np.all(np.abs(v[i:] - vfinal) < tol):
                return float(self.points[i].radius)
        return None

    def as_records(self) -> List[dict]:
        """Plain-dict rows (JSON-friendly) for logging / serialisation."""
        return [
            {
                "radius_bohr": p.radius,
                "nrep": list(p.nrep),
                "kmesh_equivalent": list(p.nrep),
                "n_atoms": p.n_atoms,
                "r_in_bohr": p.r_in,
                "kspacing_bohr_inv": p.kspacing,
                "energy_per_atom": p.energy_per_atom,
                "value": p.value,
                "converged": bool(p.converged),
            }
            for p in self.points
        ]

    def __repr__(self) -> str:
        return (
            f"InteractionRangeScan(label={self.label!r}, "
            f"n_points={len(self.points)}, "
            f"radii={self.radii.tolist()})"
        )


def _default_runner(method: str):
    """Default SCF runner: scalable closed-shell HF-CCM with ``method``.

    Uses the production C++ lattice-sum JK driver
    (:func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf_scalable`) rather than the dense
    Python padded-ERI path, because a convergence scan grows the cluster and the
    ``O(n⁴)`` padded ERI quickly exhausts memory. Pass ``runner=`` (e.g. the
    dense ``run_ccm_rhf``, or a DFT / GDF driver) to override.
    """
    from .scf import run_ccm_rhf_scalable

    def run(ccm: CCMSystem):
        return run_ccm_rhf_scalable(ccm, method=method)

    return run


def ccm_interaction_range_scan(
    unit_system,
    basis: str,
    radii_bohr: Sequence[float],
    *,
    runner: Optional[Callable[[CCMSystem], object]] = None,
    method: str = "aiccm2026dev-a",
    property_fn: Optional[Callable[[CCMSystem, object], float]] = None,
    property_label: str = "energy_per_atom",
    skip_duplicates: bool = True,
    overlap_guard: bool = False,
    **ccm_kwargs,
) -> InteractionRangeScan:
    """Scan a CCM property against the real-space interaction radius.

    For each radius ``R_c`` in ``radii_bohr`` the minimal cluster is derived
    (:func:`nrep_for_interaction_range`), an SCF is run, and the property is
    recorded -- the real-space dual of a k-mesh convergence study.

    Parameters
    ----------
    unit_system : PeriodicSystem
        The unit cell.
    basis : str
        Basis-set name.
    radii_bohr : sequence of float
        Interaction radii ``R_c`` to scan (bohr). Scanned in increasing order.
    runner : callable, optional
        ``runner(ccm) -> result``. The result is used as the SCF result for
        ``property_fn`` and, for the default energy property, must expose
        ``.energy_per_atom`` and ``.converged``. Defaults to closed-shell
        HF-CCM with ``method``.
    method : str
        Four-center method passed to the default runner (default
        ``"aiccm2026dev-a"``, the symmetric bare-1/r four-center -- quantitative
        for covalent / molecular / 1-D systems). Ignored if ``runner`` is
        supplied. A caller may pass a different runner, including the neutral
        fitted-torus GDF control, but the resulting scan then belongs to that
        declared operator and must not be relabelled as Γ-CCM. The
        *radius -> value* staircase is a geometric cluster-size parameterization;
        numerical values and convergence rates remain route-specific.
    property_fn : callable, optional
        ``property_fn(ccm, scf_result) -> float`` to scan an arbitrary property
        (e.g. a band gap). Defaults to ``scf_result.energy_per_atom``.
    property_label : str
        Name of the scanned property (for the result label).
    skip_duplicates : bool
        Skip a radius whose derived ``nrep`` equals the previous point's -- many
        radii map to the same cluster, and recomputing it is wasteful. The
        convergence staircase keeps one point per distinct cluster.
    overlap_guard : bool
        If true, call :meth:`CCMSystem.check_overlap_spectrum` before each SCF
        and skip (with a recorded non-converged sentinel) any cluster that fails
        the Fig.-6 positive-definiteness guard.
    **ccm_kwargs
        Forwarded to :class:`CCMSystem` (e.g. ``weight_tol_bohr``).

    Returns
    -------
    InteractionRangeScan
    """
    run = runner if runner is not None else _default_runner(method)
    dim = int(unit_system.dim)
    unit_vectors = np.asarray(unit_system.lattice, dtype=float).T

    scan = InteractionRangeScan(label=property_label)
    prev_nrep: Optional[tuple] = None
    for rc in sorted(float(r) for r in radii_bohr):
        nrep = nrep_for_interaction_range(unit_vectors, rc, dim=dim)
        if skip_duplicates and nrep == prev_nrep:
            continue
        prev_nrep = nrep
        ccm = CCMSystem(unit_system, basis=basis, interaction_range=rc, **ccm_kwargs)

        if overlap_guard:
            try:
                ccm.check_overlap_spectrum()
            except ValueError:
                scan.points.append(
                    InteractionRangePoint(
                        radius=rc, nrep=nrep, n_atoms=ccm.n_atoms,
                        r_in=ccm.wsc_inscribed_radius, kspacing=ccm.kspacing_equiv,
                        energy_per_atom=float("nan"), value=float("nan"),
                        converged=False,
                    )
                )
                continue

        result = run(ccm)
        e_per_atom = float(getattr(result, "energy_per_atom", float("nan")))
        converged = bool(getattr(result, "converged", True))
        if property_fn is None:
            value = e_per_atom
        else:
            value = float(property_fn(ccm, result))
        scan.points.append(
            InteractionRangePoint(
                radius=rc, nrep=nrep, n_atoms=ccm.n_atoms,
                r_in=ccm.wsc_inscribed_radius, kspacing=ccm.kspacing_equiv,
                energy_per_atom=e_per_atom, value=value, converged=converged,
            )
        )
    return scan
