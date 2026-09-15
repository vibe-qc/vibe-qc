"""Megacell (finite-supercell) periodic MP2 -- Stage 6, megacell route.

The megacell strategy (Nejad et al. 2025, Paper II -- the maintainer-selected
route, 2026-06-15) treats periodic correlation as a **finite** calculation on a
supercell: replicate the unit cell into an ``n1xn2xn3`` supercell, run a
molecular HF + MP2 on it, and report the correlation energy **per unit cell**.

Because the supercell is a finite (open-boundary) cluster, there is **no periodic
``exxdiv`` / G=0 divergence** -- the orbital energies are the ordinary molecular
ones, so the MP2 denominator ``(eᵢ+eⱼ-eₐ-e_b)`` is clean and the per-cell energy
converges to the thermodynamic limit as the megacell grows. This is exactly why
the megacell route sidesteps the boundary that blocks the Γ-Bloch route
(``handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md`` Sec. 6.5): the exxdiv contamination of
the Bloch MP2 denominator never arises in a finite cluster.

:func:`megacell_mp2` computes the canonical (untruncated) megacell MP2 energy by
reusing the molecular MP2 (:func:`vibeqc.run_mp2`). :func:`megacell_run_job`
dispatches the supercell through the molecular :func:`vibeqc.run_job` for the
**local** methods (``"dlpno-mp2"`` / ``"dlpno-ccsd"`` / ``"dlpno-ccsd(t)"``) -- the
megacell route to periodic **DLPNO-CCSD(T)**. The remaining efficiency layer is
translational-symmetry reduction of the unique pairs (Stage 5); the periodic DF /
Wannier / PAO machinery (Stages 2-4) feeds the alternative BvK route.

Verification (`tests/test_periodic_megacell_mp2.py`)
----------------------------------------------------
* The ``(1,1,1)`` megacell equals molecular MP2 under the same explicit
  all-electron recipe **exactly**.
* The per-cell correlation energy **converges** as the megacell grows.
* TDL cross-check against **PySCF.pbc KMP2** (the chosen oracle) is the next step
  -- a finite-size extrapolation study (megacell->inf vs k-mesh->inf to the same TDL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MegacellMP2Result",
    "MegacellJobResult",
    "MegacellTDLResult",
    "build_supercell_molecule",
    "megacell_mp2",
    "megacell_run_job",
    "megacell_mp2_tdl",
]


@dataclass
class MegacellMP2Result:
    """Megacell MP2 energy and bookkeeping.

    Attributes
    ----------
    nrep : tuple of int
        Supercell replication ``(n1, n2, n3)``.
    n_cells : int
        Number of unit cells in the megacell (``n1.n2.n3``).
    n_atoms : int
        Atoms in the megacell.
    e_hf : float
        Megacell HF energy (Ha, whole supercell).
    e_corr : float
        Megacell MP2 correlation energy (Ha, whole supercell).
    e_corr_per_cell : float
        ``e_corr / n_cells`` -- the per-unit-cell correlation energy that
        converges to the TDL as the megacell grows.
    e_total : float
        ``e_hf + e_corr`` (whole supercell).
    density_fit : bool
        Whether RI-MP2 (DF) or canonical MP2 was used.
    """

    nrep: Tuple[int, int, int]
    n_cells: int
    n_atoms: int
    e_hf: float
    e_corr: float
    e_corr_per_cell: float
    e_total: float
    density_fit: bool


def build_supercell_molecule(system, nrep: Sequence[int]):
    """Replicate a periodic unit cell into a finite ``n1xn2xn3`` supercell.

    Parameters
    ----------
    system : PeriodicSystem
        The periodic unit cell (provides ``lattice``, ``unit_cell``, ``charge``,
        ``multiplicity``).
    nrep : (int, int, int)
        Replication along each lattice vector.

    Returns
    -------
    Molecule
        A finite (open-boundary) molecule of the replicated atoms.
    """
    from ._vibeqc_core import Atom, Molecule

    n1, n2, n3 = (int(x) for x in nrep)
    if min(n1, n2, n3) < 1:
        raise ValueError(f"nrep components must be >= 1, got {tuple(nrep)}")

    lattice = np.asarray(system.lattice, dtype=float)
    cell_atoms = list(system.unit_cell)
    out = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                shift = (
                    i * lattice[:, 0]
                    + j * lattice[:, 1]
                    + k * lattice[:, 2]
                )
                for a in cell_atoms:
                    pos = (np.asarray(a.xyz, dtype=float) + shift).tolist()
                    out.append(Atom(int(a.Z), pos))
    # Charge / multiplicity scale with the number of cells for the supercell.
    n_cells = n1 * n2 * n3
    charge = int(system.charge) * n_cells
    # Closed-shell megacell: an even-electron neutral cell stays closed-shell;
    # carry the unit-cell multiplicity only when the cell count is 1 (a single
    # cell), else default to a closed shell (multiplicity 1).
    mult = int(system.multiplicity) if n_cells == 1 else 1
    return Molecule(out, charge, mult)


def megacell_mp2(
    system,
    basis_name: str,
    nrep: Sequence[int] = (1, 1, 1),
    *,
    density_fit: bool = False,
    aux_basis: Optional[str] = None,
    rhf_options=None,
    mp2_options=None,
) -> MegacellMP2Result:
    """Megacell (finite-supercell) periodic MP2 energy via molecular MP2.

    Parameters
    ----------
    system : PeriodicSystem
        Periodic unit cell.
    basis_name : str
        Orbital basis (e.g. ``"sto-3g"``, ``"pob-tzvp-rev2"``).
    nrep : (int, int, int)
        Supercell replication. ``(1,1,1)`` reduces to molecular MP2 on the unit
        cell; grow it to approach the thermodynamic limit.
    density_fit : bool
        Use RI-MP2 (DF) instead of canonical MP2. Default canonical (exact).
    aux_basis : str, optional
        RI auxiliary basis when ``density_fit`` is set; auto-selected if omitted.
    rhf_options, mp2_options
        Optional ``RHFOptions`` / ``MP2Options`` overrides.

    Returns
    -------
    MegacellMP2Result
    """
    from . import BasisSet, MP2Options, RHFOptions, run_mp2, run_rhf

    mol = build_supercell_molecule(system, nrep)
    basis = BasisSet(mol, basis_name)

    rhf = run_rhf(mol, basis, rhf_options or RHFOptions())
    if not getattr(rhf, "converged", True):
        raise RuntimeError(
            f"megacell_mp2: RHF on the {tuple(nrep)} supercell did not converge"
        )

    if mp2_options is None:
        mp2o = MP2Options()
        # Preserve the finite-supercell method's all-electron convergence
        # sequence; the molecular #140 policy must not silently repin a
        # periodic thermodynamic-limit study.
        mp2o.n_frozen_core = 0
    else:
        mp2o = mp2_options
    if density_fit:
        mp2o.density_fit = True
        if aux_basis:
            mp2o.aux_basis = aux_basis
        else:
            from . import default_aux_basis_for

            mp2o.aux_basis = default_aux_basis_for(basis_name, kind="ri")

    res = run_mp2(mol, basis, rhf, mp2o)

    n1, n2, n3 = (int(x) for x in nrep)
    n_cells = n1 * n2 * n3
    e_corr = float(res.e_correlation)
    return MegacellMP2Result(
        nrep=(n1, n2, n3),
        n_cells=n_cells,
        n_atoms=len(list(mol.atoms)),
        e_hf=float(res.e_hf),
        e_corr=e_corr,
        e_corr_per_cell=e_corr / n_cells,
        e_total=float(res.e_total),
        density_fit=bool(density_fit),
    )


@dataclass
class MegacellJobResult:
    """Megacell correlation via the molecular ``run_job`` driver.

    Attributes
    ----------
    nrep, n_cells, n_atoms
        Supercell shape / cell count / atom count (as in :class:`MegacellMP2Result`).
    method : str
        The ``run_job`` method used (e.g. ``"dlpno-ccsd(t)"``).
    e_total : float
        Total energy of the megacell (Ha, whole supercell).
    e_total_per_cell : float
        ``e_total / n_cells`` -- converges to the bulk energy per cell as the
        megacell grows.
    result : object
        The raw ``run_job`` result, for method-specific energies (e.g.
        ``result.dlpno_ccsd.e_corr`` / ``.e_t``).
    """

    nrep: Tuple[int, int, int]
    n_cells: int
    n_atoms: int
    method: str
    e_total: float
    e_total_per_cell: float
    result: object


def _pre_sweep_megacell_run_kwargs(molecule, method: str) -> dict[str, object]:
    """Pin the finite-supercell convention that predates issues #140/#448.

    ``frozen_core=0`` is kept separate from the DLPNO option objects because
    :func:`vibeqc.run_job` deliberately rejects two simultaneous frozen-core
    selectors. The local option objects therefore carry the old numerical
    thresholds while their ``n_frozen`` sentinel remains unset.
    """
    from .runner import _FROZEN_CORE_ROUTE_METHODS, _normalise_method_alias

    method_key = _normalise_method_alias(method)
    kwargs: dict[str, object] = {}
    if method_key in _FROZEN_CORE_ROUTE_METHODS:
        kwargs["frozen_core"] = 0

    if method_key == "dlpno-mp2":
        if int(molecule.multiplicity) > 1:
            from .dlpno.ump2 import DLPNOUMP2Options

            kwargs["dlpno_options"] = DLPNOUMP2Options(
                localise="none",
                tcut_pno=1e-8,
                tcut_pairs=0.0,
            )
        else:
            from .dlpno.mp2 import DLPNOMP2Options

            kwargs["dlpno_options"] = DLPNOMP2Options(
                localise="boys",
                tcut_pno=1e-8,
                tcut_pno_weak=1e-7,
                tcut_mkn=1e-3,
                tcut_pairs=1e-6,
                tcut_pairs_weak=1e-4,
                # #65 moved the molecular default to pno_norm="mp2". This
                # builder pins the pre-sweep convention, and tcut_pno only
                # means what the density it was cut against means, so the
                # density is pinned with the thresholds.
                pno_norm="legacy",
            )
    elif method_key in {"dlpno-ccsd", "dlpno-ccsd(t)"}:
        if int(molecule.multiplicity) > 1:
            from .dlpno.uccsd_local_solver import LocalUCCSDOptions

            kwargs["dlpno_ccsd_options"] = LocalUCCSDOptions(
                localise="boys",
                tcut_pno=1e-7,
                tcut_pairs=0.0,
                tcut_mkn=0.0,
            )
        else:
            from .dlpno.ccsd_local_solver import LocalCCSDOptions

            kwargs["dlpno_ccsd_options"] = LocalCCSDOptions(
                localise="boys",
                tcut_pno=1e-7,
                tcut_mkn=0.0,
                tcut_pairs=1e-4,
                residual_domain="pair",
                # #65 moved the molecular default to pno_norm="mp2". This
                # builder pins the pre-sweep convention, and tcut_pno only
                # means what the density it was cut against means, so the
                # density is pinned with the thresholds.
                pno_norm="legacy",
                # ... and predates the semicanonical MP2 PNO correction the
                # molecular CCSD route now applies, so that stays off too.
                pno_correction=False,
            )
    return kwargs


def megacell_run_job(
    system,
    basis: str,
    nrep: Sequence[int] = (1, 1, 1),
    *,
    method: str = "dlpno-ccsd(t)",
    output_dir: Optional[str] = None,
    num_threads: Optional[int] = None,
) -> MegacellJobResult:
    """Periodic correlation via the megacell route, dispatched through ``run_job``.

    Builds the ``n1xn2xn3`` supercell and runs the molecular :func:`vibeqc.run_job`
    on it with the requested ``method`` -- any ``run_job`` correlation method, e.g.
    ``"dlpno-mp2"``, ``"dlpno-ccsd"``, ``"dlpno-ccsd(t)"``, ``"ccsd(t)"``. This is
    the megacell route to periodic **DLPNO-CCSD(T)** (Nejad et al. 2025, Paper II):
    because a finite supercell is a molecular calculation, the ``(1,1,1)``
    megacell reproduces a molecular calculation with the same explicit recipe
    exactly, and the per-cell energy converges to the thermodynamic limit as the
    megacell grows -- with no periodic exxdiv to correct. The facade preserves
    its pre-#140/#448 all-electron and route-specific threshold convention;
    molecular unqualified defaults may evolve independently.

    Side-effect output files are suppressed; the transient ``.out`` goes to
    ``output_dir`` (a private temp dir if ``None``).

    Returns
    -------
    MegacellJobResult
    """
    import os
    import tempfile

    from . import run_job

    mol = build_supercell_molecule(system, nrep)
    n1, n2, n3 = (int(x) for x in nrep)
    n_cells = n1 * n2 * n3

    def _run(outdir: str):
        kw = _pre_sweep_megacell_run_kwargs(mol, method)
        if num_threads is not None:
            kw["num_threads"] = num_threads
        return run_job(
            mol,
            basis=basis,
            method=method,
            output=os.path.join(outdir, "megacell"),
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
            record_hostname=False,
            **kw,
        )

    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            res = _run(tmp)
    else:
        res = _run(output_dir)

    e_total = float(res.energy_total)
    return MegacellJobResult(
        nrep=(n1, n2, n3),
        n_cells=n_cells,
        n_atoms=len(list(mol.atoms)),
        method=method,
        e_total=e_total,
        e_total_per_cell=e_total / n_cells,
        result=res,
    )


@dataclass
class MegacellTDLResult:
    """Thermodynamic-limit per-cell MP2 correlation from a megacell size series.

    Attributes
    ----------
    sizes : tuple of int
        The 1-D supercell sizes ``N`` that were run.
    e_corr_total : tuple of float
        Whole-supercell correlation energy at each ``N`` (Ha).
    e_corr_per_cell : tuple of float
        Finite-``N`` per-cell *average* ``E_corr(N)/N`` -- edge-contaminated.
    increments : tuple of float
        ``E_corr(N) - E_corr(N-1)`` -- the add-one-cell energy, which converges to
        the bulk per-cell correlation.
    e_corr_per_cell_bulk : float
        **The TDL per-cell correlation energy** -- the slope ``b`` of the linear
        fit ``E_corr(N) = b.N + a``. This is the bulk value, free of the
        finite-size edge contribution.
    edge_correction : float
        The fit intercept ``a`` -- the size-independent two-end (surface) term.
    fit_max_residual : float
        ``max |E_corr(N) - (b.N + a)|`` over the series -- small => the linear
        (bulk + constant-edge) model holds and the extrapolation is trustworthy.
    """

    sizes: Tuple[int, ...]
    e_corr_total: Tuple[float, ...]
    e_corr_per_cell: Tuple[float, ...]
    increments: Tuple[float, ...]
    e_corr_per_cell_bulk: float
    edge_correction: float
    fit_max_residual: float


def megacell_mp2_tdl(
    system,
    basis_name: str,
    sizes: Sequence[int] = (1, 2, 3, 4, 5),
    *,
    axis: int = 2,
    **megacell_kwargs,
) -> MegacellTDLResult:
    """Extrapolate the per-cell MP2 correlation energy to the thermodynamic limit.

    For an open-boundary megacell the total correlation energy is
    ``E_corr(N) = b.N + a`` to leading order: ``b`` is the **bulk** per-cell
    correlation energy and ``a`` the size-independent two-end (surface)
    correction. The bulk value is therefore the *slope*, not the finite-``N``
    per-cell average ``E_corr(N)/N`` (which is edge-contaminated and converges to
    ``b`` only slowly). This runs :func:`megacell_mp2` for each 1-D size along
    ``axis`` and linear-fits ``E_corr_total`` vs ``N``.

    Parameters
    ----------
    system, basis_name
        Periodic unit cell + orbital basis.
    sizes
        1-D supercell sizes ``N`` to run (>= 2 distinct values for a fit).
    axis
        Lattice axis (0/1/2) to grow the chain along.
    **megacell_kwargs
        Forwarded to :func:`megacell_mp2` (e.g. ``density_fit``).

    Returns
    -------
    MegacellTDLResult
    """
    sizes = [int(n) for n in sizes]
    if len(set(sizes)) < 2:
        raise ValueError("megacell_mp2_tdl needs >= 2 distinct sizes for a fit")
    ax = int(axis)

    totals, per_cell = [], []
    for n in sizes:
        nrep = [1, 1, 1]
        nrep[ax] = n
        r = megacell_mp2(system, basis_name, tuple(nrep), **megacell_kwargs)
        totals.append(r.e_corr)
        per_cell.append(r.e_corr_per_cell)

    ns = np.asarray(sizes, dtype=float)
    tot = np.asarray(totals, dtype=float)
    # Linear fit E_corr_total = b.N + a; b is the bulk per-cell correlation.
    coef, *_ = np.linalg.lstsq(np.vstack([ns, np.ones_like(ns)]).T, tot, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    resid = float(np.max(np.abs(tot - (b * ns + a))))
    increments = tuple(
        float(totals[i] - totals[i - 1]) for i in range(1, len(totals))
    )

    return MegacellTDLResult(
        sizes=tuple(sizes),
        e_corr_total=tuple(totals),
        e_corr_per_cell=tuple(per_cell),
        increments=increments,
        e_corr_per_cell_bulk=b,
        edge_correction=a,
        fit_max_residual=resid,
    )
