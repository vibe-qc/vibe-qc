"""Phase G1e -- ASE bridge for periodic forces.

Thin helper that converts an :class:`ase.Atoms` with periodic
boundary conditions into a vibe-qc :class:`PeriodicSystem`, runs the
appropriate periodic SCF + gradient driver, and returns ASE-units
forces (eV/Å).

Why a separate helper instead of overloading ``VibeQC.calculate``:
the molecular ``VibeQC`` class drives ``run_rhf`` / ``run_rks`` /
... and constructs a :class:`Molecule`. Periodic systems need a
:class:`PeriodicSystem` plus a k-mesh, plus options that are
distinct from the molecular path
(:class:`PeriodicRHFOptions` /
:class:`PeriodicKSOptions`). Wrapping both behind the same
Calculator class would muddy the API. This helper keeps the
periodic path explicit; full ``Calculator``-style integration
(``periodic`` Atoms.calc with auto-routing) is a v0.6.x polish.

Use:

    from ase.build import bulk
    from vibeqc.ase_periodic import (
        atoms_to_periodic_system, run_periodic_scf, periodic_forces,
    )

    atoms = bulk("Si", "diamond", a=5.43)
    sys = atoms_to_periodic_system(atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")

    # Forces driver dispatches by functional + multiplicity.
    F_eV_per_A = periodic_forces(
        atoms, basis, kpts=[2, 2, 2],
        functional="lda",
    )

Exposes closed-shell HF (RHF) and pure-DFT (RKS/UKS) closed- and
open-shell paths. Periodic HF routes its exact exchange through the
Ewald-3D Coulomb backend at both Γ and multi-k (the bare Γ lattice
sum over-binds the exchange -- CLAUDE.md Sec.7). Open-shell HF (UHF)
periodic gradients remain deferred (they raise); hybrid-DFT periodic
forces are a follow-up.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vibeqc.ase_periodic requires ASE. Install it with `pip install ase` "
        "into your vibe-qc venv, or reinstall vibe-qc normally so its runtime "
        "dependencies are present."
    ) from exc

from . import (
    Atom,
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    compute_gradient_periodic_rhf_multi_k,
    compute_gradient_periodic_rks_gamma,
    compute_gradient_periodic_rks_multi_k,
    compute_gradient_periodic_uks_multi_k,
    monkhorst_pack,
    run_rhf_periodic_multi_k_ewald3d,
    run_rks_periodic,
    run_rks_periodic_multi_k_ewald3d,
    run_uks_periodic_multi_k_ewald3d,
)


__all__ = [
    "atoms_to_periodic_system",
    "periodic_forces",
    "VibeQCPeriodic",
]


def atoms_to_periodic_system(atoms, *, charge: int = 0,
                                multiplicity: int = 1) -> PeriodicSystem:
    """Convert an ASE :class:`Atoms` to a vibe-qc
    :class:`PeriodicSystem`.

    The cell + positions are converted from Å to bohr. The pbc flags
    determine the system dimensionality (3 for fully periodic, lower
    for slab / wire -- but vibe-qc currently expects ``dim=3``).
    """
    if atoms.pbc.sum() != 3:
        raise NotImplementedError(
            "atoms_to_periodic_system: only fully 3D-periodic Atoms "
            "are supported in v0.6 (slab / wire / molecular geometries "
            f"have pbc.sum() = {int(atoms.pbc.sum())})."
        )
    # ASE stores lattice vectors as the ROWS of atoms.cell (atoms.cell[i] = a_i);
    # vibe-qc's PeriodicSystem stores them as COLUMNS
    # (cpp/include/vibeqc/periodic.hpp:32 -- "Columns = Cartesian lattice vectors";
    # the real-space sum is r_cart = lattice @ index). Transpose at the boundary so
    # a_i lands in column i. Without the transpose, every non-orthogonal cell is
    # silently transposed; orthogonal/cubic cells are transpose-invariant and hide it.
    lattice = np.asarray(atoms.cell.array, dtype=np.float64).T / Bohr
    positions_bohr = atoms.positions / Bohr
    zs = atoms.numbers
    vq_atoms = [Atom(int(z), list(pos))
                 for z, pos in zip(zs, positions_bohr)]
    return PeriodicSystem(
        3, lattice, vq_atoms, charge=charge, multiplicity=multiplicity,
    )


def _periodic_scf_energy_and_forces(
    atoms,
    basis: BasisSet,
    *,
    kpts: Optional[Sequence[int]] = None,
    functional: Optional[str] = None,
    charge: int = 0,
    multiplicity: int = 1,
    cutoff_bohr: float = 12.0,
    nuclear_cutoff_bohr: float = 12.0,
    conv_tol_energy: float = 1e-10,
    max_iter: int = 100,
    backend: str = "ewald",
    gdf_method: str = "rsgdf",
    initial_guess: object = "AUTO",
) -> "tuple[float, np.ndarray]":
    """Run the periodic SCF + analytic gradient once.

    Returns ``(energy_hartree, forces_eV_per_A)``. Shared by
    :func:`periodic_forces` (forces only) and :class:`VibeQCPeriodic`
    (energy + forces from a single SCF). Parameter docs live on
    :func:`periodic_forces`.
    """
    sys = atoms_to_periodic_system(atoms, charge=charge,
                                      multiplicity=multiplicity)
    is_dft = functional is not None
    is_open_shell = (multiplicity != 1)

    if kpts is None:
        kpts = [1, 1, 1]
    is_gamma_only = (tuple(int(k) for k in kpts) == (1, 1, 1))

    if str(backend) not in ("ewald", "gdf"):
        raise ValueError(
            "periodic_forces: backend must be 'ewald' or 'gdf'; got "
            f"{backend!r}"
        )

    # Build appropriate options + run SCF.
    if is_dft:
        opts = PeriodicKSOptions()
        opts.functional = functional
    else:
        opts = PeriodicRHFOptions()
    from .guess import coerce_initial_guess
    opts.initial_guess = coerce_initial_guess(initial_guess)
    opts.conv_tol_energy = conv_tol_energy
    opts.max_iter = max_iter
    opts.lattice_opts.cutoff_bohr = cutoff_bohr
    opts.lattice_opts.nuclear_cutoff_bohr = nuclear_cutoff_bohr

    if str(backend) == "gdf":
        # G-PBC-002 milestone 5 (Γ) + the 2026-07-30 multi-k wiring:
        # the full GDF gradient ladder (RHF/RKS/UHF/UKS at Γ on rsgdf or
        # compcell; KRHF/KRKS/KUHF/KUKS at multi-k on rsgdf). This is
        # also the backend that fills the EWALD dispatch's
        # open-shell-HF hole below: the GDF UHF gradient exists.
        if not is_gamma_only:
            # Multi-k GDF gradients exist only on the rsgdf fit
            # (Item-4 rung 6 envelope; compcell has no multi-k
            # gradient cache).
            if str(gdf_method) != "rsgdf":
                raise NotImplementedError(
                    "periodic_forces: multi-k backend='gdf' forces "
                    "support gdf_method='rsgdf' only; got "
                    f"{gdf_method!r}"
                )
            from .periodic_k_gdf import (
                run_krhf_periodic_gdf,
                run_kuhf_periodic_gdf,
            )

            # Tuple kmesh = Γ-CENTERED mesh (the multi-k GDF driver
            # convention). The ewald backend below samples a SHIFTED
            # Monkhorst-Pack mesh for even kpts instead — the two
            # backends disagree on even meshes by construction (~44
            # mHa on MgO (2,2,2); see the production-k-sampling audit).
            kmesh_t = tuple(int(k) for k in kpts)
            mk_driver = (
                run_kuhf_periodic_gdf if is_open_shell
                else run_krhf_periodic_gdf
            )
            result = mk_driver(
                sys, basis, kmesh_t, opts,
                functional=functional,
                gdf_method="rsgdf",
                compute_gradient=True,
                progress=False,
            )
            if not result.converged:
                raise RuntimeError(
                    "periodic_forces: multi-k GDF SCF did not converge."
                )
            grad = np.asarray(result.gradient)
            return float(result.energy), -grad * (Hartree / Bohr)
        from .pbc_gdf import run_pbc_gdf_rhf, run_pbc_gdf_uhf, run_pbc_gdf_uks

        if str(gdf_method) not in ("rsgdf", "compcell"):
            raise ValueError(
                "periodic_forces: gdf_method must be 'rsgdf' or "
                f"'compcell' for backend='gdf'; got {gdf_method!r}"
            )
        gdf_common = dict(
            # rsgdf (default since the M6 full-ladder landing) is the
            # production PySCF-parity fit; compcell remains a fallback
            # knob for the AFT-corrected compensated-charge fit.
            gdf_method=str(gdf_method),
            compute_gradient=True,
            progress=False,
        )
        if is_open_shell:
            if is_dft:
                result = run_pbc_gdf_uks(
                    sys, basis, opts, functional=functional, **gdf_common
                )
            else:
                result = run_pbc_gdf_uhf(sys, basis, opts, **gdf_common)
        else:
            result = run_pbc_gdf_rhf(
                sys, basis, opts, functional=functional, **gdf_common
            )
        if not result.converged:
            raise RuntimeError(
                "periodic_forces: GDF SCF did not converge."
            )
        grad = np.asarray(result.gradient)
        forces_eV_per_A = -grad * (Hartree / Bohr)
        return float(result.energy), forces_eV_per_A

    kmesh = monkhorst_pack(sys, list(kpts))

    # Dispatch SCF.
    if is_open_shell:
        if not is_dft:
            raise NotImplementedError(
                "periodic_forces: periodic UHF gradient deferred to v0.6.x. "
                "Use a pure-DFT functional (LDA / PBE / BLYP) for open-shell "
                "periodic forces today.")
        opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
        result = run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
        if not result.converged:
            raise RuntimeError(
                "periodic_forces: UKS SCF did not converge.")
        grad = compute_gradient_periodic_uks_multi_k(
            sys, basis, result, kmesh, lattice_opts=opts.lattice_opts)
    elif is_gamma_only:
        # Γ-only path. Pure DFT (no exact exchange) is correct on the
        # simpler DIRECT_TRUNCATED SCF + Γ-only gradient drivers; HF is
        # not -- it needs Ewald-summed exchange (see the `else` branch).
        if is_dft:
            result = run_rks_periodic(sys, basis, kmesh, opts)
            if not result.converged:
                raise RuntimeError(
                    "periodic_forces: RKS SCF did not converge.")
            grad = compute_gradient_periodic_rks_gamma(
                sys, basis, result, lattice_opts=opts.lattice_opts)
        else:
            # Γ-only HF must Ewald-sum the exact-exchange (K) build. The
            # bare DIRECT_TRUNCATED Γ driver (run_rhf_periodic) leaks the
            # Madelung self-image into K and over-binds: He/STO-3G in a
            # 6-bohr box lands at -3.337 Ha vs the -2.80778 Ha molecular/
            # Ewald limit (~0.53 Ha over-bound -- the CLAUDE.md Sec.7
            # self-image-leak signature). LDA is immune (no exact
            # exchange), which is why the RKS branch above keeps the
            # direct sum. A Γ-only HF job is just a 1-point k-mesh through
            # the multi-k Ewald RHF driver -- the same backend
            # run_rhf_periodic_scf dispatches to for EWALD_3D (verified
            # identical to 4e-16 Ha). (run_rhf_periodic also can't be
            # called here regardless: its binding accepts only
            # PeriodicSCFOptions, not the PeriodicRHFOptions built above.)
            opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
            result = run_rhf_periodic_multi_k_ewald3d(
                sys, basis, kmesh, opts)
            if not result.converged:
                raise RuntimeError(
                    "periodic_forces: Γ-only RHF SCF did not converge.")
            grad = compute_gradient_periodic_rhf_multi_k(
                sys, basis, result, kmesh, lattice_opts=opts.lattice_opts)
    else:
        # Multi-k path.
        opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
        if is_dft:
            result = run_rks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
            if not result.converged:
                raise RuntimeError(
                    "periodic_forces: multi-k RKS SCF did not converge.")
            grad = compute_gradient_periodic_rks_multi_k(
                sys, basis, result, kmesh, lattice_opts=opts.lattice_opts)
        else:
            result = run_rhf_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
            if not result.converged:
                raise RuntimeError(
                    "periodic_forces: multi-k RHF SCF did not converge.")
            grad = compute_gradient_periodic_rhf_multi_k(
                sys, basis, result, kmesh, lattice_opts=opts.lattice_opts)

    # Convert from Ha/bohr to eV/Å, and flip sign (gradient -> force).
    forces_eV_per_A = -np.asarray(grad) * (Hartree / Bohr)
    return float(result.energy), forces_eV_per_A


def periodic_forces(
    atoms,
    basis: BasisSet,
    *,
    kpts: Optional[Sequence[int]] = None,
    functional: Optional[str] = None,
    charge: int = 0,
    multiplicity: int = 1,
    cutoff_bohr: float = 12.0,
    nuclear_cutoff_bohr: float = 12.0,
    conv_tol_energy: float = 1e-10,
    max_iter: int = 100,
    backend: str = "ewald",
    gdf_method: str = "rsgdf",
    initial_guess: object = "AUTO",
) -> np.ndarray:
    """Compute periodic atomic forces on an ASE :class:`Atoms`.

    Parameters
    ----------
    atoms
        Periodic ASE Atoms (pbc=True on all axes).
    basis
        Pre-built :class:`BasisSet` for the unit-cell molecule.
    kpts
        ``(nx, ny, nz)`` k-mesh. Default ``[1, 1, 1]`` (Γ-only).
    functional
        ``None`` -> HF (RHF or UHF by multiplicity). ``"lda"``, ``"pbe"``,
        ... -> KS-DFT (RKS or UKS).
    charge, multiplicity
        Net cell charge / spin multiplicity. Defaults: 0 / 1
        (closed-shell singlet).
    cutoff_bohr, nuclear_cutoff_bohr
        Lattice-sum cutoffs.
    conv_tol_energy, max_iter
        SCF convergence parameters.
    backend
        ``"ewald"`` (default) keeps the historical EWALD_3D /
        direct-truncated dispatch below. ``"gdf"`` routes Γ-only jobs
        through the compcell GDF drivers with the analytic gradient
        ladder (RHF/RKS/UHF/UKS, production AFT-on fit) — including
        open-shell HF, which the ewald backend defers. Multi-k GDF
        fails closed (G-PBC-002 Item 4).

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` forces in eV/Å (ASE units).
    """
    _energy, forces = _periodic_scf_energy_and_forces(
        atoms, basis, kpts=kpts, functional=functional, charge=charge,
        multiplicity=multiplicity, cutoff_bohr=cutoff_bohr,
        nuclear_cutoff_bohr=nuclear_cutoff_bohr,
        conv_tol_energy=conv_tol_energy, max_iter=max_iter,
        backend=backend, gdf_method=gdf_method, initial_guess=initial_guess)
    return forces


class VibeQCPeriodic(Calculator):
    """ASE Calculator for vibe-qc **periodic** SCF energy + forces.

    The periodic counterpart of :class:`vibeqc.ase.VibeQC`: hand it a
    fully-periodic ``Atoms`` (``pbc=True`` on all axes) and it routes the
    energy + analytic forces through vibe-qc's periodic SCF + gradient
    drivers, so ASE-driven workflows (``ase.optimize`` relaxation,
    ``ase.mep.NEB``, ``ase.md``) can run on vibe-qc periodic systems.

    Parameters (all via kwargs at construction)
    -------------------------------------------
    basis : str
        Basis-set name, rebuilt per geometry from the current cell.
    kpts : tuple[int, int, int]
        Monkhorst-Pack k-mesh. Default ``(1, 1, 1)`` (Γ-only).
    functional : str | None
        ``None`` -> Hartree-Fock; otherwise KS-DFT (``"lda"``, ``"pbe"``,
        ...). On the default ``backend="ewald"``, open-shell
        (``multiplicity > 1``) requires a pure-DFT functional (UKS) and
        open-shell HF raises; ``backend="gdf"`` supports open-shell HF
        and hybrids at Γ.
    charge, multiplicity : int
        Net cell charge / spin multiplicity. Defaults 0 / 1.
    cutoff_bohr, nuclear_cutoff_bohr, conv_tol_energy, max_iter
        Lattice-sum cutoffs + SCF convergence knobs (see
        :func:`periodic_forces`).
    backend : str
        ``"ewald"`` (default) or ``"gdf"`` -- the Γ-only compcell GDF
        gradient ladder (RHF/RKS/UHF/UKS, production AFT-on fit);
        multi-k GDF fails closed (see :func:`periodic_forces`).

    Example
    -------
    ::

        from ase.build import bulk
        from ase.optimize import BFGS
        from vibeqc.ase_periodic import VibeQCPeriodic

        atoms = bulk("Si", "diamond", a=5.43)
        atoms.calc = VibeQCPeriodic(basis="sto-3g", functional="pbe",
                                    kpts=(2, 2, 2))
        e = atoms.get_potential_energy()   # eV
        f = atoms.get_forces()             # eV/Å

    Energy + forces come from a single SCF per geometry; ASE's base
    Calculator caches the result until ``atoms.positions`` change.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    default_parameters = {
        "basis": "sto-3g",
        "kpts": (1, 1, 1),
        "functional": None,
        "charge": 0,
        "multiplicity": 1,
        "cutoff_bohr": 12.0,
        "nuclear_cutoff_bohr": 12.0,
        "conv_tol_energy": 1e-10,
        "max_iter": 100,
        "backend": "ewald",
        "gdf_method": "rsgdf",
        "initial_guess": "AUTO",
    }

    nolabel = True

    def calculate(  # type: ignore[override]
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        p = self.parameters
        charge = int(p["charge"])
        multiplicity = int(p["multiplicity"])
        # Basis is bound to its nuclei, so rebuild it for the current cell.
        sys = atoms_to_periodic_system(
            self.atoms, charge=charge, multiplicity=multiplicity)
        basis = BasisSet(sys.unit_cell_molecule(), p["basis"])

        energy_ha, forces_eV_per_A = _periodic_scf_energy_and_forces(
            self.atoms, basis,
            kpts=p["kpts"], functional=p["functional"],
            charge=charge, multiplicity=multiplicity,
            cutoff_bohr=float(p["cutoff_bohr"]),
            nuclear_cutoff_bohr=float(p["nuclear_cutoff_bohr"]),
            conv_tol_energy=float(p["conv_tol_energy"]),
            max_iter=int(p["max_iter"]),
            backend=str(p.get("backend", "ewald")),
            gdf_method=str(p.get("gdf_method", "rsgdf")),
            initial_guess=p["initial_guess"],
        )
        self.results["energy"] = energy_ha * Hartree
        # ASE's force_consistent free energy; identical to the total
        # energy for a non-finite-T SCF.
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = forces_eV_per_A
