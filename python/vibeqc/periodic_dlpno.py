"""DLPNO-CCSD and DLPNO-CCSD(T) on crystals through the molecular engine.

Increments B1, B2 and C of ``docs/design_dlpno_correlation_source.md``: the
Gamma-point GDF route and its extension to Gamma-centred meshes folded to
the real BvK supercell. At Gamma the reference is the multi-k GDF engine at one k-point
(:func:`~vibeqc.periodic_k_gdf.run_krhf_periodic_gdf` with
``fit_source="classic"``, the per-pair RSGDF fit the driver can retain,
and the Ewald exchange zero mode) with its fit retained; the correlation factors are that
same fit in the canonical auxiliary frame, so reference and factors are one
Hamiltonian. Open shells use KUHF with its separate spin Fock matrices and
the same spin-independent retained fit. The engine is
:func:`~vibeqc.dlpno.ccsd_local_solver.run_local_dlpno_ccsd`
or :func:`~vibeqc.dlpno.uccsd_local_solver.run_local_dlpno_uccsd`,
reading a :func:`~vibeqc.dlpno.source.gamma_gdf_source`; nothing periodic
gets its own coupled-cluster loop.

A mesh is run as the unreduced Gamma-centred mesh and folded to the real
BvK supercell by :func:`~vibeqc.dlpno.source.bvk_supercell_source`
(increment B2), so the engine sees one real supercell problem; energies
are reported in total and per cell. What this route refuses, with the
reason in the error: BIPOLE references, symmetry-reduced references,
smeared occupations and cells with ``dim < 3``. It is not a production route until its accuracy against canonical
periodic CCSD(T) and its cost against cell size are measured; the gates are
on the design page.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np

from .dlpno.ccsd_local_solver import LocalCCSDOptions, LocalCCSDResult, run_local_dlpno_ccsd
from .dlpno.source import (
    CorrelationSource, bvk_supercell_source, detect_reference_kind, gamma_gdf_source,
)
from .dlpno.uccsd_local_solver import (
    LocalUCCSDOptions, LocalUCCSDResult, run_local_dlpno_uccsd,
)

__all__ = [
    "PERIODIC_DLPNO_SCF_KWARGS",
    "PeriodicDLPNOCCSDResult",
    "correlate_periodic_reference",
    "periodic_dlpno_scf_mesh",
    "prepare_periodic_cc_options",
    "run_periodic_dlpno_ccsd",
]

#: Keyword arguments the reference SCF needs so its fit can be the
#: correlation factors: the classic RSGDF per-pair fit is the one the
#: driver can retain (the default source's Gamma fast path keeps nothing),
#: and retention must be asked for. The two sources agree on the SCF
#: energy to the fit tolerance (H2 cube/STO-3G: 6e-10 Ha).
PERIODIC_DLPNO_SCF_KWARGS = {"fit_source": "classic", "retain_fit_cache": True}


@dataclass
class PeriodicDLPNOCCSDResult:
    """Total and per-cell energies of a DLPNO-CCSD(T) run at the Gamma point
    or on a Gamma-centred mesh folded to the real BvK supercell.

    ``n_occ`` and ``n_vir`` count spatial orbitals for RHF, spin orbitals
    for UHF. The explicit per-spin counts are populated on both routes.
    """

    kmesh: tuple[int, int, int]
    n_cells: int
    n_bf: int
    n_occ: int
    n_vir: int
    n_aux: int
    e_hf: float
    e_corr: float
    e_t: Optional[float]
    e_total: float
    e_hf_per_cell: float
    e_corr_per_cell: float
    e_t_per_cell: Optional[float]
    e_total_per_cell: float
    converged: bool
    exchange_q0: str
    source_kind: str
    source_identity: str
    same_source: bool
    localisation: str
    cc_result: LocalCCSDResult | LocalUCCSDResult
    hf_result: Any
    source: CorrelationSource
    reference_kind: str = "rhf"
    n_occ_alpha: int = 0
    n_occ_beta: int = 0
    n_vir_alpha: int = 0
    n_vir_beta: int = 0


def _mesh_tuple(kmesh: Sequence[int]) -> tuple[int, int, int]:
    mesh = tuple(int(x) for x in kmesh)
    if len(mesh) != 3 or min(mesh) < 1:
        raise ValueError(
            f"run_periodic_dlpno_ccsd: kmesh must be three positive integers, got {mesh}"
        )
    return mesh  # type: ignore[return-value]


def periodic_dlpno_scf_mesh(system, kmesh: Sequence[int]):
    """The k-point set the DLPNO reference SCF runs on.

    ``(1, 1, 1)`` for the Gamma point; otherwise the unreduced Gamma-centred
    mesh in cyclic order (:func:`~vibeqc.periodic.chi.scf.cyclic_gamma_mesh`),
    which the BvK fold needs. It is built here rather than left to a
    driver's mesh default so the fold can check the reference against it.
    """
    mesh = _mesh_tuple(kmesh)
    if int(np.prod(mesh)) == 1:
        return (1, 1, 1)
    from .periodic.chi.scf import cyclic_gamma_mesh

    return cyclic_gamma_mesh(system, mesh)


def prepare_periodic_cc_options(
    cc_options: Optional[LocalCCSDOptions | LocalUCCSDOptions],
    *,
    with_triples: bool,
    reference_kind: str = "rhf",
    where: str = "run_periodic_dlpno_ccsd",
) -> LocalCCSDOptions | LocalUCCSDOptions:
    """The engine options a periodic run may use.

    The adapter localises the active occupied block itself (Pipek-Mezey,
    PBC-safe), so the engine must run with ``localise="none"``; the
    engine's own Boys localiser uses open-cluster dipole integrals. The
    dataclass default ``"boys"`` is replaced silently, any other explicit
    choice is refused with the reason. ``compute_triples`` follows
    ``with_triples``.
    """
    if reference_kind not in ("rhf", "uhf"):
        raise ValueError(f"{where}: periodic correlation supports RHF or UHF references")
    options_class = LocalUCCSDOptions if reference_kind == "uhf" else LocalCCSDOptions
    options = options_class() if cc_options is None else cc_options
    if not isinstance(options, options_class):
        raise TypeError(f"{where}: {reference_kind.upper()} needs {options_class.__name__}")
    if options.n_frozen is not None and (
        int(options.n_frozen) != options.n_frozen or options.n_frozen < 0
    ):
        raise ValueError(f"{where}: n_frozen must be a nonnegative integer or None")
    localise = str(getattr(options, "localise", "boys")).lower()
    if localise == "boys":
        options = replace(options, localise="none")
    elif localise != "none":
        raise ValueError(
            f"{where}: cc_options.localise must be 'none'; the adapter "
            "localises the occupied orbitals (Pipek-Mezey) and the "
            f"engine's localiser is not periodic-safe (got {localise!r})"
        )
    if bool(options.compute_triples) != bool(with_triples):
        options = replace(options, compute_triples=bool(with_triples))
    return options


def correlate_periodic_reference(
    system,
    basis,
    hf,
    *,
    kmesh: Sequence[int],
    options: LocalCCSDOptions | LocalUCCSDOptions,
    with_triples: bool = True,
    localise: str = "pipek-mezey",
    aux_basis: Optional[str] = None,
) -> PeriodicDLPNOCCSDResult:
    """Run the engine on a converged GDF reference through its adapter.

    ``hf`` is a :func:`~vibeqc.periodic_k_gdf.run_krhf_periodic_gdf` or
    :func:`~vibeqc.periodic_k_gdf.run_kuhf_periodic_gdf` result
    on :func:`periodic_dlpno_scf_mesh` with :data:`PERIODIC_DLPNO_SCF_KWARGS`;
    ``options`` come from :func:`prepare_periodic_cc_options` with
    ``n_frozen`` counted per cell. At Gamma the adapter is
    :func:`~vibeqc.dlpno.source.gamma_gdf_source`, on a mesh
    :func:`~vibeqc.dlpno.source.bvk_supercell_source`, which folds the
    reference to the real BvK supercell; the engine then counts frozen
    orbitals on the supercell. Energies are returned in total and per cell.
    """
    mesh = _mesh_tuple(kmesh)
    n_cells = int(np.prod(mesh))
    reference_kind = detect_reference_kind(hf)
    options = prepare_periodic_cc_options(
        options, with_triples=with_triples, reference_kind=reference_kind,
        where="correlate_periodic_reference",
    )
    if not bool(getattr(hf, "converged", False)):
        raise RuntimeError(
            "correlate_periodic_reference: the GDF reference did not converge"
        )
    aux_name = aux_basis or getattr(hf, "aux_basis_name", None)
    n_frozen_cell = int(getattr(options, "n_frozen", 0) or 0)
    if n_cells > 1:
        source = bvk_supercell_source(
            system,
            basis,
            hf,
            mesh=mesh,
            aux_basis_name=aux_name,
            n_frozen_per_cell=n_frozen_cell,
            localise=localise,
        )
        # The engine counts frozen orbitals on the supercell.
        options = replace(options, n_frozen=n_frozen_cell * n_cells)
    else:
        source = gamma_gdf_source(
            system,
            basis,
            hf,
            aux_basis_name=aux_name,
            n_frozen=n_frozen_cell,
            localise=localise,
        )
    is_uhf = reference_kind == "uhf"
    engine = run_local_dlpno_uccsd if is_uhf else run_local_dlpno_ccsd
    cc = engine(None, None, None, None, options, source=source)

    n_bf = int(source.basis.nbasis)
    ncore = int(getattr(source.reference, "ecp_total_ncore", 0) or 0)
    ne = int(source.molecule.n_electrons()) - ncore
    spin = int(source.molecule.multiplicity) - 1
    na, nb = (ne + spin) // 2, (ne - spin) // 2
    n_occ = na + nb if is_uhf else na
    n_vir = 2 * n_bf - n_occ if is_uhf else n_bf - n_occ
    e_hf = float(source.reference.energy)
    e_t = float(cc.e_t) if with_triples else None
    e_total = e_hf + float(cc.e_corr) + (e_t or 0.0)
    return PeriodicDLPNOCCSDResult(
        kmesh=mesh,
        n_cells=n_cells,
        n_bf=n_bf,
        n_occ=n_occ,
        n_vir=n_vir,
        n_aux=int(getattr(source.factors, "n_aux", 0) or 0),
        e_hf=e_hf,
        e_corr=float(cc.e_corr),
        e_t=e_t,
        e_total=e_total,
        e_hf_per_cell=e_hf / n_cells,
        e_corr_per_cell=float(cc.e_corr) / n_cells,
        e_t_per_cell=(e_t / n_cells) if e_t is not None else None,
        e_total_per_cell=e_total / n_cells,
        converged=bool(cc.converged),
        exchange_q0=source.conventions.exchange_q0,
        source_kind=source.kind,
        source_identity=source.identity,
        same_source=bool(source.conventions.same_source),
        localisation=str(getattr(source.reference, "localisation", localise)),
        cc_result=cc,
        hf_result=hf,
        source=source,
        reference_kind=reference_kind,
        n_occ_alpha=na,
        n_occ_beta=nb,
        n_vir_alpha=n_bf - na,
        n_vir_beta=n_bf - nb,
    )


def run_periodic_dlpno_ccsd(
    system,
    basis_name: str,
    *,
    kmesh: Sequence[int] = (1, 1, 1),
    aux_basis: Optional[str] = None,
    rhf_options=None,
    uhf_options=None,
    cc_options: Optional[LocalCCSDOptions | LocalUCCSDOptions] = None,
    with_triples: bool = True,
    localise: str = "pipek-mezey",
    progress=None,
    verbose: Optional[int] = None,
    **gdf_kwargs,
) -> PeriodicDLPNOCCSDResult:
    """DLPNO-CCSD, or DLPNO-CCSD(T), on a 3-D cell at the Gamma point or on a
    Gamma-centred mesh folded to the real BvK supercell.

    Non-singlet cells select KUHF and
    :class:`~vibeqc.dlpno.uccsd_local_solver.LocalUCCSDOptions`. Explicit
    ``uhf_options`` or ``LocalUCCSDOptions`` also select UHF for a singlet.
    RHF otherwise uses :class:`~vibeqc.dlpno.ccsd_local_solver.LocalCCSDOptions`.
    The corresponding ``rhf_options`` or ``uhf_options`` controls the SCF.
    ``cc_options`` use the corresponding molecular engine's shipped
    defaults unless given; ``localise`` must be ``"none"`` on
    them, because the adapter localises the active occupied block itself
    (``localise="pipek-mezey"`` here, PBC-safe) and the engine's Boys
    localiser uses open-cluster dipole integrals. ``gdf_kwargs`` go to the
    GDF driver. Each spin channel must have an integer occupied band count
    at every k-point and a global occupied-virtual gap. The adapter retains
    the SCF Fock, including its exchange zero-mode convention.
    """
    from . import BasisSet
    from .periodic_k_gdf import run_krhf_periodic_gdf, run_kuhf_periodic_gdf

    if int(system.dim) != 3:
        raise NotImplementedError(
            "run_periodic_dlpno_ccsd: the Gamma-point GDF route runs on 3-D "
            f"cells (dim=3); got dim={int(system.dim)}. Embed a chain or slab "
            "in a vacuum box."
        )
    mesh = _mesh_tuple(kmesh)
    is_uhf = (
        int(system.multiplicity) != 1 or uhf_options is not None
        or isinstance(cc_options, LocalUCCSDOptions)
    )
    if is_uhf and rhf_options is not None:
        raise ValueError("run_periodic_dlpno_ccsd: a UHF reference needs uhf_options, not rhf_options")
    options = prepare_periodic_cc_options(
        cc_options, with_triples=with_triples, reference_kind="uhf" if is_uhf else "rhf"
    )

    molecule = system.unit_cell_molecule()
    basis = BasisSet(molecule, basis_name)
    for key, value in PERIODIC_DLPNO_SCF_KWARGS.items():
        gdf_kwargs.setdefault(key, value)
    scf = run_kuhf_periodic_gdf if is_uhf else run_krhf_periodic_gdf
    hf = scf(
        system,
        basis,
        periodic_dlpno_scf_mesh(system, mesh),
        uhf_options if is_uhf else rhf_options,
        aux_basis=aux_basis,
        progress=progress,
        verbose=verbose,
        **gdf_kwargs,
    )
    if not bool(getattr(hf, "converged", False)):
        raise RuntimeError(
            "run_periodic_dlpno_ccsd: the GDF reference did not converge"
        )
    return correlate_periodic_reference(
        system,
        basis,
        hf,
        kmesh=mesh,
        options=options,
        with_triples=with_triples,
        localise=localise,
        aux_basis=aux_basis,
    )
