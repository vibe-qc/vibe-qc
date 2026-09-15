"""Runner adapter for the neutral fitted-torus real-Γ control (EXPERIMENTAL).

The per-unit-cell result adapter that wires
``jk_method='real-gamma'`` (:mod:`vibeqc.periodic.ccm.direct` --
``run_ccm_rhf_direct`` / ``run_ccm_uhf_direct`` / ``run_ccm_rks_direct`` /
``run_ccm_uks_direct``) into ``run_periodic_job``. Terminology per D89/D90:
this is the **real-Γ evaluation of the neutral fitted-torus control
Hamiltonian** -- a representation control, NOT the union-and-weight Γ-CCM
construction (``jk_method='aiccm2026dev-a'``) and not construction
evidence for it.

The CCM direct drivers return bare per-supercell API dataclasses
(``CCMSCFResult`` / ``CCMKSResult``); ``run_periodic_job``'s output stage
consumes per-unit-cell results with ``getattr``-guarded optional fields
(contract inventory: ``HANDOVER_AICCM_DIRECT_TORUS.md`` § Milestone 2d).
This module constructs that result:

* ``energy`` is per **unit cell** (the CCM per-supercell energy / N_c);
  component energies scale identically.
* ``density`` is the converged supercell-Γ density folded back to
  unit-cell lattice blocks (`_supercell_density_to_lattice_blocks`, the
  circulant projection -- exact for the converged translation-invariant
  state), presented as a ``LatticeMatrixSet`` for the density-grid
  artefacts.
* Supercell-Γ MOs are retained verbatim on ``mo_coeffs``/``mo_energies``
  (with per-spin fields open-shell); they span the ``N_c·n_μ`` supercell
  AO space, which downstream consumers must not confuse with unit-cell
  AOs -- the ``real_gamma`` diagnostics record carries ``nrep`` and
  ``n_cells`` precisely so renderers can fold like the χ-CCM-B QVF
  Γ-block does.
* ``overlap``, ``hcore``, ``fock``, and (for UKS) ``fock_beta`` are likewise
  the unscaled ``N_c·n_μ`` square supercell-Γ AO matrices.  Energy
  components are divided by ``N_c``; operators are not.
* The convention record (mission 2d): ``exchange_q0``,
  ``exchange_q0_applicability`` (active/inactive, derived from the
  presence of a full-range exact-exchange arm), the D86 executed-value
  fields (the direct SCF loops implement no mixing/damping/level shift,
  so all three are structural ``0.0`` -- recorded, not inferred), and
  ``lattice_vector_convention="columns"`` (vibe-qc lattice matrices hold
  lattice vectors as columns).

Everything here is experimental research surface: the drivers emit
``AICCM2026DevAExperimentalWarning`` and the manifest/QVF record carries
``experimental=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..._vibeqc_core import (
    BasisSet,
    GridOptions,
    LatticeSumOptions,
    PeriodicSystem,
)

__all__ = [
    "CCMRealGammaConvention",
    "CCMRealGammaResult",
    "run_real_gamma_scf",
]

#: The executed-value convention fields of the direct SCF loops. The loops
#: (`_rhf_loop`/`_rks_loop` in ccm/ri.py, the open-shell loops in
#: ccm/direct.py) implement no Fock mixing, no damping, and no level
#: shift -- these are structural zeros of the executed algorithm, recorded
#: per D86 semantics (executed values, never requested-but-ignored knobs).
_EXECUTED_FOCK_MIXING = 0.0
_EXECUTED_DAMPING = 0.0
_EXECUTED_LEVEL_SHIFT = 0.0


@dataclass(frozen=True)
class CCMRealGammaConvention:
    """Machine-readable convention record of a real-Γ direct-route run."""

    route: str                        # "real-gamma"
    exchange_q0: str                  # BVK_EWALD / STRICT_ZERO / "" (KS pure)
    exchange_q0_applicability: str    # "active" | "inactive"
    lattice_vector_convention: str    # "columns"
    nrep: tuple[int, int, int]
    n_cells: int
    cderi_build: str                  # "fold" | "supercell"
    executed_fock_mixing: float
    executed_damping: float
    executed_level_shift: float
    experimental: bool = True
    #: Screened-exchange assembly for HSE-class runs (None otherwise):
    #: (c_full, c_sr, omega_screen) -- the SR-direct (CRYSTAL-style)
    #: assembly convention; see tests/test_ccm_rsh_direct.py for the
    #: cross-code finite-size caveat.
    screened_exchange: Optional[tuple[float, float, float]] = None


@dataclass
class CCMRealGammaResult:
    """Per-unit-cell runner-contract view of a direct-route CCM result."""

    converged: bool
    n_iter: int
    energy: float                     # per unit cell (Ha)
    backend: str                      # "ccm-direct-real-gamma"
    system: PeriodicSystem
    mo_energies: np.ndarray           # supercell-Γ eigenvalues
    mo_coeffs: np.ndarray             # supercell-Γ MOs (N_c·n_mu AO rows)
    overlap: np.ndarray               # supercell-Γ overlap
    fock: Optional[np.ndarray]         # physical supercell-Γ alpha/RKS F[D]
    hcore: Optional[np.ndarray]        # supercell-Γ T + V_ne
    density: object                   # unit-cell LatticeMatrixSet (folded)
    effective_n_electrons: int        # per supercell
    real_gamma: CCMRealGammaConvention
    ccm_result: object                # the underlying CCM API result
    #: The :class:`~vibeqc.periodic.ccm.system.CCMSystem` this SCF ran on,
    #: so a correlation driver gets the identical cluster (M4b, #778).
    ccm_system: object = None
    #: The neutral cderi ``L`` this SCF actually rode, retained only when the
    #: caller passed ``retain_cderi=True`` -- it is large, and an SCF-only run
    #: must keep freeing it. Retained rather than rebuilt so the correlation
    #: rides the very array the reference converged on: the guarantee is
    #: identity rather than two builders agreeing to some tolerance, and it
    #: costs nothing, where re-deriving would also cost a second SCF.
    cderi: object = None
    #: The post-HF result when ``run_periodic_job(correlation=...)`` asked for
    #: one, else ``None`` (M4b, #778). From the NEUTRAL-RI drivers, which are
    #: construction-matched to this variant. Its own energy fields are TOTAL
    #: cyclic-cluster values; the per-unit-cell numbers are the two below,
    #: which this adapter divides.
    correlation: object = None
    #: ``correlation.e_correlation / N_c`` (Ha per unit cell), or None.
    e_correlation: Optional[float] = None
    #: SCF + correlation, per unit cell (Ha), or None.
    e_total_correlated: Optional[float] = None
    fock_mixing: float = _EXECUTED_FOCK_MIXING
    functional: Optional[str] = None
    e_xc: Optional[float] = None      # per unit cell (KS)
    e_hf_exchange: Optional[float] = None
    # Open-shell fields (None closed-shell)
    mo_energies_beta: Optional[np.ndarray] = None
    mo_coeffs_beta: Optional[np.ndarray] = None
    fock_beta: Optional[np.ndarray] = None
    density_alpha: Optional[object] = None
    density_beta: Optional[object] = None
    guess_selection: object = None


def _fold_density_to_lattice_set(ccm, D_sc, lat_opts=None):
    """Supercell-Γ density -> unit-cell ``LatticeMatrixSet`` (circulant fold)."""
    from ..._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set
    from .direct import _supercell_density_to_lattice_blocks

    unit = ccm.unit_system
    ubasis = BasisSet(unit.unit_cell_molecule(), ccm.basis_name)
    n_mu = int(ubasis.nbasis)
    base = lat_opts if lat_opts is not None else LatticeSumOptions()
    cells = direct_lattice_cells(unit, float(base.cutoff_bohr))
    blocks = _supercell_density_to_lattice_blocks(
        np.asarray(D_sc, dtype=float), ccm.nrep, n_mu, cells)
    return make_lattice_matrix_set(n_mu, cells, blocks)


def _screened_assembly_for(functional: Optional[str]):
    """(c_full, c_sr, omega_screen) for an HSE-class functional, else None."""
    if functional is None:
        return None
    from ..._vibeqc_core import Functional
    from ...periodic_screened_exchange import resolve_periodic_exchange

    func = Functional(functional, 1)
    if not bool(getattr(func, "is_range_separated", False)):
        return None
    exx = resolve_periodic_exchange(func, where="run_real_gamma_scf")
    return (float(exx.c_full), float(exx.c_sr), float(exx.omega_screen))


def run_real_gamma_scf(
    system: PeriodicSystem,
    basis_name: str,
    method: str,
    mesh: Sequence[int],
    *,
    initial_guess: object = "AUTO",
    functional: Optional[str] = None,
    exxdiv: Optional[str] = "ewald",
    cderi_build: str = "fold",
    cderi_symmetry=None,
    aux_basis: Optional[str] = None,
    ke_cutoff: float = 200.0,
    # M4b (#778): pre-build L here and keep it on the result, so a
    # correlation driver can ride the same kernel the SCF converged on.
    # Off by default -- the default path is byte-identical to before.
    retain_cderi: bool = False,
    max_iter: int = 128,
    conv_tol: float = 1e-8,
    lat_opts: Optional[LatticeSumOptions] = None,
    grid_options: Optional[GridOptions] = None,
    becke_image_radius_bohr: float = 10.0,
) -> CCMRealGammaResult:
    """Run the real-Γ direct-route SCF and adapt it to the runner contract.

    ``method`` is ``"RHF"``/``"UHF"``/``"RKS"``/``"UKS"`` (the runner's
    ``method_upper``); ``mesh`` is the BvK ``nrep`` (the runner derives it
    from the Γ-centred k-mesh argument, χ-CCM-B-style). Double hybrids are
    not wired (the composed PT2 total has no per-unit-cell SCF-state
    contract yet); use the API driver
    :func:`vibeqc.periodic.ccm.direct.run_ccm_double_hybrid_direct`.

    For KS methods, ``grid_options`` and ``becke_image_radius_bohr`` are
    forwarded unchanged to the periodic AO-grid XC builder.  The direct
    driver negotiates any external provider's required grid profile before
    building its cderi and uses the same image radius for the Becke partition
    and shifted-AO density projection.  HF methods do not consume XC-grid
    controls.
    """
    from .system import CCMSystem
    from .direct import (
        run_ccm_rhf_direct,
        run_ccm_rks_direct,
        run_ccm_uhf_direct,
        run_ccm_uks_direct,
    )

    from ...guess import InitialGuess, select_initial_guess
    selection = select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=str(method).strip().upper() in ("UHF", "UKS"),
        supported=(InitialGuess.HCORE,), driver="run_real_gamma_scf",
    )

    method_upper = str(method).strip().upper()
    nrep = tuple(int(n) for n in mesh)
    if len(nrep) != 3 or any(n < 1 for n in nrep):
        raise ValueError(
            f"run_real_gamma_scf: mesh must be three positive integers "
            f"(the BvK nrep); got {mesh!r}")
    if int(system.charge) != 0:
        raise NotImplementedError(
            "run_real_gamma_scf: charged cells are not supported by the "
            "neutral fitted-torus Hamiltonian; use a neutral unit cell"
        )
    from ...pbc_bipole_common import reject_bipole_ecp_options

    unit_basis = BasisSet(system.unit_cell_molecule(), basis_name)
    reject_bipole_ecp_options(
        object(),
        driver="run_real_gamma_scf",
        basis=unit_basis,
        system=system,
    )
    ccm = CCMSystem(system, nrep, basis_name)

    kw = dict(cderi_build=cderi_build, cderi_symmetry=cderi_symmetry,
              aux_basis=aux_basis, ke_cutoff=float(ke_cutoff),
              exxdiv=exxdiv, max_iter=int(max_iter),
              conv_tol=float(conv_tol), lat_opts=lat_opts)
    # M4b (#778): when the caller will correlate on this reference, build L
    # here and hand the SAME array to the SCF, so the correlation cannot ride
    # a different kernel than the reference converged on. Only the HF
    # branches: correlation needs an HF determinant, and the KS branches also
    # build a screened L_sr this would not cover.
    retained_cderi = None
    if retain_cderi and method_upper in ("RHF", "UHF"):
        from .direct import _neutral_cderi_for

        retained_cderi = _neutral_cderi_for(
            ccm, None, cderi_build, float(ke_cutoff), aux_basis,
            cderi_symmetry)
        # A prebuilt cderi already fixes the fitting basis, and the drivers
        # refuse aux_basis beside one rather than silently ignoring it.
        kw = dict(kw, cderi=retained_cderi, aux_basis=None)
    if method_upper == "RHF":
        res = run_ccm_rhf_direct(ccm, **kw, initial_guess=selection.requested)
    elif method_upper == "UHF":
        res = run_ccm_uhf_direct(ccm, **kw, initial_guess=selection.requested)
    elif method_upper == "RKS":
        res = run_ccm_rks_direct(
            ccm,
            functional or "pbe",
            grid_options=grid_options,
            becke_image_radius_bohr=float(becke_image_radius_bohr),
            **kw,
            initial_guess=selection.requested,
        )
    elif method_upper == "UKS":
        res = run_ccm_uks_direct(
            ccm,
            functional or "pbe",
            grid_options=grid_options,
            becke_image_radius_bohr=float(becke_image_radius_bohr),
            **kw,
            initial_guess=selection.requested,
        )
    else:
        raise NotImplementedError(
            "jk_method='real-gamma' implements RHF, UHF, RKS, and UKS; "
            f"got method={method_upper!r}")

    n_c = int(ccm.n_cells)
    applicability = getattr(res, "exchange_q0_applicability", None)
    if applicability is None:
        # HF results predate the applicability field: exact exchange is
        # always present and full-range on the HF drivers.
        applicability = "active"
    convention = CCMRealGammaConvention(
        route="real-gamma",
        exchange_q0=str(getattr(res, "exchange_q0", "") or ""),
        exchange_q0_applicability=str(applicability),
        lattice_vector_convention="columns",
        nrep=nrep,
        n_cells=n_c,
        cderi_build=str(cderi_build),
        executed_fock_mixing=_EXECUTED_FOCK_MIXING,
        executed_damping=_EXECUTED_DAMPING,
        executed_level_shift=_EXECUTED_LEVEL_SHIFT,
        screened_exchange=_screened_assembly_for(
            functional if method_upper in ("RKS", "UKS") else None),
    )

    open_shell = getattr(res, "density_alpha", None) is not None and \
        getattr(res, "density_beta", None) is not None
    # CCMUHFResult carries only per-spin densities; the KS results carry
    # the total in ``density`` (plus per-spin fields when open-shell).
    D_total = getattr(res, "density", None)
    if D_total is None:
        D_total = np.asarray(res.density_alpha) + np.asarray(res.density_beta)
    density_set = _fold_density_to_lattice_set(ccm, D_total, lat_opts)
    density_alpha_set = density_beta_set = None
    if open_shell:
        density_alpha_set = _fold_density_to_lattice_set(
            ccm, res.density_alpha, lat_opts)
        density_beta_set = _fold_density_to_lattice_set(
            ccm, res.density_beta, lat_opts)

    # CCMUHFResult exposes only per-spin MO fields; the alpha channel is
    # the conventional primary (the KS open-shell results do the same).
    mo_e = getattr(res, "mo_energies", None)
    if mo_e is None:
        mo_e = res.mo_energies_alpha
    mo_c = getattr(res, "mo_coeffs", None)
    if mo_c is None:
        mo_c = res.mo_coeffs_alpha

    overlap = np.asarray(res.overlap)
    fock_value = getattr(res, "fock", None)
    hcore_value = getattr(res, "hcore", None)
    fock_beta_value = getattr(res, "fock_beta", None)
    if method_upper in ("RKS", "UKS"):
        required_operators = [
            ("fock", fock_value),
            ("hcore", hcore_value),
        ]
        if method_upper == "UKS":
            required_operators.append(("fock_beta", fock_beta_value))
        missing = [
            name
            for name, value in required_operators
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "run_real_gamma_scf: the direct KS result omitted terminal "
                f"operator field(s) {', '.join(missing)}"
            )
    expected_shape = tuple(overlap.shape)
    for name, value in (
        ("fock", fock_value),
        ("hcore", hcore_value),
        ("fock_beta", fock_beta_value),
    ):
        if value is not None and tuple(np.asarray(value).shape) != expected_shape:
            raise RuntimeError(
                "run_real_gamma_scf: direct result operator "
                f"{name} has shape {np.asarray(value).shape}, expected the "
                f"supercell-Gamma AO shape {expected_shape}"
            )

    return CCMRealGammaResult(
        converged=bool(res.converged),
        n_iter=int(res.n_iter),
        energy=float(res.energy) / n_c,
        backend="ccm-direct-real-gamma",
        system=system,
        mo_energies=np.asarray(mo_e),
        mo_coeffs=np.asarray(mo_c),
        overlap=overlap,
        fock=(np.asarray(fock_value) if fock_value is not None else None),
        hcore=(np.asarray(hcore_value) if hcore_value is not None else None),
        density=density_set,
        effective_n_electrons=int(ccm.supercell.n_electrons()),
        real_gamma=convention,
        ccm_result=res,
        ccm_system=ccm,
        cderi=retained_cderi,
        guess_selection=getattr(res, "guess_selection", None),
        functional=(functional if method_upper in ("RKS", "UKS") else None),
        e_xc=(float(res.e_xc) / n_c if getattr(res, "e_xc", None) is not None
              else None),
        e_hf_exchange=(
            float(res.e_hf_exchange) / n_c
            if getattr(res, "e_hf_exchange", None) is not None else None),
        mo_energies_beta=(
            np.asarray(res.mo_energies_beta)
            if getattr(res, "mo_energies_beta", None) is not None else None),
        mo_coeffs_beta=(
            np.asarray(res.mo_coeffs_beta)
            if getattr(res, "mo_coeffs_beta", None) is not None else None),
        fock_beta=(
            np.asarray(fock_beta_value)
            if fock_beta_value is not None else None),
        density_alpha=density_alpha_set,
        density_beta=density_beta_set,
    )
