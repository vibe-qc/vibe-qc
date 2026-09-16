"""Periodic GDF SCF driver via compensated charges (compcell) + exxdiv='ewald'.

This is the production GDF driver for vibe-qc's periodic SCF. It pairs
:func:`vibeqc.aux_basis.build_lpq_compcell` (Sun, *J. Chem. Phys.* **147**,
164119 (2017); PySCF ``_CCGDFBuilder``) with PySCF's ``exxdiv='ewald'``
Madelung K-matrix shift (McClain, Sun, Chan, Berkelbach, *J. Chem.
Theory Comput.* **13**, 1209 (2017)) to match PySCF's
``KRHF(cell).density_fit()`` SCF total energy to µHa.

The driver is intentionally narrow in this initial landing:

* Closed-shell RHF only (UKS / open-shell not yet wired).
* Γ-only path (single k-point at the BZ origin); multi-k is the
  next milestone (see ``run_krhf_periodic_gdf`` for the multi-k
  scaffolding that this will plug into).
* Coordinated with the parallel BIPOLE chat through the
  :class:`PBCMethod` enum (defined here for this branch; once the
  BIPOLE driver lands the enum + dispatcher will move to a shared
  ``pbc_options.py``).

The Coulomb (J) and exchange (K) matrices are both built from the
compcell Lpq factor -- the "true GDF" path. This differs from
:func:`run_rhf_periodic_gamma_gdf` which on dim=3 short-circuits J to
the Ewald-3D composed builder and K to the molecular-limit real-space
kernel (see Pitfall 3 in the GDF-chat handover). For diffuse aux
(every standard JKfit aux) this distinction matters: only compcell
keeps the Lpq factor cutoff-stable, which keeps J/K stable too.

.. note:: **AFT correction is applied by default**
   (``apply_aft_correction=True``); η is largely a residual knob.

   The AFT long-range correction (PySCF ``_CCGDFBuilder.get_2c2e``
   j2c_p subtraction) conditions the compcell metric and makes the
   answer largely η-independent. With both the 2c and 3c AFT pieces
   in the matched ``libint`` convention (the default, η ≈ 1.0) the
   compcell path reaches sub-mHa PySCF parity on H2/vacuum-box.

   For µHa parity use ``gdf_method='rsgdf'`` (range-separated GDF;
   sub-µHa on H2, -0.5 µHa on LiH primitive FCC) -- the recommended
   µHa path. The ``aft_ft_convention='libcint'`` combination is a
   known +178 mHa foot-gun and emits a runtime warning; keep the
   default ``'libint'``.

   Historical (pre-AFT): without the correction, parity was
   η-tunable at η ≈ 0.25-0.3 on loose-aux vacuum boxes and the
   compensated metric was ill-conditioned on tight ionic cells.

References
----------
* Sun, *J. Comput. Chem.* **38**, 2399 (2017), DOI 10.1002/jcc.24890
  -- periodic GDF formulation.
* Sun et al., *J. Chem. Phys.* **147**, 164119 (2017),
  DOI 10.1063/1.4998644 -- eigendecomposition + threshold protocol.
* Mintmire, Sabin, Trickey, *Phys. Rev. A* **25**, 88 (1982);
  Dunlap, Connolly, Sabin, *J. Chem. Phys.* **71**, 3396 (1979);
  Whitten, *J. Chem. Phys.* **58**, 4496 (1973) -- modrho theory.
* McClain, Sun, Chan, Berkelbach, *J. Chem. Theory Comput.* **13**,
  1209 (2017), DOI 10.1021/acs.jctc.6b01184 -- exxdiv='ewald'.
* Ye, Berkelbach, *J. Chem. Phys.* **154**, 131104 (2021),
  DOI 10.1063/5.0046617 -- RSGDF alternative to compcell.
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

import enum
import os
import warnings
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    CoulombMethod,
    InitialGuess,
    LatticeSumOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFIteration,
    bloch_sum,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    ewald_nuclear_repulsion,
    nuclear_repulsion_per_cell,
)
from .aux_basis import (
    _CompcellFitState,
    _basis_fingerprint,
    _build_lpq_compcell_state,
    build_lpq_mdf,
    build_lpq_native,
    build_lpq_native_fft,
    default_aux_for,
    make_aux_basis_set,
    make_modrho_aux_basis,
)
from .guess import (
    _coerce_periodic_driver_guess,
    initial_densities_open_shell,
    initial_density_closed_shell,
)
from .linear_dependence import scf_preflight_overlap_check
from .madelung import (
    apply_exxdiv_ewald_to_K,
    exxdiv_ewald_energy_shift,
    madelung_constant_for_cell,
)
from .occupations import aufbau_occupations_per_k as _aufbau_occupations_per_k
from .periodic_rhf_ewald import _canonical_orthogonalizer
from .periodic_scf_accelerators import DynamicDamping, PeriodicSCFAccelerator
from .periodic_v_ne import compute_nuclear_lattice_dispatch
from .periodic_screened_exchange import (
    reject_periodic_gdf_unsupported_functional,
    reject_unscreened_range_separated,
)
from .progress import ProgressLogger, resolve_progress

__all__ = [
    "PBCMethod",
    "PBCExxDiv",
    "PBCGDFResult",
    "run_pbc_gdf_rhf",
]


class PBCMethod(enum.Enum):
    """Periodic SCF algorithm selector.

    Coordinated with the BIPOLE chat for the v0.8.0 release --
    ``GDF`` is owned by this branch, ``BIPOLE`` by the sibling
    chat. The enum will move to ``vibeqc.pbc_options`` once both
    drivers land.
    """

    GDF = "gdf"
    BIPOLE = "bipole"


class PBCExxDiv(enum.Enum):
    """Treatment of the G=0 self-image divergence in HF exchange.

    * ``EWALD`` -- PySCF's default. Adds the Ewald-Madelung K-matrix
      shift ``K(k) += ξ . S(k).D(k).S(k)`` per k-point, where
      ``ξ = a_M / L`` is the cell Madelung constant. The shift is
      the standard fix for the O(1/N_k) bias in HF exchange on a
      finite Monkhorst-Pack mesh (McClain et al. 2017). REQUIRED
      for µHa parity with PySCF ``exxdiv='ewald'``.

    * ``NONE`` -- Skip the shift. Reproduces PySCF
      ``exxdiv=None`` / vibe-qc's pre-2026-05-17 behaviour. Useful
      for cross-comparison and for systems where the user wants
      the unshifted K (e.g. for explicit charge-correction sweeps).
    """

    EWALD = "ewald"
    NONE = "none"


@dataclass
class PBCGDFResult:
    """SCF result from :func:`run_pbc_gdf_rhf` / :func:`run_pbc_gdf_rks`."""

    energy: float
    e_electronic: float
    e_nuclear: float
    e_coulomb: float
    e_hf_exchange: float
    e_exxdiv: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    hcore: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    scf_trace: List[SCFIteration] = field(default_factory=list)
    aux_basis_name: str = ""
    n_aux: int = 0
    n_fit: int = 0
    madelung_constant: float = 0.0
    exxdiv: str = "ewald"
    compcell_eta: float = 0.2
    backend: str = "pbc-gdf-compcell"
    gradient: Optional[np.ndarray] = None
    e_xc: float = 0.0
    functional: str = ""
    apply_aft_correction: bool = True
    aft_precision: float = 1e-10
    aft_ft_convention: str = "libint"
    v_ne_backend: str = "analytic_ft"
    v_ne_ke_cutoff: float = 200.0
    #: The RSGDF high-``|G|`` tail cutoff (Ha) this run actually applied
    #: after :func:`_auto_rsgdf_tail_ke_cutoff` resolution, or ``None``
    #: for base-mesh-only. Recorded so a consumer can tell WHICH
    #: Hamiltonian produced the energy: the Γ fast path auto-sizes a tail
    #: on the tight-core class while other routes do not, and that silent
    #: difference was worth -4.99e-01 Ha/cell against the CCM direct
    #: control on MgO/STO-3G (GitLab IID 307).
    rsgdf_tail_ke_cutoff: Optional[float] = None
    #: The BASE rsgdf reciprocal mesh (Ha) this run actually used. Recorded
    #: alongside the tail for the same auditability reason, and because a
    #: consumer's ``*_executed`` field can only be evidence if it is a real
    #: read: ``studies/aiccm-2026/run_case_cmp.py`` reported
    #: ``rsgdf_ke_cutoff_executed`` by copying its own request, so the field
    #: could not disagree with its ``*_requested`` twin (GitLab IID 307).
    rsgdf_ke_cutoff: float = 200.0
    gdf_rcut_strategy: Optional[str] = "pyscf_auto"
    gdf_rcut_precision: float = 1e-8
    gdf_linear_dep_threshold: float = 1e-9
    gdf_lattice_cutoff_bohr: float = 15.0
    gdf_nuclear_cutoff_bohr: float = 25.0
    gdf_fit_cutoff_2c: float = float("nan")
    gdf_fit_cutoff_3c: float = float("nan")
    aux_basis_fingerprint: str = ""
    # Relative accuracy estimate of the analytic gradient's fit-response
    # terms (rsgdf path): eps * lam_max / lam_min_kept of the fit
    # metric. ~3e-7 on the MgO dense-core class (near-threshold kept
    # modes), ~1e-11 on vacuum-class anchors. See the dense-core
    # gradient entry in HANDOVER_OPEN_BUGS_V015.md.
    gradient_conditioning: float = float("nan")

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def _density_from_orbitals_and_occupations(
    C: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    D = (C * np.asarray(occupations, dtype=float)[None, :]) @ C.T
    return 0.5 * (D + D.T)


def _open_shell_gamma_occupy(
    C_alpha,
    eps_alpha,
    C_beta,
    eps_beta,
    n_alpha,
    n_beta,
    smear_opts,
    *,
    preserve_column_order=False,
):
    """One chemical potential per spin, sharing the multi-k T=0 convention.

    An explicit spin-pattern hold retains its selected occupied columns.
    Ground-state filling treats a degenerate frontier as the same ensemble
    used by restricted GDF, without imposing a spurious spin splitting.
    """
    if preserve_column_order and not smear_opts.enabled:
        D_a = C_alpha[:, :n_alpha] @ C_alpha[:, :n_alpha].T
        D_b = (
            C_beta[:, :n_beta] @ C_beta[:, :n_beta].T
            if n_beta > 0
            else np.zeros_like(D_a)
        )
        occ_a = np.zeros(C_alpha.shape[1])
        occ_b = np.zeros(C_beta.shape[1])
        occ_a[:n_alpha] = occ_b[:n_beta] = 1.0
        return D_a, D_b, occ_a, occ_b, 0.0, 0.0, 0.0
    from .periodic_k_gdf import _gdf_open_shell_occupations

    a_res, b_res = _gdf_open_shell_occupations(
        [np.asarray(eps_alpha)],
        [np.asarray(eps_beta)],
        weights=[1.0],
        n_alpha=n_alpha,
        n_beta=n_beta,
        smearing=smear_opts,
    )
    occ_a = np.asarray(a_res.occupations_per_k[0], dtype=float)
    occ_b = np.asarray(b_res.occupations_per_k[0], dtype=float)
    D_a = _density_from_orbitals_and_occupations(C_alpha, occ_a)
    D_b = (
        _density_from_orbitals_and_occupations(C_beta, occ_b)
        if n_beta > 0
        else np.zeros_like(D_a)
    )
    return (
        D_a,
        D_b,
        occ_a,
        occ_b,
        float(a_res.mu),
        float(b_res.mu),
        float(a_res.entropy + b_res.entropy),
    )


def _smeared_occupation_residual_gamma(
    diagonalise,
    F_alpha,
    F_beta,
    occ_a,
    occ_b,
    n_alpha,
    n_beta,
    smear_opts,
) -> float:
    """``max_s max_i |occ_s - occ(eigs(F_s))|`` for the smeared Γ drivers.

    Occupation self-consistency residual of the smeared convergence
    test: the stored per-spin occupations (which built the density the
    current plain Fock came from) must be the Fermi filling of that
    Fock's OWN eigenvalues. The energy + commutator tests cannot see a
    violation on zero-commutator fixtures (H2-in-box class), where a
    stale extrapolated Fock freezes the occupations at a spurious fixed
    point (see the zero-commutator floor in
    ``periodic_scf_accelerators.py`` and
    ``smeared_occupation_selfconsistency_tolerance``). The comparison
    is on descending-sorted occupations: a Fermi filling is monotone in
    the (sorted) eigenvalues, and the stored array can be
    column-reordered by the SPINLOCK MOM hold.
    """
    C_a_chk, eps_a_chk = diagonalise(F_alpha)
    C_b_chk, eps_b_chk = diagonalise(F_beta)
    (_, _, occ_a_chk, occ_b_chk, _, _, _) = _open_shell_gamma_occupy(
        C_a_chk, eps_a_chk, C_b_chk, eps_b_chk, n_alpha, n_beta, smear_opts
    )
    residual = 0.0
    for stored, own in ((occ_a, occ_a_chk), (occ_b, occ_b_chk)):
        stored = np.asarray(stored, dtype=float)
        own = np.asarray(own, dtype=float)
        if stored.size == 0 or own.size == 0:
            continue
        residual = max(
            residual,
            float(np.max(np.abs(np.sort(stored)[::-1] - np.sort(own)[::-1]))),
        )
    return residual


def _gamma_open_shell_s_squared(
    n_alpha,
    n_beta,
    C_alpha,
    C_beta,
    S,
    occ_a,
    occ_b,
    smear_opts,
) -> float:
    """<S^2> for a Γ open-shell determinant. T = 0 -> the exact integer-occupation
    value (:func:`_spin_squared`, **bit-identical** to the pre-smearing path);
    T > 0 -> the fractional-occupation (ensemble-UHF) value via
    :func:`_multi_k_s_squared` at the single Γ k-point (weights = [1])."""
    from .periodic_uhf_ewald import _spin_squared

    if np.asarray(occ_a).size == 0:
        return float(_spin_squared(n_alpha, n_beta, C_alpha, C_beta, S))
    from .periodic_k_gdf import _multi_k_s_squared

    return float(
        _multi_k_s_squared(
            n_alpha,
            n_beta,
            [C_alpha],
            [C_beta],
            [S],
            [1.0],
            occ_alpha_k=[occ_a],
            occ_beta_k=[occ_b],
        )
    )


def _gamma_gradient_orbital_blocks(
    C_f: np.ndarray,
    eps_f: np.ndarray,
    n_int: int,
    occ: np.ndarray,
    smear_opts,
) -> Tuple[np.ndarray, np.ndarray]:
    """Occupied-orbital block + eigenvalues for one spin channel of the
    Γ gradient assembly.

    ``T = 0`` (smearing disabled) -> the integer hard-cutoff slice
    ``(C[:, :n], eps[:n])`` -- bit-identical to the pre-smearing wiring.

    ``T > 0`` -> the Mermin free-energy force blocks. The converged
    objective is the free energy ``A = E - T S`` (Mermin, Phys. Rev.
    137, A1441 (1965): the grand-potential functional is stationary at
    the self-consistent finite-T solution, Eqs. (1)-(4) and (9)-(10)).
    At fixed (T, N_s) the occupation- and mu-response terms in dA/dR
    vanish at self-consistency: for each spin channel

        dA/df_i = dE/df_i - T dS/df_i = eps_i - (eps_i - mu_s) = mu_s,

    (Fermi-Dirac stationarity: S = -sum_i [f ln f + (1-f) ln(1-f)],
    so -T dS/df_i = -(eps_i - mu_s) at f_i = 1/(1+exp((eps_i-mu_s)/T)))
    and the per-spin particle constraint sum_i df_i = 0 kills the
    remaining mu_s term. S has no explicit R-dependence at fixed T.
    What survives is exactly the Hellmann-Feynman + Pulay assembly at
    the FRACTIONAL-occupation density and energy-weighted density

        D_s = sum_i f_i c_i c_i^T,   W_s = sum_i f_i eps_i c_i c_i^T,

    the standard smeared-force statement (Marzari, Vanderbilt, De Vita
    & Payne, Phys. Rev. Lett. 82, 3296 (1999): with an entropy
    consistent with the occupation function, the free-energy
    derivatives carry no occupation-response terms).

    The T = 0 assemblers are occupation-agnostic once the orbital
    block is right: they build W from unit-occupation columns and the
    exchange weight quartically in the columns, so scaling column i by
    sqrt(f_i) evaluates

        C~ diag(eps) C~^T = W_s,        C~ C~^T = D_s,
        E_K-weight ~ sum_ij f_i f_j (ij|ji),

    which is exactly the smeared SCF's exchange energy
    (E_K,s = -1/2 sum_ij f_i f_j (ij|ji), bilinear in D_s). Columns
    with f_i <= 1e-14 are dropped (their W/K weight is below any
    gate). Fermi-Dirac (and Mermin) occupations are strictly in (0, 1)
    per spin, so sqrt is well-defined.  The analytic-gradient envelope
    remains Fermi-Dirac/Mermin-only because Methfessel-Paxton
    occupations can be negative and the non-Fermi-Dirac force paths do
    not yet have dedicated full-SCF validation, even though their
    generalized free-energy kernels are variational.
    """
    if np.asarray(occ).size == 0:
        return (
            np.asarray(C_f[:, :n_int], dtype=np.float64),
            np.asarray(eps_f[:n_int], dtype=np.float64),
        )
    occ = np.asarray(occ, dtype=float)
    keep = occ > 1e-14
    C_keep = np.asarray(C_f, dtype=np.float64)[:, keep]
    return (
        C_keep * np.sqrt(occ[keep])[None, :],
        np.asarray(eps_f, dtype=np.float64)[keep],
    )


def _reject_non_fermi_dirac_gradient(entry: str, smear_opts) -> None:
    """Fail closed until non-Fermi-Dirac analytic-force paths are tested.

    The Methfessel-Paxton and Marzari-Vanderbilt generalized entropies
    are consistent with their occupation kernels, so their reported
    free energies are variational. This GDF force assembly is still
    Fermi-Dirac/Mermin-only: MP occupations can be negative, while the
    fractional-block implementation takes square roots of occupations,
    and neither non-Fermi-Dirac path has a full-SCF force regression.
    """
    if smear_opts.enabled and smear_opts.flavor not in ("fermi-dirac", "mermin"):
        raise NotImplementedError(
            f"{entry}: compute_gradient supports Fermi-Dirac / Mermin "
            "smearing only -- non-Fermi-Dirac force paths are not yet "
            f"validated; run with flavor='fermi-dirac' or 'mermin', "
            f"or smearing_temperature=0 instead of {smear_opts.flavor!r}."
        )


def _gauge_lat_opts_ewald_3d(
    src: LatticeSumOptions,
    system: PeriodicSystem,
) -> LatticeSumOptions:
    """Return a clone of ``src`` with coulomb_method forced to EWALD_3D
    for the V_ne and nuclear-repulsion lattice sums on 3D-periodic
    systems. Matches PySCF's exxdiv='ewald' / CRYSTAL14 / vibe-qc's
    own EWALD_3D direct path. For dim<3 this is a passthrough.

    See ``periodic_rhf_gdf._gauge_lat_opts_for_v_ne_and_e_nuc`` for
    the original rationale (2026-05-13 gauge-regression fix).
    """
    if int(system.dim) != 3:
        return src
    dst = LatticeSumOptions()
    for attr in dir(src):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(src, attr)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            setattr(dst, attr, value)
        except Exception:
            pass
    dst.coulomb_method = CoulombMethod.EWALD_3D
    return dst


# Dimensionless tail-resolution threshold for lifting the dense-core Γ GDF
# parity hold: rsgdf_tail_ke_cutoff / zeta_max >= this ratio (zeta_max = the
# steepest AO primitive exponent). Calibrated on the exact P01 MgO/STO-3G Γ
# RHF cell (zeta_max = 299.24) against the PySCF GDF target
# -271.049457612534 Ha:
#   tail/zeta_max ~  5.3 (1600 Ha) -> -1.7 mHa
#   tail/zeta_max ~ 10.7 (3200 Ha) -> +1.3 uHa  (PySCF GDF<->RSDF ~ 1 uHa)
# so >= 10 marks the mesh regime where the tight-core AO-pair FT is resolved
# and absolute parity holds.
#
# Precision caveat (re-derived live 2026-07-16, PySCF 2.13.1): the target
# above is PySCF GDF at its DEFAULT cell precision (1e-8) -- re-run today it
# gives -271.0494591 (+1.4 uHa from the target), and the auto-sized tail
# (1.1 x 10 x zeta_max) reproduces it to -0.25 uHa
# (E = -271.0494578582, this fix's verification run). Tightening PySCF's own
# precision moves ITS answer by -163 uHa (1e-12: -271.0496211) -- i.e. the
# uHa parity claim is like-for-like at matched default thresholds; the
# absolute DF answer on a dense-core cell still carries a ~0.16 mHa shared
# truncation ladder (real-space rcut / integral thresholds) on both codes.
_RSGDF_PARITY_TAIL_RATIO = 10.0

# High-|G| tail completion is dominated by same-cell tight-core AO products.
# A conservative primitive-pair FT shell/cell screen keeps dense-core
# production tails from spending hours on exponentially dead diffuse/inter-cell
# blocks while leaving the exact unscreened builder available for direct
# validation.
_RSGDF_TAIL_PAIR_FT_SCREEN = 1.0e-10


def _gamma_dense_core_gdf_parity_held(
    system: PeriodicSystem,
    gdf_method: str,
    ao_basis: Optional[BasisSet] = None,
    tail_ke_cutoff: Optional[float] = None,
    rsgdf_ke_cutoff: float = 200.0,
) -> bool:
    """Return True for the Γ-only tight-core GDF parity-hold class.

    H2/vacuum-box and LiH/STO-3G Γ RSGDF have PySCF parity coverage. The
    release-paper P01 MgO/STO-3G Γ cell was different: at the default
    ``rsgdf_ke_cutoff`` the tight Mg/O core AO products are unresolved on
    the dense reciprocal mesh and the electronic terms sat ~0.5 Ha from PySCF
    while the Ewald nuclear term agreed. The same tight-core failure can appear
    in sparse molecular-limit boxes (for example OH/STO-3G in a 20 bohr cube):
    it is a basis reciprocal-resolution issue, not only a cell-density issue.
    The tight-core Γ class is tagged held -- UNLESS the high-|G| tail
    completion is requested with enough reciprocal support to resolve the
    steepest AO pair (``tail_ke_cutoff >= _RSGDF_PARITY_TAIL_RATIO x
    zeta_max``): with the tail, P01 reaches +1.3 uHa vs PySCF GDF at
    ``rsgdf_tail_ke_cutoff=3200`` (the validated fix; see the calibration
    note on ``_RSGDF_PARITY_TAIL_RATIO``).

    ``gdf_method='mdf'`` holds ONLY on the compact-dense-core class, not
    on the tight-basis/sparse-box class: MDF fits the steep core exactly
    in real space (Sun-Berkelbach 2017), so an unresolved-core reciprocal
    mesh is not its failure mode -- the vacuum-box heavy-atom gate
    (Ne/STO-3G/10-bohr, zeta_max=207) is VALIDATED at 0.14 mHa vs PySCF
    MDF (``tests/test_pbc_gdf_mdf.py``) and must not be tagged. What MDF
    does inherit is the Γ-only tight-ionic compact-cell instability of
    its compcell real-space Hartree lineage (G-GDF-001 probe,
    2026-07-29: MgO/STO-3G Γ mdf is non-convergent and trial-dependent
    at ke=40/60 (+1703/+2143/+418 Ha) and falsely converges at ke=80 to
    -1651 Ha vs PySCF MDF -271.0499) -- the compact class the Γ setup
    fails closed on (``_reject_gamma_dense_core_mdf``).
    """
    if str(gdf_method) not in {"rsgdf", "mdf"}:
        return False
    if int(system.dim) != 3 or not system.unit_cell:
        return False
    zeta_max = _max_ao_primitive_exponent(ao_basis) if ao_basis is not None else None
    tight_basis_requires_tail = (
        str(gdf_method) == "rsgdf"
        and zeta_max is not None
        and zeta_max > 0.0
        and _RSGDF_PARITY_TAIL_RATIO * zeta_max
        > float(rsgdf_ke_cutoff) + 1e-12
    )
    compact_dense_core = False
    if not tight_basis_requires_tail:
        z_max = max(int(atom.Z) for atom in system.unit_cell)
        if z_max < 8:
            return False
        try:
            cell_vol = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
        except Exception:
            return False
        cell_vol_per_atom = cell_vol / max(len(system.unit_cell), 1)
        compact_dense_core = cell_vol_per_atom < 250.0
        if not compact_dense_core:
            return False
    # Tail completion sized to the steepest AO primitive lifts the hold
    # (rsgdf only -- the mdf path has no tail plumbing).
    if (
        tail_ke_cutoff is not None
        and zeta_max is not None
        and str(gdf_method) == "rsgdf"
    ):
        if (
            zeta_max is not None
            and zeta_max > 0.0
            and float(tail_ke_cutoff) >= _RSGDF_PARITY_TAIL_RATIO * zeta_max
        ):
            return False
    return bool(tight_basis_requires_tail or compact_dense_core)


def _reject_legacy_gamma_gdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    entry: str,
) -> None:
    """Refuse an automatic untailed legacy fallback in the held class (#95).

    The fallback has no RSGDF tail, regardless of the requested builder or
    cutoff. Classify its actual untailed domain, including tight bases in
    sparse molecular boxes, before constructing any legacy SCF integrals.
    """
    if _gamma_dense_core_gdf_parity_held(system, "rsgdf", ao_basis=ao_basis):
        raise NotImplementedError(
            f"{entry}: legacy Gamma GDF fallback is unavailable for this "
            "tight-core basis/cell: it has no high-|G| tail correction and "
            "can return a wrong absolute energy (#95). The requested "
            "convergence or routing options are not supported by the pure "
            "Gamma route. Use an explicit kpoints=(1, 1, 1) with "
            "gdf_method='rsgdf' in run_periodic_job (kmesh=(1, 1, 1) in "
            "run_krhf_periodic_gdf) for the bulk SR/LR driver, or remove "
            "the conflicting options. An explicit tail cutoff cannot "
            "repair the legacy driver."
        )


def _max_ao_primitive_exponent(basis: BasisSet) -> Optional[float]:
    try:
        return max(float(max(shell.exponents)) for shell in basis.shells())
    except Exception:
        return None


def _auto_rsgdf_tail_ke_cutoff(
    system: PeriodicSystem,
    gdf_method: str,
    basis: BasisSet,
    tail_ke_cutoff: Optional[float],
    rsgdf_ke_cutoff: float = 200.0,
) -> Optional[float]:
    """Resolve the production default for tight-core Gamma RSGDF tails.

    ``None`` means production behavior: if this is the tight-core Gamma hold
    class, choose a tail sized past the calibrated parity threshold. The
    classifier is basis-driven so sparse molecular-limit boxes with O/Ne/Mg
    STO-3G cores get the same correction as compact ionic cells. Explicit
    numeric values are preserved, including ``0`` for fast diagnostic runs that
    intentionally keep the parity hold.
    """
    if tail_ke_cutoff is not None or str(gdf_method) != "rsgdf":
        return tail_ke_cutoff
    if not _gamma_dense_core_gdf_parity_held(
        system,
        gdf_method,
        basis,
        None,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
    ):
        return None
    zeta_max = _max_ao_primitive_exponent(basis)
    if zeta_max is None or zeta_max <= 0.0:
        return None
    return float(
        max(float(rsgdf_ke_cutoff), 1.1 * _RSGDF_PARITY_TAIL_RATIO * zeta_max)
    )


def _reject_dense_core_mdf(
    system: PeriodicSystem,
    gdf_method: str,
    ao_basis: Optional[BasisSet],
    entry: str,
) -> None:
    """Fail closed on the compact-dense-core MDF class (G-GDF-001).

    Still wrong on this class, but the cause is now measured rather than
    assumed (2026-08-03). The earlier text here said MDF "inherits the
    Γ-only tight-ionic compcell instability"; that was a hypothesis.

    **What was actually wrong, and is now fixed.** The aux 2c/3c lattice
    sums enumerated translations by ``|g| <= R_cut`` and then computed
    every shell pair at that ``g``. The physical separation is
    ``|R_P - R_Q - g|``, so the contributing translations form a ball
    centred on the intra-cell offset. On MgO (offset 6.89 bohr) that made
    the compensated Coulomb metric indefinite (-5.4e-05 against a largest
    eigenvalue of 20.8, though it is a Gram matrix) and dependent on where
    in the cell an atom sat (moving it by a lattice vector, i.e. the same
    crystal, changed the metric by 7.2e-02). MDF's Eq. 19 orthogonalisation
    divides by the square root of that metric, which is how the error
    reached +5.3e+05 Ha. Fixed in ``cpp/src/aux_eri.cpp``; invariance is
    now 1.2e-14 and pinned by
    ``tests/test_pbc_gdf_compcell.py::test_lattice_metric_is_translation_invariant``.

    **Why this still raises, and what has been RULED OUT.** The
    enumeration fix was necessary and not sufficient: post-fix MgO Γ mdf
    still lands at +4.7e+05 Ha.

    A second hypothesis -- that the remaining cause was the lattice
    cutoff, because ``pyscf_auto`` picks only 15.5 bohr for this
    compensated basis where the metric is still -2.0e-06 -- was
    **measured and REFUTED on 2026-08-03**. At a raw 28-bohr cutoff the
    compensated metric is fully positive semi-definite (+6.4e-12, residue
    6e-15 against a converged reference), and MDF is still wrong:
    ``-6.3e+04`` Ha, non-converged. The energy barely moves across
    cutoffs 20 / 24 / 28 (-62873 / -62873 / -63195 at
    ``mdf_ke_cutoff=40``). **The lattice cutoff is not the blocker.** Do
    not spend time re-tightening it.

    **The cause WAS the linear-dependence threshold, and that part is
    now FIXED** (diagnosed 2026-08-03, fixed 2026-08-14). Sweeping
    ``gdf_linear_dep_threshold`` on MgO at a 28-bohr cutoff,
    ``mdf_ke_cutoff=40``, against the PySCF MDF oracle
    ``-271.0499046``, under the pre-fix relative convention::

        1e-9 (default)  +467152 / -62874   trial-dependent
        1e-7             -337.802
        1e-5             -275.300
        1e-3             -271.146
        1e-2             -271.129

    The default kept a shelf of near-null ``J-tilde`` directions whose
    ``1/sqrt(lambda)`` amplification IS the divergence. This is the
    failure Sun 2017 Sec. II B warns about: projecting the PWs out of
    the Gaussians gives a highly singular matrix whose small
    eigenvectors must be removed.

    **Why those modes were noise, measured.** At the production rcut
    (``pyscf_auto``, ``rcut_precision`` 1e-8) the MgO ``J-tilde``
    spectrum carries up to **5.4e-06** absolute construction error
    against a converged 34-bohr reference. The smallest RETAINED
    eigenvalue at the old default was **2.9e-08, with its own error bar
    at 6.6e-08** -- the mode was smaller than its uncertainty, and
    Eq. 19 divides the Eq.-23 numerator by its square root
    (amplification ~6e+03). PySCF can afford its much tighter 1e-10
    because it builds the 2c metric at a dedicated far tighter precision
    (``precision_j2c``), exactly because 2c metric error propagates into
    the fitted tensor; vibe-qc builds ``J-tilde`` at the general
    lattice-sum precision, so its threshold must sit correspondingly
    higher. See ``aux_basis._MDF_DRESSED_METRIC_MIN_THRESHOLD``.

    **What landed.** The convention is harmonised to ABSOLUTE across all
    nine fitting-metric decompositions (PySCF's convention; MDF and
    compcell previously compared against ``linear_dep_thr * max_eig``),
    and the PW-dressed MDF metric carries a calibrated floor of 1e-3.
    The validated Ne gate improved from 0.064 mHa to **0.0024 mHa** vs
    the PySCF MDF oracle. Pinned by
    ``tests/test_pbc_gdf_mdf.py::test_metric_keep_mask_is_absolute_not_relative``,
    ``::test_mdf_dressed_metric_threshold_floor``,
    ``::test_mdf_linear_dep_threshold_ne_box_sweep``.

    **Why it STILL raises.** The divergence is gone but the accuracy is
    not there: the dense-core residual is ~92 mHa at the loosest useful
    threshold (3e-2). That is basis-set incompleteness error -- the
    threshold discards 81% of the aux space on this cell (189 -> 36
    modes) -- which is the other half of the Sun 2017 Sec. III
    trade-off, "a threshold too loose would increase the basis set
    incompleteness error". 92 mHa is not production accuracy, so the
    class stays gated (CLAUDE.md §7).

    **The remaining follow-up**, and the route to lifting this gate:
    build ``J-tilde`` accurately enough to support a PySCF-like
    threshold, i.e. vibe-qc's analogue of ``precision_j2c``. Related
    measured finding: ``precision=1e-8`` does not deliver 1e-8 in the
    compensated metric -- residues at the auto rcut are 2.4e-07 (MgO),
    7.1e-06 (LiH rocksalt), 1.9e-08 (H2 box), essentially independent of
    the compensating exponent eta, consistent with accumulation over the
    ~R^3 lattice terms rather than single-pair amplitude decay.

    Callers who want this class anyway can pass a looser
    ``gdf_linear_dep_threshold`` (~3e-2) knowingly.

    The vacuum-box heavy-atom MDF envelope (Ne/STO-3G/10-bohr, validated
    at 0.14 mHa vs PySCF MDF) is NOT this class and stays open; it has
    zero intra-cell offset and was bit-unchanged by the enumeration fix.
    """
    if str(gdf_method) != "mdf":
        return
    if not _gamma_dense_core_gdf_parity_held(system, "mdf", ao_basis):
        return
    raise NotImplementedError(
        f"{entry}: gdf_method='mdf' on a compact dense-core cell "
        "(Z>=8 in <250 bohr^3/atom) is not supported. The PW-dressed "
        "metric (Sun 2017 Eq. 20) is too ill-conditioned on this class "
        "for the auxiliary basis to survive the linear-dependence cut: "
        "the threshold that removes the divergence also discards ~81% "
        "of the aux space, leaving ~92 mHa of incompleteness error vs "
        "PySCF MDF on MgO/STO-3G (G-GDF-001; divergence fixed "
        "2026-08-14, accuracy still short). Use the default "
        "gdf_method='rsgdf' (the Gamma driver auto-sizes the high-|G| "
        "tail to reach PySCF parity on this class; multi-k accepts "
        "rsgdf_tail_ke_cutoff), or a vacuum-padded / "
        "pseudopotential-class cell for MDF."
    )


def _warn_rsgdf_gradient_conditioning(
    result,
    cache,
    plog: ProgressLogger,
    driver: str,
    warn_tol: float = 1.0e-8,
) -> None:
    """Attach + surface the fit-response conditioning estimate.

    See :func:`vibeqc.periodic_gdf_gradient.rsgdf_fit_response_conditioning`
    for the validated formula and its measurements.
    """
    from .periodic_gdf_gradient import rsgdf_fit_response_conditioning

    est = rsgdf_fit_response_conditioning(cache)
    if hasattr(result, "gradient_conditioning"):
        result.gradient_conditioning = float(est)
    if est > float(warn_tol):
        # Informational only. The dense-core class this indicator flags
        # was once suspected of a systematic gradient error; the 2026-07-29
        # e_nuc Ewald-gauge fix showed the analytic gradient exact on this
        # class (MgO/STO-3G full-SCF FD gate: |analytic - FD| < 5e-9
        # Ha/bohr across h = 1e-4 .. 8e-4). The indicator remains useful
        # as a fit-response sensitivity diagnostic near the linear-dep
        # threshold.
        plog.info(
            f"  {driver}: dense-core fit class (near-threshold metric "
            f"modes; fit-response conditioning indicator {est:.1e})."
        )


def _warn_gamma_dense_core_gdf_parity_hold(
    system: PeriodicSystem,
    gdf_method: str,
    plog: ProgressLogger,
    ao_basis: Optional[BasisSet] = None,
    tail_ke_cutoff: Optional[float] = None,
    rsgdf_ke_cutoff: float = 200.0,
) -> bool:
    # mdf on the compact-dense-core class raises here (all three Γ
    # drivers call this helper before the cderi build); every other mdf
    # cell is NOT held (see the classifier's mdf note), so past this
    # point the hold/warn text below is rsgdf-only.
    _reject_dense_core_mdf(system, gdf_method, ao_basis, "run_pbc_gdf")
    if not _gamma_dense_core_gdf_parity_held(
        system,
        gdf_method,
        ao_basis,
        tail_ke_cutoff,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
    ):
        return False
    msg = (
        "run_pbc_gdf: Gamma-only "
        f"{gdf_method} absolute-energy parity is HELD for tight-core "
        "basis/cell combinations at this reciprocal-mesh resolution: the "
        "tight core AO-pair FT is unresolved (the P01 MgO/STO-3G audit "
        "measured a ~0.5 Ha electronic offset vs PySCF at the 200 Ha default; "
        "OH/STO-3G molecular-limit boxes show the same basis-resolution "
        "failure while the Ewald nuclear term matches). Remediation: pass "
        "rsgdf_tail_ke_cutoff >= 10 x (steepest AO primitive exponent) "
        "to enable the high-|G| tail completion -- P01 reaches ~1 uHa "
        "parity at rsgdf_tail_ke_cutoff=3200 -- or use a separately "
        "validated multi-k route."
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    plog.info("  WARNING: " + msg)
    return True


def _warn_gamma_compcell_tight_ionic_cell(
    system: PeriodicSystem,
    gdf_method: str,
    driver: str,
) -> bool:
    """Warn when the Γ-only compcell Hartree cannot be trusted.

    Γ-only compcell GDF is not sufficient for tight cells with heavy
    atoms (Z > 1) where AO-pair images overlap: the Γ-only Hartree
    (G=0 dropped) cannot resolve the overlap between periodic images
    and the SCF converges to a non-physical fixed point (LiH FCC
    primitive: +579.8 Ha at HF vs PySCF -8 Ha; the energy-sanity
    guard rejects it downstream). The multi-k path is the production
    route for these cells.

    Heuristic: Z > 1 in a cell < 500 bohr³ total is tight enough that
    AO-pair images overlap. H₂ in a 12-bohr box (1728 bohr³) is safe;
    LiH FCC primitive (115 bohr³) is not.
    """
    if int(system.dim) != 3 or str(gdf_method) != "compcell":
        return False
    if not system.unit_cell:
        return False
    z_max = max(atom.Z for atom in system.unit_cell)
    cell_vol = float(abs(np.linalg.det(np.asarray(system.lattice))))
    if z_max <= 1 or cell_vol >= 500.0:
        return False
    warnings.warn(
        f"{driver}: Γ-only compcell GDF may be unreliable "
        f"for this tight ionic cell (max Z={z_max}, cell volume "
        f"{cell_vol:.0f} bohr³). The Γ-only Hartree (G=0 "
        "dropped) cannot fully resolve AO-pair overlap between "
        "periodic images. Prefer the multi-k path: "
        "vibeqc.run_krhf_periodic_gdf(..., use_compcell=True). "
        "Validated at µHa parity on LiH FCC at kmesh=(2,2,2).",
        stacklevel=3,
    )
    return True


def _gdf_backend_with_parity_hold(backend: str, parity_held: bool) -> str:
    if not parity_held or "+PARITY_HELD" in backend:
        return backend
    return f"{backend}+PARITY_HELD"


def _build_j_from_lpq(Lpq: np.ndarray, D: np.ndarray) -> np.ndarray:
    from .aux_basis import _build_coulomb_from_diagonal_factors

    # This Gamma driver represents a real density and potential. The shared
    # contraction also accepts complex auxiliary fitting coordinates.
    return np.real(_build_coulomb_from_diagonal_factors([Lpq], [D], [1.0])[0])


def _build_k_from_lpq(
    Lpq: np.ndarray, D: np.ndarray, *, workspace_byte_cap: int = 32 * 1024**2,
) -> np.ndarray:
    from .aux_basis import _accumulate_exchange_from_factors

    K = np.zeros(np.shape(D), dtype=np.result_type(Lpq, D))
    _accumulate_exchange_from_factors(K, Lpq, D, 1.0, workspace_byte_cap)
    K = np.real(K)
    return 0.5 * (K + K.T)


class _PbcGdfGammaSetup(NamedTuple):
    """Shared Γ-GDF setup outputs for ``run_pbc_gdf_{rhf,uhf,uks}``."""

    S: np.ndarray
    Hcore: np.ndarray
    X: np.ndarray
    n_kept: int
    Lpq: np.ndarray
    aux: BasisSet
    madelung: float
    e_nuc: float
    gauge_lat_opts: LatticeSumOptions
    compcell_fit_state: Optional[_CompcellFitState]
    fit_cutoff_2c: float
    fit_cutoff_3c: float
    range_separated_cache: object = None
    oneel_lat_opts: Optional[LatticeSumOptions] = None


def _pbc_gdf_gamma_setup(
    system: PeriodicSystem,
    basis: BasisSet,
    lat_opts: LatticeSumOptions,
    *,
    aux_name: str,
    aux_drop_eta: float,
    exxdiv: PBCExxDiv,
    gdf_method: str,
    compcell_eta: float,
    apply_aft_correction: bool,
    aft_precision: float,
    aft_ft_convention: str,
    rsgdf_ke_cutoff: float,
    rsgdf_tail_ke_cutoff: Optional[float],
    mdf_ke_cutoff: float,
    rcut_strategy: Optional[object],
    rcut_precision: float,
    gdf_linear_dep_threshold: float,
    linear_dep_threshold: float,
    retain_compcell_fit_state: bool,
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    scf_options=None,
    open_shell: bool = False,
    fit_screen_threshold: float = 0.0,
    plog: ProgressLogger,
) -> "_PbcGdfGammaSetup":
    """Build Gamma one-electron terms and the shared bulk fitting source.

    The SR/LR fit uses the same physical auxiliary basis, zero-mode
    convention, cutoff planner and cache reservation as the multi-k path.
    V_ne streams its Fourier panels independently. The returned fit cache
    owns the source parameters used again by analytic derivatives.
    """
    if float(fit_screen_threshold) < 0.0:
        raise ValueError(
            "pbc_gdf: fit_screen_threshold must be >= 0; "
            f"got {fit_screen_threshold}"
        )
    if float(fit_screen_threshold) > 0.0 and gdf_method != "rsgdf":
        # Loud, not silent: the Schwarz fit screen lives in the rsgdf
        # builder; a threshold on compcell/mdf would be ignored.
        raise NotImplementedError(
            "pbc_gdf: fit_screen_threshold is implemented for "
            f"gdf_method='rsgdf' only (got {gdf_method!r})."
        )
    # V_ne and the nuclear-nuclear repulsion are forced through the
    # Ewald-3D gauge for 3D-periodic systems so they match PySCF's
    # exxdiv='ewald' convention. ``lat_opts.coulomb_method`` continues
    # to control J/K routing (compcell GDF here).
    gauge_lat_opts = _gauge_lat_opts_ewald_3d(lat_opts, system)
    if int(system.dim) == 3 and gauge_lat_opts is not lat_opts:
        plog.info(
            "V_ne / e_nuc gauge: Ewald-3D "
            "(forced for 3D-periodic systems regardless of "
            f"lat_opts.coulomb_method={lat_opts.coulomb_method!r})"
        )
    oneel_lat_opts = lat_opts
    if gdf_method == "rsgdf" and int(system.dim) == 3:
        from .periodic_k_gdf import _oneel_lattice_opts, _preflight_gdf_oneel_memory

        oneel_lat_opts = _oneel_lattice_opts(
            system, basis, lat_opts, rcut_strategy=rcut_strategy,
            k_points_cart=np.zeros((1, 3)), plog=plog,
        )
        gauge_lat_opts = _gauge_lat_opts_ewald_3d(oneel_lat_opts, system)
        _preflight_gdf_oneel_memory(system, basis, oneel_lat_opts)
    with plog.stage(
        "integrals_lattice",
        detail=f"S/T/V at cutoff {oneel_lat_opts.cutoff_bohr:.2f} bohr",
    ):
        S_lat = compute_overlap_lattice(basis, system, oneel_lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, oneel_lat_opts)
        # Nuclear attraction has its own Ewald split. Stream the AO
        # Fourier panels so setup never retains an AO-by-AO-by-G bundle.
        if gdf_method == "rsgdf" and int(system.dim) == 3:
            from .periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma

            V = compute_v_ne_ewald_3d_ft_gamma(
                basis, system, gauge_lat_opts,
                ke_cutoff=float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0")),
                stream_pair_ft=True,
            )
            V_lat = None  # Γ-only path: V already bloch-summed
        else:
            V_lat = compute_nuclear_lattice_dispatch(basis, system, gauge_lat_opts)
            V = None

    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    if V is None:
        V = np.real(bloch_sum(V_lat, k_gamma))
    Hcore = 0.5 * ((T + V) + (T + V).T)
    S = 0.5 * (S + S.T)

    scf_preflight_overlap_check(S, plog=plog, label="S(Γ)", basis=basis)
    X, n_kept = _canonical_orthogonalizer(S, linear_dep_threshold)

    mol = system.unit_cell_molecule()
    with plog.stage("aux_basis", detail=aux_name):
        aux = make_aux_basis_set(
            mol,
            aux_name=aux_name,
            drop_eta=float(aux_drop_eta),
        )
    plog.info(f"aux basis: {aux_name}  ({aux.nbasis} BFs / {aux.nshells} shells)")
    compcell_fit_state = None
    range_separated_cache = None
    fit_cutoff_2c = float("nan")
    fit_cutoff_3c = float("nan")
    if gdf_method == "rsgdf":
        from .periodic_k_gdf import _build_scf_range_separated_lpq_cache

        with plog.stage("rsgdf_lpq", detail=f"SR/LR fit, aux={aux.nbasis}"):
            range_separated_cache = _build_scf_range_separated_lpq_cache(
                system, basis, aux, np.zeros((1, 3)), True,
                omega=rsgdf_omega, raw_integral_error=rsgdf_g_precision,
                ke_cutoff=rsgdf_ke_cutoff,
                linear_dep_thr=gdf_linear_dep_threshold, lat_opts=lat_opts,
                fit_screen_threshold=fit_screen_threshold, options=scf_options,
                open_shell=open_shell, progress=plog,
            )
            Lpq = range_separated_cache[(0, 0)]
            fit_cutoff_2c = range_separated_cache.source_parameters[2]
            fit_cutoff_3c = range_separated_cache.source_parameters[1]
    elif gdf_method == "compcell":
        with plog.stage(
            "compcell_lpq",
            detail=(
                f"compcell aux={aux.nbasis}, eta={compcell_eta:g}, "
                f"fit_thr={gdf_linear_dep_threshold:.1e}"
            ),
        ):
            state = _build_lpq_compcell_state(
                system,
                basis,
                aux,
                molecule=mol,
                lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                compcell_eta=float(compcell_eta),
                apply_aft_correction=bool(apply_aft_correction),
                aft_precision=float(aft_precision),
                aft_ft_convention=str(aft_ft_convention),
                rcut_strategy=rcut_strategy,
                rcut_precision=float(rcut_precision),
            )
            Lpq = state.Lpq
            fit_cutoff_2c = float(state.lat_opts_2c.cutoff_bohr)
            fit_cutoff_3c = float(state.lat_opts_3c.cutoff_bohr)
            if retain_compcell_fit_state:
                compcell_fit_state = state
    elif gdf_method == "mdf":
        with plog.stage(
            "mdf_lpq",
            detail=(
                f"mdf aux={aux.nbasis}, eta={compcell_eta:g}, "
                f"mdf_ke={mdf_ke_cutoff:g} Ha, fit_thr={gdf_linear_dep_threshold:.1e}"
            ),
        ):
            # Mixed Density Fitting (Sun-Berkelbach 2017): compensated
            # Gaussian fit (steep cores exact) + plane-wave residual. The
            # builder returns the two parts separately; combine into one
            # complex cderi [L_gauss (real->complex); cderi_pw] so the
            # (complex-aware) _build_{j,k}_from_lpq handle both uniformly
            # via W = S_L cderi[L].conj(cderi[L]).
            _mdf = build_lpq_mdf(
                system,
                basis,
                aux,
                molecule=mol,
                lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                compcell_eta=float(compcell_eta),
                mdf_ke_cutoff=float(mdf_ke_cutoff),
                rcut_strategy=rcut_strategy,
                rcut_precision=float(rcut_precision),
            )
            Lpq = np.concatenate(
                [_mdf.L_gauss.astype(np.complex128), _mdf.cderi_pw], axis=0
            )
            plog.info(
                f"MDF cderi: {_mdf.n_kept_gauss} Gaussian + {_mdf.n_pw} PW fit vectors"
            )
    else:
        raise ValueError(
            f"_pbc_gdf_gamma_setup: gdf_method must be 'compcell', 'rsgdf', "
            f"or 'mdf'; got {gdf_method!r}"
        )
    plog.info(
        f"Lpq: {Lpq.shape[0]} fit vectors, "
        f"shape=({Lpq.shape[0]}, {Lpq.shape[1]}, {Lpq.shape[2]})"
    )

    madelung = madelung_constant_for_cell(system) if exxdiv is PBCExxDiv.EWALD else 0.0
    if exxdiv is PBCExxDiv.EWALD:
        plog.info(f"exxdiv='ewald': K-shift xi = {madelung:.6f} / bohr (alpha_M/L)")

    if int(system.dim) == 3:
        # Converged Ewald nuclear energy — the exact function the analytic
        # GDF gradient differentiates (ewald_nuclear_repulsion_gradient).
        # nuclear_repulsion_per_cell(EWALD_3D) truncates its real-space sum
        # at lat_opts.nuclear_cutoff_bohr; on dense ionic cells that is
        # unconverged at the 1e-5 Ha level and carries a spurious geometry
        # dependence (MgO primitive/STO-3G at 18 bohr: 2.1e-5 Ha energy
        # error, +2.5e-5 Ha/bohr FD-slope artefact) — the source of the
        # long-standing dense-core analytic-vs-FD gradient discrepancy.
        e_nuc = float(ewald_nuclear_repulsion(system))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(system, gauge_lat_opts))
    plog.info(f"E_nuc = {e_nuc:.10f} Ha")

    return _PbcGdfGammaSetup(
        S=S,
        Hcore=Hcore,
        X=X,
        n_kept=int(n_kept),
        Lpq=Lpq,
        aux=aux,
        madelung=float(madelung),
        e_nuc=float(e_nuc),
        gauge_lat_opts=gauge_lat_opts,
        compcell_fit_state=compcell_fit_state,
        fit_cutoff_2c=fit_cutoff_2c,
        fit_cutoff_3c=fit_cutoff_3c,
        range_separated_cache=range_separated_cache,
        oneel_lat_opts=oneel_lat_opts,
    )



def _refuse_ecp_options(opts, driver: str, *, system=None) -> None:
    """Fail closed: these drivers build Hcore from bare nuclear charges
    and fill the physical electron count, so ECP options set on ``opts``
    would be dropped silently (#88). run_periodic_job routes ECP-bearing
    cells to the k-point GDF drivers at a one-point mesh; direct callers
    must do the same."""
    from .guess import _has_guess_ecp_metadata

    molecule = system.unit_cell_molecule() if isinstance(system, PeriodicSystem) else None
    if _has_guess_ecp_metadata(opts, molecule=molecule):
        raise NotImplementedError(
            f"{driver}: this driver does not apply a periodic ECP "
            "(ECP fields are set on the options). Use "
            "vibeqc.run_krhf_periodic_gdf / run_krks_periodic_gdf / "
            "run_kuhf_periodic_gdf / run_kuks_periodic_gdf at kpoints=(1, 1, 1), "
            "which run_periodic_job selects for an ECP-bearing cell."
        )

def run_pbc_gdf_rhf(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
    *,
    functional: Optional[str] = None,
    kmesh: Sequence[int] = (1, 1, 1),
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    exxdiv: Union[PBCExxDiv, str] = PBCExxDiv.EWALD,
    gdf_method: str = "compcell",
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_precision: float = 1e-10,
    aft_ft_convention: str = "libint",
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    mdf_ke_cutoff: float = 40.0,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    fit_screen_threshold: float = 0.0,
    compute_gradient: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> PBCGDFResult:
    """Closed-shell periodic RHF/RKS via compcell GDF.

    Γ-only in this initial landing. For ``kmesh != (1, 1, 1)``,
    raises ``NotImplementedError`` -- the multi-k path is the next
    milestone (see ``run_krhf_periodic_gdf`` which will pull
    compcell Lpq + exxdiv shift in once this Γ path is validated).

    Parameters
    ----------
    system, basis
        Periodic system and orbital basis (same conventions as the
        rest of the periodic stack).
    options
        :class:`PeriodicRHFOptions`. If ``None``, defaults are used.
        Relevant fields: ``max_iter``, ``conv_tol_energy``,
        ``conv_tol_grad``, ``use_diis``, ``diis_start_iter``,
        ``diis_subspace_size``, ``damping``, ``initial_guess``,
        ``lattice_opts``.
    functional
        Optional libxc functional name (e.g. ``"pbe"``, ``"pbe0"``) --
        the closed-shell KS sibling of :func:`run_pbc_gdf_uks`'s
        ``functional``. ``None`` falls back to ``options.functional``
        and, if that is empty too, the driver runs as plain RHF. For
        hybrids the exact-exchange channel is the a_x-scaled GDF ``K``
        with the same ``exxdiv='ewald'`` Madelung shift as RHF (the
        identical convention to :func:`apply_exxdiv_ewald_to_K` on the
        multi-k path); pure DFT (a_x = 0) skips ``K`` entirely. KS runs
        with ``options=None`` default to :class:`PeriodicKSOptions`,
        whose ``use_periodic_becke=True`` selects the periodic-Becke
        grid + Γ-torus density pairing (b3f74aa9). Range-separated
        hybrids raise ``NotImplementedError`` (no erf-attenuated Lpq).
    kmesh
        ``(n1, n2, n3)`` Monkhorst-Pack mesh. Γ-only ``(1,1,1)`` in
        this landing.
    aux_basis
        Aux basis name. Defaults to ``default_aux_for(basis.name)``.
    aux_drop_eta
        Per-primitive aux cull threshold (currently a no-op in
        ``make_aux_basis_set``).
    exxdiv
        :class:`PBCExxDiv` or its string value (``'ewald'`` /
        ``'none'``). Default ``EWALD`` (PySCF parity).
    gdf_method
        Lpq builder selector -- ``'compcell'`` (default) for the Sun-2017
        compensated-charge path
        (:func:`vibeqc.aux_basis.build_lpq_compcell`), or ``'rsgdf'`` for
        the historical all-reciprocal Coulomb source
        (:func:`vibeqc.aux_basis.build_lpq_native_fft`). Despite its name,
        the current production source does not evaluate real-space SR
        integrals. The replacement SR/LR source is under validation; see
        ``docs/design_native_gdf.md``. ``compcell_eta``,
        ``apply_aft_correction`` and ``aft_*`` select the compcell route's
        construction and do not affect the all-reciprocal source.
    compcell_eta
        Smooth-Gaussian exponent for the compensating basis (default
        1.0; with AFT on, the result is largely η-independent). See
        :func:`vibeqc.aux_basis.build_lpq_compcell`.
        Only used when ``gdf_method='compcell'``.
    rsgdf_omega
        Legacy split parameter (default ``0.4``). Retained in the API;
        the current all-reciprocal production source does not use it.
    rsgdf_g_precision
        Legacy LR precision (default ``1e-10``). Retained in the API;
        the current production source uses ``rsgdf_ke_cutoff`` and its
        optional tail instead.
    linear_dep_threshold
        Overlap eigenvalue floor for canonical orthogonalisation.
    gdf_linear_dep_threshold
        Aux metric eigenvalue floor for the GDF fit
        (eigendecomposition-with-threshold).
    fit_screen_threshold
        Cauchy-Schwarz screen on the three-centre fit (rsgdf only;
        ``handovers/HANDOVER_GDF_FIT_SCREENING.md``). ``0.0`` (default)
        keeps the historical dense build with the shared V_ne/cderi
        pair-FT bundle. Positive values select the memory-lean mode:
        the fit build drops AO pairs whose Schwarz bound is below the
        threshold and sweeps the G-mesh in chunks, and the V_ne FT
        streams its own chunked pair FT -- the dense
        ``(n_ao, n_ao, n_G)`` pair-FT tensor is never materialised.
        ``1e-10`` reproduces the unscreened energy to < 1e-8 Ha/cell on
        the pinned controls. Raises for compcell/mdf (the screen lives
        in the rsgdf builder). Same option as on the multi-k
        ``run_krhf/krks/kuhf_periodic_gdf`` drivers.
    compute_gradient
        If ``True``, compute the analytic GDF gradient after SCF convergence.
        This preview is restricted to 3D Gamma-point RHF with
        ``gdf_method='compcell'``, ``apply_aft_correction=False``, and the
        analytical-FT ``V_ne`` backend. Unsupported routes fail before SCF.
    progress, verbose
        Live progress logging passthrough.

    Returns
    -------
    PBCGDFResult
    """
    _refuse_ecp_options(options, "run_pbc_gdf_rhf", system=system)
    from .kpoints import _integer_counts

    kmesh = tuple(_integer_counts(kmesh, name="Gamma GDF kmesh"))
    if kmesh != (1, 1, 1):
        raise NotImplementedError(
            "run_pbc_gdf_rhf: only kmesh=(1,1,1) (Γ-only) is implemented "
            f"in this driver; got kmesh={kmesh}. For multi-k periodic "
            "SCF, see `vibeqc.run_krhf_periodic_gdf` (uses the legacy "
            "bare-aux Lpq path; works but has the bare-aux divergence on "
            "diffuse JKfit auxiliaries). Multi-k compcell GDF is a "
            "v0.9.0 milestone (will patch run_krhf_periodic_gdf to use "
            "build_lpq_compcell + apply_exxdiv_ewald_to_K)."
        )
    if isinstance(exxdiv, str):
        exxdiv = PBCExxDiv(exxdiv)

    gdf_rcut_strategy_value: Optional[str] = None
    if str(gdf_method) in {"compcell", "mdf"} and rcut_strategy is not None:
        from .lattice_screening import RcutStrategy

        if not isinstance(rcut_strategy, RcutStrategy):
            try:
                rcut_strategy = RcutStrategy(rcut_strategy)
            except Exception as exc:
                raise ValueError(
                    "run_pbc_gdf_rhf: rcut_strategy must be a "
                    f"RcutStrategy or its string value; got "
                    f"{rcut_strategy!r}: {exc}"
                ) from exc
        gdf_rcut_strategy_value = rcut_strategy.value

    if options is None:
        # KS run (functional given here or via options) -> PeriodicKSOptions,
        # so the periodic-Becke grid + torus density pairing engages by
        # default (b3f74aa9); plain-RHF fallback keeps PeriodicRHFOptions.
        from ._vibeqc_core import PeriodicKSOptions

        options = PeriodicKSOptions() if functional else PeriodicRHFOptions()
    opts = options
    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)
    guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_pbc_gdf_rhf",
        supported=periodic_guess_capabilities('gdf', 'RHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
    )

    func_name = functional or str(getattr(opts, "functional", "") or "")
    is_ks = bool(func_name)
    func = None
    alpha = 1.0
    if is_ks:
        from ._vibeqc_core import Functional

        func = Functional(func_name, 1)
        if bool(getattr(func, "is_external", False)):
            raise NotImplementedError(
                "run_pbc_gdf_rhf: the direct Gamma GDF fast path cannot "
                "supply the difference-closed periodic AO-density domain "
                "required by a full-grid external XC functional. Use "
                "run_krks_periodic_gdf(..., kmesh=(1, 1, 1)); the generic "
                "finite-torus driver owns external-XC Gamma and multi-k "
                "evaluation."
            )
        reject_periodic_gdf_unsupported_functional(
            func, where="run_pbc_gdf_rhf"
        )
        alpha = float(func.hf_exchange_fraction)
        if bool(getattr(func, "is_range_separated", False)):
            raise NotImplementedError(
                "run_pbc_gdf_rhf: range-separated hybrids need an "
                f"erf-attenuated Lpq cderi, which the Γ GDF path does not "
                f"build yet (functional={func_name!r}). Use a global hybrid "
                "(e.g. pbe0) or a pure functional."
            )
    # KS compute_gradient (G-PBC-002 milestone 2): the closed-shell KS
    # gradient composes the general-alpha compcell J/K machinery with the
    # lattice-summed XC Pulay primitive, on the same explicit
    # compcell/AFT-off preview envelope the RHF wiring already requires
    # (the guards below apply to KS and RHF alike).
    if compute_gradient and int(system.dim) != 3:
        raise NotImplementedError(
            "run_pbc_gdf_rhf: compute_gradient currently supports 3D "
            "periodic systems only."
        )
    if compute_gradient and str(gdf_method) not in ("compcell", "rsgdf"):
        raise NotImplementedError(
            "run_pbc_gdf_rhf: compute_gradient currently supports "
            "gdf_method='compcell' and 'rsgdf'; the MDF fit derivative "
            "is not implemented."
        )
    # Schwarz-screened rsgdf fits are differentiable since 2026-07-30:
    # the gradient cache mirrors the SCF's pair mask and the J/K weight
    # algebra zeroes the masked-pair derivative weights (exactly zero at
    # fixed mask).
    # AFT-corrected fits are differentiable since G-PBC-002 milestone 3b
    # (the 2c/3c AFT centre derivatives); compute_gradient supports both
    # apply_aft_correction settings on the compcell path.
    requested_v_ne_backend = os.environ.get(
        "VIBEQC_VNE_EWALD3D_BACKEND", "analytic_ft"
    ).lower()
    v_ne_backend = (
        "grid"
        if str(gdf_method) != "rsgdf" and requested_v_ne_backend == "grid"
        else "analytic_ft"
    )
    if compute_gradient and v_ne_backend != "analytic_ft":
        raise NotImplementedError(
            "run_pbc_gdf_rhf: compute_gradient requires the analytical "
            "FT Ewald V_ne backend; unset VIBEQC_VNE_EWALD3D_BACKEND=grid."
        )
    # Nuclear and fitting meshes have separate controls, shared with multi-k.
    v_ne_ke_cutoff = float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0"))

    # Hardening: detect obvious misuse before launching expensive
    # integrals. Each check fails fast with an actionable message.
    if int(system.dim) != 3:
        raise NotImplementedError(
            f"run_pbc_gdf_rhf: only dim=3 (full 3D periodic) is "
            f"implemented; got system.dim={system.dim}. For molecular-"
            "in-vacuum-box (dim=3 with large vacuum spacing) the path "
            "works fine -- make sure system.dim is set to 3."
        )

    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "run_pbc_gdf_rhf: closed-shell RHF requires even electron "
            f"count; got {n_elec}. For open-shell systems, UHF/UKS "
            "compcell GDF is a v0.9.0 milestone."
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_pbc_gdf_rhf: closed-shell RHF requires multiplicity=1; "
            f"got {system.multiplicity}. For open-shell systems, UHF/UKS "
            "compcell GDF is a v0.9.0 milestone."
        )
    # Charge-neutrality check: compcell construction assumes the unit
    # cell is charge-neutral. For charged cells (defect calculations,
    # isolated ions in vacuum boxes), the Madelung cancellation between
    # nuclear and electronic G=0 contributions breaks and the SCF
    # energy includes a divergent G=0 self-image term.
    Q_nuc = float(sum(atom.Z for atom in system.unit_cell))
    if abs(Q_nuc - n_elec) > 0.5:
        raise ValueError(
            "run_pbc_gdf_rhf: unit cell is not charge-neutral "
            f"(Q_nuclei={Q_nuc:.0f}, n_electrons={n_elec}; net charge "
            f"= {Q_nuc - n_elec:+.0f}). The compcell construction + "
            "exxdiv='ewald' shift both assume neutrality; charged "
            "cells get a divergent G=0 contribution that this driver "
            "does NOT correct. For neutral cells with an explicit "
            "charge state, use the molecular limit driver "
            "(`vibeqc.run_rhf`) in a large vacuum box, or wait for "
            "v0.9.0's charged-cell support."
        )

    # ---- Ionic-system guard: \u0393-only compcell is not sufficient
    # for tight cells with heavy atoms (Z > 1) where AO-pair images
    # overlap.  The multi-k path is the production route for these.
    _warn_gamma_compcell_tight_ionic_cell(system, gdf_method, "run_pbc_gdf_rhf")

    if not apply_aft_correction and gdf_method == "compcell":
        # The AFT long-range correction is now ON by default (v0.12.0).
        # Disabling it reverts to the bare compcell path which requires
        # manual \u03b7 tuning and diverges for tight ionic cells.
        warnings.warn(
            "run_pbc_gdf_rhf: apply_aft_correction=False disables the "
            "AFT long-range correction (ON by default since v0.12.0). "
            "Without AFT, \u03b7 is a tuning knob and tight ionic cells "
            "diverge. Prefer the default (AFT on, \u03b7=1.0, "
            "rcut_strategy='pyscf_auto') for production use.",
            stacklevel=2,
        )
    if gdf_method == "compcell" and aft_ft_convention == "libcint":
        warnings.warn(
            "run_pbc_gdf_rhf: aft_ft_convention='libcint' uses PySCF's "
            "FT convention, which is inconsistent with vibe-qc's own "
            "bare lattice sum convention (libint). This produces ~+200 mHa "
            "errors. Use the default aft_ft_convention='libint' instead.",
            stacklevel=2,
        )

    n_occ = n_elec // 2

    aux_name = aux_basis or default_aux_for(basis.name)
    if gdf_method == "rsgdf" and rsgdf_tail_ke_cutoff is not None:
        warnings.warn(
            "rsgdf_tail_ke_cutoff is obsolete for the SR/LR fit; "
            "use rsgdf_g_precision to control the raw integral error",
            DeprecationWarning, stacklevel=2,
        )
    rsgdf_tail_ke_cutoff = None
    # Tailed rsgdf gradients were held here until 2026-07-29: the
    # dense-core FD "inconsistency" (~2.5e-5 Ha/bohr, tail-independent)
    # turned out to be an e_nuc truncation artefact in the ENERGY
    # (nuclear_repulsion_per_cell at nuclear_cutoff_bohr), not a
    # gradient bug -- see _pbc_gdf_gamma_setup. With the converged
    # Ewald e_nuc the untailed and tailed MgO/STO-3G full-SCF FD gates
    # both pass at < 1e-8 Ha/bohr, so the hold is lifted.

    plog.info(
        f"PBC-GDF {'RKS ' + func_name if is_ks else 'RHF'} / "
        f"aux={aux_name}, exxdiv={exxdiv.value}, "
        + (f"alpha={alpha:g}, " if is_ks else "")
        + f"eta={compcell_eta:g}, kmesh={kmesh}, "
        f"cutoff={lat_opts.cutoff_bohr:.2f} bohr"
    )
    if rsgdf_tail_ke_cutoff is not None and gdf_method == "rsgdf":
        plog.info(
            "RSGDF high-|G| tail cutoff: "
            f"{float(rsgdf_tail_ke_cutoff):g} Ha"
        )
    parity_held = gdf_method != "rsgdf" and _warn_gamma_dense_core_gdf_parity_hold(
        system,
        gdf_method,
        plog,
        ao_basis=basis,
        tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
    )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    plog.info(
        "lattice cells: "
        f"one-electron/GDF cutoff -> "
        f"{len(direct_lattice_cells(system, lat_opts.cutoff_bohr))}, "
        "nuclear cutoff -> "
        f"{len(direct_lattice_cells(system, lat_opts.nuclear_cutoff_bohr))}"
    )

    # ---- One-electron integrals + Lpq cderi + exxdiv (shared setup) ---
    setup = _pbc_gdf_gamma_setup(
        system,
        basis,
        lat_opts,
        aux_name=aux_name,
        aux_drop_eta=aux_drop_eta,
        exxdiv=exxdiv,
        gdf_method=gdf_method,
        compcell_eta=compcell_eta,
        apply_aft_correction=apply_aft_correction,
        aft_precision=aft_precision,
        aft_ft_convention=aft_ft_convention,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        mdf_ke_cutoff=mdf_ke_cutoff,
        rcut_strategy=rcut_strategy,
        rcut_precision=rcut_precision,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        linear_dep_threshold=linear_dep_threshold,
        retain_compcell_fit_state=bool(compute_gradient),
        rsgdf_omega=rsgdf_omega, rsgdf_g_precision=rsgdf_g_precision,
        scf_options=opts, open_shell=False,
        fit_screen_threshold=fit_screen_threshold,
        plog=plog,
    )
    S, Hcore, X = setup.S, setup.Hcore, setup.X
    Lpq, aux = setup.Lpq, setup.aux
    madelung, e_nuc = setup.madelung, setup.e_nuc
    gauge_lat_opts = setup.gauge_lat_opts
    mol = system.unit_cell_molecule()
    if n_occ > setup.n_kept:
        raise RuntimeError(
            "run_pbc_gdf_rhf: canonical orthogonalisation dropped too "
            f"many directions (n_occ={n_occ}, n_kept={setup.n_kept})"
        )

    use_davidson = getattr(opts, "use_davidson", False)
    dav_opts = getattr(opts, "davidson", None)
    dav_dim = getattr(opts, "davidson_min_dim", 100)
    use_dav = use_davidson and S.shape[0] >= dav_dim
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    # ---- XC grid + density-set template (KS only) ----------------------
    grid = None
    D_set = None
    _set_xc_density = None
    if is_ks:
        from ._vibeqc_core import GridOptions, build_grid, build_xc_periodic
        from .periodic_grid import build_periodic_becke_grid
        from .periodic_rhf_gdf import _density_set_gamma

        grid_options = getattr(opts, "grid", None) or GridOptions()
        _set_xc_density = _density_set_gamma
        if bool(getattr(opts, "use_periodic_becke", False)):
            grid = build_periodic_becke_grid(
                system,
                grid_options=grid_options,
                image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
            )
            # Periodic-Becke grid pairs with the Γ-torus density (every
            # lattice block populated -> build_xc_periodic cross-cell
            # mode); the home-cell-only set is the molecular-limit
            # density that pairs with the molecular grid below (the
            # 2026-07-09 KRKS finding class, b3f74aa9 --
            # HANDOVER_AICCM_DIRECT_TORUS.md §4).
            from .periodic_rhf_gdf import _density_set_torus_gamma

            _set_xc_density = _density_set_torus_gamma
        else:
            grid = build_grid(system.unit_cell_molecule(), grid_options)
        D_set = compute_overlap_lattice(basis, system, lat_opts)

    k_gamma = np.zeros(3)  # Γ-point Bloch phase for the V_xc fold below

    def _xc_at(D_in: np.ndarray) -> Tuple[float, np.ndarray]:
        """E_xc + folded, symmetrised V_xc(Γ) for the current density."""
        _set_xc_density(D_set, D_in)
        xc = build_xc_periodic(basis, system, grid, func, D_set, lat_opts)
        V = np.real(bloch_sum(xc.V_xc, k_gamma))
        return float(xc.e_xc), 0.5 * (V + V.T)

    # ---- SCF loop -----------------------------------------------------
    def diagonalise(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Fp = X.T @ F @ X
        Fp = 0.5 * (Fp + Fp.T)
        if use_dav and dav_opts is not None:
            from vibeqc._vibeqc_core import davidson_solve

            if dav_opts.n_eig == 0:
                dav_opts.n_eig = Fp.shape[0]
            if dav_opts.guess_vectors is not None:
                pass  # already set from previous iteration
            dres = davidson_solve(Fp, dav_opts)
            if not dres.converged:
                raise RuntimeError(
                    f"Davidson did not converge after {dres.n_iter} iters"
                )
            eps, Cp = dres.eigenvalues, dres.eigenvectors
            dav_opts.guess_vectors = Cp
        else:
            eps, Cp = np.linalg.eigh(Fp)
        return X @ Cp, eps

    def occupations_from_eps(eps: np.ndarray) -> np.ndarray:
        from .smearing.apply import _global_aufbau_with_mu

        occupations, _ = _global_aufbau_with_mu([eps], [1.0], n_elec)
        return np.asarray(occupations[0], dtype=float)

    D_engine = initial_density_closed_shell(
        mol,
        basis,
        n_occ,
        InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=lat_opts,
        # READ restart (Γ-only): prior g=0 cell density (pre-resolved from
        # read_from, or read + projected from read_path). Ignored unless READ.
        read_density=getattr(opts, "read_density", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if D_engine is not None:
        plog.info(
            f"initial guess: {guess.name} (shared periodic adapter density)"
        )
        D = D_engine
        C0, eps0 = diagonalise(Hcore)
    else:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise)")
        C0, eps0 = diagonalise(Hcore)
        occ0 = occupations_from_eps(eps0)
        D = _density_from_orbitals_and_occupations(C0, occ0)
    if guess == InitialGuess.PATOM:
        J_seed = _build_j_from_lpq(Lpq, D)
        K_seed = _build_k_from_lpq(Lpq, D)
        if exxdiv is PBCExxDiv.EWALD:
            K_seed = apply_exxdiv_ewald_to_K([K_seed], [S], [D], madelung)[0]
        C0, eps0 = diagonalise(Hcore + J_seed - 0.5 * K_seed)
        D = _density_from_orbitals_and_occupations(C0, occupations_from_eps(eps0))
    D_prev = D.copy()

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"run_pbc_gdf_rhf: damping must be in [0, 1); got {damping}")
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[PeriodicSCFAccelerator] = (
        PeriodicSCFAccelerator(opts) if use_diis else None
    )
    max_iter = int(opts.max_iter)

    scf_trace: List[SCFIteration] = []
    result = PBCGDFResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=float(e_nuc),
        e_coulomb=0.0,
        e_hf_exchange=0.0,
        e_exxdiv=0.0,
        n_iter=0,
        converged=False,
        mo_energies=np.empty(0),
        mo_coeffs=np.empty((0, 0)),
        density=D.copy(),
        fock=np.empty((0, 0)),
        overlap=S,
        hcore=Hcore,
        scf_trace=scf_trace,
        aux_basis_name=aux_name,
        n_aux=int(aux.nbasis),
        n_fit=int(Lpq.shape[0]),
        madelung_constant=float(madelung),
        exxdiv=exxdiv.value,
        compcell_eta=float(compcell_eta),
        apply_aft_correction=bool(apply_aft_correction),
        aft_precision=float(aft_precision),
        aft_ft_convention=str(aft_ft_convention),
        v_ne_backend=v_ne_backend,
        v_ne_ke_cutoff=v_ne_ke_cutoff,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
        rsgdf_tail_ke_cutoff=(
            float(rsgdf_tail_ke_cutoff)
            if (rsgdf_tail_ke_cutoff is not None and gdf_method == "rsgdf")
            else None
        ),
        gdf_rcut_strategy=gdf_rcut_strategy_value,
        gdf_rcut_precision=float(rcut_precision),
        gdf_linear_dep_threshold=float(gdf_linear_dep_threshold),
        gdf_lattice_cutoff_bohr=float(lat_opts.cutoff_bohr),
        gdf_nuclear_cutoff_bohr=float(lat_opts.nuclear_cutoff_bohr),
        gdf_fit_cutoff_2c=float(setup.fit_cutoff_2c),
        gdf_fit_cutoff_3c=float(setup.fit_cutoff_3c),
        aux_basis_fingerprint=_basis_fingerprint(aux),
        backend=_gdf_backend_with_parity_hold(
            f"pbc-gdf-{gdf_method}" + ("-rks" if is_ks else ""), parity_held
        ),
        functional=func_name,
    )

    plog.banner(
        f"SCF (PBC-GDF {gdf_method}{' RKS ' + func_name if is_ks else ''})"
    )
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    E_prev = 0.0
    C_final = C0
    eps_final = eps0

    for iter_idx in range(1, max_iter + 1):
        diis_active = use_diis and iter_idx >= diis_start_iter
        D_used = (
            D
            if (iter_idx == 1 or damping == 0.0 or diis_active)
            else damping * D_prev + (1.0 - damping) * D
        )

        J = _build_j_from_lpq(Lpq, D_used)
        K = _build_k_from_lpq(Lpq, D_used) if alpha != 0.0 else None

        # exxdiv='ewald' K-shift (per Γ-point as a 1-element list). For
        # hybrids the shift rides the exact-exchange channel, scaled by
        # a_x through the -a_x/2 K contraction below -- the identical
        # convention to the multi-k apply_exxdiv_ewald_to_K route and
        # the real-Γ direct route's ξ_N seam. Pure DFT (a_x = 0) has no
        # exact-exchange channel, so no shift enters at all.
        e_exx = 0.0
        if K is not None and exxdiv is PBCExxDiv.EWALD:
            K = apply_exxdiv_ewald_to_K([K], [S], [D_used], madelung)[0]
            e_exx = exxdiv_ewald_energy_shift(
                [D_used],
                [S],
                madelung,
                hf_exchange_fraction=alpha,
                weights=[1.0],
            )

        E_xc = 0.0
        V_xc = 0.0
        if is_ks:
            E_xc, V_xc = _xc_at(D_used)

        F = Hcore + J - (0.5 * alpha * K if K is not None else 0.0) + V_xc
        F = 0.5 * (F + F.T)

        E_core = float(np.einsum("ij,ij->", D_used, Hcore))
        E_J = 0.5 * float(np.einsum("ij,ij->", D_used, J))
        # The exxdiv shift is already inside K above, so E_hf includes it.
        E_K = (
            -0.25 * alpha * float(np.einsum("ij,ij->", D_used, K))
            if K is not None
            else 0.0
        )
        E_elec = E_core + E_J + E_K + E_xc
        E_total = E_elec + float(e_nuc)

        FDS = F @ D_used @ S
        grad = FDS - FDS.T
        grad_norm = float(np.linalg.norm(grad))
        dE = E_total - E_prev
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )

        F_plain = F
        if converged:
            C_plain, eps_plain = diagonalise(F_plain)
            own_density = _density_from_orbitals_and_occupations(
                C_plain, occupations_from_eps(eps_plain),
            )
            # A commutator alone misses a wrong filling of a diagonal Fock.
            # Check the actual state, including a degenerate T=0 frontier.
            density_residual = np.linalg.norm(
                X.T @ S @ (own_density - D_used) @ S @ X
            )
            converged = density_residual <= max(
                10.0 * float(opts.conv_tol_grad), 1e-8,
            )

        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )

        if accel is not None:
            F_ex = accel.extrapolate_rhf(
                F,
                error=grad,
                density=D_used,
                energy=E_total,
                mo_coeffs=C_final,
                mo_energies=eps_final,
                n_occ=n_occ,
            )
            if diis_active:
                F = F_ex

        C_new, eps_new = diagonalise(F)
        occ = occupations_from_eps(eps_new)
        D_prev = D_used
        D = _density_from_orbitals_and_occupations(C_new, occ)

        C_final = C_new
        eps_final = eps_new
        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

        result.energy = E_total
        result.e_electronic = E_elec
        result.e_coulomb = E_J
        result.e_hf_exchange = E_K
        result.e_xc = E_xc
        result.e_exxdiv = e_exx
        result.n_iter = iter_idx
        result.mo_energies = eps_new
        result.mo_coeffs = C_new
        result.density = D_used
        result.fock = F

        if converged:
            # Retain the state that passed convergence; no extra density
            # update is allowed between the energy test and the result.
            D, F_f = D_used, F_plain
            C_f, eps_f = C_plain, eps_plain
            E_elec_f = E_elec
            result.mo_energies = eps_f
            result.mo_coeffs = C_f
            result.density = D
            result.fock = F_f
            if compute_gradient:
                from .smearing import occupations_are_per_k_integer_aufbau

                if not occupations_are_per_k_integer_aufbau(
                    [occupations_from_eps(eps_f)], n_occ,
                ):
                    raise NotImplementedError(
                        "run_pbc_gdf_rhf: analytic gradients of a T=0 "
                        "degenerate global-Aufbau ensemble are not validated"
                    )
            result.converged = True
            if compute_gradient and gdf_method == "rsgdf":
                # G-PBC-002 M6 rung 4: the rsgdf RHF gradient — the
                # rung-3 cache rebuilds the exact SCF fit (tailed and
                # Schwarz-screened envelopes included; the native tail
                # pair screen mirrors the SCF's unscreened-fit knob
                # only) and the assembly shares the one-electron terms
                # with the compcell path.
                from .periodic_gdf_gradient import compute_gdf_gradient_rsgdf_rhf_gamma

                rsgdf_grad_cache = setup.range_separated_cache
                if int(rsgdf_grad_cache.n_fit) != int(result.n_fit):
                    raise RuntimeError(
                        "run_pbc_gdf_rhf: rsgdf gradient-cache fit rank "
                        f"({rsgdf_grad_cache.n_fit}) does not match the "
                        f"converged SCF fit ({result.n_fit})."
                    )
                grad_total = compute_gdf_gradient_rsgdf_rhf_gamma(
                    system,
                    basis,
                    D=D,
                    C_occ=C_f[:, :n_occ],
                    eps_occ=eps_f[:n_occ],
                    S=S,
                    cache=rsgdf_grad_cache,
                    lattice_opts=setup.oneel_lat_opts or lat_opts,
                    gauge_lat_opts=setup.gauge_lat_opts,
                    madelung=float(result.madelung_constant),
                    v_ne_ke_cutoff=float(result.v_ne_ke_cutoff),
                    alpha_hf=(alpha if is_ks else 1.0),
                )
                if is_ks:
                    # M6 rung 5a: the closed-shell XC Pulay on the SCF's
                    # own grid/density-set conventions (M2 pattern).
                    from ._vibeqc_core import (
                        xc_lattice_gradient_contribution,
                    )

                    _set_xc_density(D_set, D)
                    grad_total = grad_total + np.asarray(
                        xc_lattice_gradient_contribution(
                            basis, system, grid, func, D_set, lat_opts
                        )
                    )
                result.gradient = grad_total
            elif compute_gradient:
                # The compcell path: for KS the wrapper dispatches on
                # result.functional and needs the SCF's XC quadrature
                # provenance (grid options + Becke mode).
                from .periodic_gdf_gradient import compute_gdf_gradient

                result.gradient = compute_gdf_gradient(
                    system,
                    basis,
                    result,
                    aux_basis_name=aux_name,
                    compcell_eta=float(compcell_eta),
                    lattice_opts=lat_opts,
                    grid_options=(grid_options if is_ks else None),
                    use_periodic_becke=(
                        bool(getattr(opts, "use_periodic_becke", False))
                        if is_ks
                        else False
                    ),
                    becke_image_radius_bohr=(
                        float(getattr(opts, "becke_image_radius_bohr", 0.0))
                        if is_ks
                        else 0.0
                    ),
                    _fit_state=setup.compcell_fit_state,
                )
            else:
                result.gradient = None
            _check_energy_sanity(result, system, plog)
            plog.converged(
                n_iter=result.n_iter,
                energy=result.energy,
                converged=True,
            )
            return result

    result.mo_coeffs = C_final
    result.mo_energies = eps_final
    result.converged = False
    _check_energy_sanity(result, system, plog)
    plog.converged(
        n_iter=result.n_iter,
        energy=result.energy,
        converged=False,
    )
    return result


def run_pbc_gdf_rks(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
    *,
    functional: Optional[str] = None,
    **kwargs,
) -> PBCGDFResult:
    """Γ-only closed-shell periodic RKS via rsgdf/compcell GDF.

    The closed-shell KS sibling of :func:`run_pbc_gdf_uks`: a thin
    wrapper over :func:`run_pbc_gdf_rhf` that requires a ``functional``
    (from the keyword or ``options.functional``). Hybrids use the
    a_x-scaled GDF exchange with the ``exxdiv='ewald'`` Madelung shift;
    pure functionals skip ``K`` entirely. See :func:`run_pbc_gdf_rhf`
    for the full parameter list.
    """
    _refuse_ecp_options(options, "run_pbc_gdf_rks", system=system)
    func = functional or str(getattr(options, "functional", "") or "")
    if not func:
        raise ValueError("run_pbc_gdf_rks requires functional=...")
    return run_pbc_gdf_rhf(
        system, basis, options, functional=str(func), **kwargs
    )


@dataclass
class PBCGDFUHFResult:
    """Result of a Γ-only open-shell UHF compcell/rsgdf GDF SCF.

    Mirrors :class:`PBCGDFResult` (closed-shell) with per-spin a/b
    orbital/density/Fock blocks and the ``<S^2>`` spin-contamination
    diagnostic, matching :class:`vibeqc.periodic_uhf_ewald.PeriodicUHFEwaldResult`.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    e_coulomb: float
    e_hf_exchange: float
    e_exxdiv: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float
    # a spin
    mo_energies_alpha: np.ndarray
    mo_coeffs_alpha: np.ndarray
    density_alpha: np.ndarray
    fock_alpha: np.ndarray
    # b spin
    mo_energies_beta: np.ndarray
    mo_coeffs_beta: np.ndarray
    density_beta: np.ndarray
    fock_beta: np.ndarray
    overlap: np.ndarray
    hcore: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    scf_trace: List[SCFIteration] = field(default_factory=list)
    aux_basis_name: str = ""
    n_aux: int = 0
    n_fit: int = 0
    madelung_constant: float = 0.0
    exxdiv: str = "ewald"
    compcell_eta: float = 1.0
    backend: str = "pbc-gdf-compcell-uhf"
    # Open-shell Fermi-Dirac smearing (Γ, per-spin global mu_a/mu_b). All zero /
    # empty at T = 0 (the integer-Aufbau default), so existing callers are
    # behavior-neutral.
    smearing_temperature: float = 0.0
    fermi_level_alpha: float = 0.0
    fermi_level_beta: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations_alpha: np.ndarray = field(default_factory=lambda: np.empty(0))
    occupations_beta: np.ndarray = field(default_factory=lambda: np.empty(0))
    fock_mixing: float = 0.0
    #: The RSGDF high-``|G|`` tail cutoff (Ha) this run actually applied
    #: after :func:`_auto_rsgdf_tail_ke_cutoff` resolution, or ``None`` for
    #: base-mesh-only. Mirrors :class:`PBCGDFResult` (GitLab IID 307): the
    #: open-shell Γ drivers auto-size the same tail on the same tight-core
    #: class and record it here, so an open-shell cross-route comparison
    #: can assert matched reciprocal support (GitLab IID 490).
    rsgdf_tail_ke_cutoff: Optional[float] = None
    #: The BASE rsgdf reciprocal mesh (Ha) actually used (IID 307/490).
    rsgdf_ke_cutoff: float = 200.0
    # G-PBC-002 UHF gradient (compcell preview): populated when the
    # driver is called with compute_gradient=True; None otherwise.
    gradient: Optional[np.ndarray] = None

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def run_pbc_gdf_uhf(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
    *,
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    exxdiv: Union[PBCExxDiv, str] = PBCExxDiv.EWALD,
    gdf_method: str = "rsgdf",
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_precision: float = 1e-10,
    aft_ft_convention: str = "libint",
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    mdf_ke_cutoff: float = 40.0,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    fit_screen_threshold: float = 0.0,
    compute_gradient: bool = False,
    check_energy_sanity: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> PBCGDFUHFResult:
    """Γ-only open-shell periodic UHF via rsgdf/compcell GDF.

    The open-shell sibling of :func:`run_pbc_gdf_rhf`: same Lpq cderi
    (spin-independent), Hartree ``J`` from the total density
    ``D_a + D_b``, per-spin exchange ``K_s`` from the shared Lpq, and the
    ``exxdiv='ewald'`` Madelung K-shift applied per spin. a/b occupations
    follow the molecule's ``multiplicity``
    (``n_a = (n_e + mult - 1) // 2``, ``n_b = (n_e - mult + 1) // 2``).
    At ``multiplicity = 1`` it reproduces :func:`run_pbc_gdf_rhf` to SCF
    tolerance (the closed-shell-limit gate).

    ``gdf_method`` defaults to ``'rsgdf'`` (2026-07-09; the open-shell
    drivers used to default to ``'compcell'`` like the RHF sibling). The
    Γ-only compcell Hartree cannot resolve overlapping AO-pair images on
    tight ionic cells -- its q-only cderi is exact only in the vacuum-box
    limit -- and the SCF converges to a non-physical fixed point
    (rocksalt LiH/STO-3G: +579.8 Ha at UHF, +1172.6 Ha at UKS-PBE, both
    ``converged=True``; the rsgdf lane lands at the sane -8.33 / -8.23 Ha
    on the same cell). ``gdf_method='compcell'`` stays selectable for its
    own development; the tight-ionic warning + the energy-sanity guard
    still fence it.

    Parameters are identical to :func:`run_pbc_gdf_rhf`; see its docstring.
    ``compute_gradient=True`` computes the analytic Γ compcell gradient
    (G-PBC-002 milestone 3b) on either AFT setting -- the 2c/3c AFT
    reciprocal-space corrections are differentiated analytically, so the
    production ``apply_aft_correction=True`` fit is supported. Fermi-Dirac
    smearing is differentiated via the Mermin free-energy force theorem
    (the gradient is dA/dR of the reported ``free_energy``; see
    :func:`_gamma_gradient_orbital_blocks`). The mdf fit derivative,
    non-Fermi-Dirac smearing flavors, and the grid ``V_ne`` backend fail
    closed.

    Additionally, ``check_energy_sanity`` (default ``True``) rejects a
    *converged* non-physical total energy with a ``RuntimeError`` instead
    of returning it -- the CLAUDE.md §7 guard; see
    :func:`_check_energy_sanity`. Pass ``False`` to bypass (diagnostics
    only). Returns a :class:`PBCGDFUHFResult` with per-spin a/b blocks and
    the ``<S^2>`` diagnostic.
    """
    _refuse_ecp_options(options, "run_pbc_gdf_uhf", system=system)
    from .periodic_uhf_ewald import _spin_squared

    if options is None:
        options = PeriodicRHFOptions()
    opts = options
    # SPINLOCK. SPIN_SCHEDULE (two-phase: lock spin then release) is delegated
    # before any SCF setup; the sub-runs re-enter with spinlock OFF. PATTERN_HOLD
    # (MOM-hold the seeded occupied set, below) runs inside the SCF loop.
    from ._vibeqc_core import SpinlockMode
    from .spinlock_periodic import check_spinlock_support, run_spin_schedule

    if (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.SPIN_SCHEDULE
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    ):
        return run_spin_schedule(
            lambda sysx, o: run_pbc_gdf_uhf(
                sysx,
                basis,
                o,
                aux_basis=aux_basis,
                aux_drop_eta=aux_drop_eta,
                exxdiv=exxdiv,
                gdf_method=gdf_method,
                compcell_eta=compcell_eta,
                apply_aft_correction=apply_aft_correction,
                aft_precision=aft_precision,
                aft_ft_convention=aft_ft_convention,
                rsgdf_omega=rsgdf_omega,
                rsgdf_g_precision=rsgdf_g_precision,
                rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                mdf_ke_cutoff=mdf_ke_cutoff,
                rcut_strategy=rcut_strategy,
                rcut_precision=rcut_precision,
                linear_dep_threshold=linear_dep_threshold,
                gdf_linear_dep_threshold=gdf_linear_dep_threshold,
                fit_screen_threshold=fit_screen_threshold,
                compute_gradient=compute_gradient,
                check_energy_sanity=check_energy_sanity,
                progress=progress,
                verbose=verbose,
            ),
            system,
            opts,
        )
    check_spinlock_support(
        opts,
        {SpinlockMode.PATTERN_HOLD, SpinlockMode.SPIN_SCHEDULE},
        "the GDF UHF driver",
    )
    lat_opts = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)
    guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_pbc_gdf_uhf",
        supported=periodic_guess_capabilities('gdf', 'UHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
    )

    if not isinstance(exxdiv, PBCExxDiv):
        exxdiv = PBCExxDiv(exxdiv)
    if int(system.dim) != 3:
        raise NotImplementedError(
            "run_pbc_gdf_uhf: only dim=3 periodic systems are supported."
        )
    # Analytic-gradient guards: mirror run_pbc_gdf_rhf's preview
    # envelope. Fail closed on anything the UHF assembly does not
    # differentiate.
    if compute_gradient and str(gdf_method) not in ("compcell", "rsgdf"):
        raise NotImplementedError(
            "run_pbc_gdf_uhf: compute_gradient currently supports "
            "gdf_method='compcell' and 'rsgdf'; the MDF fit derivative "
            "is not implemented."
        )
    # Schwarz-screened rsgdf fits are differentiable since 2026-07-30
    # (fixed-mask derivative; see run_pbc_gdf_rhf).
    # The historical hold on the incomplete AFT-off fit is released:
    # since G-PBC-002 milestone 3b the compcell AFT metric derivative is
    # implemented, so compute_gradient differentiates the production
    # AFT-on fit (and the AFT-off diagnostic fit) exactly.
    # Finite-temperature (Fermi-Dirac) smearing gradients are supported
    # since 2026-07-30 via the Mermin free-energy force theorem: the
    # assemblers run on the fractional-occupation D/W blocks
    # (_gamma_gradient_orbital_blocks) and dA/dR carries no occupation
    # or mu response at self-consistency. Non-Fermi-Dirac flavors fail
    # closed where smear_opts is resolved below
    # (_reject_non_fermi_dirac_gradient).
    if compute_gradient and os.environ.get(
        "VIBEQC_VNE_EWALD3D_BACKEND", "analytic_ft"
    ).lower() == "grid":
        raise NotImplementedError(
            "run_pbc_gdf_uhf: compute_gradient requires the analytical "
            "FT Ewald V_ne backend; unset VIBEQC_VNE_EWALD3D_BACKEND=grid."
        )
    if str(gdf_method) not in ("compcell", "rsgdf", "mdf"):
        raise ValueError(
            f"run_pbc_gdf_uhf: gdf_method must be 'compcell', 'rsgdf', or "
            f"'mdf'; got {gdf_method!r}"
        )

    n_elec = system.n_electrons()
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(f"run_pbc_gdf_uhf: multiplicity must be >= 1, got {mult}")
    if (n_elec + mult - 1) % 2 != 0:
        raise ValueError(
            f"run_pbc_gdf_uhf: n_electrons={n_elec} and multiplicity={mult} "
            "cannot be split into integer a/b occupations."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(
            f"run_pbc_gdf_uhf: invalid occupations n_a={n_alpha}, "
            f"n_b={n_beta} for n_e={n_elec}, mult={mult}."
        )

    Q_nuc = float(sum(a.Z for a in system.unit_cell))
    if abs(Q_nuc - n_elec) > 0.5:
        raise ValueError(
            "run_pbc_gdf_uhf: cell is not charge-neutral "
            f"(Q_nuclei={Q_nuc:.0f}, n_electrons={n_elec})."
        )

    aux_name = aux_basis or default_aux_for(basis.name)
    if gdf_method == "rsgdf" and rsgdf_tail_ke_cutoff is not None:
        warnings.warn(
            "rsgdf_tail_ke_cutoff is obsolete for the SR/LR fit; "
            "use rsgdf_g_precision to control the raw integral error",
            DeprecationWarning, stacklevel=2,
        )
    rsgdf_tail_ke_cutoff = None
    # Tailed rsgdf gradient hold lifted 2026-07-29 -- the dense-core FD
    # inconsistency was an e_nuc truncation artefact in the energy, not
    # a gradient bug (see run_pbc_gdf_rhf / _pbc_gdf_gamma_setup).
    plog.info(
        f"PBC-GDF UHF / aux={aux_name}, exxdiv={exxdiv.value}, "
        f"n_alpha={n_alpha}, n_beta={n_beta} (mult={mult})"
    )
    _warn_gamma_compcell_tight_ionic_cell(system, gdf_method, "run_pbc_gdf_uhf")
    if rsgdf_tail_ke_cutoff is not None and gdf_method == "rsgdf":
        plog.info(
            "RSGDF high-|G| tail cutoff: "
            f"{float(rsgdf_tail_ke_cutoff):g} Ha"
        )
    parity_held = gdf_method != "rsgdf" and _warn_gamma_dense_core_gdf_parity_hold(
        system,
        gdf_method,
        plog,
        ao_basis=basis,
        tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
    )
    if gdf_method == "compcell" and not bool(apply_aft_correction):
        warnings.warn(
            "run_pbc_gdf_uhf: apply_aft_correction=False disables the "
            "compcell AFT long-range correction. The resulting absolute "
            "energy is a diagnostic approximation, not a PySCF-parity "
            "periodic GDF result; use the AFT-on default or gdf_method='rsgdf'.",
            UserWarning,
            stacklevel=2,
        )
        parity_held = True

    # ---- One-electron integrals + Lpq cderi + exxdiv (shared setup) ---
    setup = _pbc_gdf_gamma_setup(
        system,
        basis,
        lat_opts,
        aux_name=aux_name,
        aux_drop_eta=aux_drop_eta,
        exxdiv=exxdiv,
        gdf_method=gdf_method,
        compcell_eta=compcell_eta,
        apply_aft_correction=apply_aft_correction,
        aft_precision=aft_precision,
        aft_ft_convention=aft_ft_convention,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        mdf_ke_cutoff=mdf_ke_cutoff,
        rcut_strategy=rcut_strategy,
        rcut_precision=rcut_precision,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        linear_dep_threshold=linear_dep_threshold,
        retain_compcell_fit_state=bool(compute_gradient),
        rsgdf_omega=rsgdf_omega, rsgdf_g_precision=rsgdf_g_precision,
        scf_options=opts, open_shell=True,
        fit_screen_threshold=fit_screen_threshold,
        plog=plog,
    )
    S, Hcore, X = setup.S, setup.Hcore, setup.X
    Lpq, aux = setup.Lpq, setup.aux
    madelung, e_nuc = setup.madelung, setup.e_nuc
    if max(n_alpha, n_beta) > setup.n_kept:
        raise RuntimeError(
            f"run_pbc_gdf_uhf: orthogonalisation kept {setup.n_kept} directions; "
            f"need >= {max(n_alpha, n_beta)} (n_a={n_alpha}, n_b={n_beta})."
        )

    use_davidson = getattr(opts, "use_davidson", False)
    dav_opts = getattr(opts, "davidson", None)
    dav_dim = getattr(opts, "davidson_min_dim", 100)
    use_dav = use_davidson and S.shape[0] >= dav_dim
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    # ---- SCF loop -----------------------------------------------------
    def diagonalise(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Fp = X.T @ F @ X
        Fp = 0.5 * (Fp + Fp.T)
        if use_dav and dav_opts is not None:
            from vibeqc._vibeqc_core import davidson_solve

            if dav_opts.n_eig == 0:
                dav_opts.n_eig = Fp.shape[0]
            if dav_opts.guess_vectors is not None:
                pass  # already set from previous iteration
            dres = davidson_solve(Fp, dav_opts)
            if not dres.converged:
                raise RuntimeError(
                    f"Davidson did not converge after {dres.n_iter} iters"
                )
            eps, Cp = dres.eigenvalues, dres.eigenvectors
            dav_opts.guess_vectors = Cp
        else:
            eps, Cp = np.linalg.eigh(Fp)
        return X @ Cp, eps

    # Independent spin chemical potentials, with the shared global-Aufbau
    # ensemble convention when the temperature is zero.
    smear_T = float(getattr(opts, "smearing_temperature", 0.0) or 0.0)
    if smear_T < 0.0:
        raise ValueError("run_pbc_gdf_uhf: smearing_temperature must be >= 0")
    from .smearing import (
        SmearingOptions as _SmearingOptions,
        smeared_occupation_selfconsistency_tolerance as _occ_tol_fn,
    )

    smear_opts = _SmearingOptions.from_legacy_kwarg(smear_T)
    _smeared_occ_tol = _occ_tol_fn(float(opts.conv_tol_energy))
    if compute_gradient:
        _reject_non_fermi_dirac_gradient("run_pbc_gdf_uhf", smear_opts)

    C_alpha, eps_alpha = diagonalise(Hcore)
    C_beta, eps_beta = C_alpha.copy(), eps_alpha.copy()
    (D_alpha, D_beta, occ_a, occ_b, fermi_a, fermi_b, entropy_cur) = (
        _open_shell_gamma_occupy(
            C_alpha, eps_alpha, C_beta, eps_beta, n_alpha, n_beta, smear_opts
        )
    )
    D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()

    # All density-mode guesses enter through the shared periodic adapter
    # seam. HCORE is the sole no-density result and retains the baseline
    # Hcore diagonalisation above. PATOM starts from SAD, then applies this
    # route's established one-step GDF in-field refinement.
    _atomic_spins = getattr(opts, "atomic_spins", None) or None
    _seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
    _split = initial_densities_open_shell(
        system.unit_cell_molecule(),
        basis,
        n_alpha,
        n_beta,
        _seed_guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=lat_opts,
        atomic_spins=_atomic_spins,
        read_density_alpha=getattr(opts, "read_density_alpha", None),
        read_density_beta=getattr(opts, "read_density_beta", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if _split is not None:
        D_alpha, D_beta = _split
        D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()
        plog.info(
            f"initial guess: {guess.name} (shared periodic adapter density)"
        )
        if guess == InitialGuess.PATOM:
            plog.info("initial guess: PATOM (SAD + one GDF in-field step)")
            J = _build_j_from_lpq(Lpq, D_alpha + D_beta)
            K_alpha = _build_k_from_lpq(Lpq, D_alpha)
            K_beta = _build_k_from_lpq(Lpq, D_beta)
            if exxdiv is PBCExxDiv.EWALD:
                K_alpha = apply_exxdiv_ewald_to_K(
                    [K_alpha], [S], [D_alpha], madelung
                )[0]
                K_beta = apply_exxdiv_ewald_to_K(
                    [K_beta], [S], [D_beta], madelung
                )[0]
            F_alpha_seed = 0.5 * (
                (Hcore + J - K_alpha) + (Hcore + J - K_alpha).T
            )
            F_beta_seed = 0.5 * (
                (Hcore + J - K_beta) + (Hcore + J - K_beta).T
            )
            C_alpha, eps_alpha = diagonalise(F_alpha_seed)
            C_beta, eps_beta = diagonalise(F_beta_seed)
            (
                D_alpha,
                D_beta,
                occ_a,
                occ_b,
                fermi_a,
                fermi_b,
                entropy_cur,
            ) = _open_shell_gamma_occupy(
                C_alpha,
                eps_alpha,
                C_beta,
                eps_beta,
                n_alpha,
                n_beta,
                smear_opts,
            )
            D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()

    damping = float(opts.damping)
    fock_mixing_value = float(getattr(opts, "fock_mixing", 0.0) or 0.0)
    if not (0.0 <= fock_mixing_value < 1.0):
        raise ValueError(
            "run_pbc_gdf_uhf: fock_mixing must be in [0, 1); "
            f"got {fock_mixing_value}"
        )
    if fock_mixing_value != 0.0:
        plog.info(f"fock mixing: {100.0 * fock_mixing_value:.1f}% previous Fock")
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[PeriodicSCFAccelerator] = (
        PeriodicSCFAccelerator(opts) if use_diis else None
    )

    scf_trace: List[SCFIteration] = []
    result = PBCGDFUHFResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=float(e_nuc),
        e_coulomb=0.0,
        e_hf_exchange=0.0,
        e_exxdiv=0.0,
        n_iter=0,
        converged=False,
        s_squared=0.0,
        s_squared_ideal=0.25 * (mult - 1) * (mult + 1),
        mo_energies_alpha=np.empty(0),
        mo_coeffs_alpha=np.empty((0, 0)),
        density_alpha=D_alpha.copy(),
        fock_alpha=np.empty((0, 0)),
        mo_energies_beta=np.empty(0),
        mo_coeffs_beta=np.empty((0, 0)),
        density_beta=D_beta.copy(),
        fock_beta=np.empty((0, 0)),
        overlap=S,
        hcore=Hcore,
        scf_trace=scf_trace,
        aux_basis_name=aux_name,
        n_aux=int(aux.nbasis),
        n_fit=int(Lpq.shape[0]),
        madelung_constant=float(madelung),
        exxdiv=exxdiv.value,
        compcell_eta=float(compcell_eta),
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
        rsgdf_tail_ke_cutoff=(
            float(rsgdf_tail_ke_cutoff)
            if (rsgdf_tail_ke_cutoff is not None and gdf_method == "rsgdf")
            else None
        ),
        backend=_gdf_backend_with_parity_hold(
            f"pbc-gdf-{gdf_method}-uhf", parity_held
        ),
        fock_mixing=fock_mixing_value,
    )
    plog.banner(f"SCF (PBC-GDF {gdf_method} UHF)")

    E_prev = 0.0
    C_alpha_f, eps_alpha_f = C_alpha, eps_alpha
    C_beta_f, eps_beta_f = C_beta, eps_beta
    # SPINLOCK PATTERN_HOLD: hold the seeded broken-symmetry occupied set by
    # maximum overlap (MOM) with the previous cycle for cycles
    # 2..spinlock_iterations, then release -- protects an ATOMSPIN seed from
    # collapsing to the symmetric solution. The no-smearing occupy is
    # column-order (D_s = C_s[:, :n_s] C_s[:, :n_s]ᵀ), so reordering the held
    # occupied to the front makes the fill pick them up. Mirrors the Γ UHF
    # Ewald driver and the C++ molecular path.
    from .mom import reorder_occupied_by_max_overlap as _mom_reorder

    _pattern_hold = (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.PATTERN_HOLD
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    )
    _spinlock_iters = int(getattr(opts, "spinlock_iterations", 0))
    _ca_occ_prev = None
    _cb_occ_prev = None
    F_alpha_prev_mixed: Optional[np.ndarray] = None
    F_beta_prev_mixed: Optional[np.ndarray] = None
    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        # SPINLOCK PATTERN_HOLD: the accelerator (DIIS / EDIIS / ADIIS /
        # KDIIS -- whatever PeriodicSCFAccelerator resolved) is suspended
        # (no history recorded, no extrapolation, damping stays live)
        # while the hold is active. Fock extrapolation across held-window
        # iterates steers the SCF toward the symmetric attractor by
        # continuous orbital rotation -- a collapse the occupation-selecting
        # MOM hold cannot see -- and poisons the post-release history with
        # out-of-basin iterates. The history starts fresh at release.
        hold_active = _pattern_hold and iter_idx <= _spinlock_iters
        diis_active = (
            use_diis and iter_idx >= diis_start_iter and not hold_active
        )
        if iter_idx == 1 or damping == 0.0 or diis_active:
            D_alpha_used, D_beta_used = D_alpha, D_beta
        else:
            D_alpha_used = damping * D_alpha_prev + (1.0 - damping) * D_alpha
            D_beta_used = damping * D_beta_prev + (1.0 - damping) * D_beta

        J = _build_j_from_lpq(Lpq, D_alpha_used + D_beta_used)
        K_alpha = _build_k_from_lpq(Lpq, D_alpha_used)
        K_beta = _build_k_from_lpq(Lpq, D_beta_used)
        e_exx = 0.0
        if exxdiv is PBCExxDiv.EWALD:
            K_alpha = apply_exxdiv_ewald_to_K([K_alpha], [S], [D_alpha_used], madelung)[
                0
            ]
            K_beta = apply_exxdiv_ewald_to_K([K_beta], [S], [D_beta_used], madelung)[0]
            e_exx = exxdiv_ewald_energy_shift(
                [D_alpha_used], [S], madelung, hf_exchange_fraction=1.0, weights=[1.0]
            ) + exxdiv_ewald_energy_shift(
                [D_beta_used], [S], madelung, hf_exchange_fraction=1.0, weights=[1.0]
            )

        F_alpha = 0.5 * ((Hcore + J - K_alpha) + (Hcore + J - K_alpha).T)
        F_beta = 0.5 * ((Hcore + J - K_beta) + (Hcore + J - K_beta).T)

        # UHF energy: 1/2Tr[(Da+Db)Hcore] + 1/2Tr[Da Fa] + 1/2Tr[Db Fb]
        # (the exxdiv shift is already inside the K_s folded into F_s).
        E_elec = (
            0.5 * float(np.einsum("ij,ij->", D_alpha_used + D_beta_used, Hcore))
            + 0.5 * float(np.einsum("ij,ij->", D_alpha_used, F_alpha))
            + 0.5 * float(np.einsum("ij,ij->", D_beta_used, F_beta))
        )
        E_total = E_elec + float(e_nuc)

        FDS_a = F_alpha @ D_alpha_used @ S
        FDS_b = F_beta @ D_beta_used @ S
        grad_a = FDS_a - FDS_a.T
        grad_b = FDS_b - FDS_b.T
        grad_norm = float(
            np.sqrt(np.linalg.norm(grad_a) ** 2 + np.linalg.norm(grad_b) ** 2)
        )
        dE = E_total - E_prev
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        if converged and not hold_active:
            # Smeared convergence additionally requires the stored
            # occupations to be the Fermi filling of the current plain
            # Fock's own eigenvalues -- the energy + commutator tests
            # cannot see a frozen-occupation fixed point on
            # zero-commutator fixtures (see
            # _smeared_occupation_residual_gamma).
            occ_residual = _smeared_occupation_residual_gamma(
                diagonalise,
                F_alpha,
                F_beta,
                occ_a,
                occ_b,
                n_alpha,
                n_beta,
                smear_opts,
            )
            if occ_residual > _smeared_occ_tol:
                converged = False
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )

        result.energy = E_total
        result.e_electronic = E_elec
        result.e_coulomb = 0.5 * float(
            np.einsum("ij,ij->", D_alpha_used + D_beta_used, J)
        )
        result.e_hf_exchange = -0.5 * (
            float(np.einsum("ij,ij->", D_alpha_used, K_alpha))
            + float(np.einsum("ij,ij->", D_beta_used, K_beta))
        )
        result.e_exxdiv = e_exx
        result.n_iter = iter_idx
        result.mo_energies_alpha = eps_alpha_f
        result.mo_coeffs_alpha = C_alpha_f
        result.density_alpha = D_alpha_used
        result.fock_alpha = F_alpha
        result.mo_energies_beta = eps_beta_f
        result.mo_coeffs_beta = C_beta_f
        result.density_beta = D_beta_used
        result.fock_beta = F_beta
        result.smearing_temperature = smear_T
        from .periodic_k_density import _fermi_density_entropy
        entropy_used = (
            _fermi_density_entropy([D_alpha_used], [S], [X], [1.0], 1.0)
            + _fermi_density_entropy([D_beta_used], [S], [X], [1.0], 1.0)
            if smear_T > 0.0 else 0.0
        )
        result.entropy = entropy_used
        result.free_energy = E_total - smear_T * entropy_used
        result.fermi_level_alpha = float(fermi_a)
        result.fermi_level_beta = float(fermi_b)
        result.occupations_alpha = np.asarray(occ_a, dtype=float)
        result.occupations_beta = np.asarray(occ_b, dtype=float)

        if converged:
            result.converged = True
            result.s_squared = _gamma_open_shell_s_squared(
                n_alpha, n_beta, C_alpha_f, C_beta_f, S, occ_a, occ_b, smear_opts
            )
            if compute_gradient:
                # T = 0: integer slices (bit-identical). T > 0: Mermin
                # free-energy blocks -- sqrt(f)-scaled columns so the
                # occupation-agnostic assemblers evaluate the
                # fractional-occupation D/W (the gradient is dA/dR of
                # result.free_energy; see _gamma_gradient_orbital_blocks).
                C_a_grad, eps_a_grad = _gamma_gradient_orbital_blocks(
                    C_alpha_f, eps_alpha_f, n_alpha, occ_a, smear_opts
                )
                C_b_grad, eps_b_grad = _gamma_gradient_orbital_blocks(
                    C_beta_f, eps_beta_f, n_beta, occ_b, smear_opts
                )
            if compute_gradient and gdf_method == "rsgdf":
                # G-PBC-002 M6 rung 5b: the rsgdf per-spin assembly on
                # the rung-3 cache (tailed and Schwarz-screened
                # envelopes included since 2026-07-30).
                from .periodic_gdf_gradient import compute_gdf_gradient_rsgdf_uhf_gamma

                rsgdf_grad_cache = setup.range_separated_cache
                if int(rsgdf_grad_cache.n_fit) != int(result.n_fit):
                    raise RuntimeError(
                        "run_pbc_gdf_uhf: rsgdf gradient-cache fit rank "
                        f"({rsgdf_grad_cache.n_fit}) does not match the "
                        f"converged SCF fit ({result.n_fit})."
                    )
                result.gradient = compute_gdf_gradient_rsgdf_uhf_gamma(
                    system,
                    basis,
                    D_alpha=D_alpha_used,
                    D_beta=D_beta_used,
                    C_alpha_occ=C_a_grad,
                    C_beta_occ=C_b_grad,
                    eps_alpha_occ=eps_a_grad,
                    eps_beta_occ=eps_b_grad,
                    S=S,
                    cache=rsgdf_grad_cache,
                    lattice_opts=setup.oneel_lat_opts or lat_opts,
                    gauge_lat_opts=setup.gauge_lat_opts,
                    madelung=(
                        float(madelung)
                        if exxdiv == PBCExxDiv.EWALD
                        else 0.0
                    ),
                    v_ne_ke_cutoff=float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0")),
                    alpha_hf=1.0,
                )
            elif compute_gradient:
                # In-driver analytic gradient on the compcell preview
                # envelope (G-PBC-002 milestone 1). Cache is built from the
                # SCF's own retained fit state, so every fit setting matches
                # the converged Lpq by construction; the assembly is the
                # per-spin generalisation in compute_gdf_gradient_uhf_gamma.
                from .aux_basis import make_aux_basis_set
                from .periodic_gdf_gradient import (
                    _build_compcell_gradient_cache,
                    compute_gdf_gradient_uhf_gamma,
                )

                aux_unscaled = make_aux_basis_set(
                    system.unit_cell_molecule(), aux_name=aux_name
                )
                grad_cache = _build_compcell_gradient_cache(
                    system,
                    basis,
                    aux_unscaled,
                    compcell_eta=float(compcell_eta),
                    lattice_opts=lat_opts,
                    linear_dep_thr=float(gdf_linear_dep_threshold),
                    rcut_strategy=rcut_strategy,
                    rcut_precision=float(rcut_precision),
                    fit_state=setup.compcell_fit_state,
                )
                uhf_v_ne_ke_cutoff = float(
                    os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0")
                )
                result.gradient = compute_gdf_gradient_uhf_gamma(
                    system,
                    basis,
                    D_alpha=D_alpha_used,
                    D_beta=D_beta_used,
                    C_alpha_occ=C_a_grad,
                    C_beta_occ=C_b_grad,
                    eps_alpha_occ=eps_a_grad,
                    eps_beta_occ=eps_b_grad,
                    S=S,
                    cache=grad_cache,
                    lattice_opts=lat_opts,
                    gauge_lat_opts=setup.gauge_lat_opts,
                    madelung=(
                        float(madelung)
                        if exxdiv == PBCExxDiv.EWALD
                        else 0.0
                    ),
                    v_ne_ke_cutoff=uhf_v_ne_ke_cutoff,
                )
            if check_energy_sanity:
                _check_energy_sanity(
                    result,
                    system,
                    plog,
                    driver="run_pbc_gdf_uhf",
                    raise_if_converged=True,
                )
            plog.converged(n_iter=iter_idx, energy=E_total, converged=True)
            return result

        # Skipped entirely (not even recorded) while the PATTERN_HOLD window
        # is active; see the hold_active note at the loop head.
        if accel is not None and not hold_active:
            F_alpha_ex, F_beta_ex = accel.extrapolate_uhf(
                F_alpha,
                F_beta,
                error_alpha=grad_a,
                error_beta=grad_b,
                density_alpha=D_alpha_used,
                density_beta=D_beta_used,
                energy=E_total,
                mo_coeffs_alpha=C_alpha_f,
                mo_coeffs_beta=C_beta_f,
                mo_energies_alpha=eps_alpha_f,
                mo_energies_beta=eps_beta_f,
                n_alpha=n_alpha,
                n_beta=n_beta,
            )
            if diis_active:
                F_alpha, F_beta = F_alpha_ex, F_beta_ex
        if fock_mixing_value != 0.0:
            if F_alpha_prev_mixed is not None and F_beta_prev_mixed is not None:
                F_alpha_mix = (
                    (1.0 - fock_mixing_value) * F_alpha
                    + fock_mixing_value * F_alpha_prev_mixed
                )
                F_beta_mix = (
                    (1.0 - fock_mixing_value) * F_beta
                    + fock_mixing_value * F_beta_prev_mixed
                )
                F_alpha = 0.5 * (F_alpha_mix + F_alpha_mix.T)
                F_beta = 0.5 * (F_beta_mix + F_beta_mix.T)
            F_alpha_prev_mixed = F_alpha.copy()
            F_beta_prev_mixed = F_beta.copy()

        C_alpha_f, eps_alpha_f = diagonalise(F_alpha)
        C_beta_f, eps_beta_f = diagonalise(F_beta)
        if _pattern_hold and 1 < iter_idx <= _spinlock_iters:
            if n_alpha > 0 and _ca_occ_prev is not None:
                C_alpha_f, eps_alpha_f = _mom_reorder(
                    C_alpha_f, eps_alpha_f, S, _ca_occ_prev, n_alpha
                )
            if n_beta > 0 and _cb_occ_prev is not None:
                C_beta_f, eps_beta_f = _mom_reorder(
                    C_beta_f, eps_beta_f, S, _cb_occ_prev, n_beta
                )
        if _pattern_hold and iter_idx <= _spinlock_iters:
            _ca_occ_prev = (
                np.asarray(C_alpha_f[:, :n_alpha]).copy() if n_alpha > 0 else None
            )
            _cb_occ_prev = (
                np.asarray(C_beta_f[:, :n_beta]).copy() if n_beta > 0 else None
            )
        D_alpha_prev, D_beta_prev = D_alpha_used, D_beta_used
        (D_alpha, D_beta, occ_a, occ_b, fermi_a, fermi_b, entropy_cur) = (
            _open_shell_gamma_occupy(
                C_alpha_f,
                eps_alpha_f,
                C_beta_f,
                eps_beta_f,
                n_alpha,
                n_beta,
                smear_opts,
                preserve_column_order=hold_active,
            )
        )
        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

    result.s_squared = float(
        _gamma_open_shell_s_squared(
            n_alpha, n_beta, C_alpha_f, C_beta_f, S, occ_a, occ_b, smear_opts
        )
    )
    result.converged = False
    if check_energy_sanity:
        _check_energy_sanity(
            result,
            system,
            plog,
            driver="run_pbc_gdf_uhf",
            raise_if_converged=True,
        )
    plog.converged(n_iter=result.n_iter, energy=result.energy, converged=False)
    return result


@dataclass
class PBCGDFUKSResult:
    """Result of a Γ-only open-shell UKS compcell/rsgdf GDF SCF.

    As :class:`PBCGDFUHFResult` plus the XC energy ``e_xc`` and the
    ``functional`` name. ``e_hf_exchange`` is the (a-scaled) HF-exchange
    energy for hybrids; ``0`` for pure DFT.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    e_coulomb: float
    e_hf_exchange: float
    e_xc: float
    e_exxdiv: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float
    functional: str
    mo_energies_alpha: np.ndarray
    mo_coeffs_alpha: np.ndarray
    density_alpha: np.ndarray
    fock_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    mo_coeffs_beta: np.ndarray
    density_beta: np.ndarray
    fock_beta: np.ndarray
    overlap: np.ndarray
    hcore: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    scf_trace: List[SCFIteration] = field(default_factory=list)
    aux_basis_name: str = ""
    n_aux: int = 0
    n_fit: int = 0
    madelung_constant: float = 0.0
    exxdiv: str = "ewald"
    compcell_eta: float = 1.0
    backend: str = "pbc-gdf-compcell-uks"
    # Open-shell Fermi-Dirac smearing (Γ, per-spin global mu_a/mu_b); zero/empty
    # at T = 0, so existing callers are behavior-neutral.
    smearing_temperature: float = 0.0
    fermi_level_alpha: float = 0.0
    fermi_level_beta: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations_alpha: np.ndarray = field(default_factory=lambda: np.empty(0))
    occupations_beta: np.ndarray = field(default_factory=lambda: np.empty(0))
    fock_mixing: float = 0.0
    #: The RSGDF high-``|G|`` tail cutoff (Ha) this run actually applied
    #: after :func:`_auto_rsgdf_tail_ke_cutoff` resolution, or ``None`` for
    #: base-mesh-only. Mirrors :class:`PBCGDFResult` (GitLab IID 307); the
    #: open-shell Γ drivers auto-size the same tail on the same tight-core
    #: class and record it here, so an open-shell cross-route comparison
    #: can assert matched reciprocal support (GitLab IID 490).
    rsgdf_tail_ke_cutoff: Optional[float] = None
    #: The BASE rsgdf reciprocal mesh (Ha) actually used (IID 307/490).
    rsgdf_ke_cutoff: float = 200.0
    # G-PBC-002 UKS gradient (compcell preview, milestone 4): populated
    # when the driver is called with compute_gradient=True; None otherwise.
    gradient: Optional[np.ndarray] = None

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def run_pbc_gdf_uks(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
    *,
    functional: Optional[str] = None,
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    exxdiv: Union[PBCExxDiv, str] = PBCExxDiv.EWALD,
    gdf_method: str = "rsgdf",
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_precision: float = 1e-10,
    aft_ft_convention: str = "libint",
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    mdf_ke_cutoff: float = 40.0,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    fit_screen_threshold: float = 0.0,
    compute_gradient: bool = False,
    check_energy_sanity: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> PBCGDFUKSResult:
    """Γ-only open-shell periodic UKS via rsgdf/compcell GDF.

    The DFT sibling of :func:`run_pbc_gdf_uhf`: same spin-independent Lpq,
    Hartree ``J`` from ``D_a + D_b``, native per-spin libxc ``V_xc`` from
    :func:`build_xc_periodic_uks` on the periodic Becke grid, and -- for
    hybrids -- the a-scaled per-spin exchange ``K_s`` from the shared Lpq
    with the ``exxdiv='ewald'`` Madelung shift. The HF-exchange fraction
    ``a`` is the functional's; pure DFT (``a = 0``) skips ``K`` entirely.

    ``functional`` selects the XC functional (e.g. ``"pbe"``, ``"b3lyp"``,
    ``"pbe0"``); ``None`` falls back to ``options.functional`` and, if that
    is empty too, the driver runs as plain UHF (``a = 1``, no ``V_xc``) --
    so ``run_pbc_gdf_uks(functional=None)`` reproduces
    :func:`run_pbc_gdf_uhf`/:func:`run_pbc_gdf_rhf`. a/b occupations follow
    ``multiplicity``. ``check_energy_sanity`` (default ``True``) rejects a
    *converged* non-physical total energy with a ``RuntimeError`` instead
    of returning it -- the CLAUDE.md §7 guard; see
    :func:`_check_energy_sanity`. Pass ``False`` to bypass (diagnostics
    only). Returns a :class:`PBCGDFUKSResult`.

    ``gdf_method`` defaults to ``'rsgdf'`` (2026-07-09; see
    :func:`run_pbc_gdf_uhf` -- the Γ-only compcell Hartree returned a
    converged +1172.6 Ha on rocksalt LiH/STO-3G at UKS-PBE). KS runs with
    ``options=None`` default to :class:`PeriodicKSOptions`, whose
    ``use_periodic_becke=True`` selects the periodic-Becke grid + Γ-torus
    density pairing (b3f74aa9); the old ``PeriodicRHFOptions`` fallback
    silently evaluated XC in the v0.8.x molecular-grid convention
    (-7.9638 vs -8.2339 Ha on the same LiH fixture).
    """
    _refuse_ecp_options(options, "run_pbc_gdf_uks", system=system)
    from ._vibeqc_core import (
        Functional,
        GridOptions,
        PeriodicKSOptions,
        build_grid,
        build_xc_periodic_uks,
    )
    from .periodic_grid import build_periodic_becke_grid
    from .periodic_rhf_gdf import _density_set_gamma
    from .periodic_uhf_ewald import _spin_squared

    if options is None:
        # KS run (functional given here or via options) -> PeriodicKSOptions,
        # so the periodic-Becke grid + torus density pairing engages by
        # default; plain-UHF fallback keeps PeriodicRHFOptions.
        options = (
            PeriodicKSOptions() if functional else PeriodicRHFOptions()
        )
    opts = options
    # SPINLOCK. SPIN_SCHEDULE (two-phase) is delegated before SCF setup;
    # PATTERN_HOLD (MOM-hold the seeded occupied set) runs inside the SCF loop.
    from ._vibeqc_core import SpinlockMode
    from .spinlock_periodic import check_spinlock_support, run_spin_schedule

    if (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.SPIN_SCHEDULE
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    ):
        return run_spin_schedule(
            lambda sysx, o: run_pbc_gdf_uks(
                sysx,
                basis,
                o,
                functional=functional,
                aux_basis=aux_basis,
                aux_drop_eta=aux_drop_eta,
                exxdiv=exxdiv,
                gdf_method=gdf_method,
                compcell_eta=compcell_eta,
                apply_aft_correction=apply_aft_correction,
                aft_precision=aft_precision,
                aft_ft_convention=aft_ft_convention,
                rsgdf_omega=rsgdf_omega,
                rsgdf_g_precision=rsgdf_g_precision,
                rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                mdf_ke_cutoff=mdf_ke_cutoff,
                rcut_strategy=rcut_strategy,
                rcut_precision=rcut_precision,
                linear_dep_threshold=linear_dep_threshold,
                gdf_linear_dep_threshold=gdf_linear_dep_threshold,
                fit_screen_threshold=fit_screen_threshold,
                check_energy_sanity=check_energy_sanity,
                progress=progress,
                verbose=verbose,
            ),
            system,
            opts,
        )
    check_spinlock_support(
        opts,
        {SpinlockMode.PATTERN_HOLD, SpinlockMode.SPIN_SCHEDULE},
        "the GDF UKS driver",
    )
    lat_opts = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)
    guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_pbc_gdf_uks",
        supported=periodic_guess_capabilities('gdf', 'UKS', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
    )

    if not isinstance(exxdiv, PBCExxDiv):
        exxdiv = PBCExxDiv(exxdiv)
    if int(system.dim) != 3:
        raise NotImplementedError("run_pbc_gdf_uks: only dim=3 is supported.")
    if str(gdf_method) not in ("compcell", "rsgdf", "mdf"):
        raise ValueError(
            f"run_pbc_gdf_uks: gdf_method must be 'compcell', 'rsgdf', or "
            f"'mdf'; got {gdf_method!r}"
        )
    # Analytic-gradient guards (G-PBC-002 milestone 4): the compcell
    # preview envelope of the RHF/RKS/UHF wirings, per-spin + XC.
    if compute_gradient and str(gdf_method) not in ("compcell", "rsgdf"):
        raise NotImplementedError(
            "run_pbc_gdf_uks: compute_gradient currently supports "
            "gdf_method='compcell' and 'rsgdf'; the MDF fit derivative "
            "is not implemented."
        )
    # Schwarz-screened rsgdf fits are differentiable since 2026-07-30
    # (fixed-mask derivative; see run_pbc_gdf_rhf).
    # Finite-temperature (Fermi-Dirac) smearing gradients are supported
    # since 2026-07-30 (Mermin free-energy force theorem; see
    # _gamma_gradient_orbital_blocks). Non-Fermi-Dirac flavors fail
    # closed at the smear_opts resolution below.
    if compute_gradient and os.environ.get(
        "VIBEQC_VNE_EWALD3D_BACKEND", "analytic_ft"
    ).lower() == "grid":
        raise NotImplementedError(
            "run_pbc_gdf_uks: compute_gradient requires the analytical "
            "FT Ewald V_ne backend; unset VIBEQC_VNE_EWALD3D_BACKEND=grid."
        )

    func_name = functional or str(getattr(opts, "functional", "") or "")
    is_ks = bool(func_name)
    func = Functional(func_name, 2) if is_ks else None  # spin-polarized
    if func is not None and bool(getattr(func, "is_external", False)):
        raise NotImplementedError(
            "run_pbc_gdf_uks: the direct Gamma GDF fast path cannot supply "
            "the difference-closed periodic AO-density domain required by "
            "a full-grid external XC functional. Use "
            "run_kuks_periodic_gdf(..., kmesh=(1, 1, 1)); the generic "
            "finite-torus driver owns external-XC Gamma and multi-k "
            "evaluation."
        )
    reject_periodic_gdf_unsupported_functional(
        func, where="run_pbc_gdf_uks"
    )
    # Same fail-closed rule as run_pbc_gdf_rhf's erf-attenuated-Lpq
    # guard: the fitted K is full-range only.
    reject_unscreened_range_separated(func, where="run_pbc_gdf_uks")
    alpha = float(func.hf_exchange_fraction) if func is not None else 1.0

    n_elec = system.n_electrons()
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(f"run_pbc_gdf_uks: multiplicity must be >= 1, got {mult}")
    if (n_elec + mult - 1) % 2 != 0:
        raise ValueError(
            f"run_pbc_gdf_uks: n_electrons={n_elec} and multiplicity={mult} "
            "cannot be split into integer a/b occupations."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2

    Q_nuc = float(sum(a.Z for a in system.unit_cell))
    if abs(Q_nuc - n_elec) > 0.5:
        raise ValueError(
            f"run_pbc_gdf_uks: cell is not charge-neutral "
            f"(Q_nuclei={Q_nuc:.0f}, n_electrons={n_elec})."
        )

    aux_name = aux_basis or default_aux_for(basis.name)
    if gdf_method == "rsgdf" and rsgdf_tail_ke_cutoff is not None:
        warnings.warn(
            "rsgdf_tail_ke_cutoff is obsolete for the SR/LR fit; "
            "use rsgdf_g_precision to control the raw integral error",
            DeprecationWarning, stacklevel=2,
        )
    rsgdf_tail_ke_cutoff = None
    # Tailed rsgdf gradient hold lifted 2026-07-29 -- the dense-core FD
    # inconsistency was an e_nuc truncation artefact in the energy, not
    # a gradient bug (see run_pbc_gdf_rhf / _pbc_gdf_gamma_setup).
    plog.info(
        f"PBC-GDF UKS / func={func_name or '(none->UHF)'}, alpha={alpha:g}, "
        f"aux={aux_name}, n_alpha={n_alpha}, n_beta={n_beta} (mult={mult})"
    )
    _warn_gamma_compcell_tight_ionic_cell(system, gdf_method, "run_pbc_gdf_uks")
    if rsgdf_tail_ke_cutoff is not None and gdf_method == "rsgdf":
        plog.info(
            "RSGDF high-|G| tail cutoff: "
            f"{float(rsgdf_tail_ke_cutoff):g} Ha"
        )
    parity_held = gdf_method != "rsgdf" and _warn_gamma_dense_core_gdf_parity_hold(
        system,
        gdf_method,
        plog,
        ao_basis=basis,
        tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
    )

    # ---- One-electron integrals + Lpq cderi + exxdiv (shared setup) ---
    setup = _pbc_gdf_gamma_setup(
        system,
        basis,
        lat_opts,
        aux_name=aux_name,
        aux_drop_eta=aux_drop_eta,
        exxdiv=exxdiv,
        gdf_method=gdf_method,
        compcell_eta=compcell_eta,
        apply_aft_correction=apply_aft_correction,
        aft_precision=aft_precision,
        aft_ft_convention=aft_ft_convention,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        mdf_ke_cutoff=mdf_ke_cutoff,
        rcut_strategy=rcut_strategy,
        rcut_precision=rcut_precision,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        linear_dep_threshold=linear_dep_threshold,
        retain_compcell_fit_state=bool(compute_gradient),
        rsgdf_omega=rsgdf_omega, rsgdf_g_precision=rsgdf_g_precision,
        scf_options=opts, open_shell=True,
        fit_screen_threshold=fit_screen_threshold,
        plog=plog,
    )
    S, Hcore, X = setup.S, setup.Hcore, setup.X
    Lpq, aux = setup.Lpq, setup.aux
    madelung, e_nuc = setup.madelung, setup.e_nuc
    if max(n_alpha, n_beta) > setup.n_kept:
        raise RuntimeError(
            f"run_pbc_gdf_uks: orthogonalisation kept {setup.n_kept} directions; "
            f"need >= {max(n_alpha, n_beta)} (n_a={n_alpha}, n_b={n_beta})."
        )

    use_davidson = getattr(opts, "use_davidson", False)
    dav_opts = getattr(opts, "davidson", None)
    dav_dim = getattr(opts, "davidson_min_dim", 100)
    use_dav = use_davidson and S.shape[0] >= dav_dim
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    # ---- XC grid + per-spin density-set templates (KS only) -----------
    grid = None
    D_alpha_set = D_beta_set = None
    _set_xc_density = _density_set_gamma
    if is_ks:
        grid_options = getattr(opts, "grid", None) or GridOptions()
        if bool(getattr(opts, "use_periodic_becke", False)):
            grid = build_periodic_becke_grid(
                system,
                grid_options=grid_options,
                image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
            )
            # Periodic-Becke grid pairs with the Γ-torus density (every
            # lattice block populated -> build_xc_periodic_uks cross-cell
            # mode); the home-cell-only set is the molecular-limit
            # density that pairs with the molecular grid below (the
            # 2026-07-09 KRKS finding class, spin-polarised flavour --
            # HANDOVER_AICCM_DIRECT_TORUS.md §4).
            from .periodic_rhf_gdf import _density_set_torus_gamma

            _set_xc_density = _density_set_torus_gamma
        else:
            grid = build_grid(system.unit_cell_molecule(), grid_options)
        D_alpha_set = compute_overlap_lattice(basis, system, lat_opts)
        D_beta_set = compute_overlap_lattice(basis, system, lat_opts)

    k_gamma = np.zeros(3)  # Γ-point Bloch phase for the V_xc fold below

    # ---- SCF loop -----------------------------------------------------
    def diagonalise(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Fp = X.T @ F @ X
        Fp = 0.5 * (Fp + Fp.T)
        if use_dav and dav_opts is not None:
            from vibeqc._vibeqc_core import davidson_solve

            if dav_opts.n_eig == 0:
                dav_opts.n_eig = Fp.shape[0]
            if dav_opts.guess_vectors is not None:
                pass  # already set from previous iteration
            dres = davidson_solve(Fp, dav_opts)
            if not dres.converged:
                raise RuntimeError(
                    f"Davidson did not converge after {dres.n_iter} iters"
                )
            eps, Cp = dres.eigenvalues, dres.eigenvectors
            dav_opts.guess_vectors = Cp
        else:
            eps, Cp = np.linalg.eigh(Fp)
        return X @ Cp, eps

    # Independent spin chemical potentials, with the shared global-Aufbau
    # ensemble convention when the temperature is zero.
    smear_T = float(getattr(opts, "smearing_temperature", 0.0) or 0.0)
    if smear_T < 0.0:
        raise ValueError("run_pbc_gdf_uks: smearing_temperature must be >= 0")
    from .smearing import (
        SmearingOptions as _SmearingOptions,
        smeared_occupation_selfconsistency_tolerance as _occ_tol_fn,
    )

    smear_opts = _SmearingOptions.from_legacy_kwarg(smear_T)
    _smeared_occ_tol = _occ_tol_fn(float(opts.conv_tol_energy))
    if compute_gradient:
        _reject_non_fermi_dirac_gradient("run_pbc_gdf_uks", smear_opts)

    C_alpha, eps_alpha = diagonalise(Hcore)
    C_beta, eps_beta = C_alpha.copy(), eps_alpha.copy()
    (D_alpha, D_beta, occ_a, occ_b, fermi_a, fermi_b, entropy_cur) = (
        _open_shell_gamma_occupy(
            C_alpha, eps_alpha, C_beta, eps_beta, n_alpha, n_beta, smear_opts
        )
    )
    D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()

    # Shared density-mode dispatch, with the same PATOM route-local in-field
    # refinement as UHF. HCORE alone retains the baseline density above.
    _atomic_spins = getattr(opts, "atomic_spins", None) or None
    _seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
    _split = initial_densities_open_shell(
        system.unit_cell_molecule(),
        basis,
        n_alpha,
        n_beta,
        _seed_guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=lat_opts,
        atomic_spins=_atomic_spins,
        read_density_alpha=getattr(opts, "read_density_alpha", None),
        read_density_beta=getattr(opts, "read_density_beta", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if _split is not None:
        D_alpha, D_beta = _split
        D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()
        plog.info(
            f"initial guess: {guess.name} (shared periodic adapter density)"
        )
        if guess == InitialGuess.PATOM:
            plog.info("initial guess: PATOM (SAD + one GDF in-field step)")
            J = _build_j_from_lpq(Lpq, D_alpha + D_beta)
            K_alpha = _build_k_from_lpq(Lpq, D_alpha)
            K_beta = _build_k_from_lpq(Lpq, D_beta)
            if exxdiv is PBCExxDiv.EWALD:
                K_alpha = apply_exxdiv_ewald_to_K(
                    [K_alpha], [S], [D_alpha], madelung
                )[0]
                K_beta = apply_exxdiv_ewald_to_K(
                    [K_beta], [S], [D_beta], madelung
                )[0]
            F_alpha_seed = 0.5 * (
                (Hcore + J - K_alpha) + (Hcore + J - K_alpha).T
            )
            F_beta_seed = 0.5 * (
                (Hcore + J - K_beta) + (Hcore + J - K_beta).T
            )
            C_alpha, eps_alpha = diagonalise(F_alpha_seed)
            C_beta, eps_beta = diagonalise(F_beta_seed)
            (
                D_alpha,
                D_beta,
                occ_a,
                occ_b,
                fermi_a,
                fermi_b,
                entropy_cur,
            ) = _open_shell_gamma_occupy(
                C_alpha,
                eps_alpha,
                C_beta,
                eps_beta,
                n_alpha,
                n_beta,
                smear_opts,
            )
            D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()

    damping = float(opts.damping)
    fock_mixing_value = float(getattr(opts, "fock_mixing", 0.0) or 0.0)
    if not (0.0 <= fock_mixing_value < 1.0):
        raise ValueError(
            "run_pbc_gdf_uks: fock_mixing must be in [0, 1); "
            f"got {fock_mixing_value}"
        )
    if fock_mixing_value != 0.0:
        plog.info(f"fock mixing: {100.0 * fock_mixing_value:.1f}% previous Fock")
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[PeriodicSCFAccelerator] = (
        PeriodicSCFAccelerator(opts) if use_diis else None
    )

    scf_trace: List[SCFIteration] = []
    result = PBCGDFUKSResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=float(e_nuc),
        e_coulomb=0.0,
        e_hf_exchange=0.0,
        e_xc=0.0,
        e_exxdiv=0.0,
        n_iter=0,
        converged=False,
        s_squared=0.0,
        s_squared_ideal=0.25 * (mult - 1) * (mult + 1),
        functional=func_name,
        mo_energies_alpha=np.empty(0),
        mo_coeffs_alpha=np.empty((0, 0)),
        density_alpha=D_alpha.copy(),
        fock_alpha=np.empty((0, 0)),
        mo_energies_beta=np.empty(0),
        mo_coeffs_beta=np.empty((0, 0)),
        density_beta=D_beta.copy(),
        fock_beta=np.empty((0, 0)),
        overlap=S,
        hcore=Hcore,
        scf_trace=scf_trace,
        aux_basis_name=aux_name,
        n_aux=int(aux.nbasis),
        n_fit=int(Lpq.shape[0]),
        madelung_constant=float(madelung),
        exxdiv=exxdiv.value,
        compcell_eta=float(compcell_eta),
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
        rsgdf_tail_ke_cutoff=(
            float(rsgdf_tail_ke_cutoff)
            if (rsgdf_tail_ke_cutoff is not None and gdf_method == "rsgdf")
            else None
        ),
        backend=_gdf_backend_with_parity_hold(
            f"pbc-gdf-{gdf_method}-{'uks' if is_ks else 'uhf'}",
            parity_held,
        ),
        fock_mixing=fock_mixing_value,
    )
    plog.banner(f"SCF (PBC-GDF {gdf_method} {'UKS ' + func_name if is_ks else 'UHF'})")

    E_prev = 0.0
    C_alpha_f, eps_alpha_f = C_alpha, eps_alpha
    C_beta_f, eps_beta_f = C_beta, eps_beta
    # SPINLOCK PATTERN_HOLD: see run_pbc_gdf_uhf. Hold the seeded broken-symmetry
    # occupied set by MOM for cycles 2..spinlock_iterations, then release. The
    # no-smearing occupy is column-order, so reordering the held occupied to the
    # front makes the fill pick them up.
    from .mom import reorder_occupied_by_max_overlap as _mom_reorder

    _pattern_hold = (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.PATTERN_HOLD
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    )
    _spinlock_iters = int(getattr(opts, "spinlock_iterations", 0))
    _ca_occ_prev = None
    _cb_occ_prev = None
    F_alpha_prev_mixed: Optional[np.ndarray] = None
    F_beta_prev_mixed: Optional[np.ndarray] = None
    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        # SPINLOCK PATTERN_HOLD: the accelerator (DIIS / EDIIS / ADIIS /
        # KDIIS -- whatever PeriodicSCFAccelerator resolved) is suspended
        # (no history recorded, no extrapolation, damping stays live)
        # while the hold is active. Fock extrapolation across held-window
        # iterates steers the SCF toward the symmetric attractor by
        # continuous orbital rotation -- a collapse the occupation-selecting
        # MOM hold cannot see -- and poisons the post-release history with
        # out-of-basin iterates. The history starts fresh at release.
        hold_active = _pattern_hold and iter_idx <= _spinlock_iters
        diis_active = (
            use_diis and iter_idx >= diis_start_iter and not hold_active
        )
        if iter_idx == 1 or damping == 0.0 or diis_active:
            D_alpha_used, D_beta_used = D_alpha, D_beta
        else:
            D_alpha_used = damping * D_alpha_prev + (1.0 - damping) * D_alpha
            D_beta_used = damping * D_beta_prev + (1.0 - damping) * D_beta

        J = _build_j_from_lpq(Lpq, D_alpha_used + D_beta_used)
        K_alpha = _build_k_from_lpq(Lpq, D_alpha_used) if alpha != 0.0 else None
        K_beta = _build_k_from_lpq(Lpq, D_beta_used) if alpha != 0.0 else None
        e_exx = 0.0
        if alpha != 0.0 and exxdiv is PBCExxDiv.EWALD:
            K_alpha = apply_exxdiv_ewald_to_K([K_alpha], [S], [D_alpha_used], madelung)[
                0
            ]
            K_beta = apply_exxdiv_ewald_to_K([K_beta], [S], [D_beta_used], madelung)[0]
            e_exx = exxdiv_ewald_energy_shift(
                [D_alpha_used], [S], madelung, hf_exchange_fraction=alpha, weights=[1.0]
            ) + exxdiv_ewald_energy_shift(
                [D_beta_used], [S], madelung, hf_exchange_fraction=alpha, weights=[1.0]
            )

        F_HF_alpha = J - alpha * K_alpha if K_alpha is not None else J
        F_HF_beta = J - alpha * K_beta if K_beta is not None else J

        E_xc = 0.0
        V_xc_alpha = V_xc_beta = 0.0
        if is_ks:
            _set_xc_density(D_alpha_set, D_alpha_used)
            _set_xc_density(D_beta_set, D_beta_used)
            xc = build_xc_periodic_uks(
                basis, system, grid, func, D_alpha_set, D_beta_set, lat_opts
            )
            V_xc_alpha = np.real(bloch_sum(xc.V_alpha, k_gamma))
            V_xc_beta = np.real(bloch_sum(xc.V_beta, k_gamma))
            V_xc_alpha = 0.5 * (V_xc_alpha + V_xc_alpha.T)
            V_xc_beta = 0.5 * (V_xc_beta + V_xc_beta.T)
            E_xc = float(xc.e_xc)

        F_alpha = Hcore + F_HF_alpha + V_xc_alpha
        F_beta = Hcore + F_HF_beta + V_xc_beta
        F_alpha = 0.5 * (F_alpha + F_alpha.T)
        F_beta = 0.5 * (F_beta + F_beta.T)

        E_core = float(np.einsum("ij,ij->", D_alpha_used + D_beta_used, Hcore))
        E_J = 0.5 * float(np.einsum("ij,ij->", D_alpha_used + D_beta_used, J))
        E_K = 0.0
        if K_alpha is not None:
            E_K = (
                -0.5
                * alpha
                * (
                    float(np.einsum("ij,ij->", D_alpha_used, K_alpha))
                    + float(np.einsum("ij,ij->", D_beta_used, K_beta))
                )
            )
        E_elec = E_core + E_J + E_K + E_xc
        E_total = E_elec + float(e_nuc)

        FDS_a = F_alpha @ D_alpha_used @ S
        FDS_b = F_beta @ D_beta_used @ S
        grad_a = FDS_a - FDS_a.T
        grad_b = FDS_b - FDS_b.T
        grad_norm = float(
            np.sqrt(np.linalg.norm(grad_a) ** 2 + np.linalg.norm(grad_b) ** 2)
        )
        dE = E_total - E_prev
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        if converged and not hold_active:
            # Smeared convergence additionally requires the stored
            # occupations to be the Fermi filling of the current plain
            # Fock's own eigenvalues -- the energy + commutator tests
            # cannot see a frozen-occupation fixed point on
            # zero-commutator fixtures (see
            # _smeared_occupation_residual_gamma).
            occ_residual = _smeared_occupation_residual_gamma(
                diagonalise,
                F_alpha,
                F_beta,
                occ_a,
                occ_b,
                n_alpha,
                n_beta,
                smear_opts,
            )
            if occ_residual > _smeared_occ_tol:
                converged = False
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )

        result.energy = E_total
        result.e_electronic = E_elec
        result.e_coulomb = E_J
        result.e_hf_exchange = E_K
        result.e_xc = E_xc
        result.e_exxdiv = e_exx
        result.n_iter = iter_idx
        result.mo_energies_alpha = eps_alpha_f
        result.mo_coeffs_alpha = C_alpha_f
        result.density_alpha = D_alpha_used
        result.fock_alpha = F_alpha
        result.mo_energies_beta = eps_beta_f
        result.mo_coeffs_beta = C_beta_f
        result.density_beta = D_beta_used
        result.fock_beta = F_beta
        result.smearing_temperature = smear_T
        from .periodic_k_density import _fermi_density_entropy
        entropy_used = (
            _fermi_density_entropy([D_alpha_used], [S], [X], [1.0], 1.0)
            + _fermi_density_entropy([D_beta_used], [S], [X], [1.0], 1.0)
            if smear_T > 0.0 else 0.0
        )
        result.entropy = entropy_used
        result.free_energy = E_total - smear_T * entropy_used
        result.fermi_level_alpha = float(fermi_a)
        result.fermi_level_beta = float(fermi_b)
        result.occupations_alpha = np.asarray(occ_a, dtype=float)
        result.occupations_beta = np.asarray(occ_b, dtype=float)

        if converged:
            result.converged = True
            result.s_squared = _gamma_open_shell_s_squared(
                n_alpha, n_beta, C_alpha_f, C_beta_f, S, occ_a, occ_b, smear_opts
            )
            if compute_gradient:
                # T = 0: integer slices (bit-identical). T > 0: Mermin
                # free-energy blocks (sqrt(f)-scaled columns; the
                # gradient is dA/dR of result.free_energy -- see
                # _gamma_gradient_orbital_blocks). The XC Pulay below
                # is density-based, so the fractional D_s flows into it
                # unchanged.
                C_a_grad, eps_a_grad = _gamma_gradient_orbital_blocks(
                    C_alpha_f, eps_alpha_f, n_alpha, occ_a, smear_opts
                )
                C_b_grad, eps_b_grad = _gamma_gradient_orbital_blocks(
                    C_beta_f, eps_beta_f, n_beta, occ_b, smear_opts
                )
            if compute_gradient and gdf_method == "rsgdf":
                # G-PBC-002 M6 rung 5b: the rsgdf per-spin assembly with
                # the functional's alpha (tailed and Schwarz-screened
                # envelopes included since 2026-07-30); the spin XC
                # Pulay is added below.
                from .periodic_gdf_gradient import compute_gdf_gradient_rsgdf_uhf_gamma

                rsgdf_grad_cache = setup.range_separated_cache
                if int(rsgdf_grad_cache.n_fit) != int(result.n_fit):
                    raise RuntimeError(
                        "run_pbc_gdf_uks: rsgdf gradient-cache fit rank "
                        f"({rsgdf_grad_cache.n_fit}) does not match the "
                        f"converged SCF fit ({result.n_fit})."
                    )
                grad_total = compute_gdf_gradient_rsgdf_uhf_gamma(
                    system,
                    basis,
                    D_alpha=D_alpha_used,
                    D_beta=D_beta_used,
                    C_alpha_occ=C_a_grad,
                    C_beta_occ=C_b_grad,
                    eps_alpha_occ=eps_a_grad,
                    eps_beta_occ=eps_b_grad,
                    S=S,
                    cache=rsgdf_grad_cache,
                    lattice_opts=setup.oneel_lat_opts or lat_opts,
                    gauge_lat_opts=setup.gauge_lat_opts,
                    madelung=(
                        float(madelung)
                        if exxdiv == PBCExxDiv.EWALD
                        else 0.0
                    ),
                    v_ne_ke_cutoff=float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0")),
                    alpha_hf=(alpha if is_ks else 1.0),
                )
                if is_ks:
                    from ._vibeqc_core import (
                        xc_lattice_gradient_contribution_uks,
                    )

                    _set_xc_density(D_alpha_set, D_alpha_used)
                    _set_xc_density(D_beta_set, D_beta_used)
                    grad_total = grad_total + np.asarray(
                        xc_lattice_gradient_contribution_uks(
                            basis,
                            system,
                            grid,
                            func,
                            D_alpha_set,
                            D_beta_set,
                            lat_opts,
                        )
                    )
                result.gradient = grad_total
            elif compute_gradient:
                # G-PBC-002 milestone 4: the per-spin compcell assembly
                # with the functional's alpha, plus the spin-polarised
                # XC Pulay on the SCF's own grid/density-set conventions
                # (KS only -- functional=None reproduces the UHF wiring).
                from .aux_basis import make_aux_basis_set
                from .periodic_gdf_gradient import (
                    _build_compcell_gradient_cache,
                    compute_gdf_gradient_uhf_gamma,
                )

                aux_unscaled = make_aux_basis_set(
                    system.unit_cell_molecule(), aux_name=aux_name
                )
                grad_cache = _build_compcell_gradient_cache(
                    system,
                    basis,
                    aux_unscaled,
                    compcell_eta=float(compcell_eta),
                    lattice_opts=lat_opts,
                    linear_dep_thr=float(gdf_linear_dep_threshold),
                    rcut_strategy=rcut_strategy,
                    rcut_precision=float(rcut_precision),
                    fit_state=setup.compcell_fit_state,
                )
                uks_v_ne_ke_cutoff = float(
                    os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0")
                )
                grad_total = compute_gdf_gradient_uhf_gamma(
                    system,
                    basis,
                    D_alpha=D_alpha_used,
                    D_beta=D_beta_used,
                    C_alpha_occ=C_a_grad,
                    C_beta_occ=C_b_grad,
                    eps_alpha_occ=eps_a_grad,
                    eps_beta_occ=eps_b_grad,
                    S=S,
                    cache=grad_cache,
                    lattice_opts=lat_opts,
                    gauge_lat_opts=setup.gauge_lat_opts,
                    madelung=(
                        float(madelung)
                        if exxdiv == PBCExxDiv.EWALD
                        else 0.0
                    ),
                    v_ne_ke_cutoff=uks_v_ne_ke_cutoff,
                    alpha_hf=(alpha if is_ks else 1.0),
                )
                if is_ks:
                    from ._vibeqc_core import (
                        xc_lattice_gradient_contribution_uks,
                    )

                    _set_xc_density(D_alpha_set, D_alpha_used)
                    _set_xc_density(D_beta_set, D_beta_used)
                    grad_total = grad_total + np.asarray(
                        xc_lattice_gradient_contribution_uks(
                            basis,
                            system,
                            grid,
                            func,
                            D_alpha_set,
                            D_beta_set,
                            lat_opts,
                        )
                    )
                result.gradient = grad_total
            if check_energy_sanity:
                _check_energy_sanity(
                    result,
                    system,
                    plog,
                    driver="run_pbc_gdf_uks",
                    raise_if_converged=True,
                )
            plog.converged(n_iter=iter_idx, energy=E_total, converged=True)
            return result

        # Skipped entirely (not even recorded) while the PATTERN_HOLD window
        # is active; see the hold_active note at the loop head.
        if accel is not None and not hold_active:
            F_alpha_ex, F_beta_ex = accel.extrapolate_uhf(
                F_alpha,
                F_beta,
                error_alpha=grad_a,
                error_beta=grad_b,
                density_alpha=D_alpha_used,
                density_beta=D_beta_used,
                energy=E_total,
                mo_coeffs_alpha=C_alpha_f,
                mo_coeffs_beta=C_beta_f,
                mo_energies_alpha=eps_alpha_f,
                mo_energies_beta=eps_beta_f,
                n_alpha=n_alpha,
                n_beta=n_beta,
            )
            if diis_active:
                F_alpha, F_beta = F_alpha_ex, F_beta_ex
        if fock_mixing_value != 0.0:
            if F_alpha_prev_mixed is not None and F_beta_prev_mixed is not None:
                F_alpha_mix = (
                    (1.0 - fock_mixing_value) * F_alpha
                    + fock_mixing_value * F_alpha_prev_mixed
                )
                F_beta_mix = (
                    (1.0 - fock_mixing_value) * F_beta
                    + fock_mixing_value * F_beta_prev_mixed
                )
                F_alpha = 0.5 * (F_alpha_mix + F_alpha_mix.T)
                F_beta = 0.5 * (F_beta_mix + F_beta_mix.T)
            F_alpha_prev_mixed = F_alpha.copy()
            F_beta_prev_mixed = F_beta.copy()

        C_alpha_f, eps_alpha_f = diagonalise(F_alpha)
        C_beta_f, eps_beta_f = diagonalise(F_beta)
        if _pattern_hold and 1 < iter_idx <= _spinlock_iters:
            if n_alpha > 0 and _ca_occ_prev is not None:
                C_alpha_f, eps_alpha_f = _mom_reorder(
                    C_alpha_f, eps_alpha_f, S, _ca_occ_prev, n_alpha
                )
            if n_beta > 0 and _cb_occ_prev is not None:
                C_beta_f, eps_beta_f = _mom_reorder(
                    C_beta_f, eps_beta_f, S, _cb_occ_prev, n_beta
                )
        if _pattern_hold and iter_idx <= _spinlock_iters:
            _ca_occ_prev = (
                np.asarray(C_alpha_f[:, :n_alpha]).copy() if n_alpha > 0 else None
            )
            _cb_occ_prev = (
                np.asarray(C_beta_f[:, :n_beta]).copy() if n_beta > 0 else None
            )
        D_alpha_prev, D_beta_prev = D_alpha_used, D_beta_used
        (D_alpha, D_beta, occ_a, occ_b, fermi_a, fermi_b, entropy_cur) = (
            _open_shell_gamma_occupy(
                C_alpha_f,
                eps_alpha_f,
                C_beta_f,
                eps_beta_f,
                n_alpha,
                n_beta,
                smear_opts,
                preserve_column_order=hold_active,
            )
        )
        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

    result.s_squared = float(
        _gamma_open_shell_s_squared(
            n_alpha, n_beta, C_alpha_f, C_beta_f, S, occ_a, occ_b, smear_opts
        )
    )
    result.converged = False
    if check_energy_sanity:
        _check_energy_sanity(
            result,
            system,
            plog,
            driver="run_pbc_gdf_uks",
            raise_if_converged=True,
        )
    plog.converged(n_iter=result.n_iter, energy=result.energy, converged=False)
    return result


# Positive-energy slack for the Γ-GDF energy-sanity guard (Ha). A bound
# neutral cell has E_total < 0, but cramped/artificial Bravais smoke-test
# cells can converge just above zero (the 8-bohr hexagonal H₂ coverage cell
# lands at +0.067 Ha); a positive energy beyond this slack is unphysical.
# Same convention as vibeqc.periodic_k_gdf.POSITIVE_E_SLACK_HA.
POSITIVE_E_SLACK_HA = 1.0


def _check_energy_sanity(
    result: PBCGDFResult,
    system: PeriodicSystem,
    plog: ProgressLogger,
    *,
    driver: str = "run_pbc_gdf_rhf",
    raise_if_converged: bool = False,
) -> None:
    """Post-condition: SCF total energy is in a physically sensible
    range relative to the unit cell's atomic content.

    Two failure signatures are rejected, mirroring the multi-k guard
    (:func:`vibeqc.periodic_k_gdf._check_energy_sanity`):

    * **Runaway** -- ``|E_total|`` beyond ``max(10.S Z^2, 100)`` Ha (the
      loose hydrogenic bound on absolute binding, factor-10 slack).
      Seen on tight ionic crystals when the Γ-only compcell Hartree is
      inconsistent (the LiH FCC primitive cell converges to +579.8 Ha
      at RHF/UHF and +1172.6 Ha at UKS-PBE vs PySCF's -8.24 Ha) or the
      AFT convention is mismatched (1e4-1e6 Ha range) -- the SCF
      "converges" to a numerical fixed point of the broken Fock that
      has no physical meaning.
    * **Unbound** -- ``E_total`` positive beyond
      ``POSITIVE_E_SLACK_HA``. A bound neutral cell has
      ``E_total < 0``; small positive energies are tolerated because
      cramped/artificial Bravais smoke-test cells can legitimately
      converge just above zero.

    Callers with pinned warn-and-tag behaviour (``run_pbc_gdf_rhf``;
    see ``test_pbc_gdf_rhf_production_defaults_lih_ionic``) keep the
    default ``raise_if_converged=False``: warn loudly + tag the
    backend ``+SANITY_FAILED``, but hand the value back for parity
    diagnostics. The open-shell/KS drivers pass
    ``raise_if_converged=True``: a non-physical energy reported as
    *converged* is the silent-corruption pattern CLAUDE.md §7 warns
    about, so they RAISE instead of returning it (bypass with
    ``check_energy_sanity=False``). A non-converged run warns + tags
    either way (``converged=False`` already signals failure).
    """
    E = float(result.energy)
    z_sum_sq = sum(atom.Z**2 for atom in system.unit_cell)
    sane_bound = max(10.0 * z_sum_sq, 100.0)  # loose hydrogenic + floor
    runaway = abs(E) > sane_bound
    unbound = E > POSITIVE_E_SLACK_HA
    if not (runaway or unbound):
        return

    reason = "runaway divergence" if runaway else "positive (unbound) total energy"
    msg = (
        f"{driver}: SCF total energy {E:.6e} Ha is non-physical "
        f"({reason}). A bound, neutral cell has E_total < 0 (a positive "
        f"energy beyond {POSITIVE_E_SLACK_HA:g} Ha slack is unbound) and "
        f"|E_total| < {sane_bound:.2e} Ha (loose hydrogenic bound on "
        "S_atoms Z^2/2). The SCF has converged to a numerical fixed point "
        "of a broken Fock (CLAUDE.md Sec.7 -- not a convergence-aid "
        "problem). Known trigger: Gamma-only gdf_method='compcell' on a "
        "tight ionic cell, where the Gamma-only Hartree (G=0 dropped) "
        "cannot resolve overlapping AO-pair images -- use the multi-k "
        "route (run_krhf/run_kuhf/run_kuks_periodic_gdf, validated at uHa "
        "parity on LiH FCC at kmesh=(2,2,2)) or gdf_method='rsgdf' at "
        "Gamma. Pass check_energy_sanity=False to bypass this guard "
        "(diagnostics only)."
    )
    try:
        result.backend = result.backend + "+SANITY_FAILED"
    except Exception:
        pass
    if raise_if_converged and result.converged:
        raise RuntimeError(msg)
    plog.info("  WARNING: " + msg)
