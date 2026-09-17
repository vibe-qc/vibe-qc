"""Auxiliary bases and periodic Gaussian density-fitting sources.

The production routes include compensated Gaussian fitting, mixed Gaussian /
plane-wave fitting and the historical all-reciprocal ``rsgdf`` builders.
The latter name does not imply a real-space short-range build: its current
production implementation evaluates a bare reciprocal kernel and may add a
high-energy reciprocal tail.

The private replacement source combines native real-space erfc integrals
with a compact reciprocal erf sum. It shares the finite zero-mode convention,
cutoff planner, factorization and cache admission between Gamma and multi-k.
See ``docs/design_native_gdf.md`` for equations, memory scope, validation and
the production integration still required. Auxiliary normalization and
absolute metric thresholds are explicit parts of each fitting contract.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
from scipy.special import gamma as _scipy_gamma

from ._vibeqc_core import (
    BasisSet,
    LatticeSumOptions,
    Molecule,
    PeriodicSystem,
    ShellInfo,
    compute_2c_eri_lattice,
    compute_2c_eri_lattice_blocks,
    compute_2c_eri_lattice_sr,
    compute_3c_eri_lattice,
    compute_3c_eri_lattice_blocks,
    compute_3c_eri_lattice_sr,
    compute_overlap_lattice,
)
from .periodic_gdf_blocks import (
    bloch_sum_2c_eri_blocks,
    bloch_sum_3c_eri_blocks,
)

_logger = logging.getLogger(__name__)


# --- Fitting-metric linear-dependence convention -------------------------
#
# Every density-fitting builder in this module discards near-null modes of
# its 2-centre fitting metric before forming the ``1/sqrt(lambda)``
# orthogonaliser. The threshold is ABSOLUTE: a mode is dropped when its raw
# eigenvalue falls below ``linear_dep_thr``, with no normalisation by the
# largest eigenvalue.
#
# Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
# doi:10.1063/1.4998644, Sec. II B (immediately after Eq. 21) prescribes only
# "diagonalize this singular matrix and remove the eigenvectors associated
# with small eigenvalues below a threshold" -- it states no normalisation.
# The authors' own reference implementation resolves the ambiguity: PySCF
# compares the raw eigenvalue of the un-normalised 2-centre Coulomb metric
# (``pyscf/pbc/df/rsdf_builder.py``, ``decompose_j2c`` /
# ``eigenvalue_decomposed_metric``), through one shared
# ``linear_dep_threshold`` used identically by its GDF, MDF and RSGDF
# builders. PySCF has no relative (max-eigenvalue-scaled) convention for a
# fitting metric anywhere.
#
# vibe-qc previously mixed the two: the FFT/rsgdf builders compared
# absolutely while the compcell and MDF builders compared against
# ``linear_dep_thr * max_eig``. On a metric with ``max_eig ~ 10`` that made
# the same number mean two different cuts an order of magnitude apart.
# Harmonised here (2026-08-14).
def _metric_keep_mask(
    eigvals: np.ndarray,
    linear_dep_thr: float,
    min_eig_fraction: float = 0.0,
) -> np.ndarray:
    """Absolute linear-dependence mask for a 2-centre fitting metric.

    Modes with ``lambda <= linear_dep_thr`` are discarded. See the module
    note above for the convention and its provenance.

    ``min_eig_fraction`` is an optional NOISE FLOOR, applied as
    ``thr = max(linear_dep_thr, min_eig_fraction * max_eig)``. It is a
    separate concept from the threshold and is deliberately scale-aware:
    the quantity it guards against is the metric's own CONSTRUCTION ERROR,
    which scales with the metric, whereas the threshold expresses the
    user's accuracy request in absolute terms. Each builder passes the
    floor appropriate to the matrix it decomposes (see
    :data:`_PLAIN_METRIC_MIN_EIG_FRACTION` and
    :data:`_MDF_DRESSED_METRIC_MIN_EIG_FRACTION`); ``0.0`` means no floor.
    """
    w = np.asarray(eigvals)
    thr = float(linear_dep_thr)
    frac = float(min_eig_fraction)
    if w.size and frac > 0.0:
        thr = max(thr, frac * float(w[-1]))
    return w > thr


# Noise floor for the PLAIN compensated Coulomb metric decomposed by the
# compcell and bare-native builders. Set to preserve exactly the effective
# cut those builders were validated at: they historically compared
# ``eigvals > linear_dep_thr * max_eig``, which is identical to an absolute
# comparison carrying this floor at the shipped 1e-9 default. Keeping it
# matters -- measured over 88 metric decompositions in the compcell +
# periodic-RHF-GDF suites, dropping it changed the retained rank in exactly
# one case (LiH ionic FCC, max_eig 2.814: 61 -> 62 modes), i.e. it would
# have admitted one extra mode with an eigenvalue between 1e-9 and 2.8e-9,
# whose 1/sqrt(lambda) amplification is ~3e+04. That is the same class of
# contamination this module's MDF floor exists to remove, so the plain
# metric keeps its floor rather than silently loosening.
#
# The rsgdf/FFT builders are deliberately NOT given a floor: they already
# compared absolutely before the 2026-08-14 harmonisation and are validated
# at that behaviour.
_PLAIN_METRIC_MIN_EIG_FRACTION = 1.0e-9


# Safety floor for the PW-DRESSED MDF metric ``J-tilde`` of Eq. 20 (Sun
# 2017), which is the plain Coulomb metric with the plane-wave space
# projected out. Sec. II B: "Projecting the PWs out of the Gaussian
# functions in (20) leads to a highly singular matrix" -- so J-tilde is
# deliberately rank-deficient, unlike the plain metric the compcell/rsgdf
# builders fit against, and it cannot carry that metric's threshold.
#
# WHY THIS FLOOR IS SCALE-AWARE WHILE THE THRESHOLD IS NOT. The user-facing
# ``linear_dep_thr`` is absolute (see _metric_keep_mask). The floor is
# expressed as a FRACTION OF ``max_eig`` because the thing it protects
# against -- J-tilde's own construction error -- scales with the metric,
# and J-tilde's scale varies enormously with the cell. Measured max_eig:
# 1.08e+01 (MgO/STO-3G primitive), 8.09e+00 (Ne/STO-3G 10-bohr box) but
# 9.17e-06 (H2/STO-3G 12-bohr box), where the plane waves nearly span the
# Gaussian aux space and almost nothing survives the projection. A bare
# absolute floor calibrated on the first two annihilates the third.
#
# The floor is a vibe-qc calibration, NOT a value from either paper (Sun
# 2017 Sec. II B quotes a 1e-9 default and Sec. III finds 1e-10 best;
# PySCF ships 1e-10).
#
# It USED to be set by J-tilde's construction error. At the general
# lattice-sum precision the MgO/STO-3G/def2-svp-jk spectrum carried up to
# 5.4e-06 absolute error, and the smallest retained eigenvalue at the old
# default was 2.9e-08 against its own 6.6e-08 error bar -- a mode smaller
# than its uncertainty, divided by its own square root. Since 2026-09-17
# that is no longer the binding constraint: :data:`_MDF_PRECISION_J2C`
# builds J-tilde to 1.8e-13 on the same cell (see
# :func:`_mdf_fit_lattice_opts`).
#
# THE FLOOR STAYS ANYWAY, and the reason changed. Measured on MgO with
# the accurate metric, sweeping the absolute threshold with this floor
# disabled entirely:
#
#   3e-2   -271.1420    -92.1 mHa     <- the loosest useful cut
#   1e-3   -274.2850  -3235.1 mHa
#   1e-5   -336.4730  -65423   mHa
#   1e-8   -64954.85          diverged, non-converged
#
# i.e. the small-lambda modes are still unusable, and they are unusable
# for a reason a better-built metric cannot touch (see
# ``pbc_gdf._reject_dense_core_mdf``). The floor is what keeps them out.
#
# Calibration, Gamma RHF against out-of-process PySCF MDF oracles (see
# tests/test_pbc_gdf_mdf.py::test_mdf_linear_dep_threshold_ne_box_sweep):
#
#   Ne/STO-3G/10-bohr box (the validated envelope), oracle -126.61361316,
#   as a fraction of max_eig:
#     1e-09 (old cut)   -126.6136772952   -0.064 mHa
#     1e-05 .. 1e-03    -126.6136107693   +0.0024 mHa   <- accuracy plateau
#     1e-02             -126.6127728140   +0.840 mHa    (too loose: Sun
#                                                        Sec. III
#                                                        incompleteness)
#   MgO/STO-3G primitive (gated dense-core class), oracle -271.0499046:
#     1e-09 (old cut)   +467152.38        divergent, trial-dependent
#     1e-03               -271.145742     bounded
#
# 1e-4 sits mid-plateau for Ne -- a 26x improvement in PySCF-MDF parity
# over the old cut -- removes the divergence on the dense-core class, and
# leaves the dilute H2 box untouched (1e-4 x 9.17e-06 = 9.2e-10, below the
# 1e-9 default, so max() keeps the caller's value).
_MDF_DRESSED_METRIC_MIN_EIG_FRACTION = 1.0e-4


# vibe-qc's analogue of PySCF's ``precision_j2c``: the accuracy the
# PW-dressed MDF metric ``J-tilde`` (Sun 2017 Eq. 20) is built to, which is
# NOT the general cell precision. The fit is ``W = T^T J-tilde^{-1} T``, so
# an error dJ in the metric reaches the fitted tensor as ``|T|^2 dJ /
# lambda^2`` -- two inverse powers of the very eigenvalues the
# linear-dependence cut is deciding about. No mode can be retained below
# the accuracy of the matrix it is an eigenvalue of. PySCF makes the same
# separation and for the same reason (``pyscf/pbc/df/df.py``,
# ``precision_j2c``), which is what lets it ship a 1e-10 linear-dependence
# default where a metric built at the general precision cannot support one.
#
# 1e-12 is four orders below the 1e-8 general default and costs little: a
# Gaussian lattice sum's radius grows only as ``sqrt(log(1/precision))``,
# so buying the four orders (plus the cell-count budget of
# :func:`_mdf_fit_lattice_opts`) moved MgO's metric rcut 15.5 -> 28.9 bohr
# and no measured MDF run got measurably slower. The 2-centre metric is
# the smallest of the three lattice sums MDF runs (n_aux^2 elements
# against the 3-centre's n_aux.n_orb^2), so this is the cheap place to buy
# accuracy.
#
# Delivered, against a converged 45-bohr reference (2026-09-17):
#
#              max|dJ_PQ|            max|dlambda|
#              1e-8 rcut   here      1e-8 rcut   here
#   MgO        4.51e-06  1.14e-13    5.42e-06  1.82e-13
#   LiH        4.56e-06  1.14e-13    3.79e-06  1.32e-13
#   H2 box     3.19e-09  4.44e-15    3.46e-09  6.00e-15
#   Ne box     3.84e-10  3.02e-14    2.26e-10  4.09e-14
#
# The H2 box row is the one with a live consequence for a SUPPORTED class:
# its 3.46e-09 spectrum error exceeded the 1e-9 threshold that was meant
# to cut on it, so modes inside the noise were being retained in
# production. Ne, whose floor sat far above its error either way, is
# unchanged below a nanohartree.
_MDF_PRECISION_J2C = 1.0e-12

# The three-centre companion, DEFAULT OFF (None = build the 3-centre at
# the caller's ``rcut_precision``, as before).
#
# It exists because ``L = U_keep^T (T - PW proj) / sqrt(lambda)`` divides
# the 3-centre by the same square root that divides the metric, so on
# paper a retained mode amplifies the 3-centre's construction error too.
# MEASURED 2026-09-17, and the paper argument does not bite at any
# threshold actually reachable: switching it on at 1e-10 leaves
# Ne/STO-3G/10-bohr identical to below a nanohartree and H2/STO-3G/12-bohr
# identical to all printed digits, and on MgO/STO-3G it left every energy
# in a 3e-2 .. 1e-10 threshold sweep unchanged while costing ~20% of the
# run. The metric's error is amplified by ``1/lambda`` and the 3-centre's
# only by ``1/sqrt(lambda)``, and that one extra power is the whole
# difference: at the thresholds the dressed metric admits, the 3-centre at
# ``rcut_precision`` is already well inside its budget.
#
# Kept as a knob rather than deleted because the reasoning is sound and a
# future tighter-admitting fit would need it; do not switch it on without
# re-measuring, since the 3-centre is the expensive sum of the three
# (n_aux.n_orb^2 elements over the joint AO-union-fused basis, whose rcut
# is already the longest).
_MDF_PRECISION_J3C = None


def _mdf_fit_lattice_opts(
    basis: BasisSet,
    system: PeriodicSystem,
    base_opts: "LatticeSumOptions",
    precision: float,
    multiplicity: int,
) -> "LatticeSumOptions":
    """Lattice-sum options for one half of the PW-dressed MDF fit.

    Uses :class:`~vibeqc.lattice_screening.RcutStrategy.ACCUMULATED`, so
    ``precision`` bounds the error of the SUMMED tensor rather than of its
    largest dropped term -- the distinction that made the shipped
    ``precision = 1e-8`` deliver 2.4e-07 on MgO and 7.1e-06 on LiH
    rocksalt. ``multiplicity`` carries the bound from the element error to
    the error in what is actually used: for the metric that is Weyl's
    ``|dlambda| <= ||dJ||_2 <= n_aux max|dJ_PQ|`` over its eigenvalues, and
    for the 3-centre the same row count over its contraction.
    """
    from .lattice_screening import RcutStrategy, make_lattice_opts

    return make_lattice_opts(
        basis,
        strategy=RcutStrategy.ACCUMULATED,
        base_opts=base_opts,
        precision=float(precision),
        system=system,
        extra_multiplicity=float(max(1, int(multiplicity))),
    )


def _mdf_joint_basis(
    ao_basis: BasisSet, fused: BasisSet, molecule: Molecule
) -> BasisSet:
    """The AO-union-fused basis whose reach sizes the MDF 3-centre sum."""
    shells = [
        ShellInfo(
            int(s.atom_index), int(s.l), bool(s.pure),
            list(s.exponents), list(s.coefficients), list(s.origin),
        )
        for s in list(ao_basis.shells()) + list(fused.shells())
    ]
    return BasisSet(molecule, shells, "<ao∪fused>", True)


def _mdf_dressed_metric_threshold(
    linear_dep_thr: float, n_pw: int, max_eig: float, where: str
) -> float:
    """Resolve the discard threshold for the PW-dressed MDF metric.

    ``linear_dep_thr`` is the shared GDF kwarg, whose default is sized for
    the plain Coulomb metric. When the PW block is active the metric being
    decomposed is J-tilde (Eq. 20), not that metric, so the value is raised
    to :data:`_MDF_DRESSED_METRIC_MIN_EIG_FRACTION` x ``max_eig`` when it
    falls below what J-tilde's construction accuracy supports. A caller may
    still go *looser* (the dense-core class needs ~3e-2).

    With the PW block empty (``mdf_ke_cutoff <= 0``) there is no dressing:
    J-tilde is the plain metric, and the caller's value is honoured
    unchanged so the compcell-equivalent limit stays bit-identical.
    """
    thr = float(linear_dep_thr)
    if int(n_pw) <= 0:
        return thr
    floor = _MDF_DRESSED_METRIC_MIN_EIG_FRACTION * float(max_eig)
    if thr >= floor:
        return thr
    _logger.info(
        "%s: raising the linear-dependence threshold %.1e -> %.1e for the "
        "PW-dressed metric (Eq. 20 is deliberately singular; modes below "
        "this are inside J-tilde's own construction error).",
        where,
        thr,
        floor,
    )
    return floor


def _basis_fingerprint(basis: BasisSet) -> str:
    """Return a stable content fingerprint for a native ``BasisSet``."""
    digest = hashlib.sha256(b"vibeqc-basis-fingerprint-v1\0")
    for shell in basis.shells():
        header = (
            int(shell.atom_index),
            int(shell.l),
            int(bool(shell.pure)),
            len(shell.exponents),
            len(shell.coefficients),
        )
        digest.update(":".join(str(value) for value in header).encode("ascii"))
        digest.update(b"\0")
        for values in (shell.exponents, shell.coefficients, shell.origin):
            array = np.asarray(values, dtype="<f8")
            digest.update(str(array.size).encode("ascii"))
            digest.update(b":")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _progress_info(progress: Any, message: str) -> None:
    if progress is None:
        return
    info = getattr(progress, "info", None)
    if info is None:
        return
    info(message)


__all__ = [
    "default_aux_for",
    "make_aux_basis_set",
    "build_lpq_bloch_native",
    "build_lpq_native",
    "modrho_scales",
    # Compcell pipeline (PySCF _CCGDFBuilder analogue; Sun 2017
    # DOI 10.1063/1.4998644). The fix for diffuse-aux divergence of
    # the bare image sum.
    "make_modrho_aux_basis",
    "make_compensating_basis",
    "make_fused_basis",
    "fuse_transform_matrix",
    "build_lpq_compcell",
    "build_lpq_bloch_compcell",
    # Historical LR helpers and all-reciprocal production builders.
    # The private replacement source is documented below and in
    # docs/design_native_gdf.md (Ye/Berkelbach 2021).
    "rsgdf_g_mesh",
    "rsgdf_aux_fourier_transform",
    "rsgdf_lr_2c_metric",
    "rsgdf_lr_3c_tensor",
    "rsgdf_dense_g_mesh",
    "build_lpq_native_fft",
    "build_lpq_bloch_native_fft",
    "build_lpq_bloch_native_fft_shared_q",
]


# The old sparse-mesh prototype and the current all-reciprocal route
# are distinct implementations. Historical notes attributed the sparse
# prototype's dense-cell failure to an inherent need for a tight-core
# reciprocal mesh even with real-space SR integrals. That diagnosis is
# incorrect for the decomposition in Ye/Berkelbach 2021, Eqs. (12)-(13):
# the LR Gaussian factor damps reciprocal contributions independently of
# the tightest orbital exponent; two independent real-space translations
# and consistent finite-zero-mode subtraction complete the fit. The new
# native source implements that decomposition. Exchange Madelung remains
# a later Fock correction, separate from the integral zero-mode convention.
# Historical prototype evidence remains in
# handovers/HANDOVER_GDF_V0_11_2026_05_29.md and git history.


# ============================================================
# Default aux basis pairing (mirror pyscf.df.addons.predefined_auxbasis)
# ============================================================

# Auxiliary names used when the caller does not supply one. These are
# the existing route pairings; their normalization and retained-rank
# behavior must be preserved or explicitly validated when replacing a fit.

#
# Names are matched case-insensitively; left-hand side is the orbital
# basis as known to libint, right-hand side is the libint name of the
# bundled aux .g94 file.
def default_aux_for(orbital_basis_name: str) -> str:
    """Return the recommended JK auxiliary-basis name for an AO basis.

    Reads :mod:`vibeqc.basis_registry` (the same table the molecular
    :func:`vibeqc.density_fitting.default_aux_basis_for` uses) with the
    periodic policy: a basis with no published fit (``pob-*``, the minimal
    and Pople sets) resolves to its zeta-matched def2 stand-in until a
    hand-tuned fit ships.

    Raises ``KeyError`` if nothing is registered -- this is intentional:
    rather than guess, force the caller to pass ``aux_basis=`` explicitly
    so the choice is reproducible.
    """
    from .basis_registry import default_aux_basis

    try:
        return default_aux_basis(orbital_basis_name, "jk", allow_standin=True)
    except NotImplementedError as exc:
        raise KeyError(
            f"default_aux_for: no default aux registered for orbital "
            f"basis {orbital_basis_name!r}. Pass aux_basis= explicitly, "
            f"or extend python/vibeqc/basis_library/registry.toml."
        ) from exc


def make_aux_basis_set(
    molecule,
    *,
    aux_name: str,
    drop_eta: float = 0.0,
) -> BasisSet:
    """Build an auxiliary :class:`BasisSet` for a molecule + aux name.

    Parameters
    ----------
    molecule
        A :class:`vibeqc._vibeqc_core.Molecule` (typically
        ``system.unit_cell_molecule()`` for periodic GDF).
    aux_name
        libint-recognised aux basis name (case-insensitive). vibe-qc
        bundles the def2 JKfit family, def2-universal-jkfit /
        universal-jfit, the cc-pV*-rifit / -jkfit families, and a few
        others under ``python/vibeqc/basis_library/basis/``.
    drop_eta
        Drop primitives with exponent below ``drop_eta``. Useful for
        diffuse aux bases on dense periodic systems where the most-
        diffuse primitives produce near-zero pivots in the Cholesky
        factor of the periodic 2c metric. Set to 0 (default) to keep
        all primitives. PySCF's default is ``drop_eta=0.2`` for sto-3g
        + def2-universal-jkfit on small ionic crystals; we expose it
        as a knob rather than baking a default.

    Returns
    -------
    BasisSet
        A vibe-qc BasisSet usable as the ``aux`` argument to
        :func:`compute_2c_eri_lattice` / :func:`compute_3c_eri_lattice`.

    Notes
    -----
    No modrho / charge-compensation transformation is applied. Callers
    relying on bit-equivalent parity vs PySCF's GDF metric must handle
    convergence themselves (or wait for the modrho slice).

    ``drop_eta`` is currently a no-op stub: vibe-qc's BasisSet doesn't
    yet expose a per-primitive cull. Tracked as a follow-up; for now
    the value is recorded for diagnostic purposes only.
    """
    if drop_eta < 0:
        raise ValueError(f"drop_eta must be >= 0, got {drop_eta}")
    # #480: libint2 returns zero shells on an element a basis file omits
    # rather than raising, which lets a periodic GDF fit be built with no
    # auxiliary functions on that centre -- the same silent wrong-answer
    # class measured on the molecular route. Refuse before any lattice
    # integral work. ``drop_eta`` is a documented no-op stub below, so a
    # missing centre is never a deliberate outcome of this constructor.
    from .density_fitting import make_checked_aux_basis

    aux = make_checked_aux_basis(
        molecule, aux_name, route="make_aux_basis_set"
    )
    if drop_eta > 0:
        # Per-primitive cull lives in libint internals; we can't reach
        # it without a C++-side helper. Emit a logger note so the user
        # sees the kwarg is recorded but not enforced. Lands with the
        # modrho slice (which has to touch shells anyway).
        _logger.info(
            "make_aux_basis_set: drop_eta=%g requested but per-primitive "
            "cull is a no-op until modrho slice ships.",
            drop_eta,
        )
    return aux


def _gaussian_int(n: int, alpha: np.ndarray) -> np.ndarray:
    """∫_0^inf r^n exp(-a r^2) dr.

    Closed form: ``Γ((n+1)/2) / (2 . a^((n+1)/2))``. Element-wise on
    the ``alpha`` array. Mirrors ``pyscf.gto.mole.gaussian_int``.

    Used by :func:`modrho_scales` to compute the contracted-shell
    monopole moment that the modrho rescaling normalises against.
    """
    n_half = 0.5 * (n + 1)
    return _scipy_gamma(n_half) / (2.0 * np.power(alpha, n_half))


def modrho_scales(aux_basis: BasisSet) -> np.ndarray:
    """Modrho per-AO scale factors for an auxiliary basis.

    Returns a 1-D array of length ``aux_basis.nbasis`` such that
    multiplying integrals by ``a[P]`` (and ``a[P].a[Q]`` for the 2c
    metric) produces the modrho-rescaled tensor -- equivalent to the
    PySCF ``make_modrho_basis`` transform. Each pure component's moment
    against ``r^L Y_Lm`` is ``√(1/(4pi))`` with unit-normalized real
    spherical harmonics; an s function therefore has unit total charge.

    Mintmire-Dunlap, *Phys. Rev. A* **25**, 88 (1982); PySCF
    ``pyscf/pbc/df/df.py:make_modrho_basis`` (lines 64-120).

    Algorithm (per shell ``P`` of angular momentum ``L`` with
    primitives ``(a_p, c_p)``):
      ``s = S_p c_p . gaussian_int(2L+2, a_p)``     contracted-shell scale
      ``a_P = √(2L+1) / (4pi s)``                libint rescaling factor

    The resulting scalar is then broadcast across all (2L+1) AO
    components of the shell (every component shares the same
    contraction, hence the same rescaling).

    Notes
    -----
    The shell ``coefficients`` reported by ``aux_basis.shells()`` are
    libint-normalised (post primitive normalisation). The two libraries
    represent the same physical functions with different stored radial
    coefficients. This scale includes the spherical-convention conversion
    used by :func:`make_modrho_aux_basis`; rescaling an existing integral
    and rebuilding the basis must give the same fitting metric, including
    its absolute eigenvalue threshold.
    """
    half_sph_norm = float(np.sqrt(0.25 / np.pi))
    scales = np.empty(aux_basis.nbasis, dtype=np.float64)
    bf_offset = 0
    for shell in aux_basis.shells():
        L = int(shell.l)
        exps = np.asarray(shell.exponents, dtype=np.float64)
        coeffs = np.asarray(shell.coefficients, dtype=np.float64)
        if exps.size == 0 or coeffs.size != exps.size:
            raise RuntimeError(
                f"modrho_scales: malformed aux shell at AO offset "
                f"{bf_offset}: |exp|={exps.size}, |coef|={coeffs.size}"
            )
        int1 = _gaussian_int(2 * L + 2, exps)  # one value per primitive
        s = float(np.dot(coeffs, int1))
        if abs(s) < 1e-30:
            # Degenerate contraction (sum of Gaussians integrates to
            # ~zero). Should never happen for a well-formed aux shell.
            raise RuntimeError(
                f"modrho_scales: contracted-shell scale s={s:.3e} "
                f"underflowed at AO offset {bf_offset}; aux basis is "
                "malformed or a primitive cancellation occurred."
            )
        # The normalized real spherical harmonic is sqrt((2L+1)/(4pi))
        # times libint's regular solid harmonic. Match the basis rebuilt
        # by make_modrho_aux_basis, including that per-L convention factor.
        libcint_to_libint = 1.0 / (
            float(np.sqrt(4.0 * np.pi)) / float(np.sqrt(2 * L + 1))
        )
        alpha_P = (half_sph_norm / s) * libcint_to_libint
        n_components = 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        scales[bf_offset : bf_offset + n_components] = alpha_P
        bf_offset += n_components
    if bf_offset != aux_basis.nbasis:
        raise RuntimeError(
            f"modrho_scales: shell-walking covered {bf_offset} AOs but "
            f"aux_basis.nbasis = {aux_basis.nbasis} -- internal error."
        )
    return scales


# ============================================================
# Compensated-charge (compcell) GDF pipeline
# ============================================================
#
# Algorithm (Sun, J. Chem. Phys. 147, 164119 (2017); PySCF's
# pyscf/pbc/df/gdf_builder.py:_CCGDFBuilder):
#
#   1. Rescale each aux shell so its L-th multipole = √(1/(4pi))
#      (modrho normalisation; Mintmire-Dunlap PRA 25, 88 (1982)).
#   2. Build a compensating basis "chg": for every (atom, L) present
#      in the modrho aux, add ONE smooth Gaussian shell with η~0.2
#      and coefficient √(1/(4pi)) / gaussian_int(2L+2, η). By
#      construction this shell has the SAME L-th multipole as any
#      modrho aux shell on the same atom at the same L.
#   3. Compute the 2c metric M_fused and 3c tensor T_fused on the
#      union aux ∪ chg. The fused matrix elements are divergent
#      individually (image sum on diffuse Gaussians), but the
#      compensated combination
#
#         M_compensated = A . M_fused . Aᵀ ,  A = [I_aux  -A_chg]
#
#      where A_chg[i, n_aux+j] = 1 iff aux AO i and chg AO j share
#      (atom, L, m), satisfies <aux-chg | aux-chg> with ZERO net
#      L-th multipole on each shell -- the lattice sum converges.
#   4. Similarly for T: subtract the matching chg-projection from
#      each aux row, yielding T_compensated of shape (n_aux, n_orb,
#      n_orb).
#   5. Eigendecompose-and-threshold the compensated metric, form
#      Lpq = U[:, kept].T @ T_compensated / √l_kept.
#
# What the compcell pipeline does NOT include in this initial
# landing: PySCF additionally subtracts a reciprocal-space (AFT)
# long-range Coulomb of the chg-anything block before the fuse
# transform, which makes the answer η-independent at convergence.
# Without that correction, η is a tuning knob -- for tight aux on
# typical molecules / ionic crystals, η ≈ 0.2 gives µHa parity at
# H2 / def2-svp-jk and tight lattice cutoffs. The AFT correction is
# tracked as a follow-up; if cutoff convergence misses target we
# add it then.


def make_modrho_aux_basis(
    aux_basis: BasisSet,
    molecule: Molecule,
    *,
    name: Optional[str] = None,
) -> BasisSet:
    """Rebuild ``aux_basis`` in the shared modrho multipole convention.

    For a contracted shell ``phi = S_p c_p . prim(a_p, L)`` the rescaling
    is ``c_p -> a_P . c_p`` where ``a_P = √(2L+1) / (4pi s)`` and
    ``s = S_p c_p . gaussian_int(2L+2, a_p)``. This is equivalent up to
    integral rounding to multiplying the bare-aux integrals by per-AO
    scales from :func:`modrho_scales`. Downstream
    callers can pass the modrho-aux ``BasisSet`` to any
    ``compute_*_eri_lattice`` and get the modrho-rescaled integrals
    directly, without a separate post-multiplicative pass.

    Used as the first step of the compcell GDF pipeline; the matching
    compensating basis built by :func:`make_compensating_basis` shares
    the same per-shell monopole, so ``aux - chg`` has zero net
    multipole and its lattice sum converges in real space.

    Parameters
    ----------
    aux_basis
        Source auxiliary basis (e.g. from :func:`make_aux_basis_set`).
    molecule
        :class:`vibeqc._vibeqc_core.Molecule` the aux is attached to
        (typically ``system.unit_cell_molecule()`` for periodic GDF).
    name
        Optional display name for the returned BasisSet. Defaults to
        ``f"{aux_basis.name}-modrho"``.

    Returns
    -------
    BasisSet
        A new BasisSet over the same molecule with rescaled
        contraction coefficients. Same AO count / per-AO ordering as
        the input.

    References
    ----------
    Mintmire, Sabin, Trickey, *Phys. Rev. A* **25**, 88 (1982);
    Dunlap, Connolly, Sabin, *J. Chem. Phys.* **71**, 3396 (1979);
    Whitten, *J. Chem. Phys.* **58**, 4496 (1973);
    PySCF ``pyscf/pbc/df/df.py:make_modrho_basis``.
    """
    scales = modrho_scales(aux_basis)
    new_shells = []
    offset = 0
    for shell in aux_basis.shells():
        L = int(shell.l)
        exps = np.asarray(shell.exponents, dtype=np.float64)
        coeffs = np.asarray(shell.coefficients, dtype=np.float64)
        new_coefs = (scales[offset] * coeffs).tolist()
        offset += 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        new_shells.append(
            ShellInfo(
                int(shell.atom_index),
                L,
                bool(shell.pure),
                exps.tolist(),
                new_coefs,
                list(shell.origin),
            )
        )
    label = name if name is not None else f"{getattr(aux_basis, 'name', 'aux')}-modrho"
    return BasisSet(molecule, new_shells, label, True)


def make_compensating_basis(
    aux_basis: BasisSet,
    molecule: Molecule,
    *,
    eta: float = 0.2,
    name: Optional[str] = None,
) -> BasisSet:
    """Build the smooth-Gaussian compensating basis ("chg").

    Per (atom, L) present in ``aux_basis``, add ONE single-primitive
    contracted shell:

      * exponent ``eta`` (default 0.2 -- PySCF's typical _guess_eta
        landing for tight aux on molecules + small ionic crystals)
      * coefficient ``c = √(1/(4pi)) / gaussian_int(2L+2, eta)``

    The coefficient is chosen so the chg shell's monopole equals
    ``√(1/(4pi))`` -- the same value the modrho-rescaled aux shells
    carry. With matching multipoles, ``(modrho_aux - chg)`` has zero
    net L-th multipole on each shell, and the periodic Coulomb sum
    converges in real space.

    Parameters
    ----------
    aux_basis
        The (typically modrho-rescaled) auxiliary basis whose (atom, L)
        signature determines which chg shells to add.
    molecule
        Same molecule the aux is attached to.
    eta
        Smooth-Gaussian exponent. Default 0.2 (a good landing for
        tight JKfit aux on systems with cell volumes ≳ 100 bohr^3).
    name
        Optional display name for the returned BasisSet.

    Returns
    -------
    BasisSet
        Compensating basis. For each unique (atom_index, l) pair in
        ``aux_basis``, one ``ShellInfo`` is emitted with that single
        primitive. The shells are emitted in *atom-then-L* order
        (atom_index ascending, then l ascending); this matches PySCF
        ``make_modchg_basis``'s emission order.

    References
    ----------
    PySCF ``pyscf/pbc/df/gdf_builder.py:make_modchg_basis``.
    """
    if eta <= 0:
        raise ValueError(f"make_compensating_basis: eta must be > 0, got {eta}")
    half_sph_norm = float(np.sqrt(0.25 / np.pi))

    # Collect unique (atom_index, l) pairs and remember each atom's
    # origin (chg shells share the atom centre).
    atom_l_pairs: list[tuple[int, int]] = []
    atom_origin: dict[int, list[float]] = {}
    for shell in aux_basis.shells():
        ia = int(shell.atom_index)
        L = int(shell.l)
        key = (ia, L)
        if key not in atom_l_pairs:
            atom_l_pairs.append(key)
        if ia not in atom_origin:
            atom_origin[ia] = list(shell.origin)

    atom_l_pairs.sort()

    new_shells = []
    sqrt4pi = float(np.sqrt(4.0 * np.pi))
    for ia, L in atom_l_pairs:
        c_libcint = half_sph_norm / float(_gaussian_int(2 * L + 2, np.array([eta]))[0])
        # libcint->libint convention conversion (same fix as in
        # make_modrho_aux_basis). PySCF's c formula gives libcint
        # convention; libint with pre_normalized=True interprets the
        # value as solid-harmonic-convention coefficient, producing a
        # basis function √(4pi/(2L+1)) larger than PySCF's. Dividing
        # restores the correct physical normalization.
        libcint_to_libint = 1.0 / (sqrt4pi / float(np.sqrt(2 * L + 1)))
        c = c_libcint * libcint_to_libint
        new_shells.append(
            ShellInfo(
                ia,
                L,
                True,  # always pure spherical for compensation
                [float(eta)],
                [float(c)],
                atom_origin[ia],
            )
        )
    label = name if name is not None else f"compcell-eta{eta:g}"
    return BasisSet(molecule, new_shells, label, True)


def make_fused_basis(
    modrho_aux: BasisSet,
    chg: BasisSet,
    molecule: Molecule,
    *,
    name: Optional[str] = None,
) -> BasisSet:
    """Concatenate ``modrho_aux`` and ``chg`` into one BasisSet.

    Shell order is: all aux shells first (in aux's natural order),
    then all chg shells. AO indexing follows: aux AOs occupy indices
    ``[0, n_aux)``, chg AOs occupy ``[n_aux, n_aux + n_chg)``. This
    is the layout assumed by :func:`fuse_transform_matrix` and
    :func:`build_lpq_compcell`.
    """
    aux_shells = [
        ShellInfo(
            int(s.atom_index),
            int(s.l),
            bool(s.pure),
            list(s.exponents),
            list(s.coefficients),
            list(s.origin),
        )
        for s in modrho_aux.shells()
    ]
    chg_shells = [
        ShellInfo(
            int(s.atom_index),
            int(s.l),
            bool(s.pure),
            list(s.exponents),
            list(s.coefficients),
            list(s.origin),
        )
        for s in chg.shells()
    ]
    label = name if name is not None else "fused"
    return BasisSet(molecule, aux_shells + chg_shells, label, True)


def fuse_transform_matrix(
    modrho_aux: BasisSet,
    chg: BasisSet,
) -> np.ndarray:
    """Sparse ``(n_aux, n_aux + n_chg)`` fuse-transform matrix ``A``.

    Defined so that, for any tensor ``X_fused`` over fused AO indices:

      * 2c metric: ``M_compensated = A . M_fused . A.T``
      * 3c tensor: ``T_compensated = einsum('Pq,Pmn->qmn', A, T_fused)``

    Construction: for each AO ``i`` in the aux block ``[0, n_aux)``
    belonging to a shell at ``(atom_a, L, m_i)``, set

        A[i, i]                   = +1
        A[i, n_aux + chg_index_of(atom_a, L, m_i)] = -1

    where ``chg_index_of(atom_a, L, m)`` is the AO index of the chg
    shell at ``(atom_a, L)``'s m-th component. The chg block contains
    one (and only one) shell per ``(atom, L)`` (see
    :func:`make_compensating_basis`), so the link is unambiguous.

    Multiple aux shells at the same ``(atom, L)`` all subtract the
    SAME chg shell (PySCF's ``fuse_auxcell`` behaviour).

    Parameters
    ----------
    modrho_aux
        The aux basis used to assemble the fused basis.
    chg
        The matching compensating basis.

    Returns
    -------
    np.ndarray of shape (n_aux, n_aux + n_chg)
        Sparse-but-dense-stored. Use ``A @ M_fused @ A.T`` /
        ``np.einsum('Pq,Pmn->qmn', A, T_fused)`` directly.
    """
    # Discover chg-AO layout: per (atom, L), the AO range in chg.
    chg_ao_offset: dict[tuple[int, int], int] = {}
    bf = 0
    for shell in chg.shells():
        ia = int(shell.atom_index)
        L = int(shell.l)
        if not shell.pure:
            raise RuntimeError(
                "fuse_transform_matrix: chg basis must be pure-spherical; "
                f"got non-pure shell at atom {ia} L={L}."
            )
        chg_ao_offset[(ia, L)] = bf
        bf += 2 * L + 1
    n_chg = bf
    if n_chg != chg.nbasis:
        raise RuntimeError(
            f"fuse_transform_matrix: chg AO walk = {n_chg} but "
            f"chg.nbasis = {chg.nbasis}; chg basis malformed."
        )

    n_aux = modrho_aux.nbasis
    A = np.zeros((n_aux, n_aux + n_chg), dtype=np.float64)
    bf = 0
    for shell in modrho_aux.shells():
        ia = int(shell.atom_index)
        L = int(shell.l)
        if not shell.pure:
            raise RuntimeError(
                "fuse_transform_matrix: aux basis must be pure-spherical; "
                f"got non-pure shell at atom {ia} L={L}."
            )
        n_comp = 2 * L + 1
        key = (ia, L)
        if key not in chg_ao_offset:
            raise RuntimeError(
                f"fuse_transform_matrix: aux has shell at (atom={ia}, "
                f"L={L}) with no matching chg shell. Did you build chg "
                "from the same aux?"
            )
        chg_bf = chg_ao_offset[key]
        for m in range(n_comp):
            A[bf + m, bf + m] = 1.0
            A[bf + m, n_aux + chg_bf + m] = -1.0
        bf += n_comp
    if bf != n_aux:
        raise RuntimeError(
            f"fuse_transform_matrix: aux AO walk = {bf} but "
            f"modrho_aux.nbasis = {n_aux}; aux basis malformed."
        )
    return A


def _aft_g_mesh(
    system: PeriodicSystem,
    eta: float,
    *,
    precision: float = 1e-10,
    pad_factor: float = 1.5,
    q: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reciprocal-lattice mesh for the compcell AFT correction.

    Sized by the chg basis's Gaussian decay: ``exp(-|G+q|^2/(4η)) <=
    precision`` gives ``|G+q|_max = 2 . pad_factor . √(η . ln(1/precision))``.
    Returns all reciprocal-lattice ``G`` with ``|G + q| <= |G+q|_max``,
    excluding any ``G`` for which ``|G + q| ≈ 0``:

    * ``q = None`` (or ``q = 0``): excludes ``G = 0`` -- the AFT
      ``1/G^2`` weight is singular at the origin; the compcell
      construction already cancels the monopole.
    * ``q != 0`` (inside the BZ, generic point): all ``G`` are
      retained because ``|G + q| > 0`` everywhere. ``G = 0`` is the
      typical safe case ``|q|`` and contributes; ``G = -q`` would
      cancel exactly, but that's exclusively a zone-boundary
      coincidence and is excluded by the same ``|G+q| ≈ 0`` test.
    """
    if eta <= 0:
        raise ValueError(f"_aft_g_mesh: eta must be > 0, got {eta}")
    Gq_max = (
        2.0
        * float(pad_factor)
        * float(np.sqrt(float(eta) * np.log(1.0 / float(precision))))
    )
    a = np.asarray(system.lattice, dtype=float)
    if a.shape != (3, 3):
        raise RuntimeError(
            f"_aft_g_mesh: PeriodicSystem.lattice must be 3x3, got {a.shape}"
        )
    b = 2.0 * np.pi * np.linalg.inv(a).T
    # Pad by |q| so we keep all G with |G+q| <= Gq_max. For q = 0 this
    # is the historical behaviour.
    q_arr = (
        np.zeros(3, dtype=float) if q is None else np.asarray(q, dtype=float).reshape(3)
    )
    q_norm = float(np.linalg.norm(q_arr))
    # Per-axis reciprocal box (column-norm bound); |G| <= Gq_max + |q| by the
    # triangle inequality. The previous single n_max = ceil((Gq_max+|q|) /
    # min|b_i|) clipped the ball on oblique cells. compcell AFT is 3D-only.
    # See _reciprocal_index_box_per_axis.
    n_per_axis = _reciprocal_index_box_per_axis(a, Gq_max + q_norm, 3)
    grids = [np.arange(-n, n + 1) for n in n_per_axis]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(float)
    G = idx @ b.T
    Gq = G + q_arr[None, :]
    Gq2 = (Gq**2).sum(axis=1)
    keep = (Gq2 > 1e-12) & (Gq2 <= Gq_max * Gq_max)
    return G[keep]


def _aft_fourier_transform_libcint_convention(
    basis: BasisSet,
    G_vectors: np.ndarray,
) -> np.ndarray:
    """FT of an aux basis at G-vectors using libcint's Y_lm-included
    convention (matches PySCF ``ft_ao.ft_ao``).

    This is intentionally a DIFFERENT convention from
    :func:`rsgdf_aux_fourier_transform`, which uses the libint
    Y_00-dropped + L>0 √(4pi/(2L+1)) convention (calibrated against
    libint's ``compute_2c_eri`` element-wise via Parseval). For the
    AFT correction in :func:`build_lpq_compcell` we need libcint's
    convention so the AFT subtraction stays self-consistent: PySCF's
    BARE lattice sum on the fused basis is internally consistent with
    its OWN FT convention (libcint), and analogously vibe-qc's bare
    lattice sum (libint) needs an AFT in libint's matching convention.

    The two FT conventions differ per L by:

      libint convention / libcint convention  =  √(4pi/(2L+1))

    Multiply this routine's output by ``√(4pi/(2L+1))`` to recover
    :func:`rsgdf_aux_fourier_transform`'s libint-calibrated output;
    divide that routine's output by the same factor to recover this
    one. (Either way, both conventions are self-consistent ONLY when
    paired with their matching ``(P|P)`` evaluation.)
    """
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    n_aux = basis.nbasis
    out = np.zeros((n_aux, n_G), dtype=np.complex128)

    G_norms = np.linalg.norm(G_vectors, axis=1)
    safe_norm = np.where(G_norms > 0, G_norms, 1.0)
    theta = np.arccos(np.clip(G_vectors[:, 2] / safe_norm, -1.0, 1.0))
    phi = np.arctan2(G_vectors[:, 1], G_vectors[:, 0])
    G2 = G_norms**2

    # SPEED: cache Y_lm by L. Across a typical aux basis (def2-svp-jk
    # on H/Li/etc.) the same L value appears for many shells; without
    # caching we recompute _real_sph_harm per (L, m) for every shell,
    # which dominates the AFT cost (Python overhead per call + scipy
    # sph_harm_y evaluation per G point).
    ylm_cache: dict[int, np.ndarray] = {}

    def _ylm_block(L: int) -> np.ndarray:
        if L not in ylm_cache:
            block = np.empty((2 * L + 1, n_G), dtype=np.float64)
            for idx, m in enumerate(range(-L, L + 1)):
                block[idx] = _real_sph_harm(L, m, theta, phi)
            ylm_cache[L] = block
        return ylm_cache[L]

    bf_offset = 0
    for shell in basis.shells():
        L = int(shell.l)
        n_comp = 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        if not shell.pure and L > 0:
            raise NotImplementedError(
                "_aft_fourier_transform_libcint_convention: Cartesian "
                f"shells L>0 not supported (got L={L})."
            )
        exps = np.asarray(shell.exponents, dtype=float)
        coeffs = np.asarray(shell.coefficients, dtype=float)
        R = np.asarray(shell.origin, dtype=float)
        phase = np.exp(-1j * (G_vectors @ R))

        prim_factor = (np.pi / exps)[:, None] ** 1.5 * np.exp(
            -G2[None, :] / (4.0 * exps[:, None])
        )
        if L > 0:
            prim_factor *= (G_norms[None, :] / (2.0 * exps[:, None])) ** L
        rad_contracted = np.einsum("p,pk->k", coeffs, prim_factor)

        i_to_minus_l = (-1j) ** L
        ylm_block = _ylm_block(L)  # (n_comp, n_G), cached per-L
        # Vectorized over the m index -- single broadcast multiply instead of
        # a Python loop over (-L .. +L).
        ft_block = i_to_minus_l * ylm_block * rad_contracted[None, :] * phase[None, :]
        out[bf_offset : bf_offset + n_comp, :] = ft_block
        bf_offset += n_comp

    if bf_offset != n_aux:
        raise RuntimeError(
            f"_aft_fourier_transform_libcint_convention: shell-walk "
            f"covered {bf_offset} AOs but basis.nbasis={n_aux}."
        )
    return out


def _compcell_aft_correction(
    fused: BasisSet,
    n_aux: int,
    system: PeriodicSystem,
    *,
    eta: float,
    precision: float = 1e-10,
    ft_convention: str = "libcint",
    q: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The PySCF ``_CCGDFBuilder.get_2c2e`` j2c_p subtraction.

    Returns ``j2c_p`` of shape ``(n_chg, n_aux + n_chg)`` such that
    callers subtract it from the chg-rows of the bare fused metric and
    its transpose-conjugate from the chg-columns of the aux block
    (Hermitian symmetry preservation; see PySCF gdf_builder.py:187-188):

        M_fused[n_aux:, :]      -= j2c_p
        M_fused[:n_aux, n_aux:] -= j2c_p[:, :n_aux].T.conj()

    For ``q = None`` (Γ-only) the result is real-valued and the
    transpose has no conjugate; for ``q != 0`` it is complex-valued
    and the Hermitian conjugate is required.

    Closure: the resulting M_fused, when fuse-transformed via A
    (fuse_transform_matrix), gives a compensated metric whose answer
    is η-independent at convergence and well-conditioned for tight
    ionic systems -- the missing piece in the initial compcell landing.

    The AFT correction is the analytical reciprocal-space evaluation
    of the long-range Coulomb between (chg | anything):

        j2c_p[i, j] = (4pi/V) S_G F̂_chg_i*(G+q) . F̂_anything_j(G+q)
                       / |G+q|^2

    where ``F̂`` comes from :func:`rsgdf_aux_fourier_transform` (now
    convention-corrected per the 2026-05-17 L>0 calibration). For
    ``q = 0`` the ``G = 0`` term is excluded by :func:`_aft_g_mesh`
    (singular ``1/G^2``, monopole already cancelled by the compcell
    construction). For ``q != 0`` inside the BZ, ``|G + q| > 0`` for
    all ``G`` and the full mesh is summed.

    Parameters
    ----------
    q
        Momentum shift in Cartesian bohr⁻¹, shape ``(3,)`` or
        ``None``. ``None`` is the Γ-only path (q = 0, ``G = 0``
        excluded, result is real). A nonzero ``q`` shifts every FT
        evaluation point and the Coulomb kernel to ``G + q``; the
        result is complex-valued.

    References
    ----------
    Sun, *J. Chem. Phys.* **147**, 164119 (2017), Sec.III; PySCF
    ``pyscf/pbc/df/gdf_builder.py:139-196``.
    """
    n_total = fused.nbasis
    n_chg = n_total - n_aux
    if n_chg <= 0:
        raise RuntimeError(
            "_compcell_aft_correction: n_chg <= 0; fused basis "
            f"must extend beyond aux (n_aux={n_aux}, n_total={n_total})."
        )

    q_arr = (
        np.zeros(3, dtype=float) if q is None else np.asarray(q, dtype=float).reshape(3)
    )
    is_gamma_q = float(np.linalg.norm(q_arr)) < 1e-14

    G = _aft_g_mesh(system, eta, precision=precision, q=(None if is_gamma_q else q_arr))
    if G.shape[0] == 0:
        # Match dtype to the q != 0 caller's expectation: real at Γ,
        # complex otherwise.
        dtype = np.float64 if is_gamma_q else np.complex128
        return np.zeros((n_chg, n_total), dtype=dtype)
    Gq = G + q_arr[None, :]
    Gq2 = (Gq**2).sum(axis=1)

    a = np.asarray(system.lattice, dtype=float)
    cell_volume = float(abs(np.linalg.det(a)))
    coul = (4.0 * np.pi) / Gq2 / cell_volume  # (n_G,) -- kernel
    # F̂[fused_AO, G_idx], evaluated at G + q; complex-valued.
    # Convention selected by ``ft_convention``:
    #   "libcint" -> Y_lm-included for all L (matches PySCF's ``ft_ao``
    #               bit-perfectly; right choice when the bare lattice
    #               sum is in libcint scale, e.g. via PySCF-style auto
    #               rcut + libcint integrals).
    #   "libint"  -> Y_00-dropped for L=0, √(4pi/(2L+1)) for L>=1 (matches
    #               vibe-qc's own ``compute_2c_eri`` via Parseval; right
    #               choice when the bare lattice sum is in libint scale
    #               end-to-end).
    if ft_convention == "libcint":
        F = _aft_fourier_transform_libcint_convention(fused, Gq)
    elif ft_convention == "libint":
        F = rsgdf_aux_fourier_transform(fused, Gq)
    else:
        raise ValueError(
            f"_compcell_aft_correction: ft_convention must be "
            f"'libcint' or 'libint'; got {ft_convention!r}"
        )
    F_chg = F[n_aux:]  # (n_chg, n_G)
    # j2c_p = S_G coul[G] . conj(F_chg[i, G+q]) . F[j, G+q]
    # For a real basis at Γ (q = 0), F satisfies F(-G) = conj(F(G)),
    # so the sum S_G coul . conj(F_chg) . F equals its conjugate
    # (real-valued). For q != 0 this symmetry is broken (G <-> -G doesn't
    # map G+q <-> -(G+q) unless q = 0); the result is complex.
    F_chg_w = np.conj(F_chg) * coul[None, :]  # (n_chg, n_G)
    j2c_p = F_chg_w @ F.T  # (n_chg, n_total)
    if is_gamma_q:
        return np.real(j2c_p)
    return j2c_p


def _basis_ao_atom_indices(
    basis: BasisSet,
    system: PeriodicSystem,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    """Map each basis function to its home-cell atom index by shell origin.

    Every shell origin must coincide (within ``tol``) with exactly one
    unit-cell atom position; anything else raises. Used by the AFT
    centre-derivative below, where the only atom-position dependence is
    the ``exp(-iG.R)`` phase of each shell's Fourier transform.
    """
    atom_xyz = np.asarray(
        [list(atom.xyz) for atom in system.unit_cell], dtype=np.float64
    )
    out = np.empty(int(basis.nbasis), dtype=np.int64)
    bf = 0
    for shell in basis.shells():
        origin = np.asarray(shell.origin, dtype=np.float64).reshape(3)
        dists = np.linalg.norm(atom_xyz - origin[None, :], axis=1)
        hits = np.flatnonzero(dists <= float(tol))
        if hits.size != 1:
            raise ValueError(
                "_basis_ao_atom_indices: shell origin "
                f"{origin.tolist()} matches {hits.size} unit-cell atoms "
                f"(tol={tol:g}); need exactly one."
            )
        L = int(shell.l)
        n_components = 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        out[bf : bf + n_components] = int(hits[0])
        bf += n_components
    if bf != int(basis.nbasis):
        raise RuntimeError(
            "_basis_ao_atom_indices: shell walk covered "
            f"{bf} of {int(basis.nbasis)} basis functions."
        )
    return out


def _compcell_aft_correction_2c_gradient_weighted(
    fused: BasisSet,
    n_aux: int,
    system: PeriodicSystem,
    *,
    eta: float,
    weight: np.ndarray,
    precision: float = 1e-10,
    ft_convention: str = "libcint",
) -> np.ndarray:
    """Atomic gradient of the weighted Γ-only 2c AFT correction value.

    Returns ``d/dR_A [ Σ_cf weight[c, f] · j2c_p[c, f] ]`` as an
    ``(n_atoms, 3)`` array, where ``j2c_p`` is exactly
    :func:`_compcell_aft_correction` at ``q = None`` (Γ). Callers apply
    their own sign for the SCF's *subtraction* of the correction.

    Derivation (G-PBC-002 milestone 3a). In

        j2c_p[c, f] = (4π/V) Σ_G conj(F̂_chg_c(G)) F̂_f(G) / G²
        (Sun, J. Chem. Phys. 147, 164119 (2017), Sec. III; the PySCF
        ``_CCGDFBuilder.get_2c2e`` j2c_p term)

    the reciprocal mesh (:func:`_aft_g_mesh`), the 1/G² Coulomb kernel,
    and each FT's radial/angular content depend on the lattice and the
    exponents only. The single atom-position dependence is the shell
    phase ``exp(-iG.R)``, so ``dF̂_k/dR_A = -i G F̂_k δ(atom_k = A)``
    exactly, and with

        C_d[c, f] = Σ_G coul(G) G_d conj(F̂_chg_c(G)) F̂_f(G)

    the real Γ-point gradient is

        grad[A, d] = - Σ_{c on A} row-sum(weight ∘ Im C_d)[c]
                     + Σ_{f on A} col-sum(weight ∘ Im C_d)[f].

    The derivative is complete — there is no mesh- or kernel-response
    term — which the isolation FD regression pins at the kernel
    tolerance (``tests/test_periodic_gdf_gradient.py``).
    """
    n_total = int(fused.nbasis)
    n_chg = n_total - int(n_aux)
    n_atoms = len(system.unit_cell)
    if n_chg <= 0:
        raise RuntimeError(
            "_compcell_aft_correction_2c_gradient_weighted: n_chg <= 0"
        )
    W = np.asarray(weight, dtype=np.float64)
    if W.shape != (n_chg, n_total):
        raise ValueError(
            "_compcell_aft_correction_2c_gradient_weighted: weight must "
            f"have shape (n_chg, n_fused) = ({n_chg}, {n_total}); got "
            f"{W.shape}."
        )

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    G = _aft_g_mesh(system, eta, precision=precision, q=None)
    if G.shape[0] == 0:
        return grad
    G2 = (G**2).sum(axis=1)

    a = np.asarray(system.lattice, dtype=float)
    cell_volume = float(abs(np.linalg.det(a)))
    coul = (4.0 * np.pi) / G2 / cell_volume

    if ft_convention == "libcint":
        F = _aft_fourier_transform_libcint_convention(fused, G)
    elif ft_convention == "libint":
        F = rsgdf_aux_fourier_transform(fused, G)
    else:
        raise ValueError(
            "_compcell_aft_correction_2c_gradient_weighted: ft_convention "
            f"must be 'libcint' or 'libint'; got {ft_convention!r}"
        )
    F_chg = F[n_aux:]

    atom_of = _basis_ao_atom_indices(fused, system)
    atom_of_chg = atom_of[n_aux:]

    for d in range(3):
        # C_d = Σ_G (coul·G_d) conj(F_chg) ⊗ F  — one ZGEMM per axis.
        C_d = (np.conj(F_chg) * (coul * G[:, d])[None, :]) @ F.T
        X = W * np.imag(C_d)  # (n_chg, n_fused)
        row = X.sum(axis=1)  # per chg function
        col = X.sum(axis=0)  # per fused function
        np.subtract.at(grad[:, d], atom_of_chg, row)
        np.add.at(grad[:, d], atom_of, col)
    return grad


def _per_ao_libcint_to_libint_scale(basis: BasisSet) -> np.ndarray:
    """Per-AO scale factor mapping libcint single-AO FT to libint
    single-AO FT, as a (n_orb,) array.

    libint single-AO FT (per :func:`rsgdf_aux_fourier_transform` after
    the 2026-05-17 calibration) is ``√(4pi/(2L+1))`` times the libcint
    single-AO FT, for every L (including L=0 via the Y_00 drop which
    is equivalent to multiplying by ``√(4pi) = √(4pi/(2.0+1))``).

    NOTE: this factor applies to SINGLE-AO FTs only. For AO-PAIR FTs
    a separate (sign-aware) factor applies -- see
    :func:`_per_ao_pair_libint_to_libcint_scale`.
    """
    sqrt4pi = float(np.sqrt(4.0 * np.pi))
    scales = np.empty(basis.nbasis, dtype=np.float64)
    bf = 0
    for shell in basis.shells():
        L = int(shell.l)
        n_comp = 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        scale = sqrt4pi / float(np.sqrt(2 * L + 1))
        scales[bf : bf + n_comp] = scale
        bf += n_comp
    if bf != basis.nbasis:
        raise RuntimeError(
            f"_per_ao_libcint_to_libint_scale: shell walk = {bf} vs "
            f"basis.nbasis = {basis.nbasis}"
        )
    return scales


def _per_ao_pair_libint_to_libcint_scale(basis: BasisSet) -> np.ndarray:
    """Per-AO magnitude factor converting vibe-qc's libint pair-FT
    convention to PySCF's libcint pair-FT convention.

    Empirically measured on Li-H STO-3G (see
    ``examples/debug/gdf_pair_ft_pyscf_diff.py``):

      ratio(vibe-qc/PySCF)[mu, ν, G]  =  s[mu] . s[ν]

    where the per-AO factor ``s[i]`` depends only on ``L_i``:

      L=0: s = +1                       (no convention diff)
      L>=1: s = 1/√(4pi/(2L+1))           (libint solid-harmonic Y_lm
                                         normalization vs libcint
                                         standard Y_lm)

    To convert vibe-qc's libint pair FT to PySCF's libcint pair FT:

      r̂_libcint[mu, ν, G]  =  r̂_libint[mu, ν, G] / (s[mu] . s[ν])

    The L=0 components are unchanged (since both libraries normalize
    the AO to S_ii=1, and pair density chi_mu.chi_ν is identical for
    L=0/L=0 -- verified bit-exactly to 1e-16). The L>0 components
    pick up the reciprocal-magnitude conversion.

    History
    -------
    Before 2026-05-18, this factor also carried a SIGN ``(-1)^L`` per
    axis. That sign was patching a deeper bug in
    :func:`vibeqc._aopair_ft.cartesian_gaussian_product_ft` which used
    ``(+iG)^t`` instead of the standard ``(-iG)^t`` in the FT of the
    Hermite-Gaussian chain. For same-center pairs only ``t=L_total``
    contributes, the odd-``L_total`` term gets a single sign flip, and
    the per-AO ``(-1)^L`` exactly compensated. For OFF-center pairs
    both even-``t`` and odd-``t`` terms contribute; the per-AO
    blanket sign over-corrected the even-``t`` parts, which appeared
    as the LiH primitive multi-k AFT ~109 Ha residue
    (``examples/debug/gdf_pair_ft_pyscf_diff.py`` shell-pair
    (ish=2, ish=3) Li 2p x H 1s block at ~50% relative error). The
    fix landed the ``(-iG)^t`` correction in
    ``cartesian_gaussian_product_ft`` and dropped the ``(-1)^L`` sign
    here. The magnitude factor ``1/√(4pi/(2L+1))`` comes from the
    Y_lm normalization difference (libint absorbs ``√(2L+1)/(4pi)``
    into solid-harmonic Y_lm vs libcint's standard normalization),
    is independent of the FT sign convention, and stays.

    NOTE: cross-Gaunt-coupling for arbitrary L is verified up to
    L<=2 single-prim shells. For L>2 in production basis sets, an
    extension of the per-AO calibration may be needed.
    """
    sqrt4pi = float(np.sqrt(4.0 * np.pi))
    scales = np.empty(basis.nbasis, dtype=np.float64)
    bf = 0
    for shell in basis.shells():
        L = int(shell.l)
        n_comp = 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        if L == 0:
            s = 1.0
        else:
            s = 1.0 / (sqrt4pi / np.sqrt(2 * L + 1))
        scales[bf : bf + n_comp] = s
        bf += n_comp
    if bf != basis.nbasis:
        raise RuntimeError(
            f"_per_ao_pair_libint_to_libcint_scale: shell walk = {bf} "
            f"vs basis.nbasis = {basis.nbasis}"
        )
    return scales


def _compcell_aft_correction_3c(
    fused: BasisSet,
    ao_basis: BasisSet,
    n_aux: int,
    system: PeriodicSystem,
    *,
    eta: float,
    precision: float = 1e-10,
    ft_convention: str = "libcint",
    q: Optional[np.ndarray] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
) -> np.ndarray:
    """The PySCF ``_CCGDFBuilder.add_ft_j3c`` 3c AFT subtraction.

    Returns ``j3c_p`` of shape ``(n_chg, n_orb, n_orb)`` such that the
    caller subtracts it from the chg-rows of the bare fused 3c tensor:

        T_fused[n_aux:, :, :] -= j3c_p

    Formula (PySCF gdf_builder.py:450-477):

        j3c_p[i, mu, ν] = (4pi/V) S_G F̂_chg_i*(G+q) . r̂_muν(G+q) / |G+q|^2

    where r̂_muν is the FT of the AO-pair density chi_mu(r).chi_ν(r) at the
    shifted reciprocal point. For ``q = 0`` the ``G = 0`` term is
    excluded; for ``q != 0`` inside the BZ all ``G`` are summed. The
    chg FT uses libcint convention (matches PySCF's ``ft_ao``); the
    AO-pair FT comes from :func:`_aopair_ft.ao_pair_fourier_transform`
    (libint convention) and is converted to libcint via the per-AO
    sign+scale factor :func:`_per_ao_pair_libint_to_libcint_scale`.

    Parameters
    ----------
    q
        Momentum shift in Cartesian bohr⁻¹, shape ``(3,)`` or
        ``None`` (Γ-only, real). Nonzero ``q`` returns complex.

    References
    ----------
    PySCF ``pyscf/pbc/df/gdf_builder.py:add_ft_j3c`` (lines 450-477)
    and ``weighted_ft_ao`` (lines 359-389).
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import direct_lattice_cells

    n_total = fused.nbasis
    n_chg = n_total - n_aux
    n_orb = ao_basis.nbasis
    q_arr = (
        np.zeros(3, dtype=float) if q is None else np.asarray(q, dtype=float).reshape(3)
    )
    is_gamma_q = float(np.linalg.norm(q_arr)) < 1e-14
    dtype_out = np.float64 if is_gamma_q else np.complex128

    if n_chg <= 0:
        return np.zeros((n_chg, n_orb, n_orb), dtype=dtype_out)

    G = _aft_g_mesh(system, eta, precision=precision, q=(None if is_gamma_q else q_arr))
    if G.shape[0] == 0:
        return np.zeros((n_chg, n_orb, n_orb), dtype=dtype_out)
    Gq = G + q_arr[None, :]
    Gq2 = (Gq**2).sum(axis=1)

    a = np.asarray(system.lattice, dtype=float)
    cell_volume = float(abs(np.linalg.det(a)))
    coul = (4.0 * np.pi) / Gq2 / cell_volume  # (n_G,)

    # chg-only FT in the chosen convention (libcint matches PySCF),
    # evaluated at the shifted points G + q.
    if ft_convention == "libcint":
        F_full = _aft_fourier_transform_libcint_convention(fused, Gq)
    elif ft_convention == "libint":
        F_full = rsgdf_aux_fourier_transform(fused, Gq)
    else:
        raise ValueError(
            f"_compcell_aft_correction_3c: ft_convention must be "
            f"'libcint' or 'libint'; got {ft_convention!r}"
        )
    F_chg = F_full[n_aux:]  # (n_chg, n_G)

    # AO-pair FT -- BLOCH-SUMMED at k=q. PySCF's add_ft_j3c builds the
    # j3c_p AFT correction from the *periodic* AO-pair FT
    #   r̂^(per)_muν(G+q) = S_L exp(+iq.R_L) . ∫chi_mu(r).chi_ν(r-R_L).e^{-i(G+q).r} dr
    # NOT the molecular AO-pair FT (which sets R_L=0 only). For tight
    # ionic crystals (LiH primitive FCC: Li-H separation 3.86 bohr in
    # a 7.72-bohr lattice; MgO 8-atom conv cell) lattice images of the
    # pair density contribute substantially; using the molecular pair
    # FT under-estimates j3c_p by orders of magnitude and the bare
    # 3c minus j3c_p subtraction blows up (the canonical LiH +11,686 Ha
    # Γ-only SCF symptom, 2026-05-24/26 chats). For vacuum-box systems
    # (H₂/STO-3G in a 12-bohr cube) lattice contributions to the pair
    # density are exponentially small (Li-H pair on a 7.7 bohr cell vs
    # H-H pair on a 12 bohr cell: exp(-a.Δ^2) factor is many orders of
    # magnitude smaller for the H₂ case), which is why the molecular
    # approximation accidentally worked there and the bug went
    # undiagnosed until 2026-05-26.
    #
    # The lattice-cell list comes from ``direct_lattice_cells`` with the
    # *same* cutoff as the bare T_fused 3c integral (``lat_opts``); this
    # keeps the bare-3c minus AFT-3c subtraction internally consistent
    # in the lattice cutoff. Verified on LiH primitive FCC: Bloch-sum
    # at k=0 over 959 cells (cutoff 30 bohr) reproduces PySCF's
    # ``ft_aopair(cell, Gv, ...)`` element-wise bit-exactly at every G
    # tested (``examples/debug/gdf_j3c_p_same_mesh_lih.py`` Step 7b).
    #
    # SOLID-HARMONIC NORMALIZATION (2026-05-26 fix). vibe-qc has two
    # families of FT routines that use *different* per-L solid-harmonic
    # normalisations:
    #
    #   * ``rsgdf_aux_fourier_transform`` and the C++
    #     ``compute_3c_eri[_lattice]`` both bake an explicit
    #     ``scale_L = √(4pi/(2L+1))`` into the FT output (so that they
    #     satisfy libint's Plancherel-Coulomb identity directly -- see
    #     ``rsgdf_aux_fourier_transform`` line ~2284). For the MOLECULAR
    #     ``compute_3c_eri`` this matches PySCF's libcint Y_lm
    #     normalisation element-wise (after L=1 m-permutation
    #     libint(py,pz,px) <-> libcint(px,py,pz)); verified by
    #     ``examples/debug/gdf_j3c_p_same_mesh_lih.py``.
    #
    #   * ``ao_pair_fourier_transform[_bloch]`` does NOT include the
    #     per-L ``scale_L`` (its ``cart_to_sph_matrix`` is a plain
    #     real-Y_lm transform). Its output is therefore off from
    #     PySCF's ``ft_aopair`` by ``√((2L+1)/(4pi))`` per L=mu axis and
    #     per L=ν axis.
    #
    # The Bloch-summed pair-FT is the correct *function* (lattice-
    # summed pair density at q=k) and matches PySCF's ``ft_aopair``
    # element-wise after the ``1/scales`` rescale on each L>0 axis
    # AND the L=1 m-permutation (m=-1,0,+1 -> m=+1,-1,0). Verified
    # rel < 1e-8 over the first 200 G of PySCF's production mesh.
    #
    # The rescale below mirrors the historical "libcint" path: it
    # converts the Bloch pair-FT from vibe-qc's Python real-Y_lm
    # convention into PySCF's libcint Y_lm. For the j3c_p subtraction
    # against ``T_fused`` (vibe-qc's lattice-summed bare 3c) to be
    # internally consistent, BOTH must end up in the same per-L
    # convention. Empirically the "libcint" code path (with the rescale
    # below) brings LiH primitive FCC Γ-only SCF from +11,686 Ha to
    # +20 Ha post-Bloch-fix; the "libint" code path lands at +636 Ha
    # (still a major improvement, but the convention chain is not fully
    # internally consistent -- an open question on the chg-axis
    # F_chg convention; tracked for the next M1 session).
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g_list = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g_list.size == 0:
        # Defensive: cutoff so small no cells qualified. Use the home
        # cell only -- same as molecular pair-FT (back-compat behaviour).
        R_g_list = np.zeros((1, 3), dtype=float)
    rho_pair = ao_pair_fourier_transform_bloch(
        ao_basis,
        Gq,
        R_g_list,
        k_cart=q_arr,
    )  # (n_orb, n_orb, n_G), complex even at Γ; .real for is_gamma_q.
    if ft_convention == "libcint":
        scales = _per_ao_pair_libint_to_libcint_scale(ao_basis)
        inv = 1.0 / scales
        rho_pair = rho_pair * inv[:, None, None] * inv[None, :, None]
    # else (libint): no pair-FT rescale. Internally consistent only for
    # pure-L=0 ao_basis (e.g. H₂ STO-3G); for L>0 ao_basis the libint
    # path under-represents j3c_p -- open issue; the libcint path is
    # the recommended setting until the convention is fully unified.

    # j3c_p[chg_i, mu, ν] = S_G F̂_chg_i*(G+q) . coul(G+q) . r̂_muν(G+q)
    F_chg_w = np.conj(F_chg) * coul[None, :]  # (n_chg, n_G)
    # Contract over G -- efficient as a matmul on the flattened pair axis.
    rho_flat = rho_pair.reshape(n_orb * n_orb, G.shape[0]).T  # (n_G, n_orb^2)
    j3c_p_flat = F_chg_w @ rho_flat  # (n_chg, n_orb^2)
    j3c_p = j3c_p_flat.reshape(n_chg, n_orb, n_orb)
    # Result is real at Γ (Hermitian product of real basis functions);
    # complex for q != 0 (the AO-pair density's FT no longer satisfies
    # r̂(-G) = conj(r̂(G)) at G + q != 0).
    if is_gamma_q:
        return np.real(j3c_p)
    return j3c_p


def _compcell_aft_correction_3c_gradient_weighted(
    fused: BasisSet,
    ao_basis: BasisSet,
    n_aux: int,
    system: PeriodicSystem,
    *,
    eta: float,
    weight: np.ndarray,
    precision: float = 1e-10,
    ft_convention: str = "libcint",
    lat_opts: Optional[LatticeSumOptions] = None,
) -> np.ndarray:
    """Atomic gradient of the weighted Γ-only 3c AFT correction value.

    Returns ``d/dR_A [ Σ_{c,μν} weight[c, μ, ν] · j3c_p[c, μ, ν] ]`` as
    an ``(n_atoms, 3)`` array, where ``j3c_p`` is exactly
    :func:`_compcell_aft_correction_3c` at ``q = None`` (Γ), including
    its Bloch-summed pair FT over the ``lat_opts`` cell list and its
    per-convention solid-harmonic rescale. Callers apply their own sign
    for the SCF's subtraction (G-PBC-002 milestone 3, sibling of
    :func:`_compcell_aft_correction_2c_gradient_weighted`).

    Two atom-position dependencies, each handled analytically:

    * the chg-function phase ``exp(-iG.R_c)`` (piece A, vectorised
      reciprocal sums with per-atom scatter, as in the 2c sibling);
    * the AO-pair FT centres (piece B), via the existing OpenMP kernel
      ``ao_pair_fourier_transform_gamma_gradient_weighted`` driven once
      per chg function ``c`` — the G-dependent weight
      ``coul(G) conj(F̂_chg_c(G))`` couples ``c`` to ``G``, so the
      separable (pair-weight × reciprocal-weight) kernel is applied per
      ``c`` with ``Re Σ_G coul conj(F̂_c) r̂ = Re Σ_G (coul F̂_c)
      conj(r̂)`` matching the kernel's conjugation convention (the
      ``compute_v_ne_ewald_3d_ft_gamma_gradient`` usage).

    The reciprocal mesh, Coulomb kernel, radial FT content, and the
    direct-lattice cell list depend on the lattice/exponents/cutoff
    only, so these two pieces are the complete derivative — pinned by
    the isolation FD regression
    (``tests/test_periodic_gdf_gradient.py``).
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import (
        ao_pair_fourier_transform_gamma_gradient_gweighted,
        direct_lattice_cells,
    )

    n_total = int(fused.nbasis)
    n_chg = n_total - int(n_aux)
    n_orb = int(ao_basis.nbasis)
    n_atoms = len(system.unit_cell)
    if n_chg <= 0:
        raise RuntimeError(
            "_compcell_aft_correction_3c_gradient_weighted: n_chg <= 0"
        )
    W3 = np.asarray(weight, dtype=np.float64)
    if W3.shape != (n_chg, n_orb, n_orb):
        raise ValueError(
            "_compcell_aft_correction_3c_gradient_weighted: weight must "
            f"have shape (n_chg, n_orb, n_orb) = ({n_chg}, {n_orb}, "
            f"{n_orb}); got {W3.shape}."
        )

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    G = _aft_g_mesh(system, eta, precision=precision, q=None)
    if G.shape[0] == 0:
        return grad
    G2 = (G**2).sum(axis=1)

    a = np.asarray(system.lattice, dtype=float)
    cell_volume = float(abs(np.linalg.det(a)))
    coul = (4.0 * np.pi) / G2 / cell_volume

    if ft_convention == "libcint":
        F_full = _aft_fourier_transform_libcint_convention(fused, G)
    elif ft_convention == "libint":
        F_full = rsgdf_aux_fourier_transform(fused, G)
    else:
        raise ValueError(
            "_compcell_aft_correction_3c_gradient_weighted: ft_convention "
            f"must be 'libcint' or 'libint'; got {ft_convention!r}"
        )
    F_chg = F_full[n_aux:]

    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g_list = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g_list.size == 0:
        R_g_list = np.zeros((1, 3), dtype=float)

    # Per-pair solid-harmonic conversion, exactly as the value route:
    # G-independent, so it folds into the pair weights.
    if ft_convention == "libcint":
        inv = 1.0 / _per_ao_pair_libint_to_libcint_scale(ao_basis)
        pair_conv = np.outer(inv, inv)
    else:
        pair_conv = np.ones((n_orb, n_orb), dtype=np.float64)
    W3_conv = W3 * pair_conv[None, :, :]  # (n_chg, n_orb, n_orb)

    # ---- Piece A: chg-function phase derivative ------------------------
    # P_c(G) = Σ_μν W3_conv[c, μν] r̂_μν(G) (the libint-convention pair
    # FT — the conversion already sits in W3_conv).
    rho_pair = ao_pair_fourier_transform_bloch(
        ao_basis,
        G,
        R_g_list,
        k_cart=np.zeros(3, dtype=np.float64),
    )  # (n_orb, n_orb, n_G)
    P = W3_conv.reshape(n_chg, n_orb * n_orb) @ rho_pair.reshape(
        n_orb * n_orb, G.shape[0]
    )  # (n_chg, n_G)

    atom_of_chg = _basis_ao_atom_indices(fused, system)[n_aux:]
    # d(conj F_c)/dR_A = +iG conj(F_c) on atom(c); Re[i z] = -Im[z].
    core = np.conj(F_chg) * P * coul[None, :]  # (n_chg, n_G)
    for d in range(3):
        row = -(np.imag(core) * G[None, :, d]).sum(axis=1)  # (n_chg,)
        np.add.at(grad[:, d], atom_of_chg, row)
    del rho_pair, P, core

    # ---- Piece B: AO-pair FT centre derivative -------------------------
    # One G-resolved-weight kernel call for the whole chg batch (M6
    # rung 7): Q[munu, k] = sum_c W3_conv[c, munu] (coul F_chg_c)(k).
    rw_batch = F_chg * coul[None, :]  # (n_chg, n_G)
    Q = np.ascontiguousarray(
        (
            W3_conv.reshape(n_chg, n_orb * n_orb).T @ rw_batch
        ).reshape(n_orb, n_orb, G.shape[0])
    )
    grad += np.asarray(
        ao_pair_fourier_transform_gamma_gradient_gweighted(
            ao_basis,
            G,
            R_g_list,
            Q,
            n_atoms,
        ),
        dtype=np.float64,
    )
    return grad


class _CompcellFitState(NamedTuple):
    """Private compcell fit state shared by the SCF and its gradient.

    The trailing AFT fields (G-PBC-002 milestone 3b) record whether and
    how the fit subtracted the reciprocal-space AFT corrections, so the
    gradient can differentiate exactly the metric/tensor the SCF built.
    """

    Lpq: np.ndarray
    A: np.ndarray
    M_fused: np.ndarray
    T_fused: np.ndarray
    fused_basis: BasisSet
    lat_opts_2c: LatticeSumOptions
    lat_opts_3c: LatticeSumOptions
    eigvals: np.ndarray
    eigvecs: np.ndarray
    keep_mask: np.ndarray
    linear_dep_thr: float
    apply_aft_correction: bool = False
    aft_precision: float = 1e-10
    aft_ft_convention: str = "libint"


def _build_lpq_compcell_state(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    *,
    molecule: Optional[Molecule] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_precision: float = 1e-10,
    aft_ft_convention: str = "libint",
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
) -> _CompcellFitState:
    """Build the periodic GDF ``Lpq`` tensor via compensated charges.

    This is the production GDF path (Sun et al., J. Chem. Phys. 147,
    164119 (2017), DOI 10.1063/1.4998644; PySCF ``_CCGDFBuilder``).
    The AFT long-range correction is ON by default, making the answer
    \u03b7-independent at convergence and numerically well-conditioned
    for both molecular and tight ionic cells.

    Pipeline (see also the module-level "Compensated-charge (compcell)
    GDF pipeline" comment block):

        1. ``modrho_aux = make_modrho_aux_basis(aux_basis, mol)``
        2. ``chg        = make_compensating_basis(modrho_aux, mol,
                                                 eta=compcell_eta)``
        3. ``fused      = make_fused_basis(modrho_aux, chg, mol)``
        4. ``A          = fuse_transform_matrix(modrho_aux, chg)``
        5. ``M_fused    = compute_2c_eri_lattice(fused, system, opts)``
           ``T_fused    = compute_3c_eri_lattice(ao_basis, fused, ...)``
        6. ``M = A @ M_fused @ A.T`` ; ``T = einsum('Pq,Pmn->qmn', A,
           T_fused)``
        7. ``Lpq = eigendecompose-and-threshold(M, T)``

    Parameters
    ----------
    system
        Periodic lattice + atoms.
    ao_basis
        Orbital basis on the unit-cell molecule.
    aux_basis
        Auxiliary basis (NOT pre-modrho-rescaled; this routine handles
        the rescaling internally).
    molecule
        Unit-cell molecule the bases attach to. If ``None``, taken
        from ``system.unit_cell_molecule()``.
    lat_opts
        Lattice-sum options. Defaults to ``LatticeSumOptions()`` (15
        bohr cutoff). With ``rcut_strategy='pyscf_auto'`` (the default),
        the cutoff is overridden per-basis using PySCF-style Gaussian
        decay precision control.
    linear_dep_thr
        Eigenvalue threshold for the eigendecomposition fit. Default
        ``1e-9`` matches Sun 2017 \u00a7III.B.
    compcell_eta
        Smooth-Gaussian exponent for the compensating basis. Default
        ``1.0``. With AFT correction enabled (the default), the SCF
        answer is \u03b7-independent at convergence; \u03b7 is purely a
        numerical-conditioning parameter.
    apply_aft_correction
        Apply the analytical-Fourier-transform long-range correction
        to the fused metric and 3c tensor (default ``True``). Disable
        with ``False`` to recover the bare (non-AFT) compcell path,
        which requires manual \u03b7 tuning and diverges for tight
        ionic cells.
    aft_ft_convention
        FT convention for the AFT correction. ``"libint"`` (default)
        matches vibe-qc's own bare lattice sum convention; ``"libcint"``
        matches PySCF's convention.
    aft_precision
        G-mesh precision target for the AFT correction
        (``exp(-G\u00b2_max/(4\u03b7)) \u2264 precision``). Default ``1e-10``.

    Returns
    -------
    Lpq : np.ndarray of shape (n_kept, n_orb, n_orb)
        Real-valued (Γ-only). ``n_kept <= n_aux`` after threshold
        truncation. AO indices follow libint's per-shell m-ordering;
        comparing element-wise to PySCF requires the L=1
        ``(py, pz, px) -> (px, py, pz)`` AO permutation (and analogous
        per-L perms for higher L when those conventions diverge -- for
        L=2,3 libint and libcint both use ``m = -l..+l`` real
        spherical and agree).

    References
    ----------
    Sun et al., *J. Chem. Phys.* **147**, 164119 (2017),
    DOI 10.1063/1.4998644 -- eigendecomposition + threshold protocol.
    Sun, *J. Comput. Chem.* **38**, 2399 (2017),
    DOI 10.1002/jcc.24890 -- periodic GDF formulation.
    Mintmire, Sabin, Trickey, *Phys. Rev. A* **25**, 88 (1982);
    Dunlap, Connolly, Sabin, *J. Chem. Phys.* **71**, 3396 (1979);
    Whitten, *J. Chem. Phys.* **58**, 4496 (1973) -- modrho theory.
    PySCF ``pyscf/pbc/df/gdf_builder.py:_CCGDFBuilder``.
    """
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    if molecule is None:
        molecule = system.unit_cell_molecule()

    modrho_aux = make_modrho_aux_basis(aux_basis, molecule)
    chg = make_compensating_basis(modrho_aux, molecule, eta=compcell_eta)
    fused = make_fused_basis(modrho_aux, chg, molecule)
    A = fuse_transform_matrix(modrho_aux, chg)

    # If the caller requested an auto rcut strategy, derive a per-basis
    # ``cutoff_bohr`` for the fused-basis lattice sum. This lets the
    # compcell construction use the PySCF-style tight rcut (matched to
    # Gaussian decay precision) instead of vibe-qc's flat geometric cut.
    if rcut_strategy is not None:
        from .lattice_screening import RcutStrategy, make_lattice_opts

        if not isinstance(rcut_strategy, RcutStrategy):
            try:
                rcut_strategy = RcutStrategy(rcut_strategy)
            except Exception as exc:
                raise ValueError(
                    f"build_lpq_compcell: rcut_strategy must be a "
                    f"RcutStrategy or its string value; got "
                    f"{rcut_strategy!r}: {exc}"
                )
        lat_opts_2c = make_lattice_opts(
            fused,
            strategy=rcut_strategy,
            base_opts=lat_opts,
            precision=rcut_precision,
        )
        # Also tune the 3c lattice sum: use the joint (orbital + fused)
        # basis to size the cutoff -- the 3c integrand is
        # <mu_orbital ν_orbital | P_aux>, so its decay is set by whichever
        # of the two bases is most diffuse.
        from ._vibeqc_core import ShellInfo as _SI

        joint_shells = [
            _SI(
                int(s.atom_index),
                int(s.l),
                bool(s.pure),
                list(s.exponents),
                list(s.coefficients),
                list(s.origin),
            )
            for s in list(ao_basis.shells()) + list(fused.shells())
        ]
        joint_basis = BasisSet(molecule, joint_shells, "<ao∪fused>", True)
        lat_opts_3c = make_lattice_opts(
            joint_basis,
            strategy=rcut_strategy,
            base_opts=lat_opts,
            precision=rcut_precision,
        )
    else:
        lat_opts_2c = lat_opts
        lat_opts_3c = lat_opts

    M_fused = np.asarray(compute_2c_eri_lattice(fused, system, lat_opts_2c))
    M_fused = 0.5 * (M_fused + M_fused.T)
    T_fused = np.asarray(compute_3c_eri_lattice(ao_basis, fused, system, lat_opts_3c))

    # AFT long-range correction (PySCF gdf_builder.py:139-196). Subtract
    # the reciprocal-space (chg | fused) long-range Coulomb from the
    # chg rows of M_fused, mirroring the subtraction onto the aux-chg
    # cross-block via Hermitian symmetry. The 3c tensor T_fused is
    # NOT corrected -- PySCF does the analogous AFT correction on the
    # 3c side via `add_ft_j3c`, but it cancels against the implicit
    # G=0 background in the SCF J/K builds and the Lpq fit (the
    # combined effect leaves T_compcell unchanged at machine
    # precision after the eigendecomposition-and-threshold step).
    if apply_aft_correction:
        n_aux_for_aft = modrho_aux.nbasis
        j2c_p = _compcell_aft_correction(
            fused,
            n_aux_for_aft,
            system,
            eta=float(compcell_eta),
            precision=float(aft_precision),
            ft_convention=str(aft_ft_convention),
        )
        # 3c AFT correction (PySCF gdf_builder.py:add_ft_j3c). Subtracts
        # the analytical reciprocal-space (chg | AO_pair) Coulomb from
        # the chg rows of the bare 3c tensor. Symmetric in (mu, ν) at Γ.
        j3c_p = _compcell_aft_correction_3c(
            fused,
            ao_basis,
            n_aux_for_aft,
            system,
            eta=float(compcell_eta),
            precision=float(aft_precision),
            ft_convention=str(aft_ft_convention),
            lat_opts=lat_opts_3c,
        )
        T_fused[n_aux_for_aft:, :, :] -= j3c_p
        # Restore symmetry in the AO-pair indices (the subtraction
        # leaves micro-asymmetry from floating-point reduction order).
        T_fused = 0.5 * (T_fused + T_fused.transpose(0, 2, 1))
        # j2c_p has shape (n_chg, n_fused). Subtract from chg rows of
        # M_fused; subtract its transpose from the matching aux-chg
        # cross-block (Hermitian symmetry of the metric at Γ).
        M_fused[n_aux_for_aft:, :] -= j2c_p
        M_fused[:n_aux_for_aft, n_aux_for_aft:] -= j2c_p[:, :n_aux_for_aft].T
        # Re-symmetrise: floating-point error from the two-sided
        # subtraction can leak ~1e-15 asymmetry.
        M_fused = 0.5 * (M_fused + M_fused.T)

    # Compensated metric + tensor.
    # A has shape (n_aux, n_fused); M_fused is (n_fused, n_fused); T_fused
    # is (n_fused, n_orb, n_orb). The compensated outputs index into the
    # aux block: M[i,j] = S_{P,Q} A[i,P].M_fused[P,Q].A[j,Q] and
    # T[i,m,n] = S_P A[i,P].T_fused[P,m,n].
    M = A @ M_fused @ A.T
    M = 0.5 * (M + M.T)
    T = np.einsum("iP,Pmn->imn", A, T_fused, optimize=True)

    n_aux = modrho_aux.nbasis
    n_orb = ao_basis.nbasis
    if M.shape != (n_aux, n_aux) or T.shape != (n_aux, n_orb, n_orb):
        raise RuntimeError(
            f"build_lpq_compcell: post-compensation shapes mismatch -- "
            f"M={M.shape}, T={T.shape}, expected ({n_aux},{n_aux}) / "
            f"({n_aux},{n_orb},{n_orb})."
        )

    eigvals, U = np.linalg.eigh(M)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    min_eig = float(eigvals[0]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            f"build_lpq_compcell: compensated metric has no positive "
            f"eigenvalue (max eig = {max_eig:.3e}); aux basis or "
            "lattice cutoff invalid."
        )
    # Hardening: the compensated metric is positive semi-definite by
    # construction (A . M_PSD . A.T with M_PSD = Coulomb metric on the
    # fused basis). Significant negative eigenvalues signal a broken
    # convention setup -- either the AFT correction is mismatched to
    # the bare lattice convention, or the lattice cutoff is too small
    # for the chg basis to be compensated. Surface this loudly: users
    # who don't see the warning may not realise the downstream Lpq is
    # numerically singular.
    if min_eig < -1e-2 * max_eig:
        _logger.warning(
            "build_lpq_compcell: compensated metric has SIGNIFICANT "
            "NEGATIVE eigenvalue (min eig = %.3e, max eig = %.3e, "
            "ratio = %.3e). The metric is supposed to be positive "
            "semi-definite. Likely causes: AFT convention mismatch, "
            "lattice cutoff too small for chg basis, or eta too large "
            "for the system. Consider rcut_strategy='pyscf_auto' or a "
            "smaller compcell_eta.",
            min_eig,
            max_eig,
            min_eig / max_eig,
        )
    keep_mask = _metric_keep_mask(
        eigvals, linear_dep_thr, _PLAIN_METRIC_MIN_EIG_FRACTION
    )
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"build_lpq_compcell: no eigenvalues above threshold "
            f"({float(linear_dep_thr):.3e}; max eig = {max_eig:.3e})"
        )
    drop_fraction = (n_aux - n_kept) / max(n_aux, 1)
    if drop_fraction > 0.5:
        _logger.warning(
            "build_lpq_compcell: dropped %d/%d aux modes (%.1f%%). "
            "Dropping >50%% indicates the compensated metric is highly "
            "rank-deficient -- possible AFT/convention issue or eta "
            "tuning needed. Min kept eig = %.3e, threshold = %.3e . "
            "%.3e.",
            n_aux - n_kept,
            n_aux,
            100.0 * drop_fraction,
            float(eigvals[keep_mask][0]),
            linear_dep_thr,
            max_eig,
        )
    elif n_kept < n_aux:
        _logger.info(
            "build_lpq_compcell: truncated %d/%d aux modes "
            "(min kept eig = %.3e, threshold = %.3e . %.3e).",
            n_aux - n_kept,
            n_aux,
            float(eigvals[keep_mask][0]),
            linear_dep_thr,
            max_eig,
        )

    T_flat = T.reshape(n_aux, n_orb * n_orb)
    Lpq_flat = (U[:, keep_mask].T @ T_flat) / np.sqrt(eigvals[keep_mask])[:, None]
    Lpq = Lpq_flat.reshape(n_kept, n_orb, n_orb)
    return _CompcellFitState(
        Lpq=Lpq,
        A=A,
        M_fused=M_fused,
        T_fused=T_fused,
        fused_basis=fused,
        lat_opts_2c=lat_opts_2c,
        lat_opts_3c=lat_opts_3c,
        eigvals=eigvals,
        eigvecs=U,
        keep_mask=keep_mask,
        linear_dep_thr=float(linear_dep_thr),
        apply_aft_correction=bool(apply_aft_correction),
        aft_precision=float(aft_precision),
        aft_ft_convention=str(aft_ft_convention),
    )


def build_lpq_compcell(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    *,
    molecule: Optional[Molecule] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_precision: float = 1e-10,
    aft_ft_convention: str = "libint",
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
) -> np.ndarray:
    """Build the periodic compcell GDF fit tensor."""
    return _build_lpq_compcell_state(
        system,
        ao_basis,
        aux_basis,
        molecule=molecule,
        lat_opts=lat_opts,
        linear_dep_thr=linear_dep_thr,
        compcell_eta=compcell_eta,
        apply_aft_correction=apply_aft_correction,
        aft_precision=aft_precision,
        aft_ft_convention=aft_ft_convention,
        rcut_strategy=rcut_strategy,
        rcut_precision=rcut_precision,
    ).Lpq


# Keep the detailed public API documentation on the stable ndarray wrapper.
build_lpq_compcell.__doc__ = _build_lpq_compcell_state.__doc__


def build_lpq_bloch_compcell(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    k_cart: np.ndarray,
    *,
    molecule: Optional[Molecule] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_precision: float = 1e-10,
    aft_ft_convention: str = "libint",
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    symmetrize_gamma: bool = True,
) -> np.ndarray:
    """Multi-k compcell GDF ``Lpq(k)`` from cell-resolved blocks.

    Same compcell pipeline as :func:`build_lpq_compcell` (modrho + chg
    + fuse + optional AFT correction + eigendecompose-with-threshold)
    but assembled from per-cell DF blocks via Bloch phases for an
    arbitrary k-point. At ``k_cart = 0`` and ``symmetrize_gamma=True``
    the result equals :func:`build_lpq_compcell` (regression-tested).

    For nonzero k the return value is complex-valued.

    Parameters
    ----------
    system, ao_basis, aux_basis, molecule, lat_opts,
    linear_dep_thr, compcell_eta, apply_aft_correction,
    aft_precision, aft_ft_convention, rcut_strategy, rcut_precision
        Same as :func:`build_lpq_compcell`.
    k_cart
        Cartesian k-point (bohr⁻¹), shape ``(3,)``.
    symmetrize_gamma
        If ``True`` and ``k_cart ≈ 0``, symmetrise the 3c tensor in
        the (mu, ν) AO-pair indices (analytic property at Γ; reduces
        floating-point noise).

    Returns
    -------
    Lpq : np.ndarray of shape (n_kept, n_orb, n_orb)
        complex128 for nonzero k; float64 (real-part) at Γ when
        ``symmetrize_gamma=True``. n_kept <= n_aux after threshold.

    Notes
    -----
    Uses ``compute_2c_eri_lattice_blocks`` and
    ``compute_3c_eri_lattice_blocks`` on the FUSED basis to get per-
    cell tensors, then ``bloch_sum_*`` to assemble at k. The AFT
    correction (when enabled) is applied per-k via a q-shifted
    Coulomb kernel ``coulG = 4pi / |G + k|^2`` (which is finite for
    any k inside the BZ; the G=0 special case for the Γ-only build
    only arises at k=0).
    """
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    if molecule is None:
        molecule = system.unit_cell_molecule()
    k = np.asarray(k_cart, dtype=float).reshape(3)
    is_gamma = float(np.linalg.norm(k)) < 1e-14

    modrho_aux = make_modrho_aux_basis(aux_basis, molecule)
    chg = make_compensating_basis(modrho_aux, molecule, eta=compcell_eta)
    fused = make_fused_basis(modrho_aux, chg, molecule)
    A = fuse_transform_matrix(modrho_aux, chg)
    n_aux = modrho_aux.nbasis
    n_orb = ao_basis.nbasis

    # rcut auto-tuning, same as build_lpq_compcell. For multi-k the
    # 3c integrand also includes phase factors but its Gaussian decay
    # is governed by the same exponents.
    if rcut_strategy is not None:
        from .lattice_screening import RcutStrategy, make_lattice_opts

        if not isinstance(rcut_strategy, RcutStrategy):
            try:
                rcut_strategy = RcutStrategy(rcut_strategy)
            except Exception as exc:
                raise ValueError(
                    "build_lpq_bloch_compcell: rcut_strategy must be a "
                    f"RcutStrategy or its string value; got "
                    f"{rcut_strategy!r}: {exc}"
                )
        lat_opts_2c = make_lattice_opts(
            fused,
            strategy=rcut_strategy,
            base_opts=lat_opts,
            precision=rcut_precision,
        )
        joint_shells = [
            ShellInfo(
                int(s.atom_index),
                int(s.l),
                bool(s.pure),
                list(s.exponents),
                list(s.coefficients),
                list(s.origin),
            )
            for s in list(ao_basis.shells()) + list(fused.shells())
        ]
        joint_basis = BasisSet(molecule, joint_shells, "<ao∪fused>", True)
        lat_opts_3c = make_lattice_opts(
            joint_basis,
            strategy=rcut_strategy,
            base_opts=lat_opts,
            precision=rcut_precision,
        )
    else:
        lat_opts_2c = lat_opts
        lat_opts_3c = lat_opts

    # Cell-resolved blocks -> Bloch-sum at k.
    _, m_vecs, m_blocks = compute_2c_eri_lattice_blocks(
        fused,
        system,
        lat_opts_2c,
    )
    _, t_vecs, t_blocks = compute_3c_eri_lattice_blocks(
        ao_basis,
        fused,
        system,
        lat_opts_3c,
    )
    M_fused = bloch_sum_2c_eri_blocks(m_vecs, m_blocks, k)
    T_fused = bloch_sum_3c_eri_blocks(t_vecs, t_blocks, k)

    # Hermitian symmetrise the 2c metric (M(-g)^* = M(g) up to noise).
    M_fused = 0.5 * (M_fused + M_fused.conj().T)
    if symmetrize_gamma and is_gamma:
        T_fused = 0.5 * (T_fused + np.swapaxes(T_fused, 1, 2))

    # AFT correction at k (q = k for the chg-anything subtraction). At
    # Γ (q = 0) the result is real and the G = 0 term is excluded from
    # the AFT mesh (singular 1/G^2; the compcell construction already
    # cancels the monopole). At k != 0 the result is complex, |G+q| > 0
    # everywhere inside the BZ, and the full mesh is summed.
    if apply_aft_correction:
        q_aft = None if is_gamma else k
        j2c_p = _compcell_aft_correction(
            fused,
            n_aux,
            system,
            eta=float(compcell_eta),
            precision=float(aft_precision),
            ft_convention=str(aft_ft_convention),
            q=q_aft,
        )
        j3c_p = _compcell_aft_correction_3c(
            fused,
            ao_basis,
            n_aux,
            system,
            eta=float(compcell_eta),
            precision=float(aft_precision),
            ft_convention=str(aft_ft_convention),
            q=q_aft,
            lat_opts=lat_opts_3c,
        )
        # Promote M_fused / T_fused to complex if the subtraction is
        # complex (q != 0). The Bloch-summed tensors are already complex
        # at q != 0, but the early "is_gamma" path used to return real;
        # we leave that alone and only widen here when needed.
        if np.iscomplexobj(j2c_p) and not np.iscomplexobj(M_fused):
            M_fused = M_fused.astype(np.complex128)
        if np.iscomplexobj(j3c_p) and not np.iscomplexobj(T_fused):
            T_fused = T_fused.astype(np.complex128)
        # Hermitian subtraction. At Γ this reduces to the real-symmetric
        # ``M_fused[:n_aux, n_aux:] -= j2c_p[:, :n_aux].T`` because
        # j2c_p is real and the .conj() is a no-op.
        M_fused[n_aux:, :] -= j2c_p
        M_fused[:n_aux, n_aux:] -= j2c_p[:, :n_aux].conj().T
        M_fused = 0.5 * (M_fused + M_fused.conj().T)
        T_fused[n_aux:, :, :] -= j3c_p
        if symmetrize_gamma and is_gamma:
            T_fused = 0.5 * (T_fused + np.swapaxes(T_fused, 1, 2))

    # Compensated metric + tensor (A is real; M_fused complex at k != 0).
    M = A @ M_fused @ A.T.astype(M_fused.dtype, copy=False)
    M = 0.5 * (M + M.conj().T)
    T = np.einsum(
        "iP,Pmn->imn", A.astype(T_fused.dtype, copy=False), T_fused, optimize=True
    )

    if M.shape != (n_aux, n_aux) or T.shape != (n_aux, n_orb, n_orb):
        raise RuntimeError(
            f"build_lpq_bloch_compcell: post-compensation shapes "
            f"mismatch -- M={M.shape}, T={T.shape}, expected "
            f"({n_aux},{n_aux}) / ({n_aux},{n_orb},{n_orb})."
        )

    eigvals, U = np.linalg.eigh(M)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    min_eig = float(eigvals[0]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            f"build_lpq_bloch_compcell: compensated metric has no "
            f"positive eigenvalue at k={k_cart} (max eig = {max_eig:.3e})"
        )
    if min_eig < -1e-2 * max_eig:
        _logger.warning(
            "build_lpq_bloch_compcell: compensated metric has "
            "SIGNIFICANT NEGATIVE eigenvalue at k=%s (min=%.3e, "
            "max=%.3e). See build_lpq_compcell warning for diagnosis.",
            k_cart,
            min_eig,
            max_eig,
        )
    keep_mask = _metric_keep_mask(
        eigvals, linear_dep_thr, _PLAIN_METRIC_MIN_EIG_FRACTION
    )
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"build_lpq_bloch_compcell: no eigenvalues above threshold "
            f"({float(linear_dep_thr):.3e}) at k={k_cart}"
        )

    T_flat = T.reshape(n_aux, n_orb * n_orb)
    Lpq_flat = (U[:, keep_mask].conj().T @ T_flat) / np.sqrt(eigvals[keep_mask])[
        :, None
    ]
    return Lpq_flat.reshape(n_kept, n_orb, n_orb)


def build_lpq_native(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    apply_modrho: bool = True,
    algorithm: str = "bare",
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
) -> np.ndarray:
    """Build the periodic GDF Lpq tensor entirely with vibe-qc machinery.

    ::

        M_PQ      = S_T (P_0 | Q_T)             (compute_2c_eri_lattice)
        T_{P,muν} = S_T (P_0 | mu_0 ν_T)          (compute_3c_eri_lattice)
        L . L^T   = M                            (Cholesky factor of the metric)
        Lpq       = L^{-1} . T                   (one triangular solve)

    so that ``Lpq[L, mu, ν]`` satisfies the GDF factorisation
    ``(muν|ls)_periodic ≈ S_L Lpq[L, mu, ν] . Lpq[L, l, s]``.

    Parameters
    ----------
    system
        Periodic lattice + atoms.
    ao_basis
        Orbital basis (vibe-qc BasisSet on the unit-cell molecule).
    aux_basis
        Auxiliary basis (vibe-qc BasisSet on the unit-cell molecule),
        typically built via :func:`make_aux_basis_set`.
    lat_opts
        Lattice-sum options (cutoff for the image sum, etc.). If ``None``,
        defaults to ``LatticeSumOptions()`` (15 bohr cutoff). Production
        SCF should pass an opts object whose cutoff has been tested to
        give converged Lpq elements.
    linear_dep_thr
        Absolute eigenvalue threshold for the eigendecomposition-based
        fit: modes with ``l_k <= linear_dep_thr`` are dropped from the
        auxiliary subspace. See :func:`_metric_keep_mask` for the
        convention and its provenance.

        Per Sun et al. 2017 (DOI 10.1063/1.4998644) Sec.III.B, eigen-
        decomposition with truncation is the principled handling for
        residual linear dependence in the aux metric -- Cholesky breaks
        down on larger aux bases even when the metric is nominally SPD.
        The default ``1e-9`` matches the Sun 2017 Fig. 2 calibration
        (settings ``< 1e-12`` reintroduce noise; ``> 1e-7`` over-
        truncate).

    Returns
    -------
    Lpq : np.ndarray of shape (n_kept, n_orb, n_orb)
        Real-valued (Γ-only). ``n_kept <= n_aux`` after threshold
        truncation. The dropped count is reported via the logger if
        any modes were truncated.

    Notes
    -----
    Output is in **vibe-qc (libint) AO ordering** -- both the orbital
    and aux indices follow libint's per-shell m-ordering (p-shells:
    py, pz, px) and per-atom shell grouping. Callers comparing against
    PySCF (libcint ordering) must apply the AO permutation; see the
    helper in ``periodic_rhf_gdf._build_permutation_matrix``.

    References
    ----------
    - Sun et al., *J. Chem. Phys.* **147**, 164119 (2017),
      DOI 10.1063/1.4998644 -- Sec.III.B for the eigendecomposition-+-
      threshold protocol that supersedes plain Cholesky.
    """
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    if algorithm not in ("bare", "rsgdf"):
        raise ValueError(
            f"build_lpq_native: algorithm must be 'bare' or 'rsgdf', got {algorithm!r}"
        )

    # 1. Periodic 2c metric (n_aux, n_aux).
    # 2. Periodic 3c tensor (n_aux, n_orb, n_orb).
    # Both must use the SAME convention (bare or RSGDF) so that the
    # Lpq fit (muν|ls) ≈ T^T M^{-1} T is mathematically consistent.
    if algorithm == "bare":
        # Direct image sum on the aux. Convergence depends on the aux
        # basis; tight aux + per-shell precision-controlled cutoff
        # gives a finite (truncation-controlled) M.
        M = np.asarray(compute_2c_eri_lattice(aux_basis, system, lat_opts))
        T = np.asarray(compute_3c_eri_lattice(ao_basis, aux_basis, system, lat_opts))
    else:  # rsgdf
        # Range-separated GDF (Ye & Berkelbach 2021,
        # DOI 10.1063/5.0046617). Splits 1/r = erfc(wr)/r + erf(wr)/r;
        # SR is real-space (libint Operator::erfc_coulomb on the aux
        # and AO-pair sides via compute_*_eri_lattice_sr); LR is
        # reciprocal-space (closed-form Gaussian FTs via
        # rsgdf_lr_2c_metric / rsgdf_lr_3c_tensor on a small G-mesh).
        # G-mesh is shared between 2c and 3c LR builds for efficiency.
        g_mesh = rsgdf_g_mesh(system, rsgdf_omega, precision=rsgdf_g_precision)
        M_sr = np.asarray(
            compute_2c_eri_lattice_sr(aux_basis, system, lat_opts, rsgdf_omega)
        )
        M_lr = rsgdf_lr_2c_metric(aux_basis, system, rsgdf_omega, g_mesh=g_mesh)
        M = M_sr + M_lr

        T_sr = np.asarray(
            compute_3c_eri_lattice_sr(
                ao_basis, aux_basis, system, lat_opts, rsgdf_omega
            )
        )
        T_lr = rsgdf_lr_3c_tensor(
            ao_basis, aux_basis, system, rsgdf_omega, g_mesh=g_mesh
        )
        T = T_sr + T_lr

        _logger.info(
            "build_lpq_native (rsgdf): omega=%.3f, |G|=%d, "
            "||M_SR||=%.3e, ||M_LR||=%.3e, ||T_SR||=%.3e, ||T_LR||=%.3e",
            rsgdf_omega,
            len(g_mesh),
            float(np.linalg.norm(M_sr)),
            float(np.linalg.norm(M_lr)),
            float(np.linalg.norm(T_sr)),
            float(np.linalg.norm(T_lr)),
        )
    M = 0.5 * (M + M.T)  # symmetrise (kernel does this; defensive)

    n_aux, n_orb, _ = T.shape
    if M.shape != (n_aux, n_aux):
        raise RuntimeError(
            f"build_lpq_native: 2c metric shape {M.shape} mismatches 3c "
            f"aux dimension {n_aux}; aux basis inconsistent."
        )

    # 3. Modrho rescaling (per-shell scalar; multiplicative on integrals).
    #    Sets every contracted aux shell's monopole moment to √(1/(4pi)).
    #    For the GDF identity (muν|ls) ≈ S_L Lpq[L,muν] . Lpq[L,ls], the
    #    modrho transform is invariant -- Lpq matrix elements computed
    #    here are mathematically equivalent to those without modrho,
    #    just numerically robust against the diffuse-aux ill-conditioning.
    if apply_modrho:
        alpha = modrho_scales(aux_basis)
        # M[P, Q] -> a[P] . M[P, Q] . a[Q]
        M = (alpha[:, None] * M) * alpha[None, :]
        # Re-symmetrise: outer product of the alpha array preserves
        # symmetry analytically, but mixing types can let 1e-16 drift.
        M = 0.5 * (M + M.T)
        # T[P, mu, ν] -> a[P] . T[P, mu, ν]
        T = T * alpha[:, None, None]

    # 4. Eigendecomposition + threshold (NOT Cholesky -- see docstring
    #    rationale, Sun 2017 Sec.III.B). For each eigenvalue l_k > thr.max,
    #    the corresponding Lpq column is
    #        Lpq[k, mu, ν]  =  (1/√l_k) . S_P U[P, k] . T[P, mu, ν]
    eigvals, U = np.linalg.eigh(M)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            f"build_lpq_native: 2c metric has no positive eigenvalue "
            f"(max eig = {max_eig:.3e}); the aux basis or lattice "
            f"cutoff is invalid."
        )
    keep_mask = _metric_keep_mask(
        eigvals, linear_dep_thr, _PLAIN_METRIC_MIN_EIG_FRACTION
    )
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"build_lpq_native: no eigenvalues above threshold "
            f"({float(linear_dep_thr):.3e}; max eig = {max_eig:.3e}); "
            f"check the aux basis and lattice cutoff."
        )
    if n_kept < n_aux:
        _logger.info(
            "build_lpq_native: truncated %d/%d aux modes "
            "(min kept eig = %.3e, threshold = %.3e, max eig = %.3e).",
            n_aux - n_kept,
            n_aux,
            float(eigvals[keep_mask][0]),
            float(linear_dep_thr),
            max_eig,
        )

    T_flat = T.reshape(n_aux, n_orb * n_orb)
    Lpq_flat = (U[:, keep_mask].T @ T_flat) / np.sqrt(eigvals[keep_mask])[:, None]
    return Lpq_flat.reshape(n_kept, n_orb, n_orb)


class _RsgdfDensePairFT(NamedTuple):
    """Shared dense-mesh AO-pair FT ``r̂_muν(G)`` on the ke-cutoff mesh.

    The Bloch AO-pair FT (the dominant cost of the rsgdf GDF path -- the
    C++ ``ao_pair_fourier_transform_bloch_cxx`` kernel) is computed
    identically by both the cderi build (:func:`build_lpq_native_fft`)
    and the EWALD_3D V_ne FT
    (:func:`vibeqc.periodic_v_ne.compute_v_ne_ewald_3d_ft_gamma`). On a
    single GDF SCF it was being run twice. The driver
    (:func:`vibeqc.pbc_gdf.run_pbc_gdf_rhf`) builds this bundle once via
    :func:`_rsgdf_dense_pair_ft` and threads it into both through their
    optional ``pair_ft_shared`` argument, halving that build cost.

    The ``ke_cutoff`` / ``cutoff_bohr`` / ``n_ao`` provenance fields let
    :func:`_resolve_pair_ft_shared` reject a bundle whose parameters do
    not match the consumer's -- a mismatch yields a loud error rather
    than a silently-wrong tensor.
    """

    G: np.ndarray  # (n_G, 3) -- nonzero dense G-mesh
    G2: np.ndarray  # (n_G,) -- squared norms |G|^2 (== the old G2[nz])
    pair_ft: np.ndarray  # (n_ao, n_ao, n_G) -- scaled Bloch AO-pair FT
    ke_cutoff: float
    cutoff_bohr: float
    n_ao: int


def _rsgdf_dense_pair_ft(
    ao_basis: BasisSet,
    system: PeriodicSystem,
    ke_cutoff: float,
    lat_opts: LatticeSumOptions,
) -> "_RsgdfDensePairFT":
    """Compute the shared ``(G, |G|^2, r̂_muν(G))`` bundle on the dense
    ke-cutoff FFT mesh -- the AO-pair FT common to the rsgdf cderi build
    and the EWALD_3D V_ne FT. See :class:`_RsgdfDensePairFT`.

    This is the verbatim ``(G, pair_ft)`` computation that previously
    lived inline in both consumers; factoring it here lets the driver
    run it once per SCF instead of twice.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import direct_lattice_cells

    ke_cutoff = float(ke_cutoff)
    cutoff_bohr = float(lat_opts.cutoff_bohr)
    G_all = rsgdf_dense_g_mesh(system, ke_cutoff)
    G2_all = (G_all**2).sum(axis=1)
    nz = G2_all > 0
    G = G_all[nz]
    G2 = G2_all[nz]

    # Bloch-summed pair-FT at k=0 -- S_R ∫chi_mu(r).chi_ν(r-R).exp(-iG.r) dr.
    cells = direct_lattice_cells(system, cutoff_bohr)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    pair_ft = ao_pair_fourier_transform_bloch(
        ao_basis,
        G,
        R_g,
        k_cart=np.zeros(3),
    )
    # Per-L calibration (libint pair-FT convention vs the real-Y_lm plain
    # pair-FT the C++ kernel produces): the pair-scale is the outer
    # product of per-AO single-AO scales -- confirmed bit-exact vs PySCF
    # ft_aopair on the LiH (muν|ls) tensor after L=1 m-perm.
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    # In place: pair_ft is (n_ao, n_ao, n_G) -- the bundle's dominant
    # memory -- and an out-of-place multiply transiently doubles it.
    pair_ft *= pair_scales[:, :, None]
    return _RsgdfDensePairFT(
        G, G2, pair_ft, ke_cutoff, cutoff_bohr, int(ao_basis.nbasis)
    )


def _resolve_pair_ft_shared(
    shared: "Optional[_RsgdfDensePairFT]",
    ao_basis: BasisSet,
    system: PeriodicSystem,
    ke_cutoff: float,
    lat_opts: LatticeSumOptions,
) -> "_RsgdfDensePairFT":
    """Return the dense-mesh AO-pair FT bundle to use -- the caller's
    precomputed ``shared`` (validated against the local parameters) or a
    freshly built one. Provenance mismatch is a loud error, never a
    silent wrong-tensor reuse.
    """
    if shared is None:
        return _rsgdf_dense_pair_ft(ao_basis, system, ke_cutoff, lat_opts)
    if (
        shared.ke_cutoff != float(ke_cutoff)
        or shared.cutoff_bohr != float(lat_opts.cutoff_bohr)
        or shared.n_ao != int(ao_basis.nbasis)
    ):
        raise ValueError(
            "pair_ft_shared provenance mismatch: bundle built with "
            f"(ke_cutoff={shared.ke_cutoff}, cutoff_bohr={shared.cutoff_bohr}, "
            f"n_ao={shared.n_ao}) but the consumer expects "
            f"(ke_cutoff={float(ke_cutoff)}, "
            f"cutoff_bohr={float(lat_opts.cutoff_bohr)}, "
            f"n_ao={int(ao_basis.nbasis)})."
        )
    return shared


def _rsgdf_weighted_2c_metric_gradient(
    aux_basis: BasisSet,
    system: PeriodicSystem,
    *,
    ke_cutoff: float,
    weight: np.ndarray,
    tail_ke_cutoff: Optional[float] = None,
    tail_chunk_g: int = 65536,
) -> np.ndarray:
    """Atomic gradient of the weighted Γ rsgdf 2c metric value.

    Returns ``d/dR_A [ Σ_PQ weight[P, Q] · M_PQ ]`` as an
    ``(n_atoms, 3)`` array, where ``M`` is exactly the
    :func:`build_lpq_native_fft` dense-mesh 2c metric

        M_PQ = Re (4π/V) Σ_{G != 0} conj(F̂_P(G)) F̂_Q(G) / G²

    including the optional high-|G| tail completion over
    ``sqrt(2 ke_cutoff) < |G| <= sqrt(2 tail_ke_cutoff)``.

    G-PBC-002 milestone 6 (Item-2 rung 1): the sum shares the compcell
    AFT correction's structure — the reciprocal mesh, the 1/G² kernel,
    and the radial/angular FT content depend on the lattice and
    exponents only, so the single atom-position dependence is each
    shell's ``exp(-iG.R)`` phase and the derivative is complete with

        grad[A, d] = - Σ_{P on A} row-sum(weight ∘ Im C_d)[P]
                     + Σ_{Q on A} col-sum(weight ∘ Im C_d)[Q],
        C_d[P, Q] = Σ_G coul(G) G_d conj(F̂_P(G)) F̂_Q(G)

    (the full-aux generalisation of
    :func:`_compcell_aft_correction_2c_gradient_weighted`; isolation FD
    regression in ``tests/test_periodic_gdf_gradient.py``). Both the
    base mesh and the tail accumulate in G-chunks bounded by
    ``tail_chunk_g``.
    """
    n_aux = int(aux_basis.nbasis)
    n_atoms = len(system.unit_cell)
    W = np.asarray(weight, dtype=np.float64)
    if W.shape != (n_aux, n_aux):
        raise ValueError(
            "_rsgdf_weighted_2c_metric_gradient: weight must have shape "
            f"(n_aux, n_aux) = ({n_aux}, {n_aux}); got {W.shape}."
        )
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    atom_of = _basis_ao_atom_indices(aux_basis, system)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    chunk = max(int(tail_chunk_g), 1)

    def _accumulate(G_block: np.ndarray) -> None:
        G2_block = (G_block**2).sum(axis=1)
        nz = G2_block > 0
        Gc = G_block[nz]
        if Gc.shape[0] == 0:
            return
        coul_c = (4.0 * np.pi) / G2_block[nz] / V
        F = rsgdf_aux_fourier_transform(aux_basis, Gc)
        for d in range(3):
            C_d = (np.conj(F) * (coul_c * Gc[:, d])[None, :]) @ F.T
            X = W * np.imag(C_d)
            np.subtract.at(grad[:, d], atom_of, X.sum(axis=1))
            np.add.at(grad[:, d], atom_of, X.sum(axis=0))

    G_base = rsgdf_dense_g_mesh(system, float(ke_cutoff))
    for lo in range(0, G_base.shape[0], chunk):
        _accumulate(G_base[lo : lo + chunk])

    if tail_ke_cutoff is not None and float(tail_ke_cutoff) > float(ke_cutoff):
        G_tail_all = rsgdf_dense_g_mesh(system, float(tail_ke_cutoff))
        G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
        tail_norms = np.linalg.norm(G_tail_all, axis=1)
        G_tail = G_tail_all[tail_norms > G_base_max]
        for lo in range(0, G_tail.shape[0], chunk):
            _accumulate(G_tail[lo : lo + chunk])
    return grad


def _rsgdf_weighted_3c_tensor_gradient(
    aux_basis: BasisSet,
    ao_basis: BasisSet,
    system: PeriodicSystem,
    *,
    ke_cutoff: float,
    weight: np.ndarray,
    lat_opts: Optional[LatticeSumOptions] = None,
    tail_ke_cutoff: Optional[float] = None,
    tail_chunk_g: int = 65536,
    tail_pair_ft_screen: float = 0.0,
) -> np.ndarray:
    """Atomic gradient of the weighted Γ rsgdf 3c tensor value.

    Returns ``d/dR_A [ Σ_{P,μν} weight[P, μ, ν] · T_Pμν ]`` as an
    ``(n_atoms, 3)`` array, where ``T`` is exactly the
    :func:`build_lpq_native_fft` dense-mesh 3c tensor

        T_Pμν = Re (4π/V) Σ_{G != 0} conj(F̂_P(G)) r̂^per_μν(G)
                s_μ s_ν / G²

    with the Bloch-summed periodic pair FT over the ``lat_opts`` cell
    list, the rsgdf per-AO pair scales ``s``, and the optional high-|G|
    tail completion (G-PBC-002 milestone 6, Item-2 rung 2; the full-aux
    dense-mesh sibling of
    :func:`_compcell_aft_correction_3c_gradient_weighted`).

    Two atom-position dependencies, each analytic: the aux-shell phase
    (vectorised reciprocal sums with per-atom scatter) and the AO-pair
    FT centres (the weighted OpenMP kernel
    ``ao_pair_fourier_transform_gamma_gradient_weighted`` driven once
    per aux function per G-chunk, with ``Re Σ coul conj(F̂_P) r̂ =
    Re Σ (coul F̂_P) conj(r̂)`` matching its conjugation contract).
    Mesh, Coulomb kernel, radial content, pair scales, and the cell
    list are lattice/exponent/cutoff-only, so these two pieces are the
    complete derivative (isolation FD gate in
    ``tests/test_periodic_gdf_gradient.py``).

    **Schwarz-screened fits.** Both pieces are LINEAR in the per-pair
    weight entries ``weight[P, mu, ν]``: piece A contracts the weight
    with the pair-FT values (``P = W3_flat @ rho_pair``) and piece B
    folds it into the per-pair G-resolved kernel weight
    (``Q = W3_flat.T @ (F coul)``). Callers differentiating a
    Schwarz-screened fit therefore zero the masked-pair entries of
    ``weight`` BEFORE dispatch — the masked pairs then contribute
    exactly zero to both pieces, which is the exact derivative of the
    screened objective at fixed mask (masked T entries are hard zeros,
    geometry-independent). No kernel-side masking is needed.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import (
        ao_pair_fourier_transform_gamma_gradient_gweighted,
        direct_lattice_cells,
    )

    n_aux = int(aux_basis.nbasis)
    n_orb = int(ao_basis.nbasis)
    n_atoms = len(system.unit_cell)
    W3 = np.asarray(weight, dtype=np.float64)
    if W3.shape != (n_aux, n_orb, n_orb):
        raise ValueError(
            "_rsgdf_weighted_3c_tensor_gradient: weight must have shape "
            f"(n_aux, n_orb, n_orb) = ({n_aux}, {n_orb}, {n_orb}); got "
            f"{W3.shape}."
        )
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)

    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    # The scale is G-independent, so it folds into the pair weights for
    # both derivative pieces (the C++ kernels produce the unscaled
    # libint pair FT).
    W3_scaled = W3 * pair_scales[None, :, :]
    W3_flat = W3_scaled.reshape(n_aux, n_orb * n_orb)

    atom_of_aux = _basis_ao_atom_indices(aux_basis, system)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    chunk = max(int(tail_chunk_g), 1)

    def _accumulate(G_block: np.ndarray, screen_tol: float = 0.0) -> None:
        G2_block = (G_block**2).sum(axis=1)
        nz = G2_block > 0
        Gc = np.ascontiguousarray(G_block[nz])
        if Gc.shape[0] == 0:
            return
        coul_c = (4.0 * np.pi) / G2_block[nz] / V
        F = rsgdf_aux_fourier_transform(aux_basis, Gc)
        # Piece A's value pair-FT mirrors the SCF's tail screen (M6
        # rung 9 bit-consistency); piece B's gweighted derivative
        # kernel has no screen -- its screened-block derivative is
        # direct and unamplified (~1e-9-class), documented in the M6
        # handover entry.
        rho_pair = ao_pair_fourier_transform_bloch(
            ao_basis, Gc, R_g, k_cart=np.zeros(3, dtype=np.float64),
            screen_tol=float(screen_tol),
        )  # (n_orb, n_orb, n_G), unscaled
        # ---- aux-phase piece --------------------------------------
        P = W3_flat @ rho_pair.reshape(n_orb * n_orb, Gc.shape[0])
        core = np.conj(F) * P * coul_c[None, :]  # (n_aux, n_G)
        for d in range(3):
            row = -(np.imag(core) * Gc[None, :, d]).sum(axis=1)
            np.add.at(grad[:, d], atom_of_aux, row)
        del rho_pair, P, core
        # ---- pair-FT-centre piece ---------------------------------
        # One G-resolved-weight kernel call for the whole aux batch
        # (M6 rung 7): Q[μν, k] = Σ_p W3_scaled[p, μν] (coul F_p)(k)
        # folds the n_aux separable weights into one complex per-G pair
        # weight via a single ZGEMM, so the expensive FT-derivative
        # pass runs once instead of n_aux times.
        rw_batch = F * coul_c[None, :]  # (n_aux, n_G)
        Q = np.ascontiguousarray(
            (W3_flat.T @ rw_batch).reshape(n_orb, n_orb, Gc.shape[0])
        )
        grad[:, :] += np.asarray(
            ao_pair_fourier_transform_gamma_gradient_gweighted(
                ao_basis, Gc, R_g, Q, n_atoms
            ),
            dtype=np.float64,
        )

    G_base = rsgdf_dense_g_mesh(system, float(ke_cutoff))
    for lo in range(0, G_base.shape[0], chunk):
        _accumulate(G_base[lo : lo + chunk])

    if tail_ke_cutoff is not None and float(tail_ke_cutoff) > float(ke_cutoff):
        G_tail_all = rsgdf_dense_g_mesh(system, float(tail_ke_cutoff))
        G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
        tail_norms = np.linalg.norm(G_tail_all, axis=1)
        G_tail = G_tail_all[tail_norms > G_base_max]
        for lo in range(0, G_tail.shape[0], chunk):
            _accumulate(
                G_tail[lo : lo + chunk],
                screen_tol=float(tail_pair_ft_screen),
            )
    return grad


def _rsgdf_weighted_3c_tensor_gradient_bloch(
    aux_basis: BasisSet,
    ao_basis: BasisSet,
    system: PeriodicSystem,
    *,
    ke_cutoff: float,
    weight: np.ndarray,
    k_ket: np.ndarray,
    q_cart: np.ndarray,
    lat_opts: Optional[LatticeSumOptions] = None,
    tail_ke_cutoff: Optional[float] = None,
    tail_chunk_g: int = 65536,
    g_mesh: Optional[np.ndarray] = None,
    kernel_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Atomic gradient of a complex-weighted shared-q rsgdf 3c tensor.

    Returns ``d/dR_A [ Re S_{P,muν} weight[P, mu, ν] . T_Pmuν ]`` as an
    ``(n_atoms, 3)`` array, where ``T`` is ONE bra-k slice of the
    :func:`build_lpq_bloch_native_fft_shared_q` dense-mesh 3c tensor

        T_Pmuν = (4pi/V) S_{|G+q| != 0} conj(F̂_P(G+q)) r̂^{k_ket}_muν(G+q)
                 s_mu s_ν / |G+q|²

    on the shifted sphere ``0 < |G+q| <= sqrt(2 ke_cutoff)`` with the
    ket-Bloch pair FT at ``k_ket = k_bra + q``, rsgdf per-AO pair
    scales ``s``, and the optional shifted tail completion (unscreened,
    like the shared-q SCF builder). The Bloch/complex-weight sibling of
    :func:`_rsgdf_weighted_3c_tensor_gradient` (G-PBC-002 Item-4 rung
    3a); at ``q = 0``, ``k_ket = 0`` with a real ``weight`` it reduces
    to that Γ kernel term by term.

    ``g_mesh``/``kernel_weights`` (both or neither) replace the
    internally derived mesh and 4pi/|G+q|²/V weights with an externally
    supplied already-shifted physical mesh and per-point reciprocal
    kernel weights (1/V included) — the slab-truncated route
    (:func:`_build_lpq_bloch_slab_truncated`) uses this to
    differentiate its Sundararaman-Arias-weighted 3c tensor, whose
    finite ``K(0)`` zero mode must NOT be zero-filtered: at ``G+q = 0``
    the aux-phase piece vanishes identically (its ``(G+q)_d`` factor is
    zero and single-centre aux functions carry geometry only in the
    phase) but the AO-pair centre piece is finite — the Bloch pair
    density at zero momentum still depends on the relative AO centres.
    With both defaults ``None`` this function is byte-identical to its
    pre-parameter behaviour (the default branch below is untouched).

    Two atom-position dependencies, each analytic (mesh, Coulomb
    kernel, radial FT content, pair scales, and cell list are
    lattice/exponent/cutoff-only):

    * **Aux-shell phase.** ``F̂_P`` carries ``exp(-i(G+q).R_P)``
      (:func:`rsgdf_aux_fourier_transform`), so for P on atom A
      ``d conj(F̂_P)/dR_{A,d} = +i (G+q)_d conj(F̂_P)`` and

          d/dR_{A,d} Re S = Re[ i S_G (G+q)_d core_P(G) ]
                          = - S_G (G+q)_d Im core_P(G),
          core_P(G) = conj(F̂_P) coul(G+q) S_muν W_scaled[P, muν]
                      r̂^{k_ket}_muν(G+q)

      — with a COMPLEX weight the same expression holds verbatim, the
      ``Re[...]`` picking ``-Im`` of the full complex ``core``. At Γ
      with a real weight this is exactly the Γ kernel's imag-part
      contraction.

    * **AO-pair centres.** ``Re S_P W aux_w r̂ = Re S_muν Q_muν(G)
      conj(r̂^{k_ket}_muν(G))`` with

          Q_muν(G) = S_P conj(W_scaled[P, muν]) F̂_P(G+q) coul(G+q),

      dispatched to the rung-1 Bloch gweighted kernel
      :func:`vibeqc._aopair_ft.
      ao_pair_fourier_transform_bloch_gradient_gweighted` at the ket
      momentum. For a real weight the ``conj`` is a no-op and Q is the
      Γ kernel's ``W3_flat.T @ (F * coul)``.

    **Schwarz-screened fits.** As in the Γ kernel, both pieces are
    LINEAR in the per-pair weight entries (piece A through
    ``W3_flat @ rho_pair``, piece B through the folded per-pair weight
    ``Q``), so callers differentiating a screened fit zero the
    masked-pair entries of ``weight`` before dispatch; the masked pairs
    then contribute exactly zero to both pieces — the exact fixed-mask
    derivative of the screened objective. No kernel-side masking is
    needed.
    """
    from ._aopair_ft import (
        ao_pair_fourier_transform_bloch,
        ao_pair_fourier_transform_bloch_gradient_gweighted,
    )
    from ._vibeqc_core import direct_lattice_cells

    n_aux = int(aux_basis.nbasis)
    n_orb = int(ao_basis.nbasis)
    n_atoms = len(system.unit_cell)
    W3 = np.asarray(weight, dtype=np.complex128)
    if W3.shape != (n_aux, n_orb, n_orb):
        raise ValueError(
            "_rsgdf_weighted_3c_tensor_gradient_bloch: weight must have "
            f"shape (n_aux, n_orb, n_orb) = ({n_aux}, {n_orb}, {n_orb}); "
            f"got {W3.shape}."
        )
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    if (g_mesh is None) != (kernel_weights is None):
        raise ValueError(
            "_rsgdf_weighted_3c_tensor_gradient_bloch: g_mesh and "
            "kernel_weights must be supplied together"
        )
    if kernel_weights is not None and tail_ke_cutoff is not None:
        raise ValueError(
            "_rsgdf_weighted_3c_tensor_gradient_bloch: external "
            "kernel_weights carry their own complete mesh; the high-|G| "
            "tail completion is undefined for them"
        )
    q = _canonical_reciprocal_transfer(
        system, np.asarray(q_cart, dtype=float).reshape(3)
    )
    k = np.ascontiguousarray(np.asarray(k_ket, dtype=float).reshape(3))
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)

    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    # G-independent scale: folds into the pair weights for both pieces
    # (the C++ kernels produce the unscaled libint pair FT).
    W3_flat = (W3 * pair_scales[None, :, :]).reshape(n_aux, n_orb * n_orb)

    atom_of_aux = _basis_ao_atom_indices(aux_basis, system)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    chunk = max(int(tail_chunk_g), 1)

    def _accumulate(
        G_block: np.ndarray, w_block: Optional[np.ndarray] = None
    ) -> None:
        if w_block is None:
            G2_block = (G_block**2).sum(axis=1)
            # Same zero filter as the shared-q builder (Gq2 > 1e-12).
            nz = G2_block > 1e-12
            Gc = np.ascontiguousarray(G_block[nz])
            if Gc.shape[0] == 0:
                return
            coul_c = (4.0 * np.pi) / G2_block[nz] / V
        else:
            # External kernel weights: no zero filter — a finite zero
            # mode (slab K(0)) contributes through the pair-centre
            # piece; the aux-phase piece nulls it via its (G+q)_d
            # factor.
            Gc = np.ascontiguousarray(G_block)
            if Gc.shape[0] == 0:
                return
            coul_c = np.asarray(w_block, dtype=np.float64)
        F = rsgdf_aux_fourier_transform(aux_basis, Gc)
        rho_pair = ao_pair_fourier_transform_bloch(
            ao_basis, Gc, R_g, k_cart=k
        )  # (n_orb, n_orb, n_G), unscaled
        # ---- aux-phase piece --------------------------------------
        P = W3_flat @ rho_pair.reshape(n_orb * n_orb, Gc.shape[0])
        core = np.conj(F) * P * coul_c[None, :]  # (n_aux, n_G)
        for d in range(3):
            row = -(np.imag(core) * Gc[None, :, d]).sum(axis=1)
            np.add.at(grad[:, d], atom_of_aux, row)
        del rho_pair, P, core
        # ---- pair-FT-centre piece ---------------------------------
        # One G-resolved-weight kernel call for the whole aux batch
        # (same one-ZGEMM fold as the Γ kernel, conjugated for the
        # complex weight per the docstring derivation).
        rw_batch = F * coul_c[None, :]  # (n_aux, n_G)
        Q = np.ascontiguousarray(
            (np.conj(W3_flat).T @ rw_batch).reshape(
                n_orb, n_orb, Gc.shape[0]
            )
        )
        grad[:, :] += ao_pair_fourier_transform_bloch_gradient_gweighted(
            ao_basis, Gc, R_g, k, Q, n_atoms
        )

    if g_mesh is not None:
        mesh = np.asarray(g_mesh, dtype=float)
        w_full = np.asarray(kernel_weights, dtype=np.float64)
        if mesh.ndim != 2 or mesh.shape[1] != 3 or w_full.shape != (
            mesh.shape[0],
        ):
            raise ValueError(
                "_rsgdf_weighted_3c_tensor_gradient_bloch: g_mesh must be "
                "(n_G, 3) with kernel_weights of matching length"
            )
        for lo in range(0, mesh.shape[0], chunk):
            _accumulate(mesh[lo : lo + chunk], w_full[lo : lo + chunk])
        return grad

    Gq_base = _rsgdf_shifted_dense_g_mesh(system, q, float(ke_cutoff))
    for lo in range(0, Gq_base.shape[0], chunk):
        _accumulate(Gq_base[lo : lo + chunk])

    if tail_ke_cutoff is not None and float(tail_ke_cutoff) > float(ke_cutoff):
        # Shifted-sphere tail partition with the base-boundary
        # tolerance (zero at q = 0), mirroring the shared-q builder.
        Gq_tail_all = _rsgdf_shifted_dense_g_mesh(
            system, q, float(tail_ke_cutoff)
        )
        G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
        tail_norms = np.linalg.norm(Gq_tail_all, axis=1)
        base_boundary_tolerance = (
            0.0
            if float(np.linalg.norm(q)) < 1.0e-14
            else _rsgdf_shifted_sphere_boundary_tolerance(
                system, G_base_max, q
            )
        )
        Gq_tail = Gq_tail_all[
            tail_norms > G_base_max + base_boundary_tolerance
        ]
        for lo in range(0, Gq_tail.shape[0], chunk):
            _accumulate(Gq_tail[lo : lo + chunk])
    return grad


def _rsgdf_weighted_2c_metric_gradient_bloch(
    aux_basis: BasisSet,
    system: PeriodicSystem,
    *,
    ke_cutoff: float,
    q_cart: np.ndarray,
    weight: np.ndarray,
    tail_ke_cutoff: Optional[float] = None,
    tail_chunk_g: int = 65536,
    g_mesh: Optional[np.ndarray] = None,
    kernel_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Atomic gradient of a complex-weighted shared-q rsgdf 2c metric.

    Returns ``d/dR_A [ Re S_PQ weight[P, Q] . M(q)_PQ ]`` as an
    ``(n_atoms, 3)`` array, where ``M(q)`` is exactly the
    :func:`build_lpq_bloch_native_fft_shared_q` dense-mesh 2c metric
    accumulation

        M(q)_PQ = (4pi/V) S_{|G+q| != 0} conj(F̂_P(G+q)) F̂_Q(G+q)
                  / |G+q|²

    on the shifted sphere ``0 < |G+q| <= sqrt(2 ke_cutoff)`` plus the
    optional shifted tail completion. The Bloch/complex-weight sibling
    of :func:`_rsgdf_weighted_2c_metric_gradient` (G-PBC-002 Item-4
    rung 4); at ``q = 0`` with a real weight it reduces to that Γ
    kernel byte-for-byte (`_rsgdf_shifted_dense_g_mesh` returns the
    unshifted mesh at q = 0, and ``Im(W_real . C_d) = W_real .
    Im(C_d)`` elementwise, exactly).

    Derivative algebra. The only atom-position dependence is each aux
    shell's phase ``exp(-i(G+q).R_P)`` in ``F̂_P``
    (:func:`rsgdf_aux_fourier_transform`), so for P on atom A

        d conj(F̂_P)/dR_{A,d} = +i (G+q)_d conj(F̂_P),
        d F̂_Q/dR_{A,d}       = -i (G+q)_d F̂_Q,

    and with ``C_d[P, Q] = S_G coul(G+q) (G+q)_d conj(F̂_P) F̂_Q``

        d/dR_{A,d} Re S_PQ W_PQ M_PQ
            = Re[ +i S_{P on A, Q} W_PQ C_d[P, Q] ]
            + Re[ -i S_{P, Q on A} W_PQ C_d[P, Q] ]
            = - S_{P on A} row-sum( Im(W ∘ C_d) )[P]
              + S_{Q on A} col-sum( Im(W ∘ C_d) )[Q].

    For a HERMITIAN weight the result is exact for the SCF metric even
    though the SCF Hermitizes ``M -> (M + M^H)/2`` and real-projects at
    q = 0 before eigendecomposing: with Hermitian W,
    ``Re S W dM^H = conj(Re S W dM)`` term by term, so contracting the
    raw (unhermitized) accumulation under ``Re[...]`` equals the
    contraction with the Hermitized metric's derivative; at q = 0 the
    unshifted mesh is inversion symmetric and ``Im M(0)`` vanishes
    identically as a function of geometry (``F̂(-G) = conj(F̂(G))``),
    so the discarded ``Im(W) ∘ d Im(M)`` term is exactly zero in the
    algebra (rounding-level in float). Callers pass Hermitian weights;
    non-Hermitian weights differentiate the literal ``Re S W M(q)``
    objective, nothing else.

    ``g_mesh``/``kernel_weights`` (both or neither): externally
    supplied already-shifted physical mesh + per-point kernel weights
    (1/V included), as for
    :func:`_rsgdf_weighted_3c_tensor_gradient_bloch` — used by the
    slab-truncated metric derivative. The finite slab zero mode rides
    along harmlessly here: its ``(G+q)_d`` phase-derivative factor is
    identically zero, and single-centre aux functions carry NO other
    geometry dependence, so the 2c derivative receives no zero-mode
    term (exact, not a truncation). Defaults ``None`` keep the
    pre-parameter behaviour byte-identical.
    """
    n_aux = int(aux_basis.nbasis)
    n_atoms = len(system.unit_cell)
    W = np.asarray(weight, dtype=np.complex128)
    if W.shape != (n_aux, n_aux):
        raise ValueError(
            "_rsgdf_weighted_2c_metric_gradient_bloch: weight must have "
            f"shape (n_aux, n_aux) = ({n_aux}, {n_aux}); got {W.shape}."
        )
    if (g_mesh is None) != (kernel_weights is None):
        raise ValueError(
            "_rsgdf_weighted_2c_metric_gradient_bloch: g_mesh and "
            "kernel_weights must be supplied together"
        )
    if kernel_weights is not None and tail_ke_cutoff is not None:
        raise ValueError(
            "_rsgdf_weighted_2c_metric_gradient_bloch: external "
            "kernel_weights carry their own complete mesh; the high-|G| "
            "tail completion is undefined for them"
        )
    q = _canonical_reciprocal_transfer(
        system, np.asarray(q_cart, dtype=float).reshape(3)
    )
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    atom_of = _basis_ao_atom_indices(aux_basis, system)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    chunk = max(int(tail_chunk_g), 1)

    def _accumulate(
        G_block: np.ndarray, w_block: Optional[np.ndarray] = None
    ) -> None:
        if w_block is None:
            G2_block = (G_block**2).sum(axis=1)
            # Same zero filter as the shared-q builder (Gq2 > 1e-12).
            nz = G2_block > 1e-12
            Gc = np.ascontiguousarray(G_block[nz])
            if Gc.shape[0] == 0:
                return
            coul_c = (4.0 * np.pi) / G2_block[nz] / V
        else:
            # External kernel weights: no zero filter (see docstring —
            # the zero mode contributes exactly nothing here anyway).
            Gc = np.ascontiguousarray(G_block)
            if Gc.shape[0] == 0:
                return
            coul_c = np.asarray(w_block, dtype=np.float64)
        F = rsgdf_aux_fourier_transform(aux_basis, Gc)
        for d in range(3):
            C_d = (np.conj(F) * (coul_c * Gc[:, d])[None, :]) @ F.T
            X = np.imag(W * C_d)
            np.subtract.at(grad[:, d], atom_of, X.sum(axis=1))
            np.add.at(grad[:, d], atom_of, X.sum(axis=0))

    if g_mesh is not None:
        mesh = np.asarray(g_mesh, dtype=float)
        w_full = np.asarray(kernel_weights, dtype=np.float64)
        if mesh.ndim != 2 or mesh.shape[1] != 3 or w_full.shape != (
            mesh.shape[0],
        ):
            raise ValueError(
                "_rsgdf_weighted_2c_metric_gradient_bloch: g_mesh must be "
                "(n_G, 3) with kernel_weights of matching length"
            )
        for lo in range(0, mesh.shape[0], chunk):
            _accumulate(mesh[lo : lo + chunk], w_full[lo : lo + chunk])
        return grad

    Gq_base = _rsgdf_shifted_dense_g_mesh(system, q, float(ke_cutoff))
    for lo in range(0, Gq_base.shape[0], chunk):
        _accumulate(Gq_base[lo : lo + chunk])

    if tail_ke_cutoff is not None and float(tail_ke_cutoff) > float(ke_cutoff):
        # Shifted-sphere tail partition with the base-boundary
        # tolerance (zero at q = 0), mirroring the shared-q builder.
        Gq_tail_all = _rsgdf_shifted_dense_g_mesh(
            system, q, float(tail_ke_cutoff)
        )
        G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
        tail_norms = np.linalg.norm(Gq_tail_all, axis=1)
        base_boundary_tolerance = (
            0.0
            if float(np.linalg.norm(q)) < 1.0e-14
            else _rsgdf_shifted_sphere_boundary_tolerance(
                system, G_base_max, q
            )
        )
        Gq_tail = Gq_tail_all[
            tail_norms > G_base_max + base_boundary_tolerance
        ]
        for lo in range(0, Gq_tail.shape[0], chunk):
            _accumulate(Gq_tail[lo : lo + chunk])
    return grad


def build_lpq_native_fft(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    *,
    ke_cutoff: float = 200.0,
    tail_ke_cutoff: Optional[float] = None,
    tail_chunk_g: int = 65536,
    tail_pair_ft_screen: float = 0.0,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-10,
    pair_ft_shared: "Optional[_RsgdfDensePairFT]" = None,
    fit_screen_threshold: float = 0.0,
    fit_pair_list: Optional[np.ndarray] = None,
    progress: Any = None,
) -> np.ndarray:
    """All-FT Γ-only GDF Lpq builder -- dense FFT mesh + Bloch pair-FT.

    Compared to :func:`build_lpq_native` (which uses the bare lattice
    sum or the SR+LR split with a sparse w-sized G-mesh), this routine
    computes the full periodic Coulomb 2c metric and 3c tensor directly
    in reciprocal space on a dense FFT mesh sized by ``ke_cutoff``:

    ::

        M_PQ  = (4pi/V) . S_{G!=0} F̂_P*(G) . F̂_Q(G) / G^2
        T_{P,muν} = (4pi/V) . S_{G!=0} F̂_P*(G) . r̂^per_muν(G) / G^2

    where ``F̂_P`` is :func:`rsgdf_aux_fourier_transform` and
    ``r̂^per_muν`` is the **Bloch-summed periodic pair-FT** (i.e.,
    ``S_R ∫chi_mu(r).chi_ν(r-R).exp(-iG.r) dr`` at k=0) from
    :func:`vibeqc._aopair_ft.ao_pair_fourier_transform_bloch`. The
    G = 0 contribution is omitted (standard Ewald / neutralising-
    background convention).

    The ``aux_basis`` should be passed as the **modrho-rescaled basis**
    from :func:`make_modrho_aux_basis` -- coefficients with the per-L
    ``libcint_to_libint`` factor baked in. The driver
    :func:`vibeqc.run_pbc_gdf_rhf` handles this; direct callers should
    pre-build modrho.

    Why this exists
    ---------------
    The classic SR+LR-split route (:func:`build_lpq_native` with
    ``algorithm='rsgdf'``) has a fundamental architectural limit on
    tight periodic cells with diffuse orbitals: the SR via libint
    sums one orbital over lattice images, but the LR via FT uses the
    *molecular* pair-FT (single-cell pair density). For H₂ in a 12-bohr
    vacuum box the difference is exponentially small and sub-µHa SCF
    parity to PySCF holds. For LiH primitive FCC (Li-H separation
    3.86 bohr in 7.72-bohr lattice; Li 2s diffuse), the Bloch overlap
    is ~10x the molecular overlap and the SR+LR sum doesn't equal the
    proper periodic Coulomb integral.

    The all-FT path with Bloch pair-FT (this routine) handles both
    regimes uniformly:

    * H₂ vacuum-box / STO-3G / def2-svp-jk: <= 0.05 µHa vs PySCF at
      ``ke_cutoff=200``.
    * LiH primitive FCC / STO-3G / def2-svp-jk: 18 mHa vs PySCF (10⁴x
      tighter than the previous best compcell+AFT at +11 Ha).

    The remaining LiH residue (18 mHa) is in J/K-builder territory,
    not in the 2c/3c integrals themselves -- element-wise (muν|ls)
    parity vs PySCF is at rel ≈ 5.8e-5 (block-by-block <= 1e-3).

    Parameters
    ----------
    system
        Periodic system (3D-periodic; dim < 3 not yet wired).
    ao_basis
        Orbital basis (vibe-qc BasisSet on the unit-cell molecule).
    aux_basis
        Auxiliary basis. **Must be the modrho-rescaled basis** from
        :func:`make_modrho_aux_basis`. The raw aux + apply_modrho
        post-multiply path of :func:`build_lpq_native` is NOT used here.
    ke_cutoff
        Kinetic-energy cutoff (Hartree) for the dense FFT mesh sizing.
        Default ``200`` -- mesh-converged for sub-mHa SCF on
        def2-svp-jkfit-class aux. Increase for tighter cells or more
        diffuse aux.
    lat_opts
        Lattice-sum options. Defaults to ``LatticeSumOptions()``
        (15 bohr cutoff). Used for the Bloch-pair-FT lattice-cell list
        on the 3c side.
    linear_dep_thr
        Absolute eigenvalue threshold for the 2c-metric eigendecomp
        truncation (default ``1e-10``, matching PySCF's ``LINEAR_DEP_THR``).
        Modes with ``l_k <= thr`` are dropped from the auxiliary subspace.
    pair_ft_shared
        Optional precomputed dense-mesh AO-pair FT bundle
        (:class:`_RsgdfDensePairFT`) from :func:`_rsgdf_dense_pair_ft`.
        When given, the (dominant) Bloch AO-pair FT is reused instead of
        recomputed -- the GDF driver passes the same bundle here and to
        the EWALD_3D V_ne FT so it runs once per SCF, not twice. ``None``
        (default) builds it internally; a provenance mismatch (ke_cutoff
        / cutoff_bohr / n_ao) raises rather than using a wrong tensor.
    tail_pair_ft_screen
        Optional native AO-pair FT shell/cell screen used only for the
        high-|G| tail completion. ``0.0`` keeps the exact historical
        base+tail == big-mesh construction. Positive values skip
        exponentially dead shell/cell blocks in the tail chunks; the PBC
        GDF driver uses a very small tolerance to keep the dense-core
        tail affordable without changing the base tensor.
    fit_screen_threshold, fit_pair_list
        Cauchy-Schwarz fit screen / shell-pair restriction, exactly as
        on :func:`build_lpq_bloch_native_fft` (M1 of
        ``handovers/HANDOVER_GDF_FIT_SCREENING.md``). When either is
        requested the build **delegates to the** ``k_bra = k_ket = 0``
        **Bloch builder** -- the general builder whose Γ block this
        routine is the special case of -- and returns its real part
        (exact at true Γ; the production Γ-supercell cderi
        :func:`vibeqc.periodic.ccm.neutral.ccm_neutral_cderi` uses that
        route already). The delegation brings the M2 ``tail_chunk_g``
        G-chunked base sweep with it, so the screened path never
        materialises the ``(n_ao, n_ao, n_G)`` pair FT. Because that is
        a memory-lean mode, it conflicts with ``pair_ft_shared`` (whose
        dense bundle IS that materialisation): passing both raises --
        the driver must choose. ``tail_pair_ft_screen > 0`` likewise
        raises (the Bloch builder does not implement the native tail
        pair screen; the Schwarz screen covers the pair-dropping role).
    progress
        Optional progress logger with an ``info`` method. Tail chunk
        counters are emitted here so long promotion runs show pre-SCF
        progress.

    Returns
    -------
    Lpq : (n_kept, n_orb, n_orb) float64
        Real-valued (Γ-only). ``n_kept <= aux_basis.nbasis`` after
        threshold truncation. AO indices in libint m-ordering.

    References
    ----------
    PySCF ``pyscf/pbc/df/rsdf_builder.py:_RSGDFBuilder.get_2c2e`` for
    the dense-FFT-mesh approach (read for understanding; no code
    copied, per CLAUDE.md Sec.10).
    """
    if lat_opts is None:
        lat_opts = LatticeSumOptions()

    # Screened / pair-restricted builds delegate to the (k_bra, k_ket)
    # builder at Γ -- ONE screened build path (CLAUDE.md Sec. 9 + the
    # symmetry-pair-reduction seam), and the M2 G-chunked base sweep
    # comes with it so the (n_ao, n_ao, n_G) pair FT is never
    # materialised. Real part is exact at true Γ (the (0,0) block is
    # real; see the q == 0 note in build_lpq_bloch_native_fft).
    if float(fit_screen_threshold) > 0.0 or fit_pair_list is not None:
        if pair_ft_shared is not None:
            raise ValueError(
                "build_lpq_native_fft: fit_screen_threshold / "
                "fit_pair_list is a memory-lean chunked mode and cannot "
                "consume a precomputed dense pair_ft_shared bundle (the "
                "bundle IS the (n_ao, n_ao, n_G) materialisation the "
                "screen avoids). Pass pair_ft_shared=None -- the driver "
                "must choose one or the other."
            )
        if float(tail_pair_ft_screen) > 0.0:
            raise NotImplementedError(
                "build_lpq_native_fft: tail_pair_ft_screen is not "
                "implemented on the screened (Bloch-delegated) path; "
                "the Schwarz fit screen already drops negligible pairs "
                "in both the base and tail sweeps."
            )
        k0 = np.zeros(3)
        Lpq = build_lpq_bloch_native_fft(
            system,
            ao_basis,
            aux_basis,
            k0,
            k0,
            ke_cutoff=float(ke_cutoff),
            tail_ke_cutoff=(
                float(tail_ke_cutoff) if tail_ke_cutoff is not None else None
            ),
            tail_chunk_g=int(tail_chunk_g),
            lat_opts=lat_opts,
            linear_dep_thr=float(linear_dep_thr),
            fit_screen_threshold=float(fit_screen_threshold),
            fit_pair_list=fit_pair_list,
            progress=progress,
        )
        return np.real(np.asarray(Lpq))

    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))

    # Dense-mesh AO-pair FT r̂_muν(G) + the (G, |G|^2) it lives on. Shared
    # with the EWALD_3D V_ne FT when the driver precomputes it (the
    # dominant C++ cost, otherwise run twice per SCF) -- see
    # :class:`_RsgdfDensePairFT`.
    _ft = _resolve_pair_ft_shared(
        pair_ft_shared, ao_basis, system, ke_cutoff, lat_opts
    )
    G, pair_ft = _ft.G, _ft.pair_ft
    coul = (4.0 * np.pi) / _ft.G2 / V

    aux_ft = rsgdf_aux_fourier_transform(aux_basis, G)

    # 2c metric on dense FFT mesh.
    M = np.real(aux_ft.conj() @ (aux_ft * coul[None, :]).T)

    # 3c tensor on dense FFT mesh.
    aux_w = aux_ft.conj() * coul[None, :]
    T = np.real(np.einsum("Pk,mnk->Pmn", aux_w, pair_ft))

    # High-|G| tail completion for dense-core cells (prompt-11 / P01).
    # Same rationale + construction as the (k_bra, k_ket) builder
    # build_lpq_bloch_native_fft: extend BOTH the metric and the 3c
    # G-sums over the exact complementary reciprocal shell
    # sqrt(2 ke_cutoff) < |G| <= sqrt(2 tail_ke_cutoff), chunked, with
    # the same analytic kernels -- staying inside the FT representation
    # so DF internal consistency is preserved (no SR/real-space
    # representation mixing, which amplifies near-null metric modes).
    if tail_ke_cutoff is not None and float(tail_ke_cutoff) > float(ke_cutoff):
        from ._aopair_ft import ao_pair_fourier_transform_bloch
        from ._vibeqc_core import direct_lattice_cells

        G_tail_all = rsgdf_dense_g_mesh(system, float(tail_ke_cutoff))
        G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
        tail_norms = np.linalg.norm(G_tail_all, axis=1)
        G_tail = G_tail_all[tail_norms > G_base_max]
        cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
        R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
        if R_g.size == 0:
            R_g = np.zeros((1, 3), dtype=float)
        ao_scales = _ao_scales_for_rsgdf(ao_basis)
        pair_scales = np.outer(ao_scales, ao_scales)
        chunk = max(int(tail_chunk_g), 1)
        n_tail = int(G_tail.shape[0])
        n_chunks = int((n_tail + chunk - 1) // chunk) if n_tail else 0
        _progress_info(
            progress,
            "RSGDF high-|G| tail: "
            f"{n_tail} G-points, {len(cells)} lattice cells, "
            f"chunk={chunk}, pair_screen={float(tail_pair_ft_screen):.1e}",
        )
        for lo in range(0, G_tail.shape[0], chunk):
            Gc = G_tail[lo : lo + chunk]
            chunk_idx = lo // chunk + 1
            _progress_info(
                progress,
                f"RSGDF high-|G| tail chunk {chunk_idx}/{n_chunks}: "
                f"G {lo + 1}-{lo + Gc.shape[0]} of {n_tail}",
            )
            G2c = (Gc**2).sum(axis=1)
            coul_c = (4.0 * np.pi) / G2c / V
            aux_ft_c = rsgdf_aux_fourier_transform(aux_basis, Gc)
            pair_ft_c = ao_pair_fourier_transform_bloch(
                ao_basis,
                Gc,
                R_g,
                k_cart=np.zeros(3),
                screen_tol=float(tail_pair_ft_screen),
            )
            pair_ft_c *= pair_scales[:, :, None]
            aux_w_c = aux_ft_c.conj() * coul_c[None, :]
            M += np.real(aux_w_c @ aux_ft_c.T)
            T += np.real(np.einsum("Pk,mnk->Pmn", aux_w_c, pair_ft_c))

    M = 0.5 * (M + M.T)
    T = 0.5 * (T + T.transpose(0, 2, 1))

    # Eigendecompose-and-threshold (Sun 2017 Sec.III.B protocol).
    # Absolute threshold (matches PySCF ``LINEAR_DEP_THR``). For this
    # FT path the 2c metric has slightly-negative numerical-noise
    # eigenvalues on tight cells (~ -1e-12); the threshold filter
    # drops them along with any other near-singular modes.
    eigvals, U = np.linalg.eigh(M)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            f"build_lpq_native_fft: 2c metric has no positive eigenvalue "
            f"(max eig = {max_eig:.3e}); aux or mesh invalid."
        )
    keep = _metric_keep_mask(eigvals, linear_dep_thr)
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"build_lpq_native_fft: no eigenvalues above threshold "
            f"({linear_dep_thr:.1e}); aux basis or threshold invalid."
        )
    T_flat = T.reshape(T.shape[0], -1)
    Lpq = (U[:, keep].T @ T_flat) / np.sqrt(eigvals[keep])[:, None]
    return Lpq.reshape(n_kept, T.shape[1], T.shape[2])


class _MdfCderi(NamedTuple):
    """MDF cderi -- Gaussian-fit L vectors + plane-wave residual (Sun 2017).

    Mixed Density Fitting factorises the Γ periodic ERI (Eq 22 of Sun,
    Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
    doi:10.1063/1.4998644; real AO pairs so r(-G)=r(G)*) as::

        W_muν,κl = S_i L_gauss[i,muν].L_gauss[i,κl]      <- Gaussian (real)
                + S_G cderi_pw[G,muν].conj(cderi_pw[G,κl])  <- PW residual

    with ``cderi_pw[G] = √coul(G).r_muν(G)`` (``coul(G)=4pi/(Ω|G|^2)``), so the
    PW residual is a Hermitian outer product over the *modest* residual
    mesh -- the part the compensated Gaussians cannot represent. The
    Gaussian part is the compcell 3c tensor with the PW projection
    subtracted and orthogonalised on the PW-dressed metric (Eq 19-23);
    pure compcell/rsgdf = the Gaussian part with no PW residual *and* the
    bare metric (its all-electron accuracy floor).
    """

    L_gauss: np.ndarray  # (n_kept, n_orb, n_orb) float64
    cderi_pw: np.ndarray  # (n_pw, n_orb, n_orb) complex128
    n_kept_gauss: int
    n_pw: int


def build_lpq_mdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    *,
    molecule: Optional[Molecule] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    compcell_eta: float = 1.0,
    mdf_ke_cutoff: float = 40.0,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    precision_j2c: Optional[float] = None,
    precision_j3c: Optional[float] = None,
) -> "_MdfCderi":
    """Mixed Density Fitting cderi -- Gaussian fit + plane-wave residual.

    Implements the MDF decomposition of Sun, Berkelbach, McClain & Chan,
    *J. Chem. Phys.* **147**, 164119 (2017), doi:10.1063/1.4998644 (Sec.II
    A-C), specialised to Γ. The all-electron accuracy floor of pure
    Gaussian DF (compcell / rsgdf) comes from the minimal auxiliary not
    being steep enough to fit core pair densities; MDF handles the steep
    part in **real space** (exact libint Gaussian 2c/3c on the compensated
    "fused" basis -- no mesh) and only the **smooth residual** via plane
    waves on a *modest* ``mdf_ke_cutoff`` mesh.

    The pieces (design: ``docs/design_mdf.md``):

    * Gaussian metric ``M_PQ = (phi_P|phi_Q)`` and 3c ``T_{P,muν} = (phi_P|muν)``
      on the compensated fused basis ``phi = chi - ξ`` -- identical to
      :func:`build_lpq_compcell` (real-space libint, ``rcut_strategy``
      auto-tuning), **without** its AFT long-range correction (MDF
      replaces that with the explicit PW residual).
    * PW-dressed 2c metric (Eq 20)
      ``J̃_PQ = M_PQ - S_{G!=0} coul(G) r_P(-G) r_Q(G)`` and PW projection
      of the 3c ``S_{G!=0} coul(G) r_P(-G) r_muν(G)`` (Eq 23 bracket),
      using the FT of the *compensated* aux (``rsgdf_aux_fourier_transform``
      on the fused basis) and the dense-mesh AO-pair FT.
    * Orthogonalise on ``J̃`` (Eq 19-21): diagonalise, drop eigenvalues
      ``<= linear_dep_thr`` (absolute; this removes the Gaussian<->PW
      linear dependence), ``t = U_keep/√l_keep``;
      ``L = tᵀ.(T - PW-projection)``.
    * PW residual cderi ``cderi_pw[G] = √coul(G).r_muν(G)`` (Eq 22 second
      term).

    **No-PW limit == compcell.** With ``mdf_ke_cutoff <= 0`` the PW mesh
    is empty: ``J̃ = M``, ``T`` is uncorrected, and ``L_gauss = M^{-1/2}.T``
    -- bit-identical to ``build_lpq_compcell(apply_aft_correction=False)``
    with the same ``compcell_eta`` / ``rcut_strategy`` / ``linear_dep_thr``.
    This is the increment-1 sanity gate (see
    ``tests/test_pbc_gdf_mdf.py``).

    Parameters
    ----------
    system, ao_basis, aux_basis
        Periodic system, orbital basis, **raw** auxiliary basis (modrho
        rescaling is applied internally, as in
        :func:`build_lpq_compcell`).
    molecule
        Unit-cell molecule; defaults to ``system.unit_cell_molecule()``.
    lat_opts
        Lattice-sum options for the real-space 2c/3c (and the base for the
        ``rcut_strategy`` auto-tune). Defaults to ``LatticeSumOptions()``.
    linear_dep_thr
        Absolute eigenvalue threshold for the PW-dressed-metric
        eigendecomposition (Sun 2017 Sec. II B). Because ``J̃`` (Eq. 20)
        is deliberately singular, a value below
        :data:`_MDF_DRESSED_METRIC_MIN_EIG_FRACTION` x ``max_eig`` is raised
        to it whenever
        the PW block is active -- see
        :func:`_mdf_dressed_metric_threshold`. Callers may set a *looser*
        value; the dense-core class needs ~3e-2.
    compcell_eta
        Compensating-charge exponent for the fused basis (default ``1.0``).
    mdf_ke_cutoff
        Kinetic-energy cutoff (Hartree) for the **residual** PW mesh. The
        MDF residual is smooth, so this is *modest* (default ``40``) -- not
        the dense ke≈200+ a pure-FFT GDF (:func:`build_lpq_native_fft`)
        needs. ``<= 0`` disables the PW part (the compcell-equivalent
        sanity limit).
    rcut_strategy, rcut_precision
        Real-space lattice-cutoff auto-tuning for the libint 3c, as in
        :func:`build_lpq_compcell` (default ``"pyscf_auto"``). Also sizes
        the 2c metric in the no-PW (compcell-equivalent) limit.
    precision_j2c
        Accuracy the PW-dressed metric ``J-tilde`` is built to -- vibe-qc's
        analogue of PySCF's ``precision_j2c``; default
        :data:`_MDF_PRECISION_J2C`. Whenever the PW block is active the 2c
        metric gets its own, much tighter lattice sum (see
        :func:`_mdf_fit_lattice_opts`), because Eq. 19's
        ``1/sqrt(lambda)`` makes the metric's own construction error -- not
        ``rcut_precision`` -- what decides how small a mode may honestly be
        kept. Ignored with the PW block off, where ``J-tilde`` IS the plain
        metric and the compcell-equivalent limit must stay bit-identical.

    Returns
    -------
    :class:`_MdfCderi`
        ``L_gauss`` (real, ``(n_kept, n_orb, n_orb)``) + ``cderi_pw``
        (complex, ``(n_pw, n_orb, n_orb)``); ``W`` reconstructs as
        ``S_i L.L + S_G cderi_pw.conj(cderi_pw)``. AO indices in libint
        m-ordering.

    Notes
    -----
    The G=0 ``V̄_P.r̄_muν`` term of Eq 23 IS applied on the PW-on path. It is
    an **Ewald-regularised G=0 self-term**, NOT the naïve continuous Fourier
    limit ``(4pi/Ω).lim r_P(G)/|G|^2`` (which underestimates it by ~10⁴). The
    closed form is the second-moment term ``V̄_P = -pi^{5/2}.(A @ m_fused)``
    with ``m_fused[f] = S_k c_{f,k} a_{f,k}^{-5/2}`` over s-type fused
    functions (higher-L aux carry no monopole), and ``r̄_muν = S_muν/Ω``. This
    was derived + verified during increment 2 (``docs/design_mdf.md`` Sec.4):
    the real-space libint 3c ``T`` and the reciprocal PW projection differ
    by exactly this ke-independent G=0 term (measured ``4.87e-3`` on
    H2/def2-svp-jk); the closed form matches the exact ``T_real -
    T_recip(dense)`` gap to ``1e-5`` and the constant ``-pi^{5/2}`` to
    machine precision. With it the reconstructed ``W`` matches the rsgdf
    reference to ``~1e-4`` and is mesh-stable. The higher-multipole G=0
    contribution (L>0 aux x AO higher moments) is ~1% of the monopole term
    and is currently neglected -- adequate for the modest-mesh residual but
    a knob for sub-µHa all-electron targets.
    """
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    if molecule is None:
        molecule = system.unit_cell_molecule()
    pw_on = mdf_ke_cutoff is not None and float(mdf_ke_cutoff) > 0.0
    if precision_j2c is None:
        precision_j2c = _MDF_PRECISION_J2C
    if precision_j3c is None:
        precision_j3c = _MDF_PRECISION_J3C

    # --- Gaussian part: compensated fused basis + real-space libint 2c/3c.
    #     Identical to build_lpq_compcell (sans AFT correction). ---
    modrho_aux = make_modrho_aux_basis(aux_basis, molecule)
    chg = make_compensating_basis(modrho_aux, molecule, eta=compcell_eta)
    fused = make_fused_basis(modrho_aux, chg, molecule)
    A = fuse_transform_matrix(modrho_aux, chg)

    joint_basis = _mdf_joint_basis(ao_basis, fused, molecule)

    if rcut_strategy is not None:
        from .lattice_screening import RcutStrategy, make_lattice_opts

        if not isinstance(rcut_strategy, RcutStrategy):
            try:
                rcut_strategy = RcutStrategy(rcut_strategy)
            except Exception as exc:
                raise ValueError(
                    f"build_lpq_mdf: rcut_strategy must be a RcutStrategy "
                    f"or its string value; got {rcut_strategy!r}: {exc}"
                )
        lat_opts_2c = make_lattice_opts(
            fused, strategy=rcut_strategy, base_opts=lat_opts, precision=rcut_precision
        )
        lat_opts_3c = make_lattice_opts(
            joint_basis,
            strategy=rcut_strategy,
            base_opts=lat_opts,
            precision=rcut_precision,
        )
    else:
        lat_opts_2c = lat_opts
        lat_opts_3c = lat_opts

    if pw_on:
        # J-tilde is built to its own accuracy, not the cell's (Eq. 20 is
        # deliberately singular and Eq. 19 divides by sqrt of its modes).
        lat_opts_2c = _mdf_fit_lattice_opts(
            fused, system, lat_opts_2c, precision_j2c, modrho_aux.nbasis
        )
        if precision_j3c is not None:
            lat_opts_3c = _mdf_fit_lattice_opts(
                joint_basis, system, lat_opts_3c, precision_j3c,
                modrho_aux.nbasis,
            )

    M_fused = np.asarray(compute_2c_eri_lattice(fused, system, lat_opts_2c))
    M_fused = 0.5 * (M_fused + M_fused.T)
    T_fused = np.asarray(compute_3c_eri_lattice(ao_basis, fused, system, lat_opts_3c))

    M = A @ M_fused @ A.T
    M = 0.5 * (M + M.T)
    T = np.einsum("iP,Pmn->imn", A, T_fused, optimize=True)

    n_aux = modrho_aux.nbasis
    n_orb = ao_basis.nbasis
    if M.shape != (n_aux, n_aux) or T.shape != (n_aux, n_orb, n_orb):
        raise RuntimeError(
            f"build_lpq_mdf: post-compensation shapes mismatch -- M={M.shape}, "
            f"T={T.shape}, expected ({n_aux},{n_aux}) / ({n_aux},{n_orb},{n_orb})."
        )

    # --- PW residual: modest dense mesh; compensated-aux FT + AO-pair FT. ---
    if pw_on:
        V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
        _ft = _resolve_pair_ft_shared(
            None, ao_basis, system, float(mdf_ke_cutoff), lat_opts_3c
        )
        G, pair_ft = _ft.G, _ft.pair_ft  # pair_ft: (n_orb, n_orb, n_G)
        coul = (4.0 * np.pi) / _ft.G2 / V  # (n_G,)

        # FT of the *compensated* aux phi_P = chi_P - ξ_P (Eq 5): apply the
        # fuse transform A to the fused-basis FT.
        aux_ft_fused = rsgdf_aux_fourier_transform(fused, G)  # (n_fused, n_G)
        rho_aux = A @ aux_ft_fused  # (n_aux, n_G)

        # PW-dressed 2c metric (Eq 20): J̃ = M - S_G coul r_P(-G) r_Q(G).
        # Γ + real fitting fns => r_P(-G) = conj(r_P(G)); the ±G sum is real.
        aux_w = rho_aux.conj() * coul[None, :]  # (n_aux, n_G)
        pw_2c = np.real(aux_w @ rho_aux.T)  # (n_aux, n_aux)
        pw_2c = 0.5 * (pw_2c + pw_2c.T)
        J_tilde = M - pw_2c

        # PW projection of the 3c (Eq 23 bracket): S_G coul r_P(-G) r_muν(G).
        pw_3c = np.real(np.einsum("Pk,mnk->Pmn", aux_w, pair_ft, optimize=True))
        pw_3c = 0.5 * (pw_3c + pw_3c.transpose(0, 2, 1))

        # G=0 self-term V̄_P.r̄_muν (Eq 23). The real-space libint 3c ``T``
        # carries a G=0 monopole that BOTH the G!=0 PW projection and the
        # G=0-free metric J̃ = M - PW2c exclude; subtracting it makes the
        # Gaussian fit consistent with its own (G=0-free) metric. Omitting
        # it makes the L vectors blow up as the mesh tightens (the
        # ke-independent 4.87e-3 T_real-T_recip gap; docs/design_mdf.md Sec.4).
        #
        # Closed form -- the Ewald-regularised G=0 second-moment term, NOT
        # the naïve Fourier limit (4pi/Ω)lim r_P(G)/|G|^2 (off by ~10⁴):
        #   V̄r̄_muν = -(2pi/3Ω).M2_P.S_muν ,  M2_P = (3/2)pi^{3/2} A@(S_k c_k a_k^{-5/2})
        # over s-type fused functions (higher-L aux carry no monopole),
        # giving the per-aux  V̄_P = -pi^{5/2}.(A @ m_fused)  and r̄_muν = S_muν/Ω.
        # Verified: matches the exact T_real-T_recip(dense) gap to 1e-5 and
        # the constant -pi^{5/2} to machine precision across all aux.
        m_fused = np.zeros(fused.nbasis)
        _off = 0
        for sh in fused.shells():
            nb = (2 * sh.l + 1) if sh.pure else ((sh.l + 1) * (sh.l + 2) // 2)
            if sh.l == 0:
                a = np.asarray(sh.exponents, dtype=float)
                c = np.asarray(sh.coefficients, dtype=float)
                m_fused[_off : _off + nb] = float(np.sum(c * a ** (-2.5)))
            _off += nb
        vbar = -(np.pi**2.5) * (A @ m_fused)  # (n_aux,)
        from ._vibeqc_core import bloch_sum as _bloch_sum

        S_ao = np.real(
            _bloch_sum(
                compute_overlap_lattice(ao_basis, system, lat_opts), np.zeros(3)
            )
        )
        S_ao = 0.5 * (S_ao + S_ao.T)
        vbar_rho = vbar[:, None, None] * (S_ao / V)[None, :, :]

        T_corr = T - vbar_rho - pw_3c

        # PW residual cderi: √coul(G).r_muν(G) -> (n_pw, n_orb, n_orb).
        sq = np.sqrt(coul)  # (n_G,)
        cderi_pw = (pair_ft * sq[None, None, :]).transpose(2, 0, 1).copy()
        n_pw = int(G.shape[0])
    else:
        J_tilde = M
        T_corr = T
        cderi_pw = np.zeros((0, n_orb, n_orb), dtype=np.complex128)
        n_pw = 0

    # --- Orthogonalise on the (PW-dressed) metric + assemble L (Eq 19/23). ---
    eigvals, U = np.linalg.eigh(J_tilde)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            f"build_lpq_mdf: PW-dressed metric has no positive eigenvalue "
            f"(max eig = {max_eig:.3e}); aux/mesh/eta invalid."
        )
    thr = _mdf_dressed_metric_threshold(
        linear_dep_thr, n_pw, max_eig, "build_lpq_mdf"
    )
    keep = _metric_keep_mask(eigvals, thr)
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"build_lpq_mdf: no eigenvalues above threshold ({thr:.3e}; "
            f"max eig = {max_eig:.3e})."
        )
    T_flat = T_corr.reshape(n_aux, n_orb * n_orb)
    L_flat = (U[:, keep].T @ T_flat) / np.sqrt(eigvals[keep])[:, None]
    L_gauss = L_flat.reshape(n_kept, n_orb, n_orb)

    return _MdfCderi(
        L_gauss=L_gauss, cderi_pw=cderi_pw, n_kept_gauss=n_kept, n_pw=n_pw
    )


def _fit_screen_pair_norm_bounds(
    ao_basis: BasisSet,
    R_g: np.ndarray,
    Gq2: np.ndarray,
    coul: np.ndarray,
) -> np.ndarray:
    """Per-shell-pair screening estimates for the Coulomb self-norm of
    Bloch AO-pair densities on the fit mesh -- the pair side of the
    Cauchy-Schwarz fit screen of :func:`build_lpq_bloch_native_fft`.

    For the coul-weighted mesh inner product ``<f, g> = S_G coul(G)
    f(G)* g(G)`` (``coul(G) = 4pi/(V|G+q|^2) > 0``), the three-centre fit
    element is ``T_{P,muν} = <F̂_P, r̂_muν>``, so Cauchy-Schwarz gives the
    standard direct-SCF / RI screening bound (Häser & Ahlrichs, J. Comput.
    Chem. 10, 104 (1989); Neese, Wennmohs, Hansen & Becker, Chem. Phys.
    356, 98 (2009), Sec. 3.1 -- the RI three-index analogue; Ochsenfeld,
    White & Head-Gordon, J. Chem. Phys. 109, 1663 (1998) for the
    linear-scaling pair-list philosophy):

        |T_{P,muν}| <= sqrt((P|P)) . sqrt((muν|muν))
                    =  sqrt(M_PP)  . sqrt(S_G coul(G) |r̂_muν(G)|^2)

    This routine bounds the pair factor ``sqrt(S_G coul |r̂|^2)`` WITHOUT
    computing the pair FT. Pointwise, by the triangle inequality over
    lattice cells R and primitive pairs (p, q) (Bloch phases have unit
    modulus, so the bound holds for every k_ket),

        |r̂_muν(G)| <= B_sp(G²) = scale_M scale_N S_{pq} w̃_pq
                                  (pi/g_pq)^{3/2} e^{-G²/(4 g_pq)}
        w̃_pq = S_R poly_R |c_p c_q| e^{-a_p b_q/g_pq |A_M - B_N - R|²}

    with ``g = a + b`` (Gaussian-product theorem; the same radial factor
    the C++ ``shell_pair_cell_ft_bound`` uses) and ``poly_R = ((1 +
    max|G+q| + |A_M - B_N - R|)^{l_M+l_N}) . n_cart_M . n_cart_N`` the
    deliberately conservative polynomial estimate of the
    McMurchie-Davidson / cart->sph mixing for L > 0. The expression is
    an exact upper bound for s-pairs; for higher angular momentum it is
    intended to overestimate but is not advertised as a formal proof.
    The Coulomb-norm estimate then expands over the primitive-pair cross
    terms:

        S_G coul B_sp² = S_{pq,p'q'} w̃_pq w̃_p'q' C(g_pq, g_p'q'),
        C(g, g') = S_G coul(G) (pi/g)^{3/2} (pi/g')^{3/2}
                   e^{-G²/4 (1/g + 1/g')}

    and ``C`` is precomputed once per unique exponent pair, so the whole
    bound costs ~n_shell_pairs.n_prim_pairs² -- negligible next to one
    pair-FT build.

    Returns ``(n_shells, n_shells)`` float64: ``bounds[sM, sN]`` is the
    screening estimate for ``sqrt(S_G coul |r̂_muν(G)|^2)`` for every AO
    pair of that shell pair.
    """
    shells = ao_basis.shells()
    n_sh = len(shells)
    R_arr = np.asarray(R_g, dtype=float).reshape(-1, 3)
    max_g_norm = float(np.sqrt(np.max(Gq2))) if Gq2.size else 0.0

    # Per-shell primitive data + the per-L pair-FT calibration scale
    # (matches _ao_scales_for_rsgdf).
    sh_origin = [np.asarray(sh.origin, dtype=float) for sh in shells]
    sh_exps = [np.asarray(sh.exponents, dtype=float) for sh in shells]
    sh_coefs = [np.abs(np.asarray(sh.coefficients, dtype=float)) for sh in shells]
    sh_l = [int(sh.l) for sh in shells]
    sh_scale = [
        float(np.sqrt(4.0 * np.pi / (2 * l + 1))) if l > 0 else 1.0 for l in sh_l
    ]

    # C(g, g') table over unique Gaussian-product exponents g = a_p + b_q.
    gammas: dict[float, int] = {}
    for sM in range(n_sh):
        for sN in range(n_sh):
            for g in np.add.outer(sh_exps[sM], sh_exps[sN]).ravel():
                gammas.setdefault(float(g), len(gammas))
    g_vals = np.array(sorted(gammas), dtype=float)
    g_idx = {float(g): i for i, g in enumerate(g_vals)}
    # E[i, :] = (pi/g_i)^{3/2} exp(-G²/(4 g_i)) on the mesh -> C = E W Eᵀ
    # with W = diag(coul). Accumulated in G-chunks: the full E would be
    # (n_gamma, n_G), which itself reaches GBs at production basis/mesh
    # sizes -- the very object class this screen exists to avoid.
    n_gamma = int(g_vals.shape[0])
    C_tab = np.zeros((n_gamma, n_gamma))
    pref = (np.pi / g_vals) ** 1.5
    chunk = 16384
    for lo in range(0, int(Gq2.shape[0]), chunk):
        G2c = Gq2[lo : lo + chunk]
        E_c = pref[:, None] * np.exp(-G2c[None, :] / (4.0 * g_vals[:, None]))
        C_tab += (E_c * coul[lo : lo + chunk][None, :]) @ E_c.T

    bounds = np.zeros((n_sh, n_sh))
    for sM in range(n_sh):
        A = sh_origin[sM]
        for sN in range(n_sh):
            B0 = sh_origin[sN]
            l_sum = sh_l[sM] + sh_l[sN]
            # |A - (B0 + R)| per cell; conservative polynomial factor.
            dvec = (A - B0)[None, :] - R_arr
            dist2 = np.einsum("rc,rc->r", dvec, dvec)
            if l_sum > 0:
                n_cart = ((sh_l[sM] + 1) * (sh_l[sM] + 2) // 2) * (
                    (sh_l[sN] + 1) * (sh_l[sN] + 2) // 2
                )
                poly = n_cart * (1.0 + max_g_norm + np.sqrt(dist2)) ** l_sum
            else:
                poly = np.ones_like(dist2)
            gam = np.add.outer(sh_exps[sM], sh_exps[sN])  # (nM, nN)
            red = np.multiply.outer(sh_exps[sM], sh_exps[sN]) / gam
            cc = np.multiply.outer(sh_coefs[sM], sh_coefs[sN])
            # w̃_pq = S_R poly_R |c_p c_q| exp(-red_pq dist2_R)
            w = cc * np.einsum(
                "r,pqr->pq", poly, np.exp(-np.multiply.outer(red, dist2))
            )
            w_flat = (sh_scale[sM] * sh_scale[sN]) * w.ravel()
            gi = np.array([g_idx[float(g)] for g in gam.ravel()], dtype=int)
            norm_sq = float(w_flat @ C_tab[np.ix_(gi, gi)] @ w_flat)
            bounds[sM, sN] = np.sqrt(max(norm_sq, 0.0))
    return bounds


def _rsgdf_fit_pair_keep_mask(
    ao_basis: BasisSet,
    R_g: np.ndarray,
    Gq2: np.ndarray,
    coul: np.ndarray,
    metric: np.ndarray,
    *,
    fit_screen_threshold: float,
    fit_pair_list: Optional[np.ndarray],
    progress: Optional[Any],
) -> Optional[np.ndarray]:
    """Return the q-resolved AO-pair mask used by the RSGDF 3c fit."""

    shells_ao = ao_basis.shells()
    n_sh = len(shells_ao)
    sh_sizes = [
        (2 * int(sh.l) + 1)
        if bool(sh.pure) or int(sh.l) == 0
        else ((int(sh.l) + 1) * (int(sh.l) + 2) // 2)
        for sh in shells_ao
    ]
    sh_off = np.concatenate([[0], np.cumsum(sh_sizes)]).astype(int)
    keep_sp = np.ones((n_sh, n_sh), dtype=bool)
    if fit_pair_list is not None:
        pl = np.asarray(fit_pair_list, dtype=int).reshape(-1, 2)
        keep_sp[:] = False
        keep_sp[pl[:, 0], pl[:, 1]] = True
    threshold = float(fit_screen_threshold)
    if threshold > 0.0:
        pair_bounds = _fit_screen_pair_norm_bounds(ao_basis, R_g, Gq2, coul)
        aux_max = float(np.sqrt(np.max(np.real(np.diag(metric)))))
        schwarz_keep = (aux_max * pair_bounds) >= threshold
        schwarz_keep |= schwarz_keep.T
        n_dropped = int(np.count_nonzero(keep_sp & ~schwarz_keep))
        keep_sp &= schwarz_keep
        max_dropped = (
            float(np.max(aux_max * pair_bounds[~schwarz_keep]))
            if n_dropped
            else 0.0
        )
        n_ao_kept = int(
            sum(
                sh_sizes[i] * sh_sizes[j]
                for i in range(n_sh)
                for j in range(n_sh)
                if keep_sp[i, j]
            )
        )
        _progress_info(
            progress,
            "GDF fit screen: threshold "
            f"{threshold:.1e}, kept {int(np.count_nonzero(keep_sp))}/"
            f"{n_sh * n_sh} shell pairs "
            f"({n_ao_kept}/{ao_basis.nbasis ** 2} AO pairs), "
            f"max dropped Schwarz bound {max_dropped:.2e}",
        )
    if bool(np.all(keep_sp)):
        return None

    pair_keep_ao = np.zeros((ao_basis.nbasis, ao_basis.nbasis), dtype=bool)
    for i in range(n_sh):
        for j in range(n_sh):
            if keep_sp[i, j]:
                pair_keep_ao[
                    sh_off[i] : sh_off[i + 1],
                    sh_off[j] : sh_off[j + 1],
                ] = True
    return pair_keep_ao


class _RsgdfQMetricCacheEntry(NamedTuple):
    """Exact q-only state shared by ket-resolved multi-k RSGDF fits."""

    eigvals: np.ndarray
    eigvecs: np.ndarray
    keep: np.ndarray
    pair_keep_ao: Optional[np.ndarray]


def build_lpq_bloch_native_fft(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    k_bra: np.ndarray,
    k_ket: np.ndarray,
    *,
    ke_cutoff: float = 200.0,
    tail_ke_cutoff: Optional[float] = None,
    tail_chunk_g: int = 65536,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    canonical_auxiliary_basis: bool = False,
    fit_screen_threshold: float = 0.0,
    fit_pair_list: Optional[np.ndarray] = None,
    omega_screen: float = 0.0,
    progress: Optional[Any] = None,
    _q_metric_cache: Optional[dict] = None,
) -> np.ndarray:
    r"""Per-(k_bra, k_ket) all-FT GDF cderi for tight periodic cells.

    The **(k_bra, k_ket)-resolved generalisation** of
    :func:`build_lpq_native_fft` (which is the ``k_bra = k_ket = 0``
    special case). Returns the density-fit tensor ``L_{P,muν}`` whose
    contraction reconstructs the periodic ERI block with bra AO ``mu`` at
    crystal momentum ``k_bra`` and ket AO ``ν`` at ``k_ket`` -- exactly
    what the multi-k closed-shell ``J``/``K`` builders need (J uses the
    diagonal ``k_bra = k_ket`` blocks; K(k_i) uses ``(k_i, k_j)`` for
    every ``k_j``).

    **Why the (k_bra, k_ket) resolution is required (not a q-only cache).**
    Working out the Bloch sum (bra fixed at its home cell, ket summed over
    lattice cells ``R``), the GDF cderi pair-density Fourier coefficient is

    .. math::

        \hat\rho_{\mu\nu}^{k_b,k_k}(\mathbf G+\mathbf q)
          = \sum_{\mathbf R} e^{+i\,\mathbf k_k\cdot\mathbf R}
            \int \chi_\mu(\mathbf r)\,\chi_\nu(\mathbf r-\mathbf R)\,
                 e^{-i(\mathbf G+\mathbf q)\cdot\mathbf r}\,d\mathbf r ,
        \qquad \mathbf q = \mathbf k_k - \mathbf k_b .

    The lattice-phase is the **ket momentum** :math:`k_k`, *not* the
    momentum transfer :math:`q` -- so for a fixed ``q`` the cderi still
    depends on ``k_k`` through the inter-cell (``R!=0``) terms. Collapsing
    the cache to ``q`` alone (the legacy ``_build_lpq_q_cache`` /
    single-broadcast-J architecture) is only exact in the
    **no-inter-cell-overlap limit** (vacuum-box cells such as H₂ in a
    12-bohr box, where the ``R!=0`` overlaps are exponentially small and
    the ``k_k`` dependence vanishes). For a tight ionic crystal (LiH
    primitive FCC: Li-H 3.86 bohr in a 7.72-bohr lattice, Bloch overlap
    ~10x the molecular overlap) the ``q``-only cache is wrong by hundreds
    of Ha -- the multi-k ``Lpq`` "internal inconsistency" the Sec.7 sanity
    guard catches.

    **Formula** (Sun, Berkelbach, McClain & Chan, *J. Chem. Phys.* **147**,
    164119 (2017), doi:10.1063/1.4998644 -- the range-separated periodic
    GDF; McClain, Sun, Chan & Berkelbach, *J. Chem. Theory Comput.* **13**,
    1209 (2017), doi:10.1021/acs.jctc.6b01184). With the dense FFT mesh
    ``{G}``, shifted by ``q`` so the Coulomb kernel is evaluated at
    ``G+q`` (the reciprocal support of a Bloch function at momentum ``q``):

    ::

        M(q)_{PQ}   = (4pi/V) S_{|G+q|!=0} F̂_P(G+q)* . F̂_Q(G+q) / |G+q|^2
        T(q)_{P,muν} = (4pi/V) S_{|G+q|!=0} F̂_P(G+q)* . r̂_muν(G+q; k_ket) / |G+q|^2
        L_{P,muν}    = S_{P'} [M(q)^{-1/2}]_{P P'} . T(q)_{P',muν}   (eig-fit)

    where ``F̂_P`` is :func:`rsgdf_aux_fourier_transform` and ``r̂_muν`` is
    the **ket-Bloch-summed** AO-pair FT from
    :func:`vibeqc._aopair_ft.ao_pair_fourier_transform_bloch` (C++ kernel,
    ``k_cart = k_ket``). The ``G+q = 0`` mode is dropped (Ewald /
    neutralising-background convention); it only occurs at ``q = 0`` (then
    ``G = 0``), where the exchange divergence is handled separately by the
    exxdiv-ewald Madelung shift. ``M(q)`` is complex-Hermitian for
    ``q != 0``; at ``q = 0`` the inversion-symmetric mesh makes ``M``
    real-symmetric. ``T(k,k)`` is real only at true Γ and may be complex for
    diagonal ``k_bra = k_ket != 0`` blocks because its AO pair density carries
    the ket Bloch phase.

    **Historical validation, revision-bound by the physical shifted-sphere
    correction.** The following out-of-process PySCF comparison predates D78
    and must be rerun before the exact multi-k value is quoted again. The
    Γ-only support and accumulation order are byte-preserved. LiH primitive
    FCC / sto-3g / def2-svp-jk, ``run_krhf_periodic_gdf`` at
    kmesh=(2,2,2), ``exxdiv='ewald'``::

        vibe-qc (this builder) : E = -7.92039 Ha
        pyscf KRHF.density_fit : E = -7.92200 Ha   (GDF, exxdiv='ewald')
        ΔE(cutoff 15, ke 200) = +1.6 mHa  (production default; chem-acc)
        ΔE(cutoff 30, ke 200) = -1.0 µHa  (µHa parity)

    In that pre-D78 run, the residual at the default was **lattice-cutoff
    (Bloch-pair cell-list)
    truncation**, NOT a DF-method floor: the tight-cell Bloch pair density
    extends over many cells, and the default 15-bohr ``lat_opts`` cell
    list truncates it. Raising the cutoff to 30 bohr reaches µHa parity
    for both Γ-only (+27 µHa at ke=200, -0.5 µHa at ke=800) and multi-k
    (-1.0 µHa at ke=200) -- PySCF's own GDF<->RSDF agree to ~1 µHa, so this
    was genuine parity for the then-active finite support. Re-establish the
    exact numbers with the physical shifted sphere before using this paragraph
    as current validation evidence.

    Parameters
    ----------
    system, ao_basis
        Periodic system (3D) and orbital basis (libint m-ordering).
    aux_basis
        Auxiliary basis. **Must be the modrho-rescaled basis** from
        :func:`make_modrho_aux_basis` (as for :func:`build_lpq_native_fft`).
    k_bra, k_ket
        Cartesian crystal momenta (inverse bohr) of the bra and ket AO.
        ``q = k_ket - k_bra`` sets the reciprocal mesh shift. The finite
        reciprocal support is the physical shifted sphere
        ``0 < |G + q| <= sqrt(2 * ke_cutoff)``. It is invariant under
        ``q -> q + G0`` relabelling and under time reversal, including exact
        ``+/-G/2`` Nyquist transfers.
    ke_cutoff
        Kinetic-energy cutoff (Hartree) for the dense FFT mesh. Default
        ``200`` (mesh-converged for light-atom cells; the residual ~mHa is
        the aux floor, not mesh -- verified flat across ke=200/400/600).
    tail_ke_cutoff
        Optional high-``|G+q|`` tail completion cutoff (Hartree). When set above
        ``ke_cutoff``, the M / T G-sums are extended over the exact
        complementary reciprocal shell up to this cutoff, in chunks of
        ``tail_chunk_g`` G-points. Needed for dense-core cells (Mg/O
        STO-3G) whose tight AO products are unresolved at any affordable
        base mesh -- see the tail block in the source for the full
        rationale. E(ke_cutoff=a, tail_ke_cutoff=b) is exactly
        E(ke_cutoff=b) by construction (same total shifted support).
    tail_chunk_g
        G-points per accumulation chunk, for BOTH the base-mesh M/T
        sweeps and the high-|G| tail (the builder's peak-memory knob:
        the pair-FT transient is ``(n_orb, n_orb, tail_chunk_g)``
        complex, never ``(n_orb, n_orb, n_G)``). The default 65536
        covers typical base meshes in one chunk, which reproduces the
        unchunked contraction on the same shifted support bit-for-bit.
    lat_opts
        Lattice-sum options for the Bloch pair-FT cell list. Defaults to
        ``LatticeSumOptions()``.
    linear_dep_thr
        Absolute eigenvalue floor for the ``M(q)`` truncation. This
        matches PySCF's FFT metric convention and differs deliberately
        from the legacy compcell/MDF/bare builders, where the same knob
        is relative to the largest metric eigenvalue.
    canonical_auxiliary_basis
        If false (the SCF default), return the compact eigenmode factor with
        ``n_kept`` rows. If true, rotate the truncated inverse square root
        back to the original auxiliary-AO basis and return ``n_aux`` rows.
        The latter removes arbitrary eigenvector phases between the ``q`` and
        ``-q`` blocks and is required when two independently built momentum
        blocks are contracted in post-HF theory.
    fit_screen_threshold
        Cauchy-Schwarz pre-screen of the three-centre fit (default
        ``0.0`` = off, exact). AO pairs ``muν`` whose Schwarz bound
        ``max_P sqrt((P|P)) . sqrt((muν|muν))`` on the fit mesh stays
        below the threshold are dropped from the 3c tensor (their
        ``Lpq`` entries are exactly zero). The pair-norm factor is an
        exact analytic upper bound for s-pairs and a conservative
        polynomial screening estimate for higher angular momentum. The
        default keeps the build exact; ``1e-10``-class thresholds have
        reproduced the unscreened energy to well below SCF accuracy on
        the current regression set while the significant-pair count grows
        ~linearly for local systems (Neese et al., Chem. Phys. 356, 98
        (2009), Sec. 3.1). The kept/dropped counts are logged via
        ``progress`` -- no silent truncation.
    fit_pair_list
        Optional ``(n_sp, 2)`` int array of AO **shell-pair indices** to
        which the fit is restricted (all other pairs' ``Lpq`` entries
        are zero). This is the composition seam for space-group
        pair reduction (symmetry drops *redundant* pairs, the Schwarz
        screen drops *negligible* ones; both act on one build path --
        handovers/HANDOVER_SYMMETRY_PAIR_REDUCTION.md). ``None`` (the
        default) keeps every shell pair.
    omega_screen
        Optional short-range (erfc) attenuation of the fitted kernel
        (bohr^-1). Default ``0.0`` fits the full Coulomb kernel
        (bit-identical to the historical build). For ``omega_screen > 0``
        the fit replaces ``1/r_12`` by the screened ``erfc(omega_screen
        * r_12)/r_12`` everywhere (metric, 3c tensor, and high-|G| tail
        receive the identical attenuation, so DF internal consistency is
        preserved by construction), via its Fourier transform::

            FT[erfc(w r)/r](p) = (4 pi / p^2) (1 - exp(-p^2 / (4 w^2)))

        The ``p = G+q = 0`` mode of the screened kernel is FINITE
        (``pi / omega_screen^2``, the ``p -> 0`` limit above) but is
        still excluded from the fit, exactly like the Coulomb ``G+q=0``
        drop -- for the screened kernel the excluded content is exactly
        the rank-1 term ``(pi / (omega_screen^2 V)) S_{mu nu} S_{rho
        sigma}``, which callers restore analytically (see
        ``vibeqc.periodic.ccm.neutral.ccm_sr_exchange_zero_mode_constant``).
        This is the periodic screened-exchange kernel of HSE-type
        functionals (Heyd, Scuseria & Ernzerhof, J. Chem. Phys. 118,
        8207 (2003), Eq. 2), density-fitted per the range-separated
        periodic GDF of Sun, Berkelbach, McClain & Chan, J. Chem. Phys.
        147, 164119 (2017), doi:10.1063/1.4998644. It is the *physical*
        screening parameter of the functional, NOT the numerical Ewald
        split parameter some drivers also call ``omega``.
    progress
        Optional progress logger with an ``info`` method; receives the
        fit-screen statistics line.
    _q_metric_cache
        Internal setup-local cache used by the multi-k drivers. Entries share
        only the q-dependent metric, eigensystem, and Schwarz mask; the
        ket-dependent three-centre tensor is always rebuilt.

    Returns
    -------
    Lpq : complex128 ndarray
        Density-fit tensor in libint AO m-ordering. Its shape is
        ``(n_kept,n_orb,n_orb)`` by default and
        ``(n_aux,n_orb,n_orb)`` in the canonical auxiliary basis.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import direct_lattice_cells

    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    k_bra = np.asarray(k_bra, dtype=float).reshape(3)
    k_ket = np.asarray(k_ket, dtype=float).reshape(3)
    q = _canonical_reciprocal_transfer(system, k_ket - k_bra)
    # Moving the ket by a reciprocal vector leaves exp(+i k_ket.R) unchanged
    # for every lattice translation R and makes every downstream use of q,
    # including zero-mode detection and tail partitioning, label-independent.
    k_ket = k_bra + q

    # A Bloch momentum is defined modulo a reciprocal-lattice vector, and
    # exp(+i (k+G).R) == exp(+i k.R) for every lattice translation R. Build
    # the kinetic cutoff on the physical shifted vectors p = G+q, not by
    # shifting a pre-truncated |G| ball. The latter selected different
    # boundary crescents for q and q+G0 and could not be both reciprocal-label
    # invariant and time-reversal covariant at a Nyquist transfer. The shifted
    # sphere below has both properties by construction. Its p=0 point is
    # dropped (jellium/Ewald G=0 omission).
    Gq = _rsgdf_shifted_dense_g_mesh(system, q, ke_cutoff)
    Gq2 = (Gq**2).sum(axis=1)
    nz = Gq2 > 1e-12
    Gq = Gq[nz]
    Gq2 = Gq2[nz]
    coul = (4.0 * np.pi) / Gq2 / V
    w_scr = float(omega_screen)
    if w_scr > 0.0:
        # Screened (erfc-SR) kernel weight: FT[erfc(w r)/r](p)
        #   = (4 pi / p^2) (1 - exp(-p^2 / (4 w^2)))
        # (see the omega_screen docstring). -expm1(-x) = 1 - exp(-x)
        # evaluates the small-p attenuation without cancellation.
        coul = coul * (-np.expm1(-Gq2 / (4.0 * w_scr * w_scr)))

    # G-chunked accumulation (M2 of HANDOVER_GDF_FIT_SCREENING.md): the
    # base mesh is swept in ``tail_chunk_g``-point chunks, exactly like
    # the high-|G| tail below, so the builder's peak transient is the
    # (n_orb, n_orb, chunk) pair-FT block -- NOT (n_orb, n_orb, n_G) --
    # plus the (n_aux, chunk) aux-FT block. A single chunk (the default
    # covers meshes up to 65536 points) reproduces the historical
    # unchunked build bit-for-bit. The metric sweep runs first because
    # its diagonal feeds the Cauchy-Schwarz fit screen below
    # (sqrt((P|P)) is the aux factor of the bound); the aux FT is then
    # recomputed per chunk in the 3c sweep -- (n_aux . n_G) Gaussian
    # evaluations, negligible next to one (n_orb^2 . n_G . n_cells)
    # pair-FT pass. Aux FT F̂_P(G+q) on the modrho aux (compensating
    # charge fused is the follow-up for µHa -- see docstring).
    chunk = max(int(tail_chunk_g), 1)
    n_base = int(Gq.shape[0])
    n_orb = int(ao_basis.nbasis)
    n_aux = int(aux_basis.nbasis)
    n_chunks = int((n_base + chunk - 1) // chunk) if n_base else 0

    q_key = tuple(float(value) for value in np.round(q, 14))
    q_metric = (
        _q_metric_cache.get(q_key)
        if _q_metric_cache is not None
        else None
    )

    # Sweep A -- 2c metric M(q)_PQ, (naux, naux) Hermitian. The metric
    # depends only on q, not on the ket Bloch phase. Multi-k drivers pass a
    # setup-local cache so the n_k pairs sharing q reuse this exact state;
    # Sweep B below remains per-(k_bra,k_ket).
    M: Optional[np.ndarray] = None
    if q_metric is None:
        M = np.zeros((n_aux, n_aux), dtype=np.complex128)
        for lo in range(0, n_base, chunk):
            aux_ft_c = rsgdf_aux_fourier_transform(
                aux_basis,
                Gq[lo : lo + chunk],
            )
            M += (
                aux_ft_c.conj() * coul[lo : lo + chunk][None, :]
            ) @ aux_ft_c.T
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)

    # ---- Cauchy-Schwarz fit screen (fit_screen_threshold > 0) --------
    # |T_{P,muν}| <= sqrt(M_PP) . sqrt(S_G coul |r̂_muν|^2): AO pairs whose
    # bound stays below the threshold for EVERY aux P contribute nothing
    # above threshold to the fit and are dropped from the 3c tensor --
    # the Almlöf/Häser-Ahlrichs direct-SCF screen applied to the RI fit
    # (Neese et al., Chem. Phys. 356, 98 (2009), Sec. 3.1; Häser &
    # Ahlrichs 1989; Ochsenfeld et al. 1998). See
    # :func:`_fit_screen_pair_norm_bounds` for the bound's derivation.
    # ``fit_pair_list`` optionally restricts the computed shell pairs
    # further (the seam for space-group pair reduction -- symmetry drops
    # redundant pairs, Schwarz drops negligible ones; one build path,
    # handovers/HANDOVER_SYMMETRY_PAIR_REDUCTION.md).
    if q_metric is None:
        if M is None:
            raise RuntimeError("RSGDF q-metric cache miss did not build M(q)")
        pair_keep_ao = _rsgdf_fit_pair_keep_mask(
            ao_basis,
            R_g,
            Gq2,
            coul,
            M,
            fit_screen_threshold=float(fit_screen_threshold),
            fit_pair_list=fit_pair_list,
            progress=progress,
        )
    else:
        pair_keep_ao = q_metric.pair_keep_ao

    # Sweep B -- 3c tensor T(q)_{P,muν} from the ket-Bloch-summed AO-pair
    # FT r̂_muν(G+q; k_ket): bra at home cell, ket summed over R with
    # phase exp(+i k_ket.R) (matches the C++ kernel's +k Bloch convention
    # and vibe-qc's bloch_sum). Chunked over G (see the M2 note above);
    # the per-chunk pair FT gets the per-L AO calibration (same
    # convention as build_lpq_native_fft) in place and is screened
    # before the contraction, so the dense (n_orb, n_orb, n_G) tensor
    # never materialises.
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    T = np.zeros((n_aux, n_orb, n_orb), dtype=np.complex128)
    if n_chunks > 1:
        _progress_info(
            progress,
            f"GDF fit build: {n_base} G-points in {n_chunks} chunks of "
            f"{chunk} (pair-FT transient "
            f"{n_orb * n_orb * chunk * 16 / 1e9:.2f} GB)",
        )
    for lo in range(0, n_base, chunk):
        aux_ft_c = rsgdf_aux_fourier_transform(aux_basis, Gq[lo : lo + chunk])
        pair_ft_c = ao_pair_fourier_transform_bloch(
            ao_basis, Gq[lo : lo + chunk], R_g, k_cart=k_ket
        )
        pair_ft_c *= pair_scales[:, :, None]
        if pair_keep_ao is not None:
            pair_ft_c[~pair_keep_ao] = 0.0
        aux_w_c = aux_ft_c.conj() * coul[lo : lo + chunk][None, :]
        T += np.einsum("Pk,mnk->Pmn", aux_w_c, pair_ft_c)

    # High-|G| tail completion (prompt-11 / P01 dense-core fix). Tight
    # core AO products (Mg/O 1s on STO-3G-class bases) have reciprocal
    # support far beyond any affordable base mesh: at ke_cutoff = 200 Ha
    # the truncated T (and, to a lesser degree, M) leave the MgO Gamma
    # RHF ~0.5 Ha over-bound, decaying only slowly with the cutoff
    # (-515 / -141 / -25 mHa at 200 / 400 / 800 Ha vs PySCF GDF).
    # Replacing the short-range piece with real-space integrals (the
    # PySCF RSDF compact/smooth split) mixes two integral
    # representations and was observed to amplify near-null DF metric
    # modes catastrophically. Instead, stay INSIDE the FT
    # representation: extend the same M / T G-sums over the exact
    # complementary reciprocal-lattice shell
    #     sqrt(2 ke_cutoff) < |G+q| <= sqrt(2 tail_ke_cutoff)
    # with the same analytic kernels, chunked so the (n_orb, n_orb,
    # n_G) pair-FT block stays in memory. Both the metric and the
    # 3c tensor receive the identical treatment, so DF internal
    # consistency is preserved by construction. The AO-pair FT decays
    # as exp(-|G+q|^2 / (4 (z_mu + z_nu))): only the steep-pair rows
    # survive at high G, which is exactly the under-resolved physics.
    if tail_ke_cutoff is not None and float(tail_ke_cutoff) > float(ke_cutoff):
        Gq_tail_all = _rsgdf_shifted_dense_g_mesh(
            system,
            q,
            float(tail_ke_cutoff),
        )
        G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
        tail_norms = np.linalg.norm(Gq_tail_all, axis=1)
        # Strict complement of the base shifted sphere (|G+q| <= G_max).
        base_boundary_tolerance = (
            0.0
            if float(np.linalg.norm(q)) < 1.0e-14
            else _rsgdf_shifted_sphere_boundary_tolerance(
                system,
                G_base_max,
                q,
            )
        )
        Gq_tail = Gq_tail_all[
            tail_norms > G_base_max + base_boundary_tolerance
        ]
        n_tail = Gq_tail.shape[0]
        chunk = max(int(tail_chunk_g), 1)
        for lo in range(0, n_tail, chunk):
            Gqc = Gq_tail[lo : lo + chunk]
            Gq2c = (Gqc**2).sum(axis=1)
            nzc = Gq2c > 1e-12
            Gqc = Gqc[nzc]
            if Gqc.shape[0] == 0:
                continue
            coul_c = (4.0 * np.pi) / Gq2c[nzc] / V
            if w_scr > 0.0:
                # Same screened-kernel attenuation as the base mesh: the
                # tail must complete exactly the operator the base fits.
                coul_c = coul_c * (
                    -np.expm1(-Gq2c[nzc] / (4.0 * w_scr * w_scr))
                )
            aux_ft_c = rsgdf_aux_fourier_transform(aux_basis, Gqc)
            pair_ft_c = ao_pair_fourier_transform_bloch(
                ao_basis, Gqc, R_g, k_cart=k_ket
            )
            pair_ft_c *= np.outer(ao_scales, ao_scales)[:, :, None]
            if pair_keep_ao is not None:
                # Same screened pair set as the base mesh: the tail must
                # complete exactly the pairs the base kept.
                pair_ft_c[~pair_keep_ao] = 0.0
            aux_w_c = aux_ft_c.conj() * coul_c[None, :]
            if q_metric is None:
                if M is None:
                    raise RuntimeError(
                        "RSGDF q-metric cache miss lost M(q) during tail build"
                    )
                M += aux_w_c @ aux_ft_c.T
            T += np.einsum("Pk,mnk->Pmn", aux_w_c, pair_ft_c)

    if q_metric is None:
        if M is None:
            raise RuntimeError("RSGDF q-metric cache miss did not retain M(q)")
        if float(np.linalg.norm(q)) < 1e-12:
            # q = 0: the 2c metric M is built on the inversion-symmetric mesh
            # {G} and depends only on q, so it is exactly real-symmetric here.
            # T(k,k) remains complex away from true Gamma because it carries
            # the ket Bloch phase.
            M = np.real(M)
        M = 0.5 * (M + M.conj().T)

        eigvals, U = np.linalg.eigh(M)
        max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
        if max_eig <= 0:
            raise RuntimeError(
                "build_lpq_bloch_native_fft: 2c metric has no positive "
                f"eigenvalue (max eig = {max_eig:.3e}) at q={q}; "
                "aux/mesh invalid."
            )
        keep = _metric_keep_mask(eigvals, linear_dep_thr)
        if not bool(np.any(keep)):
            raise RuntimeError(
                "build_lpq_bloch_native_fft: no eigenvalues above threshold "
                f"({linear_dep_thr:.1e}) at q={q}; "
                "aux basis or threshold invalid."
            )
        q_metric = _RsgdfQMetricCacheEntry(
            eigvals=eigvals,
            eigvecs=U,
            keep=keep,
            pair_keep_ao=pair_keep_ao,
        )
        if _q_metric_cache is not None:
            _q_metric_cache[q_key] = q_metric
    eigvals = q_metric.eigvals
    U = q_metric.eigvecs
    keep = q_metric.keep
    n_kept = int(keep.sum())
    T_flat = T.reshape(T.shape[0], -1)
    if canonical_auxiliary_basis:
        kept_vectors = U[:, keep]
        inverse_sqrt = (
            kept_vectors / np.sqrt(eigvals[keep])[None, :]
        ) @ kept_vectors.conj().T
        Lpq = inverse_sqrt @ T_flat
        n_factor = T.shape[0]
    else:
        Lpq = (U[:, keep].conj().T @ T_flat) / np.sqrt(eigvals[keep])[:, None]
        n_factor = n_kept
    return Lpq.reshape(n_factor, T.shape[1], T.shape[2])



def _rsgdf_aux_ft_cache_budget_bytes() -> int:
    """Byte budget for the shared-q builder's per-chunk aux-FT cache.

    Sweep A's aux FT is reused by Sweep B instead of being recomputed
    (a pure duplicate-work removal), but the cache grows as
    ``(n_aux, n_base)`` -- it does NOT shrink when a caller lowers
    ``tail_chunk_g``. Left unbounded it becomes a floor under the peak
    that the chunking knob cannot reach, so it gets its own budget.

    Override with ``VIBEQC_GDF_AUX_FT_CACHE_MIB`` (default 256 MiB;
    ``0`` disables caching and restores the recompute-every-chunk
    behaviour).
    """
    try:
        target_mib = float(
            os.environ.get("VIBEQC_GDF_AUX_FT_CACHE_MIB", "256.0")
        )
    except ValueError:
        target_mib = 256.0
    return int(max(0.0, target_mib) * 1024.0 * 1024.0)


def _build_coulomb_from_diagonal_factors(
    factors: Sequence[np.ndarray],
    densities: Sequence[np.ndarray],
    weights: Sequence[float],
) -> list[np.ndarray]:
    """Contract one q=0 fit in any shared auxiliary coordinate system.

    ``rho_P = sum_k w_k sum_mn conj(L[k,P,m,n]) D[k,m,n]`` and
    ``J[k,m,n] = sum_P L[k,P,m,n] rho_P``. Both the conjugation and
    the density index order matter. Replacing the first contraction by
    ``tr(L D)`` assumes every auxiliary factor is Hermitian; a complex
    unitary change of fitting coordinates violates that assumption while
    preserving the fitted Coulomb operator. Gamma and multi-k use this
    same Gram contraction, without a full conjugated factor-cache copy.
    """
    n_k = len(factors)
    if n_k == 0 or len(densities) != n_k or len(weights) != n_k:
        raise ValueError("GDF Coulomb requires matching nonempty factors, densities and weights")
    shape = np.shape(factors[0])
    if len(shape) != 3 or shape[0] < 1 or shape[1] != shape[2]:
        raise ValueError("GDF Coulomb factors must have shape (rank,nao,nao)")
    if any(np.shape(f) != shape for f in factors):
        raise ValueError("GDF Coulomb diagonal factors must share one auxiliary rank")
    if any(np.shape(d) != shape[1:] for d in densities):
        raise ValueError("GDF Coulomb density and factor AO shapes differ")
    dtype = np.result_type(*(np.asarray(a).dtype for a in (*factors, *densities)), float)
    result = [np.zeros(shape[1:], dtype=dtype) for _ in range(n_k)]
    # At most 1 MiB of conjugated factors, or one AO matrix if larger.
    panel_rank = max(1, min(shape[0], 2**20 // max(1, 16 * shape[1]**2)))
    for first in range(0, shape[0], panel_rank):
        last = min(first + panel_rank, shape[0])
        rho = np.zeros(last - first, dtype=dtype)
        for factor, density, weight in zip(factors, densities, weights):
            panel = np.asarray(factor)[first:last]
            rho += float(weight) * np.einsum(
                "Pmn,mn->P", panel.conj(), density, optimize=True,
            )
        for matrix, factor in zip(result, factors):
            matrix += np.einsum(
                "P,Pmn->mn", rho, np.asarray(factor)[first:last], optimize=True,
            )
    return [0.5 * (matrix + matrix.conj().T) for matrix in result]


def _exchange_auxiliary_panel_rank(naux, nbasis, bytes_per_auxiliary, cap):
    available = int(cap) - 32 * nbasis**2
    if available < bytes_per_auxiliary:
        raise MemoryError("GDF exchange workspace cannot hold one auxiliary panel")
    return max(1, min(naux, available // bytes_per_auxiliary))


def _accumulate_exchange_from_factors(
    output: np.ndarray,
    factors: np.ndarray,
    density: np.ndarray,
    weight: float,
    workspace_byte_cap: int,
) -> None:
    """Add weight * sum_P L_P D L_P^H with bounded auxiliary scratch.

    Gamma and multi-k share the conjugation, index order and panel budget.
    The cap covers four complex panel arrays and two AO temporaries;
    caller-owned inputs and output are separate. It is not an RSS cap.
    """
    factors, density = np.asarray(factors), np.asarray(density)
    nbasis = output.shape[0]
    if (output.shape != (nbasis, nbasis) or density.shape != output.shape
            or factors.ndim != 3 or factors.shape[1:] != output.shape):
        raise ValueError("GDF exchange factor, density and output shapes differ")
    panel_rank = _exchange_auxiliary_panel_rank(
        len(factors), nbasis, 64 * nbasis**2, workspace_byte_cap,
    )
    for first in range(0, len(factors), panel_rank):
        panel = factors[first:first + panel_rank]
        temporary = np.einsum("Lpr,rs->Lps", panel, density, optimize=True)
        output += float(weight) * np.einsum(
            "Lps,Lqs->pq", temporary, panel.conj(), optimize=True,
        )
        del temporary


class _RangeSeparatedGdfFitState(NamedTuple):
    """Borrowed source arrays for differentiation of this exact fit."""

    three_center: np.ndarray
    eigenvectors: np.ndarray
    keep: np.ndarray
    vectors: np.ndarray
    ket_kpoints: np.ndarray
    source_parameters: Optional[tuple] = None
    source_signature: str = ""


def _range_separated_gdf_source_signature(system, orbital, auxiliary):
    """Hash the geometry and normalized bases bound to retained fit arrays."""
    digest = hashlib.sha256(b"vibeqc-rsgdf-source-v1\0")
    digest.update(str(int(system.dim)).encode("ascii"))
    digest.update(np.asarray(system.lattice, dtype="<f8").tobytes())
    for atom in system.unit_cell:
        digest.update(str(int(atom.Z)).encode("ascii"))
        digest.update(b":")
        digest.update(np.asarray(atom.xyz, dtype="<f8").tobytes())
    for basis in (orbital, auxiliary):
        digest.update(_basis_fingerprint(basis).encode("ascii"))
    return digest.hexdigest()


class _RangeSeparatedGdfBatch(NamedTuple):
    factors: np.ndarray
    transfer: np.ndarray
    metric_eigenvalues: np.ndarray
    reciprocal_vector_count: int
    reserved_peak_bytes: int
    fit_state: Optional[_RangeSeparatedGdfFitState] = None


class _RangeSeparatedGdfAdmissionError(MemoryError):
    """A rejected reservation, with explicit k-batch retryability.

    Tensor storage can shrink with a k batch. Reciprocal candidate/workspace
    caps describe a single q sphere and cannot be repaired by that retry.
    Actual allocator failures remain ordinary MemoryError exceptions.
    """

    def __init__(self, message, *, retry_with_fewer_kpoints=True):
        super().__init__(message)
        self.retry_with_fewer_kpoints = bool(retry_with_fewer_kpoints)


class _RangeSeparatedGdfCutoffs(NamedTuple):
    pair_cutoff: float
    auxiliary_cutoff: float
    ke_cutoff: float
    integral_screen_error: float
    raw_integral_error: float


def _rsgdf_shell_envelopes(basis):
    """Return (a, log C) with |AO(r)| <= C exp(-a r^2).

    Shell coefficients already include primitive normalization. This is
    the same absolute polynomial bound used by the native SR screen;
    signed contraction cancellation is never used to shorten a cutoff.
    """
    envelopes = []
    for shell in basis.shells():
        alpha = np.asarray(shell.exponents, dtype=float)
        coefficients = np.asarray(shell.coefficients, dtype=float)
        angular = int(shell.l)
        if (alpha.size == 0 or alpha.shape != coefficients.shape
                or not np.isfinite(alpha).all() or np.any(alpha <= 0)
                or not np.isfinite(coefficients).all() or angular < 0):
            raise ValueError("range-separated GDF requires finite Gaussian shells")
        exponent = float(alpha.min()) / (2 if angular else 1)
        live = coefficients != 0
        terms = np.log(np.abs(coefficients[live]))
        if angular:
            half_l = angular / 2
            terms += half_l * (np.log(half_l / (alpha[live] - exponent)) - 1)
        log_coefficient = float(np.logaddexp.reduce(terms, initial=-np.inf))
        envelopes.append((exponent, log_coefficient))
    if not envelopes:
        raise ValueError("range-separated GDF requires nonempty Gaussian bases")
    # Bounds are maxima per raw matrix element. Repeated atom species
    # contribute identical envelopes, with no multiplicity in that maximum.
    return list(dict.fromkeys(envelopes))


def _rsgdf_log_lattice_gaussian_bound(decay, smallest_singular_value):
    """Uniform-in-shift bound for sum_n exp(-decay |A n-r|^2).

    |A(n-f)| >= sigma_min(A)|n-f| and the one-dimensional Gaussian
    lattice sum is at most 1 + integral exp(-c*x*x) dx. The latter
    follows from Poisson summation (maximum at integer shift) and the
    integral bound for the decreasing positive half-line Gaussian.
    No lattice points or covering spheres are allocated.
    """
    return 3 * np.logaddexp(
        0., .5 * np.log(np.pi / decay) - np.log(smallest_singular_value),
    )


def _plan_range_separated_gdf_cutoffs(
    system, orbital, auxiliary, q_transfer, *, omega: float,
    raw_integral_error: float,
) -> _RangeSeparatedGdfCutoffs:
    """Bound omitted SR/LR tails per raw M/T element, before whitening.

    Ye/Berkelbach (2021), Eqs. (12)-(13), supply the decomposition.
    Absolute Gaussian envelopes and uniform lattice-sum bounds supply
    conservative truncation estimates for *each* term, including the AO
    image tail in the LR and finite-zero-mode overlap. This does not
    certify an SCF energy error or metric rank stability. It is a private
    planner; production defaults are unchanged pending route validation.
    """
    if (not np.isfinite(omega) or omega <= 0
            or not np.isfinite(raw_integral_error) or raw_integral_error <= 0):
        raise ValueError("range-separated GDF requires positive finite omega and error")
    lattice = np.asarray(system.lattice, dtype=float)
    if int(system.dim) != 3 or not np.isfinite(lattice).all():
        raise ValueError("range-separated GDF cutoff bounds require a finite 3D cell")
    sigma = float(np.linalg.svd(lattice, compute_uv=False)[-1])
    if sigma <= 0:
        raise ValueError("range-separated GDF cutoff bounds require a nonsingular cell")
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    reciprocal_sigma = float(np.linalg.svd(reciprocal, compute_uv=False)[-1])
    volume = abs(float(np.linalg.det(lattice)))
    q_input = np.asarray(q_transfer)
    if (q_input.shape != (3,) or np.iscomplexobj(q_input)
            or not np.isfinite(q_input).all()):
        raise ValueError("range-separated GDF cutoff bounds require a finite real q")
    q = _canonical_reciprocal_transfer(system, q_input)
    fractional = lattice.T @ q / (2 * np.pi)
    distance = float(np.linalg.norm(fractional - np.rint(fractional)))
    zero_transfer = distance < 1e-12
    minimum_p = reciprocal_sigma * (1 if zero_transfer else distance)
    gamma = 1 / (4 * omega**2)
    log_weight_sum = (
        np.log(4 * np.pi / volume) - 2 * np.log(minimum_p)
        + _rsgdf_log_lattice_gaussian_bound(gamma, reciprocal_sigma)
    )
    if zero_transfer:
        log_weight_sum = np.logaddexp(
            log_weight_sum, np.log(np.pi / (volume * omega**2)),
        )
    # T has four truncation terms and one finite-domain screening term;
    # M has two truncation terms and screening. Leave additional slack
    # for floating-point evaluation of these conservative bounds.
    log_budget = np.log(raw_integral_error) - np.log(8.) - 1e-8
    pair_squared = auxiliary_squared = 1e-12
    reciprocal_prefactor = -np.inf
    orbital_envelopes = _rsgdf_shell_envelopes(orbital)
    auxiliary_envelopes = _rsgdf_shell_envelopes(auxiliary)

    def theta(decay):
        return _rsgdf_log_lattice_gaussian_bound(decay, sigma)

    def sr_bound(a, b, log_c):
        eta_squared = 1 / (1 / a + 1 / b)
        ratio = eta_squared / omega**2
        root = np.sqrt(1 + ratio)
        difference = np.sqrt(eta_squared) * ratio / (root * (root + 1))
        decay = 1 / (1 / a + 1 / b + 1 / omega**2)
        prefactor = (log_c + 2.5 * np.log(np.pi) + np.log(2 * difference)
                     - 1.5 * (np.log(a) + np.log(b)))
        return decay, prefactor

    for ap, cp in auxiliary_envelopes:
        charge = cp + 1.5 * np.log(np.pi / ap)
        for aq, cq in auxiliary_envelopes:
            decay, prefactor = sr_bound(ap, aq, cp + cq)
            auxiliary_squared = max(
                auxiliary_squared, 2 / decay * (prefactor + theta(decay / 2) - log_budget),
            )
            reciprocal_prefactor = max(
                reciprocal_prefactor, charge + cq + 1.5 * np.log(np.pi / aq),
            )
        for am, cm in orbital_envelopes:
            for an, cn in orbital_envelopes:
                combined = am + an
                beta = 1 / (1 / am + 1 / an)
                pair_charge = cm + cn + 1.5 * np.log(np.pi / combined)
                decay, prefactor = sr_bound(ap, combined, cp + cm + cn)
                pair_squared = max(
                    pair_squared,
                    2 / beta * (prefactor + theta(decay) + theta(beta / 2) - log_budget),
                    2 / beta * (charge + pair_charge + log_weight_sum
                                + theta(beta / 2) - log_budget),
                )
                auxiliary_squared = max(
                    auxiliary_squared,
                    2 / decay * (prefactor + theta(beta) + theta(decay / 2) - log_budget),
                )
                reciprocal_prefactor = max(
                    reciprocal_prefactor, charge + pair_charge + theta(beta),
                )
    # For |p|>G: 1/p^2 <= 1/G^2 and split the Gaussian exponent
    # between exp(-gamma G^2/2) and the uniform reciprocal lattice sum.
    reciprocal_prefactor += (
        np.log(4 * np.pi / volume)
        + _rsgdf_log_lattice_gaussian_bound(gamma / 2, reciprocal_sigma)
    )

    def reciprocal_log_tail(g_squared):
        return reciprocal_prefactor - np.log(g_squared) - gamma * g_squared / 2

    lower, upper = 0., 1.
    while reciprocal_log_tail(upper) > log_budget:
        upper *= 2
        if not np.isfinite(upper):
            raise ValueError("range-separated GDF reciprocal cutoff is not representable")
    for _ in range(64):
        middle = (lower + upper) / 2
        if reciprocal_log_tail(middle) > log_budget:
            lower = middle
        else:
            upper = middle
    cutoffs = np.sqrt(pair_squared), np.sqrt(auxiliary_squared), upper / 2
    if not np.isfinite(cutoffs).all():
        raise ValueError("range-separated GDF cutoff bounds are not representable")
    return _RangeSeparatedGdfCutoffs(
        *(float(np.nextafter(value, np.inf)) for value in cutoffs),
        raw_integral_error / 8, raw_integral_error,
    )


def _build_lpq_range_separated_shared_q(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    k_bra_list: np.ndarray,
    q_transfer: np.ndarray,
    *,
    omega: float,
    pair_cutoff: float,
    auxiliary_cutoff: float,
    ke_cutoff: float,
    linear_dep_thr: float,
    memory_byte_cap: int,
    native_workspace_byte_cap: int,
    image_candidate_cap: int,
    reciprocal_candidate_cap: int,
    integral_screen_error: float = 0.0,
    retain_fit_state: bool = False,
    _metric_state: Optional[dict] = None,
    canonical_auxiliary_basis: bool = False,
) -> _RangeSeparatedGdfBatch:
    """One admitted SR/LR fit for Gamma or a general shared-q batch.

    All cutoffs are explicit finite-domain inputs and require convergence.
    The numerical reservation includes mesh enumeration, raw M/T, LAPACK's
    queried work arrays, whitening and the returned factors, including the
    binding's Eigen copy of the reciprocal vectors. Caller-owned
    bases, existing factors, library global state and allocator overhead are
    separate. With ``retain_fit_state``, the returned source and eigensystem
    share their original storage with a gradient consumer. Their overlap
    with whitening is included in the reservation. This private integration
    leaf does not choose SCF defaults.

    ``canonical_auxiliary_basis`` returns M^(-1/2) T in the original
    auxiliary coordinates, annihilating the discarded auxiliary modes.
    This is the covariant fitting frame needed for auxiliary symmetry
    transport; it reconstructs the same Gram operator as the smaller
    eigenmode factors. Fit-response retention currently uses eigenmodes.
    """
    from scipy.linalg import get_lapack_funcs
    from ._vibeqc_core import compute_gdf_range_separated_integrals

    n_aux, n_orb = int(aux_basis.nbasis), int(ao_basis.nbasis)
    n_k = len(k_bra_list)
    if min(n_aux, n_orb, n_k) < 1:
        raise ValueError("range-separated GDF requires nonempty bases and k points")
    if min(memory_byte_cap, native_workspace_byte_cap, image_candidate_cap,
           reciprocal_candidate_cap) <= 0:
        raise ValueError("range-separated GDF memory and candidate caps must be positive")
    if any(not np.isfinite(value) or value <= 0.0
           for value in (omega, pair_cutoff, auxiliary_cutoff, ke_cutoff)):
        raise ValueError("range-separated GDF omega and cutoffs must be finite and positive")
    if not np.isfinite(linear_dep_thr) or linear_dep_thr < 0.0:
        raise ValueError("range-separated GDF metric threshold must be finite and nonnegative")
    if canonical_auxiliary_basis and retain_fit_state:
        raise ValueError("range-separated GDF fit-response retention requires eigenmode factors")
    metric_bytes = 16 * n_aux * n_aux
    tensor_bytes = 16 * n_k * n_aux * n_orb * n_orb
    small_arrays = 4096 + 128 * (n_aux + n_k)
    if metric_bytes + tensor_bytes + small_arrays >= int(memory_byte_cap):
        raise _RangeSeparatedGdfAdmissionError(
            "range-separated GDF raw integral reservation exceeds memory cap"
        )
    # Query the actual linked LAPACK implementation, rather than assuming
    # eigensystem work is negligible compared with the raw fitting metric.
    heevd, heevd_lwork = get_lapack_funcs(("heevd", "heevd_lwork"), dtype=np.complex128)
    work, iwork, rwork, info = heevd_lwork(n_aux, compute_v=1, lower=1)
    if info != 0:
        raise RuntimeError(f"range-separated GDF LAPACK workspace query failed ({info})")
    lwork, liwork, lrwork = int(np.ceil(work.real)), int(iwork), int(np.ceil(rwork))
    eigensolver_bytes = 16 * lwork + heevd.int_dtype.itemsize * liwork + 8 * lrwork
    mesh_workspace = 4096 + 128 * 512
    nonmesh_bytes = small_arrays + max(
        mesh_workspace,
        tensor_bytes + 2 * metric_bytes + int(native_workspace_byte_cap),
        tensor_bytes + 2 * metric_bytes + eigensolver_bytes,
        2 * tensor_bytes + 3 * metric_bytes,
    )
    if nonmesh_bytes >= int(memory_byte_cap):
        raise _RangeSeparatedGdfAdmissionError(
            "range-separated GDF factorization/workspace reservation exceeds memory cap"
        )
    k_bras = np.asarray(k_bra_list)
    if (k_bras.shape != (n_k, 3) or np.iscomplexobj(k_bras)
            or not np.isfinite(k_bras).all()):
        raise ValueError("range-separated GDF bra k points must have finite real shape (nk,3)")
    q_input = np.asarray(q_transfer)
    if q_input.shape != (3,) or np.iscomplexobj(q_input) or not np.isfinite(q_input).all():
        raise ValueError("range-separated GDF q must be a finite real three-vector")
    q = _canonical_reciprocal_transfer(system, q_input)
    k_kets = np.asarray(k_bras, dtype=np.float64) + q
    source_signature = _range_separated_gdf_source_signature(system, ao_basis, aux_basis)
    source_key = (
        source_signature, tuple(q), omega, pair_cutoff, auxiliary_cutoff,
        ke_cutoff, linear_dep_thr, integral_screen_error, bool(canonical_auxiliary_basis),
    )
    metric_hit = bool(_metric_state)
    if metric_hit and _metric_state.get("source_key") != source_key:
        raise ValueError("range-separated GDF metric state belongs to a different source")
    vectors = _rsgdf_bounded_reciprocal_sphere(
        system, q, ke_cutoff, output_byte_cap=(int(memory_byte_cap) - nonmesh_bytes) // 2,
        workspace_byte_cap=mesh_workspace, candidate_cap=reciprocal_candidate_cap,
    )
    metric, three_center, n_g = compute_gdf_range_separated_integrals(
        ao_basis, aux_basis, system, q, k_kets, vectors, omega,
        pair_cutoff, auxiliary_cutoff, metric_bytes + tensor_bytes,
        int(native_workspace_byte_cap), int(image_candidate_cap),
        float(integral_screen_error), not metric_hit,
    )
    if metric_hit:
        eigenvalues = _metric_state["eigenvalues"]
        eigenvectors = _metric_state["eigenvectors"]
        whitener = _metric_state["whitener"]
        kept = _metric_state["kept"]
    else:
        scale, hermiticity_error = 1.0, 0.0
        for column in range(n_aux):
            if not np.isfinite(metric[:, column]).all():
                raise ValueError("range-separated GDF metric contains nonfinite integrals")
            scale = max(scale, float(np.max(np.abs(metric[:, column]))))
            hermiticity_error = max(hermiticity_error, float(np.max(np.abs(
                metric[:, column] - metric[column, :].conj()
            ))))
        if hermiticity_error > 64 * np.finfo(float).eps * n_aux * scale:
            raise ValueError("range-separated GDF metric is not Hermitian; check source convergence")
        eigenvalues, eigenvectors, info = heevd(
            metric, compute_v=1, lower=1, lwork=lwork, liwork=liwork,
            lrwork=lrwork, overwrite_a=0,
        )
        if info != 0:
            raise RuntimeError(f"range-separated GDF metric diagonalization failed ({info})")
        if not np.isfinite(eigenvalues).all():
            raise ValueError("range-separated GDF metric eigenvalues are nonfinite")
        negative_tolerance = max(
            float(linear_dep_thr),
            64 * np.finfo(float).eps * n_aux * max(1.0, float(np.max(np.abs(eigenvalues)))),
        )
        if eigenvalues[0] < -negative_tolerance:
            raise ValueError("range-separated GDF metric has negative modes; converge the integral cutoffs")
        kept = np.flatnonzero(_metric_keep_mask(eigenvalues, linear_dep_thr))
        if len(kept) == 0:
            raise ValueError("range-separated GDF metric has no retained positive modes")
        whitener = np.empty((len(kept), n_aux), dtype=np.complex128)
        for row, index in enumerate(kept):
            whitener[row] = eigenvectors[:, index].conj()
            whitener[row] /= np.sqrt(eigenvalues[index])
    del metric
    if canonical_auxiliary_basis and not metric_hit:
        # LAPACK returns ascending eigenvalues, so positive-threshold
        # retention is a contiguous suffix. This column slice is a view;
        # no second eigenvector matrix is materialized. Metric storage is
        # released before the canonical whitener is allocated.
        whitener = eigenvectors[:, int(kept[0]):] @ whitener
    fit_state = None
    if retain_fit_state:
        fit_state = _RangeSeparatedGdfFitState(
            three_center, eigenvectors,
            _metric_keep_mask(eigenvalues, linear_dep_thr), vectors, k_kets,
            (omega, pair_cutoff, auxiliary_cutoff, image_candidate_cap,
             integral_screen_error, linear_dep_thr),
            source_signature,
        )
    elif _metric_state is None:
        del eigenvectors
    factor_rank = n_aux if canonical_auxiliary_basis else len(kept)
    factors = np.empty((n_k, factor_rank, n_orb, n_orb), dtype=np.complex128)
    def require_finite(array, label):
        # A full boolean mask would be another batch-sized allocation.
        # These native/NumPy outputs are contiguous, so reshape is a view.
        flat = array.reshape(-1)
        for first in range(0, flat.size, 4096):
            if not np.isfinite(flat[first:first + 4096]).all():
                raise ValueError(f"range-separated GDF {label} is nonfinite")
    for k in range(n_k):
        require_finite(three_center[k], "three-center source")
        np.matmul(whitener, three_center[k].reshape(n_aux, -1),
                  out=factors[k].reshape(factor_rank, -1))
        require_finite(factors[k], "factors")
    if _metric_state is not None and not metric_hit:
        _metric_state.update(
            source_key=source_key, eigenvalues=eigenvalues,
            eigenvectors=eigenvectors, whitener=whitener, kept=kept,
        )
    return _RangeSeparatedGdfBatch(
        factors, q, eigenvalues, int(n_g), nonmesh_bytes + 2 * vectors.nbytes,
        fit_state,
    )


def build_lpq_bloch_native_fft_shared_q(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    k_bra_list: np.ndarray,
    q_transfer: np.ndarray,
    *,
    ke_cutoff: float = 200.0,
    tail_ke_cutoff: Optional[float] = None,
    tail_chunk_g: int = 65536,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    canonical_auxiliary_basis: bool = False,
    fit_screen_threshold: float = 0.0,
    fit_pair_list: Optional[np.ndarray] = None,
    omega_screen: float = 0.0,
    progress: Optional[Any] = None,
    _q_metric_cache: Optional[dict] = None,
) -> list[np.ndarray]:
    r"""Batched shared-q RSGDF cderi: one pair-FT pass for n_k bra momenta.

    Builds :func:`build_lpq_bloch_native_fft` for every pair
    ``(k_bra_i, k_bra_i + q)`` of one canonical momentum transfer ``q``
    while sharing everything that depends only on ``q`` — the shifted
    reciprocal mesh, Coulomb weights, auxiliary FT, 2c metric
    eigensystem, Schwarz pair mask, and, crucially, the dominant
    ket-Bloch AO-pair FT pass (base mesh AND high-|G| tail), which is
    batched across the ``n_k`` ket momenta through
    :func:`vibeqc._aopair_ft.ao_pair_fourier_transform_bloch_multi`.
    On a regular full mesh this reduces the multi-k GDF cderi build
    from ``n_k^2`` to ``n_k`` pair-FT passes — the same math as
    Eqs. M(q)/T(q) of Sun, Berkelbach, McClain & Chan, J. Chem. Phys.
    147, 164119 (2017), doi:10.1063/1.4998644, evaluated with exact
    work-sharing (no screening, storage, or convention change).

    The G-chunk is divided by the batch width so the batched pair-FT
    transient ``(n_k_batch, n_orb, n_orb, chunk_eff)`` respects the same
    peak bound as the single-pair builder's documented
    ``(n_orb, n_orb, tail_chunk_g)``.

    Parameters mirror :func:`build_lpq_bloch_native_fft`;
    ``k_bra_list`` is ``(n_k, 3)`` Cartesian bra momenta and
    ``q_transfer`` is the shared momentum transfer (canonicalised
    internally exactly like the single-pair builder). Returns one Lpq
    tensor per bra momentum, in input order. Parity with per-pair
    single builds is pinned by
    ``tests/test_rsgdf_shared_q_batch.py`` (floating-point rounding
    only; the batched kernel folds the Bloch phase after the cart→sph
    transform).
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch_multi
    from ._vibeqc_core import direct_lattice_cells

    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    k_bras = np.ascontiguousarray(k_bra_list, dtype=float).reshape(-1, 3)
    n_batch = int(k_bras.shape[0])
    if n_batch == 0:
        return []
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    q = _canonical_reciprocal_transfer(
        system, np.asarray(q_transfer, dtype=float).reshape(3)
    )
    # Ket momenta: bra + canonical q (label-independent, exactly like
    # the single-pair builder's `k_ket = k_bra + q`).
    k_kets = k_bras + q[None, :]

    Gq = _rsgdf_shifted_dense_g_mesh(system, q, ke_cutoff)
    Gq2 = (Gq**2).sum(axis=1)
    nz = Gq2 > 1e-12
    Gq = Gq[nz]
    Gq2 = Gq2[nz]
    coul = (4.0 * np.pi) / Gq2 / V
    w_scr = float(omega_screen)
    if w_scr > 0.0:
        coul = coul * (-np.expm1(-Gq2 / (4.0 * w_scr * w_scr)))

    # Batched transient bound: divide the configured G-chunk by the
    # batch width so (n_batch, n_orb, n_orb, chunk_eff) stays within the
    # single-pair (n_orb, n_orb, tail_chunk_g) envelope.
    chunk = max(int(tail_chunk_g) // max(n_batch, 1), 1)
    n_base = int(Gq.shape[0])
    n_orb = int(ao_basis.nbasis)
    n_aux = int(aux_basis.nbasis)
    n_chunks = int((n_base + chunk - 1) // chunk) if n_base else 0

    q_key = tuple(float(value) for value in np.round(q, 14))
    q_metric = (
        _q_metric_cache.get(q_key)
        if _q_metric_cache is not None
        else None
    )

    # Sweep A -- 2c metric M(q), shared by construction (q-only). The
    # per-chunk aux FT is cached here and handed to Sweep B below (same
    # chunk boundaries, same order) instead of being recomputed there:
    # M(q) must finish (all base chunks) before pair_keep_ao can be
    # derived, so the two sweeps can't be fused into one pass, but
    # nothing stops Sweep B from reusing Sweep A's exact per-chunk
    # arrays -- same deterministic function, same input, so this is a
    # pure duplicate-work removal (bit-identical, not reassociated).
    #
    # The cache is BUDGETED, because it is the one piece of this build
    # whose size does not follow ``tail_chunk_g``: it accumulates one
    # (n_aux, chunk) block per chunk, i.e. (n_aux, n_base) in total, so
    # an unbounded cache silently defeats the knob callers use to bound
    # the build. Measured on a 4-batch skew LiH fit: lowering
    # ``tail_chunk_g`` from 16384 to 64 moved peak 15.6 -> 7.2 MiB and
    # then stopped dead on a 6.4 MiB cache floor. Chunks are cached
    # while they fit the budget and recomputed in Sweep B after that, so
    # the common single-chunk case keeps the dedup for free (its cache
    # IS the transient that already existed) and dense multi-chunk runs
    # stay bounded.
    M: Optional[np.ndarray] = None
    aux_ft_cache: list[Optional[np.ndarray]] = []
    if q_metric is None:
        cache_budget = _rsgdf_aux_ft_cache_budget_bytes()
        cached_bytes = 0
        M = np.zeros((n_aux, n_aux), dtype=np.complex128)
        for lo in range(0, n_base, chunk):
            aux_ft_c = rsgdf_aux_fourier_transform(
                aux_basis,
                Gq[lo : lo + chunk],
            )
            if cached_bytes + aux_ft_c.nbytes <= cache_budget:
                aux_ft_cache.append(aux_ft_c)
                cached_bytes += aux_ft_c.nbytes
            else:
                aux_ft_cache.append(None)
            M += (
                aux_ft_c.conj() * coul[lo : lo + chunk][None, :]
            ) @ aux_ft_c.T
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)

    if q_metric is None:
        if M is None:
            raise RuntimeError(
                "build_lpq_bloch_native_fft_shared_q: cache miss did not "
                "build M(q)"
            )
        pair_keep_ao = _rsgdf_fit_pair_keep_mask(
            ao_basis,
            R_g,
            Gq2,
            coul,
            M,
            fit_screen_threshold=float(fit_screen_threshold),
            fit_pair_list=fit_pair_list,
            progress=progress,
        )
    else:
        pair_keep_ao = q_metric.pair_keep_ao

    # Sweep B -- batched 3c tensors T(q; k_ket_i): ONE ket-Bloch pair-FT
    # pass per G-chunk shared across the batch, then a per-k
    # BLAS ZGEMM contraction (n_aux, n_G_chunk) x (n_G_chunk, n_orb^2).
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    # Per-AO-pair store weight handed to the pair-FT kernel: the
    # calibration scale, with the fit-screen mask folded in as an
    # exact 0.0 (which the kernel stores as a true +0.0, matching the
    # masked ASSIGNMENT this replaces rather than a multiply by zero).
    #
    # Both used to be NumPy passes over `pair_ft_c` inside the
    # accumulator below -- single-threaded work on the largest array of
    # the whole build, competing for memory bandwidth with the threads
    # running the kernel that produced it, and measurably ANTI-scaling:
    # 0.537 s at one thread against 0.671 s at sixteen. Doing them in
    # the kernel's own store adds no pass at all -- that store already
    # writes every element, from inside the parallel region -- and is
    # worth 1.5% of the cderi build, which is ~97% of a multi-k GDF SCF
    # (handovers/HANDOVER_GDF_OUTSTANDING.md, 2026-08-05).
    #
    # This is NOT the reverted 2026-08-05 micro-optimisation, which
    # merged the same two operations into one NumPy pass and could not
    # be shown to help. Here neither pass survives.
    #
    # Do not read the profile's 4.428 s / 12.5%-of-wall figure as the
    # cost of these two lines: that is the whole of `_accumulate_chunk`'s
    # own time, of which they are ~15%. The rest is the per-k ZGEMM
    # below and its neighbours.
    pair_weights = pair_scales
    if pair_keep_ao is not None:
        pair_weights = np.where(pair_keep_ao, pair_scales, 0.0)
    T_list = [
        np.zeros((n_aux, n_orb, n_orb), dtype=np.complex128)
        for _ in range(n_batch)
    ]
    if n_chunks > 1:
        _progress_info(
            progress,
            f"GDF shared-q fit build: {n_base} G-points in {n_chunks} "
            f"chunks of {chunk} x {n_batch} ket momenta (batched pair-FT "
            f"transient "
            f"{n_batch * n_orb * n_orb * chunk * 16 / 1e9:.2f} GB)",
        )

    def _accumulate_chunk(
        Gq_chunk: np.ndarray,
        coul_chunk: np.ndarray,
        aux_ft_c: Optional[np.ndarray] = None,
    ) -> None:
        if aux_ft_c is None:
            aux_ft_c = rsgdf_aux_fourier_transform(aux_basis, Gq_chunk)
        aux_w_c = aux_ft_c.conj() * coul_chunk[None, :]
        # The kernel returns the pair FT ALREADY scaled by
        # `pair_weights` (and zeroed where the fit screen masks a
        # pair), fused into its own store — see the note above.
        pair_ft_batch = ao_pair_fourier_transform_bloch_multi(
            ao_basis, Gq_chunk, R_g, k_kets, pair_weights=pair_weights
        )
        n_chunk_g = int(Gq_chunk.shape[0])
        for i_k, pair_ft_c in enumerate(pair_ft_batch):
            # T[P, mn] += aux_w[P, g] . pair_ft[mn, g]^T — a ZGEMM, so
            # the contraction threads through BLAS instead of the
            # single-threaded default-einsum C loop.
            T_list[i_k] += (
                aux_w_c @ pair_ft_c.reshape(n_orb * n_orb, n_chunk_g).T
            ).reshape(n_aux, n_orb, n_orb)

    for chunk_idx, lo in enumerate(range(0, n_base, chunk)):
        cached = aux_ft_cache[chunk_idx] if aux_ft_cache else None
        _accumulate_chunk(
            Gq[lo : lo + chunk], coul[lo : lo + chunk], aux_ft_c=cached
        )

    # High-|G| tail completion, batched identically (see the tail block
    # of build_lpq_bloch_native_fft for the physics rationale). The
    # metric tail accumulates only on a cache miss, matching the
    # single-pair builder.
    if tail_ke_cutoff is not None and float(tail_ke_cutoff) > float(ke_cutoff):
        Gq_tail_all = _rsgdf_shifted_dense_g_mesh(
            system,
            q,
            float(tail_ke_cutoff),
        )
        G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
        tail_norms = np.linalg.norm(Gq_tail_all, axis=1)
        base_boundary_tolerance = (
            0.0
            if float(np.linalg.norm(q)) < 1.0e-14
            else _rsgdf_shifted_sphere_boundary_tolerance(
                system,
                G_base_max,
                q,
            )
        )
        Gq_tail = Gq_tail_all[
            tail_norms > G_base_max + base_boundary_tolerance
        ]
        n_tail = Gq_tail.shape[0]
        for lo in range(0, n_tail, chunk):
            Gqc = Gq_tail[lo : lo + chunk]
            Gq2c = (Gqc**2).sum(axis=1)
            nzc = Gq2c > 1e-12
            Gqc = Gqc[nzc]
            if Gqc.shape[0] == 0:
                continue
            coul_c = (4.0 * np.pi) / Gq2c[nzc] / V
            if w_scr > 0.0:
                coul_c = coul_c * (
                    -np.expm1(-Gq2c[nzc] / (4.0 * w_scr * w_scr))
                )
            tail_aux_ft_c: Optional[np.ndarray] = None
            if q_metric is None:
                if M is None:
                    raise RuntimeError(
                        "build_lpq_bloch_native_fft_shared_q: cache miss "
                        "lost M(q) during tail build"
                    )
                tail_aux_ft_c = rsgdf_aux_fourier_transform(aux_basis, Gqc)
                M += (
                    tail_aux_ft_c.conj() * coul_c[None, :]
                ) @ tail_aux_ft_c.T
            _accumulate_chunk(Gqc, coul_c, aux_ft_c=tail_aux_ft_c)

    if q_metric is None:
        if M is None:
            raise RuntimeError(
                "build_lpq_bloch_native_fft_shared_q: cache miss did not "
                "retain M(q)"
            )
        if float(np.linalg.norm(q)) < 1e-12:
            M = np.real(M)
        M = 0.5 * (M + M.conj().T)

        eigvals, U = np.linalg.eigh(M)
        max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
        if max_eig <= 0:
            raise RuntimeError(
                "build_lpq_bloch_native_fft_shared_q: 2c metric has no "
                f"positive eigenvalue (max eig = {max_eig:.3e}) at q={q}; "
                "aux/mesh invalid."
            )
        keep = _metric_keep_mask(eigvals, linear_dep_thr)
        if not bool(np.any(keep)):
            raise RuntimeError(
                "build_lpq_bloch_native_fft_shared_q: no eigenvalues above "
                f"threshold ({linear_dep_thr:.1e}) at q={q}; "
                "aux basis or threshold invalid."
            )
        q_metric = _RsgdfQMetricCacheEntry(
            eigvals=eigvals,
            eigvecs=U,
            keep=keep,
            pair_keep_ao=pair_keep_ao,
        )
        if _q_metric_cache is not None:
            _q_metric_cache[q_key] = q_metric
    eigvals = q_metric.eigvals
    U = q_metric.eigvecs
    keep = q_metric.keep
    n_kept = int(keep.sum())
    out: list[np.ndarray] = []
    for T in T_list:
        T_flat = T.reshape(T.shape[0], -1)
        if canonical_auxiliary_basis:
            kept_vectors = U[:, keep]
            inverse_sqrt = (
                kept_vectors / np.sqrt(eigvals[keep])[None, :]
            ) @ kept_vectors.conj().T
            Lpq = inverse_sqrt @ T_flat
            n_factor = T.shape[0]
        else:
            Lpq = (
                U[:, keep].conj().T @ T_flat
            ) / np.sqrt(eigvals[keep])[:, None]
            n_factor = n_kept
        out.append(Lpq.reshape(n_factor, T.shape[1], T.shape[2]))
    return out


def build_lpq_bloch_mdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    k_bra: np.ndarray,
    k_ket: np.ndarray,
    *,
    molecule: Optional[Molecule] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    compcell_eta: float = 1.0,
    mdf_ke_cutoff: float = 40.0,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    precision_j2c: Optional[float] = None,
    precision_j3c: Optional[float] = None,
) -> np.ndarray:
    r"""Per-(k_bra, k_ket) Mixed Density Fitting cderi (multi-k MDF).

    The (k_bra, k_ket)-resolved generalisation of :func:`build_lpq_mdf`
    (which is the ``k_bra = k_ket = 0`` case). Returns the **combined
    complex cderi** ``[L_gauss; cderi_pw]`` (shape
    ``(n_kept + n_pw, n_orb, n_orb)``) whose contraction
    ``S_L cderi[L,muν].conj(cderi[L,κl])`` reconstructs the ket-resolved
    periodic ERI block -- exactly what the multi-k J/K builders consume from
    ``lpq_cache[(i, j)]`` (they already take the conjugate).

    Structure (``q = k_ket - k_bra``):

    * **Gaussian part -- real-space libint on the compensated fused basis,
      Bloch-summed at q** (``compute_2c/3c_eri_lattice_blocks`` +
      ``bloch_sum_*`` at ``q``, as in :func:`build_lpq_bloch_compcell`).
      This is q-only, but that is *correct for MDF*: only the steep aux
      survive the PW-dressed-metric orthogonalisation, and a steep core
      fit is local, so its q-only and ket-resolved forms coincide. The
      smooth, inter-cell part the q-only Gaussian misses is carried by the
      ket-resolved PW residual below (this is the whole point of the MDF
      split).
    * **PW residual -- ket-resolved on the q-shifted modest mesh**
      ``{G + q}``: the ket-Bloch AO-pair FT ``r̂_muν(G+q; k_ket)`` (C++
      kernel at ``k_cart = k_ket``) and the q-shifted compensated-aux FT,
      exactly as :func:`build_lpq_bloch_native_fft`. Gives the PW-dressed
      metric ``J̃(q) = M(q) - S_{G+q!=0} coul r_P(G+q)* r_Q(G+q)``, the PW
      projection of the 3c, and the residual cderi ``√coul(G+q).r_muν``.
    * **G=0 self-term V̄r̄ -- only at q = 0** (diagonal pairs ``k_bra =
      k_ket``), where the ``G + q = 0`` mode is dropped from the mesh and
      must be restored (Eq 23). For ``q != 0`` every ``G + q != 0``, so no
      G=0 term arises. The overlap is the k-resolved Bloch overlap at
      ``k_ket``.

    ``precision_j2c`` (default :data:`_MDF_PRECISION_J2C`) sets the
    accuracy the dressed metric ``J̃(q)`` is built to, independently of
    ``rcut_precision``; see :func:`build_lpq_mdf`.

    See :func:`build_lpq_mdf` for the V̄_P closed form and
    ``docs/design_mdf.md`` for the derivation. Γ-diagonal
    (``k_bra=k_ket=0``) reproduces :func:`build_lpq_mdf` to ~1e-12.

    Returns
    -------
    cderi : (n_kept + n_pw, n_orb, n_orb) complex128
        Combined Gaussian + PW residual cderi, libint m-ordering.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import bloch_sum as _bloch_sum
    from ._vibeqc_core import direct_lattice_cells

    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    if molecule is None:
        molecule = system.unit_cell_molecule()
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    k_bra = np.asarray(k_bra, dtype=float).reshape(3)
    k_ket = np.asarray(k_ket, dtype=float).reshape(3)
    q = k_ket - k_bra
    is_gamma = float(np.linalg.norm(q)) < 1e-12
    pw_on = mdf_ke_cutoff is not None and float(mdf_ke_cutoff) > 0.0
    if precision_j2c is None:
        precision_j2c = _MDF_PRECISION_J2C
    if precision_j3c is None:
        precision_j3c = _MDF_PRECISION_J3C

    # --- Gaussian part: compensated fused basis, real-space cell blocks
    #     Bloch-summed at q (q-only; see docstring). No AFT (MDF uses PW). ---
    modrho_aux = make_modrho_aux_basis(aux_basis, molecule)
    chg = make_compensating_basis(modrho_aux, molecule, eta=compcell_eta)
    fused = make_fused_basis(modrho_aux, chg, molecule)
    A = fuse_transform_matrix(modrho_aux, chg)
    n_aux = modrho_aux.nbasis
    n_orb = ao_basis.nbasis

    joint_basis = _mdf_joint_basis(ao_basis, fused, molecule)

    if rcut_strategy is not None:
        from .lattice_screening import RcutStrategy, make_lattice_opts

        if not isinstance(rcut_strategy, RcutStrategy):
            rcut_strategy = RcutStrategy(rcut_strategy)
        lat_opts_2c = make_lattice_opts(
            fused, strategy=rcut_strategy, base_opts=lat_opts, precision=rcut_precision
        )
        lat_opts_3c = make_lattice_opts(
            joint_basis, strategy=rcut_strategy, base_opts=lat_opts,
            precision=rcut_precision,
        )
    else:
        lat_opts_2c = lat_opts
        lat_opts_3c = lat_opts

    if pw_on:
        # J-tilde(q) is built to its own accuracy, not the cell's.
        lat_opts_2c = _mdf_fit_lattice_opts(
            fused, system, lat_opts_2c, precision_j2c, n_aux
        )
        if precision_j3c is not None:
            lat_opts_3c = _mdf_fit_lattice_opts(
                joint_basis, system, lat_opts_3c, precision_j3c, n_aux
            )

    _, m_vecs, m_blocks = compute_2c_eri_lattice_blocks(fused, system, lat_opts_2c)
    _, t_vecs, t_blocks = compute_3c_eri_lattice_blocks(
        ao_basis, fused, system, lat_opts_3c
    )
    M_fused = bloch_sum_2c_eri_blocks(m_vecs, m_blocks, q)
    T_fused = bloch_sum_3c_eri_blocks(t_vecs, t_blocks, q)
    M_fused = 0.5 * (M_fused + M_fused.conj().T)
    if is_gamma:
        T_fused = 0.5 * (T_fused + np.swapaxes(T_fused, 1, 2))
    M = A @ M_fused @ A.T
    M = 0.5 * (M + M.conj().T)
    T = np.einsum("iP,Pmn->imn", A, T_fused, optimize=True)

    # --- PW residual: q-shifted modest mesh, ket-Bloch pair FT. ---
    if pw_on:
        G_all = rsgdf_dense_g_mesh(system, float(mdf_ke_cutoff))
        Gq = G_all + q[None, :]
        Gq2 = (Gq**2).sum(axis=1)
        nz = Gq2 > 1e-12
        Gq = Gq[nz]
        coul = (4.0 * np.pi) / Gq2[nz] / V

        aux_ft_fused = rsgdf_aux_fourier_transform(fused, Gq)
        rho_aux = A @ aux_ft_fused  # (n_aux, n_G)

        cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
        R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
        if R_g.size == 0:
            R_g = np.zeros((1, 3), dtype=float)
        pair_ft = ao_pair_fourier_transform_bloch(ao_basis, Gq, R_g, k_cart=k_ket)
        ao_scales = _ao_scales_for_rsgdf(ao_basis)
        pair_ft = pair_ft * np.outer(ao_scales, ao_scales)[:, :, None]

        aux_w = rho_aux.conj() * coul[None, :]  # (n_aux, n_G)
        pw_2c = aux_w @ rho_aux.T  # (n_aux, n_aux) complex Hermitian
        J_tilde = M - pw_2c
        pw_3c = np.einsum("Pk,mnk->Pmn", aux_w, pair_ft, optimize=True)

        if is_gamma:
            # G=0 self-term (diagonal pair only): V̄_P = -pi^{5/2}.(A@m),
            # r̄_muν = S^{k_ket}_muν / Ω (the k-resolved Bloch overlap).
            m_fused = np.zeros(fused.nbasis)
            _off = 0
            for sh in fused.shells():
                nb = (2 * sh.l + 1) if sh.pure else ((sh.l + 1) * (sh.l + 2) // 2)
                if sh.l == 0:
                    a = np.asarray(sh.exponents, dtype=float)
                    c = np.asarray(sh.coefficients, dtype=float)
                    m_fused[_off : _off + nb] = float(np.sum(c * a ** (-2.5)))
                _off += nb
            vbar = -(np.pi**2.5) * (A @ m_fused)  # (n_aux,)
            S_ao = _bloch_sum(
                compute_overlap_lattice(ao_basis, system, lat_opts), k_ket
            )
            S_ao = 0.5 * (S_ao + S_ao.conj().T)
            T = T - vbar[:, None, None] * (S_ao / V)[None, :, :]

        T_corr = T - pw_3c
        cderi_pw = (
            (pair_ft * np.sqrt(coul)[None, None, :]).transpose(2, 0, 1).copy()
        )
        n_pw = int(Gq.shape[0])
    else:
        J_tilde = M
        T_corr = T
        cderi_pw = np.zeros((0, n_orb, n_orb), dtype=np.complex128)
        n_pw = 0

    if is_gamma:
        J_tilde = np.real(J_tilde)  # q=0 metric is real-symmetric
    J_tilde = 0.5 * (J_tilde + J_tilde.conj().T)

    eigvals, U = np.linalg.eigh(J_tilde)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            f"build_lpq_bloch_mdf: PW-dressed metric has no positive "
            f"eigenvalue (max eig = {max_eig:.3e}) at q={q}."
        )
    thr = _mdf_dressed_metric_threshold(
        linear_dep_thr, n_pw, max_eig, "build_lpq_bloch_mdf"
    )
    keep = _metric_keep_mask(eigvals, thr)
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"build_lpq_bloch_mdf: no eigenvalues above threshold "
            f"({thr:.3e}; max eig = {max_eig:.3e}) at q={q}."
        )
    T_flat = T_corr.reshape(n_aux, n_orb * n_orb)
    L_flat = (U[:, keep].conj().T @ T_flat) / np.sqrt(eigvals[keep])[:, None]
    L_gauss = L_flat.reshape(n_kept, n_orb, n_orb).astype(np.complex128)

    if n_pw:
        return np.concatenate([L_gauss, cderi_pw], axis=0)
    return L_gauss


def build_lpq_bloch_native(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    k_cart: np.ndarray,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-9,
    apply_modrho: bool = True,
    symmetrize_gamma: bool = True,
) -> np.ndarray:
    """Build a phase-assembled Lpq tensor from cell-resolved GDF blocks.

    This is the native multi-k GDF assembly primitive: C++ computes the
    expensive real-space DF blocks once, and this routine applies Bloch
    phases before fitting the AO-pair tensor in the auxiliary metric.

    At ``k_cart = 0`` and with ``symmetrize_gamma=True`` the result is
    equivalent to :func:`build_lpq_native` with ``algorithm="bare"``.
    For nonzero k the return value is complex-valued.
    """
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    k = np.asarray(k_cart, dtype=float).reshape(3)

    _, metric_vectors, metric_blocks = compute_2c_eri_lattice_blocks(
        aux_basis,
        system,
        lat_opts,
    )
    _, tensor_vectors, tensor_blocks = compute_3c_eri_lattice_blocks(
        ao_basis,
        aux_basis,
        system,
        lat_opts,
    )
    M = bloch_sum_2c_eri_blocks(metric_vectors, metric_blocks, k)
    T = bloch_sum_3c_eri_blocks(tensor_vectors, tensor_blocks, k)

    M = 0.5 * (M + M.conj().T)
    if symmetrize_gamma and np.linalg.norm(k) < 1e-14:
        T = 0.5 * (T + np.swapaxes(T, 1, 2))

    n_aux, n_orb, _ = T.shape
    if M.shape != (n_aux, n_aux):
        raise RuntimeError(
            "build_lpq_bloch_native: metric/tensor aux dimensions "
            f"do not match ({M.shape} vs tensor n_aux={n_aux})"
        )

    if apply_modrho:
        alpha = modrho_scales(aux_basis)
        M = (alpha[:, None] * M) * alpha[None, :]
        M = 0.5 * (M + M.conj().T)
        T = T * alpha[:, None, None]

    eigvals, U = np.linalg.eigh(M)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            "build_lpq_bloch_native: 2c metric has no positive "
            f"eigenvalue (max eig = {max_eig:.3e})"
        )
    keep_mask = _metric_keep_mask(
        eigvals, linear_dep_thr, _PLAIN_METRIC_MIN_EIG_FRACTION
    )
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        raise RuntimeError(
            "build_lpq_bloch_native: no eigenvalues above threshold "
            f"({float(linear_dep_thr):.3e}; max eig = {max_eig:.3e})"
        )
    if n_kept < n_aux:
        _logger.info(
            "build_lpq_bloch_native: truncated %d/%d aux modes "
            "(min kept eig = %.3e, threshold = %.3e, max eig = %.3e).",
            n_aux - n_kept,
            n_aux,
            float(eigvals[keep_mask][0]),
            float(linear_dep_thr),
            max_eig,
        )

    T_flat = T.reshape(n_aux, n_orb * n_orb)
    Lpq_flat = (U[:, keep_mask].conj().T @ T_flat) / np.sqrt(eigvals[keep_mask])[
        :, None
    ]
    return Lpq_flat.reshape(n_kept, n_orb, n_orb)


# ============================================================
# RSGDF -- long-range halves, computed analytically in reciprocal space
# ============================================================
#
# Per Ye & Berkelbach 2021 (DOI 10.1063/5.0046617), the LR contribution
# to the periodic Coulomb integral lives naturally in reciprocal space:
#
#   M^LR_PQ(w) = (4pi/Ω) . S_{G!=0} ξ̂_P*(G) . ξ̂_Q(G) .
#                                  exp(-G^2/(4w^2)) / G^2
#
# where ξ̂_P(G) is the analytical Fourier transform of aux primitive
# P. For a (real-spherical-harmonic) primitive r^l Y_lm(r̂) e^{-a r^2}
# centred at R, the FT is the closed form
#
#   F[r^l Y_lm e^{-a r^2}](G; R) =
#     (-i)^l . (pi/a)^(3/2) . (G/(2a))^l . exp(-G^2/(4a)) .
#     Y_lm(Ĝ) . e^{-iG.R}                                     (*)
#
# (Gradshteyn & Ryzhik 6.631; same convention as Obara-Saika.) For
# contracted shells, the FT is just the linear combination of
# primitive FTs. Phase combinations (-i)^{l_P} . (i)^{l_Q} make M
# real-symmetric automatically.
#
# Damping kernel exp(-G^2/(4w^2)) cuts the G-mesh at
# G_max ≈ 2w . √(ln(1/precision)). For w = 0.4, precision = 1e-8 ->
# G_max ≈ 6.8 bohr⁻¹, which on a typical ionic crystal (a ≈ 8 bohr,
# 2pi/a ≈ 0.78) means a (9, 9, 9) ≈ 729 G-vector mesh -- small.
#
# The G = 0 term is omitted under the charge-neutrality convention
# (the divergent monopole-monopole piece cancels against the nuclear
# background; cf. Sun et al. 2017 Sec.III, PySCF
# pyscf.pbc.df.aft._invG2_g0_zero).


# Per-L calibration scale factors for the analytic FT in the RSGDF
# long-range machinery.  libint's BraKet::xs_xs treats shells as
# "scalar charge" (no Y_{Lm} angular factor), while the analytic FT
# uses the conventional AO spherical-harmonic FT.  The scale factor
#   scale_L = √(4pi / (2L+1))    for L > 0,   scale_0 = 1.0
# restores the libint convention so that M^SR(w) + M^LR(w) = M_bare.
# Applied in rsgdf_aux_fourier_transform (inline) and in _ao_scales_for_rsgdf.
_RSGDF_AUX_FT_SCALE: list[float] = [
    1.0,  # L=0: Y_00 already dropped
    np.sqrt(4.0 * np.pi / 3),  # L=1: √(4pi/3) ≈ 2.047
    np.sqrt(4.0 * np.pi / 5),  # L=2: √(4pi/5) ≈ 1.585
    np.sqrt(4.0 * np.pi / 7),  # L=3: √(4pi/7) ≈ 1.340
]


def _ao_scales_for_rsgdf(basis: BasisSet) -> np.ndarray:
    """Per-AO calibration scale factors for RSGDF ao-pair FT.

    Same per-L factors as used in rsgdf_aux_fourier_transform,
    broadcast to each AO in the basis.
    """
    scales = np.ones(basis.nbasis, dtype=np.float64)
    bf = 0
    for shell in basis.shells():
        L = int(shell.l)
        nbf = 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        f = float(np.sqrt(4.0 * np.pi / (2 * L + 1))) if L > 0 else 1.0
        scales[bf : bf + nbf] = f
        bf += nbf
    return scales


# Real-spherical-harmonic Y_lm helpers via scipy. scipy provides
# complex spherical harmonics (sph_harm); we convert to the real
# basis ourselves to match libint's basis-function convention.

from scipy.special import sph_harm_y as _scipy_sph_harm_y


def _real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real spherical harmonic Y_l^m(th, phi).

    Uses the Condon-Shortley convention with sign-removed real
    combinations:
        m  > 0:  Y_l^m  = √2 . (-1)^m . Re Y_l^|m|_complex
        m  < 0:  Y_l^m  = √2 . (-1)^m . Im Y_l^|m|_complex
        m == 0:  Y_l^0  = Y_l^0_complex (real already)

    This matches libint's spherical AO ordering for L=1 min{-1,0,+1} =
    (py, pz, px) (verified empirically; see
    periodic_rhf_gdf._WITHIN_SHELL_PERM_BY_L). Higher-L libint
    convention (m=-l..+l) maps directly to the formulas above.

    th is the polar angle (from +z), phi the azimuthal (from +x toward
    +y). Both broadcastable.

    Implementation note: scipy >= 1.15 deprecated ``sph_harm`` in
    favour of ``sph_harm_y(n, m, theta, phi)`` with reversed argument
    order from the old API. We use the new API and pass arguments
    accordingly.
    """
    if m == 0:
        # Y_l^0 is real; scipy returns a complex value with zero imag.
        Y = _scipy_sph_harm_y(l, 0, theta, phi)
        return np.real(Y)
    abs_m = abs(m)
    Y_complex = _scipy_sph_harm_y(l, abs_m, theta, phi)
    sign = (-1.0) ** abs_m
    sqrt2 = np.sqrt(2.0)
    if m > 0:
        return sqrt2 * sign * np.real(Y_complex)
    return sqrt2 * sign * np.imag(Y_complex)


def _reject_transverse_collapse(dim: int, what: str) -> None:
    """Refuse to hand a ``dim < 3`` reciprocal mesh to a Coulomb kernel.

    Both G-mesh builders below span only the *periodic* axes (``axis < dim``) and
    pin every non-periodic axis at the single point ``G_perp = 0``. That is the
    correct support for a Bloch phase factor, but it is **not** a usable support
    for the Coulomb kernel ``4π/|G+q|²/V`` that every caller multiplies it by:
    dropping the transverse Fourier components replaces each AO-pair density by
    its transverse average, so the kernel degenerates from ``1/r`` into a uniform
    sheet term ``∝ 1/V`` that *vanishes* as the vacuum padding grows.

    Measured on an H₂ chain (dim=1, sto-3g, 6-bohr period), 2026-07-10: the
    resulting four-center obeys ``|g|_max · V = 92.46`` exactly across transverse
    vacuum ``D = 12/18/24`` bohr -- pure ``1/V``, no transverse structure -- while
    the mixed-boundary wire four-center is ``D``-invariant at ``1.895512``. The
    two disagree by 14 % of the wire scale even after removing the best-fit
    ``c·S⊗S`` gauge constant, so this is not a gauge choice. At SCF level the
    collapsed kernel gave ``-6.917 -> -6.568`` Ha as ``D`` went ``12 -> 30`` bohr,
    against the wire kernel's ``-1.32206067`` Ha at every ``D``.

    A ``dim == 1`` cell must use the wire kernel
    (:mod:`vibeqc.periodic.ccm.lowd_four_center`); a ``dim == 2`` cell has no
    mixed-boundary four-center yet (the slab needs the partial in-plane FT of
    ``docs/aiccm2026dev_a_lowd_greens.md`` §9). Callers that genuinely want the
    collapsed mesh for a **non-Coulomb** purpose pass
    ``allow_transverse_collapse=True`` and take responsibility for the kernel.
    """
    if dim == 3:
        return
    raise NotImplementedError(
        f"{what}: refusing to build a transverse-collapsed reciprocal mesh for a "
        f"dim = {dim} cell. Every non-periodic axis is pinned at G_perp = 0, so the "
        "Coulomb kernel 4*pi/|G+q|^2 that consumers apply to this mesh is a "
        "transverse-uniform sheet term proportional to 1/V, not 1/r -- the "
        "electron-electron repulsion vanishes as the vacuum padding grows "
        "(measured: |g|_max * V constant to 4 s.f. across D = 12/18/24 bohr). "
        "Use the mixed-boundary wire kernel for dim == 1 "
        "(vibeqc.periodic.ccm.lowd_scf.run_ccm_rhf_wire); dim == 2 needs the slab "
        "partial in-plane FT (docs/aiccm2026dev_a_lowd_greens.md section 9) and is "
        "not implemented. Pass allow_transverse_collapse=True only for a "
        "non-Coulomb use of the mesh."
    )


def _slab_truncation_geometry(
    system: PeriodicSystem,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the orthogonal bookkeeping embedding of a genuine slab."""
    if int(getattr(system, "dim", 3)) != 2:
        raise ValueError("slab truncation requires a genuine dim=2 slab")

    lattice = np.asarray(system.lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("slab truncation requires a finite 3x3 lattice")
    a1, a2, a3 = lattice.T
    plane_normal = np.cross(a1, a2)
    area = float(np.linalg.norm(plane_normal))
    a3_length = float(np.linalg.norm(a3))
    if area == 0.0 or a3_length == 0.0:
        raise ValueError("slab truncation requires non-degenerate slab vectors")
    normal = plane_normal / area
    transverse_a3 = a3 - np.dot(a3, normal) * normal
    if float(np.linalg.norm(transverse_a3)) > 1e-10 * a3_length:
        raise ValueError(
            "slab truncation requires lattice[:, 2] to be normal to the "
            "periodic slab plane"
        )
    return lattice, normal, a3_length


def _slab_probe_charge_madelung(
    system: PeriodicSystem,
    *,
    alpha: float = 0.45,
    precision: float = 1e-12,
) -> float:
    """Return the private 2D probe-charge constant for slab exchange.

    The value is ``xi_2d = -2 E`` for one unit point charge and a co-located
    neutralising sheet in the rigorous 2D Ewald gauge. It is the slab analogue
    of the 3D ``exxdiv='ewald'`` probe-charge constant: a raw finite-k exchange
    block receives ``xi_2d * S @ D @ S``. The helper remains private until the
    signed/charge-constrained slab metric and multi-k exchange gates close.
    """
    from ._vibeqc_core import (
        EwaldOptions,
        ewald_2d_point_charge_energy_with_background,
    )

    lattice, _, _ = _slab_truncation_geometry(system)
    eta = float(alpha)
    tolerance = float(precision)
    if eta <= 0.0:
        raise ValueError("_slab_probe_charge_madelung requires alpha > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError(
            "_slab_probe_charge_madelung requires 0 < precision < 1"
        )
    opts = EwaldOptions()
    opts.alpha = eta
    opts.real_cutoff_bohr = float(np.sqrt(-np.log(tolerance)) / eta)
    opts.tolerance = tolerance
    energy = ewald_2d_point_charge_energy_with_background(
        lattice,
        np.zeros((3, 1), dtype=float),
        np.ones(1, dtype=float),
        0.0,
        opts,
    )
    return -2.0 * float(energy)


def _slab_probe_charge_madelung_for_kmesh(
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
    *,
    alpha: float = 0.45,
    precision: float = 1e-12,
) -> float:
    """Return the 2D probe constant of a finite k-mesh's BvK supercell."""
    mesh_values = np.asarray(mesh, dtype=float)
    if (
        mesh_values.shape != (3,)
        or not np.all(np.isfinite(mesh_values))
        or not np.all(mesh_values == np.rint(mesh_values))
        or np.any(mesh_values < 1)
    ):
        raise ValueError(
            "_slab_probe_charge_madelung_for_kmesh requires three positive "
            "integer mesh dimensions"
        )
    mesh_array = np.rint(mesh_values).astype(int)
    if int(mesh_array[2]) != 1:
        raise ValueError(
            "_slab_probe_charge_madelung_for_kmesh requires a slab mesh "
            "with n3 == 1"
        )
    lattice, _, _ = _slab_truncation_geometry(system)
    super_lattice = lattice.copy()
    super_lattice[:, 0] *= int(mesh_array[0])
    super_lattice[:, 1] *= int(mesh_array[1])
    super_system = PeriodicSystem(2, super_lattice, [])
    return _slab_probe_charge_madelung(
        super_system, alpha=alpha, precision=precision
    )


def _slab_truncated_coulomb_kernel(
    system: PeriodicSystem,
    G_vectors: np.ndarray,
    *,
    q_cart: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the finite-cell reciprocal kernel for a genuine 2D slab.

    This is the private Inc-D foundation for the 2D GDF metric. It evaluates
    the Wigner-Seitz-truncated slab interaction, including its finite ``G=0``
    gauge value. It deliberately does *not* divide by the embedding-cell
    volume: GDF contractions apply that reciprocal-sum normalization at their
    call site.

    ``system.lattice[:, 2]`` is allowed only as the synthesized, normal
    bookkeeping vector produced by :func:`vibeqc.build.slab_2d`. A skewed
    third vector is rejected because the closed form below assumes truncation
    over ``[-L/2, L/2]`` along the slab normal. Likewise, a Bloch transfer
    ``q`` must lie in the periodic plane.

    The public slab-GDF routes remain fail-closed while this helper is private;
    wiring it into the metric requires charge/gauge-consistent 2c and 3c
    contractions plus J/K parity coverage.

    References
    ----------
    Sundararaman and Arias, Phys. Rev. B 87, 165122 (2013), Eq. (A6),
    DOI 10.1103/PhysRevB.87.165122.
    """
    _, normal, a3_length = _slab_truncation_geometry(system)

    G = np.asarray(G_vectors, dtype=float)
    if G.ndim != 2 or G.shape[1] != 3 or not np.all(np.isfinite(G)):
        raise ValueError(
            "_slab_truncated_coulomb_kernel requires finite G_vectors with "
            "shape (n_G, 3)"
        )
    if q_cart is None:
        q = np.zeros(3, dtype=float)
    else:
        q = np.asarray(q_cart, dtype=float)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError(
                "_slab_truncated_coulomb_kernel requires finite q_cart with "
                "shape (3,)"
            )
    q_normal = float(np.dot(q, normal))
    if abs(q_normal) > 1e-10 * max(1.0, float(np.linalg.norm(q))):
        raise ValueError(
            "_slab_truncated_coulomb_kernel requires q_cart to lie in the "
            "periodic slab plane"
        )

    Gq = G + q[None, :]
    Gq2 = np.einsum("Gi,Gi->G", Gq, Gq)
    G_normal = Gq @ normal
    G_plane_vectors = Gq - G_normal[:, None] * normal[None, :]
    G_plane = np.linalg.norm(G_plane_vectors, axis=1)
    half_length = 0.5 * a3_length

    kernel = np.empty(Gq2.shape, dtype=float)
    zero = Gq2 == 0.0
    nonzero = ~zero

    # Sundararaman-Arias Eq. (A6):
    # K(G) = 4*pi/G^2 * [1 - cos(G_n*L/2)*exp(-G_parallel*L/2)], G != 0;
    # K(0) = -pi*L^2/2 after removing the 2*pi/G_parallel singular part.
    # Split the numerator into two positive pieces to avoid cancellation near
    # the origin: 1-e^-x + e^-x(1-cos y).
    decay = np.exp(-G_plane[nonzero] * half_length)
    numerator = -np.expm1(-G_plane[nonzero] * half_length)
    numerator += decay * (1.0 - np.cos(G_normal[nonzero] * half_length))
    kernel[nonzero] = 4.0 * np.pi * numerator / Gq2[nonzero]
    kernel[zero] = -0.5 * np.pi * a3_length * a3_length
    return kernel


def _slab_truncated_gdf_contractions(
    system: PeriodicSystem,
    G_vectors: np.ndarray,
    aux_ft: np.ndarray,
    pair_ft: np.ndarray,
    *,
    q_cart: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract private slab-GDF 2c and 3c tensors on one reciprocal mesh.

    ``aux_ft`` and ``pair_ft`` must already be evaluated at the shifted points
    ``G_vectors + q_cart``. This helper supplies only the 2D-truncated Coulomb
    weights and the embedding-volume normalization. In particular, the finite
    slab zero mode is retained; it must not be replaced by the bulk GDF
    convention that drops ``G=0``.

    No metric factorization is performed here. The public GDF route remains
    closed until the charge-constrained auxiliary fit and the matching J/K
    gauge are validated end to end.
    """
    G = np.asarray(G_vectors, dtype=float)
    aux = np.asarray(aux_ft, dtype=np.complex128)
    pairs = np.asarray(pair_ft, dtype=np.complex128)
    if G.ndim != 2 or G.shape[1] != 3:
        raise ValueError(
            "_slab_truncated_gdf_contractions requires G_vectors with shape "
            "(n_G, 3)"
        )
    n_G = G.shape[0]
    if aux.ndim != 2 or aux.shape[1] != n_G:
        raise ValueError(
            "_slab_truncated_gdf_contractions requires aux_ft with shape "
            "(n_aux, n_G)"
        )
    if pairs.ndim != 3 or pairs.shape[2] != n_G:
        raise ValueError(
            "_slab_truncated_gdf_contractions requires pair_ft with shape "
            "(n_ao, n_ao, n_G)"
        )
    if pairs.shape[0] != pairs.shape[1]:
        raise ValueError(
            "_slab_truncated_gdf_contractions requires square AO-pair axes"
        )

    cell_volume = float(
        abs(np.linalg.det(np.asarray(system.lattice, dtype=float)))
    )
    if cell_volume == 0.0:
        raise ValueError(
            "_slab_truncated_gdf_contractions requires a nonzero cell volume"
        )
    weights = (
        _slab_truncated_coulomb_kernel(system, G, q_cart=q_cart) / cell_volume
    )

    metric = (aux.conj() * weights[None, :]) @ aux.T
    metric = 0.5 * (metric + metric.conj().T)
    three_center = np.einsum(
        "PG,G,mnG->Pmn", aux.conj(), weights, pairs, optimize=True
    )
    return metric, three_center


class _SlabGdfMetricFit(NamedTuple):
    """Signed inverse-metric factors for the private slab-GDF route."""

    factors: np.ndarray
    signs: np.ndarray
    eigenvalues: np.ndarray


class _SlabGdfBlochFit(NamedTuple):
    """Private slab-GDF block plus its Coulomb and exchange gauge metadata."""

    factors: np.ndarray
    metric_signs: np.ndarray
    metric_eigenvalues: np.ndarray
    q_cart: np.ndarray
    includes_finite_coulomb_zero_mode: bool
    primitive_exchange_probe_charge_madelung: float


def _factor_slab_truncated_gdf_metric(
    metric: np.ndarray,
    three_center: np.ndarray,
    *,
    linear_dep_thr: float = 1e-10,
) -> _SlabGdfMetricFit:
    """Return a signed inverse-square-root fit of a private slab GDF metric.

    The hard slab cutoff has a negative reciprocal ``G=0`` coefficient, so
    its finite-cell auxiliary metric can be indefinite. The signed factor
    contract retains ``sign(lambda)`` separately while scaling each retained
    mode by ``1/sqrt(abs(lambda))``. A consumer must include that signature in
    every J and K contraction; treating ``factors`` as an ordinary positive
    Gram factor is incorrect.

    Modes with ``abs(lambda) <= linear_dep_thr`` are dropped. The full metric
    spectrum is retained explicitly for the Inc-D padding/mesh diagnostics.
    """
    M = np.asarray(metric, dtype=np.complex128)
    T = np.asarray(three_center, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            "_factor_slab_truncated_gdf_metric requires a square 2c metric"
        )
    if T.ndim != 3 or T.shape[0] != M.shape[0] or T.shape[1] != T.shape[2]:
        raise ValueError(
            "_factor_slab_truncated_gdf_metric requires three_center with "
            "shape (n_aux, n_ao, n_ao)"
        )
    threshold = float(linear_dep_thr)
    if threshold <= 0.0:
        raise ValueError(
            "_factor_slab_truncated_gdf_metric requires linear_dep_thr > 0"
        )
    hermitian_error = float(np.max(np.abs(M - M.conj().T))) if M.size else 0.0
    matrix_scale = float(np.max(np.abs(M))) if M.size else 0.0
    if hermitian_error > 1e-11 * max(matrix_scale, 1.0):
        raise ValueError(
            "_factor_slab_truncated_gdf_metric requires a Hermitian metric; "
            f"max residual={hermitian_error:.3e}"
        )
    M = 0.5 * (M + M.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    max_abs_eigenvalue = (
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    )
    if max_abs_eigenvalue <= threshold:
        raise RuntimeError(
            "slab-truncated GDF metric has no eigenvalue outside the linear "
            "dependency threshold "
            f"({threshold:.1e}); max_abs={max_abs_eigenvalue:.3e}"
        )
    keep = np.abs(eigenvalues) > threshold
    retained = eigenvalues[keep]
    T_flat = T.reshape(T.shape[0], -1)
    fitted = (eigenvectors[:, keep].conj().T @ T_flat) / np.sqrt(
        np.abs(retained)
    )[:, None]
    fitted = fitted.reshape(int(np.count_nonzero(keep)), T.shape[1], T.shape[2])
    signs = np.sign(retained).astype(np.int8)
    return _SlabGdfMetricFit(fitted, signs, eigenvalues)


def _reciprocal_index_box_per_axis(
    lattice: np.ndarray, g_reach: float, dim: int
) -> list[int]:
    """Per-axis integer half-width of the reciprocal box enclosing the
    ``|G (+ q)| <= g_reach`` ball.

    The integer index of ``G = sum_i n_i b_i`` is ``n_i = G . a_i / (2 pi)``
    (biorthogonality, ``b = 2 pi a^{-T}``), so ``|n_i| <= g_reach . |a_i| /
    (2 pi)`` is the EXACT per-axis bound. ``|a_i|`` are COLUMN norms --
    vibe-qc lattice matrices carry the direct lattice vectors in columns (the
    same convention ``b = 2 pi inv(a).T`` and ``G = idx @ b.T`` rely on).

    The legacy single ``n_max = ceil(g_reach / min|b_i|)`` bound is equivalent
    ONLY for orthogonal cells (``|a_i||b_i| = 2 pi``). For oblique / non-
    symmetric lattices ``|a_i||b_i| > 2 pi``, so ``min|b_i|`` undercounts every
    axis longer than the shortest reciprocal vector and the box silently CLIPS
    G-points near the sphere boundary (measured ~18% missing on the C-diamond
    (2,1,1) cell at ke=60), breaking mesh point-group closure and under-
    converging the fit at its declared cutoff. This single source of truth
    exists because the fix first landed only in ``rsgdf_dense_g_mesh``
    (2026-07-10) and its two siblings (``rsgdf_g_mesh``, ``_aft_g_mesh``) kept
    the clipping bound until the 2026-07-14 GDF audit; routing all three here
    prevents a fourth divergence.
    """
    a = np.asarray(lattice, dtype=float)
    a_norms = np.linalg.norm(a, axis=0)  # |a_i| (columns are lattice vectors)
    return [
        int(np.ceil(g_reach * float(a_norms[axis]) / (2.0 * np.pi))) + 1
        if axis < dim
        else 0
        for axis in range(3)
    ]


def _slab_truncated_dense_g_mesh(
    system: PeriodicSystem,
    ke_cutoff: float,
    *,
    q_cart: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return full ``G+q`` support for the private truncated-slab kernel.

    Unlike :func:`rsgdf_dense_g_mesh` on a low-dimensional system, this mesh
    intentionally enumerates the synthesized normal reciprocal axis as well as
    the two physical periodic axes. Those normal Fourier modes resolve the
    slab's finite thickness; collapsing them to ``G_normal=0`` produces the
    transverse-uniform sheet interaction rejected by
    :func:`_reject_transverse_collapse`.

    The returned vectors fill one physical kinetic-energy sphere
    ``|G+q| <= sqrt(2*ke_cutoff)`` and include the exact zero point at Gamma so
    :func:`_slab_truncated_coulomb_kernel` can apply its finite zero-mode gauge.
    ``q`` is canonicalized modulo the embedding reciprocal lattice and must lie
    in the slab plane.
    """
    if ke_cutoff <= 0.0:
        raise ValueError(
            "_slab_truncated_dense_g_mesh requires ke_cutoff > 0; "
            f"got {ke_cutoff}"
        )
    lattice, normal, _ = _slab_truncation_geometry(system)
    q_input = np.zeros(3) if q_cart is None else np.asarray(q_cart, dtype=float)
    if q_input.shape != (3,) or not np.all(np.isfinite(q_input)):
        raise ValueError(
            "_slab_truncated_dense_g_mesh requires finite q_cart with shape (3,)"
        )
    q_normal = float(np.dot(q_input, normal))
    if abs(q_normal) > 1e-10 * max(1.0, float(np.linalg.norm(q_input))):
        raise ValueError(
            "_slab_truncated_dense_g_mesh requires q_cart to lie in the "
            "periodic slab plane"
        )
    q = _canonical_reciprocal_transfer(system, q_input)

    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    p_max = float(np.sqrt(2.0 * float(ke_cutoff)))
    g_reach = p_max + float(np.linalg.norm(q))
    n_per_axis = _reciprocal_index_box_per_axis(lattice, g_reach, 3)
    grids = [np.arange(-n, n + 1) for n in n_per_axis]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    indices = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(
        float
    )
    shifted = indices @ reciprocal.T + q[None, :]
    norms = np.linalg.norm(shifted, axis=1)
    boundary_tolerance = _rsgdf_shifted_sphere_boundary_tolerance(
        system, p_max, q
    )
    keep = norms <= p_max + boundary_tolerance
    shifted = shifted[keep]
    norms = norms[keep]

    sort_vectors = np.round(shifted, 12)
    sort_norms = np.round(norms, 12)
    order = np.lexsort(
        (
            sort_vectors[:, 2],
            sort_vectors[:, 1],
            sort_vectors[:, 0],
            sort_norms,
        )
    )
    return shifted[order]


def _build_lpq_bloch_slab_truncated(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    k_bra: np.ndarray,
    k_ket: np.ndarray,
    *,
    ke_cutoff: float = 200.0,
    lat_opts: Optional[LatticeSumOptions] = None,
    linear_dep_thr: float = 1e-10,
    g_chunk: int = 65536,
) -> _SlabGdfBlochFit:
    r"""Build one private Bloch cderi block with the truncated-slab kernel.

    This is the Inc-D integration seam between the full three-axis reciprocal
    support, the ket-Bloch AO-pair Fourier transform, and the signed slab
    metric factorization. The returned block is intentionally private and is
    not wired to a driver: fitted J/K parity and synthesized-``a3`` invariance
    must be established before any public slab GDF guard can be removed.

    ``aux_basis`` is expected to use the same modrho convention as the bulk
    dense-FFT builder. The reciprocal mesh contains the already shifted
    physical vectors ``p = G + q``, where ``q = k_ket - k_bra``; consequently
    each chunk is passed to the slab kernel without applying ``q`` a second
    time. The AO pair uses the ket momentum in the Bloch phase, matching
    :func:`build_lpq_bloch_native_fft`.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import direct_lattice_cells

    _, normal, _ = _slab_truncation_geometry(system)
    bra = np.asarray(k_bra, dtype=float)
    ket = np.asarray(k_ket, dtype=float)
    if (
        bra.shape != (3,)
        or ket.shape != (3,)
        or not np.all(np.isfinite(bra))
        or not np.all(np.isfinite(ket))
    ):
        raise ValueError(
            "_build_lpq_bloch_slab_truncated requires finite k_bra and "
            "k_ket with shape (3,)"
        )
    momentum_scale = max(1.0, float(np.linalg.norm(bra)), float(np.linalg.norm(ket)))
    if max(abs(float(np.dot(bra, normal))), abs(float(np.dot(ket, normal)))) > (
        1e-10 * momentum_scale
    ):
        raise ValueError(
            "_build_lpq_bloch_slab_truncated requires Bloch momenta in the "
            "periodic slab plane"
        )

    q = _canonical_reciprocal_transfer(system, ket - bra)
    ket = bra + q
    physical_vectors = _slab_truncated_dense_g_mesh(
        system, ke_cutoff, q_cart=q
    )
    if lat_opts is None:
        lat_opts = LatticeSumOptions()
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    lattice_vectors = np.array([list(cell.r_cart) for cell in cells], dtype=float)
    if lattice_vectors.size == 0:
        lattice_vectors = np.zeros((1, 3), dtype=float)

    n_aux = int(aux_basis.nbasis)
    n_ao = int(ao_basis.nbasis)
    metric = np.zeros((n_aux, n_aux), dtype=np.complex128)
    three_center = np.zeros((n_aux, n_ao, n_ao), dtype=np.complex128)
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    chunk = max(int(g_chunk), 1)
    for lo in range(0, physical_vectors.shape[0], chunk):
        vectors = physical_vectors[lo : lo + chunk]
        aux_ft = rsgdf_aux_fourier_transform(aux_basis, vectors)
        pair_ft = ao_pair_fourier_transform_bloch(
            ao_basis, vectors, lattice_vectors, k_cart=ket
        )
        pair_ft *= pair_scales[:, :, None]
        metric_chunk, three_center_chunk = _slab_truncated_gdf_contractions(
            system, vectors, aux_ft, pair_ft
        )
        metric += metric_chunk
        three_center += three_center_chunk

    if float(np.linalg.norm(q)) < 1e-12:
        metric = np.real(metric)
    metric = 0.5 * (metric + metric.conj().T)
    metric_fit = _factor_slab_truncated_gdf_metric(
        metric, three_center, linear_dep_thr=linear_dep_thr
    )
    return _SlabGdfBlochFit(
        factors=metric_fit.factors,
        metric_signs=metric_fit.signs,
        metric_eigenvalues=metric_fit.eigenvalues,
        q_cart=q,
        includes_finite_coulomb_zero_mode=True,
        primitive_exchange_probe_charge_madelung=_slab_probe_charge_madelung(
            system
        ),
    )


def _contract_slab_gdf_gamma(
    fit: _SlabGdfBlochFit,
    density: np.ndarray,
    overlap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract one private signed Gamma slab-GDF block into J and K.

    The Coulomb factors already include the finite truncated-kernel zero mode.
    Exchange additionally receives the rigorous 2D probe-charge correction.
    This helper is deliberately Gamma-only while multi-k weights and singular
    transfer ownership remain unversioned follow-ups (G-PBC-004's energy
    envelope is resolved; see `handovers/HANDOVER_GATED_ITEMS.md`).
    """
    if float(np.linalg.norm(fit.q_cart)) > 1e-12:
        raise ValueError("_contract_slab_gdf_gamma requires q_cart == 0")
    if not fit.includes_finite_coulomb_zero_mode:
        raise ValueError(
            "_contract_slab_gdf_gamma requires the finite slab Coulomb "
            "zero mode"
        )
    factors = np.asarray(fit.factors, dtype=np.complex128)
    signs = np.asarray(fit.metric_signs, dtype=float)
    D = np.asarray(density, dtype=np.complex128)
    S = np.asarray(overlap, dtype=np.complex128)
    if factors.ndim != 3 or factors.shape[1] != factors.shape[2]:
        raise ValueError(
            "_contract_slab_gdf_gamma requires factors with shape "
            "(n_fit, n_ao, n_ao)"
        )
    n_ao = factors.shape[1]
    if signs.shape != (factors.shape[0],):
        raise ValueError(
            "_contract_slab_gdf_gamma requires one metric sign per factor"
        )
    if D.shape != (n_ao, n_ao) or S.shape != (n_ao, n_ao):
        raise ValueError(
            "_contract_slab_gdf_gamma requires square density and overlap "
            "matrices matching the AO dimension"
        )

    rho = np.einsum("Lij,ij->L", factors.conj(), D, optimize=True)
    coulomb = np.einsum(
        "L,L,Lij->ij", signs, rho, factors, optimize=True
    )
    exchange = np.einsum(
        "L,Lmk,kl,Lnl->mn",
        signs,
        factors,
        D,
        factors.conj(),
        optimize=True,
    )
    exchange += fit.primitive_exchange_probe_charge_madelung * (S @ D @ S)
    coulomb = 0.5 * (coulomb + coulomb.conj().T)
    exchange = 0.5 * (exchange + exchange.conj().T)
    return np.real_if_close(coulomb), np.real_if_close(exchange)


def _contract_slab_gdf_multik_exchange(
    fits: dict[tuple[int, int], _SlabGdfBlochFit],
    densities: list[np.ndarray],
    overlaps: list[np.ndarray],
    weights: np.ndarray,
    *,
    bvk_probe_charge_madelung: float,
) -> list[np.ndarray]:
    """Contract private signed slab-GDF exchange on one finite k-mesh.

    The source-density k-point weights enter the ordinary fitted exchange.
    The exchange-divergence correction uses the probe constant of the full
    Born-von-Karman supercell implied by the mesh, not the primitive-cell
    constant carried by each individual Bloch fit.
    """
    n_k = len(densities)
    if n_k == 0 or len(overlaps) != n_k:
        raise ValueError(
            "_contract_slab_gdf_multik_exchange requires matching nonempty "
            "density and overlap lists"
        )
    k_weights = np.asarray(weights, dtype=float)
    if k_weights.shape != (n_k,) or not np.all(np.isfinite(k_weights)):
        raise ValueError(
            "_contract_slab_gdf_multik_exchange requires one finite weight "
            "per k-point"
        )
    if not np.isclose(float(np.sum(k_weights)), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            "_contract_slab_gdf_multik_exchange requires weights summing to 1"
        )
    xi = float(bvk_probe_charge_madelung)
    if not np.isfinite(xi):
        raise ValueError(
            "_contract_slab_gdf_multik_exchange requires a finite BvK probe "
            "constant"
        )

    exchange_blocks: list[np.ndarray] = []
    for i in range(n_k):
        D_i = np.asarray(densities[i], dtype=np.complex128)
        S_i = np.asarray(overlaps[i], dtype=np.complex128)
        if D_i.ndim != 2 or D_i.shape[0] != D_i.shape[1]:
            raise ValueError(
                "_contract_slab_gdf_multik_exchange requires square density "
                "matrices"
            )
        if S_i.shape != D_i.shape:
            raise ValueError(
                "_contract_slab_gdf_multik_exchange requires overlap and "
                "density matrices with matching shapes"
            )
        exchange = np.zeros_like(D_i)
        for j in range(n_k):
            try:
                fit = fits[(i, j)]
            except KeyError as exc:
                raise ValueError(
                    "_contract_slab_gdf_multik_exchange is missing fit block "
                    f"({i}, {j})"
                ) from exc
            if not fit.includes_finite_coulomb_zero_mode:
                raise ValueError(
                    "_contract_slab_gdf_multik_exchange requires the finite "
                    "slab Coulomb zero mode"
                )
            factors = np.asarray(fit.factors, dtype=np.complex128)
            signs = np.asarray(fit.metric_signs, dtype=float)
            D_j = np.asarray(densities[j], dtype=np.complex128)
            if factors.shape[1:] != D_j.shape or signs.shape != (
                factors.shape[0],
            ):
                raise ValueError(
                    "_contract_slab_gdf_multik_exchange received an "
                    "incompatible signed fit block"
                )
            exchange += float(k_weights[j]) * np.einsum(
                "L,Lpr,rs,Lqs->pq",
                signs,
                factors,
                D_j,
                factors.conj(),
                optimize=True,
            )
        exchange += xi * (S_i @ D_i @ S_i)
        exchange = 0.5 * (exchange + exchange.conj().T)
        exchange_blocks.append(np.real_if_close(exchange))
    return exchange_blocks


def rsgdf_g_mesh(
    system: PeriodicSystem,
    omega: float,
    *,
    precision: float = 1e-8,
    pad_factor: float = 1.5,
    allow_transverse_collapse: bool = False,
) -> np.ndarray:
    """Reciprocal-lattice G-vectors for the RSGDF LR sum.

    Generates the integer reciprocal-lattice points (n1, n2, n3) such
    that the corresponding Cartesian G satisfies |G| <= G_max, where
    G_max is sized by the long-range damping kernel
    ``exp(-G^2/(4w^2))`` and the requested precision:

        G_max  =  pad_factor . 2w . √(ln(1/precision))

    The factor 2 inside the sqrt comes from the half-width of the
    Gaussian damping; ``pad_factor`` (default 1.5) gives a safety
    margin against quadrature error in the reciprocal-lattice sum.

    G = 0 is included in the returned mesh; callers responsible for
    the LR Coulomb integral skip it (the 1/G^2 weight is divergent and
    the analytic continuation is handled by the charge-neutral
    background convention).

    Returns
    -------
    G_vectors : (n_G, 3) float64
        Cartesian reciprocal-lattice vectors, sorted by |G| ascending.
    """
    if omega <= 0:
        raise ValueError(f"rsgdf_g_mesh: omega must be > 0, got {omega}")

    G_max = pad_factor * 2.0 * omega * float(np.sqrt(np.log(1.0 / precision)))

    # Reciprocal lattice vectors b_i = 2pi . a^{-T}_i for the direct
    # lattice matrix a whose columns are direct lattice vectors.
    a = np.asarray(system.lattice, dtype=float)
    if a.shape != (3, 3):
        raise RuntimeError(
            f"rsgdf_g_mesh: PeriodicSystem.lattice must be 3x3, got {a.shape}"
        )
    b = 2.0 * np.pi * np.linalg.inv(a).T  # columns are reciprocal vectors

    # Range of integer multipliers. For 1D/2D systems only the active
    # primitive directions are periodic. The inactive columns are vacuum
    # embedding axes and must not seed a three-dimensional reciprocal
    # sphere: a 40-50 bohr padding direction has a tiny |b_i| and would
    # inflate the RSGDF cderi mesh by orders of magnitude, presenting as
    # a hang at "Building per-pair Lpq cache" for otherwise small chains
    # and slabs. 3D systems keep the historical full reciprocal sphere.
    dim = int(getattr(system, "dim", 3))
    if dim not in (1, 2, 3):
        raise ValueError(f"rsgdf_g_mesh: dim must be 1, 2, or 3; got {dim}")
    if not allow_transverse_collapse:
        _reject_transverse_collapse(dim, "rsgdf_g_mesh")
    # Per-axis reciprocal box (column-norm bound). The previous single
    # n_max = ceil(G_max / min|b_i|) clipped the |G| <= G_max ball on oblique
    # cells; inactive (vacuum) directions still collapse to n = 0 via the
    # axis < dim guard. See _reciprocal_index_box_per_axis.
    n_per_axis = _reciprocal_index_box_per_axis(a, G_max, dim)

    grids = [
        np.arange(-n, n + 1) if axis < dim else np.array([0])
        for axis, n in enumerate(n_per_axis)
    ]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    indices = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(float)
    G_vectors = indices @ b.T  # (N, 3) Cartesian, G = n1*b1+n2*b2+n3*b3
    G_norms = np.linalg.norm(G_vectors, axis=1)
    keep = G_norms <= G_max
    G_kept = G_vectors[keep]
    norms_kept = G_norms[keep]
    order = np.argsort(norms_kept)
    return G_kept[order]


def _canonical_reciprocal_transfer(
    system: PeriodicSystem,
    q_cart: np.ndarray,
) -> np.ndarray:
    """Return one deterministic fractional ``[-1/2, 1/2)`` q label.

    This label selection is safe because the numerical cutoff is imposed on
    the complete physical ``G+q`` sphere, not on a fixed ``G`` ball. In
    particular, the Nyquist sphere is inversion symmetric even though
    ``+1/2`` and ``-1/2`` share one canonical label.
    """

    a = np.asarray(system.lattice, dtype=float)
    b = 2.0 * np.pi * np.linalg.inv(a).T
    q = np.asarray(q_cart, dtype=float).reshape(3)
    fractional = np.linalg.solve(b, q)
    fractional = np.round(fractional, 12)
    # Subtracting an integer from a rounded fractional coordinate can add
    # a last-bit difference (e.g. 1.23 - 1 versus 0.23). Restore the same
    # quantization after wrapping so reciprocal relabels share one label.
    wrapped = np.round(fractional - np.floor(fractional + 0.5), 12)
    wrapped[np.abs(wrapped) < 5.0e-13] = 0.0
    return b @ wrapped


def _rsgdf_shifted_sphere_boundary_tolerance(
    system: PeriodicSystem,
    p_max: float,
    q_cart: np.ndarray,
) -> float:
    """Floating-point slack shared by shifted-sphere base/tail partitions."""

    q = _canonical_reciprocal_transfer(system, q_cart)
    return (
        128.0
        * np.finfo(float).eps
        * max(float(p_max) + float(np.linalg.norm(q)), 1.0)
    )


def _rsgdf_bounded_reciprocal_sphere(
    system: PeriodicSystem,
    q_cart: np.ndarray,
    ke_cutoff: float,
    *,
    output_byte_cap: int,
    workspace_byte_cap: int,
    candidate_cap: int,
    allow_empty: bool = False,
) -> np.ndarray:
    """Enumerate a physical q+G sphere with bounded temporary arrays.

    Count first, allocate the exact output once, then fill in integer-index
    order. The candidate census precedes every lattice-sized allocation.
    Unlike the legacy mesh builders, this never materializes a 3-D box.
    The cap covers numerical arrays, not Python/NumPy allocator overhead.
    """
    import math

    if int(getattr(system, "dim", 3)) != 3:
        raise ValueError("range-separated GDF requires a three-dimensional cell")
    energy = float(ke_cutoff)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("range-separated GDF reciprocal cutoff must be finite and positive")
    if min(output_byte_cap, workspace_byte_cap, candidate_cap) <= 0:
        raise ValueError("range-separated GDF mesh caps must be positive")
    lattice = np.asarray(system.lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("range-separated GDF requires a finite 3x3 lattice")
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    q_input = np.asarray(q_cart)
    if q_input.shape != (3,) or np.iscomplexobj(q_input) or not np.isfinite(q_input).all():
        raise ValueError("range-separated GDF q must be a finite real three-vector")
    q = _canonical_reciprocal_transfer(system, q_input)
    if not np.isfinite(reciprocal).all() or not np.isfinite(q).all():
        raise ValueError("range-separated GDF reciprocal lattice must be finite")
    radius = np.sqrt(energy) * np.sqrt(2.0)
    slack = _rsgdf_shifted_sphere_boundary_tolerance(system, radius, q)
    reach = (radius + slack) * np.linalg.norm(lattice, axis=0) / (2.0 * np.pi)
    fractional_q = np.linalg.solve(reciprocal, q)
    if not np.isfinite(reach).all():
        raise ValueError("range-separated GDF reciprocal extent is not finite")
    lower = [math.floor(-fractional_q[i] - reach[i]) - 1 for i in range(3)]
    upper = [math.ceil(-fractional_q[i] + reach[i]) + 1 for i in range(3)]
    widths = [hi - lo + 1 for lo, hi in zip(lower, upper)]
    candidate_count = math.prod(widths)
    context = (f"ke_cutoff={energy:.17g} Ha, q_cart={tuple(float(x) for x in q)}, "
               f"box_widths={tuple(widths)}")
    if candidate_count > int(candidate_cap):
        raise _RangeSeparatedGdfAdmissionError(
            "range-separated GDF reciprocal candidate cap exceeded: "
            f"candidates={candidate_count}, cap={int(candidate_cap)}; {context}. "
            "Reducing the k batch does not reduce this single-q search box.",
            retry_with_fewer_kpoints=False,
        )
    # A line holds integer labels, Cartesian coordinates, squared norms,
    # masks and a selected-point copy. Keep generous array headroom while
    # making the line length independent of the complete box dimensions.
    line_length = min(512, widths[2])
    workspace_required = 4096 + 128 * line_length
    if workspace_required > int(workspace_byte_cap):
        raise _RangeSeparatedGdfAdmissionError(
            "range-separated GDF reciprocal workspace cap exceeded: "
            f"required={workspace_required} bytes, cap={int(workspace_byte_cap)} bytes; "
            f"line_length={line_length}, {context}",
            retry_with_fewer_kpoints=False,
        )
    limit_squared = (radius + slack) ** 2

    def panels():
        for i in range(lower[0], upper[0] + 1):
            for j in range(lower[1], upper[1] + 1):
                origin = q + i * reciprocal[:, 0] + j * reciprocal[:, 1]
                for first in range(lower[2], upper[2] + 1, line_length):
                    labels = np.arange(first, min(first + line_length, upper[2] + 1))
                    points = labels[:, None] * reciprocal[None, :, 2] + origin
                    keep = np.einsum("ij,ij->i", points, points) <= limit_squared
                    yield points, keep

    count = 0
    for _, keep in panels():
        count += int(np.count_nonzero(keep))
        if 24 * count > int(output_byte_cap):
            raise _RangeSeparatedGdfAdmissionError(
                "range-separated GDF reciprocal output cap exceeded: "
                f"required_at_least={24 * count} bytes, cap={int(output_byte_cap)} bytes; "
                f"retained_so_far={count}, {context}"
            )
    del _, keep
    if count == 0 and not allow_empty:
        raise ValueError("range-separated GDF reciprocal cutoff contains no q+G vectors")
    result = np.empty((count, 3), dtype=np.float64)
    offset = 0
    for points, keep in panels():
        selected = points[keep]
        result[offset:offset + len(selected)] = selected
        offset += len(selected)
        del selected
    return result


def _rsgdf_shifted_dense_g_mesh(
    system: PeriodicSystem,
    q_cart: np.ndarray,
    ke_cutoff: float,
) -> np.ndarray:
    """Return ``G+q`` vectors inside one physical kinetic-energy sphere.

    The integer reciprocal-lattice box is enlarged by ``|q|`` before the
    spherical filter, so no vector satisfying
    ``|G+q| <= sqrt(2*ke_cutoff)`` is missed. Unlike shifting the already
    truncated output of :func:`rsgdf_dense_g_mesh`, this set is invariant
    under ``q -> q + G0`` and maps to its negative under time reversal.
    """

    if ke_cutoff <= 0:
        raise ValueError(
            "_rsgdf_shifted_dense_g_mesh: ke_cutoff must be > 0, "
            f"got {ke_cutoff}"
        )
    a = np.asarray(system.lattice, dtype=float)
    if a.shape != (3, 3):
        raise RuntimeError(
            "_rsgdf_shifted_dense_g_mesh: lattice must be 3x3, "
            f"got {a.shape}"
        )
    dim = int(getattr(system, "dim", 3))
    if dim != 3:
        _reject_transverse_collapse(dim, "_rsgdf_shifted_dense_g_mesh")

    q = _canonical_reciprocal_transfer(system, q_cart)
    if float(np.linalg.norm(q)) < 1.0e-14:
        # Preserve the historical Γ path byte-for-byte: its norm-only stable
        # order is already inversion symmetric and underlies tight covariance
        # and external-parity gates. Reciprocal relabels of Γ canonicalise to
        # this exact same path.
        return rsgdf_dense_g_mesh(system, ke_cutoff)
    b = 2.0 * np.pi * np.linalg.inv(a).T
    p_max = float(np.sqrt(2.0 * float(ke_cutoff)))
    # If p = G+q is inside the target sphere, then |G| <= p_max+|q|.
    # Biorthogonality gives |n_i| <= |a_i| |G| / (2*pi) for G=B n,
    # which is an exact per-axis enumeration bound for a column-vector
    # lattice convention. The extra integer is roundoff headroom only.
    g_bound = p_max + float(np.linalg.norm(q))
    a_norms = np.linalg.norm(a, axis=0)
    n_per_dim = [
        int(np.ceil(g_bound * float(a_norms[axis]) / (2.0 * np.pi))) + 1
        for axis in range(3)
    ]
    grids = [np.arange(-n, n + 1) for n in n_per_dim]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    indices = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(
        float
    )
    shifted = indices @ b.T + q[None, :]
    norms = np.linalg.norm(shifted, axis=1)
    boundary_tolerance = _rsgdf_shifted_sphere_boundary_tolerance(
        system,
        p_max,
        q,
    )
    keep = norms <= p_max + boundary_tolerance
    shifted = shifted[keep]
    norms = norms[keep]
    # Sorting by the physical vector, rather than the raw reciprocal index,
    # keeps accumulation order stable when q is relabelled by q+G0. Round only
    # the sort keys, not the vectors consumed by the numerical contraction.
    sort_vectors = np.round(shifted, 12)
    sort_norms = np.round(norms, 12)
    order = np.lexsort(
        (
            sort_vectors[:, 2],
            sort_vectors[:, 1],
            sort_vectors[:, 0],
            sort_norms,
        )
    )
    return shifted[order]


def rsgdf_dense_g_mesh(
    system: PeriodicSystem,
    ke_cutoff: float,
    *,
    allow_transverse_collapse: bool = False,
) -> np.ndarray:
    """Uniform FFT-style reciprocal-lattice G-mesh sized by kinetic-energy cutoff.

    For tight periodic cells with diffuse aux primitives, the
    :func:`rsgdf_g_mesh` (sized by the LR damping kernel
    ``2w.√ln(1/prec)``) is too coarse to resolve the Gaussian decay of
    the basis functions in reciprocal space -- leading to w-dependent,
    mesh-sensitive SCF energies on tight ionic crystals. This routine
    builds a denser mesh sized by ``G_max = √(2.ke_cutoff)`` (the
    plane-wave kinetic-energy cutoff convention used by PySCF's
    ``_RSGDFBuilder``).

    The mesh covers the integer reciprocal lattice points within a
    sphere of radius ``G_max`` -- same structure as
    :func:`rsgdf_g_mesh` but with a different sizing rule.

    Typical ``ke_cutoff`` values:

    * H₂ / 12-bohr / def2-svp-jk: 100 Ha -> ~5500 G-points.
    * LiH primitive FCC / sto-3g / def2-svp-jk: 200 Ha -> ~15000 G-points.
    * Default in :func:`build_lpq_native_fft`: 200 Ha (mesh-converged
      for sub-mHa SCF on def2-svp-jkfit-class aux).

    References
    ----------
    PySCF ``pyscf.pbc.df.aft.estimate_ke_cutoff_for_omega``;
    Ye & Berkelbach 2021 (DOI 10.1063/5.0046617) for the kinetic-energy
    sizing of the LR mesh in RSGDF.
    """
    if ke_cutoff <= 0:
        raise ValueError(f"rsgdf_dense_g_mesh: ke_cutoff must be > 0, got {ke_cutoff}")
    a = np.asarray(system.lattice, dtype=float)
    if a.shape != (3, 3):
        raise RuntimeError(f"rsgdf_dense_g_mesh: lattice must be 3x3, got {a.shape}")
    b = 2.0 * np.pi * np.linalg.inv(a).T  # columns are reciprocal vectors
    G_max = float(np.sqrt(2.0 * ke_cutoff))
    dim = int(getattr(system, "dim", 3))
    if dim not in (1, 2, 3):
        raise ValueError(f"rsgdf_dense_g_mesh: dim must be 1, 2, or 3; got {dim}")
    if not allow_transverse_collapse:
        _reject_transverse_collapse(dim, "rsgdf_dense_g_mesh")
    # Index-box bound. The integer index of G = n1 b1 + n2 b2 + n3 b3 is
    # n_i = G.a_i / (2pi) (biorthogonality), so |n_i| <= G_max |a_i| / (2pi)
    # is the EXACT per-axis bound for the |G| <= G_max ball. The historical
    # bound ceil(G_max / |b_i|) + 1 is equivalent only for orthogonal cells
    # (|a_i||b_i| = 2pi); for oblique lattices |a_i||b_i| > 2pi and that box
    # UNDERCOUNTS -- it silently drops G-points near the sphere boundary,
    # linearly more of them at higher cutoffs (fcc primitive: ~15% deficit
    # per axis). On the P01 MgO fcc cell this made every "ke_cutoff" mesh
    # subtly incomplete and broke the exact base+tail == big-mesh identity
    # of the tail completion below.
    # |a_i| are COLUMN norms: vibe-qc lattice matrices carry the vectors in
    # columns (the same convention `b = 2 pi inv(a).T` and `G = idx @ b.T`
    # above rely on). The previous axis=1 (row norms) was only correct for
    # symmetric lattice matrices (e.g. the fcc primitive cell, where every
    # historical control lived); for a non-symmetric matrix -- any (n,1,1)
    # supercell of fcc, hexagonal cells, ... -- the row norm underestimates
    # |a_i| and the box CLIPS the ball: measured 602 of 3381 G-points
    # (18%) silently missing on the C-diamond (2,1,1) cluster cell at
    # ke_cutoff = 60, which broke the mesh's point-group closure (caught
    # 2026-07-10 by the space-group pair-reduction covariance gate at
    # 4e-3) and under-converged every affected fit at its declared cutoff.
    # Column-norm per-axis box (shared with rsgdf_g_mesh / _aft_g_mesh).
    n_per_dim = _reciprocal_index_box_per_axis(a, G_max, dim)
    grids = [
        np.arange(-n, n + 1) if axis < dim else np.array([0])
        for axis, n in enumerate(n_per_dim)
    ]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(float)
    G = idx @ b.T
    G_norms = np.linalg.norm(G, axis=1)
    keep = G_norms <= G_max
    G_kept = G[keep]
    order = np.argsort(G_norms[keep])
    return G_kept[order]


def rsgdf_aux_fourier_transform(
    aux_basis: BasisSet,
    G_vectors: np.ndarray,
) -> np.ndarray:
    """Analytical FT of an auxiliary basis at given G-vectors.

    For a contracted shell at center R with primitives ``(a_p, c_p)``
    and angular momentum ``L``, the FT of basis function (L, m) is

        ξ̂_lm(G; R) = e^{-iG.R} . S_p c_p . F[r^l Y_lm e^{-a_p r^2}](G)

    with the primitive FT taken from the closed form (*) above. The
    coefficients ``c_p`` here are libint's internal post-renormalisation
    representation (what ``BasisSet.shells()`` returns) -- they already
    include libint's primitive normalisation factor, so the formula
    is applied as-is.

    Parameters
    ----------
    aux_basis
        Auxiliary basis (vibe-qc BasisSet).
    G_vectors
        Cartesian reciprocal-lattice points, shape ``(n_G, 3)``.
        Typically obtained from :func:`rsgdf_g_mesh`. The G = 0
        special case is handled (returns 0 contribution from non-S
        shells; for S shells the value at G = 0 is the monopole,
        which the LR caller multiplies by the divergent ``1/G^2`` and
        thus excludes from the sum).

    Returns
    -------
    ft : (n_aux, n_G) complex128
        ``ft[mu, k]`` is the FT of the mu-th aux basis function at G_k.
    """
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    n_aux = aux_basis.nbasis
    out = np.zeros((n_aux, n_G), dtype=np.complex128)

    # Spherical coordinates of the G-vectors (theta = polar from +z;
    # phi = azimuth from +x). At G = 0 the angles are undefined but
    # only the L = 0 component contributes (and its angular factor is
    # the constant 1/√(4pi)); we set theta = phi = 0 there and rely on
    # the radial G^L = 0 factor for L > 0 to zero the contribution.
    G_norms = np.linalg.norm(G_vectors, axis=1)
    safe_norm = np.where(G_norms > 0, G_norms, 1.0)
    theta = np.arccos(np.clip(G_vectors[:, 2] / safe_norm, -1.0, 1.0))
    phi = np.arctan2(G_vectors[:, 1], G_vectors[:, 0])

    # SPEED: cache Y_lm per L so a basis with several shells at the
    # same L doesn't recompute the spherical harmonics each time.
    ylm_cache: dict[int, np.ndarray] = {}

    bf_offset = 0
    for shell in aux_basis.shells():
        L = int(shell.l)
        n_components = 2 * L + 1 if shell.pure else (L + 1) * (L + 2) // 2
        if not shell.pure and L > 0:
            raise NotImplementedError(
                "rsgdf_aux_fourier_transform: Cartesian (non-pure) shells "
                "with L > 0 are not yet supported. vibe-qc forces pure "
                "spherical for L >= 2 by default -- this should not arise "
                "in practice."
            )
        exps = np.asarray(shell.exponents, dtype=float)
        coeffs = np.asarray(shell.coefficients, dtype=float)
        R = np.asarray(shell.origin, dtype=float)

        # Phase factor exp(-iG.R) -- same for all (l, m) on this shell.
        phase = np.exp(-1j * (G_vectors @ R))  # (n_G,)

        # Radial FT of each primitive: f_p(G) = (pi/a_p)^(3/2) .
        # (G/(2a_p))^L . exp(-G^2/(4a_p)). We sum over primitives with
        # contraction coefficients c_p first.
        # rad_per_prim[p, k] = c_p . f_p(G_k); sum over p gives
        # rad_contracted[k] = S_p c_p . f_p(G_k).
        G2 = G_norms**2  # (n_G,)
        # Broadcast-friendly: (n_prim, n_G).
        prim_factor = (np.pi / exps)[:, None] ** 1.5 * np.exp(
            -G2[None, :] / (4.0 * exps[:, None])
        )
        if L > 0:
            prim_factor *= (G_norms[None, :] / (2.0 * exps[:, None])) ** L
        rad_contracted = np.einsum("p,pk->k", coeffs, prim_factor)  # (n_G,)

        # Angular factor: (-i)^L . Y_lm(Ĝ) for each m, plus a per-L
        # convention multiplier so that the resulting FTs satisfy
        # Plancherel-for-Coulomb against libint's compute_2c_eri:
        #
        #     libint (P|Q) = (1/(2pi^2)) . ∫ F̂_P*(G) F̂_Q(G) / G^2 d^3G
        #
        # CONVENTION FIX (2026-05-10 + 2026-05-17): libint's
        # BraKet::xs_xs absorbs the Y_00 = 1/√(4pi) factor into the
        # s-shell basis function (verified empirically). For L=0 we
        # drop the Y_00 angular factor (angular = 1 instead of
        # 1/√(4pi)). For L>0 the empirical calibration on single-prim
        # shells across a in [0.25, 2.0] (see
        # examples/debug/gdf_aft_calibration.py) shows libint's
        # (P|P) = (4pi/(2L+1)) x Parseval[F̂_uncorrected]: there is a
        # per-L convention scale of √(4pi/(2L+1)) that needs to enter
        # the FT formula so |F̂|^2 gives the right Coulomb scale via
        # 1/G^2 integration.
        #
        # For L=0 the existing Y_00-drop fix is equivalent to
        # multiplying by √(4pi/(2.0+1)) = √(4pi) -- so the unified
        # convention factor is simply
        #
        #     scale_L = √(4pi / (2L+1))
        #
        # absorbed into the FT for all L. L=0 was already correct
        # via the explicit Y_00 drop; this comment is the cleaner
        # restatement.
        i_to_minus_l = (-1j) ** L
        # scale_L = √(4pi / (2L+1)). For L=0 we leave the Y_00 drop
        # in place (its effect is exactly √(4pi)) and DON'T multiply
        # again -- using scale_L = 1.0 below. For L>=1 we apply the
        # √(4pi/(2L+1)) factor.
        scale_L = float(np.sqrt(4.0 * np.pi / (2 * L + 1))) if L > 0 else 1.0

        # SPEED: cache Y_lm by L (same pattern as
        # _aft_fourier_transform_libcint_convention) so a basis with
        # multiple shells at the same L doesn't recompute the
        # spherical harmonics for every shell.
        if L == 0:
            # libint s-shell convention: no Y_00 in the basis function
            # (it's been folded into c_libint). FT angular factor = 1.
            ylm_block = np.ones((1, theta.shape[0]), dtype=np.float64)
        else:
            if L not in ylm_cache:
                block = np.empty((2 * L + 1, theta.shape[0]), dtype=np.float64)
                for idx, m in enumerate(range(-L, L + 1)):
                    block[idx] = _real_sph_harm(L, m, theta, phi)
                ylm_cache[L] = block
            ylm_block = ylm_cache[L]
        n_comp = 2 * L + 1
        ft_block = (
            scale_L
            * i_to_minus_l
            * ylm_block
            * rad_contracted[None, :]
            * phase[None, :]
        )
        out[bf_offset : bf_offset + n_comp, :] = ft_block
        bf_offset += n_comp

    if bf_offset != n_aux:
        raise RuntimeError(
            f"rsgdf_aux_fourier_transform: shell-walking covered "
            f"{bf_offset} AOs but aux.nbasis = {n_aux} -- internal error."
        )
    return out


def rsgdf_lr_2c_metric(
    aux_basis: BasisSet,
    system: PeriodicSystem,
    omega: float,
    *,
    precision: float = 1e-8,
    g_mesh: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Long-range half of the periodic 2c metric, M^LR_PQ(w).

    ::

        M^LR_PQ(w) = (4pi/Ω) . S_{G!=0} ξ̂_P*(G) . ξ̂_Q(G) .
                              exp(-G^2/(4w^2)) / G^2

    Parameters
    ----------
    aux_basis
        Auxiliary basis (typically modrho-rescaled by the caller --
        the LR formula itself is invariant under per-shell scaling,
        same as the SR kernel).
    system
        Periodic lattice + atoms (used only for the unit-cell volume
        and the reciprocal-lattice mesh).
    omega
        RSGDF range-separation parameter (bohr⁻¹). Same value used in
        ``compute_2c_eri_lattice_sr``; the SR + LR sum is independent
        of w at convergence.
    precision
        Target precision for the G-mesh truncation
        (``exp(-G^2_max/(4w^2)) <= precision``).
    g_mesh
        Pre-computed G-vectors (e.g. shared with the 3c LR build).
        If ``None``, calls :func:`rsgdf_g_mesh` to generate one.

    Returns
    -------
    M_lr : (n_aux, n_aux) float64
        Real, symmetric. Together with the SR half, recovers the
        full periodic Coulomb 2c metric.
    """
    if g_mesh is None:
        G_vectors = rsgdf_g_mesh(system, omega, precision=precision)
    else:
        G_vectors = np.ascontiguousarray(g_mesh, dtype=float)

    G2 = (G_vectors**2).sum(axis=1)
    nonzero = G2 > 0
    if not nonzero.any():
        n_aux = aux_basis.nbasis
        return np.zeros((n_aux, n_aux), dtype=float)

    # Damped Coulomb kernel in reciprocal space, restricted to G != 0.
    K = np.exp(-G2[nonzero] / (4.0 * omega**2)) / G2[nonzero]  # (n_G_kept,)

    fts = rsgdf_aux_fourier_transform(aux_basis, G_vectors[nonzero])
    # M^LR_PQ = (4pi/Ω) . S_G ξ̂_P*(G) . K(G) . ξ̂_Q(G)
    #         = (4pi/Ω) . (ξ̂* . diag(K) . ξ̂.T)_PQ
    # Compute via a Hermitian product; result is real (M is real-symmetric).
    a = np.asarray(system.lattice, dtype=float)
    cell_volume = float(np.abs(np.linalg.det(a)))
    fts_w = fts * np.sqrt(K)[None, :]  # (n_aux, n_G_kept)
    M = np.real(fts_w.conj() @ fts_w.T) * (4.0 * np.pi / cell_volume)

    # G = 0 Madelung renormalisation. The LR Coulomb kernel
    #   4pi . exp(-G^2/(4w^2)) / G^2
    # expands at G -> 0 as ``4pi/G^2 - pi/w^2 + O(G^2)``. The divergent
    # ``4pi/G^2`` piece is cancelled by the neutralising background
    # convention (charge-neutral cells); the finite ``-pi/w^2`` remainder
    # has to be added back explicitly because the discrete G-sum above
    # omits G = 0. For L >= 1 aux shells the monopole moment ``ξ̂(0)``
    # vanishes so the correction is zero; for L = 0 it is the
    # monopole-monopole self-energy term.
    #
    # Equivalent to PySCF ``_RSGDFBuilder.get_2c2e`` rsdf_builder.py:268
    # (``g0_fac = pi / w^2 / vol``) and the FT-SR G = 0 addition at
    # rsdf_builder.py:343 (``coulG_SR_at_G0 = pi / w^2 . kws``). Per-shell
    # block parity vs PySCF ``get_2c2e`` is bit-exact on He / def2-svp-jkfit
    # at w in [0.2, 1.5] (5+ sig figs).
    fts_at_0 = rsgdf_aux_fourier_transform(aux_basis, np.zeros((1, 3), dtype=float))[
        :, 0
    ].real
    M -= np.outer(fts_at_0, fts_at_0) * (np.pi / (omega**2 * cell_volume))

    M = 0.5 * (M + M.T)
    return M


def rsgdf_lr_3c_tensor(
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    system: PeriodicSystem,
    omega: float,
    *,
    precision: float = 1e-8,
    g_mesh: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Long-range half of the periodic 3-centre ERI tensor,
    T^LR_{P, mu, ν}(w).

    ::

        T^LR_{P, muν}(w) = (4pi/Ω) . S_{G!=0} ξ̂_P*(G) . r̂_{muν}(G) .
                                    exp(-G^2/(4w^2)) / G^2

    where ξ̂_P(G) is the analytic Fourier transform of aux primitive
    P (cf. :func:`rsgdf_aux_fourier_transform`) and r̂_{muν}(G) is
    the analytic Fourier transform of the AO pair density
    chi_mu(r).chi_ν(r) (cf.
    :func:`vibeqc._aopair_ft.ao_pair_fourier_transform`).

    The Bloch-summed orbital pair restricts r̂ to the reciprocal
    lattice; the resulting discrete G-sum is what's coded here. See
    ``docs/design_rsgdf_3c_lr.md`` for the derivation.

    Status (milestone 1, 2026-05-10): only s x s AO pairs supported
    (``ao_basis`` must be entirely L=0). General-L via
    McMurchie-Davidson is milestone 2 in the design doc.

    Parameters
    ----------
    ao_basis
        Orbital basis (vibe-qc BasisSet, on the unit-cell molecule).
        Must be s-only until milestone 2 lands.
    aux_basis
        Auxiliary basis. Same calling convention as
        :func:`rsgdf_lr_2c_metric` -- typically modrho-rescaled by
        the caller (the LR formula itself is invariant under
        per-shell aux scaling).
    system
        Periodic lattice + atoms (used for the unit-cell volume + G
        mesh).
    omega
        RSGDF range-separation parameter (bohr⁻¹). Must match the
        omega used in :func:`compute_3c_eri_lattice_sr`; the
        SR + LR sum is independent of w at convergence.
    precision
        Target precision for the G-mesh truncation.
    g_mesh
        Pre-computed G-vectors (share with the 2c LR build for
        efficiency). If ``None``, calls :func:`rsgdf_g_mesh`.

    Returns
    -------
    T_lr : (n_aux, n_orb, n_orb) float64
        Real-valued (Γ-only). Symmetric in (mu, ν).
    """
    from ._aopair_ft import ao_pair_fourier_transform

    if g_mesh is None:
        G_vectors = rsgdf_g_mesh(system, omega, precision=precision)
    else:
        G_vectors = np.ascontiguousarray(g_mesh, dtype=float)

    G2 = (G_vectors**2).sum(axis=1)
    nonzero = G2 > 0
    n_aux = aux_basis.nbasis
    n_orb = ao_basis.nbasis
    if not nonzero.any():
        return np.zeros((n_aux, n_orb, n_orb), dtype=float)

    G_kept = G_vectors[nonzero]
    K = np.exp(-G2[nonzero] / (4.0 * omega**2)) / G2[nonzero]  # (n_G_kept,)

    aux_ft = rsgdf_aux_fourier_transform(aux_basis, G_kept)  # (n_aux, n_G)
    pair_ft = ao_pair_fourier_transform(ao_basis, G_kept)  # (n_orb, n_orb, n_G)

    # Per-AO-pair calibration: the aux FT carries per-L convention
    # factors (√(4pi/(2L+1))); the AO-pair FT needs the same treatment,
    # applied as the outer product of the per-AO factors.
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)  # (n_orb, n_orb)
    pair_ft = np.einsum("mn,mnk->mnk", pair_scales, pair_ft)

    # T^LR_{P, muν} = (4pi/Ω) S_G ξ̂_P*(G) r̂_muν(G) K(G).
    # Reduction over G axis. Result is real (analytic property of
    # the periodic Coulomb integral at Γ).
    a = np.asarray(system.lattice, dtype=float)
    cell_volume = float(np.abs(np.linalg.det(a)))
    weight = K * (4.0 * np.pi / cell_volume)  # (n_G,)
    # Fold weight into one of the FTs to avoid an extra big array.
    aux_w = aux_ft.conj() * weight[None, :]  # (n_aux, n_G)
    # T[P, mu, ν] = S_G aux_w[P, G] . pair_ft[mu, ν, G]
    T_lr_complex = np.einsum("Pk,mnk->Pmn", aux_w, pair_ft)
    T_lr = np.real(T_lr_complex)

    # G = 0 Madelung renormalisation -- analogous to the 2c case (see
    # :func:`rsgdf_lr_2c_metric`). Replaces the divergent ``4pi/G^2``
    # piece at G -> 0 by the finite ``-pi/w^2`` remainder of the LR-Coulomb
    # Taylor expansion; the divergent piece is cancelled by the
    # neutralising background convention. The AO-pair FT at G = 0 is the
    # overlap matrix ``S_muν = ∫ chi_mu chi_ν dr`` (per-L calibration matches
    # the finite-G FT path); the aux FT at G = 0 is non-zero only for
    # L = 0 shells.
    fts_at_0 = rsgdf_aux_fourier_transform(aux_basis, np.zeros((1, 3), dtype=float))[
        :, 0
    ].real
    from ._vibeqc_core import compute_overlap as _compute_overlap

    S = np.asarray(_compute_overlap(ao_basis))
    S_cal = S * pair_scales
    T_lr -= np.einsum("p,mn->pmn", fts_at_0, S_cal) * (np.pi / (omega**2 * cell_volume))

    # Symmetrise (mu, ν) -- analytic property of the AO pair FT.
    T_lr = 0.5 * (T_lr + T_lr.transpose(0, 2, 1))
    return T_lr
