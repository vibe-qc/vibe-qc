"""User-selectable lattice-sum screening strategies.

Vibe-qc's ``compute_*_eri_lattice`` C++ kernels read a single flat
``cutoff_bohr`` from :class:`LatticeSumOptions` and include every
lattice cell whose Cartesian shift has Euclidean norm <= that value.
This is the right default for tight basis functions (overlap decays
exponentially -> most distant cells contribute zero), but for diffuse
basis functions (small primitive exponents) the flat cutoff either
(a) wastes time on cells whose contribution is below precision, or
(b) under-includes cells whose contribution still exceeds precision.

PySCF takes a different approach: ``cell.rcut`` is auto-tuned per
shell, sized by the smallest primitive exponent and a target precision,
so the effective cutoff is short for diffuse shells and longer for
tight ones. The full lattice sum then combines this short bare sum
with a reciprocal-space AFT correction that supplies the missing
long-range Coulomb.

This module exposes both strategies (and the surface for adding more)
as :class:`RcutStrategy` plus the :func:`make_lattice_opts` factory.
Drivers that want PySCF-compatible lattice sums (e.g. compcell GDF
with AFT correction) pass ``rcut_strategy=PYSCF_AUTO``; drivers that
want vibe-qc's historical flat-cutoff behaviour pass ``FLAT`` (default).

References
----------
PySCF ``pyscf/pbc/gto/cell.py:_estimate_rcut`` /
``estimate_rcut``.
"""
from __future__ import annotations

import enum
import math
from typing import Optional

import numpy as np

from ._vibeqc_core import BasisSet, LatticeSumOptions


__all__ = [
    "RcutStrategy",
    "estimate_rcut_pyscf_per_shell",
    "estimate_rcut_pyscf",
    "estimate_rcut_overlap_pair",
    "estimate_rcut_amplitude",
    "atom_pair_span_bohr",
    "bloch_overlap_cutoff_bohr",
    "make_lattice_opts",
]

# Target magnitude for the largest DROPPED primitive-pair overlap in the
# real-space sum behind S(k) = sum_R exp(i k . R) S(R). It plays the role of
# eps_s in Lippert, Hutter & Parrinello 1997 (their recommended 1e-5, Mol.
# Phys. 92, 477, p. 483 / Table 1 p. 484) and of CRYSTAL's S_c = 10^-ITOL1
# (default ITOL1 = 7, CRYSTAL23 manual section 18.3 p. 402, TOLINTEG p. 130),
# but much tighter, for the same reason PySCF hardens its own overlap sum to
# cell.precision * 1e-6 = 1e-14 (pyscf/pbc/scf/hf.py): those looser values
# screen integrals that then enter a variational energy, whereas here the
# dropped tail enters the METRIC, where an error of the same size is not a
# small perturbation but a change of signature.
#
# Measured: min eig(S(k)) over a Gamma-centred (4,4,4) mesh is non-negative
# at this tolerance for def2-SVP, def2-TZVP, 6-31G, STO-3G and pob-TZVP-rev2
# on primitive LiF, MgO and Si (worst case +3.7e-11, LiF / def2-SVP). It is
# a convergence target, not a tunable knob: loosening it re-admits the
# indefinite metric it exists to exclude.
DEFAULT_BLOCH_OVERLAP_TOL = 1e-12


class RcutStrategy(enum.Enum):
    """How to choose ``LatticeSumOptions.cutoff_bohr`` for a given basis.

    * ``FLAT`` -- keep ``base_opts.cutoff_bohr`` unchanged (vibe-qc's
      historical default; a single geometric cutoff applied to every
      lattice cell).
    * ``PYSCF_AUTO`` -- auto-tune per-shell rcut using PySCF's formula
      (Gaussian-decay-driven), then take the max across shells. Close to
      but NOT equal to PySCF's ``cell.rcut`` for the same basis +
      precision -- it runs ~0.7-0.8x of it; see
      :func:`estimate_rcut_pyscf` for the measured ratios and the three
      terms this closed form omits. The right
      choice when downstream code combines the short-range lattice sum
      with a reciprocal-space AFT correction (compcell GDF, RSGDF).

      Note for callers: :func:`make_lattice_opts` *replaces*
      ``base_opts.cutoff_bohr`` with the estimate under this strategy
      rather than taking the larger of the two, so a caller's explicit
      ``cutoff_bohr`` does not reach a sum sized this way.
    * ``OVERLAP_PAIR`` -- bound the largest *primitive-pair* overlap that
      the truncation drops, using the Gaussian-product reduced exponent.
      The right choice when the truncated real-space sum is used BARE,
      with no reciprocal-space correction to supply the missing tail --
      in particular for the Bloch overlap ``S(k) = sum_R exp(ik.R) S(R)``,
      whose positive definiteness is destroyed by a dropped tail. Reaches
      further than ``PYSCF_AUTO`` by construction: see
      :func:`estimate_rcut_overlap_pair`.
    """

    FLAT = "flat"
    PYSCF_AUTO = "pyscf_auto"
    OVERLAP_PAIR = "overlap_pair"


def estimate_rcut_pyscf_per_shell(
    alpha: float,
    L: int,
    coef: float,
    *,
    precision: float = 1e-8,
    rcut_min: float = 0.1,
) -> float:
    """Per-shell cutoff radius, PySCF-style.

    Mirrors PySCF ``pyscf/pbc/gto/cell.py:_estimate_rcut`` (the same
    formula PySCF uses to set ``cell.rcut``). The shell's value at
    ``rcut`` decays below ``precision`` based on its smallest primitive
    exponent and contraction coefficient:

        log_prec = log(precision / (|coef| . √((2L+1).a / (2pi))))
        rcut     = √(max(log_prec / (-theta), rcut_min)),  theta = a / 2

    Parameters
    ----------
    alpha
        Smallest primitive exponent of the shell (bohr⁻^2).
    L
        Angular momentum.
    coef
        Magnitude of the (max-absolute) contraction coefficient on the
        most-diffuse primitive. PySCF uses ``|c|.max()`` across
        contractions; for a single-contracted shell this is just the
        coefficient on the smallest primitive.
    precision
        Target precision for the shell-tail magnitude (default 1e-8,
        matching PySCF's ``cell.precision`` default).
    rcut_min
        Lower bound on ``rcut`` (PySCF default 0.1 bohr) so very tight
        shells don't get rcut -> 0 from numerical underflow.

    Notes
    -----
    The decay exponent is ``theta = alpha / 2``, NOT ``alpha``. What is
    being truncated is a lattice sum over AO *pairs*, and by the Gaussian
    product theorem two Gaussians of exponent ``alpha`` separated by ``R``
    overlap as ``exp(-mu R^2)`` with the reduced exponent
    ``mu = alpha*alpha/(alpha+alpha) = alpha/2``. A pair therefore reaches
    sqrt(2) times further than the single-shell amplitude ``exp(-alpha
    r^2)``, and screening on the single-shell decay truncates the sum
    early. PySCF's ``_estimate_rcut`` (``pyscf/pbc/gto/cell.py``) uses the
    same reduced exponent -- verified against PySCF 2.13.1, whose first
    body line is literally ``theta = alpha * .5``.

    # Eqs. (9.2.10) and (9.2.12) of Helgaker, Jorgensen & Olsen, *Molecular
    # Electronic-Structure Theory*, Wiley (2000), section 9.2.3 p. 341 --
    # the Gaussian product rule:
    #   exp(-a x_A^2) exp(-b x_B^2) = exp(-mu X_AB^2) exp(-p x_P^2),
    #   mu = a b / (a + b)   (the reduced exponent)
    # and Eq. (9.3.10), section 9.3.1 p. 346, whose 3D normalised s-s form is
    #   S_00(R) = (2 sqrt(a b) / (a + b))^{3/2} exp(-mu R^2).

    Until 2026-08-02 this port used ``-alpha`` and so returned a radius
    short by sqrt(2) (RCUT-PYSCF-MISPORT). The practical effect was that
    ``precision`` did not mean what it said: on compcell H2/STO-3G in a
    12-bohr box at kmesh (2,1,1) the returned energy did not move between
    ``precision=1e-6`` and ``1e-12``, sitting 2.14e-06 Ha away from the
    cutoff-converged value the corrected formula reaches at the 1e-8
    default. See ``agentic-loop/bug-claims.md``.

    Code that consumes the truncated real-space sum BARE -- above all the
    Bloch overlap, whose positive definiteness the tail decides -- should
    still use :func:`estimate_rcut_overlap_pair`, which adds an explicit
    polynomial-factor margin on top of the same reduced exponent.
    """
    if alpha <= 0:
        raise ValueError(f"estimate_rcut_pyscf_per_shell: alpha must be > 0, "
                         f"got {alpha}")
    abs_c = max(abs(float(coef)), 1e-300)
    inner = abs_c * math.sqrt((2 * L + 1) * alpha / (2.0 * math.pi))
    log_prec = math.log(precision / inner)
    # theta is the pair (Gaussian-product) reduced exponent alpha/2, not
    # the single-shell alpha -- see the Notes section above.
    theta = 0.5 * alpha
    val = max(log_prec / (-theta), rcut_min)
    return math.sqrt(val)


def estimate_rcut_pyscf(
    basis: BasisSet,
    *,
    precision: float = 1e-8,
) -> float:
    """Cell-level ``rcut`` for a basis, PySCF-style.

    Walks every shell, computes its per-shell rcut via
    :func:`estimate_rcut_pyscf_per_shell` (using the shell's smallest
    primitive exponent and the maximum-magnitude contraction
    coefficient on that primitive), and returns the max across shells.

    Follows the same construction as PySCF's ``cell.rcut`` -- per-shell
    radius from the diffuse primitive, maximum over shells, decaying with
    the pair reduced exponent ``alpha/2``.

    **It is close to, but not equal to, ``cell.rcut``.** Measured
    out-of-process against PySCF 2.13.1 on H2 at ``precision=1e-8``
    (2026-08-02):

        basis      vibe-qc    PySCF     ratio
        sto-3g     12.936     16.514    0.78
        6-31g      13.585     17.496    0.78
        def2-svp   15.438     19.991    0.77

    The remaining gap is *not* the exponent -- that now agrees exactly.
    PySCF's estimate is deliberately more conservative: it squares the
    contraction coefficient, carries a ``(r/2 + (2 alpha)^-1/2)^(2l+2)``
    polynomial factor solved by two fixed-point iterations, and adds a
    ``4 alpha^2`` penalty so the radius also covers the kinetic operator
    (whose integrand carries an extra factor of ``2 alpha r``). This
    closed form omits all three. Before the 2026-08-02 fix the ratio was
    ~0.55; the sqrt(2) recovered most of the shortfall.

    No test here runs PySCF -- CLAUDE.md § 10 forbids importing it, and
    the numbers above came from two separate processes that never shared
    an interpreter.
    """
    rcut = 0.0
    for shell in basis.shells():
        L = int(shell.l)
        exps = np.asarray(shell.exponents, dtype=float)
        coefs = np.asarray(shell.coefficients, dtype=float)
        idx = int(np.argmin(exps))
        r = estimate_rcut_pyscf_per_shell(
            float(exps[idx]), L, float(coefs[idx]),
            precision=precision,
        )
        if r > rcut:
            rcut = r
    return float(rcut)


def estimate_rcut_overlap_pair(
    basis: BasisSet,
    *,
    tol: float = DEFAULT_BLOCH_OVERLAP_TOL,
    n_refine: int = 8,
) -> float:
    """Separation beyond which every primitive-pair overlap is below ``tol``.

    Screening the real-space lattice sum on the magnitude of the pair
    OVERLAP INTEGRAL -- rather than on a geometric radius -- is the
    criterion the GPW method prescribes for itself:

    # Eq. (36) and section 4 of Lippert, Hutter & Parrinello, Mol. Phys. 92,
    # 477 (1997), doi:10.1080/00268979709482119, p. 482-483: the pairs of
    # Gaussians retained in the sum over periodic replicas are selected "on
    # the basis of the value of their overlap integral", with a screening
    # parameter eps_s (their recommended value 1e-5, tightened here -- see
    # DEFAULT_BLOCH_OVERLAP_TOL).

    It is also what the periodic-LCAO literature has used since the
    beginning: CRYSTAL's ITOL1 = -log10(S_c) thresholds the overlap of the
    two shells' "adjoined Gaussians", each an s-type GTF carrying the
    SMALLEST exponent of its contraction (Pisani & Dovesi, Int. J. Quantum
    Chem. 17, 501 (1980), doi:10.1002/qua.560170311, section 4 "Truncation
    Criteria", p. 508; Pisani, Dovesi & Roetti, *Hartree-Fock Ab Initio
    Treatment of Crystalline Systems*, Springer LNC 48 (1988),
    doi:10.1007/978-3-642-93385-1, section II.2a p. 36 and section II.4i
    p. 65).

    Unlike :func:`estimate_rcut_pyscf`, which as implemented here bounds a
    SINGLE shell's own amplitude at ``r`` (``exp(-alpha r^2)``), this bounds
    the two-centre overlap INTEGRAL that the lattice sum actually drops. The
    distinction is not cosmetic:

    # Eqs. (9.2.10) and (9.2.12) of Helgaker, Jorgensen & Olsen, *Molecular
    # Electronic-Structure Theory*, Wiley (2000), section 9.2.3 p. 341 --
    # the Gaussian product rule:
    #   exp(-a x_A^2) exp(-b x_B^2) = exp(-mu X_AB^2) exp(-p x_P^2),
    #   mu = a b / (a + b)   (the reduced exponent)
    # and Eq. (9.3.10), section 9.3.1 p. 346, whose 3D normalised s-s form is
    #   S_00(R) = (2 sqrt(a b) / (a + b))^{3/2} exp(-mu R^2).

    For a pair drawn from the SAME diffuse shell (a = b = alpha) the reduced
    exponent is mu = alpha/2, so the pair overlap reaches sqrt(2) times
    further than the shell amplitude ``exp(-alpha r^2)``. (PySCF's own
    ``_estimate_rcut`` uses ``theta = alpha/2`` for exactly this reason,
    and so does :func:`estimate_rcut_pyscf_per_shell` -- its port dropped
    the factor until 2026-08-02, when RCUT-PYSCF-MISPORT was fixed.)

    What this function still adds over the corrected
    :func:`estimate_rcut_pyscf` is the explicit polynomial-factor margin
    below. The GDF / RSGDF routes add a reciprocal-space AFT correction
    for whatever tail their cutoff drops; a bare Bloch overlap sum has no
    such correction, and a shortfall there shows up as an INDEFINITE S(k)
    away from the zone centre, so that consumer wants the margin rather
    than the bare precision-driven radius.

    The bound used per primitive pair, for normalised primitives with
    contraction coefficients c_a, c_b and angular momenta l_a, l_b:

        |S_ab(R)| <= C * (1 + sqrt(mu) R)^(l_a + l_b) * exp(-mu R^2)
        C         =  |c_a c_b| * (2 sqrt(a b) / (a + b))^(3/2)

    where C is the exact s-s prefactor and the polynomial factor is a
    deliberate over-estimate of the angular part (it omits the
    (2l-1)!! normalisation denominator), so the returned radius errs long.
    ``|S| = tol`` is solved by fixed-point iteration on

        R <- sqrt( [ ln(C/tol) + (l_a + l_b) ln(1 + sqrt(mu) R) ] / mu ) .

    Returns the maximum over all primitive pairs in the basis. This is a
    SEPARATION between two basis-function centres, not a lattice-vector
    radius -- see :func:`bloch_overlap_cutoff_bohr`, which adds the
    in-cell atom span that converts one into the other.
    """
    if not (0.0 < float(tol) < 1.0):
        raise ValueError(
            f"estimate_rcut_overlap_pair: tol must be in (0, 1); got {tol}"
        )
    tol = float(tol)

    primitives: list[tuple[float, float, int]] = []
    for shell in basis.shells():
        L = int(shell.l)
        exps = np.asarray(shell.exponents, dtype=float)
        coefs = np.asarray(shell.coefficients, dtype=float)
        for alpha, coef in zip(exps, coefs):
            if float(alpha) <= 0.0:
                continue
            primitives.append((float(alpha), abs(float(coef)), L))
    if not primitives:
        return 0.0

    rcut = 0.0
    for a, ca, la in primitives:
        for b, cb, lb in primitives:
            mu = a * b / (a + b)
            prefactor = ca * cb * (2.0 * math.sqrt(a * b) / (a + b)) ** 1.5
            if prefactor <= tol:
                # This pair is already below tolerance at zero separation.
                continue
            log_ratio = math.log(prefactor / tol)
            L = la + lb
            r = math.sqrt(log_ratio / mu)
            for _ in range(int(n_refine)):
                arg = log_ratio + L * math.log1p(math.sqrt(mu) * r)
                r = math.sqrt(max(arg / mu, 0.0))
            if r > rcut:
                rcut = r
    return float(rcut)


def estimate_rcut_amplitude(
    basis: BasisSet,
    *,
    tol: float = DEFAULT_BLOCH_OVERLAP_TOL,
    n_refine: int = 8,
) -> float:
    """Distance beyond which every normalised primitive is below ``tol``.

    The SINGLE-Gaussian counterpart of :func:`estimate_rcut_overlap_pair`:
    it bounds ``|chi(r)|`` itself rather than a two-centre integral, which
    is the right quantity when a basis function is evaluated pointwise --
    collocating Bloch AOs on a plane-wave grid, for instance. A normalised
    primitive of exponent ``alpha`` and angular momentum ``l`` is bounded by

        |chi(r)| <= |c| * (2 alpha / pi)^{3/4} * (1 + sqrt(alpha) r)^l
                        * exp(-alpha r^2)

    with the same deliberately generous polynomial factor as the pair
    estimator, solved for ``|chi| = tol`` by fixed-point iteration.

    Because one Gaussian decays with ``alpha`` while a pair decays with the
    reduced exponent ``mu = alpha*beta/(alpha+beta) <= alpha/2``, this is
    always the SHORTER of the two reaches -- by a factor approaching
    sqrt(2) for a pair drawn from the same diffuse shell. The two must not
    be substituted for one another.
    """
    if not (0.0 < float(tol) < 1.0):
        raise ValueError(
            f"estimate_rcut_amplitude: tol must be in (0, 1); got {tol}"
        )
    tol = float(tol)
    rcut = 0.0
    for shell in basis.shells():
        L = int(shell.l)
        exps = np.asarray(shell.exponents, dtype=float)
        coefs = np.asarray(shell.coefficients, dtype=float)
        for alpha, coef in zip(exps, coefs):
            alpha = float(alpha)
            if alpha <= 0.0:
                continue
            prefactor = abs(float(coef)) * (2.0 * alpha / math.pi) ** 0.75
            if prefactor <= tol:
                continue
            log_ratio = math.log(prefactor / tol)
            r = math.sqrt(log_ratio / alpha)
            for _ in range(int(n_refine)):
                arg = log_ratio + L * math.log1p(math.sqrt(alpha) * r)
                r = math.sqrt(max(arg / alpha, 0.0))
            if r > rcut:
                rcut = r
    return float(rcut)


def atom_pair_span_bohr(system) -> float:
    """Largest distance between two basis-carrying centres of the cell.

    ``direct_lattice_cells`` keeps a cell when the LATTICE vector satisfies
    ``|R| <= cutoff_bohr``, but the quantity that has to have decayed is the
    separation of the two basis-function centres, ``|r_B + R - r_A|``. Those
    differ by up to this span, so a cutoff derived from a pair separation
    must be padded by it before it is handed to ``LatticeSumOptions``.
    """
    coords = np.asarray(
        [np.asarray(atom.xyz, dtype=float) for atom in system.unit_cell],
        dtype=float,
    )
    if coords.shape[0] < 2:
        return 0.0
    diff = coords[:, None, :] - coords[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=-1)).max())


def bloch_overlap_cutoff_bohr(
    basis: BasisSet,
    system,
    *,
    tol: float = DEFAULT_BLOCH_OVERLAP_TOL,
    floor_bohr: float = 25.0,
) -> float:
    """Lattice-vector cutoff that keeps ``S(k)`` positive definite.

    ``S(k) = sum_R exp(i k . R) S(R)`` is the Gram matrix of the Bloch
    orbitals ``phi_mu^k(r) = sum_R exp(i k . R) chi_mu(r - R)`` and is
    therefore positive definite whenever the R sum is complete. Truncating
    it is not a benign approximation: at k = 0 every phase is +1 and the
    truncated sum remains a legitimate overlap, but away from the zone
    centre the dropped tail enters with alternating phase and can make the
    metric INDEFINITE. Negative eigenvalues of S(k) are a truncation
    artefact, never a basis-set property.

    Combines :func:`estimate_rcut_overlap_pair` with
    :func:`atom_pair_span_bohr`, then takes the max with ``floor_bohr`` so
    a compact basis never lands below the value the Gamma GPW / GAPW
    helpers already use.
    """
    pair_reach = estimate_rcut_overlap_pair(basis, tol=tol)
    return float(max(float(floor_bohr), pair_reach + atom_pair_span_bohr(system)))


def make_lattice_opts(
    basis: BasisSet,
    *,
    strategy: RcutStrategy = RcutStrategy.FLAT,
    base_opts: Optional[LatticeSumOptions] = None,
    precision: float = 1e-8,
    cutoff_bohr: Optional[float] = None,
    nuclear_cutoff_bohr: Optional[float] = None,
) -> LatticeSumOptions:
    """Factory: build a :class:`LatticeSumOptions` with the chosen
    cutoff strategy applied to ``cutoff_bohr``.

    Parameters
    ----------
    basis
        Basis set the lattice sum will run on (needed for
        :class:`RcutStrategy.PYSCF_AUTO` to compute per-shell rcuts).
    strategy
        :class:`RcutStrategy` choice. Default ``FLAT``.
    base_opts
        Optional :class:`LatticeSumOptions` to clone all fields from
        (``coulomb_method``, ``schwarz_threshold``, etc.). Defaults to
        a fresh ``LatticeSumOptions()`` if not supplied.
    precision
        Target precision for :class:`RcutStrategy.PYSCF_AUTO` (passed
        to :func:`estimate_rcut_pyscf`). Ignored for ``FLAT``.
    cutoff_bohr
        Manual override (any strategy). When set, this value is used
        as-is for ``cutoff_bohr``; the strategy choice still records
        intent but no auto-tuning is done.
    nuclear_cutoff_bohr
        Manual override for the (typically larger) nuclear-attraction
        lattice-sum cutoff. Defaults to ``base_opts.nuclear_cutoff_bohr``
        if not set.

    Returns
    -------
    LatticeSumOptions
        A fresh instance with ``cutoff_bohr`` set per the strategy.
        Other fields are copied from ``base_opts``.
    """
    dst = LatticeSumOptions()
    if base_opts is not None:
        for attr in dir(base_opts):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(base_opts, attr)
            except Exception:
                continue
            if callable(value):
                continue
            try:
                setattr(dst, attr, value)
            except Exception:
                pass

    if cutoff_bohr is not None:
        dst.cutoff_bohr = float(cutoff_bohr)
    elif strategy is RcutStrategy.FLAT:
        # Leave dst.cutoff_bohr at base_opts value (or the LatticeSumOptions
        # default of 15.0 if base_opts was None).
        pass
    elif strategy is RcutStrategy.PYSCF_AUTO:
        dst.cutoff_bohr = estimate_rcut_pyscf(basis, precision=precision)
    elif strategy is RcutStrategy.OVERLAP_PAIR:
        dst.cutoff_bohr = estimate_rcut_overlap_pair(basis, tol=precision)
    else:
        raise ValueError(f"make_lattice_opts: unknown RcutStrategy {strategy!r}")

    if nuclear_cutoff_bohr is not None:
        dst.nuclear_cutoff_bohr = float(nuclear_cutoff_bohr)
    return dst
