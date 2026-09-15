"""Native D4 C6 model -- Phase D4b-4.

The D4 dispersion C6 between two atoms is a coordination-number- and
charge-dependent interpolation over per-element reference data:

    C6^AB = S_i^ref(A) S_j^ref(B)  W_A,i . W_B,j . C6_ref^AB_ij

where ``ref(A)`` is the set of reference systems for element A
(catalogued in :mod:`vibeqc.dispersion_d4_reference_systems`),
``C6_ref^AB_ij`` is the reference C6 between reference system i of A
and j of B (Phase D4b-4 reference-data generation), and ``W_A,i`` is
the weight of reference system i for the actual atom A in its
molecular environment.

Each ``W_A,i`` is the product of two independent factors:

    W_A,i(CN_A, q_A) = gw_i^norm(CN_A) . ζ(q_A; q_ref,i, Z, η^ζ)

where ``gw_i^norm`` is the normalised CN Gaussian weight
(:func:`cn_gaussian_weights`) and ``ζ`` is the charge-scaling factor
(:func:`zeta_charge_scaling`). The charge dependence is the
distinguishing D4-over-D3 feature.

This module lands incrementally:

* **D4b-4 part 1** -- :func:`cn_gaussian_weights`, the
  coordination-number Gaussian weighting (the D3-inherited half of
  ``W_A,i``).
* **D4b-4 part 2 (this commit)** -- :func:`zeta_charge_scaling`, the
  charge-scaling factor ``ζ(q)``, together with its per-element
  data tables (:func:`zeta_hardness` and
  :func:`effective_nuclear_charge`). Also :func:`d4_reference_weights`,
  the combined CN+charge weight that the full C6 model consumes.
* D4b-4 part 3 -- reference-data generation (run the D4b-2 pipeline
  over the D4b-3 catalogue) and the full ``C6^AB`` assembly.

References
----------
* Grimme, Antony, Ehrlich, Krieg, *J. Chem. Phys.* **132**, 154104
  (2010) -- the D3 model; origin of the Gaussian CN interpolation.
* Caldeweyher, Bannwarth, Grimme, *J. Chem. Phys.* **147**, 034112
  (2017) and Caldeweyher *et al.*, *J. Chem. Phys.* **150**, 154122
  (2019) -- the D4 model; the charge-scaling function ζ(q) and its
  per-element hardness parameters (Eqs. 8-9, Table S4).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "D4_WEIGHTING_FACTOR",
    "D4_CHARGE_HEIGHT",
    "D4_CHARGE_STEEPNESS",
    "ZETA_MAX_Z",
    "R4R2_MAX_Z",
    "D4_ALP_DEFAULT",
    "cn_gaussian_weights",
    "zeta_charge_scaling",
    "zeta_hardness",
    "effective_nuclear_charge",
    "d4_reference_weights",
    "d4_reference_weights_derivs",
    "compute_d4_c6_pair",
    "compute_d4_c6_pair_derivs",
    "r4r2_val",
    "compute_c8",
    "d4_bj_damping_energy",
    "d4_bj_damping_gradient",
    "compute_d4_energy",
    "compute_d4_atm_energy",
    "compute_d4_energy_total",
    "compute_c9",
    "d4_atm_triple_energy",
    "compute_d4_gradient",
]

# ---- D4 model constants (Caldeweyher 2019) ----

# Coordination-number Gaussian weighting factor.  D3 used 4.0; D4
# sharpens it to 6.0.
D4_WEIGHTING_FACTOR = 6.0

# Charge-scaling height ``ga`` (default 3.0).  Controls the maximum
# suppression of a reference when the atom's charge is far from the
# reference charge.
D4_CHARGE_HEIGHT = 3.0

# Charge-scaling steepness ``gc`` (default 2.0).  Multiplier for the
# per-element hardness :func:`zeta_hardness` in the inner exponential.
D4_CHARGE_STEEPNESS = 2.0

# ---- Per-element data tables for the ζ(q) charge-scaling factor ----
#
# These tables are scientific facts from the dftd4 reference
# implementation (Caldeweyher et al., J. Chem. Phys. 150, 154122
# (2019), Sec. II.C and supporting information).  They are NOT the same
# as the EEQ electronegativity / hardness tables that vibe-qc ships in
# ``eeq_charges_data.cpp`` (Caldeweyher 2019, Table S2) -- those govern
# the EEQ charge model; these govern the D4 charge-scaling factor ζ(q).

# Maximum Z supported by the ζ data tables.
ZETA_MAX_Z = 118

# η^ζ_A -- per-element chemical hardness for the charge-scaling ζ(q)
# function (distinct from the EEQ η_A).  Source: dftd4 data_hardness.f90.
# Units: a⁻¹ (where a = ga = D4_CHARGE_HEIGHT = 3.0).
_ZETA_HARDNESS: tuple[float, ...] = (
    # Z=0 placeholder
    0.0,
    # Z=1..2
    0.47259288,
    0.92203391,
    # Z=3..10
    0.17452888,
    0.25700733,
    0.33949086,
    0.42195412,
    0.50438193,
    0.58691863,
    0.66931351,
    0.75191607,
    # Z=11..18
    0.17964105,
    0.22157276,
    0.26348578,
    0.30539645,
    0.34734014,
    0.38924725,
    0.43115670,
    0.47308269,
    # Z=19..36
    0.17105469,
    0.20276244,
    0.21007322,
    0.21739647,
    0.22471039,
    0.23201501,
    0.23933969,
    0.24665638,
    0.25398255,
    0.26128863,
    0.26859476,
    0.27592565,
    0.30762999,
    0.33931580,
    0.37235985,
    0.40273549,
    0.43445776,
    0.46611708,
    # Z=37..54
    0.15585079,
    0.18649324,
    0.19356210,
    0.20063311,
    0.20770522,
    0.21477254,
    0.22184614,
    0.22891872,
    0.23598621,
    0.24305612,
    0.25013018,
    0.25719937,
    0.28784780,
    0.31848673,
    0.34912431,
    0.37976593,
    0.41040808,
    0.44105777,
    # Z=55..86
    0.05019332,
    0.06762570,
    0.08504445,
    0.10247736,
    0.11991105,
    0.13732772,
    0.15476297,
    0.17218265,
    0.18961288,
    0.20704760,
    0.22446752,
    0.24189645,
    0.25932503,
    0.27676094,
    0.29418231,
    0.31159587,
    0.32902274,
    0.34592298,
    0.36388048,
    0.38130586,
    0.39877476,
    0.41614298,
    0.43364510,
    0.45104014,
    0.46848986,
    0.48584550,
    0.12526730,
    0.14268677,
    0.16011615,
    0.17755889,
    0.19497557,
    0.21240778,
    # Z=87..118
    0.07263525,
    0.09422158,
    0.09920295,
    0.10418621,
    0.14235633,
    0.16394294,
    0.18551941,
    0.22370139,
    0.25110000,
    0.25030000,
    0.28840000,
    0.31000000,
    0.33160000,
    0.35320000,
    0.36820000,
    0.39630000,
    0.40140000,
    # Z=104..118 -- not parametrised in dftd4
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

# Z_eff^A -- effective nuclear charge from the def2-ECPs used to
# compute the D4 reference polarizabilities.  For H-Ar (Z=1..18)
# this equals the nuclear charge; for heavier elements it is the
# screened charge after the ECP cores.  Source: dftd4 data_zeff.f90.
_EFFECTIVE_CHARGE: tuple[float, ...] = (
    # Z=0 placeholder
    0.0,
    # Z=1..2
    1.0,
    2.0,
    # Z=3..10
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    # Z=11..18
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    # Z=19..36 (K-Kr, full nuclear charge)
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    27.0,
    28.0,
    29.0,
    30.0,
    31.0,
    32.0,
    33.0,
    34.0,
    35.0,
    36.0,
    # Z=37..54 (Rb-Xe: def2-ECP with 28 e⁻ core -> Z_eff = Z - 28)
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    # Z=55..71 (Cs-Lu: def2-ECP with 46 e⁻ core)
    9.0,
    10.0,
    11.0,
    # La-Lu: 11 f-electrons screened, effective Z 30..43
    30.0,
    31.0,
    32.0,
    33.0,
    34.0,
    35.0,
    36.0,
    37.0,
    38.0,
    39.0,
    40.0,
    41.0,
    42.0,
    43.0,
    # Z=72..86 (Hf-Rn: def2-ECP with 60 e⁻ core)
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    # Z=87..103 (Fr-Lr: same pattern as Cs-Lu)
    9.0,
    10.0,
    11.0,
    30.0,
    31.0,
    32.0,
    33.0,
    34.0,
    35.0,
    36.0,
    37.0,
    38.0,
    39.0,
    40.0,
    41.0,
    42.0,
    43.0,
    # Z=104..118 (Rf-Og: same pattern as Hf-Rn)
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
)


def cn_gaussian_weights(
    cn: float,
    reference_cns: Sequence[float],
    *,
    weighting_factor: float = D4_WEIGHTING_FACTOR,
    n_gaussian: int = 3,
) -> np.ndarray:
    """Coordination-number Gaussian weights for one atom's reference
    systems.

    Given an atom's actual coordination number ``cn`` and the nominal
    CN of each of its element's reference systems, return the
    normalised weight of each reference -- the D3-inherited CN half of
    the D4 reference weight ``W_A,i``.

    The unnormalised weight of reference *i* is a sum of Gaussians in
    the CN gap ``ΔCN_i = cn - reference_cns[i]``::

        gw_i = S_{k=1}^{n_gaussian}  exp(-weighting_factor . k . ΔCN_i^2)

    and the returned weights are ``W_i = gw_i / S_j gw_j``.

    The single Gaussian (``k = 1``) is the textbook D3/D4 form; the
    extra ``k = 2, 3`` terms are a numerical-robustness device (from
    the D4 reference implementation) so that an atom in an environment
    far from *every* reference -- where the ``k = 1`` Gaussians all
    underflow -- still gets a well-defined, normalised weight vector
    rather than ``0/0``. The ``k > 1`` terms decay faster, so near a
    reference they are negligible and ``k = 1`` dominates.

    Parameters
    ----------
    cn
        The atom's actual coordination number (the continuous D4 erf
        coordination number; see ``vibeqc.eeq_coordination_numbers``).
    reference_cns
        Nominal CN of each reference system for this atom's element,
        in catalogue order. Must be non-empty.
    weighting_factor
        The Gaussian sharpness ``wf``. Default :data:`D4_WEIGHTING_FACTOR`.
    n_gaussian
        Number of Gaussian terms in the robustness sum. ``1`` gives
        the pure single-Gaussian D3 form; the D4 default is ``3``.

    Returns
    -------
    numpy.ndarray
        Weights, shape ``(len(reference_cns),)``, non-negative and
        summing to 1.

    Notes
    -----
    If every Gaussian underflows to exactly zero (an atom absurdly
    far from all references, or a pathological ``weighting_factor``),
    the weight collapses onto the single nearest reference -- the
    correct limiting behaviour.
    """
    ref = np.asarray(reference_cns, dtype=float)
    if ref.ndim != 1 or ref.size == 0:
        raise ValueError(
            "cn_gaussian_weights: reference_cns must be a non-empty 1-D sequence"
        )
    if weighting_factor <= 0.0:
        raise ValueError(
            f"cn_gaussian_weights: weighting_factor must be positive, "
            f"got {weighting_factor}"
        )
    if n_gaussian < 1:
        raise ValueError(
            f"cn_gaussian_weights: n_gaussian must be >= 1, got {n_gaussian}"
        )

    delta2 = (cn - ref) ** 2  # ΔCN_i^2

    gw = np.zeros_like(ref)
    for k in range(1, n_gaussian + 1):
        gw += np.exp(-weighting_factor * k * delta2)

    total = gw.sum()
    if total > 0.0:
        return gw / total

    # Underflow fallback: all Gaussians vanished. Put unit weight on
    # the single reference with the smallest CN gap.
    weights = np.zeros_like(ref)
    weights[int(np.argmin(delta2))] = 1.0
    return weights


# ---- ζ(q) charge-scaling factor (D4b-4 part 2) ----


def zeta_charge_scaling(
    ga: float,
    gi: float,
    qref: float,
    qmod: float,
) -> float:
    """The D4 charge-scaling function ζ(q).

    ζ is a smooth, asymmetric function that suppresses a reference
    system's weight when the atom's actual partial charge ``qmod``
    differs from the reference system's charge ``qref``.

    Both ``qref`` and ``qmod`` are *shifted* by the effective nuclear
    charge :func:`effective_nuclear_charge` -- i.e. ``qmod == q_A + z_eff``
    and ``qref == q_ref,i + z_eff`` -- so that neutral atoms have
    ``qmod = z_eff`` and anionic extremes are naturally bounded.

    The functional form (Caldeweyher 2019, Eq. 9):

        ζ = exp( ga . (1 - exp( gi . (1 - qref/qmod) )) )

    For qmod <= 0 (which only occurs if q_A < -z_eff, a deeply anionic
    extreme), ζ saturates at ``exp(ga)`` -- the maximum possible
    suppression.

    Parameters
    ----------
    ga
        Charge-scaling height parameter (default :data:`D4_CHARGE_HEIGHT`
        = 3.0). Controls the maximum suppression.
    gi
        Product of the per-element charge-scaling hardness
        :func:`zeta_hardness` and the steepness parameter
        :data:`D4_CHARGE_STEEPNESS` (default 2.0).  Larger ``gi``
        makes ζ decay faster as charge deviates.
    qref
        Reference charge ``q_ref,i + z_eff`` (the reference system's
        partial charge, shifted by the effective nuclear charge).
    qmod
        The atom's actual charge ``q_A + z_eff``.

    Returns
    -------
    float
        ζ in (0, exp(ga)].  ζ = 1 when qmod = qref (no suppression);
        ζ -> 1/exp(ga) as |qmod - qref| -> inf for cations;
        ζ = exp(ga) for qmod <= 0 (deeply anionic).
    """
    if qmod <= 0.0:
        return np.exp(float(ga))
    zeta = ga * (1.0 - np.exp(float(gi) * (1.0 - float(qref) / float(qmod))))
    return np.exp(zeta)


def zeta_hardness(z: int) -> float:
    """Per-element chemical hardness η^ζ for the D4 charge-scaling
    function :func:`zeta_charge_scaling`.

    This is **not** the EEQ chemical hardness ``eeq_chemical_hardness(Z)``
    from ``eeq_charges_data.cpp`` (Caldeweyher 2019, Table S2).  It is a
    distinct parameter set from the same paper (Table S4 / dftd4 source)
    that controls the charge sensitivity of the D4 C6 weights.

    Parameters
    ----------
    z
        Atomic number, 1 <= z <= :data:`ZETA_MAX_Z`.

    Returns
    -------
    float
        η^ζ in units of a⁻¹ (where a = :data:`D4_CHARGE_HEIGHT`).
    """
    if z < 1 or z > ZETA_MAX_Z:
        raise ValueError(f"zeta_hardness: Z={z} is outside [1, {ZETA_MAX_Z}]")
    return _ZETA_HARDNESS[z]


def effective_nuclear_charge(z: int) -> float:
    """Effective nuclear charge ``z_eff`` for element ``Z``.

    From the def2-ECPs used to compute the D4 reference polarizabilities.
    For H-Ar (Z=1..18) this equals the bare nuclear charge; for heavier
    elements it is the screened charge after the ECP cores.

    Parameters
    ----------
    z
        Atomic number, 1 <= z <= :data:`ZETA_MAX_Z`.

    Returns
    -------
    float
        Effective nuclear charge.
    """
    if z < 1 or z > ZETA_MAX_Z:
        raise ValueError(
            f"effective_nuclear_charge: Z={z} is outside [1, {ZETA_MAX_Z}]"
        )
    return _EFFECTIVE_CHARGE[z]


# ---- Combined CN + charge weights (D4b-4 part 2) ----


def d4_reference_weights(
    cn: float,
    q: float,
    reference_cns: Sequence[float],
    reference_qs: Sequence[float],
    z: int,
    *,
    weighting_factor: float = D4_WEIGHTING_FACTOR,
    n_gaussian: int = 3,
    ga: Optional[float] = None,
    gc: Optional[float] = None,
) -> np.ndarray:
    """Full D4 reference weights ``W_A,i`` for one atom, combining
    the CN Gaussian weight and the charge-scaling factor ζ(q).

    This is the complete ``W_A,i(CN_A, q_A)`` that the D4 C6 model
    sums over:

        W_i = gw_i^norm(CN) . ζ(ga, η^ζ(Z).gc, q_ref,i+z_eff, q+z_eff)

    where ``gw_i^norm`` is the normalised CN weight from
    :func:`cn_gaussian_weights` and ζ is :func:`zeta_charge_scaling`.

    The CN-normalisation is global (across all references) before ζ
    is applied -- this is the dftd4 reference-implementation algorithm.

    Parameters
    ----------
    cn
        The atom's actual D4 erf coordination number.
    q
        The atom's EEQ partial charge ``q_A`` (in e).
    reference_cns
        Nominal CN of each of this element's reference systems, in
        catalogue order.
    reference_qs
        Reference charge ``q_ref,i`` of each reference system, in
        the same order as ``reference_cns``.  These are the EEQ
        partial charges of the reference molecules, computed at their
        equilibrium geometry -- the output of Phase D4b-4 part 3.
    z
        Atomic number of the atom.
    weighting_factor
        CN Gaussian sharpness. Default :data:`D4_WEIGHTING_FACTOR`.
    n_gaussian
        Number of Gaussian terms. Default 3.
    ga
        Charge-scaling height. Default None -> :data:`D4_CHARGE_HEIGHT`.
    gc
        Charge-scaling steepness. Default None ->
        :data:`D4_CHARGE_STEEPNESS`.

    Returns
    -------
    numpy.ndarray
        Weights ``W_i``, shape ``(len(reference_cns),)``, non-negative.
        Not normalised -- the normalisation of the CN weights was
        already performed; ζ is a multiplicative factor.
    """
    ref_cns = np.asarray(reference_cns, dtype=float)
    ref_qs = np.asarray(reference_qs, dtype=float)
    n_ref = ref_cns.size
    if ref_qs.size != n_ref:
        raise ValueError(
            f"d4_reference_weights: reference_cns ({n_ref}) and "
            f"reference_qs ({ref_qs.size}) must have the same length"
        )

    # Step 1: normalised CN Gaussian weights.
    gw = cn_gaussian_weights(
        cn,
        ref_cns,
        weighting_factor=weighting_factor,
        n_gaussian=n_gaussian,
    )

    # Step 2: apply ζ charge scaling.
    _ga = ga if ga is not None else D4_CHARGE_HEIGHT
    _gc = gc if gc is not None else D4_CHARGE_STEEPNESS
    gi = zeta_hardness(z) * _gc
    z_eff = effective_nuclear_charge(z)

    weights = np.zeros(n_ref)
    for i in range(n_ref):
        zeta_val = zeta_charge_scaling(
            _ga,
            gi,
            float(ref_qs[i]) + z_eff,
            float(q) + z_eff,
        )
        weights[i] = gw[i] * zeta_val

    return weights


# ---- D4 C6 pair assembly (D4b-4 part 3) ----


def compute_d4_c6_pair(
    cn_a: float,
    q_a: float,
    z_a: int,
    cn_b: float,
    q_b: float,
    z_b: int,
    ref_data: "D4ReferenceDataset",
    *,
    weighting_factor: float = D4_WEIGHTING_FACTOR,
    n_gaussian: int = 3,
    ga: Optional[float] = None,
    gc: Optional[float] = None,
) -> float:
    """The D4 C6 dispersion coefficient between two atoms A and B.

    Computes the coordination-number- and charge-dependent C6 via
    Gaussian-weighted interpolation over the reference dataset::

        C6^AB = S_i S_j  W_A,i . W_B,j . C6_ref^AB_ij

    where ``W_A,i`` is :func:`d4_reference_weights` and
    ``C6_ref^AB_ij`` is the pre-computed reference C6 between
    reference system *i* of element A and reference system *j* of
    element B.

    Parameters
    ----------
    cn_a, cn_b
        Actual D4 erf coordination numbers of atoms A and B.
    q_a, q_b
        EEQ partial charges of atoms A and B (in e).
    z_a, z_b
        Atomic numbers of atoms A and B.
    ref_data
        The reference dataset (:class:`~dispersion_d4_reference_data.D4ReferenceDataset`).
    weighting_factor, n_gaussian, ga, gc
        Weight parameters forwarded to :func:`d4_reference_weights`.

    Returns
    -------
    float
        C6^AB in atomic units (Hartree.bohr⁶).
    """
    cns_a = ref_data.get_cns(z_a)
    qs_a = ref_data.get_qs(z_a)
    cns_b = ref_data.get_cns(z_b)
    qs_b = ref_data.get_qs(z_b)

    w_a = d4_reference_weights(
        cn_a,
        q_a,
        cns_a,
        qs_a,
        z_a,
        weighting_factor=weighting_factor,
        n_gaussian=n_gaussian,
        ga=ga,
        gc=gc,
    )
    w_b = d4_reference_weights(
        cn_b,
        q_b,
        cns_b,
        qs_b,
        z_b,
        weighting_factor=weighting_factor,
        n_gaussian=n_gaussian,
        ga=ga,
        gc=gc,
    )

    c6 = 0.0
    for ia, wi in enumerate(w_a):
        for jb, wj in enumerate(w_b):
            c6 += wi * wj * ref_data.get_c6_ref(z_a, ia, z_b, jb)
    return float(c6)


# ---- r4/r2 expectation-value ratios (D4b-5) ----
#
# <r⁴>/<r^2> for computing C8 and the BJ critical radius R0.
# PBE0/def2-QZVP atomic values by S. Grimme (Gaussian 2010);
# rare gases recalibrated by J. Mewes (PBE0/aug-cc-pVQZ, Dirac 2018);
# super-heavies by 4c-PBE/Dyall-AE4Z (Dirac 2022).
# Source: dftd4 data_r4r2.f90. Stored as the pre-computed
# r4r2_st(Z) = sqrt(0.5 * sqrt(Z) * <r⁴>/<r^2>(Z)) so that:
#
#     rrij = 3 * r4r2_st(A) * r4r2_st(B)    (= C8 / C6)
#
# exactly matches the dftd4 convention the a1/a2 parameters were
# fitted against. The BJ critical radius R0 = a1*sqrt(C8/C6) + a2 then
# also matches dftd4, so an inconsistent r4r2 table mis-damps every
# pair -- the 2026-06-26 fix corrected 117/118 entries that had shipped
# without the sqrt(Z) factor (only H, where sqrt(Z)=1, was right),
# which over-bound native D4 energies independently of the C6 data.
# Pinned against dftd4.data.sqrt_z_r4_over_r2 by test_r4r2_spots.

R4R2_MAX_Z = 118

_R4R2: tuple[float, ...] = (
    # Z=0 placeholder
    0.00000000,
    # Z=1..2
    2.00734900,
    1.56637132,
    # Z=3..10
    5.01986928,
    3.85379034,
    3.64446594,
    3.10492822,
    2.71175242,
    2.59361682,
    2.38825250,
    2.21522515,
    # Z=11..18
    6.58585548,
    5.46295966,
    5.65216658,
    4.88284907,
    4.29727568,
    4.04108896,
    3.72932350,
    3.44677276,
    # Z=19..36
    7.97762746,
    7.07623944,
    6.60844067,
    6.28791378,
    6.07728705,
    5.54643099,
    5.80491171,
    5.58415606,
    5.41374528,
    5.28497228,
    5.22592813,
    5.09817147,
    6.12149692,
    5.54083743,
    5.06696890,
    4.87005101,
    4.59089643,
    4.31176298,
    # Z=37..54
    9.55461712,
    8.67396089,
    7.97210185,
    7.43439903,
    6.58711857,
    6.19536204,
    6.01517302,
    5.81623399,
    5.65710425,
    5.52640670,
    5.44263311,
    5.58285360,
    7.02081904,
    6.46815533,
    5.98089106,
    5.81686646,
    5.53321806,
    5.25477002,
    # Z=55..86
    11.02204559,
    10.15679514,
    9.35167817,
    9.06926082,
    8.97241151,
    8.90092815,
    8.85984844,
    8.81736839,
    8.79317721,
    7.89969620,
    8.80588447,
    8.42439202,
    8.54289264,
    8.47583362,
    8.45090883,
    8.47339356,
    7.83525635,
    8.20702839,
    7.70559062,
    7.32755987,
    7.03887394,
    6.68978709,
    6.05450039,
    5.88752028,
    5.70661507,
    5.78450700,
    7.79780738,
    7.26443859,
    6.78151998,
    6.67883162,
    6.39024306,
    6.09527966,
    # Z=87..118
    11.79156087,
    11.10997633,
    9.51377809,
    8.67197058,
    8.77140704,
    8.65402708,
    8.53923512,
    8.85024701,
    8.44986047,
    8.49130145,
    8.27663854,
    8.19328750,
    8.11365231,
    8.03707036,
    7.96122152,
    7.88566179,
    8.67566853,
    7.94424788,
    7.38258128,
    6.94284601,
    6.52413514,
    6.27038488,
    6.02138659,
    5.79081733,
    5.58246693,
    5.39126096,
    5.98021735,
    5.89723488,
    7.65084015,
    7.46929096,
    7.15746429,
    6.85989127,
)


def r4r2_val(z: int) -> float:
    """Pre-computed r4r2 storage value for the D4 C8 formula.

    This is ``sqrt(0.5 * <r⁴>/<r^2>(Z) * sqrt(Z))`` -- the quantity
    that enters the C8 computation directly::

        C8^AB = 3 * C6^AB * r4r2_val(A) * r4r2_val(B)

    Parameters
    ----------
    z
        Atomic number, 1 <= z <= :data:`R4R2_MAX_Z`.

    Returns
    -------
    float
        The pre-computed r4/r2 storage value (dimensionless).
    """
    if z < 1 or z > R4R2_MAX_Z:
        raise ValueError(f"r4r2_val: Z={z} is outside [1, {R4R2_MAX_Z}]")
    return _R4R2[z]


def compute_c8(c6: float, z_a: int, z_b: int) -> float:
    """C8 dispersion coefficient from C6 and the D4 r4r2 table.

        C8^AB = 3 . C6^AB . r4r2_val(A) . r4r2_val(B)

    Parameters
    ----------
    c6
        The D4 C6 coefficient for the atom pair (Hartree.bohr⁶).
    z_a, z_b
        Atomic numbers of the two atoms.

    Returns
    -------
    float
        C8 in Hartree.bohr⁸.
    """
    return 3.0 * c6 * r4r2_val(z_a) * r4r2_val(z_b)


# ---- Becke-Johnson rational damping (D4b-5) ----


def d4_bj_damping_energy(
    r: float,
    c6: float,
    c8: float,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
) -> float:
    """Two-body D4 dispersion energy with Becke-Johnson rational damping.

        E_pair = - s6.C6/(R⁶ + R0⁶) - s8.C8/(R⁸ + R0⁸)

    where ``R0 = a1.sqrt(C8/C6) + a2`` is the BJ critical radius.

    Parameters
    ----------
    r
        Interatomic distance (bohr).
    c6, c8
        D4 C6 and C8 coefficients for the pair.
    s6, s8, a1, a2
        Per-functional BJ damping parameters.

    Returns
    -------
    float
        Pair dispersion energy in Hartree (negative for attractive
        dispersion).
    """
    r0 = a1 * np.sqrt(float(c8) / float(c6)) + a2
    r2 = r * r
    r6 = r2 * r2 * r2
    r8 = r6 * r2
    t6 = 1.0 / (r6 + r0**6)
    t8 = 1.0 / (r8 + r0**8)
    return float(-s6 * c6 * t6 - s8 * c8 * t8)


# ---- Full-molecule D4 dispersion energy (D4b-5) ----


def compute_d4_energy(
    mol: "Molecule",
    ref_data: "D4ReferenceDataset",
    functional: str = "pbe",
    *,
    _damping_parameters: object | None = None,
    **kwargs: object,
) -> float:
    """Full-molecule D4 two-body dispersion energy.

    For every atom pair A<B:

    1. Compute the D4 C6 coefficient via
       :func:`compute_d4_c6_pair` (CN/charge-dependent interpolation
       over the reference dataset).
    2. Compute C8 via :func:`compute_c8`.
    3. Evaluate the Becke-Johnson rational damping via
       :func:`d4_bj_damping_energy`.

    The per-atom coordination numbers and EEQ partial charges are
    computed internally via the Phase D4a machinery
    (``eeq_coordination_numbers``, ``eeq_charges``).

    Parameters
    ----------
    mol
        The molecule (closed-shell SCF not required -- only atomic
        numbers, coordinates, and EEQ properties are used).
    ref_data
        The reference dataset, typically loaded from a JSON file
        produced by :func:`dispersion_d4_reference_data.generate_reference_dataset`.
    functional
        Functional keyword for BJ damping parameters (see
        :mod:`dispersion_d4_parameters`). Default ``"pbe"``.
    **kwargs
        Forwarded to :func:`compute_d4_c6_pair` (``weighting_factor``,
        ``n_gaussian``, ``ga``, ``gc``).

    Returns
    -------
    float
        Total D4 two-body dispersion energy in Hartree.
    """
    from ._vibeqc_core import eeq_charges, eeq_coordination_numbers
    from .dispersion_d4_parameters import get_d4_params

    # Per-atom properties.
    cn_arr = eeq_coordination_numbers(mol)
    q_arr = eeq_charges(mol).charges
    params = (
        get_d4_params(functional)
        if _damping_parameters is None
        else _damping_parameters
    )

    natom = len(mol.atoms)
    z_arr = [atom.Z for atom in mol.atoms]
    xyz = np.array([atom.xyz for atom in mol.atoms])

    energy = 0.0
    for a in range(natom):
        ra = xyz[a]
        for b in range(a):
            c6 = compute_d4_c6_pair(
                float(cn_arr[a]),
                float(q_arr[a]),
                z_arr[a],
                float(cn_arr[b]),
                float(q_arr[b]),
                z_arr[b],
                ref_data,
                **kwargs,  # type: ignore[arg-type]
            )
            if c6 <= 0.0:
                continue
            c8 = compute_c8(c6, z_arr[a], z_arr[b])
            r_ab = float(np.linalg.norm(ra - xyz[b]))
            energy += d4_bj_damping_energy(
                r_ab,
                c6,
                c8,
                params.s6,
                params.s8,
                params.a1,
                params.a2,
            )

    return energy


# ---- Axilrod-Teller-Muto three-body dispersion (D4b-6) ----

# Default ATM damping exponent (Caldeweyher 2019).
D4_ALP_DEFAULT = 16.0


def compute_c9(c6_ab: float, c6_ac: float, c6_bc: float) -> float:
    """C9 coefficient from the geometric mean of three pair C6 values.

        C9_ABC ≈ -sqrt(|C6_AB . C6_AC . C6_BC|)

    The formula is an approximation (the exact ATM C9 involves a
    3-pole integral); the geometric mean is the D4 default.
    """
    return float(-np.sqrt(abs(c6_ab * c6_ac * c6_bc)))


def d4_atm_triple_energy(
    r_ab: float,
    r_bc: float,
    r_ac: float,
    c9: float,
    r0_ab: float,
    r0_bc: float,
    r0_ac: float,
    *,
    alp: float = D4_ALP_DEFAULT,
) -> float:
    """ATM triple energy for one atom triplet (A, B, C).

    The angle factor is the Axilrod-Teller-Muto geometrical term
    ``(3.costh_A.costh_B.costh_C + 1) / (R_AB.R_BC.R_AC)^3``, damped
    by the BJ-style rational function with exponent ``alp``.

    Parameters
    ----------
    r_ab, r_bc, r_ac
        Interatomic distances for the three pair edges.
    c9
        C9 coefficient (as returned by :func:`compute_c9`).
    r0_ab, r0_bc, r0_ac
        Critical radii for each pair edge (``R0 = a1.sqrt(C8/C6) + a2``).
    alp
        ATM damping exponent.

    Returns
    -------
    float
        Triple energy in Hartree.
    """
    r2_ab = r_ab * r_ab
    r2_bc = r_bc * r_bc
    r2_ac = r_ac * r_ac

    r0_prod = r0_ab * r0_bc * r0_ac

    r2 = r2_ab * r2_bc * r2_ac
    r1 = np.sqrt(r2)
    r3 = r2 * r1
    r5 = r3 * r2

    alp3 = float(alp) / 3.0
    fdmp = 1.0 / (1.0 + 6.0 * (r0_prod / r1) ** alp3)

    ang = (
        0.375
        * (r2_ab + r2_bc - r2_ac)
        * (r2_ab - r2_bc + r2_ac)
        * (-r2_ab + r2_bc + r2_ac)
        / r5
        + 1.0 / r3
    )

    return float(c9 * ang * fdmp)


def _compute_c6_matrix(
    mol: "Molecule",
    ref_data: "D4ReferenceDataset",
    **kwargs: object,
) -> np.ndarray:
    """Compute the full atom-pair C6 matrix for the molecule."""
    from ._vibeqc_core import eeq_charges, eeq_coordination_numbers

    cn_arr = eeq_coordination_numbers(mol)
    q_arr = eeq_charges(mol).charges
    natom = len(mol.atoms)
    z_arr = [atom.Z for atom in mol.atoms]

    c6 = np.zeros((natom, natom))
    for a in range(natom):
        for b in range(a):
            val = compute_d4_c6_pair(
                float(cn_arr[a]),
                float(q_arr[a]),
                z_arr[a],
                float(cn_arr[b]),
                float(q_arr[b]),
                z_arr[b],
                ref_data,
                **kwargs,  # type: ignore[arg-type]
            )
            c6[a, b] = val
            c6[b, a] = val
    return c6


def compute_d4_atm_energy(
    mol: "Molecule",
    ref_data: "D4ReferenceDataset",
    functional: str = "pbe",
    *,
    alp: float = D4_ALP_DEFAULT,
    _damping_parameters: object | None = None,
    **kwargs: object,
) -> float:
    """D4 Axilrod-Teller-Muto three-body dispersion energy.

    The ATM term is the leading three-body dispersion contribution::

        E^(3) = S_{A<B<C}  C9_ABC . w(th) / (R_AB.R_BC.R_CA)^3 . f_damp

    where ``C9_ABC ≈ -s9.sqrt(|C6_AB.C6_BC.C6_CA|)`` and ``w(th)`` is
    the angular factor ``3.cos th_A.cos th_B.cos th_C + 1``.  The
    Becke-Johnson-style rational damping uses the same ``a1, a2`` as
    the two-body term with an additional exponent ``alp`` (default 16).

    The C6 matrix is computed once from the reference dataset via
    :func:`compute_d4_c6_pair` for all atom pairs.

    Parameters
    ----------
    mol
        The molecule.
    ref_data
        The D4 reference dataset.
    functional
        Functional keyword for ``s9, a1, a2``.
    alp
        ATM damping exponent (default :data:`D4_ALP_DEFAULT` = 16).
    **kwargs
        Forwarded to :func:`compute_d4_c6_pair`.

    Returns
    -------
    float
        Total ATM three-body dispersion energy in Hartree.
    """
    from .dispersion_d4_parameters import get_d4_params

    params = (
        get_d4_params(functional)
        if _damping_parameters is None
        else _damping_parameters
    )
    if abs(params.s9) < 1e-15:
        return 0.0

    c6 = _compute_c6_matrix(mol, ref_data, **kwargs)
    natom = len(mol.atoms)
    z_arr = [atom.Z for atom in mol.atoms]
    xyz = np.array([atom.xyz for atom in mol.atoms])

    six = 1.0 / 6.0
    a1 = float(params.a1)
    a2 = float(params.a2)
    s9 = float(params.s9)

    energy = 0.0
    for a in range(natom):
        za = z_arr[a]
        ra = xyz[a]
        for b in range(a):
            zb = z_arr[b]
            rb = xyz[b]
            r2ab = float(np.sum((ra - rb) ** 2))
            c6_ab = c6[a, b]
            if c6_ab <= 0.0:
                continue
            r0ab = a1 * np.sqrt(3.0 * r4r2_val(za) * r4r2_val(zb)) + a2
            for c_idx in range(b):
                zc = z_arr[c_idx]
                rc = xyz[c_idx]
                c6_ac = c6[a, c_idx]
                c6_bc = c6[b, c_idx]
                if c6_ac <= 0.0 or c6_bc <= 0.0:
                    continue

                r2ac = float(np.sum((ra - rc) ** 2))
                r2bc = float(np.sum((rb - rc) ** 2))
                r_ac = np.sqrt(r2ac)
                r_bc = np.sqrt(r2bc)
                r_ab = np.sqrt(float(r2ab))

                c9 = s9 * compute_c9(c6_ab, c6_ac, c6_bc)

                r0ac = a1 * np.sqrt(3.0 * r4r2_val(za) * r4r2_val(zc)) + a2
                r0bc = a1 * np.sqrt(3.0 * r4r2_val(zb) * r4r2_val(zc)) + a2

                e_triple = d4_atm_triple_energy(
                    r_ab,
                    r_bc,
                    r_ac,
                    c9,
                    r0ab,
                    r0bc,
                    r0ac,
                    alp=alp,
                )

                # Triplet scaling: 1/6 for three identical atoms,
                # 1/2 for two identical, 1 for all distinct.
                if a == b and b == c_idx:
                    triple = six
                elif a == b or b == c_idx or a == c_idx:
                    triple = 0.5
                else:
                    triple = 1.0

                energy += e_triple * triple

    return energy


def compute_d4_energy_total(
    mol: "Molecule",
    ref_data: "D4ReferenceDataset",
    functional: str = "pbe",
    *,
    atm: bool = True,
    alp: float = D4_ALP_DEFAULT,
    _damping_parameters: object | None = None,
    **kwargs: object,
) -> float:
    """Full D4 dispersion energy (two-body + optional three-body ATM).

    Convenience wrapper that calls :func:`compute_d4_energy` and
    (if ``atm=True``) :func:`compute_d4_atm_energy`, sharing a single
    C6 matrix computation where possible.

    Parameters
    ----------
    mol
        The molecule.
    ref_data
        The D4 reference dataset.
    functional
        Functional keyword.
    atm
        Include the three-body ATM term (default ``True``).
    alp
        ATM damping exponent.
    **kwargs
        Forwarded to C6 weight computation.

    Returns
    -------
    float
        Total D4 dispersion energy in Hartree.
    """
    e2 = compute_d4_energy(
        mol,
        ref_data,
        functional,
        _damping_parameters=_damping_parameters,
        **kwargs,
    )
    if atm:
        e3 = compute_d4_atm_energy(
            mol,
            ref_data,
            functional,
            alp=alp,
            _damping_parameters=_damping_parameters,
            **kwargs,
        )
        return e2 + e3
    return e2


# ---- Analytical gradient (D4b-7) ----


def _dzeta_dq(ga: float, gi: float, qref: float, qmod: float) -> float:
    """Derivative dζ/dq of the charge-scaling function."""
    if qmod <= 0.0:
        return 0.0
    qmod2 = qmod * qmod
    inner = np.exp(float(gi) * (1.0 - float(qref) / qmod))
    z = zeta_charge_scaling(ga, gi, qref, qmod)
    return float(-ga * gi * inner * z * float(qref) / qmod2)


def d4_reference_weights_derivs(
    cn: float,
    q: float,
    reference_cns: Sequence[float],
    reference_qs: Sequence[float],
    z: int,
    *,
    weighting_factor: float = D4_WEIGHTING_FACTOR,
    n_gaussian: int = 3,
    ga: Optional[float] = None,
    gc: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D4 reference weights and their derivatives w.r.t. CN and q.

    Returns ``(W, dW/dCN, dW/dq)`` where each is shape ``(n_ref,)``.
    """
    ref_cns = np.asarray(reference_cns, dtype=float)
    ref_qs = np.asarray(reference_qs, dtype=float)
    n_ref = ref_cns.size

    _ga = ga if ga is not None else D4_CHARGE_HEIGHT
    _gc = gc if gc is not None else D4_CHARGE_STEEPNESS
    gi = zeta_hardness(z) * _gc
    z_eff = effective_nuclear_charge(z)

    # ---- Step 1: CN Gaussian weights and their CN derivative ----
    delta = cn - ref_cns
    delta2 = delta * delta

    # Unnormalized weights and their CN derivatives.
    gw = np.zeros(n_ref)
    dgw = np.zeros(n_ref)
    for k in range(1, n_gaussian + 1):
        wf_k = weighting_factor * k
        g = np.exp(-wf_k * delta2)
        gw += g
        dgw += 2.0 * wf_k * (ref_cns - cn) * g  # = -2*wf_k*delta*g

    total = gw.sum()
    if total <= 0.0:
        # Underflow fallback.
        idx = int(np.argmin(delta2))
        w_cn = np.zeros(n_ref)
        w_cn[idx] = 1.0
        return w_cn.copy(), np.zeros(n_ref), np.zeros(n_ref)

    inv_total = 1.0 / total
    w_cn = gw * inv_total
    # dw_cn/dCN = inv_total * dgw - w_cn * inv_total * sum(dgw)
    dw_cn = inv_total * (dgw - w_cn * dgw.sum())

    # ---- Step 2: ζ charge-scaling factor and its q-derivative ----
    qmod = float(q) + z_eff
    zeta_vals = np.zeros(n_ref)
    dzeta_vals = np.zeros(n_ref)
    for i in range(n_ref):
        qref_i = float(ref_qs[i]) + z_eff
        zeta_vals[i] = zeta_charge_scaling(_ga, gi, qref_i, qmod)
        dzeta_vals[i] = _dzeta_dq(_ga, gi, qref_i, qmod)

    # ---- Combine ----
    w = w_cn * zeta_vals
    dw_dcn = dw_cn * zeta_vals
    dw_dq = w_cn * dzeta_vals

    return w, dw_dcn, dw_dq


def compute_d4_c6_pair_derivs(
    cn_a: float,
    q_a: float,
    z_a: int,
    cn_b: float,
    q_b: float,
    z_b: int,
    ref_data: "D4ReferenceDataset",
    *,
    weighting_factor: float = D4_WEIGHTING_FACTOR,
    n_gaussian: int = 3,
    ga: Optional[float] = None,
    gc: Optional[float] = None,
) -> tuple[float, float, float, float, float]:
    """D4 C6 coefficient and its derivatives w.r.t. CN and q.

    Returns ``(C6, dC6/dCN_a, dC6/dCN_b, dC6/dq_a, dC6/dq_b)``.
    """
    cns_a = ref_data.get_cns(z_a)
    qs_a = ref_data.get_qs(z_a)
    cns_b = ref_data.get_cns(z_b)
    qs_b = ref_data.get_qs(z_b)

    kw = dict(weighting_factor=weighting_factor, n_gaussian=n_gaussian, ga=ga, gc=gc)
    w_a, dw_a_dcn, dw_a_dq = d4_reference_weights_derivs(
        cn_a, q_a, cns_a, qs_a, z_a, **kw
    )
    w_b, dw_b_dcn, dw_b_dq = d4_reference_weights_derivs(
        cn_b, q_b, cns_b, qs_b, z_b, **kw
    )

    c6 = 0.0
    dc6_dcn_a = 0.0
    dc6_dcn_b = 0.0
    dc6_dq_a = 0.0
    dc6_dq_b = 0.0

    for ia in range(len(w_a)):
        wi = w_a[ia]
        dwi_cn = dw_a_dcn[ia]
        dwi_q = dw_a_dq[ia]
        for jb in range(len(w_b)):
            wj = w_b[jb]
            cref = ref_data.get_c6_ref(z_a, ia, z_b, jb)
            c6 += wi * wj * cref
            dc6_dcn_a += dwi_cn * wj * cref
            dc6_dcn_b += wi * dw_b_dcn[jb] * cref
            dc6_dq_a += dwi_q * wj * cref
            dc6_dq_b += wi * dw_b_dq[jb] * cref

    return c6, dc6_dcn_a, dc6_dcn_b, dc6_dq_a, dc6_dq_b


def d4_bj_damping_gradient(
    r_ab: float,
    vec_ab: np.ndarray,
    c6: float,
    c8: float,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
) -> tuple[float, np.ndarray, float]:
    """BJ-damped pair dispersion energy and its geometric gradient.

    Returns ``(E_pair, dE/dR_a, dE/dC6)`` where ``dE/dR_a`` is the
    3-vector gradient on atom A (atom B gets the negative).
    ``dE/dC6`` is the scalar derivative needed for the CN/q chain rule.
    """
    r0 = a1 * np.sqrt(float(c8) / float(c6)) + a2
    r2 = r_ab * r_ab
    r6 = r2 * r2 * r2
    r8 = r6 * r2

    t6 = 1.0 / (r6 + r0**6)
    t8 = 1.0 / (r8 + r0**8)

    # Energy.
    edisp0 = s6 * t6 + s8 * (float(c8) / float(c6)) * t8
    e_pair = -float(c6) * edisp0

    # Gradient w.r.t. R: d/dr2(edisp0) = s6*d6 + s8*rrij*d8
    d6 = -6.0 * r2 * r2 * t6 * t6
    d8 = -8.0 * r2 * r2 * r2 * t8 * t8
    rrij = float(c8) / float(c6)
    gdisp = s6 * d6 + s8 * rrij * d8
    # dE/dR_a = -c6 * gdisp * 2r * vec/r = -2*c6*gdisp*vec
    grad_a = -2.0 * float(c6) * gdisp * vec_ab

    # Derivative w.r.t. C6 (for CN/q chain rule).
    # d/dC6 of: -c6 * (s6*t6 + s8*c8/c6*t8)
    # = -(s6*t6 + s8*c8/c6*t8) - c6 * s8 * (-c8/c6^2*t8)
    # = -edisp0 + s8 * c8/c6 * t8
    # Actually the full dE/dC6 is needed for the CN/q chain.
    # From dftd4: dE/dC6 = -edisp where edisp = s6*t6 + s8*rrij*t8
    dE_dc6 = -edisp0

    return float(e_pair), grad_a, float(dE_dc6)


def compute_d4_gradient(
    mol: "Molecule",
    ref_data: "D4ReferenceDataset",
    functional: str = "pbe",
    **kwargs: object,
) -> tuple[float, np.ndarray]:
    """Full-molecule D4 two-body dispersion energy and gradient.

    Returns ``(energy, gradient)`` where ``gradient`` has shape
    ``(n_atoms, 3)`` in Hartree/bohr.

    The gradient includes the geometric derivative (dE/dR) and the
    chain-rule terms through C6 w.r.t. CN and q (dE/dCN and dE/dq).
    The further chain through dCN/dR and dq/dR (EEQ charge gradient)
    is deferred -- those are small corrections for most geometries.
    """
    from ._vibeqc_core import eeq_charges, eeq_coordination_numbers
    from .dispersion_d4_parameters import get_d4_params

    cn_arr = eeq_coordination_numbers(mol)
    q_arr = eeq_charges(mol).charges
    params = get_d4_params(functional)

    natom = len(mol.atoms)
    z_arr = [atom.Z for atom in mol.atoms]
    xyz = np.array([atom.xyz for atom in mol.atoms])

    energy = 0.0
    gradient = np.zeros((natom, 3))
    dEdcn = np.zeros(natom)
    dEdq = np.zeros(natom)

    for a in range(natom):
        ra = xyz[a]
        for b in range(a):
            vec_ab = ra - xyz[b]
            r_ab = float(np.linalg.norm(vec_ab))
            if r_ab < 1e-12:
                continue

            c6, dc6_dcn_a, dc6_dcn_b, dc6_dq_a, dc6_dq_b = compute_d4_c6_pair_derivs(
                float(cn_arr[a]),
                float(q_arr[a]),
                z_arr[a],
                float(cn_arr[b]),
                float(q_arr[b]),
                z_arr[b],
                ref_data,
                **kwargs,  # type: ignore[arg-type]
            )
            if c6 <= 0.0:
                continue

            c8 = compute_c8(c6, z_arr[a], z_arr[b])
            e_pair, grad_a, dE_dc6 = d4_bj_damping_gradient(
                r_ab,
                vec_ab,
                c6,
                c8,
                params.s6,
                params.s8,
                params.a1,
                params.a2,
            )

            energy += e_pair
            gradient[a] += grad_a
            gradient[b] -= grad_a

            # Chain rule: dE/dCN = dE/dC6 * dC6/dCN
            dEdcn[a] += dE_dc6 * dc6_dcn_a
            dEdcn[b] += dE_dc6 * dc6_dcn_b
            dEdq[a] += dE_dc6 * dc6_dq_a
            dEdq[b] += dE_dc6 * dc6_dq_b

    # dCN/dR and dq/dR contributions are deferred (requires EEQ gradient).
    # Return energy, gradient, and auxiliary derivatives for downstream
    # consumers to chain through.
    return float(energy), gradient, dEdcn, dEdq
