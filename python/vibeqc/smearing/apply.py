"""``apply_smearing`` -- the single entry point every backend driver calls.

For ``temperature > 0`` this dispatches to the selected Fermi-Dirac,
Mermin, Methfessel-Paxton, or Marzari-Vanderbilt occupation/entropy pair; at
``temperature == 0`` it takes the shared hard-Aufbau + midgap branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Union

import numpy as np

from .aufbau import aufbau_occupations_per_k
from .fermi_dirac import fermi_dirac_occupations_per_k
from .marzari_vanderbilt import marzari_vanderbilt_occupations_per_k
from .mermin import mermin_occupations_per_k
from .methfessel_paxton import methfessel_paxton_occupations_per_k
from .options import SmearingOptions


#: Smallest frontier gap (Ha) the T = 0 hard-Aufbau fill is allowed to resolve
#: by CUTTING it (issue #543, the Python twin of issue #434).
#:
#: **This value is duplicated in C++** as the default of
#: ``KPointOccupationOptions::min_resolvable_frontier_gap``
#: (``cpp/include/vibeqc/semiempirical/kpoints_occupations.hpp``). The two fills
#: are independent implementations of one convention, and letting them drift
#: would mean the semiempirical and ab-initio multi-k routes disagreed about
#: which frontiers are resolvable. ``tests/test_smearing_frontier_resolution.py``
#: pins them EQUAL mechanically, so neither can be changed without the other.
#:
#: 1e-6 Ha is about 0.027 meV, some four orders below room-temperature kT
#: (9.5e-4 Ha); nothing physical distinguishes such a gap from zero. It was
#: checked against the real population rather than argued -- see the #434
#: measurements on the six issue-424 wave-flip fixtures.
MIN_RESOLVABLE_FRONTIER_GAP = 1.0e-6

#: Roundoff-degeneracy grouping of the T = 0 global-Aufbau fill (#339, #434,
#: #544): two sorted states are "the same level" when they lie within
#: ``ZERO_TEMPERATURE_DEGENERACY_ULPS * eps * energy_scale`` of each other,
#: the chain is transitive (each state is compared to the group's LAST
#: member), and a group's span may not exceed
#: ``DEGENERATE_GROUP_SPAN_FACTOR`` times that tolerance. Both numbers
#: duplicate the C++ kernel's ``kZeroTemperatureDegeneracyUlps`` /
#: ``kDegenerateGroupSpanFactor`` (cpp/include/vibeqc/semiempirical/
#: kpoints_occupations.hpp, exported as
#: ``_vibeqc_core.semiempirical.zero_temperature_degeneracy_ulps`` /
#: ``degenerate_group_span_factor``); ``tests/test_smearing_frontier_resolution.py``
#: pins them equal.
ZERO_TEMPERATURE_DEGENERACY_ULPS = 256.0
DEGENERATE_GROUP_SPAN_FACTOR = 16.0


def unresolved_frontier_cut(
    eps_per_k: Sequence[np.ndarray],
    occupations_per_k: Sequence[np.ndarray],
    *,
    min_resolvable_frontier_gap: float = MIN_RESOLVABLE_FRONTIER_GAP,
    occupation_tolerance: float = 1.0e-8,
    max_occupation: float | None = None,
) -> float | None:
    """Measured frontier gap when a T = 0 fill CUT a frontier it cannot resolve.

    ``max_occupation`` is the full occupancy of one state (2.0 closed-shell,
    1.0 per spin channel). Pass it whenever it is known: a shared (fractional)
    Fermi group that has NO fully occupied state below it -- the lowest states
    of a nearly empty band, or a roundoff-degenerate manifold at the bottom of
    the spectrum (#544) -- is otherwise indistinguishable from an integer fill
    at that occupancy, and would be reported as a cut. ``None`` falls back to
    the largest occupation present.

    Returns the gap in Hartree when the occupied/empty boundary falls between
    two states separated by at most ``min_resolvable_frontier_gap``, and
    ``None`` otherwise. ``None`` therefore means "nothing to worry about here",
    covering three distinct cases: a real gap, an equalized fractional Fermi
    group, and a completely empty or completely filled model space.

    Why this matters (issue #543, and #434 for the C++ twin). At that
    separation which of the two states carries the electrons is decided by
    last-ulp arithmetic, so an entire electron moves between k-points with a
    rounding difference that varies between hosts and builds -- while the SCF
    still reports convergence. Issue #424 documented exactly that as cross-host
    ``converged=True at n_iter=1`` versus ``converged=False at n_iter=500``
    branch flips on one shared binary.

    A **fractional** occupation anywhere means the fill equalized a degenerate
    Fermi group instead of cutting it. That is the correct, deterministic
    T -> 0 ensemble limit (Weinert and Davenport, Phys. Rev. B 45, 13709
    (1992)), it does not race, and it is what issue #424's regression depends
    on -- so it returns ``None`` and must never be reported here.

    This is deliberately a **stateless predicate over the realised
    occupations**, not a hook inside the fill. It re-derives nothing: with
    integer occupations the cut boundary is exactly
    ``min(empty) - max(occupied)``. Keeping it outside the fill lets each
    caller decide WHEN to ask -- which matters, because these fills run inside
    SCF loops, and a run whose intermediate iterate transits a near-degenerate
    frontier but whose converged frontier is well gapped must not be refused.
    Ask on the accepted spectrum.
    """
    if not np.isfinite(min_resolvable_frontier_gap) or (
        min_resolvable_frontier_gap < 0.0
    ):
        raise ValueError(
            "unresolved_frontier_cut: min_resolvable_frontier_gap must be "
            "finite and >= 0 (0 disables the check)"
        )
    if len(eps_per_k) != len(occupations_per_k):
        raise ValueError(
            "unresolved_frontier_cut: spectra and occupations must match "
            "k-point for k-point"
        )
    if not eps_per_k:
        return None

    energies = np.concatenate(
        [np.asarray(np.real(e), dtype=float).ravel() for e in eps_per_k]
    )
    occupations = np.concatenate(
        [np.asarray(o, dtype=float).ravel() for o in occupations_per_k]
    )
    if energies.size != occupations.size:
        raise ValueError(
            "unresolved_frontier_cut: spectra and occupations must have the "
            "same number of states"
        )
    if energies.size == 0:
        return None

    occupied = occupations > occupation_tolerance
    # A partially filled state means the fill SHARED the frontier rather than
    # cutting it. Nothing was cut, so there is nothing to resolve.
    if max_occupation is not None:
        if not np.isfinite(max_occupation) or max_occupation <= 0.0:
            raise ValueError(
                "unresolved_frontier_cut: max_occupation must be finite and > 0"
            )
        upper = float(max_occupation)
    else:
        upper = occupations.max()
    if np.any(occupied & (occupations < upper - occupation_tolerance)):
        return None
    if not occupied.any() or occupied.all():
        return None  # no boundary: empty or completely filled

    gap = float(energies[~occupied].min() - energies[occupied].max())
    return gap if gap <= min_resolvable_frontier_gap else None


@dataclass(frozen=True)
class SmearingResult:
    """Output of :func:`apply_smearing`.

    ``occupations_per_k`` -- per-band occupations, nominally bounded
    by ``[0, n_max]`` (``n_max = 2`` for closed-shell). Generalized
    MP/MV kernels can overshoot this interval.
    ``mu`` -- chemical potential in Hartree. For ``T = 0`` insulators
    this is the midgap value ``(HOMO + LUMO) / 2``.
    ``entropy`` -- electronic entropy ``S/k_B`` per unit cell. Zero
    at ``T = 0`` by construction.
    ``free_energy_correction`` -- ``-T . entropy``, ready to be
    added to the total electronic energy to form ``A = E - TS``.
    This is Mermin's free energy for Fermi-Dirac and the corresponding
    stationary generalized free energy for MP/MV.
    ``smearing`` -- echoed back so log surfaces and result structs
    don't have to re-derive it.
    """

    occupations_per_k: List[np.ndarray]
    mu: float
    entropy: float
    free_energy_correction: float
    smearing: SmearingOptions = field(default_factory=SmearingOptions)
    #: Issue #543. True when a T = 0 fill placed the occupied/empty boundary
    #: between two states it cannot resolve (see
    #: :func:`unresolved_frontier_cut`), so which one carries the electrons is
    #: decided by last-ulp arithmetic. Always False at finite temperature,
    #: where the frontier is shared rather than cut.
    frontier_cut_unresolved: bool = False
    #: The measured gap across that boundary in Hartree, ``nan`` when the
    #: fill made no unresolvable cut.
    frontier_gap: float = float("nan")


def _aufbau_with_midgap_mu(
    eps_per_k: Sequence[np.ndarray],
    n_occ_each: int,
    occ_value: float = 2.0,
) -> Tuple[List[np.ndarray], float]:
    """Hard Aufbau + midgap mu for the ``T = 0`` **single-spectrum** branch.

    ``occ_value`` is ``2.0`` for closed-shell and ``1.0`` for a single
    open-shell spin channel; the midgap mu (``(HOMO + LUMO) / 2``) is
    independent of it.

    .. warning::

       This fills the lowest ``n_occ_each`` states of **every** spectrum
       independently, so on a k-mesh it is a *per-k* Aufbau: it computes one
       global mu from the mesh and then does not use it. That is only correct
       when a single chemical potential cannot reorder anything -- one
       k-point, or a mesh with a gap at every k and no band crossing mu.
       Multi-k closed-shell callers must go through
       :func:`_global_aufbau_with_mu` (see :func:`apply_smearing`); the
       mismatch between the mu reported here and the occupations returned
       beside it is exactly the defect in GitLab #85.
    """
    occ = aufbau_occupations_per_k(eps_per_k, n_occ_each, occ_value=occ_value)
    if not eps_per_k:
        return occ, 0.0
    n_occ = int(n_occ_each)
    homos = [
        float(np.real(eps[n_occ - 1])) if n_occ > 0 else float("-inf")
        for eps in eps_per_k
    ]
    lumos = [
        float(np.real(eps[n_occ]))
        if n_occ < int(np.asarray(eps).shape[0])
        else float("inf")
        for eps in eps_per_k
    ]
    homo = max(homos) if homos else 0.0
    lumo = min(lumos) if lumos else 0.0
    mu = 0.5 * (homo + lumo) if np.isfinite(homo + lumo) else 0.0
    return occ, mu


def _fixed_subspace_fill_with_midgap_mu(
    eps_per_k: Sequence[np.ndarray],
    n_occ_each: int,
    occ_value: float = 2.0,
) -> Tuple[List[np.ndarray], float]:
    """Positional ``T = 0`` fill for a caller-chosen occupied subspace (#725).

    The maximum overlap method (Gilbert, Besley and Gill, J. Phys. Chem. A
    112, 13164 (2008), Sec. 2, Eqs. 2.7-2.8) replaces the Aufbau protocol:
    at every SCF cycle the ``n`` new orbitals with the largest projection
    onto the previous occupied space are occupied, whatever their energies.
    The multi-k drivers express that choice by permuting the columns of
    ``C(k)`` and the entries of ``eps(k)`` so the MOM-selected states lead,
    so the occupied set is *positional* here: ``occ[:n_occ_each] = occ_value``
    at every k, exactly the pattern the fixed ``C[:, :n_occ]`` density slice
    consumes. An energy-ordered fill (:func:`_global_aufbau_with_mu`) would
    re-pick the lowest states and silently undo the MOM selection, which is
    the defect of GitLab #725.

    ``mu`` is the midpoint between the highest occupied and the lowest
    empty energy over the mesh under this partition. For a non-Aufbau
    selection the two can be inverted; the value is reported, not used to
    decide anything.
    """
    occ = aufbau_occupations_per_k(eps_per_k, n_occ_each, occ_value=occ_value)
    n_occ = int(n_occ_each)
    homo = float("-inf")
    lumo = float("inf")
    for eps in eps_per_k:
        e = np.asarray(np.real(eps), dtype=float)
        if n_occ > 0:
            homo = max(homo, float(np.max(e[:n_occ])))
        if n_occ < e.shape[0]:
            lumo = min(lumo, float(np.min(e[n_occ:])))
    mu = 0.5 * (homo + lumo) if np.isfinite(homo + lumo) else 0.0
    return occ, mu


# A finite-k band calculation is one periodic finite system, so its
# occupations obey one particle constraint over the full eigenvalue sum,
# not one independent constraint at every k point. Weinert and Davenport,
# Phys. Rev. B 45, 13709 (1992), Eqs. (3) and (8), DOI
# 10.1103/PhysRevB.45.13709. The T -> 0 limit used here is
#
#     sum_k w_k sum_i f_ki = N_e,  f_ki = g below mu and 0 above mu,
#
# with one shared fractional f for an energy-degenerate group at mu.
def _global_aufbau_with_mu(
    eps_per_k: Sequence[np.ndarray],
    weights: Sequence[float],
    n_electrons_per_cell: float,
    occ_value: float = 2.0,
) -> Tuple[List[np.ndarray], float]:
    """Weighted zero-temperature Aufbau fill with one global ``mu``.

    The lowest states across the complete k-mesh are occupied subject to the
    weighted particle count. If the target cuts an energy-degenerate group,
    every member receives the same fractional occupation; this is the
    symmetry-preserving zero-temperature ensemble limit of the common Fermi
    function.
    """
    g = float(occ_value)
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("global Aufbau: occ_value must be > 0")

    eps_real = [np.asarray(np.real(e), dtype=float) for e in eps_per_k]
    if not eps_real:
        raise ValueError("global Aufbau: no k-points")
    if any(e.ndim != 1 or e.size == 0 for e in eps_real):
        raise ValueError(
            "global Aufbau: eigenvalue arrays must be non-empty 1D arrays"
        )
    if any(not np.all(np.isfinite(e)) for e in eps_real):
        raise ValueError("global Aufbau: eigenvalues must be finite")

    w_arr = np.asarray(weights, dtype=float)
    if w_arr.shape != (len(eps_real),):
        raise ValueError(
            "global Aufbau: weights length must match number of k-points"
        )
    if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
        raise ValueError("global Aufbau: weights must be finite and >= 0")
    weight_sum = math.fsum(float(w) for w in w_arr)
    weight_tol = (
        256.0 * np.finfo(float).eps * max(1.0, float(len(w_arr)))
    )
    if abs(weight_sum - 1.0) > weight_tol:
        raise ValueError("global Aufbau: weights must sum to 1")

    target = float(n_electrons_per_cell)
    if not np.isfinite(target):
        raise ValueError("global Aufbau: electron count must be finite")
    capacity = math.fsum(
        float(w) * g * float(eps.size)
        for eps, w in zip(eps_real, w_arr)
    )
    count_tol = 256.0 * np.finfo(float).eps * max(
        1.0, abs(target), abs(capacity)
    )
    if target < -count_tol or target > capacity + count_tol:
        raise ValueError(
            "global Aufbau: electron count is outside the available "
            "band capacity"
        )

    occupations = [np.zeros_like(eps) for eps in eps_real]
    positive_weight_states = [
        (float(energy), ik, ib, float(w_arr[ik]))
        for ik, eps in enumerate(eps_real)
        if w_arr[ik] > 0.0
        for ib, energy in enumerate(eps)
    ]
    if not positive_weight_states:
        raise ValueError("global Aufbau: at least one k-point must have weight")
    positive_weight_states.sort(key=lambda state: state[0])

    eps_min = positive_weight_states[0][0]
    eps_max = positive_weight_states[-1][0]
    if target <= count_tol:
        return occupations, float(eps_min)
    if target >= capacity - count_tol:
        return [np.full_like(eps, g) for eps in eps_real], float(eps_max)

    energy_scale = max(
        1.0, max(abs(state[0]) for state in positive_weight_states)
    )
    # This is a roundoff equality test, not a physical broadening. It groups
    # the archived P15 symmetry pair split by 8.33e-16 Ha while leaving its
    # 2.16e-4-Ha minimum direct gap distinct.
    energy_tol = ZERO_TEMPERATURE_DEGENERACY_ULPS * np.finfo(float).eps * energy_scale
    # Bound on a degeneracy group's span (#544); see the constants above.
    max_group_span = DEGENERATE_GROUP_SPAN_FACTOR * energy_tol

    # Transitive grouping (#544): a state joins the current group when it is
    # within energy_tol of the group's LAST member (the states are sorted, so
    # the difference is non-negative) and the group's span stays within
    # max_group_span. The pre-#544 rule compared to the group's FIRST member
    # -- a ball -- and could split a roundoff-degenerate manifold into two
    # adjacent groups one ulp apart, which the fill then integer-occupied
    # across the chop. Same rule as the C++ kernel
    # (global_zero_temperature_occupations).
    groups: List[List[Tuple[float, int, int, float]]] = []
    for state in positive_weight_states:
        if (
            not groups
            or state[0] - groups[-1][-1][0] > energy_tol
            or state[0] - groups[-1][0][0] > max_group_span
        ):
            groups.append([state])
        else:
            groups[-1].append(state)

    filled_group_capacities: List[float] = []
    filled_capacity_running = 0.0
    mu = float(eps_min)
    boundary_occ: float | None = None
    for ig, group in enumerate(groups):
        group_weight = math.fsum(state[3] for state in group)
        group_capacity = g * group_weight
        next_capacity = filled_capacity_running + group_capacity
        if target > next_capacity + count_tol:
            for _, ik, ib, _ in group:
                occupations[ik][ib] = g
            filled_group_capacities.append(group_capacity)
            filled_capacity_running = next_capacity
            continue

        if target >= next_capacity - count_tol:
            for _, ik, ib, _ in group:
                occupations[ik][ib] = g
            filled_group_capacities.append(group_capacity)
            if ig + 1 < len(groups):
                mu = 0.5 * (group[-1][0] + groups[ig + 1][0][0])
            else:
                mu = group[-1][0]
        else:
            filled_capacity = math.fsum(filled_group_capacities)
            boundary_occ = max(
                0.0, min(g, (target - filled_capacity) / group_weight)
            )
            for _, ik, ib, _ in group:
                occupations[ik][ib] = boundary_occ
            filled_group_capacities.append(boundary_occ * group_weight)
            mu = float(np.mean([state[0] for state in group]))
        break

    filled_capacity = math.fsum(filled_group_capacities)
    if abs(target - filled_capacity) > count_tol:
        raise RuntimeError(
            "global Aufbau: failed to satisfy the weighted electron count"
        )

    # Zero-weight k points do not constrain mu, but give them occupations
    # consistent with the same global step function for deterministic output.
    for ik, (eps, weight) in enumerate(zip(eps_real, w_arr)):
        if weight != 0.0:
            continue
        occupations[ik][eps < mu - energy_tol] = g
        at_mu = np.abs(eps - mu) <= energy_tol
        occupations[ik][at_mu] = (
            boundary_occ if boundary_occ is not None else 0.5 * g
        )

    return occupations, float(mu)


def apply_smearing(
    eps_per_k: Sequence[np.ndarray],
    *,
    weights: Sequence[float],
    n_electrons_per_cell: float,
    n_occ_each: int,
    smearing: Union[SmearingOptions, float, None] = None,
    fixed_occupied_subspace: bool = False,
) -> SmearingResult:
    """Compute fractional occupations + mu + entropy for one k-mesh.

    The single entry point every backend driver calls. Accepts the
    full :class:`SmearingOptions` dataclass, a bare Hartree
    ``k_B T`` float (treated as Fermi-Dirac for back-compat), or
    ``None`` (treated as ``temperature == 0``).

    ``fixed_occupied_subspace=True`` declares that the caller has already
    chosen the occupied states and ordered them first in every ``eps(k)``
    (the maximum overlap method: the drivers permute ``C(k)`` and ``eps(k)``
    so the MOM-selected columns lead). The ``T = 0`` fill is then positional,
    ``occ[:n_occ_each] = 2`` at every k, instead of the global energy
    ordering, which would re-pick the lowest states and undo the selection
    (GitLab #725). It is a zero-temperature contract: combining it with
    finite smearing raises, because a smeared Fermi function has no sharp
    subspace to hold.

    Closed-shell only at v0.10.x. Open-shell (M3) will land an
    ``apply_smearing_open_shell`` sibling that runs the same
    machinery with two per-spin electron-count constraints.
    """
    if smearing is None:
        smearing = SmearingOptions()
    elif not isinstance(smearing, SmearingOptions):
        smearing = SmearingOptions.from_legacy_kwarg(float(smearing))

    if fixed_occupied_subspace:
        if smearing.enabled:
            raise ValueError(
                "apply_smearing: fixed_occupied_subspace (maximum overlap "
                "method) is a zero-temperature contract; a smeared Fermi "
                "function has no sharp occupied subspace to hold. Disable "
                "use_mom or set smearing_temperature = 0."
            )
        # The occupied set is the caller's, not the fill's: no energetic
        # frontier was cut here, so there is nothing for #543 to report.
        occ, mu = _fixed_subspace_fill_with_midgap_mu(eps_per_k, n_occ_each)
        return SmearingResult(
            occupations_per_k=occ,
            mu=mu,
            entropy=0.0,
            free_energy_correction=0.0,
            smearing=smearing,
        )

    if not smearing.enabled:
        # T = 0 on a k-mesh is still ONE particle constraint over the full
        # eigenvalue sum, so the occupied set is chosen by the global energy
        # ordering, not by taking the lowest n_occ_each bands at every k
        # independently. Filling per k occupies a state at k1 that lies above
        # an empty state at k2, and the band summary then reports
        # CBM - VBM < 0: the nonphysical negative indirect gap of GitLab #85
        # (archived Si/def2-SVP/PBE 6x6x6: -0.080825 Ha with exactly 14
        # occupied bands forced at every k).
        #
        # Single spectrum keeps the historical hard-Aufbau path bit-for-bit:
        # with one k-point there is nothing for a shared mu to reorder. This
        # mirrors the branch the C++ k-point occupation routine already takes
        # (compute_closed_shell_kpoint_occupations in
        # cpp/src/semiempirical/kpoints_occupations.cpp -- global fill for
        # eps_per_k.size() > 1, historical path for Gamma/molecular/SECCM),
        # and the contract 60104fc01 established for the native multi-k GDF
        # route. A gapped mesh is unchanged either way: when the electron
        # count lands exactly on a group boundary the global fill returns the
        # same {0, occ_value} arrays and the same midgap mu.
        if len(eps_per_k) > 1:
            occ, mu = _global_aufbau_with_mu(
                eps_per_k,
                weights,
                float(n_electrons_per_cell),
                occ_value=2.0,
            )
        else:
            occ, mu = _aufbau_with_midgap_mu(eps_per_k, n_occ_each)
        # Issue #543: report, do not refuse. This runs inside SCF loops, so a
        # raise here would abort a run whose intermediate iterate transits a
        # near-degenerate frontier but whose converged frontier is well gapped
        # -- the same reason the C++ twin records instead of throwing (#434).
        # The caller asks on the spectrum it accepts.
        cut = unresolved_frontier_cut(eps_per_k, occ, max_occupation=2.0)
        return SmearingResult(
            occupations_per_k=occ,
            mu=mu,
            entropy=0.0,
            free_energy_correction=0.0,
            smearing=smearing,
            frontier_cut_unresolved=cut is not None,
            frontier_gap=float("nan") if cut is None else cut,
        )

    if smearing.flavor == "fermi-dirac":
        occ, mu, entropy = fermi_dirac_occupations_per_k(
            eps_per_k,
            weights,
            float(n_electrons_per_cell),
            float(smearing.temperature),
        )
    elif smearing.flavor == "mermin":
        occ, mu, entropy = mermin_occupations_per_k(
            eps_per_k,
            weights,
            float(n_electrons_per_cell),
            float(smearing.temperature),
        )
    elif smearing.flavor == "methfessel-paxton":
        occ, mu, entropy = methfessel_paxton_occupations_per_k(
            eps_per_k,
            weights,
            float(n_electrons_per_cell),
            float(smearing.temperature),
            mp_order=int(smearing.mp_order),
        )
    elif smearing.flavor == "marzari-vanderbilt":
        occ, mu, entropy = marzari_vanderbilt_occupations_per_k(
            eps_per_k,
            weights,
            float(n_electrons_per_cell),
            float(smearing.temperature),
        )
    else:
        raise NotImplementedError(
            f"apply_smearing: flavor={smearing.flavor!r} is not implemented."
        )
    return SmearingResult(
        occupations_per_k=occ,
        mu=mu,
        entropy=float(entropy),
        free_energy_correction=-float(smearing.temperature) * float(entropy),
        smearing=smearing,
    )


def apply_smearing_open_shell(
    eps_alpha_per_k: Sequence[np.ndarray],
    eps_beta_per_k: Sequence[np.ndarray],
    *,
    weights: Sequence[float],
    n_alpha: float,
    n_beta: float,
    smearing: Union[SmearingOptions, float, None] = None,
) -> Tuple[SmearingResult, SmearingResult]:
    """Per-spin occupations with SEPARATE chemical potentials mu_a, mu_b.

    The open-shell (UHF/UKS) sibling of :func:`apply_smearing`. Each spin
    channel runs the identical mu-bisection machinery with
    ``spin_degeneracy = 1`` (occupations in ``[0, 1]``) and its own
    particle-count constraint ``sum_k w_k sum_i n^s_i(k) = n_s`` -- the
    standard spin-polarized convention (CP2K / VASP / Quantum Espresso).
    mu_a and mu_b relax independently, so ferro/antiferro stationary points
    with substantially different per-spin Fermi levels occupy correctly.

    Returns ``(alpha_result, beta_result)`` -- two :class:`SmearingResult`,
    each carrying its channel's occupations (in ``[0, 1]``), mu_s, entropy,
    and ``free_energy_correction = -T . S_s``. The driver sums the two
    ``entropy`` / ``free_energy_correction`` values to form Mermin's free
    energy ``A = E - T(S_a + S_b)``.

    At ``T = 0`` (smearing disabled) each channel is per-spin hard Aufbau
    + midgap mu_s -- bit-identical to the legacy per-k Aufbau the open-shell
    drivers used, so disabling smearing stays behavior-neutral.
    """
    if smearing is None:
        smearing = SmearingOptions()
    elif not isinstance(smearing, SmearingOptions):
        smearing = SmearingOptions.from_legacy_kwarg(float(smearing))

    def _one_channel(eps_per_k: Sequence[np.ndarray], n_sigma: float) -> SmearingResult:
        if not smearing.enabled:
            occ, mu = _aufbau_with_midgap_mu(
                eps_per_k, int(round(float(n_sigma))), occ_value=1.0
            )
            return SmearingResult(
                occupations_per_k=occ,
                mu=mu,
                entropy=0.0,
                free_energy_correction=0.0,
                smearing=smearing,
            )
        if smearing.flavor == "fermi-dirac":
            occ, mu, entropy = fermi_dirac_occupations_per_k(
                eps_per_k,
                weights,
                float(n_sigma),
                float(smearing.temperature),
                spin_degeneracy=1.0,
            )
        elif smearing.flavor == "mermin":
            occ, mu, entropy = mermin_occupations_per_k(
                eps_per_k,
                weights,
                float(n_sigma),
                float(smearing.temperature),
                spin_degeneracy=1.0,
            )
        elif smearing.flavor == "methfessel-paxton":
            occ, mu, entropy = methfessel_paxton_occupations_per_k(
                eps_per_k,
                weights,
                float(n_sigma),
                float(smearing.temperature),
                spin_degeneracy=1.0,
                mp_order=int(smearing.mp_order),
            )
        elif smearing.flavor == "marzari-vanderbilt":
            occ, mu, entropy = marzari_vanderbilt_occupations_per_k(
                eps_per_k,
                weights,
                float(n_sigma),
                float(smearing.temperature),
                spin_degeneracy=1.0,
            )
        else:
            raise NotImplementedError(
                f"apply_smearing_open_shell: flavor={smearing.flavor!r} "
                "is not implemented."
            )
        return SmearingResult(
            occupations_per_k=occ,
            mu=mu,
            entropy=float(entropy),
            free_energy_correction=-float(smearing.temperature) * float(entropy),
            smearing=smearing,
        )

    return (
        _one_channel(eps_alpha_per_k, n_alpha),
        _one_channel(eps_beta_per_k, n_beta),
    )


def occupations_are_per_k_integer_aufbau(
    occupations_per_k: Sequence[np.ndarray],
    n_occ_each: int,
    occ_value: float = 2.0,
) -> bool:
    """Whether every k point carries exactly ``occ[:n_occ] = occ_value``.

    A density builder that takes a single ``n_occ`` and slices ``C[:, :n_occ]``
    at every k assumes one fixed occupied subspace of the same size everywhere,
    each of its states carrying the full degeneracy ``occ_value``. That holds
    only for this pattern. Under the T = 0 global fill a band-overlap mesh
    occupies a different number of bands at different k, and a degenerate group
    at ``mu`` takes a shared fractional occupation, so the fixed-subspace
    builder would silently drop the difference.

    ``occ_value`` is passed in rather than inferred: a mesh whose only occupied
    states are a half-filled degenerate group would otherwise present ``1.0`` as
    if it were the full closed-shell degeneracy, and be accepted by exactly the
    builder that cannot represent it.
    """
    blocks = [np.asarray(occ, dtype=float) for occ in occupations_per_k]
    if not blocks:
        return True
    n_occ = int(n_occ_each)
    g = float(occ_value)
    for block in blocks:
        if n_occ > block.size:
            return False
        expected = np.zeros_like(block)
        expected[:n_occ] = g
        if not np.array_equal(block, expected):
            return False
    return True


def require_fixed_occupied_subspace(
    occupations_per_k: Sequence[np.ndarray],
    n_occ_each: int,
    *,
    entry: str,
    occ_value: float = 2.0,
) -> None:
    """Refuse an integer-subspace density build for band-overlap occupations.

    The T = 0 fill is one global Fermi level across the mesh (#85), so it can
    return occupations that no ``C[:, :n_occ]`` slice reproduces. Drivers whose
    density build still takes that slice must not pair it with these
    occupations: the reported band edges would then describe a different state
    from the density that was actually converged. Fail closed and name the
    supported routes instead of shipping the mismatch (CLAUDE.md section 7).

    Removing this guard is the density half of GitLab #509 -- the multi-k
    routes need the same ``density-from-occupations`` build the GDF route
    already uses (``periodic_k_gdf._density_from_orbitals``).
    """
    if occupations_are_per_k_integer_aufbau(
        occupations_per_k, n_occ_each, occ_value=occ_value
    ):
        return
    blocks = [np.asarray(occ, dtype=float) for occ in occupations_per_k]
    counts = [int(np.count_nonzero(b > 0.0)) for b in blocks]
    fractional = any(
        bool(np.any((b > 0.0) & (b < float(occ_value)))) for b in blocks
    )
    if fractional:
        shape = (
            "shares a fractional occupation across a degenerate group at the "
            "Fermi level"
        )
    elif all(c == int(n_occ_each) for c in counts):
        # Equal integer counts, yet not the leading block: the occupied
        # states sit outside eps[:n_occ_each] at some k, which happens when
        # the eigenvalue list is not in energy order there (a MOM-permuted
        # spectrum handed to the energy-ordered fill, GitLab #725). Do not
        # describe that as a count mismatch.
        shape = (
            "occupies states outside the leading n_occ_each entries of the "
            "eigenvalue list at some k point (equal per-k counts, but the "
            "list is not in energy order there)"
        )
    else:
        shape = "occupies a different number of bands at different k points"
    raise NotImplementedError(
        f"{entry}: the zero-temperature Brillouin-zone fill {shape} "
        f"(per-k occupied counts {counts}, n_occ_each={int(n_occ_each)}), so "
        "the bands overlap the Fermi level. This route builds its density "
        "from a single fixed occupied subspace and cannot represent that "
        "state; continuing would converge one density and report the band "
        "edges of another. Use smearing_temperature > 0 (Fermi-Dirac / "
        "Mermin / Methfessel-Paxton / Marzari-Vanderbilt), or "
        "bz_integration='gilat', or the multi-k GDF route "
        "(run_krhf_periodic_gdf / run_krks_periodic_gdf), which builds its "
        "density from the occupations directly. Tracked as GitLab #509."
    )


def closed_shell_periodic_occupations(
    eps_per_k: Sequence[np.ndarray],
    weights: Sequence[float],
    n_electrons_per_cell: float,
    n_occ_each: int,
    smearing_temperature: float,
    *,
    fixed_occupied_subspace: bool = False,
) -> Tuple[List[np.ndarray], float, float]:
    """Driver-friendly tuple-returning wrapper around :func:`apply_smearing`.

    ``fixed_occupied_subspace`` is forwarded unchanged; the MOM drivers pass
    ``use_mom`` so their permuted occupied block survives the ``T = 0`` fill
    (GitLab #725).

    Matches the historical signature of the per-driver
    ``_occupations_from_eps`` / ``_occupations_per_k`` closures so
    the M1 driver migration is a single-line import-and-call swap
    rather than a return-shape refactor. Returns
    ``(occupations_per_k, mu, entropy)`` exactly as the per-driver
    closures did.
    """
    result = apply_smearing(
        eps_per_k,
        weights=weights,
        n_electrons_per_cell=float(n_electrons_per_cell),
        n_occ_each=int(n_occ_each),
        smearing=SmearingOptions.from_legacy_kwarg(float(smearing_temperature)),
        fixed_occupied_subspace=bool(fixed_occupied_subspace),
    )
    return result.occupations_per_k, result.mu, result.entropy


def smeared_occupation_selfconsistency_tolerance(
    conv_tol_energy: float,
) -> float:
    """Occupation tolerance for the smeared SCF's self-consistency guard.

    A smeared SCF that satisfies the energy + commutator convergence
    tests must ALSO hold occupations that are the Fermi filling of its
    own Fock's eigenvalues -- on symmetry-locked fixtures whose FDS-SDF
    commutator vanishes identically (H2-in-box class), a stale
    extrapolated Fock can freeze the occupations at a spurious fixed
    point the energy tests cannot see (dE = 0 exactly; see the
    zero-commutator extrapolation floor in
    ``periodic_scf_accelerators.py``). The guard compares the stored
    occupations against a fresh Fermi filling of the current Fock and
    rejects convergence above this tolerance.

    The Mermin free energy ``A = E - T S`` is stationary with respect
    to the occupations at the Fermi filling of the converged Fock's own
    eigenvalues (Mermin, Phys. Rev. 137, A1441 (1965)), so an
    occupation residual ``df`` costs ``O(df^2)`` in ``A``; the
    occupation tolerance commensurate with an energy tolerance
    ``tol_E`` is therefore ``sqrt(tol_E)``. The ``1e-9`` floor keeps
    the guard above the ~1e-12 stored-vs-recomputed occupation
    reproducibility of a converged plain-iteration SCF (measured
    2026-07-30, H2 2.4 bohr / 12-bohr box / sto-3g / kBT = 0.05 Ha)
    for arbitrarily tight energy tolerances.
    """
    return max(float(np.sqrt(float(conv_tol_energy))), 1.0e-9)
