"""Measured band edges of a periodic result: one classifier for every surface.

Every consumer that says "gap", "VBM", "CBM" or "metallic" about a periodic
calculation -- the ``.out`` band-extrema block, the ``.system`` manifest, the
public :func:`vibeqc.frontier_bands` accessor, the semiempirical k-route
result fields -- reads it from here, so the same spectrum can no longer be
called an 8 eV insulator by one surface and a metal by another (GitLab #176,
the reporter/accessor split its 2026-08-30 verification found).

The convention
--------------

It is the occupation convention the native k-route kernel documents on
``KPointBandEdges`` (``cpp/include/vibeqc/semiempirical/kpoints_occupations.hpp``,
issue #426), with the one case that convention got wrong repaired in both
implementations at once.

With occupation tolerance ``t`` and full occupancy ``g`` (2 for a restricted
closed-shell channel, 1 for one spin channel), a state ``(k, i)`` holding
``n`` electrons is

* a VALENCE state when ``n > t`` (it holds electrons), and
* a CONDUCTION state when ``n < g - t`` (it can accept electrons).

``indirect_gap = CBM - VBM`` over every sampled k-point, unclamped. A
fractionally occupied state belongs to both classes, so a Fermi level pinned
inside a partially filled degenerate group measures ``indirect_gap <= 0``.

**The repaired case: the Fermi level crossing a band.** Under the
zero-temperature global fill (one chemical potential for the whole mesh,
Weinert and Davenport, Phys. Rev. B 45, 13709 (1992)), a band that dips below
the Fermi level at one k-point and rises above it at another is occupied at
the first and empty at the second, and every occupation is still an integer.
The state classes above then never overlap: the highest occupied state lies
below the lowest empty one *by construction of the fill*, so ``CBM - VBM`` is
POSITIVE for a metal. That number is only the spacing of the sampled levels
around the Fermi level; it shrinks to zero as the mesh is refined, and
reporting it as a band gap is how the two-k reproducer of #176 printed an
8.2 eV insulator for a band-overlap metal.

This classifier therefore detects the crossing -- integer occupations whose
occupied-state count differs between k-points -- and reports the edges of the
band pair at the Fermi filling instead: with ``n_f = N / g`` bands' worth of
electrons per cell, the valence band is index ``ceil(n_f) - 1`` and the
conduction band index ``floor(n_f)`` in the energy-sorted spectrum at every
k (the CRYSTAL convention, and the one :func:`vibeqc.frontier_bands` and
``periodic_gapw_postscf.compute_homo_lumo`` already use). For an integer
``n_f`` this is ``(n - 1, n)`` and the measured gap is the (negative) band
overlap; for a non-integer ``n_f`` both indices name the partially filled
band and the gap is minus its width. Either way it is strictly negative
whenever the counts differ. Proof for integer ``n``: the weighted mean count
is ``n``, so some ``k1`` has at least ``n + 1`` states below ``mu`` and some
``k2`` at most ``n - 1``, whence ``eps[k1][n] < mu < eps[k2][n - 1]``.

With zero-weight k-points (a band-structure spelling) the occupied count can
also differ without any crossing: those points are filled with the step
function at the chemical potential the weighted points fix, which need not lie
inside the gap. The band-pair edges are still the right ones there and measure
a positive gap, so ``fermi_level_crosses_band`` is set only when the band-pair
gap is ``<= 0``.

``is_metallic`` is True exactly when a fractional occupation is present, the
Fermi level crosses a band, or the measured gap is ``<= 0``. It is the
structural flag a screen tests; the sign of a float alone is not enough.

Nothing changes on a gapped row: when every k holds the same integer count
the band-pair edges and the state-class edges are the same maximum and
minimum over the same floats, bit for bit.

``gap_above_fermi_manifold`` is the occupancy-partition quantity
``min(eps | n <= t) - max(eps | n > t)``: the distance to the next entirely
empty state. It is legitimate, but it is not the band gap (sec8-r4 F1,
issue #131), and it keeps its own name so it cannot be mistaken for one.

NaN appears only when a class is empty in the model space (no electrons, or
a completely filled basis). NaN never encodes a metal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


#: Occupation classification threshold. Twin of the default of the native
#: ``compute_kpoint_band_edges`` (``kpoints_occupations.hpp``);
#: ``tests/test_band_edges.py`` pins the two implementations equal on shared
#: spectra.
OCCUPATION_TOLERANCE = 1.0e-8

_NAN = float("nan")


@dataclass(frozen=True)
class BandChannel:
    """One occupation channel of a periodic result.

    ``max_occupation`` is the full occupancy of one state: 2 for a restricted
    closed-shell channel, 1 for one spin channel of an unrestricted result.
    ``restricted_open_shell`` marks the single 2/1/0 channel of an ROHF/ROKS
    result, whose singly occupied orbitals are an integer occupation class
    (valence, never conduction) rather than a fractional frontier.
    """

    spin: Optional[str]
    eps_per_k: Tuple[np.ndarray, ...]
    occupations_per_k: Tuple[np.ndarray, ...]
    max_occupation: float = 2.0
    restricted_open_shell: bool = False

    @property
    def n_kpoints(self) -> int:
        return len(self.eps_per_k)


@dataclass(frozen=True)
class BandEdges:
    """Measured valence/conduction edges of a periodic spectrum.

    Energies are Hartree. See the module docstring for the convention. The
    ``*_k`` fields index the sampled k-set (``-1`` when the edge does not
    exist); ``valence_max_per_k`` / ``conduction_min_per_k`` carry one entry
    per k-point, NaN where that k has no state of the class.

    ``fermi_level_crosses_band`` and ``fractional_fermi_group`` say WHY a
    result is metallic; ``is_metallic`` is their disjunction with a measured
    ``indirect_gap <= 0``. For an unrestricted result the combined edges are
    taken over both spin channels and ``channels`` holds the per-spin edges.
    """

    valence_band_max: float
    conduction_band_min: float
    indirect_gap: float
    direct_gap: float
    gap_above_fermi_manifold: float
    is_metallic: bool
    fermi_level_crosses_band: bool
    fractional_fermi_group: bool
    valence_band_max_k: int
    conduction_band_min_k: int
    direct_gap_k: int
    valence_max_per_k: Tuple[float, ...]
    conduction_min_per_k: Tuple[float, ...]
    n_kpoints: int
    n_bands: int
    spin: Optional[str] = None
    channels: Tuple["BandEdges", ...] = field(default=(), repr=False)

    @property
    def is_direct(self) -> bool:
        """Whether the VBM and CBM were found at the same k-point."""
        return (
            self.valence_band_max_k >= 0
            and self.valence_band_max_k == self.conduction_band_min_k
        )

    @property
    def classification(self) -> str:
        """``"metal"``, ``"insulator"``, or ``"not-measured"``.

        ``"not-measured"`` means a class is empty in the model space, so no
        gap exists to classify (NaN ``indirect_gap``).
        """
        if self.is_metallic:
            return "metal"
        if math.isfinite(self.indirect_gap):
            return "insulator"
        return "not-measured"


def _as_blocks(values: Sequence, name: str) -> List[np.ndarray]:
    blocks = [np.asarray(np.real(v), dtype=float).reshape(-1) for v in values]
    for block in blocks:
        if not np.all(np.isfinite(block)):
            raise ValueError(f"band edges: {name} must be finite")
    return blocks


def _fermi_filling(
    channel: BandChannel,
    occupations: Sequence[np.ndarray],
    weights: np.ndarray,
) -> float:
    """Bands' worth of electrons per cell, ``N / g``, in this channel.

    For the 2/1/0 channel of ROHF/ROKS the band pair at the Fermi level is the
    one bounding the singly occupied manifold, so the filling is the count of
    at-least-singly occupied states (the alpha count).
    """
    if channel.restricted_open_shell:
        return math.fsum(
            float(w) * float(np.count_nonzero(occ > OCCUPATION_TOLERANCE))
            for w, occ in zip(weights, occupations)
        )
    total = math.fsum(
        float(w) * float(np.sum(occ))
        for w, occ in zip(weights, occupations)
    )
    return total / float(channel.max_occupation)


def _normalised_weights(
    weights: Optional[Sequence[float]], n_k: int
) -> np.ndarray:
    if weights is None:
        return np.full(n_k, 1.0 / n_k)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (n_k,):
        raise ValueError(
            f"band edges: {w.size} k-point weights for {n_k} k-points"
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("band edges: k-point weights must be finite and >= 0")
    total = float(np.sum(w))
    if total <= 0.0:
        # A band-structure path: every point is zero weight, so no electron
        # count is sampled. The occupations came from elsewhere; weigh the
        # points equally to recover the filling.
        return np.full(n_k, 1.0 / n_k)
    return w / total


def _channel_state_classes(
    channel: BandChannel, tol: float
) -> Tuple[List[np.ndarray], List[np.ndarray], bool, List[int]]:
    """Valence / conduction masks per k, whether any state is fractional,
    and the per-k occupied-state counts."""
    g = float(channel.max_occupation)
    valence: List[np.ndarray] = []
    conduction: List[np.ndarray] = []
    fractional = False
    counts: List[int] = []
    for occ in channel.occupations_per_k:
        occ = np.asarray(occ, dtype=float)
        v = occ > tol
        if channel.restricted_open_shell:
            c = occ < tol
            distance = np.min(
                np.abs(occ[..., None] - np.array([0.0, 1.0, 2.0])), axis=-1
            )
            frac = bool(np.any(distance > tol))
        else:
            c = occ < g - tol
            frac = bool(np.any(v & c))
        valence.append(v)
        conduction.append(c)
        fractional = fractional or frac
        counts.append(int(np.count_nonzero(v)))
    return valence, conduction, fractional, counts


def _band_pair_indices(filling: float, n_bands: int) -> Tuple[int, int]:
    """Valence/conduction band indices at a Fermi filling of ``filling``."""
    nearest = round(filling)
    if abs(filling - nearest) <= 1.0e-8 * max(1.0, abs(filling)):
        lo, hi = int(nearest) - 1, int(nearest)
    else:
        lo = hi = int(math.floor(filling))
    lo = min(max(lo, -1), n_bands - 1)
    hi = min(max(hi, 0), n_bands)
    return lo, hi


def kpoint_band_edges(
    channels: Sequence[BandChannel],
    *,
    weights: Optional[Sequence[float]] = None,
    occupation_tolerance: float = OCCUPATION_TOLERANCE,
) -> BandEdges:
    """Band edges of a periodic spectrum from the occupations it used.

    ``channels`` is one :class:`BandChannel` for a restricted result, two
    (alpha, beta) for an unrestricted one; the combined edges are taken over
    all of them, and each channel's own edges are returned in
    :attr:`BandEdges.channels` when there is more than one.

    ``weights`` are the k-point weights of the occupations (the mesh the SCF
    integrated); they fix the Fermi filling ``N / g`` used when the Fermi
    level crosses a band. ``None`` weighs every k-point equally, which is
    exact for a full Monkhorst-Pack mesh.
    """
    tol = float(occupation_tolerance)
    if not math.isfinite(tol) or tol <= 0.0 or tol >= 1.0:
        raise ValueError("band edges: occupation tolerance must be in (0, 1)")
    if not channels:
        raise ValueError("band edges: at least one channel is required")
    n_k = channels[0].n_kpoints
    if n_k == 0:
        raise ValueError("band edges: no k-points")
    for channel in channels:
        if channel.n_kpoints != n_k or len(channel.occupations_per_k) != n_k:
            raise ValueError(
                "band edges: spectra and occupations must match k-point for "
                "k-point in every channel"
            )
    w = _normalised_weights(weights, n_k)

    per_channel: List[BandEdges] = []
    vmax_per_k = np.full(n_k, -np.inf)
    cmin_per_k = np.full(n_k, np.inf)
    partition_occupied_max = -np.inf
    partition_empty_min = np.inf
    any_fractional = False
    any_crossing = False
    n_bands = 0
    for channel in channels:
        eps_blocks = _as_blocks(channel.eps_per_k, "energies")
        occ_blocks = _as_blocks(channel.occupations_per_k, "occupations")
        for eps, occ in zip(eps_blocks, occ_blocks):
            if eps.shape != occ.shape:
                raise ValueError(
                    "band edges: spectra and occupations must match band "
                    "for band"
                )
        channel = BandChannel(
            channel.spin,
            tuple(eps_blocks),
            tuple(occ_blocks),
            channel.max_occupation,
            channel.restricted_open_shell,
        )
        n_bands = max(n_bands, max(e.size for e in eps_blocks))
        valence, conduction, fractional, counts = _channel_state_classes(
            channel, tol
        )
        band_pair = (not fractional) and len(set(counts)) > 1
        crossing = False
        ch_vmax = np.full(n_k, -np.inf)
        ch_cmin = np.full(n_k, np.inf)
        if band_pair:
            # The occupied count differs between k-points: measure the band
            # pair at the Fermi filling, not the sampled-level spacing
            # (module docstring). With positive weights this is always a
            # crossing (gap < 0, proved above). A ZERO-weight point
            # (band-structure spelling) is filled with the step function at
            # the mu the weighted points fix, which can lie outside the true
            # gap; its count then differs although no band crosses, and the
            # band pair still measures the right, positive gap.
            filling = _fermi_filling(channel, occ_blocks, w)
            for ik, eps in enumerate(eps_blocks):
                ordered = np.sort(eps)
                lo, hi = _band_pair_indices(filling, ordered.size)
                if lo >= 0:
                    ch_vmax[ik] = ordered[lo]
                if hi < ordered.size:
                    ch_cmin[ik] = ordered[hi]
            if np.any(np.isfinite(ch_vmax)) and np.any(np.isfinite(ch_cmin)):
                crossing = bool(
                    np.min(ch_cmin[np.isfinite(ch_cmin)])
                    - np.max(ch_vmax[np.isfinite(ch_vmax)])
                    <= 0.0
                )
        else:
            for ik, eps in enumerate(eps_blocks):
                if np.any(valence[ik]):
                    ch_vmax[ik] = float(np.max(eps[valence[ik]]))
                if np.any(conduction[ik]):
                    ch_cmin[ik] = float(np.min(eps[conduction[ik]]))
        ch_partition_occ = -np.inf
        ch_partition_empty = np.inf
        for ik, eps in enumerate(eps_blocks):
            occupied = valence[ik]
            if np.any(occupied):
                ch_partition_occ = max(
                    ch_partition_occ, float(np.max(eps[occupied]))
                )
            if np.any(~occupied):
                ch_partition_empty = min(
                    ch_partition_empty, float(np.min(eps[~occupied]))
                )
        per_channel.append(
            _edges_from_extrema(
                ch_vmax,
                ch_cmin,
                ch_partition_occ,
                ch_partition_empty,
                fractional=fractional,
                crossing=crossing,
                n_bands=max(e.size for e in eps_blocks),
                spin=channel.spin,
            )
        )
        vmax_per_k = np.maximum(vmax_per_k, ch_vmax)
        cmin_per_k = np.minimum(cmin_per_k, ch_cmin)
        partition_occupied_max = max(partition_occupied_max, ch_partition_occ)
        partition_empty_min = min(partition_empty_min, ch_partition_empty)
        any_fractional = any_fractional or fractional
        any_crossing = any_crossing or crossing

    if len(per_channel) == 1:
        return per_channel[0]
    combined = _edges_from_extrema(
        vmax_per_k,
        cmin_per_k,
        partition_occupied_max,
        partition_empty_min,
        fractional=any_fractional,
        crossing=any_crossing,
        n_bands=n_bands,
        spin=None,
        channel_metallic=any(e.is_metallic for e in per_channel),
    )
    return BandEdges(
        **{
            name: getattr(combined, name)
            for name in combined.__dataclass_fields__
            if name != "channels"
        },
        channels=tuple(per_channel),
    )


# ---------------------------------------------------------------------------
# Reading a result
# ---------------------------------------------------------------------------
#
# Result layouts differ by driver (#505, #122): BIPOLE carries a per-k list in
# ``mo_energies`` and no ``mo_energies_k``; native GAPW multi-k carries
# ``mo_energies_k`` / ``occupations_k``; Gamma-only drivers carry single
# arrays; unrestricted results carry ``*_alpha`` / ``*_beta`` (optionally with
# a ``_k`` suffix); ROHF/ROKS carry alpha/beta aliases of ONE restricted set
# plus the real 2/1/0 ``occupations``. Dispatch on that here, once.


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return np.asarray(value).size == 0


def _first_present(result, *names):
    for name in names:
        value = getattr(result, name, None)
        if not _is_empty(value):
            return value
    return None


def _blocks(value) -> Tuple[np.ndarray, ...]:
    """One 1-D float block per k-point (a single array is one Gamma block)."""
    if isinstance(value, (list, tuple)):
        blocks = tuple(
            np.asarray(np.real(v), dtype=float).reshape(-1) for v in value
        )
    else:
        array = np.asarray(np.real(value), dtype=float)
        if array.ndim <= 1:
            blocks = (array.reshape(-1),)
        elif array.ndim == 2:
            blocks = tuple(np.asarray(row, dtype=float) for row in array)
        else:
            raise TypeError(
                f"band edges: eigenvalue/occupation data of shape "
                f"{array.shape!r} is neither per-k nor Gamma"
            )
    return blocks


def _matching(eps: Tuple[np.ndarray, ...], occ) -> Optional[Tuple[np.ndarray, ...]]:
    if _is_empty(occ):
        return None
    try:
        occ_blocks = _blocks(occ)
    except TypeError:
        return None
    if len(occ_blocks) != len(eps):
        return None
    if any(o.shape != e.shape for o, e in zip(occ_blocks, eps)):
        return None
    return occ_blocks


def result_band_channels(result) -> List[BandChannel]:
    """The occupation channels of a periodic result, or ``[]`` when it does
    not carry both eigenvalues and occupations for every channel.

    Occupations are the only reliable frontier classifier: periodic orbital
    energies have an arbitrary gauge, so their sign cannot say how many states
    are filled. A restricted result that carries no occupations but an even
    ``n_electrons`` is read with the band-index fill ``occ[:N/2] = 2`` at every
    k, which is exactly the band-pair reading of the module docstring.
    """
    method = str(getattr(result, "method", "") or "").strip().lower()
    energies = _first_present(result, "mo_energies_k", "mo_energies")
    occupations = _first_present(result, "occupations_k", "occupations")

    if method in ("rohf", "roks") and energies is not None:
        eps = _blocks(energies)
        occ = _matching(eps, occupations)
        if occ is None:
            return []
        return [BandChannel(None, eps, occ, 2.0, restricted_open_shell=True)]

    alpha = _first_present(result, "mo_energies_alpha_k", "mo_energies_alpha")
    beta = _first_present(result, "mo_energies_beta_k", "mo_energies_beta")
    if alpha is not None or beta is not None:
        channels: List[BandChannel] = []
        for spin, values, occ_names in (
            ("alpha", alpha, ("occupations_alpha_k", "occupations_alpha")),
            ("beta", beta, ("occupations_beta_k", "occupations_beta")),
        ):
            if values is None:
                continue
            eps = _blocks(values)
            occ = _matching(eps, _first_present(result, *occ_names))
            if occ is None:
                return []
            channels.append(BandChannel(spin, eps, occ, 1.0))
        if channels:
            return channels

    if energies is None:
        return []
    eps = _blocks(energies)
    occ = _matching(eps, occupations)
    if occ is not None:
        return [BandChannel(None, eps, occ, 2.0)]
    n_electrons = int(getattr(result, "n_electrons", 0) or 0)
    if n_electrons <= 0 or n_electrons % 2:
        return []
    n_occ = n_electrons // 2
    synthesized = []
    for block in eps:
        if n_occ > block.size:
            return []
        fill = np.zeros(block.size, dtype=float)
        fill[np.argsort(block, kind="stable")[:n_occ]] = 2.0
        synthesized.append(fill)
    return [BandChannel(None, eps, tuple(synthesized), 2.0)]


def result_kpoint_weights(result, n_kpoints: int) -> Optional[np.ndarray]:
    """The k-point weights a result's occupations were integrated with."""
    for name in ("kpoint_weights", "restart_weights", "weights"):
        value = getattr(result, name, None)
        if _is_empty(value):
            continue
        w = np.asarray(value, dtype=float).reshape(-1)
        if w.size == n_kpoints:
            return w
    kmesh = getattr(result, "kmesh", None)
    value = getattr(kmesh, "weights", None)
    if not _is_empty(value):
        w = np.asarray(value, dtype=float).reshape(-1)
        if w.size == n_kpoints:
            return w
    return None


def result_band_edges(result) -> Optional[BandEdges]:
    """Measured band edges of a periodic SCF result, ``None`` if unreadable.

    ``None`` means the result does not carry eigenvalues AND occupations for
    every channel (see :func:`result_band_channels`); it never encodes a
    metal. Whether the SCF converged is the caller's question: an
    unconverged result's edges describe its last iterate, not a material.
    """
    channels = result_band_channels(result)
    if not channels:
        return None
    n_k = channels[0].n_kpoints
    if any(ch.n_kpoints != n_k for ch in channels):
        return None
    return kpoint_band_edges(
        channels, weights=result_kpoint_weights(result, n_k)
    )


def _edges_from_extrema(
    vmax_per_k: np.ndarray,
    cmin_per_k: np.ndarray,
    partition_occupied_max: float,
    partition_empty_min: float,
    *,
    fractional: bool,
    crossing: bool,
    n_bands: int,
    spin: Optional[str],
    channel_metallic: bool = False,
) -> BandEdges:
    n_k = int(vmax_per_k.size)
    has_v = np.isfinite(vmax_per_k)
    has_c = np.isfinite(cmin_per_k)
    # Ties on equal energies resolve to the first k in mesh order.
    vbm_k = int(np.argmax(np.where(has_v, vmax_per_k, -np.inf))) if has_v.any() else -1
    cbm_k = int(np.argmin(np.where(has_c, cmin_per_k, np.inf))) if has_c.any() else -1
    vbm = float(vmax_per_k[vbm_k]) if vbm_k >= 0 else _NAN
    cbm = float(cmin_per_k[cbm_k]) if cbm_k >= 0 else _NAN
    indirect = cbm - vbm if (vbm_k >= 0 and cbm_k >= 0) else _NAN
    both = has_v & has_c
    if both.any():
        direct_per_k = np.where(both, cmin_per_k - vmax_per_k, np.inf)
        direct_k = int(np.argmin(direct_per_k))
        direct = float(direct_per_k[direct_k])
    else:
        direct_k = -1
        direct = _NAN
    gap_above = (
        float(partition_empty_min - partition_occupied_max)
        if (
            math.isfinite(partition_occupied_max)
            and math.isfinite(partition_empty_min)
        )
        else _NAN
    )
    metallic = (
        bool(fractional)
        or bool(crossing)
        or bool(channel_metallic)
        or (math.isfinite(indirect) and indirect <= 0.0)
    )
    return BandEdges(
        valence_band_max=vbm,
        conduction_band_min=cbm,
        indirect_gap=indirect,
        direct_gap=direct,
        gap_above_fermi_manifold=gap_above,
        is_metallic=metallic,
        fermi_level_crosses_band=bool(crossing),
        fractional_fermi_group=bool(fractional),
        valence_band_max_k=vbm_k,
        conduction_band_min_k=cbm_k,
        direct_gap_k=direct_k,
        valence_max_per_k=tuple(
            float(v) if np.isfinite(v) else _NAN for v in vmax_per_k
        ),
        conduction_min_per_k=tuple(
            float(c) if np.isfinite(c) else _NAN for c in cmin_per_k
        ),
        n_kpoints=n_k,
        n_bands=int(n_bands),
        spin=spin,
    )
