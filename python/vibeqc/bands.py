"""Band structures, k-paths, and density-of-states for periodic systems.

Two related products:

* **Band structure.** Eigenvalues of a one-electron Fock matrix sampled
  along a path of k-points (often through high-symmetry points of the
  Brillouin zone). Plot the resulting :class:`BandStructure` to see
  the dispersion of each band.
* **Density of states (DOS).** Eigenvalues collected over a dense
  Monkhorst-Pack k-mesh, broadened with a Gaussian, projected onto an
  energy grid. Plot the resulting :class:`DensityOfStates` to see how
  the electronic states distribute in energy -- band gaps and van Hove
  singularities are obvious by inspection.

Both objects are pure-Python dataclasses so callers can post-process,
serialize, or hand-render. The matching matplotlib plotters live in
:mod:`vibeqc.plot`.

Workflow
--------

The current public entry points take a real-space Fock and overlap
lattice set (``LatticeMatrixSet``) and sample at user-supplied
k-points. For the *non-interacting* (Hcore) limit this is straightforward:
build ``Hcore_lattice = T + V`` from
:func:`vibeqc.compute_kinetic_lattice` + :func:`vibeqc.compute_nuclear_lattice`
and pass it in. Convenience wrappers
:func:`band_structure_hcore` and :func:`density_of_states_hcore`
do this for you.

Bands and DOS computed from a converged interacting Fock matrix require
the user to supply ``F_real`` from their own SCF, since vibe-qc does not
yet persist the real-space converged Fock on the
:class:`PeriodicRHFResult` / :class:`PeriodicKSResult` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    bloch_sum,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    diagonalize_bloch,
    monkhorst_pack,
)
from .basis_crystal import _ELEMENT_SYMBOLS
from .properties import _shell_nao
from .properties import _shell_to_atom as _canonical_shell_to_atom

# Hartree -> eV conversion (CODATA 2018).  Same constant as in the QVF writer;
# duplicated here to keep bands.py importable without output/.
_HARTREE_TO_EV: float = 27.211386245988

__all__ = [
    "BandStructure",
    "DensityOfStates",
    "FrontierBands",
    "KPath",
    "ProjectedDensityOfStates",
    "kpath_from_segments",
    "band_structure",
    "band_structure_hcore",
    "band_structure_projected",
    "density_of_states",
    "density_of_states_hcore",
    "density_of_states_projected",
    "density_of_states_projected_hcore",
    "ao_groups_per_atom",
    "ao_groups_per_atom_l",
    "frontier_bands",
    "per_k_band_energies",
    "require_periodic_system",
]


def require_periodic_system(system, *, feature: str) -> None:
    """Fail fast when a k-point feature is requested for a molecule.

    Band structures, DOS meshes, and k-paths are defined on the Brillouin
    zone of a lattice; a :class:`vibeqc.Molecule` (or anything else without
    ``.lattice`` / ``.unit_cell``) has neither. Without this check the
    request dies deep in the machinery with an unhelpful ``AttributeError``
    ("'Molecule' object has no attribute 'lattice'") -- raise a targeted
    ``TypeError`` at the user-facing entry point instead.
    """
    if not hasattr(system, "lattice") or not hasattr(system, "unit_cell"):
        raise TypeError(
            f"{feature}: system must be a PeriodicSystem (a cell with a "
            f"lattice); got {type(system).__name__}. Molecules have no "
            f"lattice or Brillouin zone, so band structures / DOS / "
            f"k-point sampling are not defined for them -- these are "
            f"periodic-boundary-condition features. Use vibeqc.run_job("
            f"...) for molecular spectra, or build a PeriodicSystem "
            f"(set .lattice / .unit_cell / .dim) for a crystal."
        )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class KPath:
    """A discretised k-path -- sequence of k-points with cumulative arc
    length (for the x-axis when plotting) and labeled high-symmetry
    points (for the x-axis tick marks)."""

    kpoints_cart: np.ndarray  # (N, 3), bohr⁻¹
    kpoints_frac: np.ndarray  # (N, 3), reciprocal-lattice fractional
    distances: np.ndarray  # (N,) cumulative |Δk_cart|
    labels: List[Tuple[float, str]]  # (distance, label) ticks for plotting
    # Route keys into [routes.numerics] naming the published path convention
    # this k-path follows (HPKOT via seekpath). Carried over from
    # KPoints.citation_numerics by KPoints.to_kpath, so a job that attaches a
    # band structure can cite it. Manually specified segments follow no
    # published convention and leave this empty.
    citation_numerics: Tuple[str, ...] = ()

    @property
    def n_points(self) -> int:
        return int(self.kpoints_cart.shape[0])


@dataclass
class BandStructure:
    """Eigenvalues at every k along a path.

    ``energies`` shape is ``(n_points, n_bands)`` with bands sorted in
    ascending energy at each k. Reference energy ``e_fermi`` is the
    HOMO eigenvalue (over all sampled k-points) when ``n_electrons`` is
    set, else ``None``.

    Optional ``projections`` carries Mulliken-projected band-character
    weights of shape ``(n_kpoints, n_bands, n_channels)``, and ``channels``
    names each channel (e.g. ``"Si-3s"``, ``"O-2p"``). Both are set by
    :func:`band_structure_projected`; absent for a plain band structure.
    """

    kpath: KPath
    energies: np.ndarray  # (n_points, n_bands), Hartree
    e_fermi: Optional[float] = None  # HOMO of the sampled bands, Hartree
    n_electrons_per_cell: Optional[int] = None
    # Fat-band fields — set only by band_structure_projected().
    projections: Optional[np.ndarray] = (
        None  # (n_k, n_bands, n_channels), Mulliken weight
    )
    channels: Optional[List[str]] = None  # length n_channels
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_bands(self) -> int:
        return int(self.energies.shape[1])

    @property
    def shifted_energies(self) -> np.ndarray:
        """Energies relative to ``e_fermi`` (or zero if not set).
        Property (no parens) -- was a method until v0.4 polish."""
        return self.energies - (self.e_fermi or 0.0)


@dataclass
class DensityOfStates:
    """Total density of states.

    ``energies`` is the energy grid (Hartree). ``dos`` is the
    Gaussian-broadened state count per energy unit per unit cell:

        dos(e) = (1 / N_k) . S_{k, n}  w_k . g(e - e_n(k))

    with ``g`` a unit-area Gaussian of width ``sigma``. The integral of
    ``dos`` over all energy equals ``n_bands`` (every state contributes
    one).
    """

    energies: np.ndarray  # (n_e,) Hartree
    dos: np.ndarray  # (n_e,) states / Hartree / cell
    sigma: float  # broadening, Hartree
    e_fermi: Optional[float] = None

    @property
    def shifted_energies(self) -> np.ndarray:
        """Energies relative to ``e_fermi`` (or zero if not set).
        Property (no parens) -- was a method until v0.4 polish."""
        return self.energies - (self.e_fermi or 0.0)


@dataclass
class ProjectedDensityOfStates:
    """DOS broken into per-group contributions.

    ``contributions`` maps a group label (e.g. ``"H1"``, ``"O3-p"``,
    or whatever the caller supplies) to a length-``n_e`` array. Sum of
    all contributions over groups equals :attr:`total` to machine
    precision when the projection is exhaustive (every AO assigned to
    exactly one group).

    The projection is Mulliken-style:

        w_{mu,n,k} = Re[ C*_{mu,n}(k) . (S(k) C(k))_{mu,n} ]

    so for any (n, k), S_mu w = (C^+ S C)_{nn} = 1 (orbitals normalized).
    The PDOS for group ``g`` is then

        PDOS_g(e) = S_k w_k S_n  ( S_{mu in g} w_{mu,n,k} ) . g_s(e - e_{n,k})

    with ``g_s`` a unit-area Gaussian.

    Sums are weighted by ``kmesh.weights``; the totals match
    :class:`DensityOfStates` to FP noise.
    """

    energies: np.ndarray  # (n_e,) Hartree
    total: np.ndarray  # (n_e,) states / Hartree / cell
    contributions: Dict[str, np.ndarray]  # label -> (n_e,)
    sigma: float  # broadening, Hartree
    e_fermi: Optional[float] = None

    @property
    def shifted_energies(self) -> np.ndarray:
        """Property (no parens) -- was a method until v0.4 polish."""
        return self.energies - (self.e_fermi or 0.0)

    @property
    def group_labels(self) -> List[str]:
        """Group label list (Mulliken keys). Property (no parens) --
        was a method until v0.4 polish."""
        return list(self.contributions.keys())

    def as_density_of_states(self) -> "DensityOfStates":
        """Discard the per-group split and return the total DOS -- useful
        for handing the same object to plain :func:`vibeqc.plot.dos_figure`."""
        return DensityOfStates(
            energies=self.energies,
            dos=self.total,
            sigma=self.sigma,
            e_fermi=self.e_fermi,
        )


# ---------------------------------------------------------------------------
# k-path construction
# ---------------------------------------------------------------------------


def kpath_from_segments(
    system: PeriodicSystem,
    segments: Sequence[Tuple[Sequence[float], str, Sequence[float], str]],
    *,
    points_per_segment: int = 30,
) -> KPath:
    """Stitch a piecewise-linear k-path through high-symmetry points.

    ``segments`` is a list of ``(start_frac, start_label, end_frac, end_label)``
    tuples. Adjacent segments may share an endpoint -- e.g. for a path
    Γ -> X -> M -> Γ, three segments share their successive endpoints.

    Returned ``KPath`` carries Cartesian k-points (bohr⁻¹), the matching
    fractional coordinates, the cumulative arc length along the path
    (used as the x-coordinate when plotting), and the (distance, label)
    tick marks for high-symmetry points.

    ``points_per_segment`` controls the resolution of each leg.
    """
    if not segments:
        raise ValueError("kpath_from_segments: at least one segment required")
    B = system.reciprocal_lattice()  # 3 x 3, columns = b_1, b_2, b_3

    all_frac: List[np.ndarray] = []
    labels: List[Tuple[float, str]] = []
    distances: List[float] = []
    last_cart: Optional[np.ndarray] = None
    cumulative = 0.0

    for s_idx, (k_a, lbl_a, k_b, lbl_b) in enumerate(segments):
        k_a = np.asarray(k_a, dtype=float).reshape(3)
        k_b = np.asarray(k_b, dtype=float).reshape(3)

        # Skip the duplicate point at segment boundaries.
        i_start = 0 if s_idx == 0 else 1
        for i in range(i_start, points_per_segment + 1):
            t = i / points_per_segment
            kf = (1.0 - t) * k_a + t * k_b
            kc = B @ kf
            if last_cart is not None:
                cumulative += float(np.linalg.norm(kc - last_cart))
            all_frac.append(kf)
            distances.append(cumulative)
            last_cart = kc
            if i == 0:
                labels.append((cumulative, lbl_a))
            elif i == points_per_segment:
                labels.append((cumulative, lbl_b))

    frac_arr = np.asarray(all_frac, dtype=float)
    cart_arr = (B @ frac_arr.T).T
    dist_arr = np.asarray(distances, dtype=float)

    # Coalesce duplicate-position labels at segment joins by collapsing
    # adjacent entries with the same distance and combining their labels
    # with "|" -- a common convention in band-structure plots.
    dedup: List[Tuple[float, str]] = []
    for d, lbl in labels:
        if dedup and abs(dedup[-1][0] - d) < 1e-12 and dedup[-1][1] != lbl:
            dedup[-1] = (d, f"{dedup[-1][1]}|{lbl}")
        elif not dedup or dedup[-1] != (d, lbl):
            dedup.append((d, lbl))

    return KPath(
        kpoints_cart=cart_arr,
        kpoints_frac=frac_arr,
        distances=dist_arr,
        labels=dedup,
    )


# ---------------------------------------------------------------------------
# Band structure -- sample a Fock matrix at every k along a path
# ---------------------------------------------------------------------------


def _eigenvalues_at_kpoints(
    F_terms: Sequence[LatticeMatrixSet],
    S_real: LatticeMatrixSet,
    kpoints_cart: np.ndarray,
) -> np.ndarray:
    """For each k in ``kpoints_cart`` Bloch-sum each term in ``F_terms``,
    sum them to form ``F(k)``, Bloch-sum ``S(g)``, and diagonalize to
    obtain band energies. Returned shape: ``(n_kpoints, nbf)``.

    Taking ``F_terms`` as a list (rather than a single ``LatticeMatrixSet``)
    avoids materialising ``Hcore = T + V`` as its own lattice matrix set --
    Bloch summation is linear, so we just add the per-term Bloch sums.
    """
    n_pts = kpoints_cart.shape[0]
    nbf = S_real.nbf
    out = np.empty((n_pts, nbf), dtype=float)
    for i in range(n_pts):
        k = kpoints_cart[i]
        Fk = bloch_sum(F_terms[0], k)
        for term in F_terms[1:]:
            Fk = Fk + bloch_sum(term, k)
        Sk = bloch_sum(S_real, k)
        # Symmetrize tiny imaginary drift before diagonalizing.
        Fk = 0.5 * (Fk + Fk.conj().T)
        Sk = 0.5 * (Sk + Sk.conj().T)
        bd = diagonalize_bloch(Fk, Sk)
        out[i, :] = np.asarray(bd.energies)
    return out


def band_structure(
    F_real: LatticeMatrixSet,
    S_real: LatticeMatrixSet,
    kpath: KPath,
    *,
    n_electrons_per_cell: Optional[int] = None,
) -> BandStructure:
    """Sample a real-space Fock matrix along a k-path.

    The Fock and overlap come from a converged (or non-interacting) SCF
    in real-space-lattice form. For non-interacting bands use
    :func:`band_structure_hcore`.

    If ``n_electrons_per_cell`` is supplied, the highest occupied
    eigenvalue across all sampled k-points is recorded as ``e_fermi``
    so a plotter can shift the reference. For closed-shell systems
    ``n_electrons_per_cell // 2`` bands are occupied at each k.
    """
    energies = _eigenvalues_at_kpoints([F_real], S_real, kpath.kpoints_cart)

    e_fermi: Optional[float] = None
    if n_electrons_per_cell is not None:
        if n_electrons_per_cell % 2 != 0:
            # Open-shell band-structure interpretation is more involved
            # (alpha vs beta channels). For now we only set e_fermi for
            # closed-shell systems; the array of energies is still
            # returned so the user can decide what to do.
            e_fermi = None
        else:
            n_occ = n_electrons_per_cell // 2
            occ_max = energies[:, :n_occ].max() if n_occ > 0 else None
            e_fermi = float(occ_max) if occ_max is not None else None

    return BandStructure(
        kpath=kpath,
        energies=energies,
        e_fermi=e_fermi,
        n_electrons_per_cell=n_electrons_per_cell,
    )


def band_structure_projected(
    F_real: LatticeMatrixSet,
    S_real: LatticeMatrixSet,
    system: PeriodicSystem,
    basis: BasisSet,
    kpath: KPath,
    *,
    n_electrons_per_cell: Optional[int] = None,
) -> BandStructure:
    """Band structure with Mulliken-projected orbital-character weights
    ("fat bands").

    At every k-point, computes both eigenvalues and the Mulliken
    projection weight of each band onto (atom, l) channels. The returned
    :class:`BandStructure` carries ``projections`` of shape
    ``(n_kpoints, n_bands, n_channels)`` and a ``channels`` list whose
    entries look like ``"Si-3s"``, ``"O-2p"``, etc.

    The projection is the same AO-resolved Mulliken weight used by
    :func:`density_of_states_projected`, delivered band-by-band at every
    sampled k-point instead of integrated over energy.
    """
    require_periodic_system(system, feature="band_structure_projected")
    groups = ao_groups_per_atom_l(system, basis)
    channel_labels = list(groups.keys())
    n_channels = len(channel_labels)
    if n_channels == 0:
        raise ValueError(
            "band_structure_projected: ao_groups_per_atom_l returned "
            "zero channels — is the basis empty?"
        )

    # Build a (n_ao,) array mapping each AO to its channel index.
    ao_to_channel = np.full(S_real.nbf, -1, dtype=np.int64)
    for ch_idx, label in enumerate(channel_labels):
        for ao in groups[label]:
            ao_to_channel[int(ao)] = ch_idx
    if (ao_to_channel < 0).any():
        missing = int((ao_to_channel < 0).sum())
        raise ValueError(
            f"band_structure_projected: {missing} AOs are not covered "
            f"by ao_groups_per_atom_l — the groups must partition all "
            f"AOs exactly once."
        )

    kpoints = kpath.kpoints_cart
    n_k = kpoints.shape[0]
    n_bf = S_real.nbf
    energies = np.empty((n_k, n_bf), dtype=float)
    projections = np.empty((n_k, n_bf, n_channels), dtype=float)

    for ki in range(n_k):
        Fk = bloch_sum(F_real, kpoints[ki])
        Sk = bloch_sum(S_real, kpoints[ki])
        Fk = 0.5 * (Fk + Fk.conj().T)
        Sk = 0.5 * (Sk + Sk.conj().T)
        bd = diagonalize_bloch(Fk, Sk)
        energies[ki, :] = np.asarray(bd.energies)
        Ck = np.asarray(bd.coefficients)
        # Mulliken weight: (n_ao, n_bands), column-sum = 1 per band.
        mw = _mulliken_band_weights(Ck, Sk)  # (n_bf, n_bands)
        # Accumulate into channels by summing over AOs in each channel.
        for ch_idx in range(n_channels):
            mask = ao_to_channel == ch_idx
            projections[ki, :, ch_idx] = mw[mask, :].sum(axis=0)

    e_fermi: Optional[float] = None
    if n_electrons_per_cell is not None:
        if n_electrons_per_cell % 2 != 0:
            e_fermi = None
        else:
            n_occ = n_electrons_per_cell // 2
            occ_max = energies[:, :n_occ].max() if n_occ > 0 else None
            e_fermi = float(occ_max) if occ_max is not None else None

    return BandStructure(
        kpath=kpath,
        energies=energies,
        e_fermi=e_fermi,
        n_electrons_per_cell=n_electrons_per_cell,
        projections=projections,
        channels=channel_labels,
    )


def band_structure_hcore(
    system: PeriodicSystem,
    basis: BasisSet,
    kpath: KPath,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    n_electrons_per_cell: Optional[int] = None,
) -> BandStructure:
    """Non-interacting (Hcore) band structure: eigenvalues of T + V at
    every k-point. Useful for system-shape sanity checks before
    investing in a full SCF.
    """
    require_periodic_system(system, feature="band_structure_hcore")
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    V = compute_nuclear_lattice(basis, system, opts)

    energies = _eigenvalues_at_kpoints([T, V], S, kpath.kpoints_cart)

    e_fermi: Optional[float] = None
    if n_electrons_per_cell is not None and n_electrons_per_cell % 2 == 0:
        n_occ = n_electrons_per_cell // 2
        if n_occ > 0:
            e_fermi = float(energies[:, :n_occ].max())

    return BandStructure(
        kpath=kpath,
        energies=energies,
        e_fermi=e_fermi,
        n_electrons_per_cell=n_electrons_per_cell,
    )


# ---------------------------------------------------------------------------
# Density of states
# ---------------------------------------------------------------------------


def _gaussian_dos(
    energies_per_k: np.ndarray,  # (n_k, n_bands)
    weights: np.ndarray,  # (n_k,) k-point weights, sum = 1
    energy_grid: np.ndarray,  # (n_e,)
    sigma: float,
) -> np.ndarray:
    """Sum of unit-area Gaussians centered on every (k, band) eigenvalue,
    weighted by ``weights[k]``."""
    n_e = energy_grid.size
    dos = np.zeros(n_e, dtype=float)
    inv_2sigma2 = 0.5 / (sigma * sigma)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    for k, w in enumerate(weights):
        for eps in energies_per_k[k]:
            dos += w * norm * np.exp(-((energy_grid - eps) ** 2) * inv_2sigma2)
    return dos


def density_of_states(
    F_real: LatticeMatrixSet,
    S_real: LatticeMatrixSet,
    kmesh: BlochKMesh,
    *,
    sigma: float = 0.01,
    energy_grid: Optional[np.ndarray] = None,
    n_grid: int = 401,
    pad: float = 5.0,
    n_electrons_per_cell: Optional[int] = None,
) -> DensityOfStates:
    """Total DOS computed by Gaussian-broadening every eigenvalue of
    ``F(k)`` over ``kmesh`` onto an energy grid.

    ``sigma`` is the Gaussian width in Hartree. ``energy_grid`` defaults
    to a uniform ``n_grid``-point grid spanning the eigenvalue range
    extended by ``pad.sigma`` on either side so the broadened tails fit.
    """
    kpoints = np.asarray([np.asarray(k) for k in kmesh.kpoints])
    weights = np.asarray(kmesh.weights, dtype=float)
    if abs(weights.sum() - 1.0) > 1e-8:
        # Defensive: make this work even with a mesh whose weights weren't
        # normalized by the caller.
        weights = weights / weights.sum()

    energies_per_k = _eigenvalues_at_kpoints([F_real], S_real, kpoints)

    if energy_grid is None:
        e_min = energies_per_k.min() - pad * sigma
        e_max = energies_per_k.max() + pad * sigma
        energy_grid = np.linspace(e_min, e_max, n_grid)

    dos = _gaussian_dos(energies_per_k, weights, energy_grid, sigma)

    e_fermi: Optional[float] = None
    if n_electrons_per_cell is not None and n_electrons_per_cell % 2 == 0:
        n_occ = n_electrons_per_cell // 2
        if n_occ > 0:
            e_fermi = float(energies_per_k[:, :n_occ].max())

    return DensityOfStates(
        energies=energy_grid,
        dos=dos,
        sigma=sigma,
        e_fermi=e_fermi,
    )


def _density_of_states_from_terms(
    fock_terms: Sequence[LatticeMatrixSet],
    S_real: LatticeMatrixSet,
    kmesh: BlochKMesh,
    *,
    sigma: float = 0.01,
    energy_grid: Optional[np.ndarray] = None,
    n_grid: int = 401,
    pad: float = 5.0,
    n_electrons_per_cell: Optional[int] = None,
) -> DensityOfStates:
    """Total DOS from multiple Fock lattice terms -- internal helper.

    Unlike :func:`density_of_states` which takes a single ``F_real``
    lattice matrix set, this variant accepts a sequence ``[T_real, V_real,
    F2e_real, ...]`` and Bloch-sums each term independently before adding
    them in k-space.  This lets SCF drivers that decompose the Fock
    matrix (e.g. BIPOLE's T + V + J + K split) feed their native terms
    directly without materialising a merged ``F_real``.
    """
    kpoints = np.asarray([np.asarray(k) for k in kmesh.kpoints])
    weights = np.asarray(kmesh.weights, dtype=float)
    if abs(weights.sum() - 1.0) > 1e-8:
        weights = weights / weights.sum()

    energies_per_k = _eigenvalues_at_kpoints(fock_terms, S_real, kpoints)

    if energy_grid is None:
        e_min = energies_per_k.min() - pad * sigma
        e_max = energies_per_k.max() + pad * sigma
        energy_grid = np.linspace(e_min, e_max, n_grid)

    dos = _gaussian_dos(energies_per_k, weights, energy_grid, sigma)

    e_fermi: Optional[float] = None
    if n_electrons_per_cell is not None and n_electrons_per_cell % 2 == 0:
        n_occ = n_electrons_per_cell // 2
        if n_occ > 0:
            e_fermi = float(energies_per_k[:, :n_occ].max())

    return DensityOfStates(
        energies=energy_grid,
        dos=dos,
        sigma=sigma,
        e_fermi=e_fermi,
    )


# ---------------------------------------------------------------------------
# Projected density of states (V5b)
# ---------------------------------------------------------------------------

# Spectroscopic letter for a given angular-momentum quantum number; "?" past
# h since vibe-qc only ships through l=5.
_L_LETTERS = ("s", "p", "d", "f", "g", "h")


def _shell_to_atom(basis: BasisSet) -> np.ndarray:
    """Length-``nbasis`` int array mapping each AO to the 0-based atom
    it lives on. Re-exported from :mod:`vibeqc.properties`, which owns
    the canonical derivation; ``bands`` used to carry its own copy that
    assumed pure spherical AOs and so came out short on a Cartesian
    basis. ``properties`` pulls in nothing from ``bands``, so this is
    not a cycle."""
    return _canonical_shell_to_atom(basis)


def _shell_to_l(basis: BasisSet) -> np.ndarray:
    """Length-``nbasis`` int array mapping each AO to its shell's l.

    Shares :func:`vibeqc.properties._shell_nao` with
    :func:`_shell_to_atom` so the two arrays stay the same length on a
    Cartesian basis; callers such as :func:`ao_groups_per_atom_l` mask
    one against the other."""
    per_ao: List[int] = []
    for shell in basis.shells():
        per_ao.extend([int(shell.l)] * _shell_nao(shell))
    return np.asarray(per_ao, dtype=np.int64)


def _atom_label(z: int, idx_1based: int) -> str:
    sym = _ELEMENT_SYMBOLS[z] if 0 < z < len(_ELEMENT_SYMBOLS) else f"Z{z}"
    return f"{sym}{idx_1based}"


def _l_letter(l: int) -> str:
    return _L_LETTERS[l] if 0 <= l < len(_L_LETTERS) else f"l{l}"


def ao_groups_per_atom(
    system: PeriodicSystem,
    basis: BasisSet,
) -> Dict[str, List[int]]:
    """Partition AO indices into per-atom groups.

    Returns a dict keyed by ``"<symbol><1-based-atom-index>"``
    (e.g. ``"H1"``, ``"O3"``) with values the list of AO indices on
    that atom. Useful as the ``groups`` argument to
    :func:`density_of_states_projected`."""
    ao_to_atom = _shell_to_atom(basis)
    out: Dict[str, List[int]] = {}
    for atom_idx, atom in enumerate(system.unit_cell):
        label = _atom_label(int(atom.Z), atom_idx + 1)
        ao_indices = [int(i) for i in np.where(ao_to_atom == atom_idx)[0]]
        if ao_indices:
            out[label] = ao_indices
    return out


def ao_groups_per_atom_l(
    system: PeriodicSystem,
    basis: BasisSet,
) -> Dict[str, List[int]]:
    """Partition AO indices by atom x angular momentum.

    Labels look like ``"H1-s"`` or ``"O3-p"``. Empty channels are
    dropped (so an atom with only an s shell yields one entry, not
    six)."""
    ao_to_atom = _shell_to_atom(basis)
    ao_to_l = _shell_to_l(basis)
    out: Dict[str, List[int]] = {}
    for atom_idx, atom in enumerate(system.unit_cell):
        for l in range(int(ao_to_l.max()) + 1 if ao_to_l.size else 0):
            mask = (ao_to_atom == atom_idx) & (ao_to_l == l)
            ao_indices = [int(i) for i in np.where(mask)[0]]
            if not ao_indices:
                continue
            label = f"{_atom_label(int(atom.Z), atom_idx + 1)}-{_l_letter(l)}"
            out[label] = ao_indices
    return out


def _mulliken_band_weights(C_k: np.ndarray, S_k: np.ndarray) -> np.ndarray:
    """Per-AO Mulliken weight of every band at a single k.

    ``C_k`` is ``(n_bf, n_bands)``; ``S_k`` is ``(n_bf, n_bf)``. Returns
    a real ``(n_bf, n_bands)`` matrix whose column-sum is 1 per band
    (orbitals are S-normalized by :func:`diagonalize_bloch`).
    """
    SC = S_k @ C_k  # (n_bf, n_bands), complex
    return np.real(np.conj(C_k) * SC)  # (n_bf, n_bands)


def _projected_gaussian_dos(
    energies_per_k: np.ndarray,  # (n_k, n_bands)
    weights_per_k: np.ndarray,  # (n_k, n_bands) -- Mulliken sum on group
    k_weights: np.ndarray,  # (n_k,) k-mesh weights, sum = 1
    energy_grid: np.ndarray,  # (n_e,)
    sigma: float,
) -> np.ndarray:
    """Sum of unit-area Gaussians scaled by ``weights_per_k``."""
    n_e = energy_grid.size
    pdos = np.zeros(n_e, dtype=float)
    inv_2sigma2 = 0.5 / (sigma * sigma)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    for k, wk in enumerate(k_weights):
        for n, eps in enumerate(energies_per_k[k]):
            coeff = wk * float(weights_per_k[k, n])
            if coeff == 0.0:
                continue
            pdos += coeff * norm * np.exp(-((energy_grid - eps) ** 2) * inv_2sigma2)
    return pdos


def _projected_dos_from_terms(
    F_terms: Sequence[LatticeMatrixSet],
    S_real: LatticeMatrixSet,
    kmesh: BlochKMesh,
    groups: Mapping[str, Sequence[int]],
    sigma: float,
    energy_grid: Optional[np.ndarray],
    n_grid: int,
    pad: float,
    n_electrons_per_cell: Optional[int],
) -> ProjectedDensityOfStates:
    """Shared core: takes a list of Fock terms (any number, >=1) so the
    Hcore wrapper can pass ``[T, V]`` without materialising a merged
    LatticeMatrixSet (the C++ class is read-only from Python)."""
    nbf = S_real.nbf
    for label, ao_indices in groups.items():
        bad = [i for i in ao_indices if not (0 <= int(i) < nbf)]
        if bad:
            raise IndexError(
                f"density_of_states_projected: group {label!r} contains "
                f"out-of-range AO indices {bad!r} (nbf={nbf})"
            )

    kpoints = np.asarray([np.asarray(k) for k in kmesh.kpoints])
    k_weights = np.asarray(kmesh.weights, dtype=float)
    if abs(k_weights.sum() - 1.0) > 1e-8:
        k_weights = k_weights / k_weights.sum()

    n_k = kpoints.shape[0]
    energies_per_k = np.empty((n_k, nbf), dtype=float)
    # Per-k Mulliken weight matrix M[k, n, mu]. Stored (band, ao) so the
    # group AO indices are the inner slice.
    mulliken = np.empty((n_k, nbf, nbf), dtype=float)
    for ki in range(n_k):
        Fk = bloch_sum(F_terms[0], kpoints[ki])
        for term in F_terms[1:]:
            Fk = Fk + bloch_sum(term, kpoints[ki])
        Sk = bloch_sum(S_real, kpoints[ki])
        Fk = 0.5 * (Fk + Fk.conj().T)
        Sk = 0.5 * (Sk + Sk.conj().T)
        bd = diagonalize_bloch(Fk, Sk)
        energies_per_k[ki, :] = np.asarray(bd.energies)
        Ck = np.asarray(bd.coefficients)
        mulliken[ki, :, :] = _mulliken_band_weights(Ck, Sk).T

    if energy_grid is None:
        e_min = energies_per_k.min() - pad * sigma
        e_max = energies_per_k.max() + pad * sigma
        energy_grid = np.linspace(e_min, e_max, n_grid)

    contributions: Dict[str, np.ndarray] = {}
    total = np.zeros(energy_grid.shape, dtype=float)
    for label, ao_indices in groups.items():
        ao_arr = np.asarray(list(ao_indices), dtype=np.int64)
        group_weights = mulliken[:, :, ao_arr].sum(axis=2)
        pdos = _projected_gaussian_dos(
            energies_per_k,
            group_weights,
            k_weights,
            energy_grid,
            sigma,
        )
        contributions[label] = pdos
        total += pdos

    e_fermi: Optional[float] = None
    if n_electrons_per_cell is not None and n_electrons_per_cell % 2 == 0:
        n_occ = n_electrons_per_cell // 2
        if n_occ > 0:
            e_fermi = float(energies_per_k[:, :n_occ].max())

    return ProjectedDensityOfStates(
        energies=energy_grid,
        total=total,
        contributions=contributions,
        sigma=sigma,
        e_fermi=e_fermi,
    )


def density_of_states_projected(
    F_real: LatticeMatrixSet,
    S_real: LatticeMatrixSet,
    kmesh: BlochKMesh,
    *,
    groups: Mapping[str, Sequence[int]],
    sigma: float = 0.01,
    energy_grid: Optional[np.ndarray] = None,
    n_grid: int = 401,
    pad: float = 5.0,
    n_electrons_per_cell: Optional[int] = None,
) -> ProjectedDensityOfStates:
    """Mulliken-projected DOS over a Monkhorst-Pack mesh.

    For each k we Bloch-sum ``F`` and ``S``, diagonalize to obtain
    ``(e_n, C_n)``, and compute the AO-resolved Mulliken band weight
    ``Re[C* . (S C)]``. The PDOS for a group ``g`` (a list of AO
    indices) is the energy-broadened, k-weighted sum of the Mulliken
    weights of all AOs in ``g``.

    Parameters
    ----------
    groups
        Dict ``{label: [ao_index, ...]}``. Labels become the keys of
        :attr:`ProjectedDensityOfStates.contributions`. Use
        :func:`ao_groups_per_atom` or :func:`ao_groups_per_atom_l` for
        the standard partitions; pass a custom dict to project onto
        arbitrary AO subsets.
    sigma, energy_grid, n_grid, pad, n_electrons_per_cell
        Same meaning as in :func:`density_of_states`.

    Returns
    -------
    ProjectedDensityOfStates
        Per-group DOS plus a precomputed total. The total is the sum
        over all groups (no double-counting; if your ``groups`` are not
        a partition of the AO indices, the total reflects only the
        covered AOs and the relation to the unprojected DOS no longer
        holds).
    """
    if not groups:
        raise ValueError("density_of_states_projected: groups dict is empty")
    return _projected_dos_from_terms(
        [F_real],
        S_real,
        kmesh,
        groups,
        sigma,
        energy_grid,
        n_grid,
        pad,
        n_electrons_per_cell,
    )


def density_of_states_projected_hcore(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: Sequence[int],
    *,
    projection: Union[str, Mapping[str, Sequence[int]]] = "atoms",
    sigma: float = 0.01,
    n_grid: int = 401,
    pad: float = 5.0,
    lattice_opts: Optional[LatticeSumOptions] = None,
    n_electrons_per_cell: Optional[int] = None,
) -> ProjectedDensityOfStates:
    """Hcore-projected DOS -- convenience wrapper.

    ``projection`` is either:

    * ``"atoms"`` -> :func:`ao_groups_per_atom`,
    * ``"atoms_l"`` -> :func:`ao_groups_per_atom_l`, or
    * a dict of explicit ``{label: [ao_indices]}`` groups.
    """
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    V = compute_nuclear_lattice(basis, system, opts)

    if isinstance(projection, str):
        if projection == "atoms":
            groups = ao_groups_per_atom(system, basis)
        elif projection == "atoms_l":
            groups = ao_groups_per_atom_l(system, basis)
        else:
            raise ValueError(
                "projection must be 'atoms', 'atoms_l', or an explicit "
                f"groups dict; got {projection!r}"
            )
    else:
        groups = dict(projection)
    if not groups:
        raise ValueError(
            "density_of_states_projected_hcore: projection produced no "
            "groups (system has no atoms?)"
        )

    km = monkhorst_pack(system, list(mesh))
    return _projected_dos_from_terms(
        [T, V],
        S,
        km,
        groups,
        sigma,
        None,
        n_grid,
        pad,
        n_electrons_per_cell,
    )


def density_of_states_hcore(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: Sequence[int],
    *,
    sigma: float = 0.01,
    n_grid: int = 401,
    pad: float = 5.0,
    lattice_opts: Optional[LatticeSumOptions] = None,
    n_electrons_per_cell: Optional[int] = None,
) -> DensityOfStates:
    """Non-interacting DOS for a system on a Monkhorst-Pack mesh.
    Convenience for quick band-structure overviews."""
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    V = compute_nuclear_lattice(basis, system, opts)

    km = monkhorst_pack(system, list(mesh))
    kpoints = np.asarray([np.asarray(k) for k in km.kpoints])
    weights = np.asarray(km.weights, dtype=float)
    if abs(weights.sum() - 1.0) > 1e-8:
        weights = weights / weights.sum()

    energies_per_k = _eigenvalues_at_kpoints([T, V], S, kpoints)

    e_min = energies_per_k.min() - pad * sigma
    e_max = energies_per_k.max() + pad * sigma
    energy_grid = np.linspace(e_min, e_max, n_grid)

    dos = _gaussian_dos(energies_per_k, weights, energy_grid, sigma)

    e_fermi: Optional[float] = None
    if n_electrons_per_cell is not None and n_electrons_per_cell % 2 == 0:
        n_occ = n_electrons_per_cell // 2
        if n_occ > 0:
            e_fermi = float(energies_per_k[:, :n_occ].max())

    return DensityOfStates(
        energies=energy_grid,
        dos=dos,
        sigma=sigma,
        e_fermi=e_fermi,
    )


# ===================================================================== #
# Per-k band energies from a converged periodic result                  #
# ===================================================================== #
#
# GitLab issue #505.  Reading "the bands of this result" looks trivial and
# is not: the layout differs by driver, and vibe-qc used to expose no way
# to ask.  Every consumer duck-typed the result object, and the same class
# of mistake has now been made twice, five days apart, in two different
# campaign carriers -- each time after a certified SCF, each time burning
# a full-node wall and turning a converged paper row into rc 9:
#
#   * #15 (BUG 124) -- ``float(b[n_occ - 1])`` where the selection was a
#     *degenerate manifold*, i.e. an ndarray rather than a scalar.
#   * #135 -- a carrier branched on ``mo_energies_k``, did not find it on
#     a ``PBCBipoleRHFResult``, and fell back to ``eps_k = [mo_energies]``,
#     wrapping the whole per-k list as ONE pseudo-k.  Its guard
#     ``len(b) > n_occ`` then counted *k-points* instead of *bands* and
#     passed, so ``float(row)`` raised.
#
# The layouts that actually occur:
#
#   PBCBipoleRHFResult   mo_energies   : List[np.ndarray], one per k
#                        mo_energies_k : ABSENT
#   GPW / GAPW multi-k   mo_energies_k : tuple of per-k arrays
#   Gamma-only drivers   mo_energies   : a single 1-D array
#
# so neither "has mo_energies_k" nor "mo_energies is an array" is a sound
# test on its own.  Dispatch on the *shape*, here, once.


@dataclass(frozen=True)
class FrontierBands:
    """Valence-band maximum, conduction-band minimum, and gap.

    ``homo`` / ``lumo`` are Hartree; ``gap`` is ``lumo - homo`` and may be
    negative for a metal, which is a real result and is not clamped.
    ``homo_k`` / ``lumo_k`` index the sampled k-set, so a caller can say
    *where* each extremum was found (direct vs indirect).
    """

    homo: float
    lumo: float
    gap: float
    homo_k: int
    lumo_k: int
    n_kpoints: int
    n_bands: int


def per_k_band_energies(result) -> List[np.ndarray]:
    """Return this periodic result's eigenvalues as one array per k-point.

    Accepts any converged periodic SCF result -- BIPOLE, GDF, GPW, GAPW,
    Gamma-only or multi-k -- and always returns a list of 1-D float arrays
    of equal length, ordered as the driver sampled the Brillouin zone.  A
    Gamma-only result yields a one-element list.

    Parameters
    ----------
    result
        A periodic SCF result carrying ``mo_energies_k`` or
        ``mo_energies``.

    Raises
    ------
    TypeError
        If *result* exposes neither attribute, or exposes something that
        is not eigenvalue data.
    ValueError
        If the per-k blocks do not all carry the same number of bands.
        A ragged set cannot be compared across k and silently misaligns
        every band index, so it is refused rather than padded.
    """
    raw = getattr(result, "mo_energies_k", None)
    if raw is None or (isinstance(raw, (list, tuple)) and not raw):
        raw = getattr(result, "mo_energies", None)
    if raw is None:
        raise TypeError(
            f"{type(result).__name__} exposes neither 'mo_energies_k' nor "
            "'mo_energies'; it is not a periodic SCF result with bands"
        )

    if isinstance(raw, (list, tuple)):
        blocks = [np.asarray(block, dtype=float) for block in raw]
    else:
        array = np.asarray(raw, dtype=float)
        if array.ndim == 1:
            blocks = [array]
        elif array.ndim == 2:
            blocks = [np.asarray(row, dtype=float) for row in array]
        else:
            raise TypeError(
                "periodic band energies must be 1-D (Gamma) or 2-D "
                f"(n_k, n_bands); got shape {array.shape!r}"
            )

    if not blocks:
        raise TypeError(
            f"{type(result).__name__} carries an empty band set"
        )
    flat: List[np.ndarray] = []
    for index, block in enumerate(blocks):
        if block.ndim != 1:
            raise TypeError(
                f"band block {index} must be 1-D per k-point; got shape "
                f"{block.shape!r}. A 2-D block is the #135 signature: a "
                "per-k list wrapped as one pseudo-k."
            )
        flat.append(block)
    sizes = {block.size for block in flat}
    if len(sizes) != 1:
        raise ValueError(
            "per-k band blocks carry different band counts "
            f"({sorted(sizes)}); a ragged set misaligns every band index "
            "and cannot be compared across the Brillouin zone"
        )
    return flat


def frontier_bands(result, n_occ: int) -> FrontierBands:
    """Valence/conduction extrema across the whole sampled k-set.

    The VBM is the largest occupied eigenvalue over every k-point and the
    CBM the smallest virtual one, so an indirect gap is found correctly.
    Degenerate manifolds are handled by construction: the reduction is a
    ``max`` / ``min`` over scalars, never an index into a manifold, which
    is the #15 failure.

    Parameters
    ----------
    result
        A periodic SCF result -- see :func:`per_k_band_energies`.
    n_occ
        Number of occupied bands per k-point (doubly occupied bands for a
        closed shell, i.e. ``n_electrons // 2``).

    Raises
    ------
    ValueError
        If ``n_occ`` is not strictly inside the band range.  The bound is
        checked against the number of BANDS, not the number of k-points --
        the #135 guard counted k-points and therefore passed on a set that
        had no virtual band to report at all.
    """
    blocks = per_k_band_energies(result)
    n_bands = int(blocks[0].size)
    if n_occ < 1:
        raise ValueError(f"n_occ must be >= 1; got {n_occ}")
    if n_occ >= n_bands:
        raise ValueError(
            f"n_occ={n_occ} leaves no virtual band: the result carries "
            f"{n_bands} bands at each of {len(blocks)} k-points. (Check "
            "you are not counting k-points instead of bands -- that is "
            "the issue #135 guard.)"
        )

    homo = -np.inf
    lumo = np.inf
    homo_k = lumo_k = 0
    for index, block in enumerate(blocks):
        ordered = np.sort(block)
        top = float(ordered[n_occ - 1])
        bottom = float(ordered[n_occ])
        if top > homo:
            homo, homo_k = top, index
        if bottom < lumo:
            lumo, lumo_k = bottom, index
    return FrontierBands(
        homo=homo,
        lumo=lumo,
        gap=lumo - homo,
        homo_k=homo_k,
        lumo_k=lumo_k,
        n_kpoints=len(blocks),
        n_bands=n_bands,
    )
