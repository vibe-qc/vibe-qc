"""User-facing k-point builder.

Phase K1 introduces a single :class:`KPoints` Python class that fronts
all five v0.5 k-point input modes:

1. **Monkhorst-Pack mesh**:  ``KPoints.monkhorst_pack(sys, [4,4,4])``
2. **Γ-centered mesh**:       ``KPoints.gamma_centred(sys, [4,4,4])``
3. **Custom-shifted mesh**:  ``KPoints.shifted(sys, mesh, shift)``
4. **Γ-only**:               ``KPoints.gamma(sys)``
5. *(K2)* Symmetry-reduced:  ``KPoints.monkhorst_pack(..., symmetry=True)``
                             or ``.symmetry_reduce()``
6. *(K3)* Band path:         ``KPoints.band_path(sys)``
                             (auto-detects Bravais via seekpath / HPKOT)
7. *(K4)* Explicit list:     ``KPoints.from_list(sys, k_frac, weights)``
8. *(K6)* Generalized regular / remote tables:
                             ``KPoints.optimal(sys, target_n_kpts)``,
                             ``KPoints.from_database(sys, lattice, order)``
9. *(K8)* AUTO recommender:  ``KPoints.recommend(sys)`` -- classifies
                             metal/insulator character, picks a target
                             k-spacing Δk, and returns a ready-to-use,
                             symmetry-reduced mesh bundled with a
                             recommended smearing and a human-readable
                             rationale. Optional ``verify=True``
                             convergence ladder and ``predictor=...`` ML
                             hook.

A ``KPoints`` object holds:

  - ``kpoints_cart``  (N, 3) bohr⁻¹     -- Cartesian
  - ``kpoints_frac``  (N, 3)            -- fractional in reciprocal basis
  - ``weights``       (N,)              -- sum to 1
  - ``mesh``          (3,) ints, or None for non-MP modes
  - ``shift``         (3,) ints  ``in {0, 1}`` for MP modes
  - ``ir_mapping``    (∏mesh,) ints, empty when no symmetry reduction
  - ``kind``          str -- ``"monkhorst-pack"``, ``"gamma"``,
                            ``"band-path"``, ``"explicit"``,
                            ``"generalized-regular"``, ``"database"``
  - ``labels``        list[(distance, label)] for plotting,
                      band-path mode only

It is **back-compat** with the legacy ``BlochKMesh``: every periodic SCF
entry point continues to accept either a raw ``BlochKMesh`` or a
``KPoints`` (the latter is converted at the boundary).

For ``PeriodicSystem.dim < 3``, mesh builders accept either active-axis
specs (``[n]`` for 1D, ``[n1, n2]`` for 2D) or full three-axis specs.
Inactive axes are pinned to Γ and stored as mesh size 1 / shift 0.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BlochKMesh,
    PeriodicSystem,
)
from ._vibeqc_core import (
    bloch_kmesh_from_lists as _bm_from_lists,
)
from ._vibeqc_core import (
    monkhorst_pack as _mp_native,
)

__all__ = [
    "KPoints",
    "KPointConvergence",
    "as_bloch_kmesh",
]


# ======================================================================
# AUTO k-point recommender -- tunable configuration block (Phase K8)
# ======================================================================
# Every threshold the AUTO heuristic (``KPoints.recommend``) uses lives
# here so it is easy to find and tune in one place.
#
# Δk values follow the **Materials Project KSPACING convention**
# (numerically 2pi/Å) -- identical units to
# ``KPoints.from_kspacing(units="angstrom")``, which ``recommend`` reuses
# for the actual Δk -> mesh step. The metal-vs-insulator k-density gap is
# the single biggest factor in mesh requirements: metals need roughly an
# order of magnitude more k-points than wide-gap insulators (Choudhary &
# Tavazza, npj Comput. Mater. 6, 39 (2020)).

# Target k-spacings per electronic character (2pi/Å).
DELTA_K_INSULATOR = 0.30  # large-gap insulators        (range 0.25-0.35)
DELTA_K_SMALL_GAP = 0.18  # small-gap semiconductors     (range 0.15-0.20)
DELTA_K_METAL = 0.12  # metals / Fermi-surface       (range 0.10-0.15)

# Band-gap cutoffs (eV) separating the three characters:
#   gap > GAP_INSULATOR_EV                         -> insulator
#   GAP_METAL_EPS_EV < gap <= GAP_INSULATOR_EV      -> small-gap semiconductor
#   gap <= GAP_METAL_EPS_EV                         -> metal (numerically gapless)
GAP_INSULATOR_EV = 0.5
GAP_METAL_EPS_EV = 1.0e-3

# Hexagonal/trigonal lattices take a Γ-centred grid -- a (1/2,1/2,1/2) MP
# offset breaks the three-fold rotational symmetry (cf. the same guard
# in ``KPoints.monkhorst_pack``). Detected via spacegroup number when
# available, else geometrically (a 60°/120° angle between cell vectors).
HEX_TRIGONAL_SG_RANGE = (143, 194)
HEX_ANGLE_TOL_DEG = 2.0

# Large-cell collapse: 3D cells whose volume exceeds this fold toward a
# Γ-only mesh (the BZ is already tiny). Lower-dimensional systems rely on
# the natural Δk collapse instead (a vacuum axis would inflate a raw
# det-based volume). Tunable.
GAMMA_ONLY_CELL_VOLUME_BOHR3 = 12000.0

# Convergence-ladder (``verify=True``) settings.
TOLERANCE_MEV_PER_ATOM_DEFAULT = 1.0
LADDER_SPACING_FACTOR = float(np.sqrt(2.0))  # AUTO/√2, AUTO, AUTO.√2 ...
VERIFY_MAX_REFINE_STEPS = 4


@dataclass(frozen=True)
class KPointConvergence:
    """Result of the AUTO convergence ladder (``recommend(verify=True)``).

    ``rungs`` records every spacing actually evaluated as
    ``(delta_k, mesh, energy_per_atom_hartree)`` from coarsest to finest.
    ``chosen_delta_k`` / ``chosen_mesh`` is the coarsest (cheapest) rung
    whose total energy/atom agreed with the next-finer rung to within
    ``tolerance_meV_per_atom``; ``converged`` is ``False`` when the ladder
    hit ``VERIFY_MAX_REFINE_STEPS`` without meeting the tolerance (the
    finest rung tried is then reported).
    """

    converged: bool
    tolerance_meV_per_atom: float
    rungs: Tuple[Tuple[float, Tuple[int, int, int], float], ...]
    chosen_delta_k: float
    chosen_mesh: Tuple[int, int, int]


@dataclass
class KPoints:
    """Container + builder for k-point lists used by periodic SCF and
    post-SCF tools.

    Construct via the classmethods below -- direct instantiation is
    intended for internal use only.
    """

    kpoints_cart: np.ndarray  # (N, 3), bohr⁻¹
    kpoints_frac: np.ndarray  # (N, 3), fractional
    weights: np.ndarray  # (N,), sum to 1
    kind: str = "explicit"
    mesh: Optional[Tuple[int, int, int]] = None
    shift: Optional[Tuple[int, int, int]] = None
    ir_mapping: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    labels: List[Tuple[float, str]] = field(default_factory=list)
    _system: Optional[PeriodicSystem] = field(default=None, repr=False)
    grid_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    # Populated only by the AUTO recommender (:meth:`recommend`); left as
    # defaults by every other constructor so they stay back-compatible.
    smearing: Optional["SmearingOptions"] = field(default=None, repr=False)
    rationale: str = ""
    verification: Optional["KPointConvergence"] = field(default=None, repr=False)
    # Recommended BZ-integration backend (AUTO only): None / "smearing"
    # (temperature broadening) or "gilat" (the parameter-free
    # Gilat-Raubenheimer net -- pass to a driver/run_periodic_job route
    # that accepts the ``bz_integration=`` argument.
    bz_integration: Optional[str] = None
    # Route keys into [routes.numerics] naming the published k-point
    # construction this mesh came from, so run_periodic_job can cite it
    # (CLAUDE.md § 8). A construction with no defining paper -- plain
    # Monkhorst-Pack, an explicit list, Γ -- leaves this empty; the
    # Monkhorst-Pack paper itself rides the kpoint_database route.
    citation_numerics: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Construction -- Monkhorst-Pack family
    # ------------------------------------------------------------------

    @classmethod
    def monkhorst_pack(
        cls,
        system: PeriodicSystem,
        mesh: Sequence[int],
        *,
        shift: Optional[Sequence[int]] = None,
        symmetry: bool = False,
    ) -> "KPoints":
        """Monkhorst-Pack k-mesh on the reciprocal lattice of ``system``.

        Parameters
        ----------
        system:
            Periodic system; reciprocal lattice taken from
            ``system.reciprocal_lattice()``.
        mesh:
            ``(nx, ny, nz)`` -- divisions along each reciprocal axis.
            Non-periodic axes contribute one k-point at 0 regardless
            of ``mesh[i]``.
        shift:
            Per-axis shift flags ``(s_x, s_y, s_z) in {0, 1}^3`` -- 1
            shifts the grid by half a step along that axis. Default
            is the **classical Monkhorst-Pack** convention: shift = 1
            on each axis where mesh is even, 0 where odd. To force
            Γ-centring (no offset on any axis), use
            :meth:`gamma_centred`.

            **Hex / trigonal cells**: when ``system.symmetry`` is
            populated and the spacegroup falls in the hexagonal or
            trigonal range (143-194), a non-zero shift is *refused*
            with an actionable error. The classical (1/2, 1/2, 1/2)
            offset breaks the three-fold rotational symmetry and
            inflates the IBZ count for no accuracy benefit. Use
            :meth:`gamma_centred` instead.
        symmetry:
            If ``True``, reduce the mesh to the irreducible Brillouin
            zone using spglib. Requires ``vq.attach_symmetry(system)``
            to have been called first.

        Notes
        -----
        Classical MP convention (``shift=None``) recovers the standard
        ``[1, 1, 1]`` shift for an all-even mesh and ``[0, 0, 0]`` for
        an all-odd mesh.

        **GDF-driver caveat**: the multi-k GDF drivers' tuple form
        (``run_krhf_periodic_gdf(kmesh=(n1, n2, n3))``) samples the
        **Γ-centered** mesh (PySCF ``make_kpts`` convention — what every
        PySCF-parity pin uses), so for an even mesh it is a *different
        sampling* from this method's classical-MP default (half-step
        shift, no Γ point). Measured on MgO primitive FCC / STO-3G at
        (2,2,2): ~44 mHa apart, with vibe-qc and PySCF agreeing on the
        size of the shift on both conventions. To reproduce the tuple /
        PySCF sampling through this class, use :meth:`gamma_centred`
        (or ``shift=(0, 0, 0)``).
        """
        from .bands import require_periodic_system

        require_periodic_system(system, feature="KPoints.monkhorst_pack")
        mesh_t = _mesh_tuple_for_system(system, mesh)
        if shift is None:
            dim = _system_dim(system)
            shift_t = tuple(
                1 if (i < dim and mesh_t[i] % 2 == 0) else 0 for i in range(3)
            )
        else:
            shift_t = _shift_tuple_for_system(system, shift)

        # Hex/trigonal Γ-centring guard: only fires when symmetry is
        # known. Without symmetry attached we silently let the user
        # do whatever (no information to act on).
        if any(s != 0 for s in shift_t):
            sg = getattr(system, "symmetry", None)
            sg_no = getattr(sg, "number", None) if sg is not None else None
            if sg_no is not None and 143 <= int(sg_no) <= 194:
                raise ValueError(
                    f"monkhorst_pack: spacegroup {sg_no} "
                    f"({getattr(sg, 'international_symbol', '?')}) is "
                    f"hexagonal/trigonal -- a non-zero shift "
                    f"{shift_t!r} breaks the three-fold rotational "
                    f"symmetry. Use KPoints.gamma_centred(system, mesh) "
                    f"instead, or pass shift=(0, 0, 0) explicitly."
                )

        bm = _mp_native(system, list(mesh_t), list(shift_t), bool(symmetry))
        return cls._from_bloch_kmesh(
            system,
            bm,
            kind="monkhorst-pack",
            mesh=mesh_t,
            shift=shift_t,
        )

    @classmethod
    def gamma_centred(
        cls,
        system: PeriodicSystem,
        mesh: Sequence[int],
        *,
        symmetry: bool = False,
    ) -> "KPoints":
        """Γ-centered mesh -- no shift on any axis.

        Identical to ``monkhorst_pack(..., shift=(0, 0, 0))``. The
        explicit constructor reads better in user code where the
        intent is "include the Γ point exactly".
        """
        return cls.monkhorst_pack(
            system,
            mesh,
            shift=(0, 0, 0),
            symmetry=symmetry,
        )

    @classmethod
    def shifted(
        cls,
        system: PeriodicSystem,
        mesh: Sequence[int],
        shift: Sequence[int],
        *,
        symmetry: bool = False,
    ) -> "KPoints":
        """Custom-shifted Monkhorst-Pack mesh."""
        return cls.monkhorst_pack(
            system,
            mesh,
            shift=shift,
            symmetry=symmetry,
        )

    @classmethod
    def gamma(cls, system: PeriodicSystem) -> "KPoints":
        """Single k-point at Γ. Equivalent to a 1x1x1 mesh, no shift."""
        return cls.monkhorst_pack(system, (1, 1, 1), shift=(0, 0, 0))

    # ------------------------------------------------------------------
    # K3 -- Band path via seekpath (HPKOT / Hinuma 2017)
    # ------------------------------------------------------------------

    @classmethod
    def band_path(
        cls,
        system: PeriodicSystem,
        scheme: str = "auto",
        *,
        reference_distance: float = 0.025,
        segments: Optional[Sequence] = None,
        points_per_segment: int = 30,
    ) -> "KPoints":
        """High-symmetry k-path for band-structure plots.

        Parameters
        ----------
        system:
            Periodic system. Bravais type is auto-detected from
            ``system.lattice`` and atomic positions via spglib.
        scheme:
            ``"auto"`` (default) or ``"hpkot"``: use seekpath to return
            the canonical HPKOT path for the detected Bravais lattice
            (Hinuma et al., *Comp. Mat. Sci.* 128, 140 (2017)).

            ``"manual"``: build the path from explicit
            ``segments=[(start_frac, "Γ", end_frac, "X"), ...]`` --
            this routes through the existing
            :func:`vibeqc.bands.kpath_from_segments` machinery so any
            non-standard Bravais or custom path remains expressible.
        reference_distance:
            Target spacing along the path in 2pi/Å (passed straight to
            seekpath). Smaller -> more points, smoother bands.
        segments:
            Required when ``scheme="manual"``. Each entry is
            ``(k_frac_start, label_start, k_frac_end, label_end)``.
        points_per_segment:
            Used with ``scheme="manual"``. Density per segment.

        Returns
        -------
        KPoints
            A ``KPoints`` of ``kind="band-path"`` with:

              - ``kpoints_cart`` / ``kpoints_frac``: the discretised
                path in Cartesian and fractional coordinates;
              - ``weights``: uniform 1/N (band-path has no integration
                weight, but we keep ``Sw = 1`` for type consistency);
              - ``labels``: ``[(linear_distance, label), ...]`` for
                tick marks when plotting band structures.

            Use :meth:`to_kpath` to feed
            :func:`vibeqc.bands.band_structure`.

        Notes
        -----
        seekpath operates on the **standardised primitive cell** -- if
        the input is a conventional cell (e.g. an FCC cell with 4
        atoms instead of 1), seekpath transforms automatically. The
        returned k-points are then in the primitive reciprocal basis,
        which is the right input for ``band_structure`` operating on
        a real-space lattice sum of the primitive cell. For mixed
        primitive<->conventional cell setups, build the cell as
        primitive (or run ``vq.symmetrize`` once it ships in G1).
        """
        from .bands import require_periodic_system

        require_periodic_system(system, feature="KPoints.band_path")
        if scheme.lower() == "manual":
            if segments is None:
                raise ValueError(
                    "band_path: scheme='manual' requires "
                    "segments=[(start_frac, label, end_frac, label), ...]"
                )
            return cls._from_manual_segments(
                system,
                segments,
                points_per_segment=points_per_segment,
            )
        if scheme.lower() not in ("auto", "hpkot"):
            raise ValueError(
                f"band_path: unknown scheme {scheme!r}. Supported: "
                "'auto' / 'hpkot' (seekpath HPKOT auto-detect), "
                "'manual' (user supplies segments=[...])."
            )
        return cls._from_seekpath_hpkot(
            system,
            reference_distance=reference_distance,
        )

    @classmethod
    def _from_seekpath_hpkot(
        cls,
        system: PeriodicSystem,
        *,
        reference_distance: float = 0.025,
    ) -> "KPoints":
        """Build a HPKOT band path via seekpath. Lattice is transposed
        to the rows-as-vectors convention seekpath expects."""
        try:
            import seekpath
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "vibeqc.KPoints.band_path(scheme='auto') requires "
                "seekpath. Install with `pip install seekpath`."
            ) from exc

        # vibe-qc's PeriodicSystem.lattice has *columns* = lattice
        # vectors; seekpath / spglib / ASE expects *rows*. Transpose.
        lat_rows = np.asarray(system.lattice, dtype=np.float64).T
        # Fractional positions: lat_columns @ frac = cartesian, so
        # frac = inv(lat_columns) @ cartesian.
        lat_inv = np.linalg.inv(np.asarray(system.lattice, dtype=np.float64))
        positions_frac = []
        numbers = []
        for atom in system.unit_cell:
            cart = np.asarray(atom.xyz, dtype=np.float64)
            frac = lat_inv @ cart
            positions_frac.append(frac.tolist())
            numbers.append(int(atom.Z))

        seek = seekpath.get_explicit_k_path(
            (lat_rows, positions_frac, numbers),
            reference_distance=reference_distance,
        )
        # k-points in primitive reciprocal basis (fractional).
        frac_arr = np.asarray(seek["explicit_kpoints_rel"], dtype=np.float64)
        # Cartesian via vibe-qc's reciprocal lattice (columns=b_i):
        B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
        cart_arr = (B @ frac_arr.T).T
        n = cart_arr.shape[0]
        weights = np.full(n, 1.0 / n, dtype=np.float64)
        # Tick labels: walk the explicit-kpoints labels and emit one
        # entry per non-empty label, paired with its linear coord.
        linear = np.asarray(seek["explicit_kpoints_linearcoord"], dtype=np.float64)
        labels: List[Tuple[float, str]] = []
        for i, lbl in enumerate(seek["explicit_kpoints_labels"]):
            if lbl:
                pretty = _prettify_label(lbl)
                # Coalesce adjacent same-position labels with "|" the
                # way kpath_from_segments does.
                if (
                    labels
                    and abs(labels[-1][0] - linear[i]) < 1e-12
                    and labels[-1][1] != pretty
                ):
                    labels[-1] = (
                        labels[-1][0],
                        f"{labels[-1][1]}|{pretty}",
                    )
                else:
                    labels.append((float(linear[i]), pretty))
        return cls(
            kpoints_cart=cart_arr,
            kpoints_frac=frac_arr,
            weights=weights,
            kind="band-path",
            mesh=None,
            shift=None,
            ir_mapping=np.zeros(0, dtype=np.int64),
            labels=labels,
            citation_numerics=("hpkot_band_path",),
            _system=system,
        )

    @classmethod
    def _from_manual_segments(
        cls,
        system: PeriodicSystem,
        segments: Sequence,
        *,
        points_per_segment: int = 30,
    ) -> "KPoints":
        """Build a band path from user-supplied (k_frac, label) segments
        -- adapter to the existing :func:`bands.kpath_from_segments`."""
        from .bands import kpath_from_segments

        kpath = kpath_from_segments(
            system,
            segments,
            points_per_segment=points_per_segment,
        )
        n = kpath.kpoints_cart.shape[0]
        weights = np.full(n, 1.0 / n, dtype=np.float64)
        return cls(
            kpoints_cart=np.asarray(kpath.kpoints_cart, dtype=np.float64),
            kpoints_frac=np.asarray(kpath.kpoints_frac, dtype=np.float64),
            weights=weights,
            kind="band-path",
            mesh=None,
            shift=None,
            ir_mapping=np.zeros(0, dtype=np.int64),
            labels=list(kpath.labels),
            _system=system,
        )

    # ------------------------------------------------------------------
    # K5 -- Density-based auto-mesh (KPPRA / kspacing / length)
    # ------------------------------------------------------------------

    @classmethod
    def from_kppra(
        cls,
        system: PeriodicSystem,
        n_kpts_per_atom: float,
        *,
        metallic: bool = False,
        gamma_centred: bool = False,
        symmetry: bool = False,
        warn_no_smearing: bool = True,
    ) -> "KPoints":
        """Density-based auto-mesh using the **KPPRA** convention
        (k-points per reciprocal atom, AFLOW / Curtarolo *et al.*
        2012). Picks the smallest mesh ``[N1, N2, N3]`` such that
        ``N1.N2.N3 >= n_kpts_per_atom / N_atoms_in_cell`` while keeping
        the per-axis subdivision proportional to ``|b_i|``.

        Algorithm:

          1. ``target_total = max(1, ceil(n_kpts_per_atom / N_atoms))``
          2. Per-axis weights ``w_i = |b_i|``; non-periodic axes get
             ``w_i = 0`` (forced ``N_i = 1``).
          3. Pick scale ``s`` so that ``∏ ceil(s . w_i) ≈ target_total``
             (line search over ``s in [1, 1000]``).
          4. ``N_i = max(1, ceil(s . w_i))``.

        Parameters
        ----------
        system:
            Periodic system.
        n_kpts_per_atom:
            Target density. Standard values (Materials Project /
            AFLOW empirical):

              -  100   -> very coarse (testing)
              -  500   -> light insulator
              - 1000   -> defensible insulator
              - 3000   -> semiconductor / band-edge convergence
              - 8000   -> metallic system (use with ``metallic=True``)

        metallic:
            If ``True``, multiplies the effective density by 4 so the
            mesh density is sufficient for Fermi-surface integration.
            Also emits a warning via the ``warnings`` module if the
            user hasn't enabled smearing on their SCF options yet
            (which the ``warn_no_smearing`` flag controls). The
            metallic-vs-insulator distinction is the single biggest
            factor in k-mesh density requirements (see Choudhary &
            Tavazza, *npj Comp. Mat.* 6, 39 (2020): metals need
            ~10x the k-density of insulators).
        gamma_centred:
            If ``True``, force ``shift=(0, 0, 0)``. If ``False`` (the
            default), use the classical Monkhorst-Pack auto-shift
            convention (even meshes shift=1, odd meshes shift=0). For
            hex/trigonal cells the auto-shift will refuse -- use
            ``gamma_centred=True`` there.
        symmetry:
            If ``True``, reduce to the IBZ via spglib (requires
            ``vq.attach_symmetry(system)``).
        warn_no_smearing:
            When ``metallic=True`` and this is also ``True``, emit a
            ``UserWarning`` reminding the caller to enable smearing on
            their SCF options. Default ``True``.

        Examples
        --------
        Si (2 atoms primitive cell), KPPRA = 1000 -> 4x4x4 mesh:

        >>> kp = vq.KPoints.from_kppra(si_primitive, 1000)
        >>> kp.mesh
        (4, 4, 4)

        Tungsten (1 atom BCC cell), metallic, KPPRA = 8000:

        >>> kp = vq.KPoints.from_kppra(w_bcc, 8000, metallic=True)
        >>> kp.mesh        # roughly (20, 20, 20) once metallic x4 boost
        """
        if n_kpts_per_atom <= 0:
            raise ValueError(
                f"from_kppra: n_kpts_per_atom must be positive; got "
                f"{n_kpts_per_atom!r}."
            )
        n_atoms = max(1, len(system.unit_cell))
        density = float(n_kpts_per_atom)
        if metallic:
            density *= 4.0
            if warn_no_smearing:
                import warnings

                warnings.warn(
                    "KPoints.from_kppra(metallic=True): mesh density "
                    "bumped 4x for Fermi-surface integration. Make "
                    "sure your SCF options enable smearing -- "
                    "without it metallic systems will struggle to "
                    "converge regardless of k-mesh density. See "
                    "RHFOptions / RKSOptions / similar for the "
                    "smearing knob.",
                    UserWarning,
                    stacklevel=2,
                )
        target_total = max(1, int(np.ceil(density / n_atoms)))
        mesh = _kpoint_mesh_from_density(
            system,
            target_total,
            dim=int(system.dim),
        )
        shift = (0, 0, 0) if gamma_centred else None
        kp = cls.monkhorst_pack(system, mesh, shift=shift, symmetry=symmetry)
        # The mesh is Monkhorst-Pack; what is cited here is the KPPRA
        # density convention that chose it (AFLOW, Curtarolo 2012).
        kp.citation_numerics = ("kppra",)
        return kp

    @classmethod
    def from_kspacing(
        cls,
        system: PeriodicSystem,
        kspacing: float,
        *,
        units: str = "angstrom",
        gamma_centred: bool = False,
        symmetry: bool = False,
    ) -> "KPoints":
        """Density-based auto-mesh using the **kspacing** convention
        (Materials Project / ASE / VASP ``KSPACING``). Picks
        ``N_i = max(1, ceil(|b_i| / kspacing))`` where ``|b_i|`` is the
        i-th reciprocal-lattice-vector length.

        Parameters
        ----------
        kspacing:
            Target spacing along each reciprocal axis. Default
            interpretation is **2pi/Å** (the VASP / Materials Project
            convention); pass ``units="bohr"`` for **bohr⁻¹** if
            preferred.
        units:
            ``"angstrom"`` (default) -> ``kspacing`` in 2pi/Å.
            ``"bohr"`` -> ``kspacing`` in bohr⁻¹.

        Standard values (Materials Project recommendation, in 2pi/Å):

          - 0.5  -> very coarse
          - 0.3  -> defensible insulator
          - 0.2  -> semiconductor / band-edge convergence
          - 0.1  -> metallic systems
          - 0.04 -> high-precision benchmarks

        Examples
        --------
        >>> kp = vq.KPoints.from_kspacing(si_primitive, 0.3)
        """
        if kspacing <= 0:
            raise ValueError(
                f"from_kspacing: kspacing must be positive; got {kspacing!r}."
            )
        # Convert kspacing -> bohr⁻¹ if needed. Reciprocal lattice
        # `B` is in bohr⁻¹ (vibeqc convention); kspacing in 2pi/Å
        # converts via 2pi/Å x (1 bohr / (2pi x 0.529177 Å)) = 1 / 0.529177
        # ... actually kspacing has units of [length⁻¹] (whether in
        # 2pi/Å or bohr⁻¹), so we just convert the *number* to bohr⁻¹.
        if units.lower() in ("angstrom", "ang", "a"):
            # 1/Å x (1 Å / 1.8897 bohr) -- but kspacing is "2pi/Å" really,
            # so factor is 1/1.8897 to get into "2pi/bohr" then we drop
            # the 2pi since |b_i| as returned by reciprocal_lattice() is
            # already in 2pi/bohr (i.e. bohr⁻¹).
            kspacing_internal = kspacing / ANGSTROM_TO_BOHR
        elif units.lower() in ("bohr", "au"):
            kspacing_internal = kspacing
        else:
            raise ValueError(
                f"from_kspacing: units must be 'angstrom' or 'bohr'; got {units!r}."
            )

        B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
        # Per-axis reciprocal-vector lengths.
        b_lens = np.linalg.norm(B, axis=0)
        mesh = []
        for i in range(3):
            if i >= int(system.dim):
                mesh.append(1)
            elif b_lens[i] < 1e-14:
                mesh.append(1)
            else:
                mesh.append(int(max(1, np.ceil(b_lens[i] / kspacing_internal))))
        shift = (0, 0, 0) if gamma_centred else None
        return cls.monkhorst_pack(system, tuple(mesh), shift=shift, symmetry=symmetry)

    @classmethod
    def auto(
        cls,
        system: PeriodicSystem,
        length: float,
        *,
        gamma_centred: bool = False,
        symmetry: bool = False,
    ) -> "KPoints":
        """VASP-style ``Auto`` mode -- picks ``N_i = max(1, ceil(length
        . |b_i| + 0.5))`` where ``length`` is in Å (matching VASP's
        deprecated ``Auto`` mode and the JARVIS-DFT line-density
        convention from Choudhary & Tavazza 2019).

        Empirical guidance (Choudhary & Tavazza 2019,
        *npj Comp. Mat.* 6, 39 (2020) on > 30 000 materials):

          -  10 Å -> large-gap insulators
          -  25 Å -> defensible insulator default
          -  50 Å -> semiconductors / metals
          - 100 Å -> d-metals / Fermi-surface convergence
          - 200 Å -> high-precision benchmarks

        Examples
        --------
        >>> kp = vq.KPoints.auto(si_primitive, 30)
        >>> # per-axis equivalent to KPoints.from_kspacing(si_primitive,
        >>> # 2*pi/30) -- VASP's Auto length l <-> KSPACING = 2pi/l.
        """
        if length <= 0:
            raise ValueError(f"auto: length must be positive; got {length!r}.")
        # VASP's ``Auto`` length ``l`` (Å) is equivalent to KSPACING = 2pi/l,
        # so ``auto(l)`` must agree per-axis with ``from_kspacing(2pi/l)``.
        # reciprocal_lattice() returns |b_i| in bohr⁻¹ *including* the 2pi.
        # from_kspacing forms N_i = ceil(|b_i| / (kspacing/ANGSTROM_TO_BOHR));
        # substituting kspacing = 2pi/l gives
        #   N_i = ceil(|b_i| . ANGSTROM_TO_BOHR . l / (2pi)).
        # So the per-axis coefficient multiplying ``length`` is
        # |b_i|.ANGSTROM_TO_BOHR/(2pi). The previous code used
        # |b_i|/ANGSTROM_TO_BOHR, i.e. a factor (2pi/ANGSTROM_TO_BOHR^2)≈1.76x
        # too dense per axis (~5.4x too many k-points). (Audit 2026-05-30.)
        B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
        # Crystallographic reciprocal length in cycles/Å (i.e. 1/a, the
        # 2pi divided back out) -- *not* the 2pi/Å "b_lens" of from_kspacing.
        b_lens_cycles_per_A = (
            np.linalg.norm(B, axis=0) * ANGSTROM_TO_BOHR / (2.0 * np.pi)
        )
        mesh = []
        for i in range(3):
            if i >= int(system.dim):
                mesh.append(1)
            else:
                mesh.append(int(max(1, np.ceil(length * b_lens_cycles_per_A[i] + 0.5))))
        shift = (0, 0, 0) if gamma_centred else None
        return cls.monkhorst_pack(system, tuple(mesh), shift=shift, symmetry=symmetry)

    # ------------------------------------------------------------------
    # K8 -- AUTO recommender (the electronic-structure-aware mode)
    # ------------------------------------------------------------------

    @classmethod
    def recommend(
        cls,
        system: PeriodicSystem,
        *,
        is_metal: Optional[bool] = None,
        band_gap: Optional[float] = None,
        tolerance_meV_per_atom: float = TOLERANCE_MEV_PER_ATOM_DEFAULT,
        periodicity: Optional[Union[int, str]] = None,
        delta_k: Optional[float] = None,
        gamma_centred: Optional[bool] = None,
        symmetry: bool = True,
        smearing_method: Optional[str] = None,
        bz_integration: Optional[str] = None,
        classifier: Optional[Callable] = None,
        predictor: Optional[Union[str, Callable]] = None,
        ml_predictor: Optional[Callable] = None,
        verify: bool = False,
        scf_energy_fn: Optional[Callable] = None,
        scf_result=None,
        basis=None,
    ) -> "KPoints":
        """**AUTO mode** -- turn a structure into a ready-to-use k-point spec.

        This is the electronic-structure-aware sibling of the density
        constructors (:meth:`from_kspacing` / :meth:`from_kppra` /
        :meth:`auto`): rather than asking the user for a target spacing,
        ``recommend`` *decides* the spacing Δk from the system's
        metal/insulator character, builds a symmetry-reduced
        Monkhorst-Pack mesh at that spacing, and bundles a recommended
        smearing plus a human-readable rationale.

        Treat the result as a **good starting point, not a guarantee** --
        pass ``verify=True`` (with ``scf_energy_fn``) for a convergence
        ladder that confirms (and, if needed, refines) the mesh.

        Parameters
        ----------
        system:
            Periodic system. Lattice, atoms, and (when attachable)
            spacegroup symmetry are read from it.
        is_metal, band_gap:
            Electronic-character hints. ``band_gap`` is in **eV**.
            Resolution priority (the part AUTO actually decides):

              1. ``band_gap`` given -- ``> 0.5 eV`` -> insulator
                 (Δk≈0.30); ``> 0`` and ``<= 0.5 eV`` -> small-gap
                 semiconductor (Δk≈0.18); ``≈ 0`` -> metal (Δk≈0.12).
              2. else ``is_metal`` given -- ``True`` -> metal,
                 ``False`` -> insulator.
              3. else ``classifier`` hook, if supplied (see below).
              4. else the **SAFE default**: treat as metal (the dense,
                 over-converged choice) and emit a ``UserWarning`` that
                 character was assumed.
        tolerance_meV_per_atom:
            Convergence target for ``verify=True``. Default
            1.0 meV/atom.
        periodicity:
            Optional sanity hint (``3``/``2``/``1`` or
            ``"3d"``/``"2d"``/``"slab"``/``"1d"``). The structure's own
            ``system.dim`` is authoritative; a mismatch only warns.
            Non-periodic (vacuum) axes always get ``N_i = 1``.
        delta_k:
            Optional user override for the target spacing (2pi/Å). When
            given, character classification is still used for the
            smearing recommendation but the spacing is taken verbatim.
        gamma_centred:
            ``None`` (default) lets AUTO choose -- Γ-centred for
            hexagonal/trigonal lattices, classical MP auto-shift
            otherwise. Pass ``True``/``False`` to force.
        symmetry:
            Reduce to the irreducible BZ via spglib (default ``True``).
            Best-effort: if symmetry can't be attached (e.g. some
            low-dimensional cells) AUTO falls back to the full mesh and
            says so in the rationale.
        smearing_method:
            Smearing flavour to recommend for metals/small-gap systems.
            Defaults to ``"methfessel-paxton"`` (cold smearing -- the
            standard metal choice, total energy only weakly dependent
            on the smearing width); ``"fermi-dirac"`` and ``"marzari-
            vanderbilt"`` are also selectable. All three are wired in
            the SCF smearing path (:mod:`vibeqc.smearing`).
        bz_integration:
            How to integrate the BZ for metals. ``None`` (default) /
            ``"smearing"`` keep the temperature-broadening path (AUTO
            recommends a :class:`SmearingOptions`). ``"gilat"`` selects
            the **parameter-free Gilat-Raubenheimer net** (the
            tetrahedron-family, CRYSTAL ``SHRINK IS ISP`` analogue): a
            T=0 integrator with no smearing width. AUTO drops the smearing
            recommendation (``smearing=None``) and records the choice on
            :attr:`bz_integration` for you to pass to a driver or
            :func:`vibeqc.run_periodic_job` route that accepts the
            ``bz_integration=`` argument. The net runs on the
            symmetry-reduced (IBZ) mesh -- the driver expands it to the
            full BZ internally -- so it composes with the default
            ``symmetry=True`` and dense metal meshes stay tractable.
            Wired routes include multi-k Ewald, closed-shell multi-k GDF,
            and BIPOLE RKS through :func:`vibeqc.run_periodic_job`.
        classifier:
            Optional character source.  A callable
            ``classifier(system) -> character`` or ``(character, gap_eV)``
            hook (resolution step 4), or the string ``"pre-scf"`` for
            the built-in cheap pre-SCF classifier (resolution step 3).
            ``"pre-scf"`` runs a few-cycle Gamma-only GDF SCF with the
            user-supplied ``basis``, extracts the HOMO-LUMO gap, and
            classifies from it.  Requires ``basis=``.  Explicitly
            gated -- the default is the SAFE metallic fallback.
        predictor, ml_predictor:
            Optional ML Δk predictor (advanced path). ``predictor`` is a
            callable ``predictor(features) -> (Δk, s)`` or the string
            ``"ml"`` (which requires a companion ``ml_predictor``
            callable -- no model ships by default). The **conservative
            end** of the predicted interval (``Δk - s``, i.e. the denser
            mesh) is used. See Choudhary & Tavazza (npj Comput. Mater.
            6, 39 (2020)) and conformal-quantile-regression k-spacing
            models for the idea.
        verify, scf_energy_fn:
            When ``verify=True``, run a convergence ladder
            (Δk.√2, Δk, Δk/√2, ...) and refine until the total energy/atom
            changes by less than ``tolerance_meV_per_atom`` between
            successive rungs, then return the converged mesh with the
            ladder attached as :attr:`verification`. Requires
            ``scf_energy_fn(kpoints) -> energy_hartree`` -- a callable
            that runs the user's intended method/basis for a given mesh.

        Returns
        -------
        KPoints
            A ``kind="monkhorst-pack"`` mesh (symmetry-reduced when
            possible) with three AUTO-only attributes populated:

              - :attr:`smearing` -- a :class:`vibeqc.SmearingOptions` to
                pass to the SCF driver (``None`` for clean insulators);
              - :attr:`rationale` -- e.g. ``"metal, Δk=0.12 Å⁻¹,
                Γ-centred, 12x12x8, 56 irreducible k-points; smearing
                methfessel-paxton kT=0.005 Ha"``;
              - :attr:`verification` -- the :class:`KPointConvergence`
                ladder when ``verify=True``, else ``None``;
              - :attr:`bz_integration` -- the recommended BZ-integration
                backend (``None`` / ``"gilat"``).

        Examples
        --------
        >>> spec = vq.KPoints.recommend(si_diamond, band_gap=1.1)
        >>> spec.mesh, spec.grid_type, spec.smearing
        ((7, 7, 7), 'gamma', None)
        >>> print(spec.rationale)
        insulator, Δk=0.3 Å⁻¹, Γ-centred, 7x7x7, 20 irreducible k-points

        >>> spec = vq.KPoints.recommend(tungsten_bcc, is_metal=True)
        >>> run_rks_periodic_scf(..., kpoints=spec, smearing=spec.smearing)

        >>> # Tetrahedron-family (parameter-free) integration for a metal
        >>> # -- no smearing; runs on the (efficient) IBZ mesh:
        >>> spec = vq.KPoints.recommend(tungsten_bcc, is_metal=True,
        ...                             bz_integration="gilat")
        >>> spec.smearing, spec.bz_integration, spec.is_symmetry_reduced
        (None, 'gilat', True)
        """
        import warnings

        from .smearing import (
            EV_PER_HARTREE,
            SMEARING_PRESETS,
            SmearingOptions,
        )

        _system_dim(system)  # validates dim in {1,2,3}
        if periodicity is not None:
            _check_periodicity_hint(system, periodicity, warnings)
        if tolerance_meV_per_atom <= 0:
            raise ValueError(
                "recommend: tolerance_meV_per_atom must be positive; "
                f"got {tolerance_meV_per_atom!r}."
            )
        if bz_integration is not None:
            bz_integration = str(bz_integration).strip().lower()
            if bz_integration not in ("smearing", "gilat"):
                raise ValueError(
                    "recommend: bz_integration must be None, 'smearing', "
                    f"or 'gilat'; got {bz_integration!r}."
                )
        use_gilat = bz_integration == "gilat"

        notes: List[str] = []

        # 1) Electronic character -- the part AUTO actually decides.
        character, gap_ev, needs_fallback = _classify_kpoint_character(
            is_metal,
            band_gap,
        )
        if needs_fallback:
            if scf_result is not None:
                character, gap_ev = _classify_from_scf_result(
                    scf_result,
                    system,
                )
                if character is not None:
                    notes.append(
                        f"character from prior SCF result -> {character}"
                        + (f" (gap {gap_ev:.3g} eV)" if gap_ev is not None else "")
                    )
                else:
                    pass
            if character is None and classifier is not None:
                if (
                    isinstance(classifier, str)
                    and classifier.strip().lower() == "pre-scf"
                ):
                    character, gap_ev = _classify_pre_scf(
                        system,
                        basis,
                        warnings,
                    )
                    if character is not None:
                        notes.append(
                            f"character from pre-SCF classifier -> {character}"
                            + (f" (gap {gap_ev:.3g} eV)" if gap_ev is not None else "")
                        )
                elif callable(classifier):
                    character, gap_ev = _apply_classifier_hook(classifier, system)
                    notes.append(f"character from classifier hook -> {character}")
                else:
                    raise TypeError(
                        "recommend: classifier must be a callable or the "
                        f"string 'pre-scf'; got {classifier!r}."
                    )
            if character is None:
                # SAFE default: a metal is the dense, over-converged
                # choice -- never under-sample an unknown system.
                character, gap_ev = "metal", 0.0
                warnings.warn(
                    "KPoints.recommend: no band_gap / is_metal hint and no "
                    "classifier supplied -- assuming METALLIC (the safe, "
                    "dense default) and recommending smearing. Pass "
                    "band_gap=<eV> or is_metal=<bool> to refine.",
                    UserWarning,
                    stacklevel=2,
                )
                notes.append("character assumed metallic (no hint supplied)")

        # 2) Target spacing Δk (2pi/Å).
        _uses_bundled = False
        if delta_k is not None:
            dk = float(delta_k)
            if dk <= 0:
                raise ValueError(
                    f"recommend: delta_k must be positive; got {delta_k!r}."
                )
            notes.append("Δk from user override")
        elif predictor is not None:
            dk, _uses_bundled = _delta_k_from_predictor(
                predictor,
                ml_predictor,
                system,
                character,
                gap_ev,
            )
            notes.append("Δk from ML predictor (conservative end)")
        else:
            dk = _delta_k_for_character(character)

        # 3) Grid type -- Γ-centred for hex/trigonal.
        sg_number = _spacegroup_number(system)
        hex_like = _is_hexagonal_like(system, sg_number)
        if gamma_centred is None:
            use_gamma = bool(hex_like)
            if hex_like:
                notes.append("Γ-centred (hexagonal/trigonal lattice)")
        else:
            use_gamma = bool(gamma_centred)

        # 4) Symmetry reduction -- best-effort. The Gilat-Raubenheimer net
        #    accepts a symmetry-reduced (IBZ) mesh: the multi-k driver
        #    expands it to the full BZ internally (eps_n(Rk)=eps_n(k), so
        #    occupations are constant over a symmetry star), which is what
        #    keeps dense metal meshes tractable. So gilat respects the
        #    symmetry setting like every other backend -- no full-mesh
        #    override.
        do_sym = bool(symmetry)
        if do_sym and hex_like and not use_gamma:
            # A non-Γ shift on a hex cell trips monkhorst_pack's guard
            # once symmetry is attached. Honour the user's explicit MP
            # request by skipping reduction rather than crashing.
            do_sym = False
            notes.append("MP shift on hexagonal cell -- symmetry reduction skipped")
            warnings.warn(
                "KPoints.recommend: gamma_centred=False on a "
                "hexagonal/trigonal lattice breaks the three-fold "
                "symmetry; Γ-centring is strongly recommended here.",
                UserWarning,
                stacklevel=2,
            )
        if do_sym:
            do_sym = _ensure_symmetry_attached(system, notes)

        # 5) Large-cell Γ-only collapse (3D only -- a vacuum axis would
        #    inflate a det-based volume on slabs/wires).
        force_gamma_only = False
        if int(system.dim) == 3:
            vol = _system_cell_volume_bohr3(system)
            if vol > GAMMA_ONLY_CELL_VOLUME_BOHR3:
                force_gamma_only = True
                notes.append(
                    f"Γ-only collapse (cell volume {vol:.0f} bohr^3 > "
                    f"{GAMMA_ONLY_CELL_VOLUME_BOHR3:.0f})"
                )

        # 6) Build the mesh.
        if force_gamma_only:
            kp = cls.monkhorst_pack(
                system,
                (1, 1, 1),
                shift=(0, 0, 0),
                symmetry=do_sym,
            )
        else:
            kp = cls.from_kspacing(
                system,
                dk,
                units="angstrom",
                gamma_centred=use_gamma,
                symmetry=do_sym,
            )
            if _mesh_is_gamma_only(kp.mesh, int(system.dim)):
                notes.append("Γ-only (small BZ -- Δk yields a 1x1x1 mesh)")

        # 7) Smearing recommendation -- driven by the *same* character that
        #    set Δk, so the two never disagree (an eV-vs-Hartree threshold
        #    mismatch would otherwise smear a clean insulator). Presets are
        #    the shared vibe-qc smearing presets (metal 0.005, small-gap
        #    0.002, insulator 0.0 Ha).
        if character == "metal":
            smear_temp = SMEARING_PRESETS["metal"]
            smear_reason = "metallic character (recommend AUTO)"
        elif character == "small-gap semiconductor":
            smear_temp = SMEARING_PRESETS["small-gap"]
            smear_reason = "small-gap semiconductor (recommend AUTO)"
        else:  # insulator
            smear_temp = 0.0
            smear_reason = "insulating character -- no smearing needed"
        smearing: Optional[SmearingOptions] = None
        if use_gilat:
            # Parameter-free T=0 integrator -- mutually exclusive with
            # finite-T smearing (the driver rejects combining them).
            if smear_temp > 0.0:
                notes.append("no smearing: Gilat-Raubenheimer is parameter-free (T=0)")
        elif smear_temp > 0.0:
            smearing = SmearingOptions(
                temperature=float(smear_temp),
                flavor=(smearing_method or "methfessel-paxton"),
                source="auto",
                reason=smear_reason,
            )

        kp.smearing = smearing
        kp.bz_integration = bz_integration
        kp.uses_ml_predictor = _uses_bundled
        kp.rationale = _format_kpoint_rationale(
            character,
            dk,
            kp,
            smearing,
            notes,
        )

        # 8) Optional convergence ladder.
        if verify:
            if scf_energy_fn is None:
                raise ValueError(
                    "recommend(verify=True) needs scf_energy_fn -- a "
                    "callable (kpoints) -> energy_hartree that runs your "
                    "method/basis for a trial mesh. Example:\n"
                    "    recommend(sys, band_gap=1.1, verify=True,\n"
                    "              scf_energy_fn=lambda kp: "
                    "run_rks_periodic_scf(sys, opts, kpoints=kp).energy)"
                )
            n_atoms = max(1, len(system.unit_cell))
            conv = _converge_kmesh(
                cls,
                system,
                dk,
                use_gamma=use_gamma,
                do_sym=do_sym,
                tol_meV=float(tolerance_meV_per_atom),
                energy_fn=scf_energy_fn,
                n_atoms=n_atoms,
                ev_per_hartree=EV_PER_HARTREE,
            )
            kp = cls.from_kspacing(
                system,
                conv.chosen_delta_k,
                units="angstrom",
                gamma_centred=use_gamma,
                symmetry=do_sym,
            )
            kp.smearing = smearing
            kp.bz_integration = bz_integration
            kp.uses_ml_predictor = _uses_bundled
            kp.verification = conv
            verify_note = (
                f"verified to {tolerance_meV_per_atom:g} meV/atom"
                if conv.converged
                else f"NOT converged within {VERIFY_MAX_REFINE_STEPS} refinements"
            )
            kp.rationale = _format_kpoint_rationale(
                character,
                conv.chosen_delta_k,
                kp,
                smearing,
                notes + [verify_note],
            )

        return kp

    # ------------------------------------------------------------------
    # K6 -- Generalized regular grids
    # ------------------------------------------------------------------

    @classmethod
    def generalized_regular(
        cls,
        system: PeriodicSystem,
        grid_matrix: Sequence[Sequence[int]],
    ) -> "KPoints":
        """Build a generalized regular k-point grid from an integer HNF.

        ``grid_matrix`` is a 3x3 upper-triangular Hermite normal form
        (HNF) matrix with positive diagonal. A diagonal HNF is the
        familiar Gamma-centred Monkhorst-Pack mesh; off-diagonal HNF
        entries generate sheared regular grids with the same number of
        points, ``det(grid_matrix)``.
        """
        _require_gr_3d(system)
        H = _validate_hnf_grid_matrix(grid_matrix)
        frac = _fractional_points_from_hnf(H)
        B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
        cart = (B @ frac.T).T
        weights = np.full(frac.shape[0], 1.0 / frac.shape[0], dtype=np.float64)
        return cls(
            kpoints_cart=cart,
            kpoints_frac=frac,
            weights=weights,
            kind="generalized-regular",
            mesh=None,
            shift=None,
            ir_mapping=np.zeros(0, dtype=np.int64),
            labels=[],
            citation_numerics=("generalized_regular_kgrid",),
            _system=system,
            grid_matrix=H,
        )

    @classmethod
    def optimal(
        cls,
        system: PeriodicSystem,
        target_n_kpts: int,
        *,
        metallic: bool = False,
    ) -> "KPoints":
        """Pick an on-the-fly generalized regular grid.

        Candidate grids are all 3D upper-triangular HNF matrices with
        determinant ``target_n_kpts`` (or ``4 * target_n_kpts`` for
        ``metallic=True``). The selected grid maximizes the shortest
        periodic Cartesian distance between k-points, giving the most
        uniform grid in the bounded HNF search.
        """
        _require_gr_3d(system)
        target = int(target_n_kpts)
        if target <= 0:
            raise ValueError(
                "KPoints.optimal: target_n_kpts must be positive; "
                f"got {target_n_kpts!r}"
            )
        if metallic:
            target *= 4

        best_H: np.ndarray | None = None
        best_score = -np.inf
        for H in _hnf_grid_matrices(target):
            score = _minimum_periodic_kpoint_distance(system, H)
            if score > best_score + 1.0e-14:
                best_score = score
                best_H = H
        if best_H is None:
            raise RuntimeError(
                f"KPoints.optimal: no HNF grids for determinant {target}"
            )
        return cls.generalized_regular(system, best_H)

    @classmethod
    def from_database(
        cls,
        system: PeriodicSystem,
        lattice: str,
        order: int | Sequence[int],
        *,
        special: bool = False,
        timeout: float = 10.0,
        base_url: str = "http://esd.cos.gmu.edu/tb/kpts/",
    ) -> "KPoints":
        """Fetch a weighted irreducible k-point table from the GMU/NRL
        pre-defined k-point database.

        The remote tables are in lattice-coordinate format: fractional
        reciprocal coordinates followed by normalized symmetry weights.
        No table data is bundled with vibe-qc; this constructor performs
        an explicit network lookup and converts the returned table into
        a weighted :class:`KPoints` object.

        Parameters
        ----------
        system:
            3D periodic system whose reciprocal lattice supplies the
            Cartesian conversion.
        lattice:
            Database lattice family. Supported values are ``"sc"``,
            ``"bcc"``, ``"fcc"``, and ``"hex"`` plus obvious long-form
            aliases such as ``"simple cubic"``.
        order:
            Cubic families use a single integer order, e.g. ``4`` for
            ``regular.04``. Hexagonal tables use a pair, e.g. ``(6, 3)``
            for ``regular.6.3``.
        special:
            Fetch the offset "special" table instead of the regular
            table. The GMU/NRL database provides special tables for the
            cubic families only; hexagonal tables are regular-only.
        timeout:
            Network timeout in seconds.
        base_url:
            Override for mirrors / tests. Defaults to the public GMU
            database over plain HTTP.
        """
        if _system_dim(system) != 3:
            raise ValueError(
                "KPoints.from_database currently requires a 3D PeriodicSystem"
            )
        lattice_code = _kpoint_database_lattice_code(lattice)
        order_key = _kpoint_database_order_key(
            lattice_code,
            order,
            special=special,
        )
        family = "special" if special else "regular"
        url = _kpoint_database_url(base_url, lattice_code, family, order_key)
        text = _download_kpoint_database_table(url, timeout=timeout)
        frac, weights = _parse_kpoint_database_table(text, source=url)
        B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
        cart = (B @ frac.T).T
        return cls(
            kpoints_cart=cart,
            kpoints_frac=frac,
            weights=weights,
            kind="database",
            mesh=None,
            shift=None,
            ir_mapping=np.zeros(0, dtype=np.int64),
            labels=[],
            citation_numerics=("kpoint_database",),
            _system=system,
        )

    # ------------------------------------------------------------------
    # K4 -- Explicit user-supplied k-list with optional weights
    # ------------------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        system: PeriodicSystem,
        k_frac: Sequence,
        weights: Optional[Sequence[float]] = None,
        *,
        normalise: bool = True,
    ) -> "KPoints":
        """Build a :class:`KPoints` from a user-supplied list of
        fractional k-vectors and (optional) weights.

        Parameters
        ----------
        system:
            Periodic system; reciprocal lattice taken from
            ``system.reciprocal_lattice()``.
        k_frac:
            ``(N, dim)`` / ``(N, 3)`` array (or a single length-``dim`` /
            length-3 sequence) of **fractional** k-coordinates in the
            reciprocal-lattice basis. For 1D and 2D systems, coordinates
            supplied only on the active axes are padded with zeros on the
            inactive axes. Cartesian conversion is automatic via
            ``B @ k_frac.T``.
        weights:
            Optional ``(N,)`` array of integration weights. When
            ``None``, uniform weights ``1/N`` are assigned. Negative
            weights raise ``ValueError`` (the IBZ unfolding logic
            assumes non-negative weights). Zero weights are allowed
            (e.g. band-path-like points sprinkled into an integration
            mesh -- they contribute to eigenvalue *evaluation* but not
            to BZ-integral sums).
        normalize:
            If ``True`` (default) and ``weights`` is given, the
            weights are scaled so they sum to 1.0 -- matching the
            convention used by :meth:`monkhorst_pack` and the rest of
            the periodic-SCF stack. Set to ``False`` to preserve the
            user's weights verbatim (useful when feeding raw
            multiplicities from another code's symmetry analysis).

        Returns
        -------
        KPoints
            ``kind="explicit"`` mesh; ``mesh`` and ``shift`` left
            as ``None``; ``ir_mapping`` empty.

        Examples
        --------
        Three k-points with uniform 1/3 weights:

        >>> kp = vq.KPoints.from_list(
        ...     sys,
        ...     k_frac=[[0.0, 0.0, 0.0],
        ...             [0.5, 0.0, 0.0],
        ...             [0.5, 0.5, 0.0]],
        ... )
        >>> len(kp), kp.weights.sum()
        (3, 1.0)

        Custom weights matching multiplicities of the IBZ orbits of an
        FCC mesh, normalized to sum to 1:

        >>> kp = vq.KPoints.from_list(
        ...     sys,
        ...     k_frac=[[0.0, 0.0, 0.0],
        ...             [0.5, 0.5, 0.5],
        ...             [0.5, 0.0, 0.0]],
        ...     weights=[1, 8, 6],
        ... )
        """
        k_frac_arr = _k_frac_array_for_system(system, k_frac)
        n = k_frac_arr.shape[0]
        if n == 0:
            raise ValueError("from_list: k_frac must contain at least 1 point.")

        if weights is None:
            weights_arr = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
            if weights_arr.shape != (n,):
                raise ValueError(
                    f"from_list: weights shape {weights_arr.shape} doesn't "
                    f"match k_frac length {n}."
                )
            if (weights_arr < 0).any():
                raise ValueError(
                    f"from_list: negative weight at index "
                    f"{int(np.argmin(weights_arr))} "
                    f"(value {float(weights_arr.min())}); the periodic-SCF "
                    "stack assumes non-negative weights."
                )
            total = float(weights_arr.sum())
            if normalise:
                if total <= 0.0:
                    raise ValueError(
                        "from_list: cannot normalize weights -- total is "
                        f"{total}. Either supply at least one positive "
                        "weight or pass normalize=False."
                    )
                weights_arr = weights_arr / total

        # Cartesian via vibe-qc reciprocal lattice (columns = b_i):
        # k_cart = B @ k_frac for each row.
        B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
        cart_arr = (B @ k_frac_arr.T).T

        return cls(
            kpoints_cart=cart_arr,
            kpoints_frac=k_frac_arr,
            weights=weights_arr,
            kind="explicit",
            mesh=None,
            shift=None,
            ir_mapping=np.zeros(0, dtype=np.int64),
            labels=[],
            _system=system,
        )

    # ------------------------------------------------------------------
    # K3 -- KPath conversion (back-compat with vibeqc.bands.band_structure)
    # ------------------------------------------------------------------

    def to_kpath(self):
        """Return a :class:`vibeqc.bands.KPath` view of this band path,
        for handing to :func:`vibeqc.bands.band_structure`. Only valid
        when ``self.kind == "band-path"``.

        The returned KPath uses cumulative arc length |Δk_cart| as the
        x-axis, matching the existing band-structure plotting
        conventions in vibeqc.bands."""
        if self.kind != "band-path":
            raise ValueError(
                f"to_kpath: only valid for kind='band-path' "
                f"(got kind={self.kind!r}). Use band_path(...) to "
                "construct."
            )
        from .bands import KPath

        cart = np.asarray(self.kpoints_cart, dtype=np.float64)
        # Cumulative arc length along the path.
        diffs = np.diff(cart, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        distances = np.concatenate([[0.0], np.cumsum(seg_lens)])
        return KPath(
            kpoints_cart=cart,
            kpoints_frac=np.asarray(self.kpoints_frac, dtype=np.float64),
            distances=distances,
            labels=list(self.labels),
            citation_numerics=tuple(self.citation_numerics),
        )

    # ------------------------------------------------------------------
    # Builder methods -- return a new KPoints with the requested change
    # ------------------------------------------------------------------

    def symmetry_reduce(self) -> "KPoints":
        """Reduce the mesh to the irreducible Brillouin zone via spglib.

        Requires the underlying ``PeriodicSystem`` to have
        ``vq.attach_symmetry(system)`` called on it first; raises
        otherwise.

        Idempotent -- calling on an already-reduced mesh returns ``self``.
        Currently only supports Monkhorst-Pack-derived meshes (the
        common case); raises ``NotImplementedError`` for explicit /
        band-path / arbitrary user-supplied lists, where the right
        action is "construct it symmetry-reduced from the start".
        """
        if self.is_symmetry_reduced:
            return self
        if self.kind != "monkhorst-pack" or self.mesh is None or self.shift is None:
            raise NotImplementedError(
                f"symmetry_reduce: only supported for Monkhorst-Pack "
                f"meshes (got kind={self.kind!r}). For explicit / "
                f"band-path lists, construct symmetry-aware from the "
                f"start."
            )
        return type(self).monkhorst_pack(
            self._require_system(),
            self.mesh,
            shift=self.shift,
            symmetry=True,
        )

    # ------------------------------------------------------------------
    # Internal -- convert a native ``BlochKMesh`` to a ``KPoints``
    # ------------------------------------------------------------------

    @classmethod
    def _from_bloch_kmesh(
        cls,
        system: PeriodicSystem,
        bm: BlochKMesh,
        *,
        kind: str,
        mesh: Optional[Tuple[int, int, int]] = None,
        shift: Optional[Tuple[int, int, int]] = None,
    ) -> "KPoints":
        cart = np.asarray(bm.kpoints, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(bm.weights, dtype=np.float64).reshape(-1)
        ir_map = np.asarray(bm.ir_mapping, dtype=np.int64).reshape(-1)
        # Recover fractional from Cartesian: k_cart = B . k_frac, with B
        # the reciprocal-lattice matrix whose *columns* are the b_i
        # vectors (vibe-qc convention). So k_frac = B⁻¹ . k_cart.
        B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
        # Pseudo-inverse handles low-dim systems where B has zero
        # columns along the vacuum direction(s).
        frac = np.linalg.pinv(B) @ cart.T
        frac = frac.T
        return cls(
            kpoints_cart=cart,
            kpoints_frac=frac,
            weights=weights,
            kind=kind,
            mesh=mesh,
            shift=shift,
            ir_mapping=ir_map,
            _system=system,
        )

    # ------------------------------------------------------------------
    # Down-conversion -- feed a periodic SCF driver
    # ------------------------------------------------------------------

    def to_bloch_kmesh(self) -> BlochKMesh:
        """Materialise as the legacy native :class:`BlochKMesh` so it
        can be handed to ``run_rhf_periodic_*`` / ``run_rks_periodic_*``
        without churning their signatures.

        For ``kind="monkhorst-pack"`` we re-construct via ``_mp_native``
        so the C++ ``mesh``/``is_shift``/``ir_mapping`` metadata is
        populated correctly. For ``kind="explicit"`` and
        ``kind="band-path"`` we synthesise a ``BlochKMesh`` directly
        from ``self.kpoints_cart`` + ``self.weights`` (with empty
        ``mesh``/``is_shift``/``ir_mapping`` since those are MP-specific
        concepts). This keeps the path open for any periodic-SCF
        driver to consume any ``KPoints`` flavor."""
        if self.kind == "monkhorst-pack" and self.mesh and self.shift:
            symmetry = self.ir_mapping.size > 0
            return _mp_native(
                self._require_system(),
                list(self.mesh),
                list(self.shift),
                bool(symmetry),
            )
        if self.kind in (
            "explicit",
            "band-path",
            "generalized-regular",
            "database",
        ):
            cart_list = [
                np.asarray(k, dtype=np.float64).reshape(3) for k in self.kpoints_cart
            ]
            weights_list = list(np.asarray(self.weights, dtype=np.float64).tolist())
            return _bm_from_lists(cart_list, weights_list)
        raise NotImplementedError(f"to_bloch_kmesh: kind={self.kind!r} not supported.")

    def _require_system(self) -> PeriodicSystem:
        if self._system is None:
            raise RuntimeError(
                "KPoints: missing PeriodicSystem reference (was the "
                "object constructed via a classmethod?)."
            )
        return self._system

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.kpoints_cart.shape[0])

    @property
    def n_kpoints(self) -> int:
        return len(self)

    @property
    def grid_type(self) -> str:
        """``"gamma"`` for a Γ-centred mesh (no shift), else the
        underlying ``kind`` (``"monkhorst-pack"`` for shifted MP meshes).
        Convenience view for the AUTO rationale + user reporting."""
        if self.shift is not None and all(int(s) == 0 for s in self.shift):
            return "gamma"
        return self.kind

    @property
    def full_mesh_size(self) -> int:
        """Number of k-points in the *full* (unreduced) mesh -- ``∏ mesh``
        for MP-family meshes, else the stored point count."""
        if self.mesh is None:
            return len(self)
        return int(np.prod(self.mesh))

    @property
    def is_symmetry_reduced(self) -> bool:
        return self.ir_mapping.size > 0

    def __repr__(self) -> str:
        bits = [f"kind={self.kind!r}", f"n={len(self)}"]
        if self.mesh is not None:
            bits.append(f"mesh={self.mesh}")
        if self.shift is not None:
            bits.append(f"shift={self.shift}")
        if self.is_symmetry_reduced:
            bits.append("symmetry_reduced=True")
        return f"KPoints({', '.join(bits)})"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _system_dim(system: PeriodicSystem) -> int:
    dim = int(system.dim)
    if dim not in (1, 2, 3):
        raise ValueError(f"PeriodicSystem.dim must be 1, 2, or 3; got {dim}")
    return dim


def _mesh_tuple(mesh: Sequence[int]) -> Tuple[int, int, int]:
    arr = list(mesh)
    if len(arr) != 3:
        raise ValueError(f"mesh must have length 3, got {arr!r}")
    out = tuple(int(m) for m in arr)
    if any(m < 1 for m in out):
        raise ValueError(f"mesh entries must be >= 1, got {arr!r}")
    return out  # type: ignore[return-value]


def _mesh_tuple_for_system(
    system: PeriodicSystem,
    mesh: Sequence[int],
) -> Tuple[int, int, int]:
    dim = _system_dim(system)
    arr = list(mesh)
    if len(arr) == dim:
        arr = arr + [1] * (3 - dim)
    elif len(arr) != 3:
        raise ValueError(
            f"mesh must have length {dim} for dim={dim} systems or "
            f"length 3; got {arr!r}"
        )
    out = tuple(int(m) for m in arr)
    if any(m < 1 for m in out):
        raise ValueError(f"mesh entries must be >= 1, got {arr!r}")
    return tuple(out[i] if i < dim else 1 for i in range(3))


# Length conversions used by K5 density-based auto-mesh constructors.
# 1 bohr = 0.529177210903 Å, so 1 Å = 1.8897261339213 bohr.
ANGSTROM_TO_BOHR = 1.8897261339213


def _kpoint_mesh_from_density(
    system: PeriodicSystem,
    target_total: int,
    *,
    dim: int = 3,
) -> Tuple[int, int, int]:
    """Pick ``N1, N2, N3`` minimizing over-shoot of ``target_total``
    while keeping the per-axis count proportional to ``|b_i|``.

    For dim < 3, the inactive axes are pinned to 1.
    """
    B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
    weights = np.linalg.norm(B, axis=0)
    # Inactive axes get zero weight.
    for i in range(3):
        if i >= dim:
            weights[i] = 0.0

    if target_total <= 1:
        return tuple(1 if w == 0 else 1 for w in weights)  # type: ignore

    active_weights = weights[weights > 0]
    if active_weights.size == 0:
        # Fully inactive system -- return all-1 mesh.
        return (1, 1, 1)

    # Line search over scale s. For each candidate s,
    # mesh = max(1, ceil(s * w_i)); product is monotone non-decreasing
    # in s (binary search).
    lo, hi = 0.0, 100.0  # plenty for 1000s x 1000s x 1000s in extremis
    # First widen `hi` if needed.
    while True:
        mesh = tuple(
            1
            if (i >= dim or weights[i] == 0)
            else int(max(1, np.ceil(hi * weights[i])))
            for i in range(3)
        )
        if int(np.prod(mesh)) >= target_total:
            break
        hi *= 2.0
        if hi > 1e6:
            break

    # Binary search.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mesh = tuple(
            1
            if (i >= dim or weights[i] == 0)
            else int(max(1, np.ceil(mid * weights[i])))
            for i in range(3)
        )
        if int(np.prod(mesh)) >= target_total:
            hi = mid
        else:
            lo = mid

    # Final mesh at hi.
    return tuple(
        1 if (i >= dim or weights[i] == 0) else int(max(1, np.ceil(hi * weights[i])))
        for i in range(3)
    )  # type: ignore


def _require_gr_3d(system: PeriodicSystem) -> None:
    if _system_dim(system) != 3:
        raise ValueError(
            "generalized regular k-point grids currently require a 3D PeriodicSystem"
        )


def _validate_hnf_grid_matrix(grid_matrix: Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(grid_matrix)
    if arr.shape != (3, 3):
        raise ValueError(
            f"grid_matrix must be a 3x3 integer HNF matrix; got shape {arr.shape}"
        )
    rounded = np.rint(arr.astype(float))
    if not np.allclose(arr.astype(float), rounded, atol=1.0e-12, rtol=0.0):
        raise ValueError("grid_matrix entries must be integers")
    H = rounded.astype(int)
    if H[1, 0] != 0 or H[2, 0] != 0 or H[2, 1] != 0:
        raise ValueError("grid_matrix must be upper-triangular HNF")
    diag = np.diag(H)
    if np.any(diag <= 0):
        raise ValueError("grid_matrix HNF diagonal entries must be positive")
    if not (0 <= H[0, 1] < H[1, 1]):
        raise ValueError("grid_matrix HNF entry h01 must satisfy 0 <= h01 < h11")
    if not (0 <= H[0, 2] < H[2, 2]):
        raise ValueError("grid_matrix HNF entry h02 must satisfy 0 <= h02 < h22")
    if not (0 <= H[1, 2] < H[2, 2]):
        raise ValueError("grid_matrix HNF entry h12 must satisfy 0 <= h12 < h22")
    return H


def _hnf_grid_matrices(determinant: int):
    """Yield all 3D upper-triangular HNF matrices with given determinant."""
    if determinant <= 0:
        return
    for h00 in _divisors(determinant):
        rem1 = determinant // h00
        for h11 in _divisors(rem1):
            h22 = rem1 // h11
            for h01, h02, h12 in product(range(h11), range(h22), range(h22)):
                yield np.array(
                    [
                        [h00, h01, h02],
                        [0, h11, h12],
                        [0, 0, h22],
                    ],
                    dtype=int,
                )


def _divisors(n: int) -> list[int]:
    out: list[int] = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            out.append(i)
            if i * i != n:
                out.append(n // i)
    return sorted(out)


def _fractional_points_from_hnf(H: np.ndarray) -> np.ndarray:
    Hf = np.asarray(H, dtype=float)
    diag = np.diag(H).astype(int)
    mesh = np.meshgrid(*(np.arange(int(n), dtype=float) for n in diag), indexing="ij")
    indices = np.stack(mesh, axis=-1).reshape(-1, 3)
    return (np.linalg.solve(Hf, indices.T).T % 1.0).astype(np.float64)


def _minimum_periodic_kpoint_distance(
    system: PeriodicSystem,
    H: np.ndarray,
) -> float:
    frac = _fractional_points_from_hnf(H)
    wrapped = frac - np.round(frac)
    norms_frac = np.linalg.norm(wrapped, axis=1)
    nonzero = wrapped[norms_frac > 1.0e-12]
    if nonzero.size == 0:
        return np.inf
    B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
    cart = (B @ nonzero.T).T
    return float(np.linalg.norm(cart, axis=1).min())


_KPOINT_DATABASE_LATTICE_ALIASES = {
    "sc": "sc",
    "simplecubic": "sc",
    "simple_cubic": "sc",
    "simple-cubic": "sc",
    "cubic": "sc",
    "bcc": "bcc",
    "bodycenteredcubic": "bcc",
    "body_centered_cubic": "bcc",
    "body-centered-cubic": "bcc",
    "bodycentredcubic": "bcc",
    "body_centred_cubic": "bcc",
    "body-centred-cubic": "bcc",
    "fcc": "fcc",
    "facecenteredcubic": "fcc",
    "face_centered_cubic": "fcc",
    "face-centered-cubic": "fcc",
    "facecentredcubic": "fcc",
    "face_centred_cubic": "fcc",
    "face-centred-cubic": "fcc",
    "hex": "hex",
    "hexagonal": "hex",
}


def _kpoint_database_lattice_code(lattice: str) -> str:
    key = str(lattice).strip().lower().replace(" ", "_")
    compact = key.replace("_", "").replace("-", "")
    code = _KPOINT_DATABASE_LATTICE_ALIASES.get(
        key
    ) or _KPOINT_DATABASE_LATTICE_ALIASES.get(compact)
    if code is None:
        allowed = "sc, bcc, fcc, hex"
        raise ValueError(
            "KPoints.from_database: unsupported lattice family "
            f"{lattice!r}; supported database families are {allowed}."
        )
    return code


def _kpoint_database_order_key(
    lattice_code: str,
    order: int | Sequence[int],
    *,
    special: bool,
) -> str:
    if lattice_code == "hex":
        if special:
            raise ValueError(
                "KPoints.from_database: the GMU/NRL hexagonal database "
                "provides regular tables only; pass special=False."
            )
        if isinstance(order, (str, bytes)):
            parts = [p for p in str(order).replace(",", ".").split(".") if p]
        else:
            try:
                parts = list(order)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError(
                    "KPoints.from_database: hexagonal tables require "
                    "order=(in_plane, c_axis), e.g. order=(6, 3)."
                ) from exc
        if len(parts) != 2:
            raise ValueError(
                "KPoints.from_database: hexagonal tables require exactly "
                f"two order entries; got {order!r}."
            )
        values = tuple(int(p) for p in parts)
        if values[0] <= 0 or values[1] <= 0:
            raise ValueError(
                f"KPoints.from_database: order entries must be positive; got {order!r}."
            )
        return f"{values[0]}.{values[1]}"

    if isinstance(order, SequenceABC) and not isinstance(order, (str, bytes)):
        values = list(order)
        if len(values) != 1:
            raise ValueError(
                "KPoints.from_database: cubic database families require "
                f"a single integer order; got {order!r}."
            )
        order_i = int(values[0])
    else:
        order_i = int(order)  # type: ignore[arg-type]
    if order_i <= 0:
        raise ValueError(
            f"KPoints.from_database: order must be positive; got {order!r}."
        )
    return f"{order_i:02d}"


def _kpoint_database_url(
    base_url: str,
    lattice_code: str,
    family: str,
    order_key: str,
) -> str:
    from urllib.parse import urljoin

    root = str(base_url)
    if not root.endswith("/"):
        root += "/"
    return urljoin(root, f"{lattice_code}/{family}.{order_key}")


def _download_kpoint_database_table(url: str, *, timeout: float) -> str:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "vibeqc-kpoints/0.13"})
    try:
        with urlopen(req, timeout=float(timeout)) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:  # pragma: no cover - network failure path
        raise ValueError(
            f"KPoints.from_database: database table not found at {url!r} "
            f"(HTTP {exc.code})."
        ) from exc
    except URLError as exc:  # pragma: no cover - network failure path
        raise OSError(
            f"KPoints.from_database: failed to fetch {url!r}: {exc.reason}"
        ) from exc


def _parse_kpoint_database_table(
    text: str,
    *,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    expected: int | None = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if expected is None:
            try:
                expected = int(fields[0])
            except ValueError as exc:
                raise ValueError(
                    "KPoints.from_database: first non-empty line of "
                    f"{source!r} must be the k-point count; got "
                    f"{stripped!r}."
                ) from exc
            if expected <= 0:
                raise ValueError(
                    "KPoints.from_database: table k-point count must be "
                    f"positive; got {expected}."
                )
            continue
        if len(fields) < 4:
            raise ValueError(
                "KPoints.from_database: malformed table row "
                f"{line_no} in {source!r}: expected x y z weight."
            )
        try:
            row = tuple(float(fields[i]) for i in range(4))
        except ValueError as exc:
            raise ValueError(
                "KPoints.from_database: malformed numeric value on row "
                f"{line_no} in {source!r}."
            ) from exc
        rows.append(row)  # type: ignore[arg-type]

    if expected is None:
        raise ValueError(f"KPoints.from_database: empty database table {source!r}.")
    if len(rows) != expected:
        raise ValueError(
            "KPoints.from_database: table count mismatch for "
            f"{source!r}: header says {expected}, parsed {len(rows)}."
        )
    arr = np.asarray(rows, dtype=np.float64)
    weights = arr[:, 3].copy()
    if np.any(weights < 0.0):
        raise ValueError(
            "KPoints.from_database: database table contains a negative weight."
        )
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            "KPoints.from_database: database table weights do not sum "
            "to a positive value."
        )
    weights /= total
    return arr[:, :3].copy(), weights


_LABEL_REPLACEMENTS = {
    "GAMMA": "Γ",
    "DELTA": "Δ",
    "SIGMA": "S",
    "LAMBDA": "Λ",
}


def _prettify_label(raw: str) -> str:
    """Convert seekpath's ASCII label conventions to Unicode-friendly
    forms ("GAMMA" -> "Γ"). Subscripts ("X_1" -> "X₁") are left alone
    because matplotlib renders LaTeX-like ``$X_1$`` prefixes anyway,
    and converting to Unicode ``₁/₂/...`` here would lock plotters
    out of LaTeX rendering."""
    return _LABEL_REPLACEMENTS.get(raw, raw)


def _shift_tuple(shift: Sequence[int]) -> Tuple[int, int, int]:
    arr = list(shift)
    if len(arr) != 3:
        raise ValueError(f"shift must have length 3, got {arr!r}")
    out = tuple(int(s) for s in arr)
    if any(s not in (0, 1) for s in out):
        raise ValueError(f"shift entries must be 0 or 1, got {arr!r}")
    return out  # type: ignore[return-value]


def _shift_tuple_for_system(
    system: PeriodicSystem,
    shift: Sequence[int],
) -> Tuple[int, int, int]:
    dim = _system_dim(system)
    arr = list(shift)
    if len(arr) == dim:
        arr = arr + [0] * (3 - dim)
    elif len(arr) != 3:
        raise ValueError(
            f"shift must have length {dim} for dim={dim} systems or "
            f"length 3; got {arr!r}"
        )
    out = tuple(int(s) for s in arr)
    if any(s not in (0, 1) for s in out):
        raise ValueError(f"shift entries must be 0 or 1, got {arr!r}")
    return tuple(out[i] if i < dim else 0 for i in range(3))


def _k_frac_array_for_system(system: PeriodicSystem, k_frac: Sequence):
    dim = _system_dim(system)
    arr = np.asarray(k_frac, dtype=np.float64)
    if arr.ndim == 0:
        if dim != 1:
            raise ValueError(
                f"from_list: scalar k_frac is only valid for dim=1; got dim={dim}."
            )
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if dim == 1 and arr.shape != (1,):
            arr = arr.reshape(-1, 1)
        elif arr.shape == (3,):
            arr = arr.reshape(1, 3)
        elif arr.shape == (dim,):
            arr = arr.reshape(1, dim)
        else:
            raise ValueError(
                f"from_list: k_frac must have shape (N, {dim}) or "
                f"(N, 3); got {arr.shape}."
            )
    elif arr.ndim == 2:
        if arr.shape[1] not in (dim, 3):
            raise ValueError(
                f"from_list: k_frac must have shape (N, {dim}) or "
                f"(N, 3); got {arr.shape}."
            )
    else:
        raise ValueError(
            f"from_list: k_frac must have shape (N, {dim}) or (N, 3); got {arr.shape}."
        )
    if arr.ndim != 2:
        raise ValueError(f"from_list: invalid k_frac shape {arr.shape}.")
    if arr.shape[1] == dim and dim < 3:
        padded = np.zeros((arr.shape[0], 3), dtype=np.float64)
        padded[:, :dim] = arr
        arr = padded
    if arr.shape[1] != 3:
        raise ValueError(f"from_list: k_frac must have shape (N, 3); got {arr.shape}.")
    if dim < 3 and not np.allclose(arr[:, dim:], 0.0, atol=1e-14, rtol=0.0):
        raise ValueError(
            "from_list: inactive k-vector components must be zero for "
            f"dim={dim} systems. Supply only the active coordinates or "
            "set the inactive entries to 0."
        )
    if dim < 3:
        arr = arr.copy()
        arr[:, dim:] = 0.0
    return arr


# ----------------------------------------------------------------------
# AUTO recommender (K8) helpers
# ----------------------------------------------------------------------


def _classify_kpoint_character(
    is_metal: Optional[bool],
    band_gap: Optional[float],
) -> Tuple[Optional[str], Optional[float], bool]:
    """Resolve electronic character from the user's hints.

    Returns ``(character, band_gap_eV, needs_fallback)``. ``character``
    is one of ``"insulator"`` / ``"small-gap semiconductor"`` /
    ``"metal"`` (or ``None`` when ``needs_fallback`` is ``True`` and the
    caller must apply its own policy). Priority: explicit band gap, then
    the ``is_metal`` boolean.
    """
    if band_gap is not None:
        gap = float(band_gap)
        if gap < 0.0:
            raise ValueError(f"recommend: band_gap must be >= 0 eV; got {band_gap!r}.")
        if gap <= GAP_METAL_EPS_EV:
            return "metal", 0.0, False
        if gap <= GAP_INSULATOR_EV:
            return "small-gap semiconductor", gap, False
        return "insulator", gap, False

    if is_metal is True:
        return "metal", None, False
    if is_metal is False:
        return "insulator", None, False

    return None, None, True


def _apply_classifier_hook(
    classifier,
    system: PeriodicSystem,
) -> Tuple[str, Optional[float]]:
    """Normalise a user ``classifier(system)`` result to
    ``(character, band_gap_eV)``."""
    out = classifier(system)
    if isinstance(out, str):
        return _normalise_character(out), None
    try:
        character, gap = out
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "recommend: classifier hook must return a character string or "
            f"a (character, band_gap_eV) pair; got {out!r}."
        ) from exc
    gap_ev = None if gap is None else float(gap)
    return _normalise_character(character), gap_ev


def _classify_from_scf_result(
    result,
    system: PeriodicSystem,
) -> Tuple[Optional[str], Optional[float]]:
    """Classify electronic character from a converged periodic SCF result."""
    n_elec = _system_n_electrons(system)
    if n_elec is None or n_elec <= 0:
        return None, None
    try:
        from .periodic_convergence_auto import converged_gap_hartree
        from .smearing import EV_PER_HARTREE
    except ImportError:
        return None, None
    n_alpha = int(round(n_elec)) // 2
    gap_ha = converged_gap_hartree(result, n_alpha=n_alpha)
    if gap_ha is None:
        return None, None
    gap_ev = float(gap_ha) * EV_PER_HARTREE
    if gap_ev <= GAP_METAL_EPS_EV:
        return "metal", 0.0
    if gap_ev <= GAP_INSULATOR_EV:
        return "small-gap semiconductor", gap_ev
    return "insulator", gap_ev


def _classify_pre_scf(
    system: PeriodicSystem,
    basis,
    warnings_mod,
) -> Tuple[Optional[str], Optional[float]]:
    """Cheap Γ-only pre-SCF classifier (Hcore diagonalization).

    Lazy-imports the periodic SCF machinery.  Runs a single
    Γ-point Hcore diagonalization (no SCF iterations) with the
    user-supplied basis, then classifies the HOMO-LUMO gap.  Returns
    ``(None, None)`` on any failure (basis missing, import error,
    etc.) so the caller can fall through to the SAFE default.
    """
    if basis is None:
        warnings_mod.warn(
            "KPoints.recommend(classifier='pre-scf') requires a basis= "
            "argument (e.g. basis=BasisSet(system.unit_cell_molecule(), "
            "'sto-3g')).  Falling back to SAFE metallic default.",
            UserWarning,
            stacklevel=3,
        )
        return None, None
    try:
        from ._vibeqc_core import (
            BlochKMesh,
            PeriodicRHFOptions,
            PeriodicSCFOptions,
            monkhorst_pack,
        )
        from .periodic_convergence_auto import converged_gap_hartree
        from .periodic_rhf_gdf import run_rhf_periodic_gamma_gdf
        from .smearing import EV_PER_HARTREE
    except ImportError:
        return None, None
    try:
        n_elec = _system_n_electrons(system)
        if n_elec is None or n_elec <= 0:
            return None, None
        opts = PeriodicRHFOptions()
        opts.max_iter = 3  # cheap: only a few SCF cycles
        opts.conv_tol_energy = 1e-4  # loose convergence
        opts.use_diis = False
        result = run_rhf_periodic_gamma_gdf(
            system,
            basis,
            opts,
            progress=False,
        )
        n_alpha = int(round(n_elec)) // 2
        gap_ha = converged_gap_hartree(result, n_alpha=n_alpha)
        if gap_ha is None:
            return None, None
        gap_ev = float(gap_ha) * EV_PER_HARTREE
        if gap_ev <= GAP_METAL_EPS_EV:
            return "metal", 0.0
        if gap_ev <= GAP_INSULATOR_EV:
            return "small-gap semiconductor", gap_ev
        return "insulator", gap_ev
    except Exception:
        return None, None


def _normalise_character(character: str) -> str:
    token = str(character).strip().lower()
    if token in ("metal", "metallic"):
        return "metal"
    if token in ("insulator", "insulating"):
        return "insulator"
    if token in (
        "small-gap",
        "small-gap semiconductor",
        "semiconductor",
        "small gap",
        "small_gap",
    ):
        return "small-gap semiconductor"
    raise ValueError(
        f"recommend: unknown character {character!r}; expected one of "
        "'metal', 'insulator', 'small-gap semiconductor'."
    )


def _delta_k_for_character(character: Optional[str]) -> float:
    """Map character -> target k-spacing (2pi/Å)."""
    if character == "insulator":
        return DELTA_K_INSULATOR
    if character == "small-gap semiconductor":
        return DELTA_K_SMALL_GAP
    # metal, or any unknown -> the dense, safe choice.
    return DELTA_K_METAL


def _delta_k_from_predictor(
    predictor,
    ml_predictor,
    system: PeriodicSystem,
    character: Optional[str],
    gap_ev: Optional[float],
) -> Tuple[float, bool]:
    """Resolve Δk from an ML predictor, picking the conservative end of
    the predicted interval (Δk - s -> the denser/safer mesh).

    Returns (delta_k, uses_bundled).
    """
    uses_bundled = False
    if callable(predictor):
        fn = predictor
    elif isinstance(predictor, str) and predictor.strip().lower() == "ml":
        if callable(ml_predictor):
            fn = ml_predictor
        else:
            from .data_library.ml_kpredictor import (
                build_ml_predictor as _build_ml_predictor,
            )
            from .data_library.ml_kpredictor import (
                is_ml_predictor_enabled,
            )

            if not is_ml_predictor_enabled():
                raise NotImplementedError(
                    "recommend(predictor='ml'): the bundled ML k-spacing "
                    "model is gated by VIBEQC_ML_KPOINTS=1.  Set that "
                    "environment variable to enable it, or pass your own "
                    "model via ml_predictor=<callable> returning "
                    "(delta_k, uncertainty).  See Choudhary & Tavazza, npj "
                    "Comput. Mater. 6, 39 (2020)."
                )
            fn = _build_ml_predictor()
            uses_bundled = True
    else:
        raise ValueError(
            f"recommend: unknown predictor {predictor!r}; pass a callable "
            "predictor(features) -> (delta_k, sigma), or 'ml' together "
            "with ml_predictor=<callable>."
        )

    features = _kpoint_ml_features(system, character, gap_ev)
    mean, sigma = _parse_predictor_output(fn(features))
    dk = mean - abs(sigma)
    if dk <= 0.0:
        dk = mean
    if dk <= 0.0:
        raise ValueError(f"recommend: ML predictor returned a non-positive Δk ({dk}).")
    return float(dk), uses_bundled


def _kpoint_ml_features(
    system: PeriodicSystem,
    character: Optional[str],
    gap_ev: Optional[float],
) -> dict:
    """Lightweight descriptor bundle handed to an ML Δk predictor."""
    B = np.asarray(system.reciprocal_lattice(), dtype=np.float64)
    return {
        "n_atoms": len(system.unit_cell),
        "dim": int(system.dim),
        "cell_volume_bohr3": _system_cell_volume_bohr3(system),
        "reciprocal_lengths_bohr": np.linalg.norm(B, axis=0).tolist(),
        "character": character,
        "band_gap_eV": gap_ev,
        "n_electrons": _system_n_electrons(system),
    }


def _parse_predictor_output(out) -> Tuple[float, float]:
    """Accept ``(mean, sigma)``, ``{"delta_k":, "uncertainty":}``, or a
    bare scalar (s = 0) from an ML predictor."""
    if isinstance(out, dict):
        mean = out.get("delta_k", out.get("mean"))
        if mean is None:
            raise ValueError("recommend: ML predictor dict must carry 'delta_k'.")
        sigma = out.get("uncertainty", out.get("sigma", 0.0))
        return float(mean), float(sigma)
    if isinstance(out, (int, float)):
        return float(out), 0.0
    try:
        mean, sigma = out
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "recommend: ML predictor must return (delta_k, sigma), a "
            f"dict, or a scalar; got {out!r}."
        ) from exc
    return float(mean), float(sigma)


def _spacegroup_number(system: PeriodicSystem) -> Optional[int]:
    """Spacegroup number (1-230) without forcing a mutation -- reads an
    already-attached ``system.symmetry`` else introspects via spglib
    (3D only). Returns ``None`` when undeterminable."""
    sg = getattr(system, "symmetry", None)
    number = getattr(sg, "number", None) if sg is not None else None
    if number is not None:
        return int(number)
    if int(system.dim) == 3:
        try:
            from .periodic_symmetrize import detect_spacegroup

            return int(detect_spacegroup(system).number)
        except Exception:
            return None
    return None


def _is_hexagonal_like(
    system: PeriodicSystem,
    sg_number: Optional[int],
) -> bool:
    """Hexagonal/trigonal family -- spacegroup number when known, else a
    geometric 60°/120° cell-angle test (the fallback for low-dimensional
    cells or when spglib can't classify).

    When the spacegroup number *is* known it is authoritative: the
    geometric test alone gives false positives, because FCC/BCC
    *primitive* cells also carry 60°/120° vector angles despite being
    cubic (e.g. Si diamond, SG 227)."""
    lo, hi = HEX_TRIGONAL_SG_RANGE
    if sg_number is not None:
        return lo <= sg_number <= hi
    return _geometric_hexagonal(system)


def _geometric_hexagonal(system: PeriodicSystem) -> bool:
    dim = int(system.dim)
    if dim < 2:
        return False
    A = np.asarray(system.lattice, dtype=np.float64)  # columns = a_i
    vecs = [A[:, i] for i in range(dim)]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            ni = float(np.linalg.norm(vecs[i]))
            nj = float(np.linalg.norm(vecs[j]))
            if ni < 1e-12 or nj < 1e-12:
                continue
            cos = float(np.dot(vecs[i], vecs[j]) / (ni * nj))
            cos = max(-1.0, min(1.0, cos))
            ang = float(np.degrees(np.arccos(cos)))
            if (
                abs(ang - 120.0) <= HEX_ANGLE_TOL_DEG
                or abs(ang - 60.0) <= HEX_ANGLE_TOL_DEG
            ):
                return True
    return False


def _ensure_symmetry_attached(
    system: PeriodicSystem,
    notes: List[str],
) -> bool:
    """Attach spglib symmetry if absent so the IBZ reduction can fire.
    Returns whether reduction should proceed; on failure records a note
    and returns ``False`` (AUTO falls back to the full mesh)."""
    if getattr(system, "symmetry", None) is not None:
        return True
    try:
        from ._vibeqc_core import attach_symmetry

        attach_symmetry(system)
    except Exception:
        notes.append("symmetry unavailable -- full (unreduced) mesh")
        return False
    if getattr(system, "symmetry", None) is None:
        notes.append("symmetry unavailable -- full (unreduced) mesh")
        return False
    return True


def _system_cell_volume_bohr3(system: PeriodicSystem) -> float:
    A = np.asarray(system.lattice, dtype=np.float64)
    return float(abs(np.linalg.det(A)))


def _system_n_electrons(system: PeriodicSystem) -> Optional[float]:
    try:
        z_total = sum(int(atom.Z) for atom in system.unit_cell)
    except Exception:
        return None
    charge = int(getattr(system, "charge", 0) or 0)
    return float(z_total - charge)


def _mesh_is_gamma_only(
    mesh: Optional[Tuple[int, int, int]],
    dim: int,
) -> bool:
    if mesh is None:
        return False
    return all(int(mesh[i]) == 1 for i in range(min(dim, 3)))


def _format_kpoint_rationale(
    character: Optional[str],
    delta_k: float,
    kp: "KPoints",
    smearing,
    notes: List[str],
) -> str:
    """Build the human-readable one-liner, e.g. ``"metal, Δk=0.12 Å⁻¹,
    Γ-centred, 12x12x8, 56 irreducible k-points"``."""
    if kp.mesh is not None:
        n1, n2, n3 = kp.mesh
        mesh_str = f"{n1}x{n2}x{n3}"
    else:
        mesh_str = f"{len(kp)} k-points"
    grid = "Γ-centred" if kp.grid_type == "gamma" else "Monkhorst-Pack"
    k_label = (
        "irreducible k-points" if kp.is_symmetry_reduced else "k-points (full mesh)"
    )
    parts = [
        str(character),
        f"Δk={delta_k:.3g} Å⁻¹",
        grid,
        mesh_str,
        f"{kp.n_kpoints} {k_label}",
    ]
    text = ", ".join(parts)
    if smearing is not None and smearing.temperature > 0.0:
        text += f"; smearing {smearing.flavor} kT={smearing.temperature:.4g} Ha"
    if getattr(kp, "bz_integration", None) == "gilat":
        text += "; Gilat-Raubenheimer BZ integration (parameter-free)"
    if notes:
        text += " [" + "; ".join(notes) + "]"
    return text


def _converge_kmesh(
    builder_cls,
    system: PeriodicSystem,
    base_delta_k: float,
    *,
    use_gamma: bool,
    do_sym: bool,
    tol_meV: float,
    energy_fn,
    n_atoms: int,
    ev_per_hartree: float,
) -> "KPointConvergence":
    """Convergence ladder around ``base_delta_k``.

    Evaluates a coarser rung (Δk.√2) and the base for the report, then
    walks finer (Δk/√2, Δk/2, ...) until the total energy/atom changes by
    less than ``tol_meV`` between successive rungs or the refinement cap
    is hit. The coarsest rung within tolerance of the next-finer one is
    chosen. Energies are cached by mesh so repeated meshes (small cells)
    cost a single SCF.
    """
    f = LADDER_SPACING_FACTOR
    cache: dict = {}

    def energy_at(spacing: float):
        kp = builder_cls.from_kspacing(
            system,
            spacing,
            units="angstrom",
            gamma_centred=use_gamma,
            symmetry=do_sym,
        )
        mesh = kp.mesh
        if mesh in cache:
            return cache[mesh], kp
        e = float(energy_fn(kp))
        cache[mesh] = e
        return e, kp

    def per_atom_meV(e_a: float, e_b: float) -> float:
        return abs(e_a - e_b) * ev_per_hartree * 1000.0 / max(1, n_atoms)

    rungs: List[Tuple[float, Tuple[int, int, int], float]] = []

    e_coarse, kp_coarse = energy_at(base_delta_k * f)
    rungs.append((base_delta_k * f, tuple(kp_coarse.mesh), e_coarse))
    e_base, kp_base = energy_at(base_delta_k)
    rungs.append((base_delta_k, tuple(kp_base.mesh), e_base))

    converged = False
    chosen_dk, chosen_mesh = base_delta_k, tuple(kp_base.mesh)
    prev_dk, prev_e, prev_mesh = base_delta_k, e_base, tuple(kp_base.mesh)
    for _ in range(VERIFY_MAX_REFINE_STEPS):
        fine_dk = prev_dk / f
        e_fine, kp_fine = energy_at(fine_dk)
        rungs.append((fine_dk, tuple(kp_fine.mesh), e_fine))
        if per_atom_meV(prev_e, e_fine) <= tol_meV:
            converged = True
            chosen_dk, chosen_mesh = prev_dk, prev_mesh
            break
        prev_dk, prev_e, prev_mesh = fine_dk, e_fine, tuple(kp_fine.mesh)
    else:
        chosen_dk, chosen_mesh = prev_dk, prev_mesh

    return KPointConvergence(
        converged=converged,
        tolerance_meV_per_atom=float(tol_meV),
        rungs=tuple(rungs),
        chosen_delta_k=float(chosen_dk),
        chosen_mesh=tuple(chosen_mesh),
    )


def _check_periodicity_hint(
    system: PeriodicSystem,
    periodicity: Union[int, str],
    warnings_mod,
) -> None:
    """Warn (don't fail) when the optional ``periodicity`` hint disagrees
    with the structure's own ``system.dim``."""
    mapping = {
        "3d": 3,
        "bulk": 3,
        "3": 3,
        "2d": 2,
        "slab": 2,
        "surface": 2,
        "2": 2,
        "1d": 1,
        "wire": 1,
        "chain": 1,
        "1": 1,
    }
    if isinstance(periodicity, str):
        want = mapping.get(periodicity.strip().lower())
    else:
        want = int(periodicity)
    if want is None:
        raise ValueError(
            f"recommend: unrecognised periodicity {periodicity!r}; use "
            "3/2/1 or '3d'/'2d'/'slab'/'1d'."
        )
    if want != int(system.dim):
        warnings_mod.warn(
            f"KPoints.recommend: periodicity hint ({want}D) disagrees with "
            f"system.dim ({int(system.dim)}D); using system.dim. "
            "Non-periodic axes get N_i = 1.",
            UserWarning,
            stacklevel=3,
        )


# ----------------------------------------------------------------------
# Boundary helper -- periodic SCF drivers may receive either a KPoints
# or a legacy BlochKMesh; this normalises the input at the call site.
# ----------------------------------------------------------------------


def as_bloch_kmesh(
    kpts: Union["KPoints", BlochKMesh],
) -> BlochKMesh:
    """Normalize a user-supplied k-point spec to a native
    :class:`BlochKMesh`.

    This is the boundary helper that the periodic SCF dispatchers can
    use to accept both the new :class:`KPoints` builder and the legacy
    raw ``BlochKMesh`` without touching their existing signatures.
    """
    if isinstance(kpts, KPoints):
        return kpts.to_bloch_kmesh()
    return kpts
