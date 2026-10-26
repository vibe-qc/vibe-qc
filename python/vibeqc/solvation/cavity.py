"""Cavity tessellation for implicit solvation (S1a).

Builds a molecular-shaped cavity by union of atom-centred spheres,
discretised by Lebedev-Laikov quadrature points. Each surface point
carries an effective area weight ``w_i`` (in bohr^2, so √(w_i/pi) has
units of bohr and the CPCM diagonal element 1.07/√w_i has units of
bohr⁻¹). Points sitting inside a neighbouring atomic sphere are
removed via a smooth switching function (Scalmani-Frisch 2010) so the
cavity surface -- and hence E_solv(R) -- is C¹-continuous as atoms move.

Defaults
--------
* **Radii** -- Bondi (1964) van-der-Waals radii in Å, scaled by
  ``radii_scale = 1.20`` (the GEPOL / standard CPCM convention; the
  scaled radius is what defines the solvent-excluded cavity boundary).
* **Solvent probe** -- ``solvent_probe_radius_ang = 0.0`` builds the
  van-der-Waals surface (vdW-SES); set to e.g. 1.385 Å (water radius
  / 2) to build the solvent-accessible surface (SAS).
* **Discretisation** -- 302 Lebedev-Laikov points per sphere
  (algebraic order 29). Order 110 (~80 effective surface points per
  atom after switching) is the cheap-and-good default in PySCF / ORCA;
  302 matches Gaussian/Q-Chem CPCM tightness.

References
----------
* Bondi, A. *J. Phys. Chem.* 68, 441 (1964) -- vdW radii table.
* Scalmani, G. & Frisch, M. J. *J. Chem. Phys.* 132, 114110 (2010)
  -- continuous-surface-charge (CSC) switching function with
  parameters a = 0.5 and ζ = 0.5.
* Tomasi, J., Mennucci, B. & Cammi, R. *Chem. Rev.* 105, 2999 (2005)
  -- cavity-construction summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

ANG_TO_BOHR = 1.8897261339213

# Bondi (1964) van-der-Waals radii in Ångström. Covers the elements
# CPCM users routinely encounter in organic + inorganic chemistry.
# Missing elements fall back to ``DEFAULT_RADIUS_ANG``; codes that
# need radii outside this table should pass an explicit ``radii``
# argument to ``build_cavity``.
BONDI_RADII_ANG: dict[int, float] = {
    1: 1.20,                                         # H
    2: 1.40,                                         # He
    3: 1.82, 4: 1.53, 5: 1.92, 6: 1.70, 7: 1.55,     # Li-N
    8: 1.52, 9: 1.47, 10: 1.54,                      # O-Ne
    11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10,          # Na-Si
    15: 1.80, 16: 1.80, 17: 1.75, 18: 1.88,          # P-Ar
    19: 2.75, 20: 2.31,                              # K, Ca
    28: 1.63, 29: 1.40, 30: 1.39,                    # Ni, Cu, Zn
    31: 1.87, 32: 2.11, 33: 1.85, 34: 1.90,          # Ga-Se
    35: 1.85, 36: 2.02,                              # Br, Kr
    46: 1.63, 47: 1.72, 48: 1.58, 49: 1.93,          # Pd, Ag, Cd, In
    50: 2.17, 51: 2.06, 52: 2.06, 53: 1.98, 54: 2.16,  # Sn-Xe
    78: 1.75, 79: 1.66, 80: 1.55, 81: 1.96, 82: 2.02,  # Pt-Pb
}

DEFAULT_RADIUS_ANG = 2.00  # fallback for missing elements


@dataclass(frozen=True)
class CavityTessellation:
    """Discrete representation of an implicit-solvation cavity.

    Attributes
    ----------
    points : ndarray, shape (n_pts, 3), bohr
        Cartesian coordinates of the surviving (switched-in) cavity
        surface points.
    weights : ndarray, shape (n_pts,), bohr^2
        Effective area each surviving point represents. The CPCM
        diagonal-element convention uses √w_i for the cavity self-
        interaction.
    point_atom : ndarray, shape (n_pts,), int
        Index of the parent atom (0-based) each surface point sits on.
    switching : ndarray, shape (n_pts,), in [0, 1]
        Smooth Scalmani-Frisch switching factor applied to each point
        (1 = fully outside neighbouring spheres, 0 = fully inside).
        Already folded into ``weights``; kept around for diagnostics
        and gradient assembly.
    atom_radii : ndarray, shape (n_atoms,), bohr
        Scaled atomic radius used for each sphere (includes
        ``radii_scale`` and any solvent-probe offset).
    atom_positions : ndarray, shape (n_atoms, 3), bohr
        Frozen copy of the atomic positions used to build this cavity.
        ``CavityTessellation`` is geometry-specific; rebuild on
        geometry change.
    n_points_per_sphere : int
        Lebedev order requested per sphere (before switching). Useful
        when explaining a sparse cavity ("90% of the requested 302
        points were switched out -- sphere is buried").
    switching_sigma_bohr : float
        The Scalmani-Frisch width this cavity's ``switching`` was built with.

        Recorded because the *gradient* needs it and had no way to ask (#745).
        ``SolventModel.switching_sigma_bohr`` is user-settable and reaches
        :func:`build_cavity`, but the tessellation did not carry it, so
        :func:`~vibeqc.solvation.gradient.cpcm_gradient` fell back to a
        hard-coded 0.5 and differentiated a cavity the run never built. At
        ``sigma = 0.8`` that was wrong by 12 percent of the largest gradient
        component, and at 0.35 by 0.35 percent, against a converged
        finite-difference reference.

        What made it survive is worth knowing: the wrong-sigma gradient is
        still *self-consistent*, so it satisfies translational invariance to
        1e-15 and every exact invariant the suite checks. Only an oracle that
        rebuilds the energy -- ``cpcm_gradient_fd`` -- sees it.
    """

    points: np.ndarray
    weights: np.ndarray
    point_atom: np.ndarray
    switching: np.ndarray
    atom_radii: np.ndarray
    atom_positions: np.ndarray
    n_points_per_sphere: int = 302
    switching_sigma_bohr: float = 0.5

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def total_surface_area_bohr2(self) -> float:
        return float(self.weights.sum())

    @property
    def charge_representation(self) -> str:
        """Point charges -- valid here *because* of the switching function.

        The FIXPVA point-charge kernel ``1/r_ij`` is singular at contact, and
        Lange & Herbert (doi:10.1063/1.3511297, p. 244111-8) state the
        dependency plainly: point charges "necessitate the use of an
        alternative switching function, as close approach of these point
        charges must be avoided". This construction has that switching
        function, so the kernel is sound. A construction without one is not
        entitled to inherit it -- see
        :attr:`vibeqc.solvation.fine_cavity.FineCavity.charge_representation`
        and #744.

        Every shipped Lebedev result was computed this way, so this is also
        the compatibility anchor: changing it would move every reference.
        """
        from .cpcm import POINT_CHARGE

        return POINT_CHARGE


# Lebedev-Laikov point-count -> algebraic-order table. ``scipy.integrate.
# lebedev_rule`` is parameterised by algebraic order, not point count;
# CPCM users (Gaussian, ORCA, PySCF) speak in point counts. We accept
# the more familiar count-based input and translate.
_LEBEDEV_POINTS_TO_ORDER = {
    6: 3, 14: 5, 26: 7, 38: 9, 50: 11, 74: 13, 86: 15, 110: 17,
    146: 19, 170: 21, 194: 23, 230: 25, 266: 27, 302: 29, 350: 31,
    434: 35, 590: 41, 770: 47, 974: 53,
}


def _lebedev_unit_sphere(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Lebedev-Laikov points + weights on the unit sphere.

    Returns ``(xyz, w)`` where ``xyz`` has shape ``(n, 3)`` and ``w``
    has shape ``(n,)`` with ``S w_i = 4 pi``.

    Accepts the standard CPCM Lebedev point counts (6, 14, ..., 974).
    Routes through ``scipy.integrate.lebedev_rule`` (scipy >= 1.15)
    after translating the point count to an algebraic order.
    """
    try:
        from scipy.integrate import lebedev_rule
    except ImportError as exc:  # pragma: no cover -- scipy bundled with vibe-qc
        raise RuntimeError(
            "vibeqc.solvation needs scipy.integrate.lebedev_rule "
            "(scipy >= 1.15). Upgrade scipy: pip install -U scipy."
        ) from exc

    order = _LEBEDEV_POINTS_TO_ORDER.get(int(n_points))
    if order is None:
        available = ", ".join(str(k) for k in sorted(_LEBEDEV_POINTS_TO_ORDER))
        raise ValueError(
            f"n_points_per_sphere={n_points} is not a standard Lebedev "
            f"order. Use one of: {available}."
        )

    xyz, w = lebedev_rule(order)
    # scipy returns xyz as (3, n); transpose for downstream convenience.
    # Weights are already normalised so S w = 4pi (the integration
    # measure on the unit sphere), so the surface element on a sphere
    # of radius r is r^2 . w_i -- used directly by ``build_cavity``.
    xyz = np.ascontiguousarray(xyz.T)
    w = np.ascontiguousarray(w)
    return xyz, w


def _scalmani_switch(d2: np.ndarray, r_other2: np.ndarray,
                     sigma: float = 0.5) -> np.ndarray:
    """Scalmani-Frisch (CSC) continuous switching function.

    Returns a factor in [0, 1] that smoothly cuts off a cavity point
    whose squared distance ``d2`` to a neighbouring atomic centre
    drops below ``r_other2`` (the squared sphere radius of the
    neighbour). The smoothing width is controlled by ``sigma``:

        f(d, r) = 1/2 [1 + erf((d - r) / s)]   with d = √d2, r = √r_other2.

    ``s = 0.5 bohr`` matches the Scalmani-Frisch 2010 reference (eq.
    9-12) and is the value Gaussian / Q-Chem / PySCF use for their
    smooth CPCM cavities. The error function gives a C^inf switch -- so
    derivatives of E_solv(R) are continuous and finite-difference
    gradients converge cleanly.

    Vectorised over both arguments; broadcasts to the larger shape.
    """
    from math import erf as _math_erf

    d = np.sqrt(d2)
    r = np.sqrt(r_other2)
    arg = (d - r) / sigma
    # numpy.special.erf is in scipy; use math.erf via vectorisation to
    # avoid the dep for this one call site.
    erf = np.frompyfunc(_math_erf, 1, 1)
    return 0.5 * (1.0 + erf(arg).astype(np.float64))


def atom_radii_bohr(
    atom_numbers,
    radii: Optional[dict[int, float]] = None,
    radii_scale: float = 1.20,
    solvent_probe_radius_ang: float = 0.0,
) -> np.ndarray:
    """Per-atom cavity sphere radius in bohr -- the single derivation.

    ``(Bondi[Z] * radii_scale + probe) * ANG_TO_BOHR``, with ``radii``
    overriding or extending the Bondi table.

    Extracted because it was written out three times: here, in the solvation
    gradient's ``_atom_radii_bohr``, and again wherever a new cavity
    construction needed it. Two copies of one formula is how the screening
    factor came to disagree with itself (#546, #548); a cavity radius has the
    same property, and a gradient built on radii that differ from the energy's
    would be wrong in exactly the same silent way.
    """
    table = dict(BONDI_RADII_ANG)
    if radii is not None:
        table.update(radii)
    r_ang = np.array(
        [table.get(int(z), DEFAULT_RADIUS_ANG) for z in atom_numbers],
        dtype=np.float64,
    )
    return (r_ang * radii_scale + solvent_probe_radius_ang) * ANG_TO_BOHR


def build_cavity(

    atom_positions_bohr: np.ndarray,
    atom_numbers: Iterable[int],
    *,
    radii: dict[int, float] | None = None,
    radii_scale: float = 1.20,
    solvent_probe_radius_ang: float = 0.0,
    n_points_per_sphere: int = 302,
    switching_sigma_bohr: float = 0.5,
    drop_threshold: float = 1e-8,
) -> CavityTessellation:
    """Build a CPCM/COSMO cavity tessellation around a molecule.

    Parameters
    ----------
    atom_positions_bohr : (n_atoms, 3) array, bohr
        Cartesian atomic coordinates.
    atom_numbers : iterable of int, length n_atoms
        Atomic numbers (Z) -- used to look up vdW radii.
    radii : optional dict[Z] -> Å
        Override / extend the default :data:`BONDI_RADII_ANG` table.
        Provide a value for every element you want to use that's not
        in the bundled table.
    radii_scale : float, default 1.20
        Multiplier applied to each Bondi radius. ``1.20`` is the
        standard PCM / GEPOL convention (Cossi-Scalmani 2003; Tomasi
        2005 Sec. II.A.1).
    solvent_probe_radius_ang : float, default 0.0
        Probe radius (Å) added to each atomic sphere to build the
        solvent-accessible surface (SAS). ``0.0`` builds the
        scaled-vdW (SES-like) surface -- the most common CPCM choice.
        Use ~1.385 Å for water-probe SAS.
    n_points_per_sphere : int, default 302
        Lebedev quadrature order per atomic sphere. Standard tiers:
        50 (fast / rough), 110 (PySCF default), 194 (ORCA default),
        302 (Gaussian default), 590 (tight). Must be a valid
        Lebedev order -- see scipy.integrate.lebedev_rule.
    switching_sigma_bohr : float, default 0.5
        Width (bohr) of the Scalmani-Frisch continuous switch. Larger
        values smooth the switch over a wider distance -- useful for
        very close atom pairs (~1.5 bohr apart) where the default
        0.5 bohr clip is too aggressive.
    drop_threshold : float, default 1e-8
        Points whose switched weight falls below this fraction of the
        raw Lebedev weight are dropped entirely. Saves work in the
        CPCM linear solve at the cost of a tiny discontinuity (the
        switching function continues smoothly across the cutoff but
        the discrete representation does not -- set to 0.0 to keep
        every point at the cost of more zeros in the A matrix).

    Returns
    -------
    CavityTessellation
        Frozen, geometry-tied cavity. Rebuild on geometry change.
    """
    atom_positions_bohr = np.asarray(atom_positions_bohr, dtype=np.float64)
    if atom_positions_bohr.ndim != 2 or atom_positions_bohr.shape[1] != 3:
        raise ValueError(
            f"atom_positions_bohr must be (n_atoms, 3); got "
            f"{atom_positions_bohr.shape}"
        )

    Zs = np.asarray(list(atom_numbers), dtype=int)
    if Zs.shape[0] != atom_positions_bohr.shape[0]:
        raise ValueError(
            f"atom_numbers length {Zs.shape[0]} does not match "
            f"n_atoms = {atom_positions_bohr.shape[0]}"
        )

    atom_radii_bohr_arr = atom_radii_bohr(
        Zs, radii, radii_scale, solvent_probe_radius_ang
    )

    unit_xyz, unit_w = _lebedev_unit_sphere(n_points_per_sphere)
    n_lebedev = unit_xyz.shape[0]

    # Stage 1: place all candidate surface points (raw Lebedev) around
    # each atom, scaled by that atom's radius.
    all_points = []
    all_weights = []
    all_parent = []
    all_switch = []

    n_atoms = atom_positions_bohr.shape[0]
    sigma = float(switching_sigma_bohr)

    for ia in range(n_atoms):
        R_a = atom_positions_bohr[ia]
        r_a = atom_radii_bohr_arr[ia]
        pts = R_a[None, :] + r_a * unit_xyz                # (n_leb, 3)
        # Raw surface element: w_i_raw = r_a^2 x unit_w_i  (bohr^2).
        # This is the standard sphere-surface-integral element for a
        # sphere of radius r_a.
        w_raw = (r_a ** 2) * unit_w                         # (n_leb,)

        # Stage 2: Scalmani-Frisch smooth switch -- for each candidate
        # point, multiply by ∏_{b != a} f(d_ib, r_b) so points sitting
        # inside neighbouring spheres are continuously cut.
        sw = np.ones(n_lebedev, dtype=np.float64)
        for jb in range(n_atoms):
            if jb == ia:
                continue
            R_b = atom_positions_bohr[jb]
            r_b = atom_radii_bohr_arr[jb]
            d2 = np.sum((pts - R_b[None, :]) ** 2, axis=1)  # (n_leb,)
            sw *= _scalmani_switch(d2, np.full_like(d2, r_b * r_b),
                                   sigma=sigma)

        keep = sw > drop_threshold
        if not np.any(keep):
            # Fully buried sphere (rare unless atoms are stacked or the
            # solvent_probe_radius_ang is large). Skip -- its surface
            # contributes nothing.
            continue

        all_points.append(pts[keep])
        all_weights.append(w_raw[keep] * sw[keep])
        all_switch.append(sw[keep])
        all_parent.append(np.full(int(keep.sum()), ia, dtype=int))

    if not all_points:
        raise ValueError(
            "build_cavity produced zero surface points -- every atomic "
            "sphere was fully switched out. Check atom_positions_bohr "
            "for stacked atoms or radii for missing elements."
        )

    return CavityTessellation(
        points=np.vstack(all_points),
        weights=np.concatenate(all_weights),
        point_atom=np.concatenate(all_parent),
        switching=np.concatenate(all_switch),
        atom_radii=atom_radii_bohr_arr,
        atom_positions=atom_positions_bohr.copy(),
        n_points_per_sphere=int(n_points_per_sphere),
        switching_sigma_bohr=sigma,
    )


