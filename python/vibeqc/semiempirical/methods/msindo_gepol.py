"""MSINDO GEPOL/SAS cavity for COSMO -- exact-parity tessellation.

A faithful port of MSINDO's solvent-accessible-surface construction
(``cosmo_sas.f`` + ``cosmo_gepol.f``; algorithm only -- the licensed MSINDO
2025e source is not reproduced and no source path is recorded), used by
:func:`vibeqc.semiempirical.methods.msindo_cosmo.msindo_cosmo` with
``cavity="gepol"`` to reach exact agreement with the reference-MSINDO oracle
(vs the ~1.4 mHa cavity-convention difference of vibe-qc's default
Lebedev-on-Bondi cavity).

The construction (GEPOL; Silla, Tuñón & Pascual-Ahuir, *J. Comput. Chem.* 12,
1077, 1991; Klamt-Schüürmann 1993):

1. Around each atom a **pentakisdodecahedron** (32 vertices, 60 triangular
   faces) is placed on the sphere of radius ``COSMOR(Z)`` (bohr).  Default
   ``COSMOGRID = (0, 3)``: the coarse grid is the bare 32-vertex polyhedron
   (no refinement), so each atom contributes <= 60 segments.
2. Vertices inside a neighbour atom's sphere are culled.  Each surviving
   triangular face becomes one **segment**.
3. Each face is refined ``COSMOGRID(2) - COSMOGRID(1) = 3`` times (4^3 = 64
   sub-triangles, 45 points) to give the segment centre (mean of the on-surface
   sub-triangle corners, projected to the cavity radius) and the number of
   on-surface sub-triangles ``NSASRF`` (<= 64).
4. Segment self-energy ``A_ii = 1.07/√(area)`` with
   ``area = NSASRF . COSMOR^2/(60.4^3)`` (Klamt; ``cosmo_sas.f``).  The A-matrix
   off-diagonal is ``1/r`` **capped** at ``1/2.min(A_ii, A_jj)``
   (``cosmo_mat.f``).

Defaults are RSOLVE = 0 (van-der-Waals surface, segments at the COSMOR radius),
matching the reference program (``voreinstell.f``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .msindo import MSINDO_BOHR_ANGSTROM

# MSINDO COSMO van-der-Waals radii COSMOR(Z), Ångström, Z = 1..54 (datas.f).
# Applied directly (no scale factor); the cavity sphere radius is COSMOR(Z).
_COSMOR_ANG = [
    1.224, 1.540,                                                  # H, He
    1.991, 1.683, 2.112, 1.776, 1.678, 1.656, 1.595, 1.694,        # Li-Ne
    2.724, 2.076, 2.208, 2.520, 2.160, 2.160, 2.100, 2.256,        # Na-Ar
    3.300, 2.772,                                                   # K, Ca
    3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036,  # Sc-Zn
    2.244, 2.532, 2.220, 2.280, 2.196, 2.424,                      # Ga-Kr
    3.636, 2.988,                                                   # Rb, Sr
    3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036,  # Y-Cd
    2.352, 2.424, 2.484, 2.364, 2.424, 2.640,                      # In-Xe
]

# Reference-executable geometry convention, shared with the MSINDO engine.
_BOHR = MSINDO_BOHR_ANGSTROM


def cosmor_bohr(z: int) -> float:
    """COSMO cavity radius for element ``z`` (bohr)."""
    return _COSMOR_ANG[z - 1] / _BOHR


def _pentakisdodecahedron() -> np.ndarray:
    """The 32 pentakisdodecahedron vertices on the sphere of radius √3.

    Port of ``cosmo_sas.f`` lines 86-235: 20 dodecahedron vertices + 12
    face-centre ("pentakis") points, the latter normalised to length √3.
    Returned in the MSINDO construction order (the order the face-finder and the
    refinement depend on).  Scale by ``VDWR/√3`` to place on a sphere of radius
    ``VDWR``.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phia = 2.0 + 2.0 * phi + 1.0 / phi
    phib = 2.0 + phi
    phiab = phia * phia + phib * phib          # original PHIAB (used by DNEIGH)
    phib_n = phib * math.sqrt(3.0) / math.sqrt(phiab)   # normalise to length √3
    phia_n = phia * math.sqrt(3.0) / math.sqrt(phiab)

    verts = np.zeros((32, 3), dtype=np.float64)
    m = 0
    for i in (0, 1):
        si = (-1.0) ** i
        for j in (0, 1):
            sj = (-1.0) ** j
            verts[m] = (si, sj, -1.0)
            verts[m + 1] = (si, sj, 1.0)
            verts[m + 2] = (0.0, si / phi, phi * sj)
            verts[m + 3] = (phi * sj, 0.0, si / phi)
            verts[m + 4] = (si / phi, phi * sj, 0.0)
            verts[m + 5] = (si * phia_n, sj * phib_n, 0.0)
            verts[m + 6] = (0.0, si * phia_n, sj * phib_n)
            verts[m + 7] = (si * phib_n, 0.0, sj * phia_n)
            m += 8
    return verts


def _dneigh_coarse(vdwr: float) -> float:
    """Squared-distance neighbour threshold for the coarse 32-vertex grid
    (``cosmo_sas.f`` line 236)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phia = 2.0 + 2.0 * phi + 1.0 / phi
    phib = 2.0 + phi
    phiab = phia * phia + phib * phib
    return (1.0 + (1.0 - 1.0 / phi) ** 2 + (1.0 - phi) ** 2) * vdwr ** 2 * 25.0 / phiab


def _gepol_refine(points, angrid, ngrid, ngridmax, dneigh, vdwr,
                  neigh_pos, neigh_r, atom_pos, refine):
    """Port of ``COSMO_GEPOL`` (cosmo_gepol.f): refine an atom-local point grid
    ``refine`` times (each triangle -> 4 by edge-midpoint insertion, projected to
    the sphere of radius ``vdwr``), then cull points inside neighbour spheres.

    ``points`` is an (ngridmax, 3) array, atom-local (centred on the atom);
    ``angrid`` an (ngridmax,) bool mask.  Both are mutated in place.  ``ngrid``
    is the number of live points at entry.  ``neigh_pos`` / ``neigh_r`` are the
    neighbour atom positions (global, bohr) and their cavity radii;
    ``atom_pos`` is this atom's position (global).  Returns nothing.
    """
    nlgrid = 0
    m = ngrid
    for _d in range(1, refine + 1):
        maxscal = 0.0
        m0 = m
        for jx in range(ngrid):
            for kx in range(max(nlgrid, jx + 1), ngrid):
                d = points[jx] - points[kx]
                if float(d @ d) > dneigh:
                    continue
                if m0 >= ngridmax:
                    raise RuntimeError("GEPOL refine overflow (NGRIDMAX)")
                mid = 0.5 * (points[jx] + points[kx])
                scal = float(mid @ mid)
                maxscal = max(maxscal, vdwr * vdwr / scal)
                points[m0] = mid * vdwr / math.sqrt(scal)
                m0 += 1
        nlgrid = ngrid
        m = m0
        if _d < refine:
            ngrid = m
        dneigh = dneigh * maxscal / 4.0

    # Cull points inside a neighbour atom's cavity sphere.
    for npos, nr in zip(neigh_pos, neigh_r):
        shift = atom_pos - npos
        for jx in range(ngridmax):
            if not angrid[jx]:
                continue
            g = points[jx] + shift          # atom-local point - (neighbour - atom)
            if math.sqrt(float(g @ g)) <= nr:
                angrid[jx] = False


# Refined-triangle sizes for COSMOGRID(2)-COSMOGRID(1) = 3 (cosmo_sas.f 155-163):
# 45 points, up to 64 sub-triangles.
_NGRIDMAXRF = 45
_NSUBTRI = 64


@dataclass(frozen=True)
class GepolCavity:
    """GEPOL/SAS tessellation: one apparent-surface-charge segment per surviving
    triangular face, with the MSINDO segment self-term."""

    points: np.ndarray        # (n_seg, 3) bohr -- segment centres
    a_diag: np.ndarray        # (n_seg,) -- A-matrix diagonal 1.07/√area
    area: np.ndarray          # (n_seg,) bohr^2 -- segment area (NSASRF.SURFCE)
    seg_atom: np.ndarray      # (n_seg,) int -- owning atom index

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])


def _cpp_gepol_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    return getattr(_indo, "gepol_build_cavity", None)


def build_gepol_cavity(Z, coords_bohr, *, cosmogrid=(0, 3),
                       rsolve=0.0, rscale=0.1) -> GepolCavity:
    """Build the MSINDO GEPOL/SAS cavity (``cosmo_sas.f``).

    Parameters
    ----------
    Z : sequence[int]
    coords_bohr : (n_atoms, 3) array, bohr
    cosmogrid : (int, int), default (0, 3)
        ``(coarse, fine)`` refinement levels.  Only the default -- coarse grid =
        bare pentakisdodecahedron, fine = 3 sub-divisions -- is validated.
    rsolve, rscale : float
        Solvent probe radius (Angstrom) and its dimensionless SAS-contraction
        scale.  Default ``rsolve = 0`` builds the van-der-Waals surface
        (segments at COSMOR).
    """
    coarse, fine = cosmogrid
    if coarse != 0:
        raise NotImplementedError(
            "build_gepol_cavity validated only for COSMOGRID=(0, fine) "
            f"(coarse-grid refinement not ported); got coarse={coarse}.")
    Z = [int(z) for z in Z]
    C = np.asarray(coords_bohr, dtype=np.float64)
    n_atoms = len(Z)

    kernel = _cpp_gepol_kernel()
    if kernel is not None:
        native = kernel(Z, C, int(coarse), int(fine), float(rsolve), float(rscale))
        return GepolCavity(
            points=np.asarray(native.points, dtype=np.float64),
            a_diag=np.asarray(native.a_diag, dtype=np.float64),
            area=np.asarray(native.area, dtype=np.float64),
            seg_atom=np.asarray(native.seg_atom, dtype=int),
        )

    # ``cosmo_sas.f`` accepts RSOLVE in Angstrom and converts it with BOHR.
    rsolvetmp = rsolve / _BOHR if rsolve else 0.0

    seed = _pentakisdodecahedron()                 # (32,3), radius √3
    refine_fine = fine - coarse

    seg_pts: list[np.ndarray] = []
    seg_adiag: list[float] = []
    seg_area: list[float] = []
    seg_atom: list[int] = []

    for n in range(n_atoms):
        vdwr = rsolvetmp + cosmor_bohr(Z[n])       # cavity sphere radius (bohr)
        # Neighbours whose spheres can overlap atom n (cosmo_sas.f 204-216).
        neigh_pos, neigh_r = [], []
        for mm in range(n_atoms):
            if mm == n:
                continue
            vdwrn = (rsolvetmp + cosmor_bohr(Z[mm]) + vdwr) ** 2
            d = C[n] - C[mm]
            if float(d @ d) <= vdwrn + vdwr:
                neigh_pos.append(C[mm])
                neigh_r.append(rsolvetmp + cosmor_bohr(Z[mm]))

        # Coarse grid: the 32 vertices at radius vdwr, culled against neighbours
        # (COSMO_GEPOL with refine=0 -> cull only).
        cgrid = seed * vdwr / math.sqrt(3.0)
        angrid = np.ones(32, dtype=bool)
        dneigh = _dneigh_coarse(vdwr)
        _gepol_refine(cgrid, angrid, 32, 32, dneigh, vdwr,
                      neigh_pos, neigh_r, C[n], refine=0)

        surfce = vdwr ** 2 / (60.0 * 4.0 ** fine)  # area of one finest sub-tri

        # Find faces (triples of mutually-neighbouring coarse vertices) and emit
        # a segment for each, refining it for the centre + on-surface count.
        for i in range(32):
            for j in range(i + 1, 32):
                dij = cgrid[i] - cgrid[j]
                if float(dij @ dij) > dneigh:
                    continue
                for k in range(j + 1, 32):
                    djk = cgrid[j] - cgrid[k]
                    if float(djk @ djk) > dneigh:
                        continue
                    dik = cgrid[k] - cgrid[i]
                    if float(dik @ dik) > dneigh:
                        continue
                    # (i,j,k) is a face. Accept if any corner is on-surface, else
                    # only if the face centre is not buried (cosmo_sas.f 270-291).
                    if not (angrid[i] or angrid[j] or angrid[k]):
                        centre = cgrid[i] + cgrid[j] + cgrid[k]
                        centre = centre * vdwr / math.sqrt(float(centre @ centre))
                        buried = False
                        for npos, nr in zip(neigh_pos, neigh_r):
                            g = centre + C[n] - npos
                            if math.sqrt(float(g @ g)) <= nr:
                                buried = True
                                break
                        if buried:
                            continue
                    seg = _refine_face(cgrid[i], cgrid[j], cgrid[k],
                                       angrid[i], angrid[j], angrid[k],
                                       vdwr, dneigh, neigh_pos, neigh_r, C[n],
                                       refine_fine)
                    if seg is None:
                        continue
                    centre_local, nsasrf = seg
                    vdwrn = rsolvetmp * rscale + cosmor_bohr(Z[n])
                    norm = math.sqrt(float(centre_local @ centre_local))
                    pt = vdwrn / norm * centre_local + C[n]
                    area = surfce * nsasrf
                    seg_pts.append(pt)
                    seg_adiag.append(1.07 / math.sqrt(area))
                    seg_area.append(area)
                    seg_atom.append(n)

    if not seg_pts:
        raise ValueError("build_gepol_cavity produced no segments.")
    return GepolCavity(
        points=np.array(seg_pts, dtype=np.float64),
        a_diag=np.array(seg_adiag, dtype=np.float64),
        area=np.array(seg_area, dtype=np.float64),
        seg_atom=np.array(seg_atom, dtype=int),
    )


def _refine_face(ci, cj, ck, ai, aj, ak, vdwr, dneigh,
                 neigh_pos, neigh_r, atom_pos, refine):
    """Refine one coarse face into sub-triangles; return (centre_local, nsasrf).

    ``centre_local`` is the (un-normalised) sum over on-surface sub-triangle
    corners (cosmo_sas.f 296-391); ``nsasrf`` the number of on-surface
    sub-triangles.  Returns ``None`` if the face contributes no live segment.
    """
    pts = np.zeros((_NGRIDMAXRF, 3), dtype=np.float64)
    angrid = np.ones(_NGRIDMAXRF, dtype=bool)
    pts[0], pts[1], pts[2] = ci, cj, ck
    angrid[0], angrid[1], angrid[2] = ai, aj, ak
    dneigh_rf = dneigh
    if refine > 0:
        # COSMO_GEPOL refines in place + culls (it updates dneigh internally; we
        # need the *final* dneigh for sub-triangle finding, so replicate it).
        dneigh_rf = _gepol_refine_face(pts, angrid, 3, _NGRIDMAXRF, dneigh,
                                       vdwr, neigh_pos, neigh_r, atom_pos, refine)
    ngrid_rf = 3 if refine > 0 else 0

    sas = np.zeros(3, dtype=np.float64)
    nsasrf = 0
    tris: list[tuple[int, int, int]] = []
    for irf in range(_NGRIDMAXRF):
        if not angrid[irf]:
            continue
        for jrf in range(max(irf + 1, ngrid_rf), _NGRIDMAXRF):
            if not angrid[jrf]:
                continue
            d = pts[jrf] - pts[irf]
            if float(d @ d) > dneigh_rf:
                continue
            for krf in range(jrf + 1, _NGRIDMAXRF):
                if not angrid[krf]:
                    continue
                d = pts[jrf] - pts[krf]
                if float(d @ d) > dneigh_rf:
                    continue
                d = pts[krf] - pts[irf]
                if float(d @ d) > dneigh_rf:
                    continue
                tris.append((irf, jrf, krf))
    # Accumulate corners of each fully-on-surface sub-triangle (cosmo_sas.f
    # 353-382 processes in reverse; for a sum that's order-irrelevant to the
    # ~µHa we target).
    for (irf, jrf, krf) in tris:
        if not (angrid[irf] and angrid[jrf] and angrid[krf]):
            continue
        sas += pts[irf] + pts[jrf] + pts[krf]
        nsasrf += 1
    if nsasrf == 0:
        return None
    return sas, nsasrf


def _gepol_refine_face(points, angrid, ngrid, ngridmax, dneigh, vdwr,
                       neigh_pos, neigh_r, atom_pos, refine) -> float:
    """As :func:`_gepol_refine` but returns the final (post-loop) ``dneigh``
    needed by the per-segment sub-triangle finder."""
    nlgrid = 0
    m = ngrid
    for _d in range(1, refine + 1):
        maxscal = 0.0
        m0 = m
        for jx in range(ngrid):
            for kx in range(max(nlgrid, jx + 1), ngrid):
                d = points[jx] - points[kx]
                if float(d @ d) > dneigh:
                    continue
                if m0 >= ngridmax:
                    raise RuntimeError("GEPOL face refine overflow")
                mid = 0.5 * (points[jx] + points[kx])
                scal = float(mid @ mid)
                maxscal = max(maxscal, vdwr * vdwr / scal)
                points[m0] = mid * vdwr / math.sqrt(scal)
                m0 += 1
        nlgrid = ngrid
        m = m0
        if _d < refine:
            ngrid = m
        dneigh = dneigh * maxscal / 4.0
    for npos, nr in zip(neigh_pos, neigh_r):
        shift = atom_pos - npos
        for jx in range(ngridmax):
            if not angrid[jx]:
                continue
            g = points[jx] + shift
            if math.sqrt(float(g @ g)) <= nr:
                angrid[jx] = False
    return dneigh


def build_gepol_A_matrix(cavity: GepolCavity) -> np.ndarray:
    """MSINDO COSMO segment interaction matrix A (``cosmo_mat.f`` 221-240).

    Diagonal ``A_ii`` = the GEPOL self-term; off-diagonal ``A_ij = 1/r_ij``
    **capped** at ``1/2.min(A_ii, A_jj)`` (the near-segment regularisation).
    """
    pts = cavity.points
    adiag = cavity.a_diag
    try:
        from vibeqc import _vibeqc_core as _core
    except ImportError:
        _core = None
    if _core is not None and hasattr(_core, "cpcm_build_capped_A_matrix"):
        return np.asarray(
            _core.cpcm_build_capped_A_matrix(pts, adiag),
            dtype=np.float64,
        )
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    with np.errstate(divide="ignore"):
        A = 1.0 / dist
    # Cap each off-diagonal at half the smaller of the two diagonal self-terms.
    cap = 0.5 * np.minimum(adiag[:, None], adiag[None, :])
    np.fill_diagonal(cap, np.inf)               # don't cap the diagonal
    A = np.minimum(A, cap)
    np.fill_diagonal(A, adiag)
    return A
