"""Vibe-view-ready **periodic** QVF emission for the Γ-CCM (``aiccm2026dev-a``).

The molecular QVF writer (:func:`vibeqc.output.formats.qvf.write_qvf`) emits a
``structure`` section with ``pbc=[false,false,false]`` and no lattice when handed
a :class:`~vibeqc.Molecule` -- which is what the CCM supercell is. vibe-view's
periodic renderer (cell box, ``N1×N2×N3`` tiling, minimum-image bonds, the
"wrap orbital to cell centre" toggle, the Wannier-centre overlay) stays dormant
until the writer emits the periodic fields. This module supplies them for a
converged CCM result:

* **structure.pbc + lattice_vectors** -- from a dimension-correct
  :class:`~vibeqc.PeriodicSystem` carrying the **Born-von-Kármán supercell**
  (torus) lattice, the periodicity of the grid data vibe-view tiles by. ``dim``
  sets ``pbc=[i<dim]`` (1-D chain ``[T,F,F]``, 2-D sheet ``[T,T,F]``); vacuum
  axes keep the (roomy) transverse box.
* **volume.density + volume.orbital torus grids** -- evaluated with the periodic
  (image-summed) AO evaluator :func:`~vibeqc.ewald_j.evaluate_ao_periodic`, so
  the wrap is *in the data*: a Wannier function straddling a cell face renders
  whole. Each grid spans exactly one supercell along every axis
  (``n_i · step_i == L_i``), so tiles abut seamlessly.
* **x_ccm.wannier_centers overlay** -- per localized orbital, a centre + spread.
  Centres/spreads are computed **from the torus grids** by circular mean over the
  periodic axes (NOT the naive position operator ``<r²>-<r>²``, which aliases to
  negative spreads for wrapping Wannier functions), so they are consistent with
  the shipped ``volume.orbital`` fields and physically sane.
* **bonds** -- Mayer bond orders (order-coloured by the viewer).

Reference: Peintinger & Bredow, *J. Comput. Chem.* **35**, 839 (2014) (CCM).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = ["write_ccm_periodic_qvf"]

_BOHR_TO_ANG = 0.529177210903


def _torus_grid_points(lat_cols: np.ndarray, ns: tuple[int, int, int]) -> np.ndarray:
    """Cartesian points (bohr) of a uniform grid spanning one cell, origin at 0.

    Fractional index ``i/n`` over ``[0,1)`` so ``n_i·step_i == L_i`` (the last
    point wraps to the first) -- the seamless-tiling requirement.
    """
    fx, fy, fz = (np.arange(n) / n for n in ns)
    frac = np.stack(np.meshgrid(fx, fy, fz, indexing="ij"), axis=-1).reshape(-1, 3)
    return frac @ np.asarray(lat_cols, float).T


def _periodic_center_spread(psi3d, lat_cols, pbc):
    """Wrap-aware centre (bohr, cartesian) + spread (bohr²) of ``|psi|²`` on the
    torus grid. Circular mean over periodic axes; ordinary mean over vacuum axes;
    minimum-image second moment. Positive by construction."""
    ns = psi3d.shape
    w = np.abs(np.asarray(psi3d, float)) ** 2
    tot = float(w.sum())
    if tot <= 0.0:
        return np.zeros(3), 0.0
    cfrac = np.zeros(3)
    dfx = []
    for ax in range(3):
        marg = w.sum(axis=tuple(a for a in range(3) if a != ax))
        f = np.arange(ns[ax]) / ns[ax]
        if pbc[ax]:
            m = complex((marg * np.exp(2j * np.pi * f)).sum())
            cfrac[ax] = (np.angle(m) / (2.0 * np.pi)) % 1.0
        else:
            cfrac[ax] = float((marg * f).sum() / marg.sum())
        d = f - cfrac[ax]
        if pbc[ax]:
            d = (d + 0.5) % 1.0 - 0.5
        dfx.append(d)
    lat = np.asarray(lat_cols, float)
    center = lat @ cfrac
    dfrac = np.stack(np.meshgrid(*dfx, indexing="ij"), axis=-1)      # (nx,ny,nz,3)
    dcart = dfrac @ lat.T
    spread = float((w * (dcart ** 2).sum(-1)).sum() / tot)
    return center, spread


def _add_vendor_section(qvf_path, section_id, kind, payload_obj, member="data"):
    """Rewrite the .qvf zip adding a vendor (``x_*``) section with one JSON member."""
    body = json.dumps(payload_obj, indent=2).encode()
    sha = hashlib.sha256(body).hexdigest()
    member_path = f"{section_id}/{member}.json"
    tmp = str(qvf_path) + ".tmp"
    with zipfile.ZipFile(qvf_path) as zin:
        names = zin.namelist()
        manifest = json.loads(zin.read("manifest.json"))
        manifest["sections"].append({
            "id": section_id, "kind": kind,
            "members": {member: {"path": member_path, "format": "json", "sha256": sha}},
        })
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for n in names:
                if n != "manifest.json":
                    zout.writestr(n, zin.read(n))
            zout.writestr(member_path, body)
            zout.writestr("manifest.json", json.dumps(manifest, indent=2).encode())
    shutil.move(tmp, qvf_path)


def _mayer_bonds_supercell(ccm, ba) -> list[tuple[int, int, float]]:
    """Map primitive-cell Mayer bonds (atom_i, atom_j, translation) to supercell
    (i, j, order) atom-index pairs for the QVF ``bonds`` section (best effort;
    the home-cell representative bond per primitive pair)."""
    nrep = np.asarray(ccm.nrep, int)
    n_unit = ccm.n_atoms // ccm.n_cells
    # supercell cell ordering is ((i·n2)+j)·n3+k with per-cell contiguous atoms.
    def lin(cell):
        i, j, k = (int(c) % int(n) for c, n in zip(cell, nrep))
        return (i * nrep[1] + j) * nrep[2] + k
    out = []
    for b in ba.bonds:
        gi = lin((0, 0, 0)) * n_unit + int(b.atom_i)
        gj = lin(b.translation) * n_unit + int(b.atom_j)
        if gi != gj:
            out.append((gi, gj, float(b.order)))
    return out


def write_ccm_periodic_qvf(
    ccm,
    result,
    stem,
    *,
    dim: int,
    basis: Optional[str] = None,
    method: str = "RHF",
    orbitals: str = "wannier",
    n_orbitals: Optional[int] = None,
    localize_method: str = "pipek-mezey",
    grid_spacing_bohr: float = 0.18,
    image_radius: int = 1,
    bonds: bool = True,
    center_vacuum: bool = True,
    wall_seconds: float = 0.0,
) -> Path:
    """Emit a vibe-view-ready periodic ``.qvf`` from a converged Γ-CCM result.

    Parameters
    ----------
    ccm : CCMSystem
    result : CCMSCFResult
        Converged closed-shell CCM SCF result (``.density``, ``.mo_coeffs``,
        ``.mo_energies``).
    stem : path-like
        Output stem; the ``.qvf`` suffix is added.
    dim : {1, 2, 3}
        The **true** periodicity, written to ``structure.pbc = [i < dim]``. The
        lattice is always the full 3×3 BvK supercell (vacuum axes carry the box).
    orbitals : {"wannier", "canonical", "none"}
        Which orbital grids to ship. ``"wannier"`` (default) localizes the
        occupieds (:func:`localise_ccm`) and emits an ``x_ccm.wannier_centers``
        overlay; ``"canonical"`` emits a HOMO-window of canonical MOs.
    n_orbitals : int, optional
        Number of orbitals to emit (default: all occupied for ``"wannier"``,
        min(occ, 4) around HOMO for ``"canonical"``).
    grid_spacing_bohr : float
        Target voxel step; per-axis ``n_i = round(L_i / spacing)``.

    Returns
    -------
    Path
        The written ``.qvf`` (``validate_qvf`` asserted valid before return).
    """
    from vibeqc import Atom, PeriodicSystem
    from vibeqc.ewald_j import evaluate_ao_periodic
    from vibeqc.output.formats.qvf import validate_qvf, write_qvf
    from vibeqc.output.plan import OutputPlan

    if dim not in (1, 2, 3):
        raise ValueError(f"write_ccm_periodic_qvf: dim must be 1/2/3, got {dim!r}")
    if orbitals not in ("wannier", "canonical", "none"):
        raise ValueError("orbitals must be 'wannier' / 'canonical' / 'none'")

    lat_cols = np.asarray(ccm.cluster_lattice, dtype=float)     # columns = BvK vectors
    pbc = [i < dim for i in range(3)]

    # Atoms: optionally centre along vacuum axes so the (open) transverse density
    # sits in the box interior and does not wrap across a vacuum face.
    Z = [int(a.Z) for a in ccm.supercell.atoms]
    R = np.array([list(a.xyz) for a in ccm.supercell.atoms], dtype=float)
    if center_vacuum:
        L = np.diag(lat_cols)
        for ax in range(3):
            if not pbc[ax] and L[ax] > 0:
                R[:, ax] += 0.5 * L[ax] - R[:, ax].mean()
    atoms = [Atom(z, r.tolist()) for z, r in zip(Z, R)]
    sys_periodic = PeriodicSystem(int(dim), lat_cols, atoms, 0, 1)

    # Torus grid spanning exactly one supercell.
    L = np.linalg.norm(lat_cols, axis=0)                       # |a_i| per axis
    ns = tuple(max(8, int(round(li / grid_spacing_bohr))) for li in L)
    pts = _torus_grid_points(lat_cols, ns)
    basis_set = ccm.basis
    chi = evaluate_ao_periodic(basis_set, sys_periodic, pts, image_radius=image_radius)

    D = np.asarray(result.density, dtype=float)
    C = np.asarray(result.mo_coeffs, dtype=float)
    n_occ = ccm.supercell.n_electrons() // 2
    span = (lat_cols / np.array(ns, float)).T                  # voxel vectors (bohr)
    origin = np.zeros(3)

    rho = np.einsum("gi,ij,gj->g", chi, D, chi, optimize=True).reshape(ns)
    volume_data = {"Electron density": (rho.astype(np.float32), origin, span)}

    mo_data: list[dict[str, Any]] = []
    centers_payload: list[dict[str, Any]] = []
    if orbitals == "wannier":
        from .localize import localise_ccm

        wan = localise_ccm(result, ccm, method=localize_method)
        Cw = np.asarray(wan.C_loc, dtype=float)
        # Prefer the VALENCE Wannier functions (bonds / lone pairs) over the tight
        # 1s cores -- cores localize ~all charge onto one atom and render as dull
        # tight blobs. Rank core-last by Mulliken charge concentration.
        sel = list(range(Cw.shape[1]))
        chg = np.asarray(getattr(wan, "charges", np.zeros((Cw.shape[1], 1))), float)
        if chg.shape[0] == Cw.shape[1] and chg.shape[1] > 1:
            frac = chg / np.clip(chg.sum(1, keepdims=True), 1e-30, None)
            core = frac.max(1) > 0.9
            valence = [i for i in sel if not core[i]]
            if valence:                              # keep cores only if that's all
                sel = valence
        if n_orbitals is not None:
            sel = sel[:n_orbitals]
        for k, i in enumerate(sel):
            psi = (chi @ Cw[:, i]).reshape(ns).astype(np.float32)
            mo_data.append({
                "label": f"Wannier w{i}", "data": psi, "origin": origin, "span": span,
                "band_index": int(i), "energy_eh": 0.0, "occupation": 2.0,
                "spin": "both", "component": "real",
            })
            c, s = _periodic_center_spread(psi, lat_cols, pbc)
            centers_payload.append({
                "center": (c * _BOHR_TO_ANG).tolist(),
                "spread": float(s * _BOHR_TO_ANG ** 2),
                "orbital_ref": f"vol_mo_{k}", "label": f"w{i}",
            })
    elif orbitals == "canonical":
        n = min(n_orbitals or 4, n_occ)
        idxs = list(range(max(0, n_occ - n), min(C.shape[1], n_occ + 1)))
        for k, idx in enumerate(idxs):
            psi = (chi @ C[:, idx]).reshape(ns).astype(np.float32)
            occ = 2.0 if idx < n_occ else 0.0
            mo_data.append({
                "label": ("HOMO" if idx == n_occ - 1 else
                          "LUMO" if idx == n_occ else f"MO {idx}"),
                "data": psi, "origin": origin, "span": span, "band_index": int(idx),
                "energy_eh": float(result.mo_energies[idx]), "occupation": occ,
                "spin": "both", "component": "real",
            })

    bonds_data = None
    if bonds:
        try:
            from .properties import ccm_mayer_bond_orders

            bonds_data = _mayer_bonds_supercell(ccm, ccm_mayer_bond_orders(result, ccm))
        except Exception:
            bonds_data = None

    plan = OutputPlan.from_run_job_kwargs(
        output=str(stem), method=method, basis=basis or ccm.basis_name, functional=None)

    class _Res:
        converged = bool(getattr(result, "converged", True))
        energy = float(getattr(result, "energy", 0.0))
        fermi_energy = float(result.mo_energies[n_occ - 1])

    qvf_path = write_qvf(
        str(stem), plan, system=sys_periodic, result=_Res(), method=method,
        basis=basis or ccm.basis_name, volume_data=volume_data,
        mo_data=mo_data or None, bonds_data=bonds_data or None,
        wall_seconds=wall_seconds)

    if centers_payload:
        _add_vendor_section(qvf_path, "x_ccm_wannier_centers",
                            "x_ccm.wannier_centers", {"centers": centers_payload})

    v = validate_qvf(qvf_path)
    if not v.get("valid"):
        raise RuntimeError(f"write_ccm_periodic_qvf: invalid QVF: {v.get('errors')}")

    # Self-check: the torus density integrates to N_e (a gross-error gate on the
    # wrap / image_radius). A few-% shortfall is expected for all-electron cores
    # on a uniform viewer grid (the sharp 1s under-resolves as a Riemann sum);
    # 2% catches a real leak (a lost AO is >> that) without flagging core cusp.
    dV = abs(np.linalg.det(lat_cols)) / float(np.prod(ns))
    n_e = ccm.supercell.n_electrons()
    got = float(rho.sum() * dV)
    if abs(got - n_e) > max(0.05, 0.02 * n_e):
        raise RuntimeError(
            f"write_ccm_periodic_qvf: density integral {got:.4f} != N_e={n_e} "
            "(check image_radius / grid resolution -- a >2% shortfall is a leak, "
            "not core under-resolution).")
    return Path(qvf_path)
