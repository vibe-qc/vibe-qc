"""Defect + adsorbate ladder on a B1 rocksalt(100) surface cell.

Status: geometry-corrected scaffold (2026-08-20). Earlier numerical
anchors used an x-striped lattice with fourfold, rather than rocksalt's
sixfold, bulk coordination and an inexact rank-one replacement for the
Parry/de Leeuw K=0 kernel. Those conclusions are withdrawn. This script
now uses the primitive neutral B1(100) checkerboard and the complete K=0
kernel; its outputs must be treated as a fresh validation series.

Systems (4-plane checkerboard, 2x2 in-plane, a = 4.212 A):
  pristine     16 Mg + 16 O, n_occ = 64
  O vacancy    16 Mg + 15 O (F0 center), n_occ = 61
  +H2O          16 Mg + 15 O + H2O above the vacancy, n_occ = 65

The vacancy and adsorbate break primitive-cell translation symmetry.
All three systems therefore use the full 2x2 surface supercell as the
finite-group cell (group order one), so their energy differences have a
common normalization.
"""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
A = 4.212


def _surface_primitives() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([A / 2.0, A / 2.0, 0.0]),
        np.array([-A / 2.0, A / 2.0, 0.0]),
    )


def _topology(coords_bohr, n):
    t1, t2 = _surface_primitives()
    supercell_vectors = (n * t1, n * t2)
    topology = build_seccm_topology(
        coords_bohr,
        [vector * BOHR for vector in supercell_vectors],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=[vector * BOHR for vector in supercell_vectors],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def checkerboard_positions(
    n_planes: int, n: int = 2
) -> tuple[list[np.ndarray], list[int]]:
    """Primitive B1(100) checkerboards with adjacent planes site-swapped."""
    if n_planes < 1:
        raise ValueError("plane count must be positive")
    t1, t2 = _surface_primitives()
    unlike_offset = np.array([A / 2.0, 0.0, 0.0])
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(n_planes):
        z_offset = np.array([0.0, 0.0, k * A / 2.0])
        for i in range(n):
            for j in range(n):
                home = i * t1 + j * t2 + z_offset
                if k % 2 == 0:
                    atoms.extend((home, home + unlike_offset))
                    zs.extend((12, 8))
                else:
                    atoms.extend((home + unlike_offset, home))
                    zs.extend((12, 8))
    return atoms, zs


def _molecule(coords_ang, zs) -> Molecule:
    return Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, coords_ang)],
        0,
        1,
    )


def pristine(n_planes: int = 4, n: int = 2):
    coords, zs = checkerboard_positions(n_planes, n)
    return _molecule(coords, zs), _topology(
        [np.asarray(c) * BOHR for c in coords], n
    )


def o_vacancy(n_planes: int = 4, n: int = 2):
    """Remove the surface O at (a/2, 0, 0) (plane 0)."""
    coords, zs = checkerboard_positions(n_planes, n)
    target = np.array([A / 2.0, 0.0, 0.0])
    for idx, (c, z) in enumerate(zip(coords, zs)):
        if z == 8 and np.linalg.norm(c - target) < 1.0e-6:
            del coords[idx]
            del zs[idx]
            break
    else:
        raise RuntimeError("vacancy site not found")
    return _molecule(coords, zs), _topology(
        [np.asarray(c) * BOHR for c in coords], n
    )


def vacancy_with_h2o(h2o_height_ang: float, n_planes: int = 4, n: int = 2):
    """H2O above the vacancy site, O-down, at the given height above the
    surface plane (z = 0). The slab occupies z >= 0, so the adsorbate is
    placed in the negative-z vacuum. Water geometry: frozen gas-phase-like."""
    mol, topo = o_vacancy(n_planes, n)
    coords = [np.asarray(a.xyz) / BOHR for a in mol.atoms]
    zs = [a.Z for a in mol.atoms]
    # H2O: O at (a/2, 0, h); H at +-x, with the HOH angle 104.5 deg.
    d_oh = 0.9572  # A
    half_angle = 52.25 * np.pi / 180.0
    o_pos = np.array([A / 2.0, 0.0, -h2o_height_ang])
    h1 = o_pos + d_oh * np.array(
        [np.sin(half_angle), 0.0, -np.cos(half_angle)]
    )
    h2 = o_pos + d_oh * np.array(
        [-np.sin(half_angle), 0.0, -np.cos(half_angle)]
    )
    coords.append(o_pos); zs.append(8)
    coords.append(h1); zs.append(1)
    coords.append(h2); zs.append(1)
    return _molecule(coords, zs), _topology(
        [np.asarray(c) * BOHR for c in coords], n
    )


def run(mol, topo, label: str, max_iter: int = 4000):
    n_at = len(mol.atoms)
    result = run_gfn2_seccm(
        mol, topo, ewald_gamma=True, max_iter=max_iter,
    )
    q = np.asarray(result.charges)
    print(
        f"{label}: converged={result.converged} physical={result.physical_basin} "
        f"n_iter={result.n_iter} "
        f"E/atom={result.energy * result.group_order / n_at:.6f} "
        f"q_rms={float(np.sqrt((q ** 2).mean())):.4f}"
    )
    return result


def main() -> None:
    # Reference: pristine 4-plane slab and isolated H2O (molecular driver).
    mol, topo = pristine(4)
    e_pristine = run(mol, topo, "pristine 4-plane")

    mol, topo = o_vacancy(4)
    e_vac = run(mol, topo, "O vacancy")
    print(
        f"vacancy raw cost (no chem. pot.): "
        f"{e_vac.energy - e_pristine.energy:.6f} Ha/(2x2 surface cell) "
        f"({(e_vac.energy - e_pristine.energy) * 27.2114:.3f} eV)"
    )

    # Isolated H2O in the same frozen geometry (molecular driver, the
    # SECCM molecular-limit delegate - bit-for-bit the same code path).
    from vibeqc._vibeqc_core import semiempirical as _se_cxx
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    d_oh = 0.9572
    half_angle = 52.25 * np.pi / 180.0
    h2o_coords = [
        np.array([0.0, 0.0, 0.0]),
        d_oh * np.array([np.sin(half_angle), 0.0, np.cos(half_angle)]),
        d_oh * np.array([-np.sin(half_angle), 0.0, np.cos(half_angle)]),
    ]
    h2o_mol = _molecule(h2o_coords, [8, 1, 1])
    h2o_ref = _se_cxx.xtb.run_gfn2_xtb(h2o_mol, load_gfn2_params())
    print(f"isolated H2O (molecular): {h2o_ref.energy:.8f} Ha")

    # Adsorbate height scan above the vacancy.
    print("H2O height scan (adsorbate binding per H2O):")
    best_h = None
    best_e = None
    for h in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        mol_a, topo_a = vacancy_with_h2o(h)
        e_ads = run(mol_a, topo_a, f"H2O @ {h:.1f} A")
        e_bind = e_ads.energy - e_vac.energy - h2o_ref.energy
        print(
            f"  h={h:.1f} A  E_bind={e_bind:.6f} Ha "
            f"({e_bind * 27.2114:.3f} eV)"
        )
        if best_e is None or e_bind < best_e:
            best_e = e_bind
            best_h = h
    print(f"best binding: {best_e:.6f} Ha at h={best_h} A")


if __name__ == "__main__":
    main()
