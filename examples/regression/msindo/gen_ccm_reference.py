"""Regenerate ``ccm_reference.json`` — periodic CCM oracle reference data.

Runs the MSINDO oracle (out-of-process, CLAUDE.md §10) on a small set of 1-D
cyclic clusters and records, per cluster, the converged energies, the
Madelung-nuclear (Wigner-Seitz-weighted core–core) energy, and the per-atom WS
neighbour/weight table.  These static numbers are the parity targets for the
pure-Python periodic engine (``vibeqc.semiempirical.methods.msindo_ccm``).

Usage::

    MSINDO_ORACLE=/tmp/msindo_oracle .venv/bin/python \\
        examples/regression/msindo/gen_ccm_reference.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from runner_msindo import run_ccm, run_ccm_gradient


def he_chain(n: int, a: float):
    real = [("He", i * a, 0.0, 0.0) for i in range(n)]
    return real, [(n * a, 0.0, 0.0)]


def hf_ionic(a: float, n_units: int):
    """Alternating ...H-F-H-F... ionic chain (1-D NaCl analogue), spacing ``a``."""
    real = []
    for u in range(n_units):
        real.append(("H", (2 * u) * a, 0.0, 0.0))
        real.append(("F", (2 * u + 1) * a, 0.0, 0.0))
    return real, [(2 * n_units * a, 0.0, 0.0)]


def hf_dilute(bond: float, period: float):
    """Dilute HF-molecule chain — each WS cell holds only its covalent partner,
    so the CCM energy must reduce to the isolated-molecule value."""
    return [("F", 0.0, 0.0, 0.0), ("H", bond, 0.0, 0.0)], [(period, 0.0, 0.0)]


def he_sc(a: float):
    """3-D simple-cubic He, 2×2×2 supercell (8 atoms), period 2a — a covalent
    3-D periodic test (closed-shell, NOEWALD)."""
    real = [("He", i * a, j * a, k * a)
            for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    t = 2 * a
    return real, [(t, 0.0, 0.0), (0.0, t, 0.0), (0.0, 0.0, t)]


def _rocksalt(cation, anion, a):
    """Rocksalt conventional cubic cell (4 cation + 4 anion), lattice ``a``."""
    fcc = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    real = [(cation, fx * a, fy * a, fz * a) for fx, fy, fz in fcc]
    real += [(anion, (fx + 0.5) * a, fy * a, fz * a) for fx, fy, fz in fcc]
    return real, [(a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)]


def hf_rocksalt(a: float):
    """3-D rocksalt H-F (MgO structure type with H/F), the precursor to MgO."""
    return _rocksalt("H", "F", a)


def mgo_rocksalt(a: float):
    """3-D rocksalt **MgO** (Mg₄O₄ conventional cell) — the real ionic oxide."""
    return _rocksalt("Mg", "O", a)


def he_layer(a: float):
    """2-D square He monolayer (2×2 cell), covalent surface test (NOEWALD)."""
    real = [("He", i * a, j * a, 0.0) for i in (0, 1) for j in (0, 1)]
    return real, [(2 * a, 0.0, 0.0), (0.0, 2 * a, 0.0)]


def hf_layer(a: float):
    """2-D flat H-F checkerboard monolayer (2×2 cell) — in-plane Madelung
    (out-of-plane separation is zero)."""
    real = [("H", 0, 0, 0), ("F", a, 0, 0), ("F", 0, a, 0), ("H", a, a, 0)]
    return real, [(2 * a, 0.0, 0.0), (0.0, 2 * a, 0.0)]


def mgo_100_slab(a: float):
    """2-D **MgO(100)** bilayer slab (8 atoms), periodic in-plane — the oxide
    adsorption substrate; exercises the out-of-plane 2-D Ewald terms."""
    real = [("Mg", 0, 0, 0), ("O", a, 0, 0), ("O", 0, a, 0), ("Mg", a, a, 0),
            ("O", 0, 0, a), ("Mg", a, 0, a), ("Mg", 0, a, a), ("O", a, a, a)]
    return real, [(2 * a, 0.0, 0.0), (0.0, 2 * a, 0.0)]


def _mgo_100_adsorbate(a: float, adsorbate):
    """MgO(100) slab + one adsorbate per cell (a periodic adlayer)."""
    real, trans = mgo_100_slab(a)
    return list(real) + list(adsorbate), trans


def mgo_100_water(a: float):
    """MgO(100) + H₂O adlayer — the headline oxide-adsorption system."""
    return _mgo_100_adsorbate(a, [
        ("O", a, 0.0, a + 2.2), ("H", a + 0.76, 0.0, a + 2.76),
        ("H", a - 0.76, 0.0, a + 2.76)])


def mgo_100_nh3(a: float):
    """MgO(100) + NH₃ adlayer (N down toward a surface Mg)."""
    z = a + 2.2
    return _mgo_100_adsorbate(a, [
        ("N", a, 0.0, z), ("H", a + 0.94, 0.0, z + 0.38),
        ("H", a - 0.47, 0.81, z + 0.38), ("H", a - 0.47, -0.81, z + 0.38)])


def mgo_100_ch4(a: float):
    """MgO(100) + CH₄ adlayer (an organic; one H toward a surface O)."""
    z = a + 2.6
    return _mgo_100_adsorbate(a, [
        ("C", a, 0.0, z), ("H", a, 0.0, z - 1.09), ("H", a + 1.03, 0.0, z + 0.36),
        ("H", a - 0.51, 0.89, z + 0.36), ("H", a - 0.51, -0.89, z + 0.36)])


CLUSTERS = [
    ("hf_dilute_chain", *hf_dilute(0.917, 3.0), False,
     "molecular-limit anchor: CCM1D reduces to isolated HF"),
    ("he5_a2.5_noewald", *he_chain(5, 2.5), False,
     "5-atom He chain (odd N, all WS weights 1.0); s-only periodic, no Madelung"),
    ("hfionic_a1.6_n1_noewald", *hf_ionic(1.6, 1), False,
     "1-unit H-F ionic chain, no Madelung"),
    ("hfionic_a1.6_n1_madelung", *hf_ionic(1.6, 1), True,
     "1-unit H-F ionic chain with Madelung (CCM1D Ewald)"),
    ("hfionic_a1.6_n2_noewald", *hf_ionic(1.6, 2), False,
     "2-unit H-F ionic chain, no Madelung"),
    ("hfionic_a1.6_n2_madelung", *hf_ionic(1.6, 2), True,
     "2-unit H-F ionic chain with Madelung (CCM1D Ewald)"),
    ("he_sc_2x2x2_a2.5_noewald", *he_sc(2.5), False,
     "3-D simple-cubic He (2x2x2 supercell), covalent periodic, no Madelung"),
    ("hf_rocksalt_a4.0_noewald", *hf_rocksalt(4.0), False,
     "3-D rocksalt H-F (MgO structure type), no Madelung"),
    ("hf_rocksalt_a4.0_madelung", *hf_rocksalt(4.0), True,
     "3-D rocksalt H-F (MgO structure type) with 3-D Ewald Madelung"),
    ("mgo_rocksalt_a4.21_noewald", *mgo_rocksalt(4.21), False,
     "3-D rocksalt MgO (Mg4O4 cell), no Madelung"),
    ("mgo_rocksalt_a4.21_madelung", *mgo_rocksalt(4.21), True,
     "3-D rocksalt MgO (Mg4O4 cell) with 3-D Ewald Madelung — real ionic oxide"),
    ("he_layer_a2.5_noewald", *he_layer(2.5), False,
     "2-D square He monolayer, covalent surface, no Madelung"),
    ("hf_layer_a1.6_madelung", *hf_layer(1.6), True,
     "2-D flat H-F checkerboard monolayer with 2-D Ewald Madelung (in-plane)"),
    ("mgo_100_slab_a2.105_madelung", *mgo_100_slab(2.105), True,
     "2-D MgO(100) bilayer slab with 2-D Ewald Madelung — adsorption substrate"),
    ("mgo_100_water_a2.105_madelung", *mgo_100_water(2.105), True,
     "2-D MgO(100) slab + H2O adlayer (CCM2D Madelung) — oxide adsorption"),
    ("mgo_100_nh3_a2.105_madelung", *mgo_100_nh3(2.105), True,
     "2-D MgO(100) slab + NH3 adlayer (CCM2D Madelung)"),
    ("mgo_100_ch4_a2.105_madelung", *mgo_100_ch4(2.105), True,
     "2-D MgO(100) slab + CH4 adlayer (organic; CCM2D Madelung)"),
]


def _displace(builder_out, atom, shift):
    """Return (atoms, translations) with one atom shifted in x — a
    non-equilibrium geometry that exposes nonzero nuclear forces."""
    real, trans = builder_out
    real = [list(a) for a in real]
    real[atom][1] += shift
    return [tuple(a) for a in real], trans


# Distorted (non-equilibrium) clusters for analytic-gradient parity.
GRADIENT_CLUSTERS = [
    ("hfionic_chain_distorted",
     [("H", 0.1, 0, 0), ("F", 1.5, 0, 0), ("H", 3.4, 0, 0), ("F", 4.7, 0, 0)],
     [(6.4, 0, 0)], False, "asymmetric H-F chain (CCM1D NOEWALD)"),
    ("mgo_bulk_distorted", *_displace(mgo_rocksalt(4.21), 4, 0.15), True,
     "MgO rocksalt with one O displaced (CCM3D Madelung)"),
]


def main() -> None:
    clusters = []
    for name, real, trans, ewald, note in CLUSTERS:
        res = run_ccm(real, trans, title=name, ewald=ewald)
        assert not res.wrong_neighbors, f"{name}: invalid cyclic cluster!"
        assert res.converged, f"{name}: SCF did not converge"
        clusters.append({
            "name": name,
            "note": note,
            "dim": len(trans),
            "ewald": ewald,
            "real_atoms": [[s, x, y, z] for s, x, y, z in real],
            "translations": [list(t) for t in trans],
            "reference": {
                "total_energy": res.total_energy,
                "electronic_energy": res.electronic_energy,
                "binding_energy": res.binding_energy,
                "madelung_nuclear_energy": res.madelung_nuclear_energy,
                "scf_cycles": res.scf_cycles,
            },
            "ws_cells": res.ws_cells,
        })
        print(f"{name:28s} tot={res.total_energy} "
              f"mad-nuc={res.madelung_nuclear_energy} cyc={res.scf_cycles}")

    gradient_clusters = []
    for name, real, trans, ewald, note in GRADIENT_CLUSTERS:
        grad = run_ccm_gradient(real, trans, ewald=ewald)
        gradient_clusters.append({
            "name": name,
            "note": note,
            "dim": len(trans),
            "ewald": ewald,
            "real_atoms": [[s, x, y, z] for s, x, y, z in real],
            "translations": [list(t) for t in trans],
            "gradient_ha_per_bohr": grad,
        })
        print(f"{name:28s} analytic gradient captured ({len(grad)} atoms)")

    out = {
        "_comment": (
            "MSINDO periodic Cyclic Cluster Model (CCM) reference data for "
            "vibe-qc parity validation. Generated out-of-process (CLAUDE.md §10) "
            "by examples/regression/msindo/gen_ccm_reference.py against a local "
            "MSINDO 2025 build. 1-D cyclic clusters: He chain (covalent, "
            "NOEWALD) and alternating H-F ionic chains (with/without Madelung). "
            "Geometry in Angstrom, energy in Hartree. madelung_nuclear_energy = "
            "MSINDO 'MADELUNG-NUCLEAR ENERGY' (WS-weighted core-core point-charge "
            "sum). ws_cells = per-atom WS neighbour table (1-based origin ids + "
            "partial-count weights, PRINTOPTS=CCMWSC)."
        ),
        "method": "msindo-ccm",
        "source_version": "MSINDO 2025 (Jul. 2024)",
        "units": {"geometry": "angstrom", "energy": "hartree",
                  "gradient": "hartree/bohr"},
        "clusters": clusters,
        "gradient_clusters": gradient_clusters,
    }
    path = Path(__file__).with_name("ccm_reference.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path} ({len(clusters)} clusters, "
          f"{len(gradient_clusters)} gradient clusters)")


if __name__ == "__main__":
    main()
