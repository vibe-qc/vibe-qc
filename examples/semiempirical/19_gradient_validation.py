"""Semiempirical gradient FD validation — Priority 6b.

Validates analytic gradients against finite differences for all
semiempirical methods that support them (DFTB0, GFN2-xTB, PM6).

Run:  .venv/bin/python examples/semiempirical/19_gradient_validation.py

Produces markdown report at examples/semiempirical/19_gradient_validation.md
with per-method RMSD and max absolute deviation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical.dftb0 import DFTB0Model, SCCDFTBModel
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.methods.pm6 import PM6Model

HERE = Path(__file__).resolve().parent

THETA = np.deg2rad(104.5 / 2)
R_OH = 1.81

MOLECULES = {
    "H2": Molecule([Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])]),
    "H2O": Molecule(
        [
            Atom(8, [0, 0, 0]),
            Atom(1, [R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
            Atom(1, [-R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
        ]
    ),
    "CH4": Molecule(
        [
            Atom(6, [0, 0, 0]),
            Atom(1, [1.186, 1.186, 1.186]),
            Atom(1, [-1.186, -1.186, 1.186]),
            Atom(1, [-1.186, 1.186, -1.186]),
            Atom(1, [1.186, -1.186, -1.186]),
        ]
    ),
    "CO2": Molecule(
        [
            Atom(6, [0, 0, 0]),
            Atom(8, [2.2, 0, 0]),
            Atom(8, [-2.2, 0, 0]),
        ]
    ),
}


def _displace(mol: Molecule, a: int, xyz: list[float]) -> Molecule:
    atoms = []
    for j, at in enumerate(mol.atoms):
        atoms.append(Atom(at.Z, xyz if j == a else list(at.xyz)))
    return Molecule(atoms, charge=mol.charge, multiplicity=mol.multiplicity)


def fd_gradient(mol: Molecule, energy_fn, h: float = 0.001) -> np.ndarray:
    """Finite-difference gradient for molecule using energy_fn."""
    n = len(mol.atoms)
    grad = np.zeros((n, 3))
    for a in range(n):
        for c in range(3):
            xyz_p = list(mol.atoms[a].xyz)
            xyz_m = list(mol.atoms[a].xyz)
            xyz_p[c] += h
            xyz_m[c] -= h
            ep = energy_fn(_displace(mol, a, xyz_p))
            em = energy_fn(_displace(mol, a, xyz_m))
            grad[a, c] = (ep - em) / (2 * h)
    return grad


def validate_method(name: str, make_model, mols: dict) -> list[dict]:
    """Run FD validation for one method across all molecules."""
    rows = []
    for mname, mol in mols.items():
        t0 = time.perf_counter()
        try:
            m = make_model(mol)
            e = m.energy()
            if not hasattr(m, "gradient"):
                rows.append(
                    {
                        "molecule": mname,
                        "method": name,
                        "rmsd": None,
                        "max_err": None,
                        "max_grad": None,
                        "time_s": time.perf_counter() - t0,
                        "error": "no gradient()",
                    }
                )
                continue
            g_analytic = np.asarray(m.gradient())
            g_fd = fd_gradient(mol, lambda x: make_model(x).energy())
            diff = g_analytic - g_fd
            rmsd = float(np.sqrt(np.mean(diff**2)))
            max_err = float(np.max(np.abs(diff)))
            max_grad = float(np.max(np.abs(g_analytic)))
            rows.append(
                {
                    "molecule": mname,
                    "method": name,
                    "rmsd": rmsd,
                    "max_err": max_err,
                    "max_grad": max_grad,
                    "time_s": time.perf_counter() - t0,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "molecule": mname,
                    "method": name,
                    "rmsd": None,
                    "max_err": None,
                    "max_grad": None,
                    "time_s": time.perf_counter() - t0,
                    "error": str(exc)[:80],
                }
            )
    return rows


def generate_report(all_rows: list[dict]) -> str:
    lines = [
        "# Semiempirical gradient FD validation",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"**FD step:** h = 0.001 bohr",
        "",
        "## Results",
        "",
        "| Molecule | Method | RMSD (Ha/bohr) | Max |g| (Ha/bohr) | Max err | Time |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sorted(all_rows, key=lambda x: (x["molecule"], x["method"])):
        if r["error"]:
            lines.append(
                f"| {r['molecule']} | {r['method']} | — | — | — | {r['error'][:50]} |"
            )
        else:
            lines.append(
                f"| {r['molecule']:8s} | {r['method']:10s} | "
                f"{r['rmsd']:.2e} | {r['max_grad']:.4f} | {r['max_err']:.2e} | "
                f"{r['time_s']:.2f}s |"
            )

    lines.append("")
    lines.append("## Summary")
    for method in sorted(set(r["method"] for r in all_rows)):
        mr = [r for r in all_rows if r["method"] == method and r["error"] is None]
        if mr:
            avg_rmsd = np.mean([r["rmsd"] for r in mr])
            avg_max_err = np.mean([r["max_err"] for r in mr])
            lines.append(
                f"- **{method}**: {len(mr)} molecules, "
                f"avg RMSD = {avg_rmsd:.2e}, avg max err = {avg_max_err:.2e} Ha/bohr"
            )

    return "\n".join(lines) + "\n"


def main():
    gfn2_params = load_gfn2_params()

    methods = [
        ("DFTB0", lambda mol: DFTB0Model(mol)),
        ("SCC-DFTB", lambda mol: SCCDFTBModel(mol)),
        ("GFN2-xTB", lambda mol: GFN2Model(mol, gfn2_params, warn=False)),
        ("PM6", lambda mol: PM6Model(mol)),
    ]

    all_rows = []
    for name, make_fn in methods:
        print(f"Validating {name}...")
        rows = validate_method(name, make_fn, MOLECULES)
        all_rows.extend(rows)
        ok = sum(1 for r in rows if r["error"] is None)
        print(f"  {ok}/{len(rows)} molecules OK")

    report = generate_report(all_rows)
    out_path = HERE / "19_gradient_validation.md"
    out_path.write_text(report)
    print(report)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
