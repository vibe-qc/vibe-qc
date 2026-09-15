"""N₂ triple-bond dissociation — where a single determinant fails.

Breaking the N≡N triple bond is the textbook case of **static
correlation**: at stretched geometries the exact wavefunction is a
near-equal mix of the σ/π bonding configuration and the σ*/π*
antibonding ones, and no single Slater determinant can represent that.
Restricted Hartree–Fock is forced to keep the bonding orbitals doubly
occupied, mixes in high-energy ionic terms (N⁺···N⁻), and climbs far
above the correct two-atom limit — the classic RHF dissociation
catastrophe. The single-reference correlated methods inherit the
problem (MP2 diverges; CCSD develops a non-variational bump).

The cure is a complete active space holding all six bonding/antibonding
configurations on an equal footing: **CAS(6,6)** — the 6 electrons of
the σ and two π bonds in the 6 valence orbitals σ2p / π2p×2 / π*2p×2 /
σ*2p. In STO-3G that freezes the lowest ``(14 − 6)/2 = 4`` MOs
(2× N 1s, σ2s, σ*2s) as a properly *dressed* doubly-occupied core — the
frozen-core dressing (effective one-electron term + E_core offset) is
what makes ``method="fci", active_space=(6, 6)`` report a real total
energy you can put on a dissociation curve.

This script scans the bond length and compares three curves:

  * RHF                — single determinant (wrong at dissociation)
  * CASCI(6,6)         — full CI in the active space on RHF orbitals
                         (``method="fci", active_space=(6, 6)``)
  * CASSCF(6,6)        — additionally relaxes the orbitals at each
                         geometry (``method="casscf"``)

What to watch: the RHF curve, measured relative to its own
dissociation limit, overshoots by hundreds of kcal/mol in the
bond-breaking region, while the CAS curves stay smooth and physical.
The RHF–CAS gap *grows with bond length* — the fingerprint of static
correlation.

Run:
    python examples/wavefunction/n2_dissociation_active_space.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from vibeqc import Atom, Molecule
from vibeqc.runner import run_job

HARTREE_TO_KCAL = 627.5094740631

# Bond lengths in bohr.  N₂ equilibrium is ≈ 2.07 bohr (1.098 Å); the scan
# runs from a compressed bond out to near-dissociation.
BOND_LENGTHS_BOHR = [1.9, 2.07, 2.3, 2.6, 3.0, 3.5, 4.5, 6.0]


def _energy(mol: Molecule, method: str, out_dir: Path, **kw) -> float | None:
    """run_job wrapper: quiet, no side-car files; None if it fails to converge."""
    try:
        result = run_job(
            mol,
            basis="sto-3g",
            method=method,
            output=out_dir / f"n2_{method}",
            write_xyz_file=False,
            write_molden_file=False,
            write_population_file=False,
            citations=False,
            **kw,
        )
    except Exception as exc:  # dissociation can stress SCF / orbital opt
        print(f"   [warn] {method} failed at this geometry: {exc}")
        return None
    return result.energy


def scan(out_dir: Path) -> list[dict]:
    rows = []
    for r in BOND_LENGTHS_BOHR:
        mol = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, r])])
        e_rhf = _energy(mol, "rhf", out_dir)
        # CASCI(6,6): full CI inside the active space, on the (frozen-core
        # dressed) RHF reference.  This is the path the active_space fix
        # makes report a correct absolute total energy.
        e_casci = _energy(mol, "fci", out_dir, active_space=(6, 6))
        # CASSCF(6,6): same active space, orbitals relaxed at this geometry.
        e_casscf = _energy(mol, "casscf", out_dir, active_space=(6, 6))
        rows.append(
            {"R": r, "RHF": e_rhf, "CASCI": e_casci, "CASSCF": e_casscf}
        )
    return rows


def _rel(rows: list[dict], key: str) -> list[float | None]:
    """Energies relative to the longest-bond point, in kcal/mol."""
    ref = rows[-1][key]
    if ref is None:
        return [None] * len(rows)
    return [
        None if row[key] is None else (row[key] - ref) * HARTREE_TO_KCAL
        for row in rows
    ]


def main() -> None:
    print("=" * 74)
    print(" N₂ dissociation — single determinant vs active space (STO-3G)")
    print("=" * 74)

    with tempfile.TemporaryDirectory() as tmp:
        rows = scan(Path(tmp))

    rel_rhf = _rel(rows, "RHF")
    rel_casci = _rel(rows, "CASCI")
    rel_casscf = _rel(rows, "CASSCF")

    # ── Absolute energies ──
    print("\nAbsolute total energies (Ha):")
    print(f" {'R/bohr':>7s}  {'E(RHF)':>15s}  {'E(CASCI66)':>15s}  {'E(CASSCF66)':>15s}")
    print(" " + "-" * 60)
    for row in rows:
        def fmt(x: float | None) -> str:
            return "      n/a      " if x is None else f"{x:>15.8f}"

        print(
            f" {row['R']:>7.2f}  {fmt(row['RHF'])}  "
            f"{fmt(row['CASCI'])}  {fmt(row['CASSCF'])}"
        )

    # ── Shape of the curves: energy relative to the dissociation point ──
    print("\nRelative to the longest-bond point (kcal/mol) — the curve shape:")
    print(f" {'R/bohr':>7s}  {'ΔE(RHF)':>12s}  {'ΔE(CASCI)':>12s}  {'ΔE(CASSCF)':>12s}")
    print(" " + "-" * 52)
    for r_, a, b, c in zip(BOND_LENGTHS_BOHR, rel_rhf, rel_casci, rel_casscf):
        def fmt(x: float | None) -> str:
            return "     n/a    " if x is None else f"{x:>12.1f}"

        print(f" {r_:>7.2f}  {fmt(a)}  {fmt(b)}  {fmt(c)}")

    # ── The lesson, quantified: non-parallelity of RHF vs CASCI ──
    pairs = [
        (row["R"], row["RHF"], row["CASCI"])
        for row in rows
        if row["RHF"] is not None and row["CASCI"] is not None
    ]
    if len(pairs) >= 2:
        gaps = [(r_, (e_rhf - e_cas) * HARTREE_TO_KCAL) for r_, e_rhf, e_cas in pairs]
        r_near, gap_near = min(gaps, key=lambda t: abs(t[0] - 2.07))
        r_far, gap_far = gaps[-1]
        print("\n" + "=" * 74)
        print(" The static-correlation fingerprint:")
        print(
            f"   RHF − CASCI gap near equilibrium (R≈{r_near:.2f}): "
            f"{gap_near:8.1f} kcal/mol"
        )
        print(
            f"   RHF − CASCI gap near dissociation (R={r_far:.2f}):  "
            f"{gap_far:8.1f} kcal/mol"
        )
        print(
            f"   The gap GROWS by {gap_far - gap_near:.1f} kcal/mol as the bond"
        )
        print("   breaks — that is correlation a single determinant cannot")
        print("   capture, and exactly where CI / CASSCF are mandatory.")
        print("=" * 74)


if __name__ == "__main__":
    main()
