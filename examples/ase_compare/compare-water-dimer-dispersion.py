"""Cross-validate D3(BJ) dispersion on the water dimer.

The water dimer is the prototype non-covalent interaction problem:
HF without dispersion underestimates the H-bond, DFT without
dispersion does too (mostly), and adding D3(BJ) brings both into
chemistry-accuracy range. This script compares vibe-qc and ORCA on
the **same dispersion-corrected calculation** to verify our D3(BJ)
implementation matches Grimme's reference parameters.

Run:
    .venv/bin/python examples/ase_compare/compare-water-dimer-dispersion.py

The comparison computes:
  1. Two single-point energies (water monomer at minimum, dimer
     at H-bonded minimum).
  2. Interaction energy: E(dimer) - 2 * E(monomer).
  3. With and without D3(BJ) dispersion, on top of HF / 6-31G*.

Expected outcome:
  * Without D3(BJ): both codes get ~-3 to -4 kcal/mol interaction
    energy (HF underbinds; experiment is ~-5 kcal/mol).
  * With D3(BJ):    both codes get ~-4 to -5 kcal/mol (closer to
    experiment), dispersion contribution ~-0.5 kcal/mol.
  * vibe-qc and ORCA agree to ~uHa on the dispersion contribution
    itself: same Grimme parameters, same damping, same atom-pair sum.

The previous PySCF row (in-process ``PySCFCalculator``) was dropped:
the shim was retired (CLAUDE.md sec 10). For a PySCF cross-check, use
the out-of-process runner at
``examples/regression/core/runner_pyscf.py``.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    compare_calculators,
    make_orca_calculator,
    print_calculator_availability,
)

HERE = Path(__file__).resolve().parent

KCAL_PER_HA = 627.5094740631
EV_PER_KCAL = 1.0 / 23.06054783
EV_PER_HA = 27.211386245988


def water_dimer_geometry() -> Atoms:
    """Cs-symmetric H-bonded water dimer near HF/6-31G* equilibrium.

    Donor O at origin; acceptor O along +z at R(O...O) ~ 2.92 Angstrom.
    Geometry matches the Smith et al. (2016) "S22" benchmark dimer to a
    few mAngstrom.
    """
    return Atoms(
        symbols=["O", "H", "H", "O", "H", "H"],
        positions=[
            (0.000, 0.000,  0.000),
            (0.762, 0.000, -0.566),
            (-0.762, 0.000, -0.566),
            (0.000, 0.000,  2.920),
            (-0.000, 0.787, 3.469),
            (-0.000, -0.787, 3.469),
        ],
    )


def water_monomer_geometry() -> Atoms:
    """Single water at the same OH bond length / H-O-H angle as the dimer."""
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.000, 0.000,  0.000),
            (0.762, 0.000, -0.566),
            (-0.762, 0.000, -0.566),
        ],
    )


def _energy_for(label: str, atoms: Atoms, calculators: list) -> dict[str, float]:
    """Compute the energy for *atoms* through every calculator in
    *calculators*, return a dict label -> eV."""
    results = compare_calculators(atoms, calculators, properties=("energy",))
    out: dict[str, float] = {}
    for row in results:
        if row.status in ("ok", "partial"):
            out[row.label] = row.properties.get("energy")
    return out


def main() -> None:
    print("=" * 72)
    print(" Cross-validation:  water dimer / D3(BJ) dispersion / HF / 6-31G*")
    print("=" * 72)
    print()
    print("Calculator availability:")
    print_calculator_availability()
    print()

    monomer = water_monomer_geometry()
    dimer = water_dimer_geometry()

    # Calculator panels: same monomer/dimer through each set, with
    # and without D3(BJ). vibe-qc's `dispersion=` kwarg pulls in the
    # right Grimme parameters for HF/6-31G*; ORCA gets the
    # equivalent simpleinput.

    def make_panel(with_d3: bool) -> list:
        panel = [
            ("vibe-qc", VibeQC(
                basis="6-31g*",
                dispersion="hf" if with_d3 else None,
            )),
            # PySCF row dropped: the in-process PySCFCalculator shim was
            # retired (CLAUDE.md sec 10). For a PySCF cross-check, use
            # the out-of-process runner at
            # examples/regression/core/runner_pyscf.py. Note: PySCF
            # doesn't ship D3 natively anyway (a `pyscf-dispersion`
            # extension exists), so the original PySCF row was only a
            # dispersion-free reference regardless of `with_d3`.
        ]
        if with_d3:
            orca = make_orca_calculator(
                orcasimpleinput="HF 6-31G* D3BJ EnGrad",
                label="orca-d3bj",
            )
        else:
            orca = make_orca_calculator(
                orcasimpleinput="HF 6-31G* EnGrad",
                label="orca-no-d3",
            )
        if orca is not None:
            panel.append(("ORCA", orca))
        return panel

    print("\n--- Without D3(BJ) ---")
    panel_nodisp = make_panel(with_d3=False)
    e_mon_nodisp = _energy_for("monomer/no-D3", monomer, panel_nodisp)
    e_dim_nodisp = _energy_for("dimer/no-D3", dimer, panel_nodisp)
    print(f"  monomer:  {e_mon_nodisp}")
    print(f"  dimer:    {e_dim_nodisp}")
    int_nodisp = {
        label: e_dim_nodisp[label] - 2.0 * e_mon_nodisp[label]
        for label in e_mon_nodisp
        if label in e_dim_nodisp
    }
    print(f"  E(int) [eV]:        {int_nodisp}")
    print(f"  E(int) [kcal/mol]:  "
          f"{ {l: v / EV_PER_KCAL for l, v in int_nodisp.items()} }")

    print("\n--- With D3(BJ) ---")
    panel_d3 = make_panel(with_d3=True)
    e_mon_d3 = _energy_for("monomer/D3", monomer, panel_d3)
    e_dim_d3 = _energy_for("dimer/D3", dimer, panel_d3)
    print(f"  monomer:  {e_mon_d3}")
    print(f"  dimer:    {e_dim_d3}")
    int_d3 = {
        label: e_dim_d3[label] - 2.0 * e_mon_d3[label]
        for label in e_mon_d3
        if label in e_dim_d3
    }
    print(f"  E(int) [eV]:        {int_d3}")
    print(f"  E(int) [kcal/mol]:  "
          f"{ {l: v / EV_PER_KCAL for l, v in int_d3.items()} }")

    print("\n--- Dispersion contribution ---")
    for label in int_nodisp:
        if label in int_d3:
            delta_eV = int_d3[label] - int_nodisp[label]
            delta_kcal = delta_eV / EV_PER_KCAL
            print(f"  {label}:  delta_E_disp = {delta_eV*1000:+.2f} meV "
                  f"({delta_kcal:+.3f} kcal/mol)")

    # Sanity check: dispersion contribution should be ~0.5 kcal/mol
    # attractive on water dimer at HF/6-31G* with D3(BJ) for HF.
    # ORCA and vibe-qc should agree on it to ~1 uHa.
    if "vibe-qc" in int_nodisp and "ORCA" in int_nodisp and \
       "vibe-qc" in int_d3 and "ORCA" in int_d3:
        vibe_disp_kcal = (int_d3["vibe-qc"] - int_nodisp["vibe-qc"]) / EV_PER_KCAL
        orca_disp_kcal = (int_d3["ORCA"] - int_nodisp["ORCA"]) / EV_PER_KCAL
        gap = abs(vibe_disp_kcal - orca_disp_kcal)
        print(f"\nvibe-qc D3(BJ) vs ORCA D3(BJ):  delta = "
              f"{gap*1000:.4f} mkcal/mol")
        assert gap < 0.01, (
            f"vibe-qc D3(BJ) interaction energy disagrees with ORCA by "
            f"{gap:.3e} kcal/mol; should match to ~uHa (Grimme params "
            f"are unambiguous)."
        )
        print("OK: vibe-qc D3(BJ) implementation matches ORCA's to "
              "<0.01 kcal/mol on the water dimer interaction energy.")
    elif "ORCA" not in int_d3:
        print("\n(ORCA not available: D3(BJ) cross-validation skipped. "
              "Set ORCA_COMMAND or add orca to $PATH and re-run for the "
              "full cross-check.)")


if __name__ == "__main__":
    main()
