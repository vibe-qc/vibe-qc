"""Emit self-contained vibe-qc input files from the structure database.

Reads the 39-compound structure database from
:mod:`vibe_basis.io.structures` and writes one ``.py`` per
(compound × basis × method) to ``examples/basisset_dev/inputs/``.
For every non-AFM cubic compound it also emits a CRYSTAL14 ``.d12``
parity sidecar via :func:`vibe_basis.backends.crystal.emit_input`.

Every emitted ``.py`` is **fully self-contained**: lattice vectors
and fractional coordinates are baked in, no runtime CIF /
Materials Project / ASE fetch.

Run once per structure-database update::

    .venv/bin/python examples/basisset_dev/_generator.py

Requires ``vibe-basis`` installed in the active venv (the test-set
database + CRYSTAL14 input emitter live there; see
``vibe-basis/README.md``). Output files are committed to the repo
so the periodic chat can pick them up directly when
REQUIREMENTS-PERIODIC R1-R5 land.

Phase scope: Γ-point RHF inputs for every cubic compound (Phases
1-3 = 39 compounds; AFM ones skip the .d12 step pending R3).
Reference HF energies from PT2013 SI Table 2 are listed in each
input's docstring as the acceptance target. Multi-k + PW1PW
variants will be added when R2 / R4 land.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vibe_basis.backends.crystal import emit_input as _emit_crystal
from vibe_basis.io.structures import STRUCTURES, Structure, StructureAtom

HERE = Path(__file__).resolve().parent

OUT_DIR = HERE / "inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Element symbol per Z, for human-readable comments in emitted files.
SYMBOL = {
    1: "H", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    19: "K", 20: "Ca", 35: "Br", 38: "Sr",
}


# ---------- Reference energies ----------------------------------------------
#
# PT2013 SI Table 2 — HF total energies per unit cell (Hartree).
# Listed for the 12 alkali / alkaline-earth halides + hydrides; KBr
# is from VO2019 (no HF table for that one in PT2013).
# Used to populate the docstring of each emitted input so a user can
# diff their SCF result against the published reference.

PT2013_T2_HF: dict[str, float] = {
    "LiCl": -467.087468,
    "NaCl": -621.495944,
    "LiF":  -107.055717,
    "NaF":  None,        # NaF SCF didn't converge in the paper; "-" in T2
    "KF":   -698.704515,
    "CaF2": -875.945290,
    "K2O":  -1273.185420,
    "MgO":  -274.681754,
    "CaO":  -751.806904,
    "LiH":   -8.062837,
    "NaH":  -162.453809,
    "KH":   -599.723256,
}  # type: ignore[dict-item]
# (The dict carries Optional[float]; the KBr / extras fall back to
# None below when looked up.)


# ---------- Emission --------------------------------------------------------


def _format_lattice(struct: Structure) -> str:
    """Render the 3×3 lattice matrix in **bohr** as a Python literal.

    vibe-qc's PeriodicSystem expects bohr; the structure database
    is in Å so the conversion happens here, once per emit.
    """
    A_TO_BOHR = 1.8897259886
    rows = struct.lattice_matrix_angstrom()
    out: list[str] = ["    lattice = np.array(["]
    for row in rows:
        scaled = [v * A_TO_BOHR for v in row]
        out.append(
            "        [{:.10f}, {:.10f}, {:.10f}],".format(*scaled)
        )
    out.append("    ])  # bohr")
    return "\n".join(out)


def _format_atoms(struct: Structure) -> str:
    """Render the unit-cell atoms as a Python literal.

    Cartesian coordinates are computed from fractional × lattice (in
    bohr). Each atom line carries a comment with the element symbol
    so the file is browsable without referring back to Z numbers.
    """
    A_TO_BOHR = 1.8897259886
    rows = struct.lattice_matrix_angstrom()
    lattice_bohr = [[v * A_TO_BOHR for v in row] for row in rows]

    def cart(fxyz: tuple[float, float, float]) -> tuple[float, float, float]:
        x = sum(fxyz[i] * lattice_bohr[i][0] for i in range(3))
        y = sum(fxyz[i] * lattice_bohr[i][1] for i in range(3))
        z = sum(fxyz[i] * lattice_bohr[i][2] for i in range(3))
        return (x, y, z)

    out: list[str] = ["    unit_cell = ["]
    for atom in struct.unit_cell:
        sym = SYMBOL.get(atom.Z, f"Z{atom.Z}")
        x, y, z = cart(atom.fxyz)
        out.append(
            "        vq.Atom({Z:>2d}, [{x:>14.10f}, {y:>14.10f}, {z:>14.10f}]),  # {sym} frac=({fx:.3f},{fy:.3f},{fz:.3f})".format(
                Z=atom.Z, x=x, y=y, z=z, sym=sym,
                fx=atom.fxyz[0], fy=atom.fxyz[1], fz=atom.fxyz[2],
            )
        )
    out.append("    ]")
    return "\n".join(out)


def _input_filename(struct: Structure, basis: str, method: str) -> str:
    """Canonical filename for an input file.

    Pattern: ``<compound>_<basis>_<method>.py``. Lower-case,
    ``-`` for separators inside the basis name, ``_`` for the
    triple separator. Underscore is intentional rather than ``/``
    so the file system layout stays flat.
    """
    return f"{struct.name.lower()}_{basis.lower()}_{method.lower()}.py"


def _docstring(struct: Structure, basis: str, method: str) -> str:
    """Generate the docstring for one input file.

    Captures: paper-table provenance, reference energy from PT2013
    SI Table 2 if available, structure source, blocker on R3 for
    AFM compounds, and acceptance criterion the user can apply
    when running the input.
    """
    ref_lines: list[str] = []
    if method == "rhf" and struct.name in PT2013_T2_HF:
        e_ref = PT2013_T2_HF.get(struct.name)
        if e_ref is None:
            ref_lines.append(
                "    Reference: PT2013 SI Table 2 lists this compound but "
                "the HF SCF did not converge in the paper (n-dash entry)."
            )
        else:
            ref_lines.append(
                f"    Reference: PT2013 SI Table 2 (HF) lists "
                f"E = {e_ref:.6f} Ha per unit cell with CRYSTAL standard "
                "basis at converged k-mesh. The pob-TZVP value in the "
                "same row is typically within ~10 mHa."
            )
    if struct.afm_pattern:
        ref_lines.append(
            f"    ANTIFERROMAGNETIC ({struct.afm_pattern}). R3's blocker, a "
            "broken-symmetry initial guess, shipped 2026-06-14/16 as "
            "ATOMSPIN (vq.PeriodicOptions.atomic_spins), validated against "
            "PySCF.pbc via "
            "examples/regression/parity_periodic_atomspin_vs_pyscf.py. "
            "The Γ-only RHF run below still converges to a non-physical "
            "spin-restricted solution because it doesn't use UKS + "
            "atomic_spins; treat this file's energy as a placeholder until "
            "it's updated to a physical AFM UKS run. The remaining gate for "
            "a production-size result is REQUIREMENTS-PERIODIC R2 "
            "(multi-k), not R3."
        )
    ref_lines.append(
        "    Acceptance gate (when REQUIREMENTS-PERIODIC R2 multi-k "
        "lands): SCF energy at converged Pack-Monkhorst grid agrees "
        "with the cited PT2013 reference to 1 mHa."
    )

    return f'''"""{struct.formula} — {method.upper()} on {basis} (Γ-only).

This input is generated by ``examples/basisset_dev/_generator.py``
and is intentionally self-contained: lattice vectors and fractional
coordinates are baked in, no external CIF / Materials Project / ASE
fetch at runtime. Edit ``vibe_basis.io.structures`` (in the
``vibe-basis`` package) and re-run the generator rather than
tweaking this file directly.

Compound metadata:
    formula:        {struct.formula}
    space group:    {struct.spacegroup}
    crystal system: {struct.crystal_system}
    a, b, c (Å):    {struct.a:.4f}, {struct.b:.4f}, {struct.c:.4f}
    α, β, γ (deg):  {struct.alpha:.2f}, {struct.beta:.2f}, {struct.gamma:.2f}
    n_atoms in cell: {len(struct.unit_cell)}
    multiplicity:   {struct.multiplicity}
    structure src:  {struct.structure_source}
    appears in:     {", ".join(struct.pob_tables)}

{chr(10).join(ref_lines)}

Today's status: Γ-point only because vibe-qc multi-k periodic SCF
(REQUIREMENTS-PERIODIC R2) is not yet wired for production-size
basis sets. The generator emits the multi-k variant alongside this
Γ-point version once R2 lands. The functional registry alias
``"PW1PW"`` (R4) is the second prerequisite for direct comparison
against PT2013 SI Table 1.
"""'''


def emit(struct: Structure, basis: str, method: str) -> Path:
    """Emit one input .py file. Returns the written path."""
    filename = _input_filename(struct, basis, method)
    docstring = _docstring(struct, basis, method)
    lattice_block = _format_lattice(struct)
    atoms_block = _format_atoms(struct)

    if method == "rhf":
        runner = "vq.run_rhf_periodic_gamma_scf"
        opts_class = "vq.PeriodicSCFOptions"
    else:
        raise NotImplementedError(
            f"method={method!r}: only 'rhf' is wired for the Phase-1 generator. "
            "Add PW1PW / B3LYP variants once R4 / hybrid-periodic lands."
        )

    path = OUT_DIR / filename
    text = f'''{docstring}

import numpy as np

import vibeqc as vq


def main() -> None:
{lattice_block}

{atoms_block}

    sysp = vq.PeriodicSystem(dim=3, lattice=lattice, unit_cell=unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "{basis}")

    opts = {opts_class}()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    # Cutoffs sized for the conventional cubic cell: lattice constant
    # plus a generous margin for Coulomb tail screening. Tighten if
    # the SCF doesn't converge cleanly.
    opts.lattice_opts.cutoff_bohr = max(15.0, 3.0 * float(np.linalg.norm(lattice[0])))
    opts.lattice_opts.nuclear_cutoff_bohr = opts.lattice_opts.cutoff_bohr * 1.5
    opts.max_iter = 60

    print(f"  cell:       {{lattice[0][0]:.4f}} bohr (= {struct.a:.4f} Å)")
    print(f"  basis:      {basis} ({{basis.nshells}} shells, {{basis.nbasis}} fns)")
    print(f"  electrons:  {{sysp.unit_cell_molecule().n_electrons()}} per cell")

    result = {runner}(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5, progress=True,
    )

    print()
    print(f"  E ({method.upper()}, Γ-only) = {{result.energy:.8f}} Ha per unit cell")
    print(f"  converged   = {{result.converged}}  ({{result.n_iter}} iters)")


if __name__ == "__main__":
    main()
'''
    path.write_text(text)
    return path


def emit_crystal_d12(struct: Structure, basis: str, method: str) -> Optional[Path]:
    """Emit a CRYSTAL14 ``.d12`` input as a parity sidecar.

    Thin wrapper: delegates deck-building to
    :func:`vibe_basis.backends.crystal.emit_input` (single source
    of truth for the .d12 format), then writes the returned text to
    ``examples/basisset_dev/inputs/<compound>_<basis>_<method>.d12``.

    Returns the written path, or ``None`` if the upstream emitter
    declined (AFM compound without R3, missing crystal-asymm data,
    or unrecognized method). See ``vibe_basis.backends.crystal``
    for the full decision tree.
    """
    text = _emit_crystal(struct, basis, method)
    if text is None:
        return None
    out_path = OUT_DIR / f"{struct.name.lower()}_{basis.lower()}_{method.lower()}.d12"
    out_path.write_text(text)
    return out_path


def emit_all() -> list[Path]:
    """Emit every input — vibe-qc .py + (where applicable) CRYSTAL .d12."""
    written: list[Path] = []
    for struct in STRUCTURES.values():
        # For Phase 1 we only emit RHF + pob-TZVP. Add more (basis,
        # method) tuples here as features land in vibe-qc.
        for basis in ("pob-tzvp",):
            for method in ("rhf",):
                written.append(emit(struct, basis, method))
                d12 = emit_crystal_d12(struct, basis, method)
                if d12 is not None:
                    written.append(d12)
    return written


def main() -> None:
    written = emit_all()
    n_py = sum(1 for p in written if p.suffix == ".py")
    n_d12 = sum(1 for p in written if p.suffix == ".d12")
    print(f"Emitted {len(written)} files to "
          f"{OUT_DIR.relative_to(HERE.parents[1])}/")
    print(f"  {n_py} vibe-qc .py inputs")
    print(f"  {n_d12} CRYSTAL14 .d12 parity sidecars")


if __name__ == "__main__":
    main()
