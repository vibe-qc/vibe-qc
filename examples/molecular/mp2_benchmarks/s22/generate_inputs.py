"""Emit vibe-qc + ORCA inputs for every (S22 system × body × MP2 variant).

Layout produced::

    inputs/
      vibeqc/
        s22-NN-<slug>/
          dimer__mp2.py        dimer__rimp2.py        dimer__scsmp2.py
          dimer__sosmp2.py     dimer__b2plyp.py
          monoA__rimp2.py      monoB__rimp2.py        (CP-style ghost monomers
                                                       — only RI variants for cost)
      orca/
        s22-NN-<slug>/
          dimer__mp2.inp       dimer__rimp2.inp       dimer__scsmp2.inp
          dimer__sosmp2.inp    dimer__b2plyp.inp
          monoA__rimp2.inp     monoB__rimp2.inp

"Body" semantics:
  * ``dimer``  — full dimer geometry, dimer basis.
  * ``monoA``  — monomer A atoms with monomer-only basis. (Single-body
                 reference for the monomer correlation energy. We do
                 NOT emit ghost-atom CP corrections in this generator
                 — that's a Phase-C add-on if interaction energies
                 need to land within Jurečka 2006 reference to <0.1
                 kcal/mol. See the locally generated REPORT.md.)
  * ``monoB``  — same for monomer B.

Variant → method mapping::

    mp2      canonical RMP2 (no DF)                                 vq.run_mp2 / ORCA ! MP2
    rimp2    RI-RMP2 (cc-pVTZ/C auxiliary)                          vq.run_mp2(density_fit=True) / ORCA ! RI-MP2 cc-pVTZ/C
    scsmp2   SCS-RI-MP2 (Grimme 2003 c_os=6/5, c_ss=1/3)            vq.run_scs_mp2 / ORCA ! RI-SCS-MP2 cc-pVTZ/C
    sosmp2   SOS-RI-MP2 (Jung 2004 c_os=1.3, c_ss=0)                vq.run_sos_mp2 / ORCA ! RI-SOS-MP2 cc-pVTZ/C
    b2plyp   B2PLYP (Grimme 2006, RI-J/K + RI-MP2 correction)       vq.run_b2plyp / ORCA ! B2PLYP RIJK def2/JK cc-pVTZ/C

All variants share the converger profile in ``converger.py``.
Every MP2-bearing route is pinned to the historical all-electron protocol on
both sides.  This is benchmark evidence, not an assertion about vibe-qc's
current unqualified frozen-core default.

Cost-aware variant selection:
  * For systems with more than ``CANONICAL_MP2_MAX_ATOMS`` atoms in
    the dimer body, the ``mp2`` (canonical, no-DF) variant is skipped
    — it scales O(N^5) and would take hours on cc-pVTZ for the larger
    S22 dimers. We still get to compare vibe-qc ↔ ORCA at the RI-MP2
    level for those systems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from converger import (
    CONV_TOL_ENERGY,
    CONV_TOL_GRAD,
    DIIS_SUBSPACE,
    LINEAR_DEP_THRESHOLD,
    MAX_ITER,
    ORCA_SCF_BLOCK,
    ORCA_SIMPLE_TOLS,
    vibeqc_options_snippet,
)

HERE = Path(__file__).resolve().parent
GEOM_DIR = HERE / "geometries"
INPUTS_DIR = HERE / "inputs"

ORBITAL_BASIS = "cc-pvtz"
ORBITAL_BASIS_ORCA = "cc-pVTZ"
RI_AUX_VQ = "cc-pvtz-ri"               # MP2/correlation auxiliary
RI_AUX_ORCA = "cc-pVTZ/C"              # same Gaussians
JK_AUX_VQ = "def2-universal-jkfit"     # B2PLYP RKS-SCF DF aux
JK_AUX_ORCA = "def2/JK"                # same Gaussians

VARIANTS = ("mp2", "rimp2", "scsmp2", "sosmp2", "b2plyp")
BODIES = ("dimer", "monoA", "monoB")

# Skip canonical (4-index) MP2 above this atom count. cc-pVTZ on a
# 24-atom system is ~600 basis functions; canonical MP2 = O(N^5) is
# minutes-to-hours per single-point.
CANONICAL_MP2_MAX_ATOMS = 14


def read_xyz(path: Path) -> tuple[list[tuple[str, float, float, float]], str]:
    """Return (atoms, comment) where atoms = [(symbol, x, y, z), ...]."""
    lines = path.read_text().splitlines()
    n = int(lines[0])
    comment = lines[1]
    atoms = []
    for ln in lines[2 : 2 + n]:
        sym, x, y, z = ln.split()
        atoms.append((sym, float(x), float(y), float(z)))
    return atoms, comment


SYM_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18,
}


def s22_xyz_path(system_stem: str, body: str) -> Path:
    """system_stem looks like 's22-02-water-dimer'."""
    if body == "dimer":
        return GEOM_DIR / f"{system_stem}.xyz"
    return GEOM_DIR / f"{system_stem}-{body}.xyz"


def s22_systems() -> list[str]:
    """Stems of all 22 S22 systems, sorted by index."""
    return sorted(
        p.stem for p in GEOM_DIR.glob("s22-*.xyz")
        if "mono" not in p.stem
    )


# ---------------------------------------------------------------------
# vibe-qc input writer
# ---------------------------------------------------------------------

def _vibeqc_atoms_block(atoms, indent: str = "    ") -> str:
    """Emit a list-of-Atom python literal for the Molecule constructor.
    All coordinates passed in Angstrom — converted inside the script
    using vq.ANGSTROM constant pattern (we just multiply in-line)."""
    lines = []
    for sym, x, y, z in atoms:
        z_int = SYM_TO_Z[sym]
        # Angstrom → Bohr (1 Å = 1.8897259886 bohr); use the conversion
        # in-line for clarity. Same factor as vq.ANGSTROM_TO_BOHR but
        # explicit so the input is self-contained.
        bx, by, bz = x * 1.8897259886, y * 1.8897259886, z * 1.8897259886
        lines.append(f"{indent}vq.Atom({z_int:>2d}, [{bx:>14.8f}, {by:>14.8f}, {bz:>14.8f}]),  # {sym}")
    return "\n".join(lines)


VIBEQC_HEADER = '''"""S22-{idx:02d} {slug} — body={body}, variant={variant}.

Reference: Jurečka, Šponer, Černý, Hobza, PCCP 8, 1985 (2006).
Generated by: examples/molecular/mp2_benchmarks/s22/generate_inputs.py

Method:   {method}
Basis:    {orbital_basis}
{aux_line}Core:     all-electron (explicit historical benchmark convention)
Converger:    plain DIIS (matched to ORCA NoSOSCF NoTRAH SCFCONV8)
"""

from pathlib import Path
import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

mol = vq.Molecule(
    [
{atoms}
    ],
    charge=0,
    multiplicity=1,
)

basis = vq.BasisSet(mol, "{orbital_basis}")
'''


def write_vibeqc_input(system_stem: str, body: str, variant: str, out_dir: Path) -> Path:
    """Generate one vibe-qc input file."""
    atoms, _ = read_xyz(s22_xyz_path(system_stem, body))
    idx = int(system_stem.split("-")[1])
    slug = "-".join(system_stem.split("-")[2:])

    atoms_block = _vibeqc_atoms_block(atoms)
    rhf_opts_snippet = vibeqc_options_snippet("rhf_opts").rstrip()

    if variant == "mp2":
        method = "canonical RMP2"
        aux_line = ""
        body_text = f'''
rhf_opts = vq.RHFOptions()
{rhf_opts_snippet}
hf = vq.run_rhf(mol, basis, rhf_opts)
assert hf.converged, f"RHF did not converge: {{hf}}"

mp2_opts = vq.MP2Options()
mp2_opts.n_frozen_core = 0
mp2_opts.density_fit = False
mp2 = vq.run_mp2(mol, basis, hf, mp2_opts)

result = {{
    "system": "{system_stem}",
    "body": "{body}",
    "variant": "mp2",
    "e_hf": hf.energy,
    "e_correlation": mp2.e_correlation,
    "e_total": mp2.e_total,
    "e_os": mp2.e_os,
    "e_ss": mp2.e_ss,
    "n_frozen_core": mp2.n_frozen_core,
    "n_scf_iter": hf.n_iter,
    "scf_converged": hf.converged,
}}
'''
    elif variant == "rimp2":
        method = "RI-RMP2"
        aux_line = f"Aux:      {RI_AUX_VQ} (correlation fitting)\n"
        body_text = f'''
rhf_opts = vq.RHFOptions()
{rhf_opts_snippet}
hf = vq.run_rhf(mol, basis, rhf_opts)
assert hf.converged, f"RHF did not converge: {{hf}}"

mp2_opts = vq.MP2Options()
mp2_opts.n_frozen_core = 0
mp2_opts.density_fit = True
mp2_opts.aux_basis = "{RI_AUX_VQ}"
mp2 = vq.run_mp2(mol, basis, hf, mp2_opts)

result = {{
    "system": "{system_stem}",
    "body": "{body}",
    "variant": "rimp2",
    "e_hf": hf.energy,
    "e_correlation": mp2.e_correlation,
    "e_total": mp2.e_total,
    "e_os": mp2.e_os,
    "e_ss": mp2.e_ss,
    "n_frozen_core": mp2.n_frozen_core,
    "n_scf_iter": hf.n_iter,
    "scf_converged": hf.converged,
}}
'''
    elif variant == "scsmp2":
        method = "SCS-RI-RMP2 (Grimme 2003: c_os=6/5, c_ss=1/3)"
        aux_line = f"Aux:      {RI_AUX_VQ} (correlation fitting)\n"
        body_text = f'''
rhf_opts = vq.RHFOptions()
{rhf_opts_snippet}
hf = vq.run_rhf(mol, basis, rhf_opts)
assert hf.converged, f"RHF did not converge: {{hf}}"

mp2 = vq.run_scs_mp2(mol, basis, hf,
                     density_fit=True,
                     aux_basis="{RI_AUX_VQ}",
                     frozen_core=False)

result = {{
    "system": "{system_stem}",
    "body": "{body}",
    "variant": "scsmp2",
    "e_hf": hf.energy,
    "e_correlation": mp2.e_correlation,
    "e_total": mp2.e_total,
    "e_os": mp2.e_os,
    "e_ss": mp2.e_ss,
    "n_frozen_core": mp2.n_frozen_core,
    "n_scf_iter": hf.n_iter,
    "scf_converged": hf.converged,
}}
'''
    elif variant == "sosmp2":
        method = "SOS-RI-RMP2 (Jung 2004: c_os=1.3, c_ss=0)"
        aux_line = f"Aux:      {RI_AUX_VQ} (correlation fitting)\n"
        body_text = f'''
rhf_opts = vq.RHFOptions()
{rhf_opts_snippet}
hf = vq.run_rhf(mol, basis, rhf_opts)
assert hf.converged, f"RHF did not converge: {{hf}}"

mp2 = vq.run_sos_mp2(mol, basis, hf,
                     density_fit=True,
                     aux_basis="{RI_AUX_VQ}",
                     frozen_core=False)

result = {{
    "system": "{system_stem}",
    "body": "{body}",
    "variant": "sosmp2",
    "e_hf": hf.energy,
    "e_correlation": mp2.e_correlation,
    "e_total": mp2.e_total,
    "e_os": mp2.e_os,
    "e_ss": mp2.e_ss,
    "n_frozen_core": mp2.n_frozen_core,
    "n_scf_iter": hf.n_iter,
    "scf_converged": hf.converged,
}}
'''
    elif variant == "b2plyp":
        method = "B2PLYP (Grimme 2006 double hybrid)"
        aux_line = f"Aux:      {JK_AUX_VQ} (RKS DF) + {RI_AUX_VQ} (MP2 correction)\n"
        rks_opts_snippet = vibeqc_options_snippet("rks_opts").rstrip()
        body_text = f'''
rks_opts = vq.RKSOptions()
{rks_opts_snippet}
db = vq.run_b2plyp(mol, basis,
                   density_fit=True,
                   aux_basis="{JK_AUX_VQ}",
                   density_fit_mp2=True,
                   aux_basis_mp2="{RI_AUX_VQ}",
                   rks_options=rks_opts)

result = {{
    "system": "{system_stem}",
    "body": "{body}",
    "variant": "b2plyp",
    "e_hf": db.rks.energy,             # RKS-hybrid energy (NOT pure HF)
    "e_correlation": db.mp2.e_correlation,
    "e_total": db.e_total,
    "e_os": db.mp2.e_os,
    "e_ss": db.mp2.e_ss,
    "n_frozen_core": db.mp2.n_frozen_core,
    "n_scf_iter": db.rks.n_iter,
    "scf_converged": db.rks.converged,
}}
'''
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    body_text += '''
out_path = (HERE / STEM).with_suffix(".result")
out_path.write_text(repr(result) + "\\n")
print(repr(result))
'''

    text = (
        VIBEQC_HEADER.format(
            idx=idx,
            slug=slug,
            body=body,
            variant=variant,
            method=method,
            orbital_basis=ORBITAL_BASIS,
            aux_line=aux_line,
            atoms=atoms_block,
        )
        + body_text
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{body}__{variant}.py"
    out_path.write_text(text)
    return out_path


# ---------------------------------------------------------------------
# ORCA input writer
# ---------------------------------------------------------------------

def _orca_simple_input(variant: str) -> str:
    """The ! line for a given variant.

    Note 1: ORCA's MaxCore is set in the %maxcore block by the per-host
    wrapper (run-orca.sh). We omit it here so the per-host job
    description can decide based on cgroup memory budget.

    Note 2: ``NoFrozenCore`` is included on every MP2-bearing variant.
    ORCA and vibe-qc now both have published frozen-core defaults, but this
    historical benchmark contract deliberately compares all-electron
    correlation.  The generated vibe-qc decks pin ``n_frozen_core=0`` or
    ``frozen_core=False`` and the ORCA decks pin both ``NoFrozenCore`` and
    ``FrozenCore FC_NONE``.  No archived result is silently reinterpreted
    when either program changes an unqualified default.
    """
    # ORCA_SIMPLE_TOLS already pins NoSOSCF/NoTRAH/SCFCONV8/
    # NoFrozenCore/NoAutoStart from converger.py (the single source
    # of truth for the matched-converger profile). Don't re-pin here.
    base = f"{ORBITAL_BASIS_ORCA} {ORCA_SIMPLE_TOLS}"
    if variant == "mp2":
        return f"! MP2 {base}"
    if variant == "rimp2":
        return f"! RI-MP2 {RI_AUX_ORCA} {base}"
    if variant == "scsmp2":
        # Bare ``! SCS-MP2`` runs *canonical* SCS-MP2 in ORCA; the
        # ``cc-pVTZ/C`` aux is silently ignored without an explicit
        # ``! RI-MP2`` directive. Pair them so this matches the vibe-qc
        # side (vq.run_scs_mp2 uses density_fit=True).
        return f"! RI-MP2 SCS-MP2 {RI_AUX_ORCA} {base}"
    if variant == "sosmp2":
        # Same gotcha as scsmp2 above — keep the two ! lines aligned.
        return f"! RI-MP2 SOS-MP2 {RI_AUX_ORCA} {base}"
    if variant == "b2plyp":
        # RIJK = RI for both J and K (B2PLYP has 53% HF exchange).
        # def2/JK aux for the SCF, cc-pVTZ/C aux for the MP2 step.
        # NoFrozenCore applies to the MP2 correction step inside B2PLYP.
        return f"! B2PLYP RIJK {JK_AUX_ORCA} {RI_AUX_ORCA} {base}"
    raise ValueError(f"unknown variant: {variant!r}")


def write_orca_input(system_stem: str, body: str, variant: str, out_dir: Path,
                     nprocs: int = 1) -> Path:
    atoms, _ = read_xyz(s22_xyz_path(system_stem, body))
    simple = _orca_simple_input(variant)

    lines = [
        simple,
        ORCA_SCF_BLOCK.rstrip(),
    ]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.append("*xyz 0 1")
    for sym, x, y, z in atoms:
        lines.append(f"  {sym:<2s} {x:>15.8f} {y:>15.8f} {z:>15.8f}")
    lines.append("*")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{body}__{variant}.inp"
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------
# Top-level: walk the matrix.
# ---------------------------------------------------------------------

def variants_for(body_atom_count: int) -> Iterable[str]:
    """Skip canonical MP2 above CANONICAL_MP2_MAX_ATOMS atoms."""
    for v in VARIANTS:
        if v == "mp2" and body_atom_count > CANONICAL_MP2_MAX_ATOMS:
            continue
        yield v


def generate_all(systems: list[str] | None = None,
                 nprocs_orca: int = 1) -> dict[str, int]:
    """Generate inputs for every (system, body, variant) combination.

    Returns counts: {'vibeqc_inputs': N, 'orca_inputs': N, 'skipped_mp2': K}.
    """
    systems = systems or s22_systems()
    counts = {"vibeqc_inputs": 0, "orca_inputs": 0, "skipped_mp2": 0}
    for stem in systems:
        for body in BODIES:
            atoms, _ = read_xyz(s22_xyz_path(stem, body))
            for variant in VARIANTS:
                if variant == "mp2" and len(atoms) > CANONICAL_MP2_MAX_ATOMS:
                    counts["skipped_mp2"] += 1
                    continue
                vq_dir = INPUTS_DIR / "vibeqc" / stem
                orca_dir = INPUTS_DIR / "orca" / stem
                write_vibeqc_input(stem, body, variant, vq_dir)
                write_orca_input(stem, body, variant, orca_dir,
                                 nprocs=nprocs_orca)
                counts["vibeqc_inputs"] += 1
                counts["orca_inputs"] += 1
    return counts


if __name__ == "__main__":
    counts = generate_all(nprocs_orca=1)
    print(f"  vibe-qc inputs: {counts['vibeqc_inputs']}")
    print(f"  ORCA inputs:    {counts['orca_inputs']}")
    print(f"  Canonical MP2 skipped (>{CANONICAL_MP2_MAX_ATOMS} atoms): {counts['skipped_mp2']}")
