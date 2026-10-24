"""Hydrogen iodide, RHF/def2-SVP with the def2 effective core potential on iodine.

System: HI at the experimental bond length, R(H-I) = 1.609 angstrom.
Method: ``run_job(method="rhf", basis="def2-svp")``. def2-SVP covers
iodine only together with its 28-electron def2-ECP (Kr core plus 4d), and
the bundled ``.ecp`` sidecar of the basis attaches it automatically: the 28
core electrons are removed, the iodine centre carries the effective nuclear
charge 25, and the semilocal projectors are integrated by libecpint. Nothing
ECP-specific has to be requested; the script checks what was applied with
``vibeqc.ecp_requirements("def2-svp", [1, 53])`` before the run and with the
``ecp_*`` fields of the RHFResult after it.

``localize=False`` is passed because the default localised-orbital (IBO)
section of the QVF archive needs an all-electron reference basis and cannot
be built for a valence-only ECP job; leaving it at the default still runs but
emits a ``UserWarning`` ("compatibility_fallback ... IAO analysis needs an
all-electron reference basis") and skips that QVF section.

Files (next to this script):
    input-hi-ecp.out       banner (with the libecpint line), RHF trace, orbitals, charges,
                           and a generic ECP citation set (eight entries, not iodine-specific)
    input-hi-ecp.molden    valence orbitals
    input-hi-ecp.system    provenance manifest (records the libecpint build)
    plus .xyz / .population.* / .bibtex / .references / .qvf sidecars

Success: E(RHF) = -297.231532 Ha with 26 valence electrons in 13 doubly
occupied orbitals (the .out header still prints the physical count
n_electrons=54; ``Molecule.n_electrons()`` is always the all-electron
number); ``ecp_requirements`` reports n_core = 28 for I with bundled data;
the result carries ``ecp_total_ncore = 28``, ``ecp_effective_charges =
[25, 1]``, ``ecp_operator_applied = True`` and ``ecp_provenance_verified =
True``. In this build the .out has no dedicated ECP block: the provenance
lives in those result fields, in the libecpint entry of the banner and
manifest, and in the generic pseudopotential citations of the reference list
(they do not identify the iodine potential), all of which the script
echoes. Tolerance: 1e-6 Ha (vibe-qc 0.17.x development build, Apple M5 Max,
10 to 45 s depending on machine load).
"""

import re
from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
ANG = 1.8897259886
BASIS = "def2-svp"

mol = vq.Molecule([
    vq.Atom(53, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, 0.0, 1.609 * ANG]),
])

print(f"\n  ECP requirements of {BASIS} for H and I:")
for req in vq.ecp_requirements(BASIS, [int(a.Z) for a in mol.atoms]):
    print(f"    {req.symbol} (Z={req.Z}): n_core={req.n_core}, bundled data={req.has_data}")

result = vq.run_job(mol, basis=BASIS, method="rhf", localize=False, output=str(HERE / STEM))

n_physical = mol.n_electrons()
n_core = int(result.ecp_total_ncore)
n_valence = n_physical - n_core
print(f"\n  E(RHF/{BASIS}, def2-ECP on I) = {result.energy:16.8f} Ha  "
      f"(converged={result.converged}, {result.n_iter} iterations)")
print(f"  electrons: {n_physical} physical, {n_core} in the ECP core, "
      f"{n_valence} treated explicitly ({n_valence // 2} doubly occupied orbitals)")
print(f"  effective nuclear charges (atom order): "
      f"{[float(q) for q in result.ecp_effective_charges]}")
print(f"  ECP operator applied: {result.ecp_operator_applied}; "
      f"provenance verified: {result.ecp_provenance_verified}")

out_text = (HERE / f"{STEM}.out").read_text(encoding="utf-8")
banner = next((ln.strip(" ║") for ln in out_text.splitlines() if "libecpint" in ln),
              "(no libecpint line)")
print(f"  .out banner: {banner}")
manifest = [ln for ln in (HERE / f"{STEM}.system").read_text(encoding="utf-8").splitlines() if "libecpint" in ln]
print(f"  .system manifest: {manifest[0].strip() if manifest else '(no libecpint entry)'}")

# The .out records ECPs only through its reference list, a generic ECP citation set
# that is not specific to iodine; print those entries.
refs = out_text.split("## References", 1)[-1]
entries = [" ".join(e.split()) for e in re.split(r"\n\s*\n", refs) if re.match(r"\s*\[\d+\]", e)]
ecp_refs = [e for e in entries if re.search(r"seudopotential|effective core", e, re.I)]
print(f"  .out reference list: {len(ecp_refs)} pseudopotential entries")
for entry in ecp_refs:
    print(f"    {entry[:110]}")
