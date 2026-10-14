"""Water dimer binding energy with the HF-3c and PBEh-3c composites, gCP shown.

System: the water dimer of examples/h2o_dimer.xyz (Cs, R(O-O) = 2.95
angstrom, linear hydrogen bond along z) and its two monomers cut out at the
dimer geometry, so the binding energy is a vertical interaction energy
without monomer relaxation. The geometry is read from that file when the
script sits in the examples tree and is embedded below as a fallback.
Methods: ``run_job(method="hf-3c")`` and ``run_job(method="pbeh-3c")``.
Each composite keyword resolves its own basis (MINIX for HF-3c, def2-mSVP
for PBEh-3c), functional, D3(BJ) damping, geometric counterpoise (gCP)
parameters and, for HF-3c, the short-range bond (SRB) correction.

What the returned object carries: ``result.e_scf`` is the bare SCF energy
and ``result.energy_total`` is SCF + D3(BJ) only. The gCP and SRB terms are
printed in the .out under "Composite total" but are not folded into
``energy_total``. This script rebuilds the full composite total with
``vibeqc.composites.composite_correction_terms`` (the same function the
runner's final single point and the optimiser use) and evaluates the gCP
term on its own with ``vibeqc.compute_gcp`` to show the basis-set
superposition correction it contributes to the binding energy.

Files (next to this script): one set per method and fragment,
    input-h2o-dimer-3c-<method>-{dimer,monoA,monoB}.out / .molden / .system / ...

Success (kcal/mol, negative = bound; measured contributions in brackets):
HF-3c binding is -5.33 at bare SCF, -6.48 with D3(BJ) [-1.15], and -5.16
for the full composite once gCP [+1.57] and SRB [-0.25] are added. PBEh-3c
binds by -6.69 at bare SCF, -6.92 with D3(BJ) [-0.24] and -4.40 for the
full composite after the damped gCP [+2.53]. The gCP term always opposes
binding here: it is the composite's built-in counterpoise correction for
the over-binding of the small basis, and it is the largest single change
in both recipes. The number from ``compute_gcp`` equals the gCP term of
``composite_correction_terms`` exactly. Tolerance: 0.01 kcal/mol on each
binding energy (vibe-qc 0.17.x development build, Apple M5 Max, about 50 s,
dominated by the PBEh-3c runs).
"""

from pathlib import Path

import vibeqc as vq
from vibeqc.composites import composite_correction_terms

HERE = Path(__file__).parent
STEM = Path(__file__).stem
ANG = 1.8897259886
KCAL = 627.5094740631

XYZ = HERE.parent / "h2o_dimer.xyz"
if XYZ.exists():
    dimer = vq.Molecule.from_xyz(XYZ)
else:  # the same six atoms, angstrom -> bohr
    dimer = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 0.960 * ANG]),
        vq.Atom(1, [0.0, 0.929 * ANG, -0.240 * ANG]),
        vq.Atom(8, [0.0, 0.0, 2.950 * ANG]),
        vq.Atom(1, [0.0, 0.759 * ANG, 3.538 * ANG]),
        vq.Atom(1, [0.0, -0.759 * ANG, 3.538 * ANG]),
    ])
atoms = list(dimer.atoms)
fragments = {
    "dimer": dimer,
    "monoA": vq.Molecule([vq.Atom(a.Z, list(a.xyz)) for a in atoms[:3]]),
    "monoB": vq.Molecule([vq.Atom(a.Z, list(a.xyz)) for a in atoms[3:]]),
}

GCP_INPUT = {                 # what each recipe hands to compute_gcp
    "hf-3c": dict(basis_name="minix"),
    "pbeh-3c": dict(basis_name="def2-msvp", damping=(4.0, 6.0)),
}

for method in ("hf-3c", "pbeh-3c"):
    recipe = vq.resolve_composite(method)
    energies = {}
    for label, mol in fragments.items():
        result = vq.run_job(mol, method=method, output=str(HERE / f"{STEM}-{method}-{label}"))
        terms = {name: e for name, e, _grad in composite_correction_terms(mol, recipe)}
        gcp = vq.compute_gcp(mol, **GCP_INPUT[method])
        energies[label] = {
            "scf": float(result.e_scf),
            "scf+d3": float(result.energy_total),
            "composite": float(result.e_scf) + sum(terms.values()),
            "gcp": float(gcp.energy),
            "gcp_recipe": terms.get("gCP", 0.0),
            "srb": terms.get("SRB", 0.0),
            "d3": terms.get("D3-BJ", 0.0),
        }

    def bind(key):
        return (energies["dimer"][key] - energies["monoA"][key] - energies["monoB"][key]) * KCAL

    print(f"\n  {method}: basis {recipe.basis}, functional {recipe.functional or 'HF'}, "
          f"gCP on {recipe.gcp_basis}"
          + (f" damped {recipe.gcp_damping.dmp_scal:g}/{recipe.gcp_damping.dmp_exp:g}"
             if recipe.gcp_damping else "")
          + (", SRB on" if recipe.sr_mod is not None else ""))
    print(f"    {'fragment':8s} {'E_SCF (Ha)':>16s} {'E_gCP (Ha)':>12s} {'E_SRB (Ha)':>12s} "
          f"{'E_D3 (Ha)':>12s} {'composite (Ha)':>16s}")
    for label, e in energies.items():
        print(f"    {label:8s} {e['scf']:16.8f} {e['gcp']:12.8f} {e['srb']:12.8f} "
              f"{e['d3']:12.8f} {e['composite']:16.8f}")
    print(f"    binding energy (kcal/mol): SCF {bind('scf'):+.3f}, SCF+D3 {bind('scf+d3'):+.3f}, "
          f"full composite {bind('composite'):+.3f}")
    print(f"    contributions to binding (kcal/mol): D3 {bind('d3'):+.3f}, "
          f"gCP {bind('gcp'):+.3f}, SRB {bind('srb'):+.3f}")
    print(f"    compute_gcp vs recipe gCP term on the dimer: "
          f"|difference| = {abs(energies['dimer']['gcp'] - energies['dimer']['gcp_recipe']):.1e} Ha")
