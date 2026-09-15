"""Canonical orthogonalisation on a tight H2 with a diffuse basis.

Run:
    .venv/bin/python input-h2-tight-canonical-orth.py

Produces:
    output-h2-tight-canonical-orth.out     — default-path SCF trace
    output-h2-tight-canonical-orth.molden  — projected MO basis
    output-h2-tight-canonical-orth.qvf     — vibe-view archive
    output-h2-tight-canonical-orth.system  — runtime/output manifest

Demonstrates how vibe-qc now handles near-linearly-dependent AO bases
— the situation where two atoms sit close enough that their diffuse
primitives start overlapping with near-unit magnitude, pushing the
overlap matrix S toward singularity. Before the canonical-
orthogonalisation path landed, the RHF driver would abort with
"RHF: AO basis is linearly dependent" on this geometry; with it,
the calculation converges cleanly and the energy is stable across a
wide range of threshold choices.

Two H atoms at 0.5 bohr with aug-cc-pVTZ (46 basis functions) is the
stress-test case from the commit message of the canonical-orth work
— artificially short to keep the diagnostic dramatic.

What gets shown:

1. The pre-flight diagnostic: ``vq.check_linear_dependence(basis)``
   reports the min eigenvalue of S, condition number, and the basis
   functions that dominate the near-null space.
2. A default ``run_rhf`` call converges without any extra plumbing.
3. A sweep over ``RHFOptions.linear_dep_threshold`` from 1e-9 to 1e-3
   shows the canonical-orthogonalisation projection kicking in — the
   kept-MO count drops while the total energy stays stable to ~µHa.
"""

import vibeqc as vq


def banner(title: str) -> None:
    bar = "-" * 68
    print(f"\n{bar}\n  {title}\n{bar}")


mol = vq.Molecule([
    vq.Atom(1, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, 0.0, 0.5]),           # 0.5 bohr — artificially tight
])
basis = vq.BasisSet(mol, "aug-cc-pvtz")

banner(f"H2 @ R=0.5 bohr, aug-cc-pVTZ — n_basis = {basis.nbasis}")

# ---- 1. Pre-flight linear-dependence diagnostic ------------------------
banner("Pre-flight linear-dependence check")
report = vq.check_linear_dependence(basis)
print(vq.format_linear_dependence_report(report, max_eigvals_shown=6))

# ---- 2. Default-path SCF -----------------------------------------------
banner("Default-path SCF (linear_dep_threshold = 1e-7)")
result = vq.run_job(
    mol,
    basis="aug-cc-pvtz",
    method="rhf",
    output="output-h2-tight-canonical-orth",
    output_qvf=True,
)
n_bf, n_kept = result.mo_coeffs.shape
print(f"  converged    = {result.converged}")
print(f"  iterations   = {result.n_iter}")
print(f"  E(RHF)       = {result.energy:.10f} Ha")
print(f"  MO basis     = ({n_bf}, {n_kept})  "
      f"[dropped {n_bf - n_kept} near-null directions]")

# ---- 3. Threshold sweep ------------------------------------------------
banner("Threshold sweep")
print(f"  {'threshold':>10s}  {'kept':>5s}  {'iters':>5s}  {'E (Ha)':>16s}  {'ΔE (µHa)':>10s}")
ref_energy = None
for thr in (1e-9, 1e-7, 1e-5, 1e-3):
    opts = vq.RHFOptions()
    opts.linear_dep_threshold = thr
    r = vq.run_rhf(mol, basis, opts)
    n_kept = r.mo_coeffs.shape[1]
    if ref_energy is None:
        ref_energy = r.energy
    de_uha = (r.energy - ref_energy) * 1e6
    print(f"  {thr:10.0e}  {n_kept:5d}  {r.n_iter:5d}  "
          f"{r.energy:16.10f}  {de_uha:+10.3f}")

banner("Takeaway")
print("  Raising the threshold drops progressively more near-null directions;")
print("  E stays stable to ~1 µHa across three decades of threshold choice.")
print("  On well-conditioned bases the default 1e-7 never projects — the")
print("  machinery only activates where it's genuinely needed.")
