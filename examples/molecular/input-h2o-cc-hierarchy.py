"""Water, the coupled-cluster and coupled-pair hierarchy measured against CCSD(T).

System: H2O/STO-3G at the fixed geometry shared by these examples (bohr).
Methods: one ``run_job(method=...)`` call each for ccd, lccd, cepa(0),
cepa(1), qcisd, qcisd(t), cc2, ccsd, ccsd(t), bccd, bccd(t), cc3 and
ccsdt, all on the same RHF reference with the default chemical frozen core
(the O 1s orbital). STO-3G has no RI auxiliary basis, so the DF-CCSD kernel
is told ``density_fit=False`` and runs on exact four-index integrals; CC3
and CCSDT are dense iterative solvers and take no auxiliary basis anyway.
Both dense triples solvers finish in about 15 s each here, so they are in;
they grow steeply with basis size, and CC3 on H2O/6-31G already takes over
five minutes, which is why this example stays in STO-3G.

Every correlation energy in the table is E_total minus the single RHF
energy of the canonical reference. That matters for the two Brueckner rows:
their ``CCSDResult.e_hf`` field holds the energy of the Brueckner
determinant, not of RHF, so subtracting it would compare correlation
energies defined against different references.

Files (next to this script): one set per method,
    input-h2o-cc-hierarchy-<method>.out / .system / .xyz
with ``(``/``)`` in the method name replaced by ``_``. The Molden,
population and QVF sidecars are switched off to keep the directory
readable; drop those three keywords to get them back.

Success (E_total in Ha, difference to CCSD(T) in mHa): CCSD(T) has
E_corr = -0.055472 Ha and CCSDT -0.055501 Ha, so (T) recovers the
iterative triples to 0.03 mHa; CC3 matches CCSD(T) to 0.0001 mHa and
QCISD(T) and BCCD(T) to 0.004 mHa. The doubles-only and coupled-pair
models sit within 1 mHa: QCISD +0.06, CCSD +0.08, BCCD +0.09, CEPA(1) -0.22,
CCD +0.40, LCCD -0.54, CEPA(0) -0.91. The one clear outlier is CC2 at
+15.99 mHa (it recovers only 71 percent of the CCSD(T) correlation
energy): CC2 approximates the doubles equations to make excitation
energies cheap, and its ground-state energy is a by-product, not a target.
Numbers from a vibe-qc 0.17.x development build on Apple M5 Max; the whole
loop takes 65 to 160 s depending on machine load.
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])

METHODS = [
    "ccd", "lccd", "cepa(0)", "cepa(1)",
    "qcisd", "qcisd(t)", "cc2",
    "ccsd", "ccsd(t)", "bccd", "bccd(t)",
    "cc3", "ccsdt",
]
DENSE_TRIPLES = {"cc3", "ccsdt"}      # own solvers; no density-fitting switch


def file_tag(method: str) -> str:
    return method.replace("(", "_").replace(")", "")


totals = {}
e_rhf = None
for method in METHODS:
    kwargs = {} if method in DENSE_TRIPLES else {"density_fit": False}
    result = vq.run_job(
        mol,
        basis="sto-3g",
        method=method,
        output=str(HERE / f"{STEM}-{file_tag(method)}"),
        write_molden_file=False,
        write_population_file=False,
        output_qvf=False,
        **kwargs,
    )
    cc = getattr(result, "ccsd", None)
    if cc is not None:                      # CCSDResult wrapper (CC kernel variants)
        totals[method] = float(cc.e_total)
        if method == "ccsd":                # canonical RHF energy, not a Brueckner one
            e_rhf = float(cc.e_hf)
    else:                                   # CC3Result / CCSDTResult
        totals[method] = float(result.e_total)

e_ref = totals["ccsd(t)"]

print("\n  H2O/STO-3G, RHF reference, chemical frozen core; deltas relative to CCSD(T)")
print(f"  {'method':10s} {'E_total (Ha)':>16s} {'E_corr (Ha)':>14s} {'delta (mHa)':>12s}")
for method in METHODS:
    e_tot = totals[method]
    print(f"  {method:10s} {e_tot:16.8f} {e_tot - e_rhf:14.8f} "
          f"{(e_tot - e_ref) * 1000.0:12.4f}")
print(f"  {'E(RHF)':10s} {e_rhf:16.8f}")
print(f"  CC2 recovers {100.0 * (totals['cc2'] - e_rhf) / (e_ref - e_rhf):.1f} percent "
      "of the CCSD(T) correlation energy")
