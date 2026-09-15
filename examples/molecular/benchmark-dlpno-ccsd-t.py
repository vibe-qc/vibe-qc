"""Retained pre-#140/#448 DLPNO-CCSD(T) validation evidence.

Measured with an explicitly pinned legacy recipe (def2-SVP, all-electron,
tcut_pno=1e-7, tcut_mkn=0, tcut_pairs=1e-4, residual_domain="pair",
triples_mode="t1"):

    mol    canon CCSD(T)    DLPNO error    recovery
    H2O      -76.177021     +0.083 kcal/mol   99.95 %
    NH3      -56.357944     -0.123 kcal/mol  100.01 %
    CH4      -40.360454     -0.670 kcal/mol  100.54 %
    HF      -100.141044     +0.128 kcal/mol   99.95 %
    CO      -112.955086     +0.006 kcal/mol  100.01 %
    N2      -109.180282     -0.727 kcal/mol  100.33 %
    H2CO    -114.125245     -0.843 kcal/mol  100.30 %
    -------------------------------------------------
    MAE = 0.37 kcal/mol, max = 0.84 kcal/mol -- all within chemical
    accuracy (< 1 kcal/mol).

The (T1) correction removes the ~0.1 kcal/mol (T0) semicanonical error.
Pair screening (tcut_pairs) treats the weakest pairs at MP2 level.  The
complete combination below is the archived protocol: although its
``tcut_pairs=1e-4`` coordinate equals NormalPNO, its ``tcut_pno=1e-7`` and
``tcut_mkn=0`` coordinates do not.

Run:
    .venv/bin/python examples/molecular/benchmark-dlpno-ccsd-t.py
"""

from vibeqc import BasisSet, CCSDOptions, RHFOptions, run_ccsd, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd

ANGSTROM = 1.8897259886
HARTREE_TO_KCAL = 627.5094740631

# Equilibrium-ish geometries (Angstrom) for a small CCSD(T)-tractable set.
MOLECULES = {
    "H2O": [(8, [0, 0, 0]), (1, [0, 0.757, 0.587]), (1, [0, -0.757, 0.587])],
    "NH3": [
        (7, [0, 0, 0.116]),
        (1, [0, 0.939, -0.271]),
        (1, [0.813, -0.469, -0.271]),
        (1, [-0.813, -0.469, -0.271]),
    ],
    "CH4": [
        (6, [0, 0, 0]),
        (1, [0.629, 0.629, 0.629]),
        (1, [-0.629, -0.629, 0.629]),
        (1, [-0.629, 0.629, -0.629]),
        (1, [0.629, -0.629, -0.629]),
    ],
    "HF": [(9, [0, 0, 0]), (1, [0, 0, 0.917])],
    "CO": [(6, [0, 0, 0]), (8, [0, 0, 1.128])],
    "N2": [(7, [0, 0, 0]), (7, [0, 0, 1.098])],
    "H2CO": [
        (6, [0, 0, 0]),
        (8, [0, 0, 1.208]),
        (1, [0, 0.943, -0.588]),
        (1, [0, -0.943, -0.588]),
    ],
}
BASIS, AUX = "def2-svp", "def2-svp-rifit"


def _molecule(atoms):
    return Molecule(
        [Atom(z, [c * ANGSTROM for c in p]) for z, p in atoms], charge=0, multiplicity=1
    )


def main():
    print(f"DLPNO-CCSD(T) vs canonical CCSD(T)  /{BASIS}, legacy pinned recipe")
    print(f"  {'mol':<6}{'canon CCSD(T)':>15}{'DLPNO err':>14}{'recovery':>11}")
    abs_errors = []
    for name, atoms in MOLECULES.items():
        mol = _molecule(atoms)
        basis = BasisSet(mol, BASIS)
        rhf_opts = RHFOptions()
        rhf_opts.max_iter = 100
        rhf_opts.conv_tol_energy = 1e-10
        rhf = run_rhf(mol, basis, rhf_opts)

        cc_opts = CCSDOptions()
        cc_opts.n_frozen_core = 0
        cc_opts.compute_triples = True
        cc_opts.density_fit = True
        cc_opts.aux_basis = AUX
        canon = run_ccsd(mol, basis, rhf, cc_opts)
        e_canon = canon.e_ccsd_correlation + canon.e_t

        df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
        dlpno = run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            LocalCCSDOptions(
                n_frozen=0,
                tcut_pno=1e-7,
                tcut_mkn=0.0,
                tcut_pairs=1e-4,
                residual_domain="pair",
                triples_mode="t1",
                compute_triples=True,
                max_nbf=400,
                # Retained evidence: both defaults that moved after it was
                # recorded are pinned back -- the pair density (#65) and the
                # MP2 PNO-truncation correction (#222).
                pno_norm="legacy",
                pno_correction=False,
            ),
        )
        e_dlpno = dlpno.e_corr + dlpno.e_t

        err = e_dlpno - e_canon
        abs_errors.append(abs(err) * HARTREE_TO_KCAL)
        print(
            f"  {name:<6}{canon.e_ccsd_t:>15.6f}"
            f"{err * HARTREE_TO_KCAL:>+10.3f} kcal{100 * e_dlpno / e_canon:>9.2f} %"
        )

    mae = sum(abs_errors) / len(abs_errors)
    print(f"  {'-' * 44}")
    print(
        f"  MAE = {mae:.3f} kcal/mol,  max = {max(abs_errors):.3f} kcal/mol"
        f"  ({'PASS' if max(abs_errors) < 1.0 else 'CHECK'}: chemical accuracy < 1)"
    )


if __name__ == "__main__":
    main()
