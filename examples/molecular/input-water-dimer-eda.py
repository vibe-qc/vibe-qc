"""Water dimer: counterpoise interaction energy and its LMO-EDA / Kitaura-Morokuma partition.

Status: PREVIEW for ``vibeqc.eda.eda_lmo`` and ``vibeqc.eda.eda_morokuma``.
They return real components for two closed-shell Hartree-Fock fragments
computed in the dimer basis, and the terms sum to the interaction energy.
Kohn-Sham, ECP, open-shell and other-basis inputs raise ``ValueError``; the
Kohn-Sham form, open-shell fragments and a ``run_job`` keyword do not exist
yet, so this script assembles the three SCF runs itself.

System: the water dimer of examples/h2o_dimer.xyz (Cs, R(O-O) = 2.95
angstrom; read from that file when the script sits in the examples tree,
embedded below otherwise), donor = atoms 1-3, acceptor = atoms 4-6.
Method: RHF/6-31G* on the dimer and on each monomer in the dimer basis.
vibe-qc has no ghost-atom keyword, but a ``Molecule`` accepts atoms with
Z = 0, and the explicit-shell constructor ``BasisSet(molecule, shells,
name=...)`` puts the dimer's shells on them; the only change is
``InitialGuess.HCORE``, because the default SAD guess has no atomic density
for Z = 0. Monomers computed this way carry the other fragment's basis
functions, which is the counterpoise correction (Boys and Bernardi 1970).

The partition follows one path: the isolated fragments, the promolecule
(P_A + P_B), the frozen state (one determinant of the mutually
Loewdin-orthogonalised occupied orbitals) and the relaxed dimer. LMO-EDA (Su
and Li 2009) reports electrostatic, exchange, repulsion and polarisation plus
charge transfer; Kitaura-Morokuma (1971, 1976) reports electrostatic, the
exchange repulsion EX (the sum of the LMO-EDA exchange and repulsion) and one
lumped polarisation, charge-transfer and coupling term. The script also
rebuilds the electrostatic term and the frozen-state energy from the
integrals, as an independent check of the library numbers.

Files (next to this script):
    input-water-dimer-eda.out    energies, the interaction energy with and without
                                 counterpoise, both partitions and the integral cross-check

Success: counterpoise-corrected Delta E_int = -4.13 kcal/mol (-4.86 without
it, so a BSSE of 0.73 kcal/mol). LMO-EDA: electrostatic -7.88, exchange
-7.13, repulsion +12.50, polarisation plus charge transfer -1.62 kcal/mol.
Kitaura-Morokuma: electrostatic -7.88, EX +5.37, polarisation plus charge
transfer -1.62 kcal/mol. Both sets sum to Delta E_int, and the integral
cross-check reproduces the electrostatic term and EX. The sign pattern is the
textbook hydrogen bond at Hartree-Fock level: electrostatics nearly twice the
binding, Pauli repulsion cancelling most of it, relaxation adding the rest.
The split depends on the chosen intermediate states, so compare terms only
within one basis. RHF has no dispersion. Tolerance: 0.01 kcal/mol on the
counterpoise energy and each term (about 2 s on a vibe-qc 0.17.x development
build, Apple M5 Max).
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.eda import eda_lmo, eda_morokuma

HERE = Path(__file__).parent
STEM = Path(__file__).stem
ANG = 1.8897259886
KCAL = 627.5094740631
BASIS = "6-31g*"

XYZ = HERE.parent / "h2o_dimer.xyz"
if XYZ.exists():
    real_atoms = [(int(a.Z), list(a.xyz)) for a in vq.Molecule.from_xyz(XYZ).atoms]
else:                                                     # the same six atoms, angstrom -> bohr
    real_atoms = [(z, [c * ANG for c in p]) for z, p in [
        (8, [0.000, 0.000, 0.000]), (1, [0.000, 0.000, 0.960]), (1, [0.000, 0.929, -0.240]),
        (8, [0.000, 0.000, 2.950]), (1, [0.000, 0.759, 3.538]), (1, [0.000, -0.759, 3.538]),
    ]]
DONOR, ACCEPTOR = [0, 1, 2], [3, 4, 5]


def molecule(real):
    """The dimer geometry with every atom outside `real` turned into a ghost (Z = 0)."""
    return vq.Molecule([vq.Atom(z if i in real else 0, list(p))
                        for i, (z, p) in enumerate(real_atoms)], 0, 1)


dimer = molecule(DONOR + ACCEPTOR)
basis = vq.BasisSet(dimer, BASIS)
shells = basis.shells()                                   # every shell remembers its own centre


def scf(mol):
    opts = vq.RHFOptions()
    if any(a.Z == 0 for a in mol.atoms):
        opts.initial_guess = vq.InitialGuess.HCORE        # SAD has no atomic density for Z = 0
        b = vq.BasisSet(mol, shells, name=BASIS)          # the full dimer basis on the monomer
    else:
        b = basis
    res = vq.run_rhf(mol, b, opts)
    assert res.converged
    return res


def free_monomer_energy(indices):
    """Monomer energy in its own basis only (no ghost functions), for the BSSE estimate."""
    sub = vq.Molecule([vq.Atom(real_atoms[i][0], list(real_atoms[i][1])) for i in indices], 0, 1)
    return float(vq.run_rhf(sub, BASIS).energy)


r_ab, r_a, r_b = scf(dimer), scf(molecule(DONOR)), scf(molecule(ACCEPTOR))
e_ab, e_a, e_b = float(r_ab.energy), float(r_a.energy), float(r_b.energy)
e_int = e_ab - e_a - e_b
e_int_uncorrected = e_ab - free_monomer_energy(DONOR) - free_monomer_energy(ACCEPTOR)

# Independent cross-check from the integrals (the same block is in the tutorial).
S = np.asarray(vq.compute_overlap(basis))
T = np.asarray(vq.compute_kinetic(basis))
V_A = np.asarray(vq.compute_nuclear(basis, molecule(DONOR)))      # attraction to A's nuclei only
V_B = np.asarray(vq.compute_nuclear(basis, molecule(ACCEPTOR)))
eri = np.asarray(vq.compute_eri(basis))                           # (36, 36, 36, 36)


def J(P):
    return np.einsum("pqrs,rs->pq", eri, P, optimize=True)


def K(P):
    return np.einsum("prqs,rs->pq", eri, P, optimize=True)


def nuclear_repulsion(mol):
    nuc = [(a.Z, np.asarray(a.xyz)) for a in mol.atoms if a.Z > 0]
    return sum(zi * zj / np.linalg.norm(ri - rj)
               for k, (zi, ri) in enumerate(nuc) for zj, rj in nuc[k + 1:])


def hf_energy(P, mol):
    """RHF energy expression of an arbitrary (not necessarily self-consistent) density."""
    return float(np.trace(P @ (T + V_A + V_B)) + 0.5 * np.trace(P @ (J(P) - 0.5 * K(P)))
                 + nuclear_repulsion(mol))


P_a, P_b = np.asarray(r_a.density), np.asarray(r_b.density)

# 1. Electrostatics: the frozen monomer charge distributions interacting classically.
e_nn = (nuclear_repulsion(dimer) - nuclear_repulsion(molecule(DONOR))
        - nuclear_repulsion(molecule(ACCEPTOR)))
e_es = float(np.trace(P_a @ V_B) + np.trace(P_b @ V_A) + np.trace(P_a @ J(P_b))) + e_nn

# 2. The frozen state: antisymmetrise the product of the two monomer determinants.
C = np.hstack([np.asarray(r_a.mo_coeffs)[:, :5], np.asarray(r_b.mo_coeffs)[:, :5]])
w, U = np.linalg.eigh(C.T @ S @ C)                    # Loewdin-orthogonalise the ten occupied MOs
C_orth = C @ U @ np.diag(w ** -0.5) @ U.T
e_frozen = hf_energy(2.0 * C_orth @ C_orth.T, dimer)

# The library functions. The Fock arguments are accepted for backward
# compatibility and are not used; the terms need the basis and the atom indices.
common = dict(e_total=e_ab, e_frag1=e_a, e_frag2=e_b,
              fock_total=None, fock_frag1=None, fock_frag2=None,
              density_total=np.asarray(r_ab.density), density_frag1=P_a, density_frag2=P_b,
              overlap=S, basis=basis, molecule=dimer,
              frag1_atoms=DONOR, frag2_atoms=ACCEPTOR)
lmo = eda_lmo(**common)
morokuma = eda_morokuma(**common)
ex_by_hand = e_frozen - e_a - e_b - e_es

lines = [
    f"Water dimer RHF/{BASIS} with ghost-atom monomers (36 functions in every calculation)",
    f"E(AB) = {e_ab:.8f}  E(A) = {e_a:.8f}  E(B) = {e_b:.8f} Ha",
    f"counterpoise-corrected interaction energy = {e_int * KCAL:.2f} kcal/mol",
    f"uncorrected (monomers in their own basis)  = {e_int_uncorrected * KCAL:.2f} kcal/mol "
    f"(BSSE {(e_int - e_int_uncorrected) * KCAL:.2f} kcal/mol)",
    "",
    "LMO-EDA (kcal/mol, counterpoise basis):",
    f"  electrostatic                     {lmo.e_elstat * KCAL:+8.2f}",
    f"  exchange                          {lmo.e_exch * KCAL:+8.2f}",
    f"  repulsion (Pauli)                 {lmo.e_rep * KCAL:+8.2f}",
    f"  polarisation + charge transfer    {lmo.e_pol * KCAL:+8.2f}",
    f"  sum                               {lmo.e_int * KCAL:+8.2f}"
    f"   (terms add up to Delta E_int to "
    f"{abs(lmo.e_elstat + lmo.e_exch + lmo.e_rep + lmo.e_pol + lmo.e_disp - e_int):.1e} Ha)",
    "",
    "Kitaura-Morokuma (kcal/mol):",
    f"  electrostatic (ES)                {morokuma.e_elstat * KCAL:+8.2f}",
    f"  exchange repulsion (EX)           {morokuma.e_exch * KCAL:+8.2f}",
    f"  polarisation + CT + coupling      {morokuma.e_pol * KCAL:+8.2f}",
    "",
    "Cross-check from the integrals (kcal/mol):",
    f"  electrostatic {e_es * KCAL:+.2f}  (library {lmo.e_elstat * KCAL:+.2f}),  "
    f"EX {ex_by_hand * KCAL:+.2f}  (library {morokuma.e_exch * KCAL:+.2f}),  "
    f"frozen-state energy difference {abs(e_frozen - lmo.extra['e_frozen']):.1e} Ha",
]
text = "\n".join(lines)
print("\n" + "\n".join("  " + ln for ln in lines))
(HERE / f"{STEM}.out").write_text(text + "\n")
