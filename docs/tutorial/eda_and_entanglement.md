---
myst:
  html_meta:
    "description": "Hands-on vibe-qc tutorial for two ways of asking why atoms bind: an energy decomposition of the water-dimer hydrogen bond at RHF/6-31G* with counterpoise ghost atoms (electrostatic, exchange, Pauli repulsion and polarisation terms, in the LMO-EDA and Kitaura-Morokuma schemes), and the orbital-entanglement measures (single-orbital entropy, mutual information, entanglement bond order) computed from an FCI wavefunction of LiH and used to choose an active space."
    "og:title": "vibe-qc tutorial - energy decomposition and orbital entanglement"
---

# Energy decomposition and orbital entanglement

An interaction energy is one number, and a bond order is one more.
Neither says *why* two fragments bind, nor *which* orbitals carry the
correlation that a single determinant misses. This page treats the two
questions with the two tools vibe-qc offers for them. **Energy
decomposition analysis** (EDA) splits the interaction energy of two
fragments into physically labelled pieces: the classical electrostatics
of the frozen fragment densities, the Pauli repulsion that
antisymmetrisation adds, and the stabilisation gained when the orbitals
are allowed to relax and to delocalise across the fragments. **Orbital
entanglement** measures, borrowed from quantum information theory, take
a correlated wavefunction and assign every orbital an entropy that says
how far it is from being simply occupied or empty, and every pair of
orbitals a mutual information that says how strongly they are correlated
with each other.

The two examples are small enough to run in seconds: the water-dimer
hydrogen bond at RHF/6-31G*, with the fragments computed in the dimer
basis through ghost atoms, and the FCI wavefunction of LiH in STO-3G,
whose six orbitals separate cleanly into a core, a bonding pair and
spectators. Both halves end in the theory section, which defines each
quantity and states where it helps and where it misleads.

```{admonition} Status: preview
:class: note

`vibeqc.eda.eda_lmo` and `eda_morokuma` return the LMO-EDA and
Kitaura-Morokuma terms of two **closed-shell Hartree-Fock** fragments
computed in the dimer basis, and the terms add up to the interaction
energy. They refuse everything else with a `ValueError` rather than return
a wrong number: Kohn-Sham energies, ECP runs, open-shell densities and
fragments computed in another basis. The Kohn-Sham form, open-shell
fragments, the Kitaura-Morokuma split of polarisation, charge transfer and
coupling, and a `run_job` keyword are not implemented, so you assemble the
three SCF runs yourself, as below. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

## Part 1: the water-dimer interaction energy

A decomposition starts from three SCF calculations: the dimer AB, and
each monomer alone. For the components to be comparable, the monomers
must be described in the *same basis* as the dimer, so each monomer
calculation carries the other's basis functions on **ghost atoms**,
which is also what removes the basis-set superposition error from the
interaction energy (the counterpoise correction of
{vibeqc-cite}`boys_bernardi_1970`). vibe-qc has no ghost-atom keyword,
but a `Molecule` accepts atoms with $Z = 0$, and the explicit-shell
`BasisSet` constructor puts the dimer's shells on them; the only thing to
change is the initial guess, because the default SAD guess has no atomic
density for $Z = 0$.

```python
import numpy as np
import vibeqc as vq
from vibeqc.eda import eda_lmo, eda_morokuma

ANG = 1.8897259886
KCAL = 627.5094740631

# Water dimer (examples/h2o_dimer.xyz): donor first, acceptor second, R(O..O) = 2.95 A.
coords = [
    (8, [0.000, 0.000, 0.000]), (1, [0.000, 0.000, 0.960]), (1, [0.000, 0.929, -0.240]),
    (8, [0.000, 0.000, 2.950]), (1, [0.000, 0.759, 3.538]), (1, [0.000, -0.759, 3.538]),
]
DONOR, ACCEPTOR = [0, 1, 2], [3, 4, 5]

def molecule(real):
    """The dimer geometry with every atom outside `real` turned into a ghost (Z = 0)."""
    return vq.Molecule([vq.Atom(z if i in real else 0, [c * ANG for c in p])
                        for i, (z, p) in enumerate(coords)], 0, 1)

dimer = molecule(DONOR + ACCEPTOR)
basis = vq.BasisSet(dimer, "6-31g*")
shells = basis.shells()                        # every shell remembers its own centre

def scf(mol):
    opts = vq.RHFOptions()
    if any(a.Z == 0 for a in mol.atoms):
        opts.initial_guess = vq.InitialGuess.HCORE   # SAD has no atomic density for Z = 0
        b = vq.BasisSet(mol, shells, name="6-31g*")  # the full dimer basis on the monomer
    else:
        b = basis
    res = vq.run_rhf(mol, b, opts)
    assert res.converged
    return res

r_ab, r_a, r_b = scf(dimer), scf(molecule(DONOR)), scf(molecule(ACCEPTOR))
e_int = r_ab.energy - r_a.energy - r_b.energy
print(f"E(AB) = {r_ab.energy:.8f}  E(A) = {r_a.energy:.8f}  E(B) = {r_b.energy:.8f} Ha")
print(f"counterpoise-corrected interaction energy = {e_int * KCAL:.2f} kcal/mol")

# The decomposition. The Fock arguments are accepted and not used; the terms
# are functionals of the densities, so the call needs the basis, the molecule
# and the atom indices of each fragment.
S = np.asarray(vq.compute_overlap(basis))
common = dict(
    e_total=r_ab.energy, e_frag1=r_a.energy, e_frag2=r_b.energy,
    fock_total=None, fock_frag1=None, fock_frag2=None,
    density_total=np.asarray(r_ab.density), density_frag1=np.asarray(r_a.density),
    density_frag2=np.asarray(r_b.density), overlap=S,
    basis=basis, molecule=dimer, frag1_atoms=DONOR, frag2_atoms=ACCEPTOR,
)
lmo = eda_lmo(**common)
print(lmo.summary())
km = eda_morokuma(**common)
print(f"Kitaura-Morokuma: ES {km.e_elstat * KCAL:+.2f}  EX {km.e_exch * KCAL:+.2f}  "
      f"PL+CT+MIX {km.e_pol * KCAL:+.2f} kcal/mol")
```

On a vibe-qc 0.17 development build (Apple M5 Max) the three SCFs and both
decompositions together take about two seconds (36 basis functions):

```
E(AB) = -152.02576771  E(A) = -76.00957528  E(B) = -76.00960659 Ha
counterpoise-corrected interaction energy = -4.13 kcal/mol
Energy Decomposition Analysis (LMO-EDA)
----------------------------------------------------------------
  E_total (supersystem)   =  -152.02576771 E_h
  E_fragment_1 (isolated) =   -76.00957528 E_h
  E_fragment_2 (isolated) =   -76.00960659 E_h
----------------------------------------------------------------
  dE_int (total)          =    -0.00658583 E_h      -4.13 kcal/mol

  Electrostatic           =    -0.01255786 E_h      -7.88 kcal/mol
  Exchange                =    -0.01136365 E_h      -7.13 kcal/mol
  Repulsion (Pauli)       =     0.01991724 E_h      12.50 kcal/mol
  Polarisation (+CT)      =    -0.00258156 E_h      -1.62 kcal/mol
----------------------------------------------------------------
Kitaura-Morokuma: ES -7.88  EX +5.37  PL+CT+MIX -1.62 kcal/mol
```

The interaction energy is $-4.13$ kcal/mol with the counterpoise
correction, against $-4.86$ kcal/mol if the monomers are computed in
their own 6-31G* bases (a BSSE of 0.73 kcal/mol, typical for a double-zeta
basis; the ghost-basis monomer energies are 0.55 mHa lower than the
free-monomer ones). The RHF/6-31G* value is close to the accepted
Hartree-Fock limit of about $-3.7$ kcal/mol and well below the correlated
binding energy of about $-5$ kcal/mol, because Hartree-Fock has no
dispersion. The five terms of `eda_lmo` (`e_elstat`, `e_exch`, `e_rep`,
`e_pol`, `e_disp`) add up to `e_int` exactly. `eda_morokuma` reports the
Kitaura-Morokuma terms of the same states: its `e_exch` is the full
exchange repulsion EX, which is the LMO-EDA exchange plus repulsion
($-7.13 + 12.50 = +5.37$), `e_rep` is zero, and `e_pol` lumps polarisation,
charge transfer and coupling into one number because the restricted SCF
variations that separate them are not implemented.

### Reading the terms

This is the classic picture of a hydrogen bond at the Hartree-Fock level.
The electrostatic attraction between the frozen monomer densities
($-7.9$ kcal/mol) is almost twice the total binding. The Pauli cost of
making the two closed shells orthogonal, $+5.4$ kcal/mol as one number in
Kitaura-Morokuma, is split by LMO-EDA into an exchange stabilisation of the
promolecule ($-7.1$) and a repulsion beyond it ($+12.5$); together they
cancel most of the electrostatics and leave a frozen-density interaction of
$-2.5$. Orbital relaxation, which lumps together the polarisation of each
monomer by the other's field and the charge transfer from the acceptor's
lone pair into the donor's O-H $\sigma^*$ orbital, adds the remaining
$-1.6$. Only the sum is an observable. The split depends on the chosen
intermediate states, and the polarisation and charge-transfer terms in
particular are not basis-set stable, so compare them within one basis. The
order of magnitude and the sign pattern are what every scheme agrees on.

### An independent check from the integrals

The Kitaura-Morokuma electrostatic term and EX need nothing beyond the
three densities and the one- and two-electron integrals, all of which
vibe-qc exposes. The block below rebuilds them in about thirty lines, as
an independent check that is also a compact statement of what the library
computes. It continues the script above.

```python
T = np.asarray(vq.compute_kinetic(basis))
V_A = np.asarray(vq.compute_nuclear(basis, molecule(DONOR)))     # attraction to A's nuclei only
V_B = np.asarray(vq.compute_nuclear(basis, molecule(ACCEPTOR)))
eri = np.asarray(vq.compute_eri(basis))                           # (36, 36, 36, 36), 13 MB
J = lambda P: np.einsum("pqrs,rs->pq", eri, P, optimize=True)
K = lambda P: np.einsum("prqs,rs->pq", eri, P, optimize=True)

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
e_nn = nuclear_repulsion(dimer) - nuclear_repulsion(molecule(DONOR)) - nuclear_repulsion(molecule(ACCEPTOR))
e_es = float(np.trace(P_a @ V_B) + np.trace(P_b @ V_A) + np.trace(P_a @ J(P_b))) + e_nn

# 2. The frozen state: antisymmetrise the product of the two monomer determinants.
C = np.hstack([np.asarray(r_a.mo_coeffs)[:, :5], np.asarray(r_b.mo_coeffs)[:, :5]])
w, U = np.linalg.eigh(C.T @ S @ C)              # Lowdin-orthogonalise the ten occupied MOs
C_orth = C @ U @ np.diag(w ** -0.5) @ U.T
e_frozen = hf_energy(2.0 * C_orth @ C_orth.T, dimer)
e_ex = e_frozen - r_a.energy - r_b.energy - e_es      # Kitaura-Morokuma EX

# 3. Everything the relaxed dimer gains beyond the frozen, antisymmetrised state.
e_relax = r_ab.energy - e_frozen

print(f"electrostatic {e_es * KCAL:+.2f}   EX {e_ex * KCAL:+.2f}   "
      f"polarisation + charge transfer {e_relax * KCAL:+.2f}   "
      f"sum {(e_es + e_ex + e_relax) * KCAL:+.2f} kcal/mol")
```

```
electrostatic -7.88   EX +5.37   polarisation + charge transfer -1.62   sum -4.13 kcal/mol
```

The three numbers match the library's Kitaura-Morokuma terms, and they sum
to the interaction energy by construction.

### What the function refuses

Because the terms are functionals of the fragment densities, the function
evaluates the Hartree-Fock energy expression on the densities you supply
and raises a `ValueError` when it does not reproduce the supplied energies
to `consistency_tol` (1e-6 hartree). Kohn-Sham energies, ECP runs and
fragments computed in a different basis therefore fail loudly. It also
refuses densities that are not closed-shell SCF densities, which excludes
`fragment_density_matrix`: that helper returns a projection of the dimer
orbitals and not a fragment density, so use the density of a fragment run
in the dimer basis, as above.

## Part 2: orbital entanglement from an FCI wavefunction

The entanglement measures need the one-orbital and two-orbital reduced
density matrices of a correlated state. For a single determinant they
are trivial (every orbital is occupied or empty, all entropies are
zero), so the interesting input is a CI vector. `run_job(method="fci")`
returns one, together with the determinant labels, and a partial trace
over the other orbitals gives the reduced density matrices directly. The
helper below does that trace with the fermionic sign bookkeeping; the
measures themselves come from `vibeqc.entanglement`.

```python
import numpy as np
import vibeqc as vq
from vibeqc.entanglement import (correlation_clusters, entanglement_bond_order,
                                 mutual_information, single_orbital_entropy)

ANG = 1.8897259886

def _parity(seq, target):
    """Sign of the permutation that reorders `seq` into `target`."""
    pos = {v: k for k, v in enumerate(seq)}
    perm = [pos[v] for v in target]
    inversions = sum(1 for x in range(len(perm)) for y in range(x + 1, len(perm)) if perm[x] > perm[y])
    return -1 if inversions % 2 else 1

def orbital_rdm(ci, dets, norb, orbitals):
    """Reduced density matrix of the spatial `orbitals`, traced over all others.

    `dets` are vibe-qc's (alpha_occ, beta_occ) tuples; the creation-operator
    string is the alpha orbitals in ascending order followed by the beta
    orbitals. Local basis bit k is spin-orbital `sub[k]` occupied.
    """
    sub = [s for o in orbitals for s in (o, norb + o)]          # alpha_o, beta_o, ...
    blocks = {}
    for c, (aocc, bocc) in zip(ci, dets):
        occ = list(aocc) + [norb + q for q in bocc]
        local = [s for s in sub if s in occ]
        env = [g for g in occ if g not in sub]
        index = sum(1 << k for k, s in enumerate(sub) if s in occ)
        vec = blocks.setdefault(tuple(env), np.zeros(2 ** len(sub)))
        vec[index] += _parity(occ, local + env) * c
    return sum(np.outer(v, v) for v in blocks.values())

def entanglement(mol, basis_name, label):
    res = vq.run_job(mol, basis=basis_name, method="fci", output=label, verbose=0)
    norb = vq.BasisSet(mol, basis_name).nbasis
    ci, dets = res.ci_coeffs, res.ci_labels
    one = np.array([orbital_rdm(ci, dets, norb, [i]) for i in range(norb)])
    two = np.zeros((norb, norb, 16, 16))
    for i in range(norb):
        for j in range(i + 1, norb):
            two[i, j] = two[j, i] = orbital_rdm(ci, dets, norb, [i, j])
    s = single_orbital_entropy(one)
    info = mutual_information(one, two)
    print(f"== {label}: E(FCI) = {res.energy:.8f} Ha, {len(dets)} determinants")
    print("  s_i          :", np.round(s, 4))
    print("  total S      :", round(float(s.sum()), 4))
    off = info.copy()
    np.fill_diagonal(off, 0.0)
    print("  strongest I_ij:", sorted(((round(float(off[i, j]), 4), i, j) for i in range(norb)
                                       for j in range(i + 1, norb)), reverse=True)[:3])
    print("  clusters (I > 0.1):", correlation_clusters(info, threshold=0.1),
          " (I > 0.01):", correlation_clusters(info, threshold=0.01))
    return s, info

lih = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 1.5957 * ANG])], 0, 1)
s, info = entanglement(lih, "sto-3g", "lih_fci")
print("  I_ij matrix (nats):")
for row in info:
    print("   ", " ".join(f"{x:6.3f}" for x in row))
```

Runtime is about two seconds, almost all of it the RHF plus FCI job; the
partial traces over 225 determinants are instantaneous.

```
== lih_fci: E(FCI) = -7.88239151 Ha, 225 determinants
  s_i          : [0.0008 0.1307 0.0723 0.0062 0.0062 0.1159]
  total S      : 0.3322
  strongest I_ij: [(0.1648, 1, 5), (0.0802, 1, 2), (0.063, 2, 5)]
  clusters (I > 0.1): [{1, 5}]  (I > 0.01): [{1, 2, 5}]
  I_ij matrix (nats):
     0.002  0.000  0.000  0.000  0.000  0.000
     0.000  0.261  0.080  0.009  0.009  0.165
     0.000  0.080  0.145  0.001  0.001  0.063
     0.000  0.009  0.001  0.012  0.001  0.003
     0.000  0.009  0.001  0.001  0.012  0.003
     0.000  0.165  0.063  0.003  0.003  0.232
```

Read the entropies against the canonical RHF orbitals of LiH in STO-3G:
orbital 0 is the Li 1s core, 1 the Li-H $\sigma$ bond, 2 an empty
Li-centred $\sigma$-type orbital, 3 and 4 the degenerate Li 2p$_\pi$
pair, and 5 the $\sigma^*$ antibond. The core has an entropy of
$8\times10^{-4}$ nats: it is doubly occupied in every important
determinant and correlating it buys nothing, which is the quantitative
version of "freeze the core". The bond (0.131) and its antibond (0.116)
have the largest entropies and by far the largest mutual information
(0.165): they are the correlated pair, and $I_{15}$ is exactly the
left-right correlation of the Li-H bond that a single determinant
cannot describe. Orbital 2 is next (0.072, coupled to both), the
$\pi$ pair is essentially uncorrelated (0.006). `correlation_clusters`
turns the matrix into a decision: at a threshold of 0.1 nats the only
cluster is $\{1, 5\}$, a CAS(2,2); at 0.01 it grows to $\{1, 2, 5\}$, a
CAS(2,3); the core and the $\pi$ orbitals never join. Run
`run_job(method="casci", active_space=(2, 2))` on those two orbitals and
you recover most of the FCI correlation energy at a fraction of the
determinant count, which is what entropy-based active-space selection
promises. The diagonal of the `mutual_information` matrix holds $2s_i$
by convention, and `entanglement_bond_order(info)` normalises
$\sqrt{I_{ij}}$ by the largest entry of whatever matrix it receives, so
zero the diagonal first if you want the off-diagonal pairs scaled to
the strongest pair.

### Variation: entropies grow with static correlation

The same function on H$_2$ in 6-31G at three bond lengths shows what
the numbers do when a bond breaks.

```python
for r in (0.74, 1.5, 3.0):
    h2 = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, r * ANG])], 0, 1)
    entanglement(h2, "6-31g", f"h2_{r}")
```

```
== h2_0.74: E(FCI) = -1.15167254 Ha, 16 determinants
  s_i          : [0.0759 0.0654 0.0185 0.0429]
  total S      : 0.2027
  strongest I_ij: [(0.0829, 0, 1), (0.0442, 1, 3), (0.04, 0, 3)]
  clusters (I > 0.1): []  (I > 0.01): [{0, 1, 2, 3}]
== h2_1.5: E(FCI) = -1.05434746 Ha, 16 determinants
  s_i          : [0.3299 0.3487 0.0598 0.0163]
  total S      : 0.7547
  strongest I_ij: [(0.603, 0, 1), (0.093, 1, 2), (0.0307, 0, 3)]
  clusters (I > 0.1): [{0, 1}]  (I > 0.01): [{0, 1, 2, 3}]
== h2_3.0: E(FCI) = -0.99745483 Ha, 16 determinants
  s_i          : [0.7306 0.719  0.0518 0.0409]
  total S      : 1.5423
  strongest I_ij: [(1.3569, 0, 1), (0.0989, 0, 2), (0.0764, 1, 3)]
  clusters (I > 0.1): [{0, 1}]  (I > 0.01): [{0, 1, 2, 3}]
```

At the equilibrium distance every entropy is below 0.08 and no pair
crosses 0.1: the RHF determinant is a good starting point and dynamic
correlation is all that is left, spread thinly over all four orbitals.
At 3.0 A the $\sigma_g$ and $\sigma_u$ entropies have risen to 0.73,
close to the maximum of $\ln 2 = 0.693$ for a state that is an equal
mixture of "doubly occupied" and "empty" (the extra few hundredths come
from the small admixture of the higher orbitals), and their mutual
information, 1.357, approaches $2\ln 2 = 1.386$, the value for a
perfectly entangled pair. The minimal-basis limit makes this exact:
for H$_2$ in STO-3G there are only two orbitals, the two-orbital state
is the pure FCI state with zero entropy, and $I_{01} = s_0 + s_1$
holds to machine precision, which is a useful check of any partial
trace routine.

## Theory

### Energy decomposition

Take two closed-shell fragments $A$ and $B$ with self-consistent
densities $\mathbf P_A$, $\mathbf P_B$ and energies $E_A$, $E_B$ in a
common basis, and the dimer with energy $E_{AB}$. The interaction energy
$\Delta E_\text{int} = E_{AB} - E_A - E_B$ is decomposed along a path of
intermediate, non-variational states:

$$
\Delta E_\text{int} = \Delta E_\text{es} + \Delta E_\text{exch} + \Delta E_\text{rep}
                    + \Delta E_\text{pol} + \Delta E_\text{disp} .
$$

The **electrostatic** term is the classical Coulomb interaction of the
two frozen charge distributions,

$$
\Delta E_\text{es} = \mathrm{Tr}[\mathbf P_A \mathbf V_B] + \mathrm{Tr}[\mathbf P_B \mathbf V_A]
                   + \mathrm{Tr}[\mathbf P_A \mathbf J(\mathbf P_B)] + E_{nn}^{AB},
$$

with $\mathbf V_X$ the attraction to the nuclei of $X$, $\mathbf J$ the
Coulomb matrix and $E_{nn}^{AB}$ the inter-fragment nuclear repulsion.
It is attractive for a hydrogen bond because the dipoles are aligned,
and it is the only term with a classical multipole expansion at long
range. The **Pauli** (exchange-repulsion) energy is the change when the
simple product of the two fragment determinants is antisymmetrised, that
is, when the occupied orbitals of $A$ and $B$ are made mutually
orthogonal (by a Lowdin transformation in the check above) and the energy
of the resulting single determinant is evaluated with the full
Hamiltonian. It is repulsive at short range, it is what makes molecules
have a size, and it grows exponentially as the closed shells
interpenetrate. Kitaura-Morokuma ({vibeqc-cite}`morokuma_eda_1971`)
report it as one term, EX. LMO-EDA ({vibeqc-cite}`su_li_eda_2009`) splits
it in two at the *promolecule*, the unorthogonalised sum $\mathbf P_A +
\mathbf P_B$: $\Delta E_\text{exch}$ is the inter-fragment exchange
energy of the promolecule, and $\Delta E_\text{rep}$ is the further
repulsion that orthogonalisation adds, so that $\mathrm{EX} =
\Delta E_\text{exch} + \Delta E_\text{rep}$. Together, $\Delta
E_\text{es}$ and the Pauli energy are the *frozen-density* interaction,
the energy of the dimer before any orbital is allowed to change. The
**polarisation** term is the variational relaxation from that frozen
state to the converged dimer. The full Kitaura-Morokuma scheme splits it
further into the polarisation of each fragment within its own orbital
space, the charge transfer between them and a coupling term, using
restricted SCF variations; vibe-qc does not implement those and reports
one lumped term. LMO-EDA defines its terms through localised orbitals so
that the scheme extends to DFT densities, where the frozen and
antisymmetrised states also carry an exchange-correlation contribution;
the vibe-qc implementation is Hartree-Fock only, and every term it
returns is invariant to rotations among the occupied orbitals of a
fragment, so no localisation is carried out.
A **dispersion** term appears when the method has one, as the
difference of D3 or D4 corrections between dimer and monomers, or as the
correlation part of the interaction in a correlated EDA; at the
Hartree-Fock level it is zero, which is why the water dimer above is
underbound.

The scheme has two known weak points. The polarisation and
charge-transfer terms are not separately basis-set stable: as the basis
grows toward completeness each fragment's own functions can describe
the other's region, and "charge transfer" migrates into "polarisation";
compare them only within one basis. And only the sum is an observable;
the components depend on the chosen intermediate states. The
electrostatic and Pauli terms are robust in sign and magnitude across
schemes, and the ordering of the components across a series of related
complexes is more informative than any one number.

### Orbital entanglement

Choose an orthonormal orbital basis (here the canonical RHF orbitals)
and write the correlated wavefunction in the Fock space of those
orbitals. Each spatial orbital $i$ is a four-state system,
$\{|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle\}$.
Tracing the density operator $|\Psi\rangle\langle\Psi|$ over all other
orbitals gives the **one-orbital reduced density matrix** $\rho_i$, and
for a state with definite particle number and $S_z$ it is diagonal with
entries

$$
p_{\uparrow\downarrow} = \langle n_{i\uparrow} n_{i\downarrow}\rangle,\quad
p_{\uparrow} = \langle n_{i\uparrow}\rangle - p_{\uparrow\downarrow},\quad
p_{\downarrow} = \langle n_{i\downarrow}\rangle - p_{\uparrow\downarrow},\quad
p_0 = 1 - p_\uparrow - p_\downarrow - p_{\uparrow\downarrow},
$$

expressible through the one- and two-particle reduced density matrices
(`vibeqc.solvers.make_rdm12` returns them in the spin-summed convention,
in which $\langle n_{i\uparrow}n_{i\downarrow}\rangle = \tfrac12\Gamma_{iiii}$).
The **single-orbital entropy** ({vibeqc-cite}`legeza_entanglement_2003`)
is the von Neumann entropy of that matrix,

$$
s_i = -\mathrm{Tr}\,\rho_i \ln\rho_i = -\sum_\alpha p_\alpha \ln p_\alpha ,
\qquad 0 \le s_i \le \ln 4 ,
$$

zero for an orbital that is always doubly occupied or always empty, and
$\ln 2$ for one that is doubly occupied in half of the wavefunction and
empty in the other half, the situation of each $\sigma$ orbital of
dissociated H$_2$. The **two-orbital reduced density matrix** $\rho_{ij}$
is the $16\times16$ trace over all orbitals but $i$ and $j$; its
entropy $s_{ij}$ measures the uncertainty of the pair as a whole, and the
**mutual information**

$$
I_{ij} = s_i + s_j - s_{ij}
$$

measures how much of the individual uncertainties is shared, that is,
how strongly the two orbitals are correlated. It vanishes for
independent orbitals, is bounded by $2\min(s_i, s_j)$, and for a pure
two-orbital system equals $s_i + s_j$. The sum of the single-orbital
entropies is the total quantum information of the orbital basis, and
Szalay and co-workers showed how the multi-orbital generalisation of
$I_{ij}$ organises bonding into correlated clusters
({vibeqc-cite}`szalay_multiorbital_2017`); Ding, Matito and Schilling
take the idea further and derive bond orders from the entanglement of
maximally entangled atomic orbitals ({vibeqc-cite}`entanglement_bonds_2025`),
the construction behind `entanglement_bond_order`, which returns
$\sqrt{I_{ij}}$ normalised to its largest entry.

Because the trace is taken over fermionic modes, the partial trace must
respect the sign of the creation-operator string: `orbital_rdm` above
reorders every determinant so that the traced-out orbitals come last and
multiplies by the parity of that permutation. Without the parity the
two-orbital matrices, and hence the mutual information, are wrong for
non-adjacent orbital pairs, although the one-orbital entropies, which
depend only on diagonal elements, survive.

Where the measures help is the choice of an active space and the
diagnosis of static correlation: an orbital with $s_i$ above a fraction,
typically 10 to 20 percent, of the largest entropy belongs in the active
space, and the mutual-information graph shows which orbitals must go in
together (Stein and Reiher, *J. Chem. Theory Comput.* **12**, 1760
(2016), automated exactly this rule on top of DMRG). Where they mislead
is in their basis dependence: the entropies are properties of the
wavefunction *in a given orbital basis*, so localising or rotating the
orbitals changes every $s_i$ and $I_{ij}$ (only the total wavefunction
is invariant), and a comparison between molecules is meaningful only
with orbitals chosen the same way. The closed-form
`entanglement_from_density` in the same module works from a
single-determinant density alone; its entropies come from the natural
occupations and its "mutual information" is a heuristic product of
occupation factors, useful as a quick flag for fractional occupations in
a smeared or broken-symmetry density but not the quantity defined
above.

## Resources

- [Bond analysis](../user_guide/bond_analysis.md): the EDA and
  entanglement sections of the user guide, beside Wiberg indices, IAO
  charges and the NBO helper.
- [Multireference tutorial](casscf_multireference.md): what to do with
  the active space once the entropies have chosen it.
- [Natural orbitals](natural_orbitals.md): the occupation-number view of
  the same correlated densities.
- [Post-SCF properties](post_scf_properties.md) and the water dimer
  section there, for the population-analysis view of the same hydrogen
  bond.
- `examples/h2o_dimer.xyz` and `examples/molecular/input-h2o-dimer-opt.py`:
  the geometry used here and its optimisation.
- [Feature matrix](../features.md) and
  [Methods: theory and status](../methods.md#energy-decomposition-and-orbital-entanglement).

Cost: the EDA is three SCFs plus one evaluation of the Hartree-Fock
energy expression on the frozen states, and the cross-check above builds
the four-index integral tensor ($n_\text{bf}^4$ doubles) explicitly; the entanglement
analysis needs a CI vector, so it is limited by the FCI or CASCI that
produces it, and the partial traces scale as the number of determinants
times the number of orbital pairs.

## References

- **LMO-EDA.** P. Su and H. Li, "Energy decomposition analysis of
  covalent bonds and intermolecular interactions," *J. Chem. Phys.*
  **131**, 014102 (2009), {vibeqc-cite}`su_li_eda_2009`.
- **Morokuma decomposition.** K. Morokuma, "Molecular orbital studies of
  hydrogen bonds. III. C=O...H-O hydrogen bond in H2CO...H2O and
  H2CO...2H2O," *J. Chem. Phys.* **55**, 1236 (1971),
  {vibeqc-cite}`morokuma_eda_1971`; K. Kitaura and K. Morokuma, "A new
  energy decomposition scheme for molecular interactions within the
  Hartree-Fock approximation," *Int. J. Quantum Chem.* **10**, 325
  (1976), the ES/EX/PL/CT partition computed by hand above.
- **Counterpoise correction.** S. F. Boys and F. Bernardi, "The
  calculation of small molecular interactions by the differences of
  separate total energies," *Mol. Phys.* **19**, 553 (1970),
  {vibeqc-cite}`boys_bernardi_1970`.
- **Orbital entropies.** O. Legeza and J. Solyom, "Optimizing the
  density-matrix renormalization group method using quantum information
  entropy," *Phys. Rev. B* **68**, 195116 (2003),
  {vibeqc-cite}`legeza_entanglement_2003`.
- **Correlation theory of the bond.** Sz. Szalay, G. Barcza,
  T. Szilvasi, L. Veis and O. Legeza, "The correlation theory of the
  chemical bond," *Sci. Rep.* **7**, 2237 (2017),
  {vibeqc-cite}`szalay_multiorbital_2017`.
- **Entanglement bond orders.** L. Ding, E. Matito and C. Schilling,
  "Chemical bonding concepts emerge naturally from maximally entangled
  atomic orbitals," *Nat. Commun.* **17**, 4732 (2026),
  {vibeqc-cite}`entanglement_bonds_2025`.
- **Automated active spaces.** C. J. Stein and M. Reiher, "Automated
  selection of active orbital spaces," *J. Chem. Theory Comput.* **12**,
  1760 (2016).

## Next

- [Multireference: CASSCF, CASPT2, NEVPT2](casscf_multireference.md)
  uses the active space the entropies point to.
- [QTAIM](qtaim_topology.md) partitions the same water dimer in real
  space and finds the hydrogen-bond critical point.
- [Dispersion](dispersion.md) supplies the term that is missing from
  the Hartree-Fock decomposition above.
