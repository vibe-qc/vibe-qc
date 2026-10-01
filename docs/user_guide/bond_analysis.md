# Bond analysis: Wiberg, NBO, EDA, IBOs, and orbital entanglement

Chemists rarely stop at a total energy. The questions that follow a
converged SCF -- how strong is this bond, where did the charge go,
what holds these fragments together -- are answered by bond-analysis
methods, and vibe-qc ships four complementary families of them. Each
family looks at the same density matrix through a different lens:
Wiberg indices count shared electron pairs, the provisional NBO tools
classify natural orbitals, EDA splits an interaction energy into physical
components, and entanglement measures read bonding straight off the
quantum-information structure of the wavefunction.

For **energy-resolved** periodic bonding analysis (COHP/COOP curves,
band-by-band bonding character), use the canonical
[COOP/COHP module](coop_cohp.md) instead -- the methods on this page
give *integrated* per-bond scalars.

```{admonition} What ships where
:class: note

Wiberg bond indices are computed automatically on every population dump.
The `.population.{txt,json}` sidecars retain an NPA field, but it is empty
and reports `unavailable.npa` until the full occupancy-weighted
Natural Atomic Orbital construction is implemented. This prevents a
Löwdin population from being reported under the NPA label. The provisional
NBO search, EDA, and entanglement analyses are Python-API workflows you call
on an SCF result.
```

## Wiberg bond indices and the delocalization index

The Wiberg bond index counts the shared electron pairs between two
atoms from the squared density-matrix elements in the Löwdin
(symmetrically orthogonalised) basis. It is less basis-set sensitive
than the Mayer bond order and complements it: Mayer contracts through
the overlap matrix, Wiberg through the orthogonalised density.

```python
import vibeqc as vq
from vibeqc.bond_analysis import wiberg_bond_orders, bond_order_summary

mol = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, 1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "sto-3g")
result = vq.run_rhf(mol, basis)

W = wiberg_bond_orders(result, basis, mol)   # (n_atoms, n_atoms)
print("O-H Wiberg index:", W[0, 1])

# All bond-order metrics side by side:
summary = bond_order_summary(
    result, basis, mol,
    compute_mayer=True,
    compute_wiberg=True,
    compute_delocalization=True,
)
print(summary.summary())
```

`delocalization_index` computes an AO-approximated delocalization
index (DI) from the Löwdin-basis density; the exact DI is defined
over QTAIM atomic basins (see the [QTAIM guide](qtaim.md)). Periodic
Gamma-point results go through the same code via
`vibeqc.bond_analysis.periodic_wiberg_bond_orders` and
`periodic_delocalization_index`.

References: Wiberg, *Tetrahedron* **24**, 1083 (1968); Matito, Solà,
Salvador & Duran, *Faraday Discuss.* **135**, 325 (2007); Outeiral,
Vincent, Martín Pendás & Popelier, *Chem. Sci.* **9**, 5517 (2018).

## NPA gate and the provisional NBO search

Natural Population Analysis derives atomic charges from Natural
Atomic Orbital occupations. A global symmetric orthogonalization alone
produces Löwdin populations, not NPA. `npa_charges` therefore raises
`NotImplementedError` until vibe-qc implements the complete
occupancy-weighted NAO construction. Population sidecars preserve the `npa`
key as an empty list and explain the gate in `unavailable.npa`. They skip
the unimplemented API, so this limitation does not appear as a failed
calculation in `errors`. The text section says `npa: not implemented`.

The separate, provisional orbital classifier remains available:

```python
import numpy as np
from vibeqc.nbo import nbo_search, donor_acceptor_analysis
from vibeqc._vibeqc_core import compute_overlap

S = np.asarray(compute_overlap(basis))
P = np.asarray(result.density)

# Provisional NBO classification (BD / LP / CR / BD* / RY*):
nbos = nbo_search(P, S, basis, mol)
print(nbos.summary())
```

`donor_acceptor_analysis` adds second-order perturbative E(2)
stabilisation energies between donor and acceptor NBOs when you also
supply the Fock matrix in the orthogonalised basis.

References: Reed, Weinstock & Weinhold, *J. Chem. Phys.* **83**, 735
(1985); Foster & Weinhold, *J. Am. Chem. Soc.* **102**, 7211 (1980);
Reed, Curtiss & Weinhold, *Chem. Rev.* **88**, 899 (1988).

## Energy decomposition analysis (EDA)

EDA answers "why do these two fragments bind" by partitioning the
interaction energy of two closed-shell fragments into electrostatic,
exchange, Pauli-repulsion, polarisation and dispersion terms.
`vibeqc.eda.eda_lmo` implements the LMO-EDA partition of Su and Li (2009)
and `vibeqc.eda.eda_morokuma` the Kitaura-Morokuma partition (1971, 1976) of
the same states, both as post-processing of three converged Hartree-Fock
runs: the dimer and the two fragments, each fragment computed **in the
dimer basis** so that the interaction energy is counterpoise corrected.
vibe-qc has no ghost-atom keyword, but a `Molecule` accepts atoms with
`Z = 0`, and the explicit-shell `BasisSet` constructor puts the dimer's
shells on them. The default SAD guess has no atomic density for `Z = 0`,
so the fragment runs start from the core Hamiltonian.

```python
import vibeqc as vq
from vibeqc.eda import eda_lmo

ANG = 1.8897259886   # Angstrom to bohr
xyz = [  # water dimer, R(O..O) = 2.95 A
    (8, (0.000, 0.000, 0.000)), (1, (0.000, 0.000, 0.960)), (1, (0.000, 0.929, -0.240)),
    (8, (0.000, 0.000, 2.950)), (1, (0.000, 0.759, 3.538)), (1, (0.000, -0.759, 3.538)),
]
frag1, frag2 = [0, 1, 2], [3, 4, 5]

def molecule(real):
    """The dimer with every atom outside `real` turned into a ghost (Z = 0)."""
    return vq.Molecule([vq.Atom(z if i in real else 0, [c * ANG for c in p])
                        for i, (z, p) in enumerate(xyz)], 0, 1)

dimer = molecule(frag1 + frag2)
basis = vq.BasisSet(dimer, "6-31g*")
shells = basis.shells()

def scf(mol):
    opts = vq.RHFOptions()
    if any(a.Z == 0 for a in mol.atoms):
        opts.initial_guess = vq.InitialGuess.HCORE      # SAD has no density for Z = 0
        return vq.run_rhf(mol, vq.BasisSet(mol, shells, name="6-31g*"), opts)
    return vq.run_rhf(mol, basis, opts)

ab, a, b = scf(dimer), scf(molecule(frag1)), scf(molecule(frag2))
eda = eda_lmo(
    ab.energy, a.energy, b.energy,
    None, None, None,                        # Fock matrices: accepted, not used
    ab.density, a.density, b.density,
    vq.compute_overlap(basis),
    basis=basis, molecule=dimer, frag1_atoms=frag1, frag2_atoms=frag2,
)
print(eda.summary())
```

```
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
```

The partition follows one path: the two fragments, the *promolecule*
(`P_1 + P_2`, the fragment orbitals as they are), the *frozen state* (one
determinant of the mutually orthogonalised occupied orbitals), and the
relaxed dimer.

| Term | Definition |
|---|---|
| `e_elstat` | Classical Coulomb energy of the two frozen charge distributions, nuclei included. |
| `e_exch` | Inter-fragment exchange energy of the promolecule state (LMO-EDA). |
| `e_rep` | Pauli repulsion beyond exchange: the frozen state minus the promolecule. |
| `e_pol` | Converged dimer minus frozen state: polarisation and charge transfer together. |
| `e_disp` | Difference of additive dispersion corrections, if you pass them (`e_disp`, `e_disp_frag1`, `e_disp_frag2`). |

The five terms add up to `e_int = e_total - e_frag1 - e_frag2` exactly.
`eda_morokuma` reports the Kitaura-Morokuma terms of the same states:
`e_exch` is then the full exchange repulsion EX (the sum of the LMO-EDA
exchange and repulsion, +5.37 kcal/mol here), `e_rep` is zero, and
`e_pol` is polarisation, charge transfer and coupling as one number. The
restricted SCF variations that separate those three are not implemented.

What to expect: the electrostatic attraction of a hydrogen bond is larger
than the binding energy, Pauli repulsion cancels most of it, and
relaxation adds a smaller stabilisation. Only the sum is an observable;
the split depends on the chosen intermediate states, and the polarisation
and charge-transfer terms are not basis-set stable. Compare them within
one basis.

**Scope, enforced by the code.** The terms are functionals of the fragment
densities, so `eda_lmo` needs the integrals and takes `basis`, `molecule`
and the atom indices of each fragment beside the SCF results. It evaluates
the Hartree-Fock energy expression on the supplied densities and refuses,
with a `ValueError`, energies that do not match it to
`consistency_tol` (1e-6 Hartree): Kohn-Sham energies, ECP runs and
fragments computed in another basis all fail here rather than return
wrong numbers. It also refuses densities that are not closed-shell SCF
densities. That excludes `fragment_density_matrix`, which returns a
projection of the dimer orbitals and not a fragment density; use the
density of a fragment run in the dimer basis. Open-shell fragments and the
Kohn-Sham form of LMO-EDA are not implemented. The Fock-matrix arguments
of the older signature are still accepted and are ignored.

References: Su & Li, *J. Chem. Phys.* **131**, 014102 (2009);
Morokuma, *J. Chem. Phys.* **55**, 1236 (1971); Kitaura & Morokuma,
*Int. J. Quantum Chem.* **10**, 325 (1976).

## Orbital entanglement measures

Quantum-information analysis reads bonding from the entanglement
structure of the wavefunction: the single-orbital entropy measures
how strongly an orbital participates in correlation, and the mutual
information between two orbitals quantifies their bonding
entanglement. vibe-qc offers an occupation-number approximation directly
from a spin-restricted, spin-summed AO density matrix:

```python
from vibeqc.entanglement import entanglement_from_density

ent = entanglement_from_density(P, S)
print("total quantum information:", ent.total_information)
print("correlation clusters:", ent.correlation_clusters)
```

The density is transformed as $S^{1/2} P S^{1/2}$, using the same
Loewdin convention as Wiberg bond orders. Its eigenvalues are the
spin-summed natural occupations in $[0, 2]$. The binary entropy uses
per-spin occupations $n_i$, obtained by dividing those eigenvalues by two:
$s_i = -n_i\ln n_i - (1-n_i)\ln(1-n_i)$.
Closed-shell RHF densities give zero entropy, mutual information and
entanglement bond orders; fractional occupations, such as those of
stretched H2 CASCI, give positive occupation entropy. Endpoint deviations
within $10^{-14}$ per spin are treated as eigensolver roundoff.

```{warning}
`entanglement_from_density` reports an occupation-based approximation.
Its mutual information and bond orders are heuristic estimates; a
one-particle density alone does not determine full orbital entanglement.
Zero entropy for a closed-shell determinant refers to its natural orbitals,
not to an arbitrary choice of localized orbitals. Supply the spin-summed
density with `Tr(P @ S)` equal to the electron count.
```

For multi-determinantal wavefunctions (CAS, CC densities), feed the
one- and two-orbital reduced density matrices to
`vibeqc.entanglement.single_orbital_entropy` and
`vibeqc.entanglement.mutual_information` directly.

References: Legeza & Sólyom, *Phys. Rev. B* **68**, 195116 (2003);
Szalay, Barcza, Szilvási, Veis & Legeza, *Sci. Rep.* **7**, 2237
(2017); Ding, Matito & Schilling, *Nat. Commun.* **17**, 4732 (2026),
DOI `10.1038/s41467-026-73527-w`.

## Intrinsic bond orbitals (IBOs) and IAO charges

The methods above all read *scalars* off the density matrix. IBOs answer
the complementary question -- **what do the bonds look like** -- by
rotating the occupied orbitals into a set that reproduces the Lewis
structure: one orbital per bond, core, or lone pair, with no Lewis
pattern assumed anywhere in the construction.

```python
import vibeqc as vq

# On by default for closed-shell molecular jobs: writes one localized
# wavefunction section per criterion for vibe-view to switch between.
res = vq.run_job(molecule=mol, basis="def2-svp", method="rhf")

res = vq.run_job(..., localize=False)     # off
res = vq.run_job(..., localize="ibo")     # just one criterion
res = vq.run_job(..., localize=["ibo", "boys"])
```

### Three criteria, because they disagree

The default emits IBO, Foster-Boys and Pipek-Mezey. That is not redundancy:
the criteria answer the same question differently, and on benzene they
visibly part company.

| benzene/def2-SVP | 1-centre | 2-centre | 3-centre |
|---|---|---|---|
| IBO | 6 | 12 | **3** (the pi system) |
| Foster-Boys | 6 | **15** | 0 |
| Pipek-Mezey | 6 | 12 | **3** |

Boys maximises the orbital dipole spread and so mixes sigma with pi, turning
the aromatic pi system into three more two-centre "banana" bonds. IBO and
Pipek-Mezey maximise atomic populations and keep sigma and pi apart. Neither
picture is wrong; which one you want depends on the question. Comparing them
is the point.

Whichever criterion runs, the *descriptors* -- atomic populations, centre
counts, charges -- are computed on the IAO yardstick, so the numbers are
directly comparable across the three. The IAO charges are identical for all
three by construction: localization is a unitary rotation inside the
occupied space, and the charges are a property of that space.

Or directly, on a converged occupied set:

```python
from vibeqc.iao import analyse_ibo

analysis = analyse_ibo(mol, basis, C_occ)
print(analysis.charges)           # IAO partial charges, one per atom
print(analysis.n_centres)         # 1 = core/lone pair, 2 = bond, 3+ = delocalised
print(analysis.atom_populations)  # n_A(i); each row sums to 1
print(analysis.centroids)         # <i|r|i> in bohr
```

For methane this gives one carbon core plus four equivalent C-H bonds;
for benzene, six cores, twelve two-centre sigma bonds, and a pi system
that genuinely cannot be squeezed onto two centres.

**IAO charges instead of Mulliken.** The partial charges that come with
this are basis-set stable in a way Mulliken charges are not -- the CH4
carbon moves only from -0.47 to -0.50 between def2-SVP and cc-pVTZ. That
stability is the reason IBOs are built on IAO populations rather than the
Mulliken populations that Pipek-Mezey uses, and it is why
`vibeqc.localise.pipek_mezey_localise` is *not* the right tool for
interpretation even though it is available (it exists for DLPNO domain
construction, where the criterion is cost, not physical meaning).

The difference is not subtle. On the same CH4/def2-SVP wavefunction,
Mulliken puts -0.144 on carbon where IAO puts -0.473. When `localize="ibo"`
is set, the IAO charges are written into the QVF's `atom_properties`
section alongside the Mulliken, Löwdin and Hirshfeld rows, so vibe-view's
charge picker offers them directly.

Use `iao_analysis=True` to request IAO charges, spin populations and
spin-resolved IAO-Wiberg bonds independently of localization or QVF. The
requested section is then included in `.out` and the population text/JSON
sidecars. See [IAO population analysis](iao_population.md) for its definition,
API, support limits and periodic integration plan.

```{admonition} Scope and cost
:class: note

Closed-shell molecular SCF only. Measured against the SCF that produced
them, all three criteria together cost about **+16 %** on a typical small
job (H2O/def2-TZVPP) and **+62 %** on benzene/def2-SVP. Nearly all of the
aromatic case is Foster-Boys, whose p=2 functional converges slowly on
degenerate pi systems -- ~89 Jacobi sweeps against IBO's 14. If that matters
for a batch, `localize="ibo"` keeps the cheapest and most interpretable one.

Jobs
using ECPs are refused -- an all-electron minimal reference basis cannot
partition a valence-only target basis -- as are elements past Z=86. In
each case the job completes and warns; only the localized section is
omitted. Periodic (Wannier) localization and open-shell support are
tracked in `handovers/HANDOVER_IBO.md`.
```

Reference: Knizia, *J. Chem. Theory Comput.* **9**, 4834 (2013),
[doi:10.1021/ct400687b](https://doi.org/10.1021/ct400687b). The minimal
reference basis is Huzinaga MINI (Huzinaga *et al.*, *Gaussian Basis Sets
for Molecular Calculations*, Elsevier 1984).

## Citations

Every method on this page carries its defining papers in the citation
database. The DOIs are pinned mechanically by
`tests/test_citations.py::test_bond_analysis_routes_pin_dois`. See
[Citations](citations.md) for the full provenance surface.
