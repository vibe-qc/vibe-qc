# Pt cluster with LANL2DZ: heavy-element ECPs

This tutorial exercises vibe-qc's libecpint integration on a
Pt atom and a Pt₂ dimer. Pt is element 78, well past the
all-electron-basis horizon, so an effective core potential
(ECP) is mandatory. ECP energies and analytic SCF gradients are a
**production** route for RHF, UHF, RKS and UKS (label vocabulary in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)),
and the [Theory](#theory) section below states what an ECP replaces and
what it costs. Numbers on this page are from a vibe-qc 0.17.10 development
build on an Apple M-series laptop.

**You will**:

- Run a single Pt atom with `LANL2DZ` orbital basis + LANL2DZ
  ECP (manual recipe).
- Run a Pt₂ dimer using the shipped `vq.auto_ecp_centers(...)`
  helper.
- Compare singlet vs triplet states.

## Theory

**What an ECP replaces.** Heavy elements have many electrons, and the core
orbitals (for platinum the 1s through 4f shells) contribute almost nothing to
bonding. They do, however, demand relativistic treatment: scalar relativity
lowers the Pt 1s eigenvalue by hundreds of hartrees and indirectly shifts the
valence shells by a chemically relevant amount. An **effective core
potential** removes $n_\text{core}$ electrons from the problem and folds their
repulsion, the Pauli exclusion from the core and the relativistic effects into
an operator acting on the remaining valence electrons. The valence
Hamiltonian is

$$
\hat H_v = \sum_i \Bigl[ -\tfrac12\nabla_i^2 - \frac{Z - n_\text{core}}{r_i}
  + \hat U_\text{ECP}(i) \Bigr] + \sum_{i<j}\frac{1}{r_{ij}},
$$

with the electron-nucleus attraction reduced to the effective charge
$Z - n_\text{core}$ (for Pt, 78 - 60 = 18 valence electrons) and
$\hat U_\text{ECP}$ a one-electron operator on each ECP centre.

**The semilocal operator.** The ECP is *semilocal*: a local radial function
plus angular-momentum projectors about the centre,

$$
\hat U_\text{ECP} = U_L(r) + \sum_{l=0}^{L-1}\sum_{m=-l}^{l} |lm\rangle\,\bigl[U_l(r) - U_L(r)\bigr]\,\langle lm| ,
\qquad
U_l(r) = \sum_k d_{lk}\, r^{\,n_{lk}-2}\, e^{-\zeta_{lk} r^2},
$$

where $|lm\rangle$ are spherical harmonics about the centre, $L$ is one more
than the highest angular momentum in the core, and the expansion
coefficients $d_{lk}$, powers $n_{lk}$ and exponents $\zeta_{lk}$ are tabulated
by the ECP family. The parameters are fitted so that the valence
eigenvalues and orbital shapes outside the core radius match an all-electron
relativistic atomic reference; scalar relativity therefore enters through
the fit, at no extra cost in the SCF. The valence basis is small, the SCF
is small, and libecpint evaluates the semilocal integrals (Shaw and Hill
2017, in the references). The two families vibe-qc ships:

* **Hay-Wadt LANL2DZ** (Hay and Wadt 1985; {vibeqc-cite}`hay_wadt_lanl_1985`),
  pseudopotentials fitted to all-electron atomic calculations that include the
  mass-velocity and Darwin (Cowan-Griffin) scalar relativistic terms. Cheap,
  robust, suitable for routine d-block work. The matching orbital basis is
  also called `LANL2DZ`, and is double-zeta.
* **Stuttgart-Köln MDF / MWB family** (Dolg, Stoll, Cao et al.), fitted to
  all-electron Dirac-Coulomb-based reference data. More accurate than LANL2DZ
  for spectroscopy and thermochemistry; pairs with the Karlsruhe def2-* and
  dhf-* orbital basis families. Five vibe-qc-bundled libraries cover the
  periodic table: `ecp10mdf` (rows 4+), `ecp28mdf` (rows 5+), `ecp46mdf`
  (post-Cd), `ecp60mdf` (f-block), `ecp78mdf` (actinides).

**Where it helps and where it fails.** An ECP makes heavy-element chemistry
reachable at the cost of a valence basis, and it captures the scalar
relativistic contraction and expansion of the valence shells. It does not
contain spin-orbit coupling (a scalar-relativistic method), it freezes the
core (no core polarisation or core-valence correlation unless the semicore
shells are kept in the valence space, which is the difference between the
small-core `ecp60mdf` and the large-core choices), and the pseudo-orbitals are
nodeless in the core region, so properties that probe the nucleus (hyperfine
couplings, contact densities) are not reliable. The result is only as good as
the valence basis and the pairing of basis and ECP; LANL2DZ is a
qualitative-to-semiquantitative choice.

vibe-qc wires both families through libecpint. Current installs ship six XML
ECP libraries (`lanl2dz` plus the five Stuttgart-Köln MDF libraries) and 44
bundled `.ecp` basis sidecars, which attach the right ECP automatically when
you name the orbital basis. For the full library / orbital-basis pairing table
see [`user_guide/ecp.md`](../user_guide/ecp.md).

## Pt atom, manual `ECPCenter` recipe

Always-available; ships on `main`.

```python
import vibeqc as vq

# The Pt ground state is 5d9 6s1, a 3D3 triplet, so multiplicity = 3.
# Spin state lives on the Molecule, not on the SCF options.
mol = vq.Molecule(
    [vq.Atom(78, [0.0, 0.0, 0.0])],
    multiplicity=3,
)
basis = vq.BasisSet(mol, "lanl2dz")        # valence-only orbital basis (22 BFs)

# One ECPCenter per heavy atom.
ec = vq.ECPCenter()
ec.Z = 78                                  # atomic number
ec.xyz = [0.0, 0.0, 0.0]                   # bohr (Cartesian)

opts = vq.UHFOptions()
opts.ecp_centers = [ec]
opts.ecp_library = "lanl2dz"

result = vq.run_uhf(mol, basis, opts)
print(f"Pt UHF energy: {result.energy:.6f} Ha")
print(f"  iterations:  {result.n_iter}")
print(f"  <S^2>:       {result.s_squared:.4f}")
# Pt UHF energy: -118.226798 Ha
#   iterations:  29
#   <S^2>:       2.0006  (triplet)
```

On this build the wiring above is redundant for a bundled basis: a plain
`vq.run_uhf(mol, basis, vq.UHFOptions())` with the `lanl2dz` basis attaches
the ECP from the basis's `.ecp` sidecar and gives the same
`-118.226798` Ha, as does `run_job(mol, basis="lanl2dz", method="uhf")`.
The explicit `ECPCenter` recipe stays useful when you want to override the
pairing (for instance a Stuttgart-Köln potential with `ecp_library="ecp60mdf"`).
The SCF trace shows the near-degenerate d/s manifold at work: the energy is
converged to 1e-11 Ha by iteration 9, while the commutator norm needs
29 iterations to reach the default gradient tolerance.

## Pt2 dimer: the auto-helper recipe

For more than one heavy atom, `vq.auto_ecp_centers(...)` picks the ECP
library and builds the per-atom `ECPCenter` list, sparing you the
boilerplate. Here it wires both Pt atoms of a singlet dimer at 10 bohr:

```python
# 10-bohr Pt-Pt separation. Singlet (paired) lives on the Molecule.
dimer = vq.Molecule(
    [
        vq.Atom(78, [0.0, 0.0, -5.0]),
        vq.Atom(78, [0.0, 0.0, +5.0]),
    ],
    multiplicity=1,
)
basis = vq.BasisSet(dimer, "lanl2dz")      # 44 BFs

# auto_ecp_centers picks library + builds per-atom ECPCenter.
opts = vq.UHFOptions()
opts.ecp_centers, opts.ecp_library = vq.auto_ecp_centers(dimer, "lanl2dz")

result = vq.run_uhf(dimer, basis, opts)
print(f"Pt2 singlet UHF: {result.energy:.6f} Ha")
```

```{note}
`vq.auto_ecp_centers` ships on `main` (exported from
`vibeqc.__init__`). The manual recipe below is equivalent and
useful when you want to override the Stuttgart-MDF default
mapping per element row:

    opts.ecp_centers = [
        vq.ECPCenter(Z=78, xyz=[0.0, 0.0, -5.0]),
        vq.ECPCenter(Z=78, xyz=[0.0, 0.0, +5.0]),
    ]
    opts.ecp_library = "lanl2dz"

Same numbers; the helper is a convenience layer on top.
```

## Singlet vs triplet

Compare the singlet and triplet Pt₂, the energy gap measures
the spin coupling between the two Pt atoms at this separation:

```python
# Each spin state needs its own Molecule (multiplicity is a Molecule
# property). Same dimer geometry, two flavours:
dimer_singlet = vq.Molecule(
    [vq.Atom(78, [0.0, 0.0, -5.0]), vq.Atom(78, [0.0, 0.0, +5.0])],
    multiplicity=1,
)
dimer_triplet = vq.Molecule(
    [vq.Atom(78, [0.0, 0.0, -5.0]), vq.Atom(78, [0.0, 0.0, +5.0])],
    multiplicity=3,
)
basis = vq.BasisSet(dimer_singlet, "lanl2dz")

opts = vq.UHFOptions()
opts.ecp_centers, opts.ecp_library = vq.auto_ecp_centers(dimer_singlet, "lanl2dz")

e_singlet = vq.run_uhf(dimer_singlet, basis, opts).energy
e_triplet = vq.run_uhf(dimer_triplet, basis, opts).energy

print(f"Singlet:    {e_singlet:.6f} Ha")
print(f"Triplet:    {e_triplet:.6f} Ha")
print(f"D (S->T):   {1000 * (e_triplet - e_singlet):+.2f} mHa")
print(f"            {(e_triplet - e_singlet) * 27.2114:.4f} eV")
```

A run gave

```text
Pt2 singlet UHF: -236.430980 Ha  (47 iterations)
Pt2 triplet UHF: -236.442456 Ha  (33 iterations)
D (S->T):        -11.48 mHa
```

At 10 bohr the two Pt atoms hardly interact, so the singlet and triplet SCF
energies would agree if both runs described two independent atoms in the same
state. They do not: two isolated triplet atoms would give
2 x (-118.226798) = -236.453596 Ha, which lies 22.6 mHa below the singlet
result and 11.1 mHa below the triplet result. The dimer SCFs have converged to
different, higher d/s occupation patterns (the UHF singlet here is not a
broken-symmetry combination of two triplet atoms), so the sign and size of
`D (S->T)` at this separation reflect which stationary point each run found,
not a spin-coupling energy. Compare final energies with the two-atom sum and
check the orbital occupations before reading a singlet-triplet gap from
transition-metal UHF. At shorter separation (try `xyz=[0.0, 0.0, ±2.5]`,
typical Pt-Pt bond length ~2.7 Å = 5.1 bohr) the states become genuinely
different, but the same caution applies.

## What this tells you

- **Heavy-element chemistry is reachable at all.** Without
  the LANL2DZ ECP, a single Pt atom in an all-electron basis
  would need ~60+ basis functions per Pt for the core +
  valence; with `lanl2dz` + ECP it's 22.
- **Convergence needs more iterations** for d-block metals because the
  density of states near the Fermi level is high: 29 iterations for the
  single Pt atom and 33 to 47 for the Pt₂ dimers here, against about 10 for
  a light closed-shell molecule. If your Pt SCF stalls, try
  [EDIIS+DIIS hybrid extrapolation](../user_guide/scf_convergence.md).
- **The auto-helper saves the boilerplate.**
  `auto_ecp_centers(mol, basis_name)` is a one-liner replacement
  for the manual `ECPCenter` building. Use it in tutorial /
  sample-script code; reserve the manual recipe for production
  scripts that want explicit control over the ECP metadata.

## Other heavy elements

The pattern generalises:

| Element | Atomic number | Library |
|---------|---------------|---------|
| Fe | 26 | (all-electron def2-* OK; no ECP needed) |
| Br | 35 | def2-svp + ecp10mdf, OR lanl2dz |
| I | 53 | def2-svp + ecp28mdf, OR lanl2dz |
| Ag | 47 | def2-svp + ecp28mdf |
| Au | 79 | def2-svp + ecp46mdf |
| Pt | 78 | def2-svp + ecp46mdf, OR lanl2dz |
| U  | 92 | dhf-tzvp + ecp78mdf |

Match the orbital-basis choice and ECP library per
[`user_guide/ecp.md`](../user_guide/ecp.md) § Library-selection
table.

## What the `.out` reports

A `run_job(mol, basis="lanl2dz", method="uhf")` on the Pt atom writes an
`.out` whose header, SCF trace and stability check look like this on the
current build (the file has no dedicated ECP block; the potential is visible
through the citations and the energy):

```text
  Job: UHF  basis=lanl2dz
  Atoms (bohr)
     1  Z= 78       0.00000000      0.00000000      0.00000000
  charge=0  multiplicity=3  n_electrons=78
  ...
  initial_guess = AUTO -> PATOM
  converged in 29 iterations; E = -118.2267984711 Ha
  <S^2> = 2.000608  (ideal 2.0000, delta = +0.0006)
  internal stability: stable (lowest Hessian eigenvalue 2.318e-07)
  ...
  Atomic charges
  #  elem  Mulliken    Löwdin  Hirshfeld
  1   Z78  0.000000  0.000000   0.000000
```

The `n_electrons=78` in the header is the nuclear-charge count of the atom;
the reduced-Z valence treatment is carried by the ECP, not by that line. The
`.references` and `.bibtex` files include Hay and Wadt 1985 for the
LANL2DZ potential and the libecpint integral paper (Shaw and Hill 2017), and
the `.system` manifest records the libecpint version. The Mulliken, Löwdin
and Hirshfeld charges are computed on the valence AO basis, so the core
electrons removed by the ECP contribute nothing to the population analysis;
a neutral Pt atom gives zero charge by construction.

## References

- **Hay-Wadt LANL2DZ.** P. J. Hay, W. R. Wadt, "Ab initio
  effective core potentials for molecular calculations.
  Potentials for the transition metal atoms Sc to Hg,"
  *J. Chem. Phys.* **82**, 270 (1985).
  [doi:10.1063/1.448799](https://doi.org/10.1063/1.448799). The
  original LANL2DZ definition. Cite this whenever `lanl2dz` is
  used.
- **Stuttgart-Köln MDF / MWB ECPs.** D. Andrae, U. Häußermann,
  M. Dolg, H. Stoll, H. Preuß, "Energy-adjusted ab initio
  pseudopotentials for the second and third row transition
  elements," *Theor. Chim. Acta* **77**, 123 (1990).
  [doi:10.1007/BF01114537](https://doi.org/10.1007/BF01114537). The
  founding paper for the ECP*mdf family. Subsequent papers cover
  specific element rows, see
  [`user_guide/ecp.md`](../user_guide/ecp.md) § Citations for the
  per-row breakdown.
- **libecpint integral evaluation.** R. A. Shaw, J. G. Hill,
  "Prescreening and efficiency in the evaluation of integrals
  over ab initio effective core potentials," *J. Chem. Phys.*
  **147**, 074108 (2017).
  [doi:10.1063/1.4986887](https://doi.org/10.1063/1.4986887). The
  integral library vibe-qc links against; fires automatically in
  the bundled [citation database](../user_guide/citations.md)
  whenever any `ecp_centers` is set.

## See also

- [`user_guide/ecp.md`](../user_guide/ecp.md), full ECP API
  + library-selection table + citations.
- [`user_guide/basis_sets.md`](../user_guide/basis_sets.md), 
  orbital-basis selection, including which orbital bases are
  designed to be paired with an ECP.
- [Open-shell systems (UHF, UKS, UMP2)](open_shell.md), 
  UHF basics for radical / triplet systems.
- [RIJCOSX SCF and analytic gradients on glycine](rijcosx_glycine.md), 
  RIJCOSX is the natural partner for ECP-using hybrid DFT on
  larger heavy-element systems (the heavy-element side keeps the
  basis count down; RIJCOSX keeps the Fock-build cost down).
