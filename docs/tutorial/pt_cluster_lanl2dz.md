# Pt cluster with LANL2DZ: heavy-element ECPs

This tutorial exercises vibe-qc's libecpint integration on a
Pt atom and a Pt₂ dimer. Pt is element 78, well past the
all-electron-basis horizon, so an effective core potential
(ECP) is mandatory.

**You will**:

- Run a single Pt atom with `LANL2DZ` orbital basis + LANL2DZ
  ECP (manual recipe).
- Run a Pt₂ dimer using the shipped `vq.auto_ecp_centers(...)`
  helper.
- Compare singlet vs triplet states.

## Background, why ECPs at all

Heavy elements have a lot of electrons. A faithful all-electron
calculation of platinum (Z = 78) with a triple-zeta basis would
need ~60 basis functions per atom, and the core orbitals, the
1s through 4d shells, contribute almost nothing to chemistry
because they sit deep in the nuclear well and rarely change
between molecular environments. They do, however, **demand
relativistic treatment**: scalar relativity moves the 1s
eigenvalue of Pt by hundreds of Hartrees and indirectly shifts
the valence by a chemically-relevant ~0.1-1 Ha.

An **effective core potential** replaces the core electrons and
the nuclear attraction with a small set of fitted radial
operators (Gaussian projectors typically), parametrised so the
valence eigenvalues + densities of the all-electron atomic
reference (with scalar relativity included) match. The valence
basis is then small, the SCF is small, and the relativistic
treatment is baked into the ECP for free. The most common
families:

* **Hay-Wadt LANL2DZ** (Hay & Wadt 1985), pseudopotentials
  derived from non-relativistic reference + Cowan-Griffin
  scalar relativity. Cheap, robust, suitable for routine
  d-block work. The matching orbital basis is also called
  `LANL2DZ`.
* **Stuttgart-Köln MDF / MWB family** (Dolg, Stoll, Cao et al.)
  - Multi-electron DF / WB fits to all-electron reference with
  Dirac-Coulomb scalar relativity. More accurate than LANL2DZ
  for spectroscopy / thermochemistry; pairs with the Dolg /
  Karlsruhe def2-* and dhf-* orbital basis families. Five
  vibe-qc-bundled libraries cover the periodic table:
  `ecp10mdf` (rows 4+), `ecp28mdf` (rows 5+), `ecp46mdf`
  (post-Cd), `ecp60mdf` (f-block), `ecp78mdf` (actinides).

vibe-qc wires both families through libecpint (Shaw & Hill 2017).
Current installs ship six XML ECP libraries (`lanl2dz` plus the five
Stuttgart-Köln MDF libraries) and 44 bundled `.ecp` basis sidecars for
automatic ECP selection.

For the full library / orbital-basis pairing table see
[`user_guide/ecp.md`](../user_guide/ecp.md).

## Pt atom, manual `ECPCenter` recipe

Always-available; ships on `main`.

```python
import vibeqc as vq

# Pt ground state is 5D0 (multiplicity = 5 in the textbook
# Russell-Saunders sense), but for SCF the simplest convergent
# guess on a closed-shell-friendly grid is the triplet. Spin
# state lives on the Molecule, not on the SCF options.
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
# Pt UHF energy: -118.227... Ha
#   iterations:  ~125  (near-degeneracy makes this slow to converge)
#   <S^2>:       ~2.0  (triplet)
```

If you forget the ECP wiring, the SCF will refuse to start
with `"canonical orth dropped too many basis directions"`, 
the symptom of trying to fit valence orbitals to a nucleus
that still expects core electrons. Add `ecp_centers` +
`ecp_library` and the SCF goes through.

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

At 10 bohr the two Pt atoms are non-interacting; both states
are degenerate to within SCF convergence noise. At shorter
separation (try `xyz=[0.0, 0.0, ±2.5]`, typical Pt-Pt bond
length ~2.7 Å = 5.1 bohr), the singlet wins, Pt₂ has a
formal triple-bond singlet ground state.

## What this tells you

- **Heavy-element chemistry is reachable at all.** Without
  the LANL2DZ ECP, a single Pt atom in an all-electron basis
  would need ~60+ basis functions per Pt for the core +
  valence; with `lanl2dz` + ECP it's 22.
- **Convergence is slow** for d-block metals because the
  density of states near the Fermi level is high. ~125
  iterations for the single Pt atom is normal; the molecular
  Pt₂ converges faster because the bonding gaps the orbitals.
  If your Pt SCF stalls, try
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

The `.out` carries an ECP block right after the basis-set summary:

```text
ECP library:        lanl2dz
ECP centres:        1
  Z=78 at (0.000000, 0.000000, 0.000000) bohr — 60 core electrons removed
ECP integrals:      computed via libecpint v1.0.7

Basis summary:
  Z=78 (Pt) — lanl2dz orbital basis, 22 functions
                                (5s, 5p, 6s, 6p, 5d valence shells)
Total electrons: 18 (78 nuclear, 60 in ECP core)
```

After SCF + properties:

```text
Mulliken populations (valence only — ECP core not in the AO basis)
  atom    Z     N_valence    charge
  Pt      78    18.000000    -0.000  ← neutral atom by construction
```

The `N_valence` matches the molecule's nominal valence electron
count (18 for Pt, 5s² 5p⁶ 6s¹ 5d⁹ in the all-electron ground
state, all promoted into the LANL2DZ valence). The Mulliken
charge is computed on the valence AO basis only, *core
electrons are removed from the basis*, so there is no
contribution to the population analysis from them.

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
