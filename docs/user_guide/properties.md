# Post-SCF properties

Supported molecular `run_job` calculations report **atomic charges**,
**bond orders**, and the **dipole moment** in their output. Individual
analyses can be unavailable even when the SCF succeeds; inspect the report
and population sidecars rather than treating a missing field as zero.
This page covers molecular SCF properties, with polarizability on
closed-shell RHF only; see
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

## Theory

**Population analyses.** Every partial-charge model splits the electron
count $N = \mathrm{tr}(\mathbf P\mathbf S)$ among the atoms; the models
differ in *how*. Mulliken ({vibeqc-cite}`mulliken_1955`) assigns each basis
function to its own centre and divides every overlap population equally,

$$
q_A^{\mathrm{Mulliken}} = Z_A - \sum_{\mu\in A}(\mathbf P\mathbf S)_{\mu\mu},
$$

which is trivial to compute but depends strongly on the basis: a diffuse
function on A describes density near B and still counts for A. Loewdin
({vibeqc-cite}`lowdin_1950`) first transforms to the symmetrically
orthogonalised basis, the orthonormal set closest in the least-squares
sense to the original AOs, and then reads the diagonal,

$$
q_A^{\mathrm{Loewdin}} = Z_A - \sum_{\mu\in A}\big(\mathbf S^{1/2}\mathbf P\mathbf S^{1/2}\big)_{\mu\mu},
$$

which removes the equal-split arbitrariness but is still a basis-space
partition. Hirshfeld ({vibeqc-cite}`hirshfeld_1977`) partitions real space
instead: each atom receives the molecular density in proportion to its own
free-atom (promolecule) density,

$$
q_A^{\mathrm{Hirshfeld}} = Z_A - \int w_A(\mathbf r)\,\rho(\mathbf r)\,d\mathbf r,
\qquad
w_A(\mathbf r) = \frac{\rho_A^{0}(\mathbf r)}{\sum_B \rho_B^{0}(\mathbf r)},
$$

integrated on the Becke-Lebedev grid; because the stockholder weights are
smooth functions of position, the charges barely move with the basis. In
vibe-qc the promolecule densities are the SAD initial-guess atomic
densities expanded in the molecular basis, which is why the values sit
slightly below tabulated Slater-atom Hirshfeld charges (see the caveats).
On H2O/STO-3G at the geometry of the examples below the three models give
$q_{\mathrm O} = -0.42$, $-0.28$ and $-0.26$ e respectively (vibe-qc 0.17.x
development build); the ordering, Mulliken most negative and Hirshfeld
least, is typical. For ECP calculations $Z_A$ is the effective core charge
in all three formulas.

**Hirshfeld on ECP calculations.** Since v0.17.16 the promolecule of an ECP
calculation is built from *valence-only* free atoms, running the same ECP
operator the SCF used, the atomic calculation the SAD guess uses on an ECP
atom. The promolecule then integrates to the valence electron count of the
molecule, as the molecular density does. Before, the free atom was either
unrepresentable in a valence basis (the Hirshfeld column disappeared from
the `.out` on H2S/LANL2DZ) or, in an all-electron-sized basis, kept the
replaced core's electrons (ZnH+ with 6-31G and ecp10mdf integrated to 31
electrons for a 20-electron molecule). Hirshfeld charges of ECP systems
computed before v0.17.16 should be recomputed; all-electron results are
unchanged. On H2S/LANL2DZ at the experimental geometry the `.out` now gives
$q_{\mathrm S} = -0.105$ e and $q_{\mathrm H} = +0.053$ e, with a promolecule
of 8.0 electrons (vibe-qc 0.17.17 development build).

When you call `hirshfeld_charges` (or `mulliken_charges`, `loewdin_charges`
and the dipole) yourself on an ECP result, pass the effective charges
$Z_A - n_{\mathrm{core},A}$ as `nuclear_charges`; the runner, the FD Hessian,
the ASE calculator and the population writers do this for you. Without them
the bare $Z_A$ is used and every ECP atom carries a spurious $n_{\mathrm{core}}$:
the same H2S call returns $q_{\mathrm S} = +9.89$ e.

```python
# doc-audit: skip - requires the converged ECP result and its matching mol/basis
import vibeqc as vq
from vibeqc.ecp_metadata import effective_nuclear_charges_from

z_eff = effective_nuclear_charges_from(mol, result)
q = vq.mulliken_charges(result, basis, mol, nuclear_charges=z_eff)
h = vq.hirshfeld_charges(result, basis, mol, nuclear_charges=z_eff)
dip = vq.dipole_moment(result, basis, mol, nuclear_charges=z_eff)
```

Use the metadata on the result that actually ran. A basis name alone is
not sufficient to reconstruct the ECP Hamiltonian. For an all-electron
result the same helper returns the ordinary nuclear charges.

**Dipole moment and its origin.** With the origin $\mathbf O$,

$$
\boldsymbol\mu(\mathbf O) = \sum_A Z_A(\mathbf R_A - \mathbf O) - \int(\mathbf r - \mathbf O)\,\rho(\mathbf r)\,d\mathbf r
 = \boldsymbol\mu(\mathbf 0) - Q\,\mathbf O,
$$

where $Q = \sum_A Z_A - N$ is the net charge. For a neutral molecule the
origin drops out; for an ion the dipole changes by $-Q\,\Delta\mathbf O$
under an origin shift, so a reported value is meaningful only together with
its origin. `dipole_moment` defaults to the centre of mass and records it in
`.origin`. On the H2O+ doublet cation at the same geometry, moving the
origin from the centre of mass by 5.11 bohr along $z$ changes $|\mu|$ from
2.52 to 15.51 D: exactly $Q\,\Delta O = 5.11$ au $= 13.0$ D added to the
$z$ component.

**Static and dynamic polarizability.** An external field $\mathbf F$ adds
$-\boldsymbol\mu\cdot\mathbf F$ to the Fock operator; the first-order
response of the orbitals follows from the coupled-perturbed Hartree-Fock
equations, a linear system in the occupied-virtual rotation amplitudes,

$$
\mathbf A\,\mathbf U^{x} = -\mathbf B^{x},
\qquad
[\mathbf A\mathbf v]_{ai} = (\varepsilon_a - \varepsilon_i)\,v_{ai}
 + \sum_{bj}\big[4(ai|bj) - (ab|ij) - (aj|ib)\big]v_{bj},
$$

with $\mathbf B^{x}$ the occupied-virtual block of the dipole integrals in
direction $x$. The static polarizability is then
$\alpha_{xy} = -2\sum_{ai}U^{x}_{ai}\,\mu_{y,ai}$, a symmetric $3\times3$
tensor in atomic units. Replacing the static response by the
imaginary-frequency TD-HF response gives $\alpha(i\omega)$, a real, smooth,
monotonically decaying function of $\omega$ whose Casimir-Polder integral
$C_6^{AB} = \tfrac{3}{\pi}\int_0^\infty\alpha_A(i\omega)\,\alpha_B(i\omega)\,d\omega$
is what the D4 dispersion model consumes. vibe-qc solves the CPHF system
with a preconditioned conjugate-gradient iteration that never forms
$\mathbf A$ (one Fock-like build per iteration) for closed-shell RHF
references.

## What you'll see in the `.out`

```
Atomic charges
------------------------------------------------------------------
   #  elem    Mulliken      Löwdin   Hirshfeld
------------------------------------------------------------------
   1     O   -0.910152   -0.646928   -0.365368
   2     H    0.455076    0.323464    0.182684
   3     H    0.455076    0.323464    0.182684
------------------------------------------------------------------
        sum  -0.000000   -0.000000    0.000000

Bond orders (Mayer)
----------------------------------------
        atoms    bond order
----------------------------------------
O1   — H2          0.7628
O1   — H3          0.7628

Dipole moment (origin: centre of mass)
----------------------------------------------------
   component     a.u. (e·bohr)       Debye
----------------------------------------------------
           x       -0.00000000     -0.0000
           y       -0.00000000     -0.0000
           z       -0.81069504     -2.0606
         |μ|        0.81069504      2.0606
```

The excerpt illustrates the report layout; it is not a required numerical
match. The Hirshfeld column requires a successful grid and SAD promolecule
calculation. If that analysis fails, the other available sections can still
be written. Review missing analyses and their diagnostics in
[the population sidecars](output_files.md#output-h2opopulationtxtjson-properties-dump-v08x).
Calling a property helper directly does not provide this best-effort wrapper:
its errors propagate to the caller.

Bond orders below 0.10 are suppressed from the table to keep the
output compact; the full matrix is available programmatically.

## Programmatic access

This block is standalone. Use the installed engine environment; positions
are in bohr and the geometry is fixed. Check convergence before evaluating
properties, and retain the same molecule and basis as the SCF calculation.

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, 1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)
if not result.converged:
    raise RuntimeError("Water RHF did not converge; inspect result.scf_trace")

# Per-atom charges (numpy array, length = n_atoms)
mul = vq.mulliken_charges(result, basis, mol)
low = vq.loewdin_charges(result, basis, mol)

# Hirshfeld returns a dataclass; `.charges` is its bare-array view.
# Inspect the integrated electron count as well as the partial charges.
h = vq.hirshfeld_charges(result, basis, mol)
print("Hirshfeld q(O):", h.charges[0])
print("Grid integration sanity:",
      f"integral(rho) = {h.molecule_norm:.4f}; expected {mol.n_electrons()}")

# Bond-order matrix (n_atoms × n_atoms numpy)
B = vq.mayer_bond_orders(result, basis, mol)
print("O–H bond order:", B[0, 1])

# Dipole moment (with conversions)
dip = vq.dipole_moment(result, basis, mol)
print(f"|μ| = {dip.total_debye:.3f} Debye  (origin: {dip.origin})")
print(f"μ = ({dip.x:.4f}, {dip.y:.4f}, {dip.z:.4f}) a.u.")
```

The charge, bond-order and dipole helpers accept the result from the molecular SCF
drivers (`run_rhf`, `run_uhf`, `run_rks`, `run_uks`). UHF / UKS results
are handled correctly, the total density `D_α + D_β` is used for
populations and dipole, and Mayer bond orders use the spin-resolved
formula `2·[(P_α S)(P_α S) + (P_β S)(P_β S)]` which reduces to the
closed-shell expression when `D_α = D_β`.

Beyond these built-ins, the [bond analysis guide](bond_analysis.md)
covers Wiberg bond indices, the NPA implementation gate and provisional
NBO search, energy decomposition analysis, and orbital-entanglement
measures. Wiberg data land automatically in the
`.population.{txt,json}` sidecars. The reserved NPA field is empty with a
structured error until full occupancy-weighted NAOs are available.

### `dipole_moment` vs `compute_dipole`

These are two different things and the names are close enough to trip
over, so, explicitly:

| Call | Returns | Use it for |
|---|---|---|
| `vq.dipole_moment(result, basis, mol, *, origin=None)` | `DipoleMoment`: `.x/.y/.z`, `.total`, `.total_debye`, `.origin` | the **physical dipole moment** of a converged SCF |
| `vq.compute_dipole(basis, origin=(0,0,0))` | `DipoleIntegrals`: `.x/.y/.z`, each an `(n_bf, n_bf)` matrix | the raw **AO dipole-integral matrices** `⟨μ|r − O|ν⟩`, when you are building your own response or localisation code |

`compute_dipole` is an integral builder: it knows nothing about a
molecule or a density, takes no `Molecule` and no density matrix, and
its signature has been `(basis, origin)` since the binding was
introduced. Passing `(molecule, basis, density)`, a plausible-looking
guess, raises

```text
TypeError: compute_dipole(): incompatible function arguments.
```

If you want the dipole moment, call `dipole_moment`; `run_job` also
writes it into the `.out` properties block and the `.qvf`
`dipole_moment` metadata with no extra call at all.

### Static polarizability sidecar

`run_job` has no polarizability option. For a closed-shell RHF job, reuse its
returned result for a separate CPHF calculation. This standalone example
runs the SCF once and writes a JSON analysis file in your run directory:

```python
import json
import numpy as np
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, 1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])
result = vq.run_job(mol, basis="def2-svp", method="rhf", output="water")
if not result.converged:
    raise RuntimeError("Water RHF did not converge; inspect water.out")

basis = vq.BasisSet(mol, "def2-svp")
alpha = np.asarray(vq.dipole_polarizability_rhf(result, basis, mol))
mu = vq.dipole_moment(result, basis, mol)

with open("water.properties.json", "w", encoding="utf-8") as stream:
    json.dump({
        "method": "RHF",
        "basis": "def2-svp",
        "polarizability_au": alpha.tolist(),
        "polarizability_isotropic_au": float(np.trace(alpha) / 3.0),
        "dipole_au": [mu.x, mu.y, mu.z],
        "dipole_total_debye": mu.total_debye,
        "dipole_origin_bohr": list(mu.origin),
    }, stream, indent=2)
```

This user-written JSON is separate from the runner's managed `.system`
manifest. Keep it alongside the job output, and use a new output stem when
preserving an earlier calculation. The ordinary properties helpers read the
reference density; a correlated total energy on a result wrapper does not
make these quantities correlated-density properties.

`dipole_polarizability_rhf` returns the symmetric `(3, 3)` static
tensor in atomic units; pass a precomputed `eri=` if you are also
running `dynamic_polarizability_rhf` on the same system, so the ERI
tensor is built once. CPHF uses a dense four-index AO ERI tensor, even though
its response solve avoids forming the full orbital Hessian. Budget for
$O(N_\mathrm{bf}^4)$ integral storage before scaling beyond small examples;
see [Memory management](memory.md). Both live in `vibeqc.cphf`; only
`dipole_polarizability_rhf` is re-exported at the top level, so import the
dynamic one from the module. The property fields and JSON serialization are covered by
`tests/test_properties.py`
(`test_property_sidecar_dipole_and_polarizability`); that test uses a separate
low-level SCF for its analysis.

`dynamic_polarizability_rhf(result, basis, mol, frequencies)` evaluates
$\alpha(i\omega)$ on the imaginary axis for a scalar or a sequence of
nonnegative frequency magnitudes (atomic units). A scalar input returns a
`(3, 3)` tensor; a sequence returns a `(n_freq, 3, 3)` array. These are
imaginary-axis responses, not real-frequency absorption spectra:

```python
import numpy as np
import vibeqc as vq
from vibeqc.cphf import dynamic_polarizability_rhf

mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]),
                   vq.Atom(1, [0.0, 1.43, -0.98]),
                   vq.Atom(1, [0.0, -1.43, -0.98])])   # bohr
basis = vq.BasisSet(mol, "sto-3g")
result = vq.run_rhf(mol, basis)
if not result.converged:
    raise RuntimeError("Water RHF did not converge; inspect result.scf_trace")

omegas = [0.0, 0.25, 0.5, 1.0]
alpha_iw = np.asarray(dynamic_polarizability_rhf(result, basis, mol, omegas))
for w, a in zip(omegas, alpha_iw):
    print(f"alpha_iso(i*{w:.2f}) = {np.trace(a) / 3:.4f} au")
```

At RHF/STO-3G this prints 2.1533, 1.9874, 1.6217 and 0.9486 au (vibe-qc
0.17.x development build): the $\omega = 0$ value equals the static
isotropic polarizability from `dipole_polarizability_rhf`, and the decay
with $\omega$ is the monotone behaviour the Casimir-Polder integral relies
on. A minimal basis has no diffuse functions, so these absolute values are
far below experiment; the recipe, not the number, is the point.

## Atomisation energies

`run_job(..., atomization=True)` reports the atomisation energy of a
mean-field calculation,

$$
D_{\mathrm{at}} = \sum_A E(\text{atom } A) - E(\text{molecule})
\qquad (> 0 \text{ for a bound molecule}),
$$

with every free-atom reference computed at the **same** method, basis,
functional and grid as the molecule, in its Hund's-rule ground-state
multiplicity (a spin-polarised atomic SCF). The references are cached per
`(Z, method, basis, functional, grid)`, so a job pays at most one atomic
SCF per distinct element. The block is written for `rhf`, `uhf`, `rks`,
`uks`, `rohf` and `roks` once the molecular SCF has converged; a post-SCF
method gets no atomisation block, because its atoms would need post-SCF
references too. Ground-state multiplicities are tabulated for the main-group
elements H to Kr (plus K and Ca); the 3d transition metals are deliberately
excluded, because a black-box atomic UHF is not reliable for their
near-degenerate configurations, so a system containing one raises
`NotImplementedError` rather than returning a wrong reference (the `.out`
then carries an "atomization not available" line instead of the block).

```python
from vibeqc import Atom, Molecule, run_job
from vibeqc.atomization import atomization_energy

h2o = Molecule([Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 1.43, -0.98]),
                Atom(1, [0.0, -1.43, -0.98])])   # bohr

# run_job writes an "## Atomization energy" block into the .out file
res = run_job(h2o, basis="sto-3g", method="rhf", atomization=True,
              output="h2o_atomization")

if not res.converged:
    raise RuntimeError("Water RHF did not converge; inspect h2o_atomization.out")

# The same numbers through the library call (free atoms are cached per element)
atm = atomization_energy(h2o, res.energy, "rhf", "sto-3g")
print(f"{atm.atomization:.6f} Ha = {atm.atomization_kcal:.1f} kcal/mol")
print(atm.atomic_energies)   # {8: E(O, triplet), 1: E(H, doublet)}
```

At RHF/STO-3G this prints 0.212252 Ha = 133.2 kcal/mol, from
E(O) = -73.80415 Ha and E(H) = -0.46658 Ha against
E(H2O) = -74.94957 Ha (vibe-qc 0.17.x development build); the `.out` block
lists the same references, their sum, the molecule and the value per atom.
`vibeqc.atomization.atomization_energy(molecule, e_molecule, method, basis,
functional=..., grid_level=..., grid_options=...)` returns an
`AtomizationResult` with `e_molecule`, `e_atoms_sum`, `atomization`,
`atomization_per_atom`, `atomic_energies` and the `atomization_kcal`
property. Hartree-Fock atomisation energies are far too small (STO-3G water
is a coincidence of basis-set error); use a correlated or DFT level for a
number to compare with experiment, and remember that no zero-point
correction is included.

MSINDO does not use this machinery: its parametrisation carries its own
atomic reference energies (`ATENG`), so `run_job(method="msindo")` reports
a **binding energy** $E_{\mathrm{total}} - \sum_A E_{\mathrm{ATENG}}(A)$ in
its own "Atomization / binding energy" block and on
`result.binding_energy`. It is negative for a bound molecule, the opposite
sign convention, and it is the quantity the method was parametrised
against.

## Validation

vibe-qc's Mulliken charges and dipole-moment components match PySCF's to
better than `1e-6` on H₂O / 6-31G\* (verified in
`tests/test_properties.py`). Hirshfeld charges are validated in
`tests/test_hirshfeld_charges.py`: the weight-normalisation identity
`Σ q_A = molecule.charge` holds to < `1e-5` e on the default grid,
the grid sweep is block-size-independent to floating-point round-off,
and the open-shell (UHF) path is exercised on the OH radical. The Mayer bond order on H₂ at the
equilibrium bond length comes out near 1.0 as expected for a single
covalent bond; in H₂O at HF/6-31G\* the O-H bond orders are about 0.76
and the H…H "bond order" is below `0.005`.

## Choosing a charge model

| Model | Cost | Basis sensitivity | Best for |
|---|---|---|---|
| Mulliken  | trivial | high (diffuse-fn sensitive) | trend across a series at fixed basis |
| Löwdin    | one symmetric orth | medium | same, less diffuse-sensitive |
| Hirshfeld | one grid build + AO eval + SAD promol | low (real-space) | quantitative-ish per-atom charges; charge-dependent dispersion methods (D4) |

Hirshfeld is the recommended default when you want a per-atom charge
number that doesn't shift wildly across reasonable basis choices.
It's also the natural input to charge-dependent dispersion methods
- vibe-qc's v0.10.0 D2b refinement plumbs ``hirshfeld_charges`` into
the D4 C6 scaling step (see [`docs/roadmap.md` § v0.10.0](../roadmap.md)).

## Caveats

- **Mulliken charges depend strongly on the basis set**, especially on
  diffuse functions. Use them for trends along a series, not for
  quantitative charge assignment. Löwdin charges are usually less
  basis-sensitive.
- **Hirshfeld charges in vibe-qc are ~0.04-0.06 e more negative on
  heavy atoms than ORCA-reported values** because the promolecule
  is constructed in the *molecular* basis (via the SAD initial guess)
  rather than from tabulated Slater-type atomic densities. Same
  qualitative trend, same response to chemistry, but expect a
  small offset when cross-comparing to ORCA Hirshfeld tables.
- **The default origin for the dipole is the center of mass**, computed
  from a built-in atomic-mass table covering H-Kr. For heavier elements
  pass an explicit `origin=` (a 3-vector in bohr).
- **Dipole values for charged species are origin-dependent**; callers
  should choose an origin that matches the convention they want to
  report (center of charge, nuclear centroid, etc.).
- **Periodic analyses are route-specific.** Periodic runners do write output
  and population sidecars on supported routes. Use their accepted density,
  overlap and k-point conventions, not these molecular recipes on a cell
  converted to `Molecule`. For example, BIPOLE bulk output explicitly marks
  the ordinary molecular dipole and Hirshfeld partition as unsupported.
  See [periodic population output](output_files.md#output-h2opopulationtxtjson-properties-dump-v08x)
  for availability and missing-data semantics.
