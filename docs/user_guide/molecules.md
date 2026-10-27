# Molecules

The `Molecule` class holds a list of `Atom` objects, a total charge,
and a spin multiplicity.

```python
from vibeqc import Atom, Molecule

mol = Molecule(
    atoms=[
        Atom(8, [0.0,  0.0,  0.0]),      # positions in bohr
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ],
    charge=0,
    multiplicity=1,
)
```

## Reading from files

XYZ files store positions in angstrom; `from_xyz` converts them to bohr.
Save this illustrative hydroxyl geometry as `oh.xyz` in your run directory:

```text
2
OH geometry in angstrom; charge and multiplicity are supplied in Python
O  0.0  0.0  0.00
H  0.0  0.0  0.97
```

The same nuclear geometry can describe different electronic systems:

```python
from vibeqc import from_xyz

radical = from_xyz("oh.xyz", charge=0, multiplicity=2)   # OH, 9 electrons
hydroxide = from_xyz("oh.xyz", charge=-1, multiplicity=1)  # OH-, 10 electrons
print(radical.n_electrons(), hydroxide.n_electrons())   # 9 10
```

Paths are relative to the process working directory. `from_xyz` reads the
first XYZ frame; it does not infer molecular charge or multiplicity from the
comment line. Supply both explicitly for ions and radicals. Omitting
`multiplicity=2` for neutral OH leaves the default singlet and raises
`ValueError`.

The current lightweight XYZ parser recognises element symbols H through Kr.
For other structure formats or heavier elements, see
[ASE structure input](ase_integration.md#reading-structures-via-ase-io).
ASE coordinates are also in angstrom; do not pass them unchanged to `Atom`,
whose coordinates are in bohr. An ASE periodic structure needs a
`PeriodicSystem` to retain its cell and boundary conditions; see
[Periodic systems](periodic_systems.md).

## Programmatic construction

`Atom(Z, xyz)` takes the atomic number and a Cartesian position in
**bohr**.  The position can be a list, tuple, or NumPy array. The
underlying molecule API exposes:

```python
# doc-audit: skip - inspect mol from the programmatic example above
mol.atoms                # list[Atom]      - property
mol.charge               # int             - property
mol.multiplicity         # int             - property
mol.n_electrons()        # int, derived from atomic numbers − charge
mol.nuclear_repulsion()  # Σ_{A<B} Z_A Z_B / R_AB  (Hartree)
```

Cached values (`atoms`, `charge`, `multiplicity`) are exposed as
**properties**, access them without parentheses. Computed quantities
(`n_electrons()`, `nuclear_repulsion()`) remain methods because they
do non-trivial work each call.

## Configuring open-shell systems

vibe-qc's spin information lives **on the `Molecule`**, not on the
SCF-driver options. Triplet O₂:

```python
import vibeqc as vq
o2 = vq.Molecule(
    atoms=[vq.Atom(8, [0, 0, +1.14]),
           vq.Atom(8, [0, 0, -1.14])],
    multiplicity=3,                          # 2S + 1 = 3 → triplet
)
basis = vq.BasisSet(o2, "6-31g*")
result = vq.run_uhf(o2, basis)               # α/β occupations derived
if not result.converged:
    raise RuntimeError("O2 UHF did not converge; inspect result.scf_trace")
```

Set the target state with `Molecule(charge=..., multiplicity=...)`.
`multiplicity` means $2S+1$, not the number of unpaired electrons or
$N_\alpha-N_\beta$. In particular, `multiplicity=2` requests a doublet.
Do not set `UHFOptions().spin`; that is not a supported option.

`Molecule.__init__` checks the following before any calculation:

- The electron count $N = \sum_A Z_A - q$ must be nonnegative.
- The multiplicity $M$ must be at least 1.
- $N - (M - 1)$ must be nonnegative and even. Thus **even electron
  counts need odd multiplicities**, and odd counts need even multiplicities.

The spin partition is $N_\alpha = (N + M - 1)/2$ and
$N_\beta = (N - M + 1)/2$. Examples:

| System | `charge` | `multiplicity` | Electrons | $N_\alpha$, $N_\beta$ |
|---|---|---|---|---|
| H₂O singlet | 0 | 1 | 10 | 5, 5 |
| OH radical doublet | 0 | 2 | 9 | 5, 4 |
| OH⁻ singlet | -1 | 1 | 10 | 5, 5 |
| O₂ triplet | 0 | 3 | 16 | 9, 7 |

Incompatible values raise `ValueError`. Passing these checks establishes a
valid electron partition, not that the requested state is the ground state.
For charged species, set `charge` alongside `multiplicity`:

```python
import vibeqc as vq

oh_minus = vq.Molecule(
    atoms=[vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.83])],
    charge=-1,                               # OH-
    multiplicity=1,                          # singlet
)
```

The native molecule properties are read-only. To change charge or spin,
construct a new `Molecule` with the desired values, then construct its basis
and run a new calculation. Use distinct result names when comparing states.

### Restricted vs unrestricted open shell (UHF vs ROHF)

> See the dedicated [ROHF and ROKS](rohf.md) page for this family in depth:
> the ROHF-vs-UHF decision, analytic gradients, optimisation/frequencies,
> spin-pure CAS references, and periodic status.

For an open-shell system you can choose between unrestricted and
restricted open-shell mean-field references:

* **`run_uhf` / `method="uhf"`**, unrestricted: independent α and β
  spatial orbitals. Its larger variational space can lower the energy
  relative to ROHF, but it need not be **a spin eigenfunction** (⟨S²⟩ deviates from the ideal `S(S+1)`,
  *spin contamination*).
* **`run_rohf` / `method="rohf"`**, restricted open shell: one set of
  spatial orbitals, doubly-occupied (closed) + singly-occupied (open) +
  virtual. **Spin-pure by construction** (⟨S²⟩ = `S(S+1)` *exactly*), at
  the cost of a slightly higher energy than UHF. This is the standard
  reference for spin-pure post-HF (ROHF-MP2 / ROHF-CC) and a cleaner CAS
  start than UHF.

```python
import vibeqc as vq

ch3 = vq.Molecule(
    atoms=[vq.Atom(6, [0, 0, 0]),
           vq.Atom(1, [+2.03, 0, 0]),
           vq.Atom(1, [-1.02, +1.76, 0]),
           vq.Atom(1, [-1.02, -1.76, 0])],
    multiplicity=2,                          # doublet radical
)
basis = vq.BasisSet(ch3, "6-31g*")
rohf = vq.run_rohf(ch3, basis)
if not rohf.converged:
    raise RuntimeError("CH3 ROHF did not converge; inspect rohf.scf_trace")
# or, end-to-end with output files + citations:
vq.run_job(ch3, basis="6-31g*", method="rohf", output="ch3_rohf")
```

ROHF uses Roothaan's single effective Fock operator (Roothaan, *Rev.
Mod. Phys.* **32**, 179 (1960)). The spin partition
(`n_alpha − n_beta = multiplicity − 1`) is taken from
`Molecule.multiplicity`, exactly as for UHF. Convergence knobs live on
`ROHFOptions` (`max_iter`, `conv_tol_energy`, `conv_tol_grad`,
`use_diis`, `level_shift`, `density_fit` / `aux_basis`).

The DFT counterpart is **`run_roks` / `method="roks"`** (restricted
open-shell Kohn-Sham), which reuses the same Roothaan coupling with the
spin-polarised XC potential. Degenerate ROKS frontier shells are averaged
instead of assigning one arbitrary SOMO component; linear OH(2Pi), for
example, uses occupations `2, 2, 2, 1.5, 1.5, 0, ...` across the pi pair.

```python
# doc-audit: skip - continues with ch3 and basis from the preceding example
roks = vq.run_roks(ch3, basis, functional="pbe")
if not roks.converged:
    raise RuntimeError("CH3 ROKS did not converge; inspect roks.scf_trace")
vq.run_job(ch3, basis="6-31g*", method="roks", functional="b3lyp",
           output="ch3_roks")
```

ROKS supports LDA, GGA, meta-GGA, global-hybrid, and
range-separated-hybrid functionals (B3LYP, PBE0, TPSS, r²SCAN, ωB97X,
…). Double hybrids run through `vibeqc.run_double_hybrid` so the ROKS
SCF half is combined with the required ROHF-MP2 correction.

**ROHF** has an **analytic gradient** (`vibeqc.compute_rohf_gradient`), so
`optimize_molecule(method="rohf")` and harmonic **frequencies**
(`run_job(method="rohf", hessian=True)`) run at full speed. **ROKS**
geometry optimisation uses finite-difference forces for now (its analytic
XC-gradient term is pending), correct but slower; use UKS for large fast
DFT relaxations.

ROHF/ROKS also reach **atomization** (`run_job(atomization=True)`, with
spin-pure free-atom references), **PES scans** (`vibeqc.scan`), the
**molecular optimiser** (`vibeqc.molecular_optimize.optimize_molecule`),
and **spin-pure CAS references** (`cas_reference="rohf"`).

!!! note "Current scope"
    ROHF: single points, **analytic** gradients, geometry optimisation,
    **Hessians / frequencies**, scans, atomization, CAS references.
    ROKS: single points + FD geometry optimisation (analytic gradient /
    Hessian pending). ROHF is also available through the ASE calculator
    (`VibeQC(restricted_open=True)`, analytic forces). Molecular NEB, IRC,
    and dimer routes accept ROHF/ROKS. Gamma-point periodic ROHF/ROKS is
    available on the maintained-preview GPW route; broader periodic backends
    remain limited to their documented envelope. Unsupported
    combinations raise a clear error.
