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

XYZ coordinates (positions in Ångström are auto-converted to bohr):

```python
from vibeqc import from_xyz
mol = from_xyz("examples/h2o.xyz")
mol = from_xyz("examples/oh.xyz", charge=-1)
```

Anything ASE reads, CIF, PDB, POSCAR, Gaussian inputs, etc.:

```python
from ase.io import read
atoms = read("crystal.cif")
# Convert to a vibe-qc Molecule (for molecules) or build a
# PeriodicSystem (for solids) - see the ASE integration page.
```

## Programmatic construction

`Atom(Z, xyz)` takes the atomic number and a Cartesian position in
**bohr**.  The position can be a list, tuple, or NumPy array. The
underlying molecule API exposes:

```python
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
```

There is **no `spin` field on `UHFOptions`, `UKSOptions`,
`PeriodicRHFOptions`, or any other SCF-options class**, setting
`UHFOptions().spin = 2` raises `AttributeError`. The single source
of truth for spin is `Molecule.multiplicity`, which keeps every
RHF / UHF / RKS / UKS run against the same molecule consistent.
This is a common point of confusion for users coming from PySCF
(where `mol.spin` lives on the molecule but the SCF driver also
accepts spin overrides) or from input-file codes (where spin sits
in the run block).

`Molecule.__init__` validates that `multiplicity` is consistent
with the electron count: `n_electrons` and `multiplicity` must have
the same parity, otherwise construction raises `ValueError`. For
charged species set `charge=` alongside `multiplicity=`:

```python
oh_minus = vq.Molecule(
    atoms=[vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.83])],
    charge=-1,                               # OH⁻
    multiplicity=1,                          # singlet
)
```

### Restricted vs unrestricted open shell (UHF vs ROHF)

> See the dedicated [ROHF and ROKS](rohf.md) page for this family in depth:
> the ROHF-vs-UHF decision, analytic gradients, optimisation/frequencies,
> spin-pure CAS references, and periodic status.

For an open-shell system you can choose between unrestricted and
restricted open-shell mean-field references:

* **`run_uhf` / `method="uhf"`**, unrestricted: independent α and β
  spatial orbitals. Variationally lowest single determinant, but **not a
  spin eigenfunction** (⟨S²⟩ deviates from the ideal `S(S+1)`, 
  *spin contamination*).
* **`run_rohf` / `method="rohf"`**, restricted open shell: one set of
  spatial orbitals, doubly-occupied (closed) + singly-occupied (open) +
  virtual. **Spin-pure by construction** (⟨S²⟩ = `S(S+1)` *exactly*), at
  the cost of a slightly higher energy than UHF. This is the standard
  reference for spin-pure post-HF (ROHF-MP2 / ROHF-CC) and a cleaner CAS
  start than UHF.

```python
ch3 = vq.Molecule(
    atoms=[vq.Atom(6, [0, 0, 0]),
           vq.Atom(1, [+2.03, 0, 0]),
           vq.Atom(1, [-1.02, +1.76, 0]),
           vq.Atom(1, [-1.02, -1.76, 0])],
    multiplicity=2,                          # doublet radical
)
basis = vq.BasisSet(ch3, "6-31g*")
rohf = vq.run_rohf(ch3, basis)               # spin-pure: <S^2> = 0.75 exactly
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
roks = vq.run_roks(ch3, basis, functional="pbe")   # spin-pure KS
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
