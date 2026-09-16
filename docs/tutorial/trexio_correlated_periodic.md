# TREXIO for CI wavefunctions and periodic restarts

After the [molecular exchange tutorial](trexio_exchange.md), two further
checks matter: keeping CI coefficients attached to their actual orbital
basis, and keeping a periodic density attached to its k-point sampling.
Both examples below retain the default QVF output and request TREXIO
explicitly. Run them from a source checkout using an environment with
`vibe-qc[trexio]` installed.

## LiH with a frozen core and an active space

This example uses LiH at a separation of 3 bohr, STO-3G, and CASSCF with two
active electrons in two active orbitals. One spatial core orbital remains
doubly occupied. This is a small interchange model, not a basis-converged
LiH benchmark.

```sh
export OMP_NUM_THREADS=1
python examples/trexio/correlated_wavefunction.py --method casscf \
    --output-dir ~/vibeqc-runs/trexio/lih-casscf
python examples/trexio/correlated_wavefunction.py --method casscf --backend text \
    --output-dir ~/vibeqc-runs/trexio/lih-casscf-text
```

Download the complete
{download}`correlated_wavefunction.py <../../examples/trexio/correlated_wavefunction.py>`.
Its calculation is:

```{literalinclude} ../../examples/trexio/correlated_wavefunction.py
:language: python
:start-after: "# BEGIN calculation"
:end-before: "# END calculation"
:dedent: 4
```

The automatic export contains the determinant expansion and optimized
CASSCF orbitals together. Replacing those orbitals with the original RHF
coefficients changes the represented wavefunction even if the determinant
coefficients are unchanged. Frozen core occupation must also be present in
every determinant.

The script checks normalization of the CI vector, spin-resolved RDM traces,
the four-electron AO density, and the frozen-core bits:

```{literalinclude} ../../examples/trexio/correlated_wavefunction.py
:language: python
:start-after: "# BEGIN checks"
:end-before: "# END checks"
:dedent: 4
```

For this small example there are six spatial MOs, so one 64-bit word per
spin suffices. Do not copy its one-word bit check into a general determinant
reader without handling larger orbital spaces.

The total energy should be about **-7.881214343110 Ha**. The output ends in
`PASS casscf`, with four determinants and CI norm close to one. Orbital
energies are absent because canonical SCF eigenvalues would not describe
this optimized correlated orbital set. The density comes from the full
stored 1-RDM, including its off-diagonal elements.

Change `--method` to `casci` for a fixed-orbital active-space calculation or
`fci` for all four electrons in all six orbitals. FCI does not use the
`active_space` argument and is only inexpensive here because the model is
tiny. The script accepts both backends for every method.

### Independent CI energy reconstruction

With PySCF installed in the reference interpreter:

```sh
python examples/regression/runner_trexio_pyscf.py \
    ~/vibeqc-runs/trexio/lih-casscf/lih-casscf.trexio.h5
```

The checker forms a Hamiltonian from its own integrals and contracts it with
the stored CI vector. Its result must report `"wavefunction": "CI"`,
`"verdict": "pass"`, and an energy difference below `1e-8` Ha. It does not
rerun a separate CASSCF optimization. That distinction lets it test whether
the file describes the original state.

The job exporter selects root zero. Use the low-level `root=` option for a
different computed state; see [CI and CASSCF wavefunctions](../user_guide/trexio.md#ci-and-casscf-wavefunctions).
A CI density can seed an SCF calculation, but READ is not a continuation of
the determinant solver or CASSCF optimizer. Coupled-cluster amplitudes and
higher RDMs require explicitly supplied fields rather than automatic export.

## A periodic two-k-point wavefunction

The periodic example uses one He atom in a cubic cell of side 7 bohr,
RHF/STO-3G, and a `(2, 1, 1)` k-point mesh. It deliberately uses a small GDF
cutoff and disables automatic cutoff convergence to keep the IO exercise
small. SCF convergence is still required. Its energy is not a converged
prediction for solid helium.

```sh
python examples/trexio/periodic_restart.py \
    --output-dir ~/vibeqc-runs/trexio/helium
python examples/trexio/periodic_restart.py --backend text \
    --output-dir ~/vibeqc-runs/trexio/helium-text
```

Download {download}`periodic_restart.py <../../examples/trexio/periodic_restart.py>`.
The same settings are used for the original calculation and the restart:

```{literalinclude} ../../examples/trexio/periodic_restart.py
:language: python
:start-after: "# BEGIN calculation"
:end-before: "# END calculation"
:dedent: 4
```

### Keep the sampling with the orbitals

```{literalinclude} ../../examples/trexio/periodic_restart.py
:language: python
:start-after: "# BEGIN checks"
:end-before: "# END checks"
:dedent: 4
```

There are two restricted orbital blocks, identified by `k_point` 0 and 1.
The per-cell electron count is
$N=\sum_k w_k\sum_i f_{ik}=2$, not the unweighted sum over both blocks.
For an unrestricted calculation, include both spin blocks at each k point.
The reader's densities are also unweighted; apply the weights when
integrating a cell quantity.

`data.lattice` uses column vectors in bohr. `data.kpoints` gives Cartesian
inverse-bohr vectors, while raw `pbc_k_point` holds reduced coordinates.
Complex coefficients remain complex even when this particularly small
case happens to have real-valued orbitals. A molecular overlap matrix is
not the Bloch overlap at a nonzero k point; the molecular overlap check
from the first tutorial must not be reused here.

### READ with the same target mesh

```{literalinclude} ../../examples/trexio/periodic_restart.py
:language: python
:start-after: "# BEGIN restart"
:end-before: "# END restart"
:dedent: 4
```

The output should end in `PASS periodic`, a weighted electron count of 2,
and a READ energy difference below `1e-8` Ha. The second run creates
`helium-read.qvf` and the ordinary job files. The first run's TREXIO file
remains the explicit restart source.

The target lattice, basis, k points and weights must be compatible with the
source; READ can map a different ordering of the same mesh, but it does not
interpolate a density onto a new mesh. Multi-k AO integral arrays are omitted
because the corresponding TREXIO datasets have no k-index dimension.
Foreign one- or two-dimensional files need an explicit `dim=` when calling
`periodic_system()` unless they carry vibe-qc's dimension metadata.

The molecular PySCF checker rejects periodic files. This example verifies
preservation and restart consistency; it does not establish independent
periodic integral agreement. For physical convergence, continue with
[periodic SCF convergence](periodic_scf_convergence.md) and
[Bloch orbitals and k points](kpoints_brillouin_bloch.md).
