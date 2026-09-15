# Good practices

A short catalog of working conventions that aren't obvious if
you've never operated a quantum-chemistry program from a clone.
None of these are vibe-qc-specific, they apply to running
Gaussian / ORCA / NWChem / PySCF / CP2K just as much, but
nobody tells you, and everybody learns the hard way. Read this
once before you start.

Use this page for operating habits. Use
[Planning a calculation](tutorial/planning_a_calculation.md) for the scientific
decision sequence and the error-budget worksheet.

## File organisation

**One project = one directory, outside the repo.** Vibe-qc writes
its outputs (`.out`, `.molden`, `.traj`, `.cube`, `.xsf`) into the
**current working directory**. If you run inside the cloned source
tree, those files land next to `cpp/` and `python/`, at best
clutter, at worst a `git clean` away from being deleted.

The convention used throughout these docs:

```text
~/vibeqc-runs/
├── water-pbe/
│   ├── water.py
│   ├── water.out
│   ├── water.molden
│   └── water.traj
├── h2o-trimer-mp2/
│   └── ...
└── lih-bulk-bands/
    └── ...
```

**Geometries in a sibling directory** if you reuse them across
methods, saves you from `cp h2o.xyz ../next-project/` every
time you start a comparison:

```text
~/vibeqc-runs/
├── geometries/
│   ├── h2o.xyz
│   ├── glycine.xyz
│   └── lih_4.5bohr.cif
├── water-pbe/
└── water-b3lyp/
```

**Keep the output family together.** The `.out` file is the readable record,
while `.system` is the machine-readable status and provenance manifest.
Retain `.references` and `.bibtex` with anything you may publish, plus `.qvf`,
geometry, trajectory, or checkpoint artifacts needed for later analysis.
Moving only the final energy into a spreadsheet discards the information
needed to reproduce it.

## Naming

**Convention:** ``<system>_<method>_<basis>.py``. Six months
later you can ``ls | grep glycine`` and find every variant you
ran:

```text
glycine_rhf_631gss.py
glycine_rks_pbe_def2tzvp.py
glycine_uks_b3lyp_def2tzvp.py
glycine_mp2_ccpvdz.py
```

Match the ``output=`` argument so the side-effect files line up:

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("glycine.xyz")
run_job(mol, basis="def2-tzvp", method="rks", functional="PBE",
        output="glycine_rks_pbe_def2tzvp")
# produces: glycine_rks_pbe_def2tzvp.out
#           glycine_rks_pbe_def2tzvp.molden
```

For periodic runs, encode the k-mesh too:
``lih_rks_pbe_sto3g_k4x4x4.py``.

## Reproducibility

**Pin a tagged version for any calculation going into a paper.**
``git checkout v0.15.0`` (or whichever release you're targeting)
**before** running ``setup_native_deps.sh`` and ``pip install``.
The banner will then read ``Release v0.15.0`` instead of the
moving ``main`` target, and your numbers stay reproducible if a
future commit changes a default. See
[picking a build to test against](installation.md#picking-a-build-to-test-against)
for the full workflow.

**Version-control the input script alongside the manuscript draft.**
Treat ``glycine_mp2_ccpvdz.py`` like Methods-section text, it
*is* the methods section, in executable form.

**Save the banner.** Every ``.out`` file starts with a labeled
box recording vibe-qc version, codename, git revision, dirty-tree
flag, and linked native-library versions:

```text
vibe-qc <version> "<codename>"
git: <sha> (<branch>[, dirty])
libint: <version>  libxc: <version>  spglib: <version>
```

That single block is your provenance line, don't strip it when
copy-pasting into a SI appendix. The version + git revision + linked
library versions together pin the *exact* binary that produced the
numbers, down to the libint / libxc / spglib commits.

If you used uncommitted local changes, the banner will read
``dirty`` (between the SHA and the closing parenthesis). **Don't
ship dirty-tree numbers in a paper.** Either commit + tag and
re-run, or document the diff in your SI.

See [example scripts and generated outputs](example_outputs.md) for
canonical calculations you can rerun locally.

## Performance hygiene

**Always set ``OMP_NUM_THREADS`` explicitly.** OpenMP defaults to
"every core on the machine," which is rude on shared nodes and
pessimal on workstations with hyperthreaded cores (you usually
want physical cores, not logical):

```sh
OMP_NUM_THREADS=4 ~/path/to/vibeqc/.venv/bin/python water.py
```

**Estimate memory before a large run.** Vibe-qc has a pre-flight
memory budget estimator, it'll tell you the integral-storage
footprint before the SCF starts, so you don't OOM 30 minutes in.
See [the memory user guide](user_guide/memory.md). For molecular SCF, compare
the $8N_\mathrm{bf}^4$ conventional tensor with direct SCF before reducing a
scientifically necessary basis. The
[direct-SCF tutorial](tutorial/direct_scf_memory_tradeoff.md) shows how CPU
time, wall time, and peak RSS should be measured together.

**Smallest basis that answers the question.** DZ before TZ before
QZ. Run a basis-convergence sweep on a small representative
system (5-10 minutes) before committing to a 12-hour QZ
production run. Same for k-meshes (3×3×3 → 6×6×6 → 8×8×8) and
DFT grids, each parameter has a noise floor, find it once per
project.

## Trusting a number

**Converge one approximation at a time against the uncertainty your result
needs.** There is no universal 1 mEh threshold and no universal order for
basis, grid, k mesh, geometry, and method. A reaction energy, force, frequency,
and band gap can respond very differently to the same numerical change.

For a sequence $Q_1,Q_2,\ldots$, record

$$
\delta_i = Q_i-Q_{i-1}
$$

alongside CPU time and peak RSS. Hold every other choice fixed while measuring
$\delta_i$, then stop when it is small compared with the uncertainty allowed
in the final conclusion. Periodic calculations normally converge the Coulomb
route and real-space cutoffs before interpreting a k-mesh or smearing sweep.
The full protocol is in [Planning a calculation](tutorial/planning_a_calculation.md).

**Cross-check against an independent implementation.** Pick a small variant
of the real system and run both programs as separate processes. Match geometry
units, charge, multiplicity, basis contractions, ECPs, frozen core, functional,
dispersion, grids, Coulomb convention, k points, smearing, and convergence
thresholds before interpreting a difference. The external program is a
reference calculation, never a vibe-qc runtime dependency. See
[Cross-validation](tutorial/cross_validation.md).

**Sanity-check the symmetry.** If you set up a high-symmetry
system but the SCF result has a tiny dipole or a spurious
splitting, your input geometry probably has noise in the last few
decimals. Snap to the symmetry first, then run.

## Long-running calculations

**Always capture stdout to a logfile.** SSH connections drop;
terminal scrollback is finite. ``tee`` keeps a copy on disk
alongside the run:

```sh
~/path/to/vibeqc/.venv/bin/python water.py 2>&1 | tee water.log
```

**For anything over 10 minutes, detach.** ``nohup`` is the
minimum:

```sh
nohup ~/path/to/vibeqc/.venv/bin/python water.py > water.log 2>&1 &
```

For multi-day runs use ``screen`` or ``tmux`` so you can
reattach and see the live output:

```sh
screen -dmS mycalc ~/path/to/vibeqc/.venv/bin/python water.py
screen -r mycalc          # re-attach later
# Ctrl-A, D to detach again
```

For a batch of remote jobs, use the shipped [vq queue](user_guide/queue.md)
instead of maintaining PIDs manually. See [the running guide](running.md) for
the local, remote, and scheduler workflows.

## When SCF diverges

First classify the calculation. Molecular and periodic symptoms do not have
the same escalation path.

### Molecular calculation

Simplify until a controlled reference converges, then add complexity back one
change at a time:

1. Verify geometry units, charge, multiplicity, electron count, and intended
   electronic state.
2. Test a smaller basis in the same family. If the smaller basis converges,
   inspect overlap eigenvalues and diffuse functions before changing the SCF
   algorithm.
3. Test a documented initial guess appropriate to the state.
4. Inspect whether two states or occupation patterns are competing.
5. Only then use the molecular convergence aids in
   [SCF convergence](user_guide/scf_convergence.md), recording every override.

Increasing `max_iter` is useful only when the residual is steadily falling.
It does not cure a plateau, two-cycle, wrong state, or linearly dependent
basis.

### Periodic calculation

An unexplained periodic oscillation, impossible absolute energy, or stationary
point that disagrees with a matched reference is a possible implementation
bug. Preserve the input and trace; do not hide the symptom with damping,
level shifting, quadratic fallback, or threshold tuning.

Smearing is appropriate when a physically metallic calculation has a
verified Fermi-surface occupation problem. Treat the smearing temperature and
k mesh as part of the model, converge both, and report whether the compared
quantity is energy or free energy. Smearing is not a generic repair for an
insulator or for a gauge, Madelung, image-summing, or Coulomb-route defect.
See [Periodic SCF convergence](tutorial/periodic_scf_convergence.md) and
[Troubleshooting](troubleshooting.md#scf-didnt-converge).

## Common gotchas

| Symptom | Likely cause | Fix |
|---|---|---|
| ``ModuleNotFoundError: No module named 'vibeqc'`` | Ran the wrong Python (system, not venv) | Use ``~/path/to/vibeqc/.venv/bin/python`` or activate the venv. See [running](running.md). |
| ``.venv/bin/python: no such file or directory`` | Ran from outside the repo with the relative-path shorthand | Use the absolute path; the venv lives where you cloned. |
| Wildly wrong energy (factor-of-2-off, or sign-flipped) | Bohr vs Ångström unit confusion in coordinates | Vibe-qc internals are bohr; pass Ångström via ``Molecule.from_xyz()``. |
| Wrong number of electrons | Charge / multiplicity mismatch | ``print(mol.n_electrons())``, does it match what you expect? |
| ``.cube`` won't open in Avogadro | File is 0 bytes (disk filled during write) | ``df -h ~`` and ``ls -la *.cube``. |
| ``Could not find Libxc / Libint2 / FFTW3`` at import | Vendored install missing or interrupted | Re-run ``./scripts/setup_native_deps.sh``, finished deps short-circuit. |
| Molecular SCF energy oscillates between two values | competing states, poor guess, basis pathology, or accelerator failure | verify state and basis first, then follow [molecular SCF convergence](user_guide/scf_convergence.md) |
| Periodic SCF oscillates or reaches an impossible energy | possible gauge, Madelung, image-summing, or route defect | preserve a minimal reproducer and compare with a matched reference; do not mask it with convergence aids |
| ``dirty`` flag in your output banner | Local uncommitted changes | Either commit + tag, or document the diff in your SI before publishing. |

## Backups

The boring rule that everyone learns the hard way: **back up
``~/vibeqc-runs/``** like you back up everything else. The
input scripts cost minutes to rewrite; the converged
calculations cost hours-to-days. Keep them.

A throwaway one-liner for nightly tarball + offsite ``rsync``:

```sh
tar czf ~/backup/vibeqc-runs-$(date +%Y%m%d).tar.gz ~/vibeqc-runs
rsync -av ~/backup/ user@backup-host:~/vibeqc-backup/
```

Or use whatever your group's existing backup story is, but
**have one**.
