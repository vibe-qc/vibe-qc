# Auto-citations: from `.out` to manuscript bibliography

You finished a hybrid-DFT/dispersion calculation, the energies look
right, and now the journal wants a bibliography. In most QC codes
that's the moment you fire up Google Scholar and start
cross-referencing your input file against the manuals. vibe-qc writes
the bibliography for you. This tutorial walks through the workflow
from input to ready-to-paste BibTeX.

We'll use a single small system, the water dimer with PBE0/D3(BJ)
on def2-TZVP, because it exercises a representative slice of the
citation database: a hybrid functional (two papers), a dispersion
correction (two papers), an Ahlrichs basis, libint, libxc, and
DIIS.

## The input

Save the script and geometry side by side:

```
~/vibeqc-runs/water-dimer/
    input-water-dimer.py
    water-dimer.xyz
```

`water-dimer.xyz`:

```text
6
H2O dimer — Cs symmetric, R(O...O) ≈ 2.91 Å
O   0.00000   0.00000  -1.45500
H   0.00000   0.75700  -2.06200
H   0.00000  -0.75700  -2.06200
O   0.00000   0.00000   1.45500
H   0.00000   0.57100   2.02900
H   0.00000  -0.57100   2.02900
```

`input-water-dimer.py`:

```python
from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE / "water-dimer.xyz")

run_job(
    mol,
    basis="def2-tzvp",
    method="rks",
    functional="PBE0",
    dispersion="d3bj",
    output=HERE / "output-water-dimer-pbe0",
)
```

Run it from a venv-activated shell:

```sh
python input-water-dimer.py
```

## What lands in the directory

After ~5 s on a modern laptop:

```
output-water-dimer-pbe0.out
output-water-dimer-pbe0.system
output-water-dimer-pbe0.molden
output-water-dimer-pbe0.xyz
output-water-dimer-pbe0.bibtex          # ← new in v0.8.x
output-water-dimer-pbe0.references      # ← new in v0.8.x
```

The `.out` file ends with a `## References` block; the `.bibtex`
sibling carries the BibTeX entries; the `.references` sibling has
a numbered Chicago-style plain-text version. We'll look at each in
turn.

## Reading the `.out` reference block

Open `output-water-dimer-pbe0.out` and scroll to the bottom:

```text
## References

Please cite the references below when reporting results
from this run. The corresponding BibTeX entries are
written to the .bibtex sibling.

  [1] Peintinger, Michael F. (2026). vibe-qc: a quantum-chemistry code for
     molecules and solids. [Software v0.8.0, MPL-2.0]. <https://vibe-qc.com/>

  [2] Valeev, Edward F. Libint: A library for the evaluation of molecular
     integrals of many-body operators over Gaussian functions. [Software].
     <https://github.com/evaleev/libint>

  [3] Weigend, Florian and Ahlrichs, Reinhart (2005). Balanced basis sets of
     split valence, triple zeta valence and quadruple zeta valence quality for
     H to Rn: design and assessment of accuracy. Physical Chemistry Chemical
     Physics, 7(18), 3297--3305. doi:10.1039/B508541A

  [4] Lehtola, Susi, Steigemann, Conrad, et al. (2018). Recent developments in
     libxc - A comprehensive library of functionals for density functional
     theory. SoftwareX, 7, 1--5. doi:10.1016/j.softx.2017.11.002

  [5] Perdew, John P., Burke, Kieron, and Ernzerhof, Matthias (1996).
     Generalized Gradient Approximation Made Simple. Physical Review Letters,
     77(18), 3865--3868. doi:10.1103/PhysRevLett.77.3865

  [6] Adamo, Carlo and Barone, Vincenzo (1999). Toward reliable density
     functional methods without adjustable parameters: The PBE0 model. Journal
     of Chemical Physics, 110(13), 6158--6170. doi:10.1063/1.478522

  [7] Pulay, Péter (1980). Convergence acceleration of iterative sequences.
     The case of SCF iteration. Chemical Physics Letters, 73(2), 393--398.
     doi:10.1016/0009-2614(80)80396-4

  [8] Pulay, Péter (1982). Improved SCF convergence acceleration. Journal of
     Computational Chemistry, 3(4), 556--560. doi:10.1002/jcc.540030413

  [9] Grimme, Stefan, Antony, Jens, et al. (2010). A consistent and accurate
     ab initio parametrization of density functional dispersion correction
     (DFT-D) for the 94 elements H-Pu. Journal of Chemical Physics, 132(15),
     154104. doi:10.1063/1.3382344

  [10] Grimme, Stefan, Ehrlich, Stephan, and Goerigk, Lars (2011). Effect of
     the damping function in dispersion corrected density functional theory.
     Journal of Computational Chemistry, 32(7), 1456--1465.
     doi:10.1002/jcc.21759
```

Each entry is independently justified by what the job did:

| # | Why it fired |
|---|---|
| [1] | Software citation, always first |
| [2] | libint, used for every two-electron integral |
| [3] | def2-TZVP, the basis set |
| [4] | libxc, fires whenever a functional is set |
| [5] | PBE component of PBE0 (Perdew-Burke-Ernzerhof 1996) |
| [6] | PBE0-specific hybrid mixing (Adamo-Barone 1999) |
| [7] | Pulay's original DIIS paper (default SCF accelerator) |
| [8] | Pulay's DIIS follow-up |
| [9] | Grimme's D3 base parametrisation |
| [10] | D3(BJ) damping-function extension |

The runtime walks the same routes every time. Swap PBE0 for B3LYP
and entries [5]/[6] are replaced by Becke 1993 + LYP 1988 +
Stephens 1994 + VWN 1980. Drop the dispersion keyword and [9]/[10]
disappear. Make it a periodic job and Togo-Tanaka 2018 (spglib)
joins the list.

## Pulling the BibTeX into your manuscript

`output-water-dimer-pbe0.bibtex` carries the same entries in BibTeX
form. Excerpt:

```bibtex
@software{peintinger_vibeqc,
  author      = {Peintinger, Michael F.},
  title       = {vibe-qc: a quantum-chemistry code for molecules and solids},
  year        = 2026,
  url         = {https://vibe-qc.com/},
  version     = {0.8.0},
  license     = {MPL-2.0},
  note        = {Always cite. A peer-reviewed publication is forthcoming; this software citation is the canonical reference until then.}
}

@article{weigend_ahlrichs_def2_2005,
  author      = {Weigend, Florian and Ahlrichs, Reinhart},
  title       = {Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H to Rn: design and assessment of accuracy},
  journal     = {Physical Chemistry Chemical Physics},
  volume      = 7,
  number      = 18,
  pages       = {3297--3305},
  year        = 2005,
  doi         = {10.1039/B508541A}
}

@article{adamo_barone_1999,
  author      = {Adamo, Carlo and Barone, Vincenzo},
  title       = {Toward reliable density functional methods without adjustable parameters: The PBE0 model},
  journal     = {Journal of Chemical Physics},
  volume      = 110,
  number      = 13,
  pages       = {6158--6170},
  year        = 1999,
  doi         = {10.1063/1.478522}
}

@article{grimme_d3bj_2011,
  author      = {Grimme, Stefan and Ehrlich, Stephan and Goerigk, Lars},
  title       = {Effect of the damping function in dispersion corrected density functional theory},
  journal     = {Journal of Computational Chemistry},
  volume      = 32,
  number      = 7,
  pages       = {1456--1465},
  year        = 2011,
  doi         = {10.1002/jcc.21759}
}
```

Drop the file into your LaTeX project's bibliography sources and
reference each entry by its `bibtex_key`:

```latex
% main.tex
\usepackage[backend=biber, style=chem-acs]{biblatex}
\addbibresource{output-water-dimer-pbe0.bibtex}

\begin{document}
Single-point energies were computed at the PBE0~\cite{adamo_barone_1999}
level with the D3(BJ)~\cite{grimme_d3_2010,grimme_d3bj_2011} dispersion
correction and the def2-TZVP~\cite{weigend_ahlrichs_def2_2005} basis set,
as implemented in vibe-qc~\cite{peintinger_vibeqc}.
\end{document}
```

That's the whole loop. `biber main` resolves every key against the
auto-generated `.bibtex` file. No hand-copying, no Google Scholar.

## Regenerating citations after the fact

If you lost the `.bibtex` / `.references` files (a cluster transfer
dropped them, the run predates the citation surface, …) the
`vibeqc-cite` console script reads the `.system` manifest and
re-emits them:

```sh
# Print the plain-text refs to stdout:
vibeqc-cite output-water-dimer-pbe0

# Or rewrite the sibling files:
vibeqc-cite output-water-dimer-pbe0 --write

# BibTeX only, to stdout (handy for grep / piping):
vibeqc-cite output-water-dimer-pbe0 --bibtex-only
```

The CLI loads the same database the SCF would have walked at run
time, so the output is bit-identical to what `run_job` would have
written. Useful for pre-v0.8.x runs whose manifests predate the
auto-citation work and for porting a paper draft to a different
machine without re-running every SCF.

## Combining bibliographies across runs

If your manuscript involves several runs (HF reference + PBE + PBE0,
say), append the `.bibtex` files into one source. `biber` is
deduplicating-aware as long as the keys match, and `bibtex_key`
values are deliberately stable across runs:

```sh
cat output-*.bibtex > manuscript-references.bibtex
```

Each file starts with the same `peintinger_vibeqc` + `valeev_libint`
preamble, but `biber` merges them. The same applies to anyone
combining a vibe-qc bibliography with their existing manuscript
references, keys never clash because the database uses
`<first_author>_<subject>_<year>` and the standard chemistry
literature naming convention picks the same authors.

## When a route is missing

If you use a feature that isn't in the bundled database yet, say
you pin a custom libxc id by passing `functional="xc_id_502"`, the
runtime emits a warning rather than a crash:

```text
# --- citation routing warnings ---
# no citation route for functional 'xc_id_502'
```

This block is appended to `.references`. The fix is in two layers:

1. **For your manuscript right now**: look up the specific functional
   in the libxc references list (`xc_func_info` on the libxc CLI, or
   the [`libxc` documentation](https://libxc.gitlab.io/)) and add the
   matching citation to your `.bibtex` by hand.
2. **For everyone else**: extend the database. The contract is
   spelled out in
   [Archived `AGENTS.md` § "Citation database ownership"](https://vibe-qc.com/docs/)
   and the schema in the [citations user guide](../user_guide/citations.md);
   patches welcome. If you added a functional to vibe-qc *and*
   forgot the database entry, CI will catch it via
   `tests/test_citations.py::test_required_functional_has_a_route`.

## Inspecting the bibliography without running SCF

For tutorial writing or manuscript drafting it's often useful to see
the bibliography for a planned calculation *before* running it. The
same machinery is exposed as a Python API:

```python
from vibeqc.output import OutputPlan
from vibeqc.output.citations import load_default_database

plan = OutputPlan.from_run_job_kwargs(
    output="preview",
    method="RKS",
    basis="def2-tzvp",
    functional="PBE0",
)

db = load_default_database()
cits = db.assemble_from_plan(plan, dispersion="d3bj")

for i, c in enumerate(cits, 1):
    print(f"[{i}] ({c.bibtex_key}) {c.title}")
```

prints the same ordered key list the SCF would have emitted. The
[citations user guide](../user_guide/citations.md) has the full API
reference, including the per-call flags for periodic / ECP / FFTW /
ASE.

## Resources

This tutorial completes in under 10 seconds on a modern laptop;
peak memory is <200 MB. The bibliography assembly itself is
O(routes) and runs in microseconds, it adds no measurable overhead
on top of the SCF.

## References

The papers above are themselves the references for this tutorial.
The auto-citation surface itself rests on:

- **vibe-qc citation database design.** See
  [`docs/design_output_module.md`](https://github.com/vibe-qc/vibe-qc/blob/main/docs/design_output_module.md)
  for the full schema and the v0.8.x → v1.0 roadmap.
- **The dev-chat contract.**
  [Archived `AGENTS.md` § "Citation database ownership"](https://vibe-qc.com/docs/)
  spells out the rule that adding a feature without updating the
  database fails CI.

## Next

- [Citations user guide](../user_guide/citations.md), schema,
  routing semantics, basissetdev sibling DB, and the Python API in
  full.
- [Input scripts and output files](../user_guide/output_files.md), 
  the full `.out` / `.system` / `.molden` / `.xyz` reference.
- [Citing vibe-qc](../citing.md), the canonical software citation
  plus backup tables for ad-hoc citations.
