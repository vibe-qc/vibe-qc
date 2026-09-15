---
myst:
  html_meta:
    "description": "Use Claude (the AI assistant) to drive vibe-qc calculations: point Claude at the docs, describe your chemistry in plain English, get a runnable input. End-to-end worked example on α-Al2O3 corundum."
    "og:title": "Using Claude to drive vibe-qc, talk chemistry, get a calculation"
    "og:description": "How to use Claude as the front-end to vibe-qc. URL-paste workflow, conversation patterns, worked corundum example, pitfalls."
    "og:type": "website"
    "og:url": "https://vibe-qc.com/docs/using_claude.html"
---

# Using Claude to drive vibe-qc

vibe-qc is a Python-first quantum-chemistry code, but most chemists
think in chemistry, not in `Atom(8, [0.0, 0.0, 0.0])`. This page
shows how to use **Claude**, an AI assistant, to bridge that gap:
describe your system in plain English, get a runnable input back.

vibe-qc was itself [built with Claude](https://vibe-qc.com/2026/04/22/i-vibe-coded-a-quantum-chemical-program-in-one-day/),
so the workflow is well-tested in both directions, Claude is good
at writing vibe-qc inputs because vibe-qc was written for and with
Claude.

## Setup: point Claude at the docs (the key step)

The single most important habit is to **give Claude the URL of the
documentation site at the start of every session**. Don't rely on
Claude's training knowledge of vibe-qc, pin it to the live docs.

Open your Claude surface of choice ([claude.ai](https://claude.ai),
[Claude Code](https://claude.com/claude-code), or the
[Anthropic API](https://docs.claude.com/)) and paste something like:

> *Please read [https://vibe-qc.com/](https://vibe-qc.com/) and use
> it as the reference for vibe-qc. I want to run a calculation; I'll
> describe what I need and you write the input.*

Claude fetches the index, follows the toctree, and now has access
to the API, tutorials, basis-set guide, user-guide chapters, and
the example library. This is the difference between *Claude making
things up* and *Claude looking it up*. **Don't skip this step.**
It's worth ~90% of the quality difference between a useful Claude
session and a frustrating one.

```{tip}
**The docs site mirrors `release` (the latest tagged version).** If
you're on `main` (development), give Claude the GitLab repo URL
instead — `https://github.com/vibe-qc/vibe-qc/` — so it can
read the in-flight changes. Tell it which branch you're on.
```

## A first session, end-to-end: corundum (α-Al₂O₃)

A worked example with a real-world ambiguity. The user is a
materials chemist who wants to compute properties of corundum and
isn't a Python expert; they let Claude pick the chemistry and write
the code.

```{admonition} Conversation pattern
:class: note

The dialogue below is paraphrased — exact phrasing varies — but the
*shape* of the exchange is what matters: ask in chemistry language,
let Claude interrogate the docs, refine, iterate.
```

### Turn 1, "What method should I use?"

> **You:** I want to run a calculation for corundum (α-Al₂O₃). What
> method would you recommend?

Claude reads `docs/user_guide/`, the tutorials, and the support page,
then answers something like:

> **Claude:** Corundum is a wide-gap (≈9 eV) ionic insulator, 
> exactly the regime vibe-qc's solid-state stack is designed for.
> A reasonable starting point:
>
> - **Method:** Kohn-Sham DFT with the PBE functional. Hybrid
>   functionals (B3LYP, ωB97X-D) give better band gaps but cost
>   ~5× more per Fock build.
> - **Basis:** **pob-TZVP-rev2**, designed specifically for
>   periodic ionic crystals (Vilela Oliveira et al. 2019). The pob
>   family ships with vibe-qc and is what
>   [Solid-state walkthrough: LiH rocksalt with `pob-TZVP`](tutorial/lih_pob_tzvp_solid_state.md)
>   walks through.
> - **k-mesh:** start with 4×4×4 Monkhorst-Pack on the conventional
>   cell; converge to 8×8×8 for production.
>
> One caveat I want to flag before writing code: **α-Al₂O₃ is
> hexagonal** (R-3c), so the input must use the real hexagonal lattice
> vectors, not the older cubic stuffed-fluorite smoke-test geometry.
> vibe-qc's periodic lattice infrastructure is Born-von-Karman general
> (triclinic through cubic); for production parity we should run the
> native GDF/FFTDF route and validate side-by-side against external
> CRYSTAL / PySCF outputs.
>
> For a quick smoke test only, there is still a small cubic Al₂O₃ model
> at
> [`examples/periodic/input-al2o3-cubic-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-al2o3-cubic-rhf.py),
> but it is not corundum and should not be used as the final physics
> model.
>
> Do you want the real corundum validation input or the cheap cubic
> smoke-test input?

The lesson: **Claude checked the lattice/model distinction before
writing input**, instead of silently swapping real corundum for a cubic
surrogate. This is the kind of catch that makes the URL-paste setup
worth it.

### Turn 2, "Let's do the smoke test first."

> **You:** Let's do the cheap cubic smoke test first. Can you fetch the
> structure and create the input?

> **Claude:** I'll mirror the cubic stuffed-fluorite model that
> ships at
> [`examples/periodic/input-al2o3-cubic-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-al2o3-cubic-rhf.py),
> with PBE instead of RHF and pob-TZVP-rev2 instead of STO-3G:
>
> ```python
> # input-al2o3-cubic-pbe-pob.py
> from pathlib import Path
> import numpy as np
> import vibeqc as vq
>
> ANG2BOHR = 1.0 / 0.529177210903
> A_ANG    = 5.05
> A        = A_ANG * ANG2BOHR
> al_frac  = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
> o_frac   = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
>
> unit_cell = (
>     [vq.Atom(13, [fx*A, fy*A, fz*A]) for fx, fy, fz in al_frac]
>     + [vq.Atom( 8, [fx*A, fy*A, fz*A]) for fx, fy, fz in o_frac]
> )
> sysp  = vq.PeriodicSystem(3, np.diag([A, A, A]), unit_cell)
> basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp-rev2")
> kpts  = vq.KPoints.monkhorst_pack(sysp, [4, 4, 4], symmetry=True)
>
> opts = vq.PeriodicKSOptions()
> opts.functional       = "pbe"
> opts.lattice_opts.coulomb_method      = vq.CoulombMethod.EWALD_3D
> opts.lattice_opts.cutoff_bohr         = 14.0
> opts.lattice_opts.nuclear_cutoff_bohr = 25.0
> opts.conv_tol_energy = 1e-7
> opts.max_iter        = 60
> opts.initial_guess   = vq.InitialGuess.SAD
>
> result = vq.run_rks_periodic_scf(sysp, basis, kpts, opts)
> print(f"E = {result.energy:.6f} Ha   converged={result.converged}")
> ```

### Turn 3, "What options do I have?"

> **You:** Before I run it, what knobs are there? What can I tune?

> **Claude:** The high-leverage knobs, roughly in order of importance:
>
> | Knob | Default | When to change |
> |---|---|---|
> | `opts.functional` | (required) | Try `"b3lyp"` for better gaps; `"pbesol"` for solids; `"ωb97x-d"` for noncovalent |
> | `kpts` mesh | 4×4×4 | Tighten to 6×6×6 / 8×8×8 to converge band properties |
> | `cutoff_bohr` | 10.0 | Tighten if linear-dependence preflight warns |
> | `auto_optimize_truncation` | True (v0.7+) | Leave on; it bisects cutoffs jointly |
> | `damping` | 0.85 | Lower (0.5) if SCF oscillates; see [Periodic SCF convergence: damping, DIIS, and level shifts](tutorial/periodic_scf_convergence.md) |
> | `initial_guess` | SAD | HCORE if SAD fails on heavy elements |
> | `dispersion` | none | `"d3bj"` if you care about packing energies |
>
> Full reference is in
> [user guide § SCF convergence](user_guide/scf_convergence.md) and
> [user guide § linear dependence](user_guide/linear_dependence.md).

### Turn 4, "What basis should I use?"

> **You:** Why pob-TZVP-rev2 specifically? What else could I use?

> **Claude:** Per the
> [linear-dependence chapter](user_guide/linear_dependence.md)
> and the [basis-set survey](user_guide/basis_sets.md):
>
> - **pob-TZVP-rev2**, primary recommendation. Diffuse primitives
>   removed exactly so that the AO overlap matrix stays
>   well-conditioned in tight ionic cells. Cite Vilela Oliveira et
>   al., *J. Comput. Chem.* **40**, 2364 (2019).
> - **pob-TZVP**, the original (Peintinger 2013). Solid for most
>   ionic systems; a few elements got tightened in -rev2.
> - **pob-DZVP-rev2**, smaller, faster; use for screening or when
>   you don't need TZ-quality wavefunction.
> - **STO-3G**, only for parity tests against PySCF.pbc; not
>   chemistry-quality.
> - **def2-TZVP / cc-pVTZ**, *don't* use for solids without
>   `vq.make_basis(..., exp_to_discard=0.1)`. Their diffuse tails
>   blow up the overlap matrix in dense crystals.

### Turn 5, "Can I run a geometry optimization?"

> **You:** Can I optimize the geometry?

> **Claude:** Periodic atomic gradients shipped in v0.6.0
> (codename *Pulay's Owl*) and the Γ-only / multi-k drivers are
> available at `vq.compute_gradient_periodic_rks_multi_k`. So yes,
> for atoms-within-the-cell relaxation. **Lattice parameter
> optimisation** (the stress tensor, v0.8 roadmap item) isn't shipped
> yet, for now you'd scan the lattice constant manually and fit an
> equation of state.
>
> The cleanest path right now is via ASE:
>
> ```python
> from vibeqc.ase_periodic import atoms_to_periodic_system, periodic_forces
> from ase.optimize import BFGS
> # ... build atoms ...
> # set calculator that returns periodic_forces; relax internal coords
> ```
>
> The `neb_reaction_path` tutorial on NEB has the closest pattern; let me know if you
> want me to adapt it.

### Turn 6, running and interpretation

> **You:** Save the input, run it, and tell me what the output says.

(Claude Code can run the script directly; on claude.ai you'd run it
locally and paste the `.out` back.)

> **You (after running):** Here's the .out file. What converged?
> [pastes contents of `input-al2o3-cubic-pbe-pob.out`]

> **Claude:** Reading the output:
>
> - **SCF**: converged in 11 iterations to energy difference 4.2e-8
>   Ha. Active settings dump confirms `auto_optimize_truncation`
>   tightened `cutoff_bohr` from 14.0 → 16.5 to keep the overlap
>   PSD at all 8 IBZ k-points.
> - **Linear dependence**: preflight = `warn` (λ_min ≈ 8.4e-6 at
>   the X-point); the auto-optimiser handled it.
> - **Total energy**: -549.832 Ha per cell. With pob-TZVP-rev2 and
>   PBE on this 5-atom Al₂O₃ stoichiometry model, that's
>   ≈110 Ha/atom, in the right ballpark for the basis + functional.
> - **Band gap** (from the orbital table): 5.7 eV at Γ. PBE
>   under-binds gaps; the experimental gap of corundum is ~9 eV, so
>   the ~3 eV under-bind is the textbook PBE error. A hybrid
>   (B3LYP / HSE06) would close most of that.
>
> No warnings beyond the expected linear-dep tightening. The .system
> manifest shows it ran on your machine in 73 s.

That's a complete first session. The chemist asked about chemistry;
Claude wrote and ran the code; the output was interpreted in
plain English with quantitative caveats about what to trust.

## A second session, Claude Code in the loop: water dimer with dispersion

The corundum example focuses on the *conversation pattern* (read →
recommend → write); this one shows the complete **Claude Code**
loop where Claude also runs the script and reads the output back, 
no copy-paste between you and the model. The chemistry is simpler
(no FFT-Poisson caveat to dodge), so the dialogue is shorter; the
point is the execution loop.

### Turn 1, the question

> **You:** Compute the H-bond binding energy of the water dimer at
> PBE / cc-pVTZ + D3(BJ). I want to know how much D3 contributes
> versus the bare DFT number.

> **Claude:** A few framing points before I write code:
>
> - **What we're computing.** The H-bond binding energy
>   $\Delta E = E_\text{dimer} - 2\,E_\text{monomer}$, with both
>   numbers from a converged geometry optimisation.
> - **What D3 does here.** H-bonds are dominated by electrostatics
>   + induction; London dispersion contributes a smaller, but real,
>   correction, typically 0.3-0.7 kcal/mol on the water dimer.
>   D3(BJ) captures it cheaply on top of any DFT functional. PBE
>   without dispersion under-binds the dimer by ~10-15 %.
> - **Reference number.** [Klopper et al.](https://doi.org/10.1063/1.480168)
>   gives 5.02 kcal/mol at CCSD(T)/CBS. Anything within ~0.5 kcal
>   of that is in the right ballpark.
> - **A caveat I want to flag.** I'm *not* doing BSSE counterpoise
>   in this run. With cc-pVTZ on water dimer the BSSE is around
>   0.3-0.5 kcal/mol, non-trivial relative to D3. If you want
>   publication-quality binding energies, ask me to add the
>   counterpoise step (or use pob-DZVP-rev2 which is much less
>   BSSE-prone in solid-state contexts).
>
> Plan: two `run_job` calls, both `optimize=True` so we get the
> relaxed geometry at the chosen level. I'll run each twice, once
> bare PBE, once PBE-D3(BJ), and assemble the comparison at the
> end. Sound good?

### Turn 2, go

> **You:** Yes, run it.

```{admonition} What Claude Code does next
:class: note

The Claude Code CLI has direct access to `Bash`, `Write`, and
`Edit` tools, so it writes the files into your current working
directory and runs them inline. On `claude.ai` you'd save the
scripts yourself and paste the outputs back; the result is the
same, the round-trip is just split.
```

Claude writes `input-water-dimer-pbe.py` and
`input-water-monomer-pbe.py` (optimisation, no D3), then their
`-d3bj` siblings. Each is ~15 lines, follows the
`output=HERE / Path(__file__).stem` convention so outputs land
next to the input. It runs all four (~30-90 s wall-clock per job
on a current laptop with cc-pVTZ), watches the SCF traces stream
by, and surfaces the converged energies once everything finishes.

### Turn 3, the answer

> **Claude:** All four converged. Pulling the numbers out of the
> `.out` files:
>
> | Run | E (Ha) | converged | iters |
> |---|---|---|---|
> | `input-water-monomer-pbe.out`     | -76.345182 | ✓ | 11 |
> | `input-water-monomer-pbe-d3bj.out` | -76.346014 | ✓ | 11 |
> | `input-water-dimer-pbe.out`       | -152.698220 | ✓ | 13 |
> | `input-water-dimer-pbe-d3bj.out`  | -152.700928 | ✓ | 13 |
>
> Binding energies (1 Ha = 627.51 kcal/mol):
>
> $$
> \Delta E_\text{PBE}    = -152.6982 - 2 \times (-76.3452) = -0.00786 \text{ Ha} = -4.93 \text{ kcal/mol}
> $$
> $$
> \Delta E_\text{PBE-D3} = -152.7009 - 2 \times (-76.3460) = -0.00892 \text{ Ha} = -5.59 \text{ kcal/mol}
> $$
>
> **D3 contribution: 0.66 kcal/mol**, ~12% of the binding, right
> in the textbook 0.3-0.7 kcal/mol range for water-dimer dispersion.
>
> **Compared to reference (CCSD(T)/CBS, 5.02 kcal/mol):**
> - PBE bare: -4.93 kcal/mol → under-binds by ~0.1 kcal/mol
> - PBE-D3: -5.59 kcal/mol → over-binds by ~0.6 kcal/mol
>
> That over-binding is the BSSE I flagged, counterpoise would
> typically peel ~0.4-0.5 kcal/mol off the cc-pVTZ binding,
> which would put PBE-D3 within ~0.2 kcal/mol of the reference.
> If you want me to add the counterpoise step, just say.

That's a research-grade answer in three turns. The chemist never
opened a Python file; the .py scripts are still there next to the
.out files for reproducibility / handoff to a collaborator.

## Workflow patterns

The corundum example covers most of what you need; here are the
common variants in shorter form.

### Single-point energies

> *I want the RKS-B3LYP / 6-31G* energy of methanol.*

Claude writes a `Molecule` with the methanol coordinates (it knows
the standard equilibrium geometry, but **always sanity-check**:
ask for the bond lengths back), uses
[`run_job`](api/index.md) for one-line
SCF + .out + .molden + .system output.

### Geometry optimisation (molecular)

> *Optimise the H₂O dimer at HF/6-31G* and tell me the H-bond
> length.*

`run_job(..., optimize=True)` does the BFGS opt + final SCF in one
call; .traj animates in `ase gui input-foo.traj`. Claude reads the
final geometry from the .out and reports the H···O distance.

### Functional sweep

> *Compare PBE / B3LYP / ωB97X-D on benzene's HOMO-LUMO gap.*

Claude generates a small driver loop (one job per functional, all
writing to per-functional output stems), then assembles the
comparison table from the `.out` files.

### Basis-set convergence

> *Show me the cc-pVXZ convergence of the H₂O atomisation energy.*

Same pattern, driver script, one row per basis, summary table.
Claude knows the cc-pVDZ → cc-pVQZ → CBS extrapolation pattern
from [Basis-set convergence](tutorial/basis_convergence.md).

### Vibrational frequencies + thermochemistry

> *Compute the IR spectrum of formaldehyde at HF/6-31G\*\* and tell
> me the harmonic ZPE.*

`run_job` doesn't (yet) bundle a Hessian path, so Claude wires
[`compute_hessian_rhf_analytic`](api/index.md) directly per
[Vibrational frequencies](tutorial/vibrational_frequencies.md). Output is a
table of frequencies + IR intensities.

### Periodic SCF on a crystal

The corundum walk-through above is the full version. Variants:

> *Compute the band structure of silicon.*
>
> *What's the cohesive energy of NaCl with pob-TZVP?*

For each, Claude builds the unit cell, picks a sensible k-mesh,
runs the SCF, then either follows up with `KPoints.band_path` (for
bands) or runs the dimer/atom references (for cohesive energies).

## What to paste back to Claude

Anything text-formatted is fair game and helpful:

- **`.out` files**, perfect for "what does this say" / "is it
  converged" / "what's the HOMO-LUMO gap" questions.
- **`.system` manifests**, useful if you're asking reproducibility
  questions (which build, which native-dep versions, which thread
  count).
- **`.perf` files**, per-iteration wall clock; pasteable for
  "why is this slow" questions.
- **The full structured-log NDJSON**, if you've enabled
  `structured_log=True`, the file is machine-readable and Claude
  can answer aggregate questions like "what was the largest energy
  step in this SCF".
- **A diff between two outputs**, `diff -u run-a.out run-b.out`,
  paste it; ask "what changed and is it significant".

## What Claude can't see

- **Binary files**, `.molden`, `.cube`, `.traj`. These need a
  viewer (Avogadro, VMD, `ase gui`); Claude can suggest the right
  command, but can't render the result. Workaround: tell Claude
  what you're seeing in the viewer, ask for interpretation.
- **Plots / images**, most surfaces support image upload but not
  all. If yours doesn't, paste a description of the plot
  (e.g. "the SCF energy oscillates between -75.6 and -75.8 every
  3 iterations after iteration 12") and Claude can reason about
  the dynamics.
- **`.scf-trace.npz` / numpy binaries**, load them in Python first,
  paste the output of `np.array_str(...)`.

## Pitfalls and how to dodge them

```{warning}
**Always verify chemistry references** before publishing. Claude
can hallucinate experimental band gaps, bond lengths, and DOIs.
The same goes for "PBE gives X eV gap on system Y" — if it
matters for your paper, look it up.
```

- **Claude seems to be making things up about vibe-qc.** Re-paste
  the documentation URL and ask Claude to confirm by quoting the
  relevant section. If the docs got updated since Claude's last
  fetch, give it the specific page URL.

- **Method / basis / functional name typos.** vibe-qc accepts
  `"pbe"` and `"PBE"` for functionals, `"6-31g*"` and `"6-31G*"`
  for Pople-style bases, but for some XC names libxc is strict.
  When in doubt, ask Claude to point at the
  [user guide functionals page](user_guide/functionals.md) and
  copy the exact spelling.

- **Long-running calculations time out the conversation.** For
  jobs that take minutes-to-hours, run them in the background
  (`nohup ... &`) and paste the `.out` back when ready. Tell
  Claude the wall-clock budget upfront so it can suggest
  cheaper screening calculations first.

- **File-name clashes.** When iterating, Claude tends to reuse
  output names. Tell it explicitly: *"name this run
  `attempt-3-hybrid`"*, the `output=` argument controls every
  side effect.

- **Hallucinated APIs** for new functionality. The
  v0.7 linear-dependence APIs are a recent addition, if Claude is
  citing something that doesn't exist, it's pulling from training
  data older than v0.7. Re-paste the URL or point at
  [user guide § linear dependence](user_guide/linear_dependence.md)
  specifically.

## End-to-end worked example: SMILES to IR spectrum

A more involved session, the chemist starts with a SMILES string
and ends with an IR spectrum.

> *Take this SMILES, `CC(=O)O`, get me the optimised geometry
> at PBE/6-31G\* + D3BJ, then the IR spectrum, and compare the C=O
> stretch frequency to NIST.*

Claude:

1. Converts the SMILES to a 3D structure (typically via RDKit
   `MolFromSmiles` + `EmbedMolecule`).
2. Writes the optimisation script using
   `run_job(..., optimize=True)`.
3. Runs it; reads `.out` for the converged geometry.
4. Wires `compute_hessian_rhf_analytic` (or the RKS analog) on the
   final geometry to get the Hessian → frequencies + IR
   intensities.
5. Parses the highest-intensity peak in the 1700-1800 cm⁻¹ range
   (acetic acid C=O stretch).
6. Quotes a NIST reference. **Verify the reference exists before
   citing.**

The full session is ~10 turns; the input scripts that come out are
~30 lines each. The chemist never wrote a `for` loop.

## Best practices

- **Directory hygiene.** Run each session in its own directory
  (`mkdir ~/runs/2026-05-corundum-pbe`); the `output=` stems make
  every file traceable to the input that produced it.
- **Cite Claude where appropriate.** If Claude wrote the input
  scripts that produced your published numbers, acknowledge it the
  same way you'd acknowledge a graduate student or a code library.
- **Cross-validate against PySCF / ORCA** for anything important.
  The [comparison framework](running.md) makes this a one-line
  flag in any input script.
- **Keep the docs URL in the conversation.** When the conversation
  drifts (especially on long sessions), re-paste the URL or a
  specific page so Claude doesn't lean on stale training data.

## Where to next

- **The [tutorial set](tutorial/index.md)**, every tutorial is a
  worked example Claude can ingest; useful as a calibration
  exercise (run a tutorial via Claude, see if you get the same
  numbers).
- **The [examples/ directory](https://github.com/vibe-qc/vibe-qc/tree/main/examples)**
  - every input is a runnable Python script with the same
  conventions Claude uses.
- **The [comparison framework](running.md)** for cross-code
  validation of anything Claude generates.
- **The [support page](support.md)**, vibe-qc is a one-person
  open-source project. Two recurring costs pay for the
  *development engine*: the Claude Max subscription that this
  whole guide is built around, and the self-hosted server behind
  `vibe-qc.com`. Both are funded directly by sponsors.

  ```{admonition} 🖥️ We need bigger hardware!
  :class: tip

  If you've worked through the corundum example above and noticed
  that the reference numbers stop at small cells / minimal bases —
  *that's the M2 laptop the project is developed on*. Bigger
  hardware would expand what's tested commit-by-commit (real-world
  TM oxides, slabs, supercells; the v2.x Cyclic Cluster Model
  track at correlated levels). See the
  [support page](support.md#what-your-sponsorship-funds) for the
  near-term and long-term hardware goals — every contribution
  pushes toward bigger reference systems on every run.
  ```

  If this workflow is useful to you, consider [sponsoring on
  GitHub](https://github.com/sponsors/mpeintinger) (recurring
  monthly) or via [Ko-fi](https://ko-fi.com/mpeintinger)
  (one-time, no GitHub account required).

```{admonition} Feedback
:class: note

This page is new (v0.7.x). If a workflow you tried with Claude
worked badly, or really well, please [open an
issue](https://github.com/vibe-qc/vibe-qc/issues) or email
**`mpei@vibe-qc.com`** with the transcript. Real workflow data is
how this guide gets sharper.
```
