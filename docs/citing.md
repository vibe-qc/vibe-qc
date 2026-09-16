---
myst:
  html_meta:
    "description": "How to cite vibe-qc and its underlying libraries in published work. Software citation, basis-set citations, integral-library citation."
    "og:title": "How to cite vibe-qc"
    "og:description": "Citation guidance for vibe-qc, including the pob-TZVP basis set paper, libint, libxc, and the cyclic-cluster-model thesis work."
    "og:type": "website"
    "og:url": "https://vibe-qc.com/docs/citing.html"
---

# How to cite vibe-qc

If you use vibe-qc in published work, please cite it. The repository
ships a [`CITATION.cff`](https://github.com/vibe-qc/vibe-qc/blob/main/CITATION.cff)
file at the root that GitLab and citation managers (Zotero, Mendeley,
the Sphinx-rendered "Cite this software" button on the repository page)
parse automatically. This page expands what to cite *in addition to*
the vibe-qc software itself, depending on which features you used.

```{admonition} Every run emits its own bibliography
:class: tip

Every `run_job` call writes a complete
[`{stem}.bibtex`](user_guide/citations.md) and `{stem}.references`
sibling file, plus a `## References` block at the bottom of
`{stem}.out`. The runtime walks the same set of papers listed here and
assembles them in the right order for the specific functional / basis /
dispersion / library combination your job exercised. The bundled
[citation database](user_guide/citations.md) is the single source of
truth.

In other words: you should not normally need to hand-cite from this
page. Open the `.bibtex` next to your `.out`, drop it into your
manuscript's bibliography, and `\cite{}` the entries by their
`bibtex_key`. The tables below remain as a backup for users on older
vibe-qc versions and for ad-hoc citations outside `run_job`.
```

## The software itself

Always cite this. APA-style:

> Peintinger, M. F. (2026). *vibe-qc: a quantum-chemistry code for
> molecules and solids* (version 0.17.4) [Software]. Mozilla Public
> License 2.0. <https://vibe-qc.com/>

BibTeX:

```bibtex
@software{peintinger_vibeqc_2026,
  author  = {Peintinger, Michael F.},
  title   = {{vibe-qc}: a quantum-chemistry code for molecules and solids},
  year    = {2026},
  version = {0.17.4},
  license = {MPL-2.0},
  url     = {https://vibe-qc.com/},
}
```

A vibe-qc-specific peer-reviewed publication is planned once a tagged
release covers a publishable scope. Until then, the software citation
above is the canonical reference.

## When you use the pob-TZVP basis sets

The pob-TZVP, pob-DZVP-rev2, and pob-TZVP-rev2 basis sets bundled with
vibe-qc were developed by the project author and collaborators. The
references below are rendered from the
[citation database](user_guide/citations.md), the same source
`run_job` walks to assemble each job's `.bibtex` sibling. Edit
`python/vibeqc/output/citations/database.toml` to update them; the
docs follow.

The originating paper (covers pob-TZVP):

```{vibeqc-cite-entry} peintinger_pob_tzvp_2013
```

The revised (`-rev2`) variants extend coverage to heavier elements:

```{vibeqc-cite-entry} vilela_oliveira_pob_rev2_2019
```

Or, equivalently, render whichever set of papers the runtime would
cite for a specific basis-set name:

```{vibeqc-cite-route} basis_sets pob-tzvp-rev2
```

## When you use DFT (any functional)

vibe-qc evaluates exchange-correlation functionals via
[libxc](https://libxc.gitlab.io/). Please cite the libxc paper:

> Lehtola, S., Steigemann, C., Oliveira, M. J. T., & Marques, M. A. L.
> (2018). Recent developments in libxc, A comprehensive library of
> functionals for density functional theory. *SoftwareX*, **7**,
> 1-5. [doi:10.1016/j.softx.2017.11.002](https://doi.org/10.1016/j.softx.2017.11.002)

You should also cite the specific functional you used (e.g. PBE,
B3LYP, ωB97X-D), see the libxc documentation for per-functional
references.

## When you use Gaussian integrals (any HF / DFT / MP2 calculation)

All Gaussian integrals are evaluated via
[libint](https://github.com/evaleev/libint). The standard libint
citation is:

> Valeev, E. F. *Libint: A library for the evaluation of molecular
> integrals of many-body operators over Gaussian functions.*
> Available at <https://github.com/evaleev/libint>.

## When you use D3(BJ) dispersion

D3(BJ) corrections via the
[`dftd3`](https://github.com/dftd3/simple-dftd3) interface require:

> Grimme, S., Antony, J., Ehrlich, S., & Krieg, H. (2010). A
> consistent and accurate ab initio parametrization of density
> functional dispersion correction (DFT-D) for the 94 elements
> H-Pu. *J. Chem. Phys.*, **132**, 154104.
> [doi:10.1063/1.3382344](https://doi.org/10.1063/1.3382344)
>
> Grimme, S., Ehrlich, S., & Goerigk, L. (2011). Effect of the
> damping function in dispersion corrected density functional theory.
> *J. Comput. Chem.*, **32**(7), 1456-1465.
> [doi:10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)

## When you use crystal-symmetry detection

vibe-qc's spacegroup detection wraps
[spglib](https://github.com/spglib/spglib):

> Togo, A., & Tanaka, I. (2018). *Spglib: a software library for
> crystal symmetry search.* arXiv:1808.01590.
> <https://arxiv.org/abs/1808.01590>

## Method, convergence & analysis references (emitted automatically)

vibe-qc routes a defining-paper citation into your `.references` /
`.bibtex` whenever a job actually exercises one of the features below.
You don't need to look these up, keep what lands in the output. The
table gives the headline reference; the emitted entry carries the full
bibliographic record.

| When your job used … | Headline reference(s) |
|---|---|
| RI-J Coulomb fitting (`density_fit=True`) | Whitten 1973; Dunlap 1979; Eichkorn 1995, 1997 |
| RIJCOSX chain-of-spheres exchange (`cosx=True`) | Neese 2009 |
| Analytic forces (geometry optimisation, `optimize=True`) | Pulay 1969 + Hellmann-Feynman |
| Harmonic frequencies / IR (`hessian=True`) | Wilson-Decius-Cross 1955 |
| Population analysis, Mulliken/Löwdin charges, Mayer bond orders, Hirshfeld charges (best-effort) | Mulliken 1955; Löwdin 1950; Mayer 1983; Hirshfeld 1977 |
| Second-order SCF convergence (SOSCF / TRAH thresholds) | Bacskay 1981 / Neese 2000; Helmich-Paris 2021 |
| Linear-response TDDFT (`run_tddft_*`) | Runge-Gross 1984 + Casida 1995; Hirata-Head-Gordon 1999 (TDA) |
| Explicit SAD / SAP / extended-Hückel SCF guess | van Lenthe 2006; Lehtola 2019; Hoffmann 1963 |

The population-analysis references fire for an ordinary molecular
`run_job`, it writes the `.population.*` siblings (Mulliken + Löwdin
charges, Mayer bond orders) by default, so those three references are
part of the standard reference set unless you pass
`write_population_file=False`. Hirshfeld 1977 additionally joins the
set when the best-effort Hirshfeld charges actually compute and reach
the `.out` "Atomic charges" table, that column needs a Becke-Lebedev
grid plus a SAD promolecule build, either of which can fail
independently, so (unlike the always-on Mulliken/Löwdin/Mayer triad)
it is cited only when it is computed, never on a grid/promolecule
build failure.

## Conceptual / methodological references

The Cyclic Cluster Model that the v2.x roadmap builds on is rooted in
the project author's PhD work:

> Peintinger, M. F., & Bredow, T. (2014). The cyclic cluster model
> at Hartree-Fock level. *Journal of Computational Chemistry*,
> **35**(11), 839-846.
> [doi:10.1002/jcc.23550](https://doi.org/10.1002/jcc.23550)
> *(Free Access, cover article for the issue.)*

This paper does not need to be cited for routine vibe-qc use today
(the CCM features are pending v2.x), but it's the right reference if
you want to cite the methodological lineage.

## Quick chooser

| What you used | Cite |
|---|---|
| Any vibe-qc calculation | The software itself (`CITATION.cff`) |
| pob-TZVP / pob-DZVP-rev2 / pob-TZVP-rev2 | + Peintinger 2013 (and 2019 for `-rev2`) |
| DFT (any functional) | + libxc + functional reference |
| HF / DFT / MP2 (i.e. integrals) | + libint |
| D3(BJ) dispersion | + Grimme 2010 + 2011 |
| Periodic system | + spglib (symmetry) |
| Density fitting (RI-J / RIJCOSX) | + Eichkorn 1995/1997 (+ Neese 2009 for COSX) |
| Geometry optimisation / frequencies | + Pulay 1969 (forces) / Wilson-Decius-Cross 1955 (Hessian) |
| Population analysis (charges + bond orders) | + Mulliken 1955 / Löwdin 1950 / Mayer 1983 |
| SOSCF / TRAH convergence | + Bacskay 1981 / Helmich-Paris 2021 |
| TDDFT excited states | + Runge-Gross 1984 + Casida 1995 |
| Future v2.x CCM work | + Peintinger 2014 |

For DFT runs in particular, the table above only lists the families
the bundled database covers. If you used a
functional with no route (e.g. a libxc id passed directly), the
runtime emits a `# no citation route for functional 'foo'` warning
into the `.references` file, that is the cue to add an entry to
[`database.toml`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/output/citations/database.toml)
in your PR or to look up the matching paper by hand.
