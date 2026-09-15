# DFT functional comparison

Every density-functional approximation is a model for the
exchange-correlation energy $E_{\text{xc}}[\rho]$. They climb a
well-known hierarchy, "Jacob's ladder", trading cost for physics:

| Rung | Input | Cost | Examples |
|---|---|---|---|
| 1. LDA | density $\rho$ only | cheapest | Slater, SVWN, PW92 |
| 2. GGA | $\rho$ + $\nabla\rho$ | +~10% | PBE, BLYP, PW91 |
| 3. meta-GGA | + $\tau = \sum_i \lvert\nabla\phi_i\rvert^2 / 2$ | +~20% | TPSS, SCAN, M06-L |
| 4. hybrid | + fraction of HF exchange | +~2× | B3LYP, PBE0, M06-2X |
| 5. range-separated hybrid | + long-range HF exchange | +~3× | ωB97X, CAM-B3LYP |

Going up the ladder usually improves accuracy at the cost of more
compute per SCF step. vibe-qc currently supports rungs 1, 2, and 4
(LDA, GGA, hybrid) via named functionals; anything libxc catalogs
is also reachable through its numeric ID. This tutorial sweeps a set
and compares them on water.

## A single-line functional swap

The `method` and `functional` arguments are all that change between a
Hartree-Fock run and any DFT rung. The loop below runs the same
experimental-geometry water molecule at HF and five functionals,
selecting each either by name or by libxc numeric ID:

```python
from vibeqc import Atom, Molecule, run_job

mol = Molecule([
    Atom(8, [0.0,  0.00,  0.00]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])

for label, functional in (
    ("HF",           None),
    ("LDA (SVWN)",   "SVWN"),
    ("GGA (PBE)",    "PBE"),
    ("GGA (BLYP)",   "BLYP"),
    ("hybrid B3LYP", "B3LYP"),
    ("hybrid PBE0",  "406"),            # via libxc ID
):
    run_job(
        mol,
        basis="6-31g*",
        method="rhf" if functional is None else "rks",
        functional=functional,
        output=f"water_{label.replace(' ', '_').lower()}",
        write_molden_file=False,
    )
```

Each call writes an `.out` file you can eyeball; below we'll pull
the same numbers more compactly.

## What the numbers say

Running the sweep above and extracting total energy, dipole, and
frontier-orbital levels (HF/6-31G\* experimental-geometry water):

| Method | $E$ (Ha) | $\lvert\mu\rvert$ (D) | $\varepsilon_{HOMO}$ (eV) | $\varepsilon_{LUMO}$ (eV) | gap (eV) |
|---|---:|---:|---:|---:|---:|
| HF             | −76.00668 | 2.061 | −13.56 | +6.11 | 19.67 |
| LDA (SVWN)     | −75.83424 | 1.988 |  −6.32 | +1.61 |  7.93 |
| GGA (PBE)      | −76.31284 | 1.914 |  −6.23 | +1.65 |  7.88 |
| GGA (BLYP)     | −76.37826 | 1.892 |  −6.11 | +1.50 |  7.60 |
| hybrid (B3LYP) | −76.36387 | 1.945 |  −7.85 | +2.28 | 10.13 |
| hybrid (PBE0)  | −76.31857 | 1.971 |  −8.32 | +2.70 | 11.02 |

Experimental reference: dipole 1.855 D (gas phase); vertical ionisation
potential 12.6 eV ≈ −$\varepsilon_{HOMO}$ by Koopmans' theorem.

Reading across:

- **Total energies drift**, LDA is too high (electronic correlation
  is underestimated); GGAs drop ~0.3-0.5 Ha; BLYP sits lowest here,
  with B3LYP just above it (its 20 % exact exchange pulls back part
  of the GGA over-binding). Absolute totals aren't meaningful to
  compare across functionals, they live on different reference
  surfaces. (B3LYP row = vibe-qc's default VWN5 flavor; the
  Gaussian-flavor `b3lyp/g` lands ~37 mHa lower on this system, 
  see [functionals](../user_guide/functionals.md) § B3LYP.)
- **Dipole moments are well-behaved.** HF overpolarises; GGAs pull
  back and undershoot slightly; hybrids land in between. All are
  within 0.2 D of the experimental value.
- **HF HOMO is unphysically deep**: −13.56 eV vs. experimental IP
  of 12.6 eV. HF's lack of correlation is compensated by Koopmans'
  theorem only partially.
- **Pure DFT HOMOs are too shallow** (−6 to −6.3 eV), the classic
  DFT gap-underestimation problem driven by self-interaction error.
- **Hybrids sit in the middle** because they mix in ~20-25% exact HF
  exchange, which partially corrects both the self-interaction and
  the overbound electrons. B3LYP (−7.95 eV) and PBE0 (−8.32 eV) are
  much closer to experiment.
- **B3LYP beats PBE on total energy** because the LYP correlation
  functional is particularly well-parametrised for first-row
  chemistry. This is an empirical match, not a systematic
  improvement, don't extrapolate to transition metals.

```{figure} ../_static/plots/water-functional-comparison.png
:alt: Three-panel bar chart comparing dipole moment, HOMO level, and HOMO-LUMO gap of water across HF, SVWN, PBE, BLYP, B3LYP, and PBE0 with experimental references.
:width: 80%
:align: center

H₂O / 6-31G\* across Jacob's ladder, with experimental references
(red dashed). **Left:** dipole moment — every method is within 0.21 D
of the gas-phase value, with hybrids landing between HF
(over-polarised) and pure GGAs (slightly under-polarised). **Centre:**
HOMO level interpreted via Koopmans' theorem against the
experimental vertical IP $-12.6$ eV. HF is too deep by ~1 eV; pure
DFT is far too shallow (~6 eV); B3LYP and PBE0 — by mixing in
20–25% exact exchange — sit much closer to experiment. **Right:**
HOMO–LUMO gap. The well-known DFT gap-underestimation problem is
visible: pure GGAs give 7–8 eV, hybrids 10–11 eV, HF 19–20 eV;
none of these match the optical gap, which is what TDDFT exists to
compute. Regenerated by
`[`examples/plots/water-functional-comparison.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/plots/water-functional-comparison.py)`.
```

## Functional names vibe-qc knows

Via name (fast path):

| Name | Family | HF exchange | libxc components |
|---|---|---:|---|
| `LDA` / `SVWN`  | LDA         | 0 | Slater + VWN5 |
| `SLATER`        | LDA         | 0 | Slater exchange only |
| `PBE`           | GGA         | 0 | X_PBE + C_PBE |
| `BLYP`          | GGA         | 0 | X_B88 + C_LYP |
| `B3LYP`         | hybrid GGA  | 0.20 | XC_HYB_GGA_XC_B3LYP5 (VWN5; ORCA convention) |
| `B3LYP/G`       | hybrid GGA  | 0.20 | XC_HYB_GGA_XC_B3LYP (VWN3; Gaussian convention) |

Via libxc ID (escape hatch for anything else libxc ships, 500+
functionals):

```python
opts.functional = "406"          # PBE0 (XC_HYB_GGA_XC_PBEH)
opts.functional = "402,101"      # separate X and C IDs, comma-separated
```

Look up IDs at
[tddft.org/programs/libxc/functionals](https://tddft.org/programs/libxc/functionals/).

**Now in reach** in vibe-qc (molecular RKS + UKS): the τ-dependent
meta-GGA family, `tpss`, `tpssh`, `m06-l`, `m06-2x`. The kinetic-
energy density τ is built from the AO gradients on the DFT grid;
parity to PySCF.dft is at grid-quadrature precision. See
[`functionals.md`](../user_guide/functionals.md) § Meta-GGA
functionals for the coverage matrix (periodic + analytic gradients
still pending). Currently **still out of reach**: range-separated
hybrids (ωB97X, ωB97X-D, ωB97X-V, CAM-B3LYP, HSE06), they need
the range-separation machinery. Meta-GGAs that additionally depend
on the density laplacian ∇²ρ (TB09, etc.) are also out of reach
until the grid extends to second derivatives. Track status on
the [roadmap](../roadmap.md).

## Cost across the ladder

On water / 6-31G\* (11 basis functions; same machine, 12 OpenMP
threads), SCF wall-clock time was roughly:

| Method         | time (s) | iters |
|---|---:|---:|
| HF             | 0.01 |  10 |
| LDA (SVWN)     | 0.23 |   9 |
| GGA (PBE)      | 0.33 |   9 |
| GGA (BLYP)     | 0.30 |   9 |
| hybrid (B3LYP) | 0.34 |   9 |
| hybrid (PBE0)  | 0.32 |   9 |

Two observations:

1. **HF is cheaper than DFT** here because the DFT grid evaluation
   dominates at small N, where the four-index ERI sum is tiny.
   This flips at around 40-60 basis functions, pure DFT becomes
   cheaper than HF as the grid cost amortises.
2. **Hybrids aren't meaningfully more expensive at this size.** The
   HF-exchange sum is a small fraction of the SCF cost when the
   system is this small. For bigger systems the hybrid overhead
   becomes real (~2× vs. pure DFT).

## Picking a functional

A starter heuristic for general organic chemistry:

- **B3LYP**, by far the most-published hybrid; "safe default" for
  geometries and binding energies on organic molecules. Known to
  underbind dispersion-dominated systems, always pair with
  [DFT-D3](dispersion.md).
- **PBE**, the go-to for periodic calculations; faster than hybrids,
  reasonable for geometries. Use with D3(BJ) for molecular crystals.
- **PBE0**, hybrid cousin of PBE with 25% HF exchange. Better HOMO
  levels and reaction barriers than PBE; commonly used for TMs.
- **LDA**, now mostly a teaching functional. Over-binds bonds,
  over-polarises, HOMO is too shallow. Still useful for quick
  screens on metallic systems where its simplicity is a virtue.

Rules of thumb that survive basis / method changes:

1. **No functional is universally best.** Benchmarks are the only
   defensible way to pick for a specific system class.
2. **Most DFT errors cancel in relative energies.** Reaction
   thermochemistry, conformational energies, and barriers are often
   within 1-2 kcal/mol across good GGA/hybrid choices. Absolute
   energies are much less stable.
3. **Polar / ionic / H-bonded systems need dispersion.** Add
   D3(BJ), it's a free lunch that always lowers the intermolecular
   error at negligible cost.
4. **Transition metals are harder.** B3LYP can be wrong by 10+
   kcal/mol on spin states; `pbe0`, `m06-2x`, and `wb97x`
   typically do better, all three ship in vibe-qc (see
   [`functionals.md`](../user_guide/functionals.md) for the
   meta-GGA family and [Range-separated hybrids: ωB97X and the HOMO ≈ −IP property](range_separated_wb97x.md)
   for ωB97X).

## DFT-D3 in the loop

Adding dispersion for each functional (using
[`dispersion="d3bj"`](dispersion.md)):

```python
for functional in ("PBE", "BLYP", "B3LYP"):
    r = run_job(
        mol, basis="6-31g*",
        method="rks", functional=functional,
        dispersion="d3bj",
        output=f"water_{functional}_d3",
        write_molden_file=False,
    )
    print(f"{functional:8s}  E_SCF = {r.energy:.6f}   "
          f"E_disp = {r.e_dispersion:+.6e}   "
          f"E_total = {r.energy_total:.6f}")
```

Dispersion is a small correction for isolated water (~0.7 mHa ≈
0.4 kcal/mol of intra-molecular contribution) but dominates
intermolecular binding in dimers, layered materials, and host-guest
systems.

## Sanity checks before you trust the number

- **Basis saturation.** See
  [basis-set convergence](basis_convergence.md). A good functional
  on an inadequate basis is worse than a bad functional on a good
  one.
- **Grid quality.** All numbers above use the default DFT grid
  (Treutler-Ahlrichs, ≈ 46k points/atom). For meta-GGAs and hybrids
  pushed to ~8 significant figures, bump `RKSOptions.grid.n_radial`
  to 99 or 120.
- **SCF tolerance.** Defaults are tight (`conv_tol_energy = 1e-8 Ha`).
  For convergence-sensitive properties (especially low-frequency
  vibrations) tighten `conv_tol_grad` too.

## Theory, the functionals in detail

The `molecular_dft` tutorial introduced the Kohn-Sham framework and Jacob's ladder in
the abstract. Here we look at the specific functionals used above,
each of which represents a different philosophy for approximating
$E_\text{xc}[\rho]$.

### LDA, the local-density approximation

LDA is the "at every point, pretend you're at the matching point in
a uniform electron gas" ansatz:

$$
E_\text{xc}^\text{LDA}[\rho] = \int \rho(\mathbf{r}) \, \varepsilon_\text{xc}^\text{HEG}\!\bigl(\rho(\mathbf{r})\bigr) \, \mathrm{d}^3 r
$$

where $\varepsilon_\text{xc}^\text{HEG}$ is the per-electron XC energy of
the homogeneous electron gas, known analytically for exchange
(Slater / Dirac formula) and numerically for correlation (Ceperley-
Alder quantum Monte Carlo, 1980; VWN parametrisation, 1980).
vibe-qc's ``SVWN`` is Slater exchange + VWN5 correlation, the
canonical LDA. LDA over-binds bonds by ~0.2 Å of length and ~10
kcal/mol of dissociation energy on typical organics; now used mostly
for teaching, quick screens, and metallic solids.

### GGA, adding the gradient

Generalised-gradient approximations (GGAs) lift the "uniform"
assumption by adding $|\nabla\rho|$ as a second argument:

$$
E_\text{xc}^\text{GGA}[\rho] = \int \rho \, \varepsilon_\text{xc}^\text{GGA}\bigl(\rho, |\nabla\rho|\bigr) \, \mathrm{d}^3 r.
$$

**PBE** (Perdew-Burke-Ernzerhof 1996) is the non-empirical GGA, 
every parameter fixed by physical constraints (UEG limit,
Lieb-Oxford bound, correct slowly-varying expansion). Workhorse for
periodic calculations. **BLYP** (Becke 1988 exchange + Lee-Yang-Parr
1988 correlation) is an empirical GGA with different strengths, 
better for organic thermochemistry at the cost of transferability.

### Hybrid functionals and self-interaction

Pure DFT has **self-interaction error**: the Coulomb term $J[\rho]$
includes an electron's interaction with itself, and the local
exchange potential only partially cancels it. HF exchange cancels
self-interaction exactly (the diagonal $K_{ii} = J_{ii}$ term).
Hybrid functionals mix in a fixed fraction $\alpha$ of HF exchange:

$$
E_\text{xc}^\text{hybrid} = \alpha \, E_\text{x}^\text{HF} + (1 - \alpha) \, E_\text{x}^\text{GGA} + E_\text{c}^\text{GGA}.
$$

**B3LYP** (Becke 1993, reparametrised Stephens et al. 1994) uses
$\alpha = 0.20$ with three empirical coefficients fit to a
thermochemistry benchmark; the "3" refers to three parameters:

$$
E_\text{xc}^\text{B3LYP} = 0.20 \, E_\text{x}^\text{HF} + 0.80 \, E_\text{x}^\text{S} + 0.72 \, \Delta E_\text{x}^\text{B88} + 0.81 \, E_\text{c}^\text{LYP} + 0.19 \, E_\text{c}^\text{VWN}.
$$

**PBE0** (Adamo-Barone 1999) is parameter-free, PBE's GGA with a
quarter of HF exchange mixed in on rigorous adiabatic-connection
grounds:

$$
E_\text{xc}^\text{PBE0} = 0.25 \, E_\text{x}^\text{HF} + 0.75 \, E_\text{x}^\text{PBE} + E_\text{c}^\text{PBE}.
$$

The tradeoff: hybrids are ~2× more expensive than pure DFT per SCF
step (the HF-exchange sum scales $\mathcal{O}(N^4)$), but self-
interaction is reduced and band gaps / barriers / HOMO levels are
better. For organics where reaction energies matter, the cost is
nearly always worth paying.

### Why this is empirical, not systematic

Unlike coupled cluster (where CCSDTQ → CCSDT → CCSD(T) → CCSD → MP2
is a systematic hierarchy converging to exactness), there is no
knob you can turn to make a DFT functional systematically more
accurate. Different functionals are **different physical models for
the same unknown object**. Benchmarking against CCSD(T) or
experiment on a target system class is the only defensible way to
choose. The [GMTKN55 benchmark](https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn)
(Grimme's group, 2017) is the canonical test set.

## Resources

~250 MB peak RAM, ~5 s on one core (Apple M2 baseline) for the
four-functional H₂O sweep. Hybrid functionals (B3LYP, PBE0, M06)
add ~30% over a pure GGA at this size due to the HF exchange
contribution to the Fock build; range-separated hybrids
($\omega$B97X-D) add another factor of ~2 because the long-range
exchange goes through a separately-screened lattice sum.

## References

- **LDA correlation parametrisation.** S. H. Vosko, L. Wilk, and
  M. Nusair, "Accurate spin-dependent electron liquid correlation
  energies for local spin density calculations: a critical
  analysis," *Can. J. Phys.* **58**, 1200 (1980).
- **Uniform-gas correlation reference (QMC).** D. M. Ceperley and
  B. J. Alder, "Ground state of the electron gas by a stochastic
  method," *Phys. Rev. Lett.* **45**, 566 (1980).
- **BLYP exchange + correlation.** A. D. Becke, "Density-functional
  exchange-energy approximation with correct asymptotic behavior,"
  *Phys. Rev. A* **38**, 3098 (1988); C. Lee, W. Yang, and R. G.
  Parr, "Development of the Colle-Salvetti correlation-energy
  formula into a functional of the electron density," *Phys. Rev.
  B* **37**, 785 (1988).
- **PBE.** J. P. Perdew, K. Burke, and M. Ernzerhof, "Generalized
  gradient approximation made simple," *Phys. Rev. Lett.* **77**,
  3865 (1996); erratum ibid. **78**, 1396 (1997).
- **Hybrid DFT, original.** A. D. Becke, "A new mixing of Hartree-
  Fock and local density-functional theories," *J. Chem. Phys.*
  **98**, 1372 (1993); the three-parameter form (same author):
  "Density-functional thermochemistry. III. The role of exact
  exchange," *J. Chem. Phys.* **98**, 5648 (1993).
- **B3LYP as used in practice.** P. J. Stephens, F. J. Devlin, C. F.
  Chabalowski, and M. J. Frisch, "Ab initio calculation of
  vibrational absorption and circular dichroism spectra using
  density functional force fields," *J. Phys. Chem.* **98**, 11623
  (1994). This is the paper that defines "B3LYP" as every code now
  implements it (Becke's 1993 half-and-half mixed VWN for
  correlation; Stephens et al. swap in LYP).
- **PBE0.** C. Adamo and V. Barone, "Toward reliable density
  functional methods without adjustable parameters: the PBE0
  model," *J. Chem. Phys.* **110**, 6158 (1999).
- **GMTKN55 benchmark set.** L. Goerigk, A. Hansen, C. Bauer,
  S. Ehrlich, A. Najibi, and S. Grimme, "A look at the density
  functional theory zoo with the advanced GMTKN55 database for
  general main group thermochemistry, kinetics, and noncovalent
  interactions," *Phys. Chem. Chem. Phys.* **19**, 32184 (2017).

## Next

- [Vibrational frequencies](vibrational_frequencies.md), how the
  functional affects IR spectra.
- [Dispersion corrections](dispersion.md), the quick way to fix
  non-covalent interactions across any functional.
