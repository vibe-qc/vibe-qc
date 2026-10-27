# Range-separated hybrids: ωB97X and the HOMO ≈ −IP property

Conventional hybrid functionals, B3LYP, PBE0, mix a *fixed*
fraction of exact (Hartree-Fock) exchange at every inter-electron
distance: 20 % for B3LYP, 25 % for PBE0. That single number is a
compromise. A **range-separated** (also "long-range-corrected" or
CAM) hybrid makes the exact-exchange fraction *depend on the
inter-electron distance*, typically a small fraction at short
range, ramping to a large fraction (often 100 %) at long range.

This tutorial introduces vibe-qc's first range-separated hybrid,
**ωB97X** (Chai & Head-Gordon 2008), with a worked example that
needs only ground-state SCF: the **HOMO eigenvalue as an estimate
of the ionization potential**. Range-separated functionals are
dramatically better at this than conventional hybrids, and it's
the cleanest single-number demonstration of why the long-range
exchange matters.

```{admonition} Status: preview
:class: note

Range-separated hybrids are production for molecular energies, but their
analytic nuclear gradients are incomplete: the RKS/UKS gradient kernel
omits the long-range exact-exchange term of the functional (and, for the
-V functionals, the VV10 nonlocal term), so it is not the derivative of the
reported energy (GitLab #571). `compute_gradient_rks` and
`compute_gradient_uks` raise `NotImplementedError` naming the missing
term unless you pass `allow_incomplete=True`, and a geometry
optimisation through `run_job(optimize=True)` falls back to finite
differences of the energy, which is correct but several times slower. The
scope is molecular RKS/UKS with direct SCF. See
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

## Theory

**Why the HOMO should equal minus the ionisation potential.** In exact
Kohn-Sham DFT the eigenvalue of the highest occupied orbital equals minus the
first ionisation potential, $\varepsilon_\text{HOMO} = -\text{IP}$
(the Perdew-Parr-Levy-Balduz result, in the references; Janak's theorem
in the exact-functional limit). Approximate
functionals violate this badly. The spurious self-interaction error of
(semi-)local exchange makes the exchange-correlation potential decay
exponentially instead of as $-1/r$ at large $r$; the HOMO sits in that
wrong asymptotic potential and comes out far too shallow. For water in
def2-TZVP the run below gives B3LYP $-8.59$ eV and PBE0 $-8.97$ eV against
an experimental vertical IP of 12.62 eV.

**The range-separated exchange.** A range-separated (long-range-corrected)
hybrid splits the Coulomb operator with the error function
({vibeqc-cite}`vydrov_scuseria_lcwpbe_2006` is the long-range-corrected
construction with a semilocal short-range part),

$$
\frac{1}{r_{12}}
 = \underbrace{\frac{1 - [\alpha + \beta\,\mathrm{erf}(\omega r_{12})]}{r_{12}}}_{\text{semi-local DFT exchange}}
 + \underbrace{\frac{\alpha + \beta\,\mathrm{erf}(\omega r_{12})}{r_{12}}}_{\text{exact HF exchange}} ,
$$

so the exact-exchange fraction is $\alpha$ at $r_{12}\to 0$ and $\alpha+\beta$
at $r_{12}\to\infty$. In Fock-matrix terms the exchange becomes
$\mathbf K \to \alpha\,\mathbf K + \beta\,\mathbf K_\text{erf}(\omega)$, where
$\mathbf K_\text{erf}$ is built from integrals of $\mathrm{erf}(\omega r_{12})/r_{12}$
(vibe-qc's direct-SCF kernel builds it alongside the ordinary $\mathbf K$).
With $\alpha + \beta = 1$ the functional has 100 percent exact exchange at
long range, which restores the $-1/r$ asymptotic potential, removes the bulk
of the self-interaction error, and brings $\varepsilon_\text{HOMO}$ much
closer to $-\text{IP}$. ωB97X ({vibeqc-cite}`chai_headgordon_wb97x_2008`) uses
$\alpha = 0.157706$, $\beta = 0.842294$ (so $\alpha+\beta = 1$) and
$\omega = 0.3\ \text{bohr}^{-1}$, with a short-range exchange and
correlation form fitted to thermochemistry, kinetics and noncovalent data;
ωB97X-D ({vibeqc-cite}`chai_headgordon_wb97xd_2008`) re-fits it with an
empirical dispersion term and a different $\omega = 0.2$, $\alpha=0.222$.

**Tuning and its limits.** The ionisation-potential condition can be used
in reverse to choose $\omega$: the optimally tuned range-separated functional
of a given molecule has the $\omega$ for which $-\varepsilon_\text{HOMO}(N)$
equals the total-energy difference $E(N-1) - E(N)$ (Stein and Kronik and
coworkers). ωB97X does not do that; its $\omega$ is a fixed compromise fitted
once for all molecules, so its HOMO is a good but not exact estimate (11.60
eV for water, 1.0 eV short of experiment, against an error of about 4 eV for
the global hybrids). The price is the erf-attenuated exchange: it has no
three-centre density-fitting path here (so it needs direct SCF), it roughly
doubles the exchange cost of the Fock build, it makes open-shell SCF
stiffer, and its analytic nuclear gradient is incomplete (the status note
above). Where it helps: orbital energies, ionisation potentials and electron
affinities from Kohn-Sham eigenvalues, charge-transfer and Rydberg
character, and long-range interactions. Where it does not: routine
closed-shell total energies and geometries, for which a global hybrid with
dispersion is as accurate and cheaper.

## Running ωB97X

ωB97X is a libxc functional like any other, just name it:

```python
import vibeqc as vq

mol = vq.Molecule([                      # coordinates in bohr
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "def2-tzvp")

opts = vq.RKSOptions()
opts.functional = "wb97x"
result = vq.run_rks(mol, basis, opts)

print(f"E(ωB97X/def2-TZVP) = {result.energy:.6f} Ha")
# E(ωB97X/def2-TZVP) = -76.435213 Ha
```

vibe-qc reads the $(\omega, \alpha, \beta)$ triple from libxc at
construction and the SCF driver builds a **second, erf-attenuated
exchange matrix** $K_{\text{erf}}(\omega)$ alongside the ordinary
$K$. You can inspect the parameters:

```python
fn = vq.Functional("wb97x")
print(fn.is_range_separated)   # True
print(fn.rsh_omega)            # 0.3
print(fn.cam_alpha)            # 0.157706  (short-range HF fraction)
print(fn.cam_beta)             # 0.842294  (the long-range ramp)
```

```{important}
**ωB97X runs through direct SCF only.** The erf-attenuated $K$ is
built by the direct-SCF Fock kernel; there is no erf-attenuated
3-centre integral path yet, so `density_fit=True` with a
range-separated functional is rejected with a clear error. The
molecular drivers select the direct builder automatically for RSH
functionals; just leave `density_fit` at its default.
```

## The worked example: HOMO ≈ −IP across three functionals

Compute water at a fixed geometry with three functionals and
read the HOMO eigenvalue off each:

```python
# input-wb97x-homo.py
import vibeqc as vq

mol = vq.Molecule([                      # coordinates in bohr
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "def2-tzvp")

for func in ("b3lyp", "pbe0", "wb97x"):
    opts = vq.RKSOptions()
    opts.functional = func
    res = vq.run_rks(mol, basis, opts)

    n_occ = mol.n_electrons() // 2
    homo_ha = res.mo_energies[n_occ - 1]
    homo_ev = homo_ha * 27.211386
    print(f"{func:8s}  E_HOMO = {homo_ha:+.5f} Ha = {homo_ev:+.2f} eV "
          f"→ IP estimate {-homo_ev:.2f} eV")
```

The script prints one line per functional. The experimental
vertical ionization potential of water is **12.62 eV** (NIST).
On a vibe-qc 0.17.10 development build (43 basis functions; the three SCFs
took two to three and a half minutes each on a heavily loaded shared
machine with two threads, so allow a few minutes) the output was:

| Functional | E (Ha) | $\varepsilon_\text{HOMO}$ (eV) | HOMO-derived IP | Error vs experiment |
|---|---:|---:|---:|---:|
| B3LYP | -76.421345 | -8.59 | 8.59 eV | -4.03 eV |
| PBE0 | -76.373204 | -8.97 | 8.97 eV | -3.65 eV |
| **ωB97X** | -76.435213 | -11.60 | **11.60 eV** | **-1.02 eV** |

B3LYP and PBE0 underestimate the IP by roughly 4 eV, the well-known
consequence of their wrong asymptotic exchange potential. ωB97X, with
100 % long-range exact exchange, cuts that error by about a factor of
four, but it does not close it: 1 eV remains, because ω is fixed
at 0.3 bohr⁻¹ rather than tuned for this molecule (see the Theory
section) and because a Kohn-Sham eigenvalue also carries the error of the
short-range and correlation parts. The gap between the conventional hybrids
and ωB97X, about 3 eV, is the headline: it is large on the scale of
orbital energies, and it is mostly an artifact of the asymptotic exchange
potential, not of the chemistry.

```{tip}
This is also why range-separated hybrids are the recommended
functionals for **frontier-orbital reasoning** (predicting
ionization, electron affinity, and charge-transfer excitations in
principle; the TDDFT solvers here still refuse range-separated kernels). For total energies and geometries
of routine closed-shell organics, B3LYP-D3(BJ) and PBE0-D3(BJ)
remain fine; the long-range exchange matters most for orbital
energies and CT.
```

## Open-shell ωB97X, the stiff-convergence caveat

ωB97X's 100 % long-range exact exchange makes open-shell SCF
**stiffer** than it is with a conventional hybrid. Closed-shell
molecules and well-behaved radicals (O₂, CH₃, atomic radicals)
converge with the defaults. An orbital-near-degenerate case,
the OH ²Π radical is the classic one, converges slowly and can plateau
near the gradient tolerance with plain DIIS. A level shift is the standard
remedy:

```python
mol = vq.Molecule(                       # OH radical, r = 1.83 bohr
    [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.83])],
    multiplicity=2,
)
basis = vq.BasisSet(mol, "def2-svp")

opts = vq.UKSOptions()
opts.functional  = "wb97x"
opts.level_shift = 0.5        # ← speeds up the OH ²Π SCF
result = vq.run_uks(mol, basis, opts)
```

On a vibe-qc 0.17.10 development build, the OH radical at ωB97X/def2-SVP
converged without a shift in 35 iterations and with `level_shift = 0.5` in
14, both to `E = -75.64629350` Ha, so on this build the shift speeds up a
slow SCF rather than rescuing a failed one. A run that does plateau is the
case the shift is for.

This is standard QC practice for stiff open-shell SCF, not a
vibe-qc quirk. See [Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS](ediis_diis_hybrid.md) and
[scf_convergence](../user_guide/scf_convergence.md) for the full
convergence-aid story.

```{note}
The **Newton / TRAH** second-order accelerators are disabled for
RSH functionals, because their orbital-Hessian exchange response is not
yet range-separation aware. DIIS + SOSCF + the quadratic fallback
carry convergence. **Analytic gradients** for RSH functionals are
also not yet available; finite-difference gradients work
(`run_job(method="rks", functional="wb97x", optimize=True)` uses
the FD path automatically).
```

## What's shipped vs queued

The table below summarises which range-separated features are live in
v0.8.0 and which are still queued, from ωB97X energies and gradients
through ωB97X-D, HSE06, and the periodic RSH path:

| | Status |
|---|---|
| ωB97X molecular RKS / UKS energy | ✅ shipped (v0.8.0), parity to PySCF.dft at grid precision |
| ωB97X finite-difference gradient / geometry opt | ✅ works |
| ωB97X analytic gradient | ⏳ queued |
| ωB97X-D molecular RKS / UKS (`run_wb97x_d`) | ✅ shipped, range-separated XC + intrinsic CHG dispersion |
| ωB97X-V / ωB97M-V | ✅ shipped (VV10 nonlocal correlation landed); see [Modern functionals: VV10 and the ωB97X-V / ωB97M-V wave](vv10_modern_functionals.md) |
| HSE06 (screened RSH) | ✅ shipped, molecular RKS / UKS, via the erf-form CAM machinery (negative long-range coefficient; no separate erfc kernel) |
| Periodic range-separated hybrids | ⏳ queued, periodic erf-attenuated K |

## Resources

Water / def2-TZVP / ωB97X: 43 basis functions, direct SCF, the
erf-attenuated K roughly doubles the per-iteration Fock cost
vs a conventional hybrid. ωB97X needed 13 SCF iterations here against 8 for
B3LYP and PBE0. The three-functional comparison above took 2 to 3.5
minutes per functional on a shared, heavily loaded machine at two threads;
with the default def2-SVP basis or a free workstation it is far quicker.

## References

- **ωB97X.** J.-D. Chai, M. Head-Gordon, "Systematic optimization
  of long-range corrected hybrid density functionals,"
  *J. Chem. Phys.* **128**, 084106 (2008).
  [doi:10.1063/1.2834918](https://doi.org/10.1063/1.2834918).
- **Long-range correction, the original idea.** H. Iikura,
  T. Tsuneda, T. Yanai, K. Hirao, "A long-range correction scheme
  for generalized-gradient-approximation exchange functionals,"
  *J. Chem. Phys.* **115**, 3540 (2001).
  [doi:10.1063/1.1383587](https://doi.org/10.1063/1.1383587).
- **CAM-B3LYP, the CAM partitioning.** T. Yanai, D. P. Tew,
  N. C. Handy, "A new hybrid exchange-correlation functional using
  the Coulomb-attenuating method (CAM-B3LYP)," *Chem. Phys. Lett.*
  **393**, 51 (2004).
  [doi:10.1016/j.cplett.2004.06.011](https://doi.org/10.1016/j.cplett.2004.06.011).
- **Optimal tuning of ω.** T. Stein, H. Eisenberg, L. Kronik, R. Baer,
  "Fundamental gaps in finite systems from eigenvalues of a generalized
  Kohn-Sham method," *Phys. Rev. Lett.* **105**, 266802 (2010).
  The tuning condition quoted in the Theory section.
- **HOMO = −IP in exact KS-DFT.** J. P. Perdew, R. G. Parr,
  M. Levy, J. L. Balduz, "Density-Functional Theory for Fractional
  Particle Number: Derivative Discontinuities of the Energy,"
  *Phys. Rev. Lett.* **49**, 1691 (1982).
  [doi:10.1103/PhysRevLett.49.1691](https://doi.org/10.1103/PhysRevLett.49.1691).

The `wb97x` route is registered in the bundled
[citation database](../user_guide/citations.md), so every job
using ωB97X auto-emits Chai & Head-Gordon 2008 in its `.bibtex`
and `.references` siblings.

## Next

- [User guide, functionals](../user_guide/functionals.md)
  § "Range-separated hybrids (ωB97X, ωB97X-D)", the API reference
  this tutorial is the worked companion to.
- [DFT functional comparison](functional_comparison.md)
  - Jacob's-ladder walk-through (LDA → GGA → hybrid); ωB97X is the
  rung above the conventional hybrids.
- [Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS](ediis_diis_hybrid.md), the
  stiff-convergence toolkit, relevant for open-shell ωB97X.
