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

## One-paragraph theory

In exact Kohn-Sham DFT the highest occupied orbital energy equals
minus the ionization potential, $\varepsilon_{\text{HOMO}} =
-\text{IP}$ (Janak's theorem in the exact-functional limit).
Approximate functionals violate this badly: the spurious
**self-interaction error** of (semi-)local exchange makes the
exchange-correlation potential decay too fast at large $r$, it
falls off exponentially instead of as $-1/r$. The HOMO sits in
that wrong asymptotic potential and comes out far too shallow;
B3LYP typically underestimates the IP of a small organic molecule
by 3-5 eV when read off the HOMO eigenvalue.

A range-separated hybrid splits the Coulomb operator
$$
\frac{1}{r_{12}} = \underbrace{\frac{1-[\alpha+\beta\,\text{erf}(\omega r_{12})]}{r_{12}}}_{\text{semi-local DFT exchange}} + \underbrace{\frac{\alpha+\beta\,\text{erf}(\omega r_{12})}{r_{12}}}_{\text{exact HF exchange}}
$$
so the exact-exchange fraction is $\alpha$ at $r_{12}\to 0$ and
$\alpha+\beta$ at $r_{12}\to\infty$. With $\alpha+\beta = 1$ the
functional has **100 % exact exchange at long range**, which
restores the correct $-1/r$ asymptotic potential, removes the
bulk of the self-interaction error, and brings
$\varepsilon_{\text{HOMO}}$ back close to $-\text{IP}$.

ωB97X uses $\alpha = 0.1577$, $\beta = 0.8423$ (so
$\alpha+\beta=1$), and a range-separation parameter
$\omega = 0.3\ \text{bohr}^{-1}$.

## Running ωB97X

ωB97X is a libxc functional like any other, just name it:

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("water.xyz")
basis = vq.BasisSet(mol, "def2-tzvp")

opts = vq.RKSOptions()
opts.functional = "wb97x"
result = vq.run_rks(mol, basis, opts)

print(f"E(ωB97X/def2-TZVP) = {result.energy:.6f} Ha")
```

vibe-qc reads the $(\omega, \alpha, \beta)$ triple from libxc at
construction and the SCF driver builds a **second, erf-attenuated
exchange matrix** $K_{\text{erf}}(\omega)$ alongside the ordinary
$K$. You can inspect the parameters:

```python
fn = vq.Functional("wb97x")
print(fn.is_range_separated)   # True
print(fn.rsh_omega)            # 0.3
print(fn.cam_alpha)            # 0.1577  (short-range HF fraction)
print(fn.cam_beta)             # 0.8423  (the long-range ramp)
```

```{important}
**ωB97X runs through direct SCF only.** The erf-attenuated $K$ is
built by the direct-SCF Fock kernel; there is no erf-attenuated
3-centre integral path yet, so `density_fit=True` with a
range-separated functional is rejected with a clear error. The
molecular drivers select the direct builder automatically for RSH
functionals — just leave `density_fit` at its default.
```

## The worked example: HOMO ≈ −IP across three functionals

Compute water at a fixed geometry with three functionals and
read the HOMO eigenvalue off each:

```python
# input-wb97x-homo.py
import vibeqc as vq

mol = vq.Molecule.from_xyz("water.xyz")
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
Run the script yourself for the exact numbers; the qualitative
picture is fixed by the physics and looks like this:

| Functional | HOMO-derived IP (def2-TZVP) | Error vs experiment |
|---|---:|---:|
| B3LYP | ≈ 7-8 eV | ≈ −5 eV |
| PBE0 | ≈ 8 eV | ≈ −4.5 eV |
| **ωB97X** | **≈ 12-13 eV** | **≈ ±0.5 eV** |

B3LYP and PBE0 underestimate the IP by 4-5 eV, the well-known
consequence of their wrong asymptotic exchange potential. ωB97X,
with 100 % long-range exact exchange, lands within roughly half
an eV. This is *not* a fitted result, it's a structural
consequence of getting the $-1/r$ tail right. The size of the
gap between the conventional hybrids and ωB97X is the headline:
4-5 eV is enormous on the scale of orbital energies, and it is
entirely an artifact of the asymptotic exchange potential, not
of the chemistry.

```{tip}
This is also why range-separated hybrids are the recommended
functionals for **frontier-orbital reasoning** (predicting
ionization, electron affinity, and — once TD-DFT lands —
charge-transfer excitations). For total energies and geometries
of routine closed-shell organics, B3LYP-D3(BJ) and PBE0-D3(BJ)
remain fine; the long-range exchange matters most for orbital
energies and CT.
```

## Open-shell ωB97X, the stiff-convergence caveat

ωB97X's 100 % long-range exact exchange makes open-shell SCF
**stiffer** than it is with a conventional hybrid. Closed-shell
molecules and well-behaved radicals (O₂, CH₃, atomic radicals)
converge with the defaults. An orbital-near-degenerate case, 
the OH ²Π radical is the classic one, can plateau near the
gradient tolerance with plain DIIS. The fix is a level shift:

```python
mol = vq.Molecule.from_xyz("oh.xyz", multiplicity=2)
basis = vq.BasisSet(mol, "def2-tzvp")

opts = vq.UKSOptions()
opts.functional  = "wb97x"
opts.level_shift = 0.5        # ← clears the OH ²Π plateau
result = vq.run_uks(mol, basis, opts)
```

This is standard QC practice for stiff open-shell SCF, not a
vibe-qc quirk. See [Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS](ediis_diis_hybrid.md) and
[scf_convergence](../user_guide/scf_convergence.md) for the full
convergence-aid story.

```{note}
The **Newton / TRAH** second-order accelerators are disabled for
RSH functionals — their orbital-Hessian exchange response is not
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

Water / def2-TZVP / ωB97X: ~60 basis functions, direct SCF, the
erf-attenuated K roughly doubles the per-iteration Fock cost
vs a conventional hybrid. The three-functional comparison above
runs in well under a minute on a laptop.

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
