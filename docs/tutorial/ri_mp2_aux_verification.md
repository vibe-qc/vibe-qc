# Verifying an RI-MP2 auxiliary basis

When you reach for RI-MP2 in production, you trade O(N⁵) canonical
AO→MO work for an O(N⁴) auxiliary-basis fit, and you inherit the
fit's residual on top of the basis-set-incompleteness error you'd
have had anyway. For most workflows the per-zeta RIfit family
(`cc-pvNz-ri`, `def2-Nzv-rifit`) is tight enough that the fit
residual stays well below ~µHa per atom. But when you move to a new
system class, switch zeta, or accidentally pair an orbital basis
with the wrong aux family, the residual can balloon by an order of
magnitude, and the silent failure mode is a wrong SCS / SOS /
double-hybrid number, not a crash.

This tutorial walks through `MP2Options.report_ri_residual`, the
opt-in diagnostic added in
[#mp2-ri-residual](../user_guide/mp2_and_double_hybrids.md#verifying-the-ri-fit-report_ri_residual),
on water with two aux choices. It complements the
[double-hybrid B2PLYP tutorial](double_hybrid_b2plyp.md) (which is
where you'd actually consume this diagnostic in anger, the post-SCF
MP2 step in a B2PLYP run inherits the same RI fit error).

## The structural fingerprint of a Coulomb-metric Dunlap fit

RI-MP2 builds the four-index ERI from the Coulomb-metric fit

$$(ia|jb)_{\mathrm{RI}} = \sum_{PQ} (ia|P)\, [V^{-1}]_{PQ}\, (Q|jb),
\quad V_{PQ} = (P|Q).$$

The fit residual has a **definite sign pattern** in the Grimme
decomposition: $E_{\mathrm{os}} = \sum (ia|jb)^2 / \Delta$ is
over-estimated (less negative than canonical), and $E_{\mathrm{ss}} =
\sum [(ia|jb)^2 - (ia|jb)(ib|ja)] / \Delta$ is under-estimated (more
negative than canonical). The two errors partially cancel in the
unscaled $E_{\mathrm{corr}} = E_{\mathrm{os}} + E_{\mathrm{ss}}$ but
survive SCS / SOS scaling, because SCS weights the OS bucket
(`c_os = 6/5`) more than three times as heavily as the SS bucket
(`c_ss = 1/3`).

A wrong-direction signature in the diagnostic (`ΔE_os < 0`,
`ΔE_ss > 0`, or both with the same sign) almost certainly indicates
a numerical bug in the fit pipeline, not a fit-basis issue. The
diagnostic is exactly the right tool to catch that quickly.

## Running the diagnostic

The example runs RHF on water/cc-pVTZ, then an SCS-MP2 with `cc-pvtz-ri`
and `report_ri_residual=True`, which reports the per-bucket fit error
alongside the OS and SS correlation energies:

```python
import vibeqc as vq

mol = vq.Molecule(
    [
        vq.Atom(8, [0.0,  0.00,  0.00]),
        vq.Atom(1, [0.0,  1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)
basis = vq.BasisSet(mol, "cc-pvtz")

rhf_opts = vq.RHFOptions(); rhf_opts.conv_tol_energy = 1e-10
hf = vq.run_rhf(mol, basis, rhf_opts)

opts = vq.MP2Options()
opts.density_fit = True
opts.aux_basis = "cc-pvtz-ri"
opts.c_os = 6.0 / 5.0          # SCS-MP2 (Grimme 2003)
opts.c_ss = 1.0 / 3.0
opts.report_ri_residual = True  # ← opt-in flag

mp2 = vq.run_mp2(mol, basis, hf, opts)

assert mp2.ri_residual_reported
print(f"E_os = {mp2.e_os:+.8f} Ha   "
      f"ΔE_os = {mp2.e_os_ri_residual:+.2e}")
print(f"E_ss = {mp2.e_ss:+.8f} Ha   "
      f"ΔE_ss = {mp2.e_ss_ri_residual:+.2e}")
```

Output:

```
E_os = -0.20613056 Ha   ΔE_os = +5.20e-05
E_ss = -0.06591474 Ha   ΔE_ss = -2.43e-05
```

`ΔE_os > 0` and `ΔE_ss < 0`, the expected Coulomb-metric Dunlap
fingerprint. With `c_os = 6/5` and `c_ss = 1/3`:

$$
\Delta E_{\mathrm{SCS}} \approx 1.2 \cdot (+5.20 \times 10^{-5})
+ 0.333 \cdot (-2.43 \times 10^{-5})
= +5.4 \times 10^{-5} \mathrm{\,Ha} \approx 0.034 \mathrm{\,kcal/mol}.
$$

For chemical accuracy (1 kcal/mol) this is well inside tolerance; for
chasing sub-µHa parity against a reference calculation it is not.

## Aux-basis sweep, confirming a genuine fit residual

A fit residual should shrink monotonically as you enlarge the
aux basis. Re-running on `cc-pvqz-ri` (242 functions vs 141 for
`cc-pvtz-ri`):

| Aux basis | $n_{\mathrm{aux}}$ | $\Delta E_{\mathrm{os}}$ | $\Delta E_{\mathrm{ss}}$ | $\Delta E_{\mathrm{SCS}}$ |
|---|---:|---:|---:|---:|
| cc-pvtz-ri | 141 | +5.2e-5 | −2.4e-5 | +5.4e-5 |
| cc-pvqz-ri | 242 | +2.0e-5 | −4.4e-6 | +2.2e-5 |

Both rows show the same sign pattern; the magnitudes drop roughly
2×, the right signature for a fit-residual sweep.

If you run the same sweep and the residual stays flat or grows with
aux size, that is *not* a fit-residual signature, it's evidence of
a code bug (Cholesky / SVD propagation, mis-routed weights), an
ill-conditioned $V_{PQ}$, or an aux family that is misaligned with
the orbital basis. Open an issue with the diagnostic output and the
$V_{PQ}$ condition number (the latter is on
`vibeqc.density_fitting.DensityFitting(...).info.metric_condition_estimate`).

## When to pair this diagnostic with a code

* **Before publishing a new aux-basis recommendation.** Run the
  diagnostic on a representative set; the per-system bucket residual
  is your "this aux is fit for this purpose" evidence.
* **Before cross-code parity work.** If two codes' RI-MP2 numbers
  disagree by more than the per-code RI residuals against canonical,
  one of them is doing something other than the standard
  Coulomb-metric fit (e.g. one might be running canonical silently, 
  always confirm both codes are running RI as advertised).
* **As a unit-test guard.** Pin the residual magnitude alongside
  the energy itself in your test suite, a regression in the RI
  pipeline shows up as a residual shift, not just an energy shift.

## Cost note

The diagnostic **doubles the runtime of an RI-MP2 call**, it forces
the O(N⁵) canonical AO→MO transform alongside the O(n_aux · …) RI
build. Keep it off in production loops; only switch it on when you
need the answer it gives.

```python
opts.report_ri_residual = False   # ← default, production
```

## References

- **RI-MP2 (the algorithm).** F. Weigend, M. Häser, "RI-MP2: first
  derivatives and global consistency," *Theor. Chem. Acc.* **97**,
  331 (1997).
  [doi:10.1007/s002140050269](https://doi.org/10.1007/s002140050269).
  The original derivation of the Coulomb-metric-fitted MP2
  amplitudes vibe-qc implements.
- **Dunlap-fit theory.** B. I. Dunlap, "Robust and variational fitting:
  Removing the four-center integrals from center stage in quantum
  chemistry," *J. Mol. Struct. THEOCHEM* **529**, 37 (2000).
  [doi:10.1016/S0166-1280(00)00528-5](https://doi.org/10.1016/S0166-1280%2800%2900528-5).
  The "robust" fitting framework that pins the sign pattern of the
  residual.
- **Aux-basis families.** F. Weigend, A. Köhn, C. Hättig, "Efficient
  use of the correlation consistent basis sets in resolution of the
  identity MP2 calculations," *J. Chem. Phys.* **116**, 3175 (2002).
  [doi:10.1063/1.1445115](https://doi.org/10.1063/1.1445115) (the
  cc-pVNZ-RI family); F. Weigend, "Accurate Coulomb-fitting basis
  sets for H to Rn," *Phys. Chem. Chem. Phys.* **8**, 1057 (2006).
  [doi:10.1039/B515623H](https://doi.org/10.1039/B515623H) (the
  def2-*-rifit family).
- **SCS-MP2.** S. Grimme, *J. Chem. Phys.* **118**, 9095 (2003).
  [doi:10.1063/1.1569242](https://doi.org/10.1063/1.1569242), the
  scaling weights `c_os = 6/5`, `c_ss = 1/3` used in this tutorial.

All four fire automatically in the bundled
[citation database](../user_guide/citations.md) routes for
`density_fit=True` MP2 / `c_os` / `c_ss` jobs.

## Further reading

* [MP2 and double hybrids, Verifying the RI fit](../user_guide/mp2_and_double_hybrids.md#verifying-the-ri-fit-report_ri_residual)
  - reference page covering the same material plus the diagnostic's
  formal definition and limitations.
* [Density fitting](../user_guide/density_fitting.md), the J / K /
  MP2 fit-aux families vibe-qc ships, autodetect helper, $V_{PQ}$
  conditioning diagnostics.
* [Double hybrid B2PLYP](double_hybrid_b2plyp.md), the place
  where the post-SCF RI-MP2 step (and therefore this diagnostic)
  actually changes a publishable number.
* [`examples/molecular/input-h2o-ri-mp2-residual.py`](../../examples/molecular/input-h2o-ri-mp2-residual.py)
  - the runnable script behind this tutorial.
* [`examples/molecular/mp2_benchmarks/`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/mp2_benchmarks/README.md)
  - multi-system vibe-qc ↔ ORCA validation suite; the diagnostic is
  the right pre-flight check before regenerating its tables.
