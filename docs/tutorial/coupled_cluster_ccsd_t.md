# Canonical CCSD(T): the gold-standard reference

Coupled cluster with singles, doubles, and perturbative triples,
CCSD(T), is the accuracy reference that every cheaper method (MP2, DFT,
and the [DLPNO local approximation](dlpno_local_correlation.md)) is
measured against. vibe-qc ships it as a **production** route at
benchmark scale (label defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)):
dense $O(N^6)$ CCSD kernels with closed-shell, open-shell, and spin-pure
references reachable from one call. Feasibility depends on occupied and
virtual dimensions and the available memory, not a universal atom-count limit. Density fitting is on by default;
`density_fit=False` switches to exact four-index integrals. Very large
jobs are stopped by a dense memory gate instead of being attempted (see
[What is and is not available](#what-is-and-is-not-available)). This page
is the worked companion to the [CCSD user guide](../user_guide/ccsd.md);
it walks through each reference type with reference numbers from a
vibe-qc 0.17.10 development build on an Apple M-series laptop, and the
[Theory](#theory) section explains what is being computed. The wider
ladder of coupled-pair and approximate-triples methods (CC2, CCD, CEPA,
QCISD, CC3, CCSDT) is run on one molecule in
[The coupled-cluster family](coupled_cluster_family.md).

## Before running

Use the installed engine's Python in a run directory outside the checkout.
Coordinates in the examples are in bohr and energies are in hartree. Run the
first block before the continuations; it defines `vq`, the water molecule,
and a helper that checks both the SCF and coupled-cluster stages. The radical
examples define a different molecule and use distinct output stems.

Start with the cc-pVDZ water calculation. Atom count alone does not establish
that a correlated calculation fits: inspect the occupied/virtual dimensions,
frozen-core choice and memory estimate before increasing the basis. The
historical numbers below are not new measurements from your installation.

## Closed-shell CCSD(T) in one call

For a closed-shell molecule, `method="ccsd(t)"` runs RHF and then the
spin-adapted CCSD(T) kernel automatically:

```python
import vibeqc as vq

def require_cc_convergence(result):
    if not result.converged:
        raise RuntimeError("SCF reference did not converge")
    if not result.ccsd.converged:
        raise RuntimeError("Coupled-cluster amplitudes did not converge; inspect cc_trace")
    return result.ccsd

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])

result = vq.run_job(mol, basis="cc-pvdz", method="ccsd(t)", output="h2o_ccsd_t")
cc = require_cc_convergence(result)
water_ccsd_t = result
print(f"E(CCSD correlation) = {cc.e_ccsd_correlation:.6f} Ha")
print(f"E((T) increment)    = {cc.e_t:.6f} Ha")
print(f"E(CCSD(T) total)    = {result.energy_total:.6f} Ha")
```

The post-SCF result is attached as `result.ccsd`, a `CCSDResult`
carrying `converged`, `e_hf`, `e_ccsd_correlation`, `e_ccsd`, `e_t`, `e_ccsd_t`
(= `e_total`), and the per-iteration `cc_trace`. The RI auxiliary basis
is resolved automatically from the orbital basis name. The wrapper's
`result.converged` describes the SCF reference; check `result.ccsd.converged`
for the correlated stage. Use `energy_total` for the correlated total rather
than the forwarded SCF `energy`. Neither convergence flag establishes that
the reference is suitable for the scientific question.

A run with the defaults (frozen oxygen 1s core, density fitting) printed:

```
E(CCSD correlation) = -0.214651 Ha
E((T) increment)    = -0.003264 Ha
E(CCSD(T) total)    = -76.238722 Ha
```

`run_job` freezes chemical cores by default, so these numbers are not the
all-electron ones that other programs usually quote. With
`frozen_core=False` the same call gave a (T) increment of -0.003287 Ha and
E(CCSD(T)) = -76.240756 Ha. The all-electron ORCA 6.1.1 reference total is
E(CCSD(T)) = -76.2406 Ha (`tests/test_ccsd.py`), and vibe-qc reproduces its
(T) increment to under 3e-5 Ha; the (T) term is nearly independent of
density fitting, and the remaining difference is the cc-pVDZ
density-fitting error on the total, 0.14 mHa in the run above. To remove
that error entirely (strict conventional parity), add `density_fit=False`
to the `CCSDOptions`: the coupled-cluster step then runs on exact
four-index integrals, matching conventional kernels in other programs to a
few nanohartree (see the
[user guide](../user_guide/ccsd.md#conventional-non-df-integrals)).

```python
# doc-audit: skip - continuation of the setup and preceding examples
conv = vq.run_job(mol, basis="cc-pvdz", method="ccsd(t)", frozen_core=False,
                  ccsd_options=vq.CCSDOptions(density_fit=False),
                  output="h2o_ccsd_t_conventional_all_electron")
require_cc_convergence(conv)
print(f"E(CCSD(T), conventional, all electron) = {conv.energy_total:.6f} Ha")
# E(CCSD(T), conventional, all electron) = -76.240615 Ha
```

The all-electron conventional total, -76.240615 Ha, agrees with ORCA's
-76.2406 Ha to the four digits ORCA is quoted to.

### Reading the T1 diagnostic

Whether CCSD(T) can be trusted for a given molecule is decided by the
reference, not by the code. The converged singles amplitudes are on the
result, and their size is the standard warning light (the
[Theory](#theory) section says why):

```python
# doc-audit: skip - continuation of the setup and preceding examples
import numpy as np

t1 = np.asarray(result.ccsd.t1_amplitudes)        # (n_occ_corr, n_virtual)
t1_diagnostic = np.linalg.norm(t1) / np.sqrt(t1.shape[0])
print(f"T1 diagnostic = {t1_diagnostic:.4f}")
# T1 diagnostic = 0.0095
```

Water sits well below the 0.02 rule of thumb for closed-shell organic
molecules. A value above it, or a (T) increment that is more than a few
percent of the CCSD correlation energy (here 1.5 percent), says the single
determinant is a poor reference and CCSD(T) should not be quoted as a
benchmark.

## CCSD without the triples: the selector

`method="ccsd"` runs plain CCSD; `method="ccsd(t)"` adds the
Raghavachari perturbative triples. You can also set the triples mode
explicitly, which is handy for comparing the two:

```python
# doc-audit: skip - continuation of the setup and preceding examples
for label, triples in (("none", "none"), ("perturbative", "(t)"),
                       ("lambda", "A-CCSD(T)")):
    res = vq.run_job(mol, basis="cc-pvdz", method="ccsd", triples=triples,
                     output=f"h2o_triples_{label}")
    require_cc_convergence(res)
    print(triples, res.energy_total)
```

`triples="none"` disables the correction even when the method keyword is
`ccsd(t)`, and the output and citation method label follow the effective
choice.

`triples="A-CCSD(T)"` runs the closed-shell asymmetric/Lambda triples
correction and records the effective method as `a-ccsd(t)`.

## Open-shell radicals: an automatic UHF reference

For an open-shell molecule (`multiplicity > 1`), the same
`method="ccsd(t)"` call runs UHF and then the spin-orbital UCCSD(T)
kernel, with no extra flags:

```python
# doc-audit: skip - continuation of the setup and preceding examples
ch3 = vq.Molecule([
    vq.Atom(6, [ 0.000,  0.000, 0.0]),
    vq.Atom(1, [ 2.039,  0.000, 0.0]),
    vq.Atom(1, [-1.019,  1.766, 0.0]),
    vq.Atom(1, [-1.019, -1.766, 0.0]),
], multiplicity=2)

result = vq.run_job(ch3, basis="cc-pvdz", method="ccsd(t)", output="ch3_uccsd_t")
require_cc_convergence(result)
print(f"E(UCCSD(T)) = {result.ccsd.e_ccsd_t:.6f} Ha")
```

The spin-orbital kernel evaluates the coupled-cluster equations directly
on the UHF reference, reproducing canonical UCCSD(T). For the methyl
radical in cc-pVDZ the (T) increment is near -0.00263 Ha, matching ORCA
to the same tolerances as the closed-shell case; the all-electron ORCA
reference total is -39.7182 Ha (`tests/test_uccsd.py`). With
`frozen_core=False` on the call above, vibe-qc gave a (T) increment of
-0.002630 Ha and E(UCCSD(T)) = -39.718273 Ha; the default frozen-core call
gives a different total, so use the all-electron form when reproducing that
archived value. If the UHF
reference fails to converge, CCSD is skipped (no numbers from an
unconverged reference); pass `uhf_options=vq.UHFOptions(level_shift=...)`
for stiff radicals.

## Spin-pure open shells: a ROHF reference

The default open-shell path uses UHF, which can carry minor spin
contamination. For a spin-pure restricted-open-shell reference, pass
`ccsd_reference="rohf"`; the same spin-orbital kernel runs on ROHF
orbitals (identical alpha and beta spatial orbitals, per-spin Fock
matrices):

```python
# doc-audit: skip - continuation of the setup and preceding examples
rohf_result = vq.run_job(ch3, basis="cc-pvdz", method="ccsd(t)",
                         ccsd_reference="rohf", output="ch3_rohf_ccsd_t")
require_cc_convergence(rohf_result)
print(f"ROHF-based CCSD(T) total = {rohf_result.energy_total:.6f} Ha")
```

## Cheaper triples: frozen natural orbitals

The (T) cost is dominated by the virtual-space size. Frozen natural
orbitals (FNO) keep the dominant virtual natural orbitals of the MP2
density and discard the rest, recovering almost all the correlation at a
fraction of the cost. This continuation returns to the water molecule and
compares with `water_ccsd_t` retained from the first block. Turn it on through
`CCSDOptions`:

```python
# doc-audit: skip - continuation of the setup and preceding examples
result = vq.run_job(
    mol, basis="cc-pvdz", method="ccsd(t)",
    ccsd_options=vq.CCSDOptions(fno=True, fno_keep_fraction=0.6),
    output="h2o_fno_ccsd_t_60pct",
)
require_cc_convergence(result)
print(f"FNO change = {(result.energy_total - water_ccsd_t.energy_total) * 1000:+.3f} mHa")
print(result.ccsd.n_virtual_kept, "of", result.ccsd.n_virtual_total, "virtuals kept")
print(f"E(FNO-CCSD(T)) = {result.energy_total:.6f} Ha")
# 11 of 19 virtuals kept
# E(FNO-CCSD(T)) = -76.236223 Ha
```

Keeping 60 percent of the cc-pVDZ virtuals costs 2.5 mHa against the
untruncated -76.238722 Ha, which shows how aggressive that fraction is for
so small a basis; the saving is worth having only when the virtual space
runs to hundreds of functions (cc-pVTZ and larger), where the default
occupation threshold is the safer knob.

Selection is by MP2 occupation threshold (`fno_occ_threshold`, default
1e-5) or a fixed `fno_keep_fraction`. The retained virtuals are
semicanonicalized so the (T) stays exact in the truncated space, and a
delta-MP2 correction (`fno_delta_mp2=True`, the default) adds back most
of the small correlation lost to truncation. With no truncation,
FNO-CCSD(T) reproduces canonical CCSD(T) to 1e-6 Ha
(`tests/test_ccsd_fno.py`, `MICRO_HA` tolerance). The high-level FNO route uses the same
chemical-core frozen-core default as canonical `run_job(method="ccsd(t)")`;
pass `frozen_core=False` alongside `CCSDOptions(fno=True, ...)` only when
matching an all-electron reference. FNO is closed-shell only so far.

## Frozen core

The high-level driver freezes chemical cores automatically for canonical and
DLPNO MP2/CCSD, restricted and unrestricted, including the FNO convenience
route. It follows ORCA 6.1 Table 2.69: ORCA's frozen-electron count is
converted to closed-shell spatial orbitals and summed over the atoms. The
uniform all-electron escape is:

```python
# doc-audit: skip - continuation of the setup and preceding examples
all_electron = vq.run_job(
    mol, basis="cc-pvdz", method="ccsd(t)", frozen_core=False,
    output="h2o_ccsd_t_all_electron",
)
require_cc_convergence(all_electron)
```

The low-level API takes an explicit spatial-orbital count:

```python
# doc-audit: skip - continuation of the setup and preceding examples
opts = vq.CCSDOptions(triples="(t)",
                      n_frozen_core=vq.chemical_core_orbital_count(mol))
```

## What is and is not available

```{note}
The canonical engine is a dense $O(N^6)$ **molecular** kernel: production
at benchmark scale, accurate but not large-scale. For bigger systems use
the [DLPNO local approximation](dlpno_local_correlation.md).

* References: RHF (closed-shell), UHF (open-shell default), and ROHF
  (`ccsd_reference="rohf"`) are all supported.
* Density fitting is the supported default integral path
  (`density_fit=False` for conventional integrals). The dense CC iteration
  and the FNO setup remain production-size memory gates: coverage of
  formic-acid-dimer/cc-pVTZ-sized jobs waits on measured end-to-end
  bounded-memory runs, and a job that will not fit is refused before it
  starts. See the [CCSD user guide](../user_guide/ccsd.md#requested-memory-triples).
* **No analytic CCSD(T) gradient yet**, energies are single-point.
  `run_job(optimize=True)` with a coupled-cluster method relaxes the
  mean-field reference surface and tags the result with the method it
  actually differentiated; it does not walk the CCSD(T) surface (see
  [Geometry optimization](../user_guide/geometry_optimization.md)).
* FNO is closed-shell only; open-shell FNO is a roadmap item.
* Ground-state CC2 is available for closed-shell RHF references through
  `method="cc2"`. Ground-state CC3 and full iterative CCSDT are available
  through `method="cc3"` and `method="ccsdt"` for compact benchmark spaces.
  CC3 currently requires a closed-shell RHF reference. They are dense
  determinant-space routes: CC3 on water in 6-31G already takes more than five
  minutes, so the [coupled-cluster family tutorial](coupled_cluster_family.md)
  runs them in STO-3G.
* `A-CCSD(T)` and `CCSD[T]` / `CCSD+T(CCSD)` are implemented for
  closed-shell CCSD.
* Periodic CCSD is a later-release item.
```

## Theory

Coupled-cluster theory writes the exact ground state as an exponential
of excitation operators acting on a reference determinant, and CCSD(T)
is the truncation that has become the working benchmark for
single-reference molecules. The paragraphs below give the ansatz, the
CCSD equations in words, the perturbative triples, and the cost and
reliability limits that decide when the number can be trusted.

### The exponential ansatz and CCSD

With the Hartree-Fock determinant $|\Phi_0\rangle$ as reference, coupled
cluster ({vibeqc-cite}`cizek_cc_1966`) takes

$$
|\Psi_\text{CC}\rangle = e^{\hat T}\,|\Phi_0\rangle,
\qquad
\hat T = \hat T_1 + \hat T_2 + \cdots,
\qquad
\hat T_1 = \sum_{ia} t_i^a\,\hat a_a^\dagger \hat a_i,
\quad
\hat T_2 = \tfrac14 \sum_{ijab} t_{ij}^{ab}\,\hat a_a^\dagger \hat a_b^\dagger \hat a_j \hat a_i ,
$$

where $i,j$ label occupied and $a,b$ virtual spin orbitals and the $t$ are
the cluster amplitudes. Truncating at $\hat T_1 + \hat T_2$ gives CCSD
({vibeqc-cite}`purvis_bartlett_ccsd_1982`). The amplitudes are not found
by minimising an energy; they are fixed by projecting the
similarity-transformed Schrodinger equation
$\bar H|\Phi_0\rangle = E|\Phi_0\rangle$, $\bar H = e^{-\hat T}\hat H e^{\hat T}$,
onto the singles and doubles, which leaves a set of coupled nonlinear
equations (terms up to fourth order in the amplitudes) solved iteratively
with DIIS. Once they converge, the energy needs only $t_i^a$ and $t_{ij}^{ab}$,

$$
E_\text{CCSD} = E_\text{HF} + \sum_{ia} f_{ia}\,t_i^a
 + \tfrac14 \sum_{ijab} \langle ij\|ab\rangle \bigl(t_{ij}^{ab} + 2\,t_i^a t_j^b\bigr),
$$

with $f_{ia}$ the Fock matrix (zero for canonical Hartree-Fock orbitals)
and $\langle ij\|ab\rangle$ the antisymmetrised two-electron integrals. The
exponential form is what makes CCSD **size extensive**: the energy of two
non-interacting fragments is the sum of their energies, which the
linear-wavefunction truncation CISD does not satisfy (the
[coupled-cluster family tutorial](coupled_cluster_family.md) shows the
gap numerically). The closed-shell kernel is spin-adapted; UHF and ROHF
references run the same equations in spin orbitals.

### The perturbative triples

Connected triples are the leading correction CCSD leaves out, and
iterating them costs $O(N^8)$. The (T) correction
({vibeqc-cite}`raghavachari_ccsdt_1989`) evaluates them once, after the CCSD
amplitudes converge, by fourth- and fifth-order perturbation theory,

$$
E_{(T)} = \frac{1}{36} \sum_{ijkabc}
  \bigl[t_{ijk}^{abc}(c) + t_{ijk}^{abc}(d)\bigr]\,
  D_{ijk}^{abc}\;t_{ijk}^{abc}(c),
\qquad
D_{ijk}^{abc} = \varepsilon_i + \varepsilon_j + \varepsilon_k - \varepsilon_a - \varepsilon_b - \varepsilon_c ,
$$

where $t(c)$ are the connected triples built from the doubles amplitudes
and the integrals, $t(d)$ the disconnected ones built from the singles, and
$\varepsilon$ the orbital energies. The whole thing is one non-iterative
pass over occupied triples and virtual triples, $O(o^3 v^4)$, so CCSD
scales as $N^6$ and (T) as $N^7$: doubling the molecule multiplies the
cost by about 64 and 128. It is negative and small, 1.5 percent of the
correlation energy for water above. This combination recovers the
correlation energy of a well-behaved closed-shell molecule typically to
within about 1 kcal/mol of full CI in a given basis, which is why it is
called the gold standard.

### What "gold standard" does and does not cover

The claim is conditional on a single dominant determinant. When static
correlation matters (stretched bonds, diradicals, many transition-metal
complexes) the singles amplitudes grow, the (T) term is no longer a small
perturbation, and it can overshoot badly; the T1 diagnostic
$\|\mathbf t_1\|_F / \sqrt{n_\text{occ}}$ used in the section above (with
$n_\text{occ}$ the number of correlated occupied orbitals, which equals the
spin-orbital form divided by the square root of the number of correlated
electrons) is the standard early warning, with values above about 0.02 (Lee and Taylor,
*Int. J. Quantum Chem. Symp.* **23**, 199 (1989)) marking doubtful
references. Two approximations then set the practical error bars beyond
the method itself. Density fitting the four-index integrals
({vibeqc-cite}`deprince_sherrill_df_ccsd_2013`) cuts the cost and
storage of the integral step at an error of order 0.1 mHa on the total
(0.14 mHa above) and about 5e-6 Ha on the (T) increment; the
$N^6$ contraction cost is unchanged. Frozen natural orbitals
({vibeqc-cite}`taube_bartlett_fno_2008`) attack that cost instead, by
truncating the virtual space to the natural orbitals of the MP2 density
that carry occupation above a threshold, with a correction for the
discarded MP2 energy (`fno_delta_mp2`). For systems beyond the reach of
either, the DLPNO tutorial applies the same equations in a localised
basis.

## References

- **The (T) correction.** K. Raghavachari, G. W. Trucks, J. A. Pople,
  M. Head-Gordon, "A fifth-order perturbation comparison of electron
  correlation theories," *Chem. Phys. Lett.* **157**, 479 (1989).
- **Density-fitted CCSD(T) and frozen natural orbitals.** A. E. DePrince,
  C. D. Sherrill, "Accuracy and efficiency of coupled-cluster theory
  using density fitting, frozen natural orbitals, and a t1-transformed
  Hamiltonian," *J. Chem. Theory Comput.* **9**, 2687 (2013).
- **Coupled cluster.** J. Cizek, *J. Chem. Phys.* **45**, 4256 (1966);
  G. D. Purvis III and R. J. Bartlett, *J. Chem. Phys.* **76**, 1910 (1982)
  (CCSD).
- **FNO.** A. G. Taube and R. J. Bartlett, "Frozen natural orbital
  coupled-cluster theory: Forces and application to decomposition of
  nitroethane," *J. Chem. Phys.* **128**, 164101 (2008).
- **T1 diagnostic.** T. J. Lee and P. R. Taylor, *Int. J. Quantum Chem.
  Symp.* **23**, 199 (1989).

The `ccsd` / `ccsd(t)` routes are registered in the bundled
[citation database](../user_guide/citations.md), so a run auto-emits the
right references in its `.bibtex` and `.references` siblings.

## Next

- [CCSD user guide](../user_guide/ccsd.md), the full API reference
  (options, result fields, the AutoCI-style `citype=` selector).
- [The coupled-cluster family](coupled_cluster_family.md), the ladder from
  MP2 through CC2, CCD, CEPA and QCISD to CC3 and CCSDT on one molecule.
- [DLPNO local correlation](dlpno_local_correlation.md), the
  near-linear-scaling approximation to the CCSD(T) computed here.
- [Natural orbitals](natural_orbitals.md), the natural-orbital machinery
  FNO-CCSD(T) builds on.
- [RI-MP2](ri_mp2_aux_verification.md), the cheaper correlated method
  that CCSD(T) is the reference for.
