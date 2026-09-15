# Canonical CCSD(T): the gold-standard reference

Coupled cluster with singles, doubles, and perturbative triples,
CCSD(T), is the accuracy reference that every cheaper method (MP2, DFT,
and the [DLPNO local approximation](dlpno_local_correlation.md)) is
measured against. vibe-qc ships it as a dense O(N^6) molecular pilot on
a density-fitting substrate, with closed-shell, open-shell, and
spin-pure references all reachable from one call. This page is the
worked companion to the [CCSD user guide](../user_guide/ccsd.md); it
walks through each reference type with the validated reference numbers.

## Closed-shell CCSD(T) in one call

For a closed-shell molecule, `method="ccsd(t)"` runs RHF and then the
spin-adapted CCSD(T) kernel automatically:

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])

result = vq.run_job(mol, basis="cc-pvdz", method="ccsd(t)", output="h2o_ccsd_t")
cc = result.ccsd
print(f"E(CCSD correlation) = {cc.e_ccsd_correlation:.6f} Ha")
print(f"E((T) increment)    = {cc.e_t:.6f} Ha")
print(f"E(CCSD(T) total)    = {result.energy_total:.6f} Ha")
```

The post-SCF result is attached as `result.ccsd`, a `CCSDResult`
carrying `e_hf`, `e_ccsd_correlation`, `e_ccsd`, `e_t`, `e_ccsd_t`
(= `e_total`), and the per-iteration `cc_trace`. The RI auxiliary basis
is resolved automatically from the orbital basis name.

For water in cc-pVDZ the (T) increment is about -0.00328 Ha. vibe-qc
reproduces ORCA 6.1.1's conventional CCSD(T) here to under 3e-5 Ha on
the (T) increment (which is density-fitting insensitive) and to within
roughly 1 mHa on the total (the cc-pVDZ density-fitting error). The
all-electron ORCA reference total is E(CCSD(T)) = -76.2406 Ha
(`tests/test_ccsd.py`); note that `run_job` freezes chemical cores by
default, so pass `frozen_core=False` to match that all-electron convention.
To remove the density-fitting error
entirely (strict conventional parity), add `density_fit=False` to the
`CCSDOptions`: the coupled-cluster step then runs on exact four-index
integrals, matching conventional kernels in other programs to a few
nanohartree (see the
[user guide](../user_guide/ccsd.md#conventional-non-df-integrals)).

## CCSD without the triples: the selector

`method="ccsd"` runs plain CCSD; `method="ccsd(t)"` adds the
Raghavachari perturbative triples. You can also set the triples mode
explicitly, which is handy for comparing the two:

```python
for triples in ("none", "(t)", "A-CCSD(T)"):
    res = vq.run_job(mol, basis="cc-pvdz", method="ccsd", triples=triples)
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
ch3 = vq.Molecule([
    vq.Atom(6, [ 0.000,  0.000, 0.0]),
    vq.Atom(1, [ 2.039,  0.000, 0.0]),
    vq.Atom(1, [-1.019,  1.766, 0.0]),
    vq.Atom(1, [-1.019, -1.766, 0.0]),
], multiplicity=2)

result = vq.run_job(ch3, basis="cc-pvdz", method="ccsd(t)")
print(f"E(UCCSD(T)) = {result.ccsd.e_ccsd_t:.6f} Ha")
```

The spin-orbital kernel evaluates the coupled-cluster equations directly
on the UHF reference, reproducing canonical UCCSD(T). For the methyl
radical in cc-pVDZ the (T) increment is near -0.00262 Ha, matching ORCA
to the same tolerances as the closed-shell case; the all-electron ORCA
reference total is -39.7182 Ha (`tests/test_uccsd.py`). Use
`frozen_core=False` on the call above when reproducing that archived
all-electron value. If the UHF
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
result = vq.run_job(oh_radical, basis="cc-pvdz",
                    method="ccsd(t)", ccsd_reference="rohf")
```

## Cheaper triples: frozen natural orbitals

The (T) cost is dominated by the virtual-space size. Frozen natural
orbitals (FNO) keep the dominant virtual natural orbitals of the MP2
density and discard the rest, recovering almost all the correlation at a
fraction of the cost. Turn it on through `CCSDOptions`:

```python
result = vq.run_job(
    mol, basis="cc-pvtz", method="ccsd(t)",
    ccsd_options=vq.CCSDOptions(fno=True, fno_keep_fraction=0.6),
)
print(result.ccsd.n_virtual_kept, "of", result.ccsd.n_virtual_total, "virtuals kept")
```

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
all_electron = vq.run_job(
    mol, basis="cc-pvdz", method="ccsd(t)", frozen_core=False
)
```

The low-level API takes an explicit spatial-orbital count:

```python
opts = vq.CCSDOptions(triples="(t)",
                      n_frozen_core=vq.chemical_core_orbital_count(mol))
```

## What is and is not available

```{note}
The canonical engine is a dense O(N^6) **molecular** pilot, accurate but
not large-scale; for bigger systems use the
[DLPNO local approximation](dlpno_local_correlation.md).

* References: RHF (closed-shell), UHF (open-shell default), and ROHF
  (`ccsd_reference="rohf"`) are all supported.
* Density fitting is the supported default integral path. Production-size
  memory coverage remains gated on bounded end-to-end BUG 63 runs.
* **No analytic CCSD(T) gradient yet**, energies are single-point;
  geometry optimization runs through finite differences.
* FNO is closed-shell only; open-shell FNO is a roadmap item.
* Ground-state CC2 is available for closed-shell RHF references through
  `method="cc2"`. Ground-state CC3 and full iterative CCSDT are available
  through `method="cc3"` and `method="ccsdt"` for compact benchmark spaces.
  CC3 currently requires a closed-shell RHF reference.
* `A-CCSD(T)` and `CCSD[T]` / `CCSD+T(CCSD)` are implemented for
  closed-shell CCSD.
* Periodic CCSD is a later-release item.
```

## References

- **The (T) correction.** K. Raghavachari, G. W. Trucks, J. A. Pople,
  M. Head-Gordon, "A fifth-order perturbation comparison of electron
  correlation theories," *Chem. Phys. Lett.* **157**, 479 (1989).
- **Frozen natural orbitals for CCSD(T).** A. E. DePrince,
  C. D. Sherrill, "Accuracy and efficiency of coupled-cluster theory
  using density fitting, frozen natural orbitals, and a t1-transformed
  Hamiltonian," *J. Chem. Theory Comput.* **9**, 2687 (2013).

The `ccsd` / `ccsd(t)` routes are registered in the bundled
[citation database](../user_guide/citations.md), so a run auto-emits the
right references in its `.bibtex` and `.references` siblings.

## Next

- [CCSD user guide](../user_guide/ccsd.md), the full API reference
  (options, result fields, the AutoCI-style `citype=` selector).
- [DLPNO local correlation](dlpno_local_correlation.md), the
  near-linear-scaling approximation to the CCSD(T) computed here.
- [Natural orbitals](natural_orbitals.md), the natural-orbital machinery
  FNO-CCSD(T) builds on.
- [RI-MP2](ri_mp2_aux_verification.md), the cheaper correlated method
  that CCSD(T) is the reference for.
