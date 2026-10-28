# Local correlation with DLPNO

Canonical MP2 and coupled cluster scale steeply with system size
because they correlate every occupied orbital with every other one in
a delocalised basis. Domain-based local pair-natural-orbital (DLPNO)
methods exploit the short-sightedness of dynamic correlation instead:
occupied orbitals are localised, each orbital *pair* gets its own
compact virtual space (the pair natural orbitals), and pairs that are
far apart are estimated rather than solved. The art is that all of
this is controlled by a handful of thresholds, and as the thresholds
go to zero, the canonical answer comes back *exactly*.

```{note}
The output blocks and the threshold table on this page were measured on the
2026-09-29 development build. Afterwards, main changed three defaults of the
closed-shell route (#190): PAO domains now follow differential overlap, the
diagonal-pair singles PNOs use a scaled cut, and the amplitude equations are
solved in a compiled PNO-space kernel. Water's domains do not change, but the
DLPNO-CCSD and (T) energies and the loose, normal and tight errors below may
move at the micro- to millihartree level on a current build, so re-run the
snippets before quoting any number from here. The
[DLPNO guide](../user_guide/dlpno_mp2.md#pao-domains) describes the new
domains.
```

The DLPNO routes are **production** (label defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)):
the reduced-scaling closed- and open-shell solvers are validated against
their canonical limit, but the retained sub-kcal/mol accuracy evidence is
pinned to explicit threshold recipes and is not yet a claim for the current
chemical-core/NormalPNO default, so state the thresholds when you publish.
The [Theory](#theory) section at the end explains what the thresholds do.

This tutorial runs DLPNO-MP2 on water, shows the threshold
convergence against canonical RI-MP2, and finishes with the
local DLPNO-CCSD solver and DLPNO-CCSD(T). The
matching ready-to-run scripts are
[`input-h2o-dlpno-mp2.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-mp2.py),
[`input-h2o-dlpno-ccsd.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-ccsd.py),
and
[`input-h2o-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-ccsd-t.py).

All four public MP2/CCSD DLPNO routes, restricted and unrestricted, now use
the same project convention: ORCA 6.1 chemical-core counts and the supported
subset of Liakos NormalPNO. Pass `frozen_core=False` for the uniform
all-electron escape, and use `dlpno_thresholds="loose"`, `"normal"`, or
`"tight"` for a named published preset. The local open-shell CCSD route uses
the same three Liakos coordinates for its spin-orbital pairs. It does not yet
claim Saitow 2017's additional `TScaleTCutPairs=0.33`, which assumes a single
ROHF/QRO-like spatial-orbital set rather than independent UHF alpha and beta
spaces.

## Running the companion scripts

Copy a companion script into a separate directory and run it with the
installed engine's Python, for example:

```sh
python input-h2o-dlpno-mp2.py > console.log
```

The scripts write job artifacts next to the copied script and print their
additional comparisons to stdout. Keep `console.log` alongside the `.out`
and `.system` files; do not append analysis to a finalized job artifact.
Use a fresh directory for each recipe so repeated runs do not replace the
previous evidence.

The MP2 script compares local correlation with a separate canonical RI-MP2
calculation, sharing the RHF orbitals, chemical frozen core and fitting
basis. Its signed difference is local minus canonical in microhartree.
The CCSD script instead compares default and full-domain **local** CCSD on
one RHF reference; it does not run an independent canonical solver. Both
scripts stop on unconverged references or iterative amplitudes. Their
recovery percentages describe these water calculations, not an accuracy
bound for other geometries or relative energies. The CCSD(T) script reports
one default calculation and provides no independent triples comparison.

## Before running

Use a separate run directory and the installed engine's Python. Coordinates
below are in bohr; run the first block before the continuations. The one-call
example uses the chemical-core default, while the threshold experiment is
explicitly all-electron on both sides. Compare only results with matching
geometry, orbital and auxiliary bases, core count and reference settings.

Check both stages: the high-level wrapper forwards SCF convergence, while
`dlpno_mp2.converged` or `dlpno_ccsd.converged` describes the local solver.
Retain output files with the resolved thresholds rather than inferring them
from a preset name in an older example. Historical recovery percentages are
not a verdict on a new system or a changed threshold recipe.

## One-call DLPNO-MP2

The whole method is reachable through `run_job` with
`method="dlpno-mp2"`. This runs DLPNO-MP2 on water in a cc-pVDZ basis and
pulls the correlation energy off the result object (numbers below are from
a vibe-qc 0.17.10 development build on an Apple M-series laptop; the run
takes a few seconds):

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])

result = vq.run_job(mol, basis="cc-pvdz", method="dlpno-mp2", output="h2o_dlpno")
r = result.dlpno_mp2
if not result.converged or not r.converged:
    raise RuntimeError("SCF or DLPNO-MP2 did not converge")
print(r.e_corr, result.energy_total)
```

The `.out` file carries the decomposed solver block, iterated pair
energies, the PNO-truncation correction, and the distant-pair
estimate:

```
  DLPNO-MP2 (Pinski 2015; RI: cc-pvdz-ri)
  ---------------------------------------
  E(RHF reference)       =   -76.0243138804 Ha
  Threshold convention   = NormalPNO (supported subset)
  Truncation thresholds  = tcut_pairs 0.0001  tcut_pno 3.33e-07  tcut_mkn 0.001
  Inactive cutoffs       = tcut_pairs_weak, tcut_pno_weak
  Frozen core orbitals = 1 (ORCA 6.1 published count-only default)
  pairs kept / screened  = 10 / 0
  avg PNOs per pair      =           17.8
  E(iterated pairs)      =    -0.1982820663 Ha
  E(PNO truncation corr) =    -0.0000058960 Ha
  E(distant-pair est.)   =     0.0000000000 Ha
  E(DLPNO-MP2 corr)      =    -0.1982879623 Ha
  E(DLPNO-MP2 total)     =   -76.2226018427 Ha
```

With the oxygen 1s orbital frozen, water has 4 correlated localised
occupied orbitals, which give 4 diagonal and 6 off-diagonal pairs, 10 in
all; none is screened because every centroid lies within 8 bohr of every
other. Each pair keeps on average 17.8 of the 19 virtual orbitals, which
is little compression for so small a molecule (the saving appears when the
virtual space grows to hundreds of functions). Canonical frozen-core
RI-MP2 gives `-0.19832991` Ha for the same molecule, so the default
DLPNO-MP2 recovers 99.979 percent of the correlation energy, 42 microhartree
short.

## The experiment every DLPNO user should run once

How much correlation energy does the PNO truncation cost, and does
the method really converge to the canonical answer? Sweep `TCutPNO`
against canonical RI-MP2 with the *same* fitting basis:

```python
# doc-audit: skip - continues with mol and vq from the first example
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.mp2 import DLPNOMP2Options, run_dlpno_mp2

basis = vq.BasisSet(mol, "cc-pvdz")
hf = vq.run_rhf(mol, basis)
if not hf.converged:
    raise RuntimeError("RHF did not converge for the threshold study")

mp2_opts = vq.MP2Options()
mp2_opts.density_fit = True
mp2_opts.aux_basis = "cc-pvdz-ri"
canonical = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="mp2",
    frozen_core=False,
    mp2_options=mp2_opts,
    output="h2o_canonical_all_electron",
).mp2

df = DensityFitting(basis, vq.BasisSet(mol, "cc-pvdz-ri"),
                    aux_basis_name="cc-pvdz-ri")

for label, opts in (
    ("1e-07", DLPNOMP2Options(n_frozen=0, tcut_pairs=1e-6,
                               tcut_pairs_weak=1e-4, tcut_pno=1e-7,
                               tcut_pno_weak=1e-6, tcut_mkn=1e-3)),
    ("1e-08", DLPNOMP2Options(n_frozen=0, tcut_pairs=1e-6,
                               tcut_pairs_weak=1e-4, tcut_pno=1e-8,
                               tcut_pno_weak=1e-7, tcut_mkn=1e-3)),
    ("1e-09", DLPNOMP2Options(n_frozen=0, tcut_pairs=1e-6,
                               tcut_pairs_weak=1e-4, tcut_pno=1e-9,
                               tcut_pno_weak=1e-8, tcut_mkn=1e-3)),
    ("0*",    DLPNOMP2Options(n_frozen=0, tcut_pairs=0.0,
                               tcut_pairs_weak=0.0, tcut_pno=0.0,
                               tcut_pno_weak=0.0, tcut_mkn=0.0)),
):
    r = run_dlpno_mp2(mol, basis, hf, df, opts)
    if not r.converged:
        raise RuntimeError(f"DLPNO-MP2 did not converge for threshold {label}")
    avg = sum(r.pno_per_pair.values()) / len(r.pno_per_pair)
    print(f"{label:>6s}  {r.e_corr:14.8f}  "
          f"{r.e_corr / canonical.e_correlation:9.4%}  {avg:5.1f}")
```

Output (the code pins the all-electron convention and explicit thresholds on
both sides, so the rows do not depend on the project defaults; canonical
all-electron RI-MP2 correlation energy is `-0.20074079` Ha):

```
 1e-07     -0.20030243   99.7816%   14.4
 1e-08     -0.20051444   99.8872%   15.5
 1e-09     -0.20060605   99.9329%   17.0
    0*     -0.20074079  100.0000%   19.0
```

Three things to read off this table:

* **The error is controlled by one knob.** Tightening `TCutPNO` by 10×
  buys roughly a factor-of-2 smaller truncation error (0.22, 0.11 and 0.07
  percent of the correlation energy) at the cost of one to one and a half
  extra PNOs per pair.
* **The last row is the exactness limit**, with pair, PNO, weak-pair, and
  domain screens all zero (hence the `*`): DLPNO-MP2 reproduces
  canonical RI-MP2 to better than 1 µHa. This identity is asserted
  permanently in `tests/test_dlpno_mp2.py`, on canonical *and*
  Boys-localised occupieds.
* **Recovery percentages are measured, not advertised.** Remaining DLPNO
  accuracy and reduced-scaling work is tracked in the canonical
  [GitLab issue list](https://github.com/vibe-qc/vibe-qc/issues).

## Frozen core and distant-pair screening

The default frozen-core count follows ORCA 6.1 Table 2.69. ORCA tabulates
frozen electrons; vibe-qc converts that closed-shell count to spatial
orbitals, sums it over the atoms, and reports the resolved orbital count in
the method block. The same high-level switch works for canonical/DLPNO and
restricted/unrestricted MP2/CCSD routes:

```python
# doc-audit: skip - continues with mol and vq from the first example
default = vq.run_job(
    mol, basis="cc-pvdz", method="dlpno-mp2",
    dlpno_thresholds="normal",
    output="h2o_fc",
)
all_electron = vq.run_job(
    mol, basis="cc-pvdz", method="dlpno-mp2",
    frozen_core=False,       # ORCA NoFrozenCore
    dlpno_thresholds="normal",  # hold thresholds fixed while changing frozen core
    output="h2o_all_electron_normal",
)
```

In the zero-threshold limit, frozen-core DLPNO-MP2 is validated against a
canonical frozen-core reference to ≤ 1 µHa; at the default thresholds the
difference is the truncation error of the sweep above. Pair screening only acts on pairs whose
localised-orbital centroids are ≥ 8 bohr apart (nothing in a water
monomer qualifies); the screened pairs contribute through the
`e_distant` component instead of silently vanishing.

At the low level, `n_frozen` remains an explicit spatial-orbital count. For
example,
`options_from_dlpno_thresholds(LocalCCSDOptions, "tight")` returns a
`LocalCCSDOptions` object ready for `dlpno_ccsd_options=...`. Only thresholds
implemented by that solver are applied and reported; an omitted field is
unsupported, not silently zero.

## DLPNO-CCSD: the local solver

The same PNO machinery carries the coupled-cluster ansatz.
`method="dlpno-ccsd"` runs the **local solver**: each pair's amplitudes
remain in its PNO basis, and the default `residual_domain="pno"` contracts
every residual term in the PNO space of the amplitude that carries the
contracted index (Riplinger and Neese 2013, Sec. II C) in a compiled kernel.
No full-system amplitude tensor is formed. It is FCI-anchored; in the
full-domain limit it reproduces canonical closed-shell CCSD bit-for-bit
(≤ 1 µHa; on H₂ that is FCI), and at any truncation it reproduces the
atom-based extended-domain contraction (`residual_domain="extended"`) to the
convergence tolerance. The project default is the supported subset of
NormalPNO. The retained recovery percentages
below describe their explicitly labelled historical recipes rather than a
newly measured current-default value.

```python
# doc-audit: skip - continues with mol and vq from the first example
result = vq.run_job(mol, basis="cc-pvdz", method="dlpno-ccsd",
                    output="h2o_cc")
cc = result.dlpno_ccsd
if not result.converged or not cc.converged:
    raise RuntimeError("SCF or DLPNO-CCSD did not converge")
print(f"E_corr(CCSD) = {cc.e_corr:.8f}  total = {result.energy_total:.8f}")
```

Output:

```
E_corr(CCSD) = -0.20802771  total = -76.23234159
```

This is the default chemical-core, NormalPNO result; the run reports its
own resolved core count and thresholds in the `.out` (10 pairs, 17.8 PNOs
per pair here, as for DLPNO-MP2). For a custom run, pass
`dlpno_ccsd_options=LocalCCSDOptions(tcut_pno=1e-9)` (from
`vibeqc.dlpno.ccsd_local_solver`) to tighten the PNO truncation toward
the canonical limit, or tune `coupling_radius` (bohr; **default 12**),
how far each pair's occupied coupling set reaches, the lever that bounds
the occupied coupling. It is exact at a radius wider than the molecule
and a controlled approximation below the PNO truncation error once
fragments separate: in a chain of H₂ molecules the average
coupled-occupied count saturates while the system grows, instead of
tracking it. On a water this small everything is within the default
radius, so it is a no-op; the radius earns its keep on extended systems
(set `coupling_radius=0` for exact full coupling).

DLPNO fits the correlation integrals against an RI auxiliary basis, which
the runner auto-resolves from the orbital basis (the `RI:` line in the
log). Correlation-consistent (`cc-pVnZ`) and `def2` orbital bases have a
bundled default RI aux; **minimal / Pople bases** (`sto-3g`, `6-31g*`, …)
do not, and a DLPNO run on one raises
`NotImplementedError: No default ri-aux registered …`. Supply an aux
explicitly through the option object to override:
`dlpno_ccsd_options=LocalCCSDOptions(aux_basis="cc-pvdz-ri")` (closed
shell) or `DLPNOUCCSDPilotOptions(aux_basis="cc-pvdz-ri")` (open shell).
DLPNO on a minimal orbital basis is rarely physically meaningful, but the
override keeps it available for smoke tests and parity checks.

### CCSD(T): DLPNO-(T1)

`method="dlpno-ccsd(t)"` runs the same local solver and then the
rotated-occupied **(T1)** triples on the converged amplitudes
(`vibeqc.dlpno.triples_local`). The occupied space is rotated to canonical
orbitals and the full canonical (T) is evaluated on the rotated PNO
amplitudes, so the (T0) semicanonical approximation and the TNO truncation
are both gone and the PNO truncation is the only approximation left. Its cost
is that of a canonical (T), so on large molecules ``triples_mode="local"``
(below) is the mode that scales. The open-shell local solver separately exposes
`triples_mode="t1-iterative"`, a spin-orbital generalization of the 2018 Guo
et al. closed-shell Eq. (2); Guo et al., J. Chem. Phys. 152, 024116 (2020) is
the dedicated open-shell method-family source. Unlike the rotated route, it
can preserve a screened local triple list.
At full domains it reproduces canonical CCSD(T) to machine precision (the
(T1) parity ratchet). Set
``triples_mode="local"`` for DLPNO-(T0), ORCA's default triples (diagonal
localised Fock, per-triple TNO domains, a compiled kernel whose cost follows
the local triple list). The (T) vanishes identically for two-electron
systems.

```python
# doc-audit: skip - continues with mol and vq from the first example
result = vq.run_job(mol, basis="cc-pvdz", method="dlpno-ccsd(t)",
                    output="h2o_cct")
cc = result.dlpno_ccsd
if not result.converged or not cc.converged:
    raise RuntimeError("SCF or DLPNO-CCSD did not converge")
print(f"E_corr(CCSD) = {cc.e_corr:.8f}  (T) = {cc.e_t:.8f}  "
      f"total = {result.energy_total:.8f}")
# E_corr(CCSD) = -0.20802771  (T) = -0.00280334  total = -76.23514493
```

The `.out` names the triples algorithm ("rotated-occupied (T1)
compatibility correction") and reports 14 occupied triples with an average
triples-natural-orbital domain of 19 functions.

The (T) uses a spatial closed-shell kernel (no spin-orbital redundancy).
The O(N⁶) correctness pilot is still available by passing a
`DLPNOCCSDPilotOptions`, useful as an independent benchmark anchor.

## Is DLPNO-CCSD(T) accurate enough?

The exactness limit above answers the question in principle; production
runs use truncated domains. At the shipped defaults (chemical frozen core,
NormalPNO, cc-pVDZ) on 17 closed-shell molecules from water to C₁₀H₂₂, the
mean absolute error against canonical DF-CCSD(T) is 0.26 kcal/mol for CCSD
and 0.38 for CCSD(T). ORCA 6.1.1 with `NormalPNO` on the same inputs gives
0.28 and 0.87; its larger (T) error is the semicanonical (T0) it uses by
default, where vibe-qc's `(T1)` evaluates the canonical (T) on the PNO
amplitudes. With the C₁₀H₂₂ TightPNO run included, TightPNO brings vibe-qc
to 0.14 / 0.19 over the same 17 molecules. The per-molecule table,
the TightPNO comparison with ORCA and the open findings are in the
[user guide](../user_guide/dlpno_mp2.md#accuracy).

### Historical def2-SVP recipe

Before the chemical-core and NormalPNO defaults (#140, #448) the benchmark
ran all-electron def2-SVP with `tcut_pno=1e-7`, `tcut_pairs=1e-4`,
`tcut_mkn=0`, `residual_domain="pair"` and DLPNO-(T1) on seven small
molecules (MAE 0.37 kcal/mol; 0.17 with `tcut_pno=1e-8`, next to a reported
ORCA 6.1 value of 0.16). The checked-in
[`benchmark-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/benchmark-dlpno-ccsd-t.py)
reproduces that recipe explicitly:

```python
# doc-audit: skip - continues with mol and vq from the first example
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

LocalCCSDOptions(n_frozen=0, tcut_pno=1e-7, tcut_pairs=1e-4,
                 tcut_mkn=0.0, residual_domain="pair",
                 triples_mode="t1", compute_triples=True)
```

With `tcut_pno=1e-8` and `triples_mode="exact"` the same options give the
O(N⁷) exact-(T) oracle. Those numbers belong to that recipe, not to the
defaults measured above.

### Cost

Accuracy is only half the story; DLPNO exists because its cost should grow
far more slowly than canonical CCSD(T)'s O(N⁷). What is measured for this
implementation: the default closed-shell `(T1)` costs a canonical (T)
(C₁₀H₂₂/cc-pVDZ took 16 hours on 16 cores, against 11 minutes for the whole
run with `triples_mode="local"`), and the CCSD step keeps no dense
amplitude tensor, so canonical CCSD(T) becomes intractable past roughly
three to four waters in the reach walkthrough of
[`benchmark-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/benchmark-dlpno-ccsd-t.py)
while DLPNO-CCSD(T) carries on. Whether the CCSD step's cost grows
sub-quadratically with molecule size has not been measured; the C20/C40/C60
alkane ladder with a `coupling_radius=0` control is #190.

## When to reach for DLPNO

* **DLPNO-MP2**, whenever you would run RI-MP2 and the system is big enough
  that you care. It supports closed-shell RHF and open-shell UHF references;
  energies only (no gradients yet). Historical recovery percentages earlier
  on this page belong to their explicit all-electron threshold sweep, not the
  current default.
* **DLPNO-CCSD**, the local solver: coupled-cluster quality with PNO
  truncation control, full-domain == canonical CCSD.
  `coupling_radius` defaults to 12 bohr, a no-op on compact molecules,
  bounding the occupied coupling on extended systems (set 0 for the exact
  full-coupling reference). The default `residual_domain="pno"` contracts
  each residual term in PNO spaces in the compiled `dlpno_ccsd_pno_solve`
  kernel; `"extended"` (the atom-based extended domain, contracted once per
  distinct coupling-set and domain group) is kept as its exactness oracle.
  `tcut_pno` cuts pair-density occupation numbers, and which density is used
  is selectable with `pno_norm` (`"mp2"` by default, Riplinger and Neese's
  published Eq. 23; `"iepa"` for the pre-2013 LPNO convention; `"legacy"`
  for vibe-qc's historical density); see the user guide. The solver carries a generous, overridable `max_nbf`
  guard.
* **DLPNO-CCSD(T)**, the local solver followed by the triples, exact ==
  canonical CCSD(T) at full domains. Closed shell defaults to (T1), the
  canonical (T) on the rotated amplitudes, with the local (T0)
  (`triples_mode="local"`) as the option that scales; open shell defaults to
  its local (T0). The opt-in open-shell ``triples_mode="t1-iterative"`` correction is an
  experimental, capped dense spin-orbital generalization and is not itself a
  reduced-scaling triples phase.
  `DLPNOCCSDPilotOptions` / `DLPNOUCCSDPilotOptions` explicitly select the
  dense, capped correctness pilots for benchmark anchors.

Every `dlpno-*` job emits its method papers (Pinski 2015, Riplinger
2013, Foster-Boys 1960, …) into the `.references` / `.bibtex` outputs
automatically.

## Theory

The steep cost of canonical correlation methods comes from delocalised
orbitals: every occupied pair correlates through every virtual pair, and
the number of amplitudes grows as $o^2v^2$. Dynamic correlation is
short-sighted, though, so a pair of well-separated localised orbitals
contributes almost nothing, and each remaining pair needs only a small
virtual space adapted to that pair. The domain-based local pair-natural-orbital
scheme ({vibeqc-cite}`riplinger_dlpno_2013`; MP2 form
{vibeqc-cite}`pinski_dlpno_mp2_2015`) turns those two observations into a
controlled approximation in four steps.

### Localisation, domains and pair screening

The occupied orbitals are localised (Foster-Boys) to $\{i\}$. The virtual
space is spanned by projected atomic orbitals, atom-centred functions from
which the occupied space has been projected out,

$$
\tilde\chi_\mu = \Bigl(1 - \sum_i |i\rangle\langle i|\Bigr)\chi_\mu ,
$$

so a *domain* can be assigned to each localised orbital: the atoms whose
population in it exceeds `tcut_mkn`, together with their PAOs. A pair domain
is the union of the two orbital domains. Before any integral is built,
every pair is classified by a multipole estimate of its correlation energy,
which decays as $R_{ij}^{-6}$ with the distance between the two orbitals:
pairs below `tcut_pairs` are *distant* and keep only that estimate
(`e_distant`), and the rest are iterated. This step makes the number of
pairs grow linearly with the molecule.

### Pair natural orbitals and the thresholds

Within each pair domain the semicanonical MP2 amplitudes $T^{ij}$ define a
pair density, whose eigenvectors are the pair natural orbitals (PNOs),

$$
D^{ij} = \tilde T^{ij\dagger}\,T^{ij} + \tilde T^{ij}\,T^{ij\dagger},
\qquad
\tilde T^{ij} = \frac{4\,T^{ij} - 2\,T^{ij\,\mathsf T}}{1 + \delta_{ij}} .
$$

PNOs with occupation number above `tcut_pno` are kept, typically ten to
twenty per pair against hundreds of virtuals, and the energy discarded
with the rest is estimated at the semicanonical level and added back as
$\Delta E^\text{PNO}_{ij}$. The amplitudes are then iterated in the pair PNO
bases; because localised orbitals are not canonical, the occupied Fock
matrix couples different pairs through overlap matrices between their PNO
spaces. The whole approximation is three numbers, `tcut_pairs`, `tcut_pno`
and `tcut_mkn`, and the published Loose/Normal/Tight sets of
{vibeqc-cite}`liakos_dlpno_thresholds_2015` are named combinations of them
(`dlpno_thresholds="loose" | "normal" | "tight"`). With every threshold at
zero, no pair is screened, the PNOs span the full pair virtual space, and the
equations are the canonical ones in a rotated basis: the result equals
canonical RI-MP2 (or CCSD, or CCSD(T)) with the same fitting basis to the
microhartree, which is the identity the test suite pins and the last row of
the sweep above shows. The truncation error is smooth and systematic, not
variational: tightening a threshold moves the energy toward the canonical
value, but a DLPNO energy is not a bound.

### Coupled cluster and the triples

DLPNO-CCSD keeps the amplitudes in the PNO bases and contracts the CCSD
residual in the extended PAO domain before projecting back
(`residual_domain="extended"`), so no full-system amplitude tensor is formed.
The perturbative triples ({vibeqc-cite}`riplinger_triples_2013`) use one
triples-natural-orbital domain per occupied triple. The original (T0)
takes the occupied Fock matrix as diagonal in the localised basis; the (T1)
correction of {vibeqc-cite}`guo_dlpno_t1_2018` restores the off-diagonal
coupling and removes that semicanonical error, which is why
`dlpno-ccsd(t)` defaults to it. A truncated (T1) still has finite PNO and
TNO domains, so it approaches the canonical (T) only in the full-domain
limit. Open-shell molecules (`multiplicity > 1`) route automatically to the
spin-orbital solvers, which keep independent alpha and beta localised
orbitals and pair PNO spaces
({vibeqc-cite}`saitow_openshell_dlpno_2017` is the method-family source;
the local open-shell CCSD route adopts the same three Liakos coordinates but
not that paper's `TScaleTCutPairs` refinement, which assumes a common
spatial-orbital set).

### Where it helps and where it fails

Local correlation pays off when the molecule is large and gapped: cost
grows near-linearly, memory stays bounded, and the error is controlled by
the thresholds. It shares every limitation of the canonical method it
approximates: a single dominant determinant is required, energies only (no
gradients), and the error is the sum of the method error and a domain error
that grows for strongly delocalised systems, so relative energies are best
compared with identical thresholds and identical frozen-core conventions
on both sides. The dense $O(N^6)$ open-shell pilot solvers remain opt-in
correctness anchors, not production routes.

## See also

* [DLPNO methods user guide](../user_guide/dlpno_mp2.md), full
  threshold table, result-object reference, limitations.
* [MP2 and double hybrids](../user_guide/mp2_and_double_hybrids.md), 
  the canonical methods DLPNO converges to.
* Tutorial [38, RI-MP2 auxiliary-basis verification](ri_mp2_aux_verification.md)
  - the RI fitting error underneath everything here.
