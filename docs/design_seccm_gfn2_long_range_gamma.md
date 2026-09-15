# Design: self-consistent long-range gamma for GFN2-xTB-SECCM

**Status (issue #444):** non-molecular 3-D `ewald_gamma=True` fails closed
at the native route boundary, before supercell matrix assembly. Exact
zero-image molecular delegation and the 1-D/2-D paths are unchanged.
The internal 3-D shell-gamma seam remains available for diagnostic tests,
not production energies. Finite WS truncation does not cure the
thermodynamic-limit defect. No replacement GFN2 hardness mapping or
finite-distance energy change is introduced here.

## Literature and the long-range/short-range partition

[GFN2-xTB, Bannwarth, Ehlert and Grimme (2019)](https://doi.org/10.1021/acs.jctc.8b01176)
defines a shell-resolved charge interaction with Klopman-Ohno damping.
For shell hardnesses (h_a,h_b), the implemented pair kernel is

```text
eta_ab = 2 / (h_a + h_b)
gamma_ab(R) = 1 / sqrt(R^2 + eta_ab^2)
gamma_ab(R) - 1/R = -eta_ab^2 / (2 R^3) + O(R^-5).
```

Here `eta_ab` is a damping length, not the hardness itself. A consistent
point-Coulomb partition at a finite WS boundary can be written

```text
Gamma_ab = Ewald_Coulomb_ab
         + sum_WS w [gamma_ab(R_image) - 1/R_image]
         + on_site_ab.
E_charge = (1/2) dq^T Gamma dq.
```

Nonzero-distance records enter the bracket; the same-atom on-site block
is handled separately, with the Ewald self convention retained. The
one-half in the energy avoids pair double counting. This identity separates
point electrostatics from the damped remainder; it does not establish
convergence of the latter as the cluster grows.

The corresponding CCM embedding partition is the exact Ewald potential
minus the bare point-Coulomb contribution already owned by the WS cell.
[Janetzko, Bredow and Jug, JCP 116, 8994-9004 (2002)](https://doi.org/10.1063/1.1473802),
especially equations 19-20, supplies that long-range CCM construction.
[Bredow, Geudtner and Jug (2001)](https://doi.org/10.1002/1096-987X%2820010115%2922%3A1%3C89%3A%3AAID-JCC9%3E3.0.CO%3B2-7)
uses finite Evjen-weighted point-charge shells and defers exact Ewald
embedding on page 93. It is not the source of the later exact operator.
The citation database retains both works under their distinct roles;
`janetzko_ccm_long_range_2002` is the existing long-range entry.

## Why the 3-D WS remainder has no thermodynamic limit

The number of sites in a three-dimensional radial shell grows as
`R^2 dR`, so the leading `R^-3` remainder accumulates as `dR/R`.
For a repeated charge pattern its leading charge-weighted coefficient is
proportional to `sum_ab dq_a dq_b eta_ab^2`. Neutrality
`sum_a dq_a = 0` does not generally cancel a pair-dependent coefficient.
Shell-dependent hardness also means a single chemical element is not a
general escape from the problem.

For the neutral first-shell Mg4O4 patterns below the coefficient is
`16 [1/h_Mg^2 + 1/h_O^2 - 8/(h_Mg+h_O)^2]`, strictly positive for unequal
positive hardnesses. The regression checks this analytic non-cancellation
independently of the two finite-size energies; the latter are not used to
fit an asymptotic rate.

WS ownership makes every fixed-cluster matrix finite. Increasing the
cluster grows the owned remainder, however, and can accumulate the same
logarithmic defect. Neither Ewald-alpha independence, finite-cutoff parity
with another implementation, small charges, nor SCC convergence proves
the existence of the thermodynamic limit. The earlier version of this
design incorrectly called the WS convention convergent in 3-D; that claim
is withdrawn.

In one and two dimensions the corresponding far-field absolute remainder
scales as `integral R^(d-4) dR` and converges. This dimensional argument
does not certify every other GFN2 channel or solve an SCC instability.
The shipped 1-D wire and 2-D Parry/Heyes paths, including the exact pairwise
2-D K=0 term, remain unchanged. The separate SCC-DFTB contracts tracked by
#211/#425 are not modified by this GFN2 containment.

An exponential remainder such as the density-derived SCC-DFTB gamma in
[Elstner et al. (1998), equations 17-18](https://doi.org/10.1103/PhysRevB.58.7260)
has a convergent tail, but that fact does not define a GFN2 unequal-shell
parameter mapping. The staged method decision is not an implemented or
validated GFN2 replacement. This milestone neither adds nor advertises
an Elstner default.

## Fixed-charge cluster discriminator

The native diagnostic seam
`_gfn2_seccm_ewald_shell_gamma_from_records` is tested without an SCC solve.
Both probes use a cubic side of 4.212 angstrom, the site order

```text
(0,0,0), (.5,.5,0), (.5,0,.5), (0,.5,.5),
(.5,.5,.5), (0,0,.5), (0,.5,0), (.5,0,0).
```

The historical synthetic Mg4O4 probe assigns
`[Mg,O,O,O,Mg,Mg,Mg,O]`. It is **not B1 rocksalt**: sites 0 and 5
are nearest neighbors of the same species. The physical B1 control assigns
`[Mg,Mg,Mg,Mg,O,O,O,O]` to the same sites. Each conventional cell is
replicated `n x n x n`. Only each atom's first enumerated shell carries
`dq = -1` for Mg or `+1` for O; every other shell carries zero.
This is a prescribed ionic diagnostic, not self-consistent occupations.

| Probe | n=1, 8 atoms | n=2, 64 atoms | Difference |
| --- | ---: | ---: | ---: |
| Synthetic cubic Mg4O4 | +0.361531884674 | +0.464251887966 | +0.102720003292 |
| B1 rocksalt control | +0.144635757778 | +0.255217985354 | +0.110582227576 |

Values are `0.5 dq^T Gamma dq / n^3` in Ha per conventional cell, with
`alpha = sqrt(pi) / volume^(1/3)`. These are total fixed-charge kernel
energies, not the remainder alone. Independent contraction of WS records
isolates the size step entirely to `KO-1/R`; the on-site and point-Coulomb
contributions per cell are invariant. Two points discriminate cluster drift
but are not a fitted asymptotic logarithmic rate or a prediction of its sign
at all sizes. No n=3 value or extrapolated limit is claimed.

## Production boundary and regression scope

The public Python and raw native calculation entries reject non-molecular
3-D `ewald_gamma`, including a group-order-one topology with nonzero image
records. The guard follows option/restart validation and exact molecular
delegation, and precedes matrix assembly. Existing validation precedence,
#410 failed-state classification, #421 diagnostic-only symmetry reporting,
and molecular Newton refusal are preserved. `include_aes=False` does not
create a bypass: it selects the cyclic engine even for zero-image records.

The internal kernel still pins finite-cluster bookkeeping, the eta-to-zero
point-Coulomb anchor, and skew-cell enumeration. Those tests do not authorize
production 3-D use. Named #130 Cu recipes now assert the explicit rejection;
their old energy, size and state assertions remain historical evidence,
not reachable production expectations. Earlier MgO/corundum bulk convergence
and lattice claims likewise do not establish thermodynamic validity.
Molecular, 1-D/2-D and non-`ewald_gamma` calculations are not reparameterized.
