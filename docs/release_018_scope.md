# 0.18.0 candidate scope: Berkelbach's Badger

This candidate freezes periodic correctness and the existing bounded MANTA
controls. Candidate preparation is not a release announcement: the exact
candidate still requires its release tests, documentation and publication gates.
The scientific source is `46d08d59737d6c4efd68a2f3285e750be99aaf88`;
subsequent candidate changes repair binding topology/converter headers, test
fixtures and the documented MANTA example work allowance. The allowance changes
no physical fixture or scientific tolerance. Each candidate retains its own
source, native and CI identities.

## Included numerical scope

| Route | Accepted configuration and comparison | Boundary |
| --- | --- | --- |
| Periodic GPW PBE | The fixed H2 directional derivative controls at cell lengths 9 and 16 bohr, cutoff 25, maximum 40 SCF iterations and finite-difference step 1e-5 give errors 4.90266e-11 and 4.50180e-11, below the original 2e-9 tolerance. | Selected energy/Fock consistency controls; no bulk exchange accuracy claim. |
| Periodic RIJCOSX | Dense-cell inputs are refused; the molecular and vacuum controls remain usable. | Dense periodic RIJCOSX is unsupported. |
| Periodic ECP | Existing single-point route and refusal controls are retained. The source-matched Cl2 Ewald UKS derivative preview at Gamma and a 3x1x1 mesh gives maximum bond finite-difference error 7.76e-9 Ha/bohr against 2e-6, and molecular-limit offset 6.026e-6 Ha against 1e-5. | This preview supersedes the earlier blanket ECP gradient refusal only for admitted source-matched cases. It is not general ECP gradient, stress or all-component physical-support qualification; unsupported-route guards remain. |
| Gamma GDF PBE0 | The original diamond/def2-SVP control gives energy error 9.911666154494014e-8 Ha and T2 splitting 3.3373773227207693e-6, both below 1e-5; the diffuse-cell refusal control also passes. | No all-cell or all-basis convergence extrapolation. |
| MANTA | The four frozen profiles described below exercise image, full-Coulomb exchange, nuclear and finite-source RI coupling admission. | Separate component certificates, not a bound on a relaxed total energy or a material prediction. |

The original settings, supports and basis coefficients remain in the selected
fixtures. Passing a process or a release test does not replace convergence,
reference matching or the route's numerical criteria.

## Frozen MANTA examples

Use the existing [control guide](user_guide/manta_release_controls.md) and
`benchmarks/manta_release_controls.py` driver. The driver serializes
the exact basis, geometry, mesh shift, physical supports and resource caps;
this release does not substitute different automatic selector settings.

- `image-tight`: shifted two-point all-electron Be RHF, RI-J and full-Coulomb
  exchange, fixed occupations and an unreduced mesh; pair radius 10 bohr,
  auxiliary radius 12.1 bohr, and image targets 1e-6.
- `ordinary-rhf`: the same family with joint controls. Image/nuclear targets
  1e6 and exchange targets 1e12 are deliberately loose admission fixtures,
  not chemical accuracy recommendations. Finite-source RI coupling targets
  are 1e-6 for J and Coulomb energy.
- `global-pbe0`: shifted two-point Be PBE0 with the fixed shared continuous
  periodic quadrature. Current energy/Fock/density errors are 5.33e-15 Ha,
  9.44e-16 and 2.04e-10, against 8e-11, 8e-11 and 5e-9. The accepted control
  energy is -9.47932169663847 Ha; the older grid's energy is not reused.
- `compact-tight`: Gamma He RHF with the custom compact basis, exact finite
  coupling, pair radius 1 bohr, nuclear radius 18 bohr and reciprocal cutoff
  512 Eh. Image, exchange and nuclear component targets are 1e-8.

The driver retains four-GiB memory and 1e12 shell/primitive limits. Its scalar
and kernel allowances are 7e18 for image-tight and 2e18 for the other profiles,
matching the accepted corrected-candidate integration. These limits are
admission allowances, not measured cost or speedup claims. Inspect every
requested certificate and its reference object. Do not add unlike image,
exchange, nuclear and finite-fitting certificates and call the sum a relaxed
SCF error. Tiny analytic tails do not certify floating-point arithmetic.

## Explicit exclusions

Spectroscopy and oscillator-strength claims are excluded from this release
scope. Existing molecular response acceptance is retained independently and
does not promote general DFT response or spectra.


The matched Gamma/2x2x2 LiH range-separated GDF controls are restricted to
the demonstrated configuration: fcc lattice parameter 7.7158 bohr, STO-3G,
69-function def2-SVP-JK auxiliary basis, short-range fitting, explicit metric
cutoff 1e-13, and SCF energy/gradient thresholds 1e-11/1e-8. Both meshes
converged against matched PySCF 2.6.2 references: absolute energy differences
1.31490e-9 and 4.22104e-9 Ha pass the unchanged 1e-7 Ha gate. The original
Gamma result and recovered 2x2x2 calculation retain their separate job identities;
the earlier timeout is not a numerical failure or a passing result. This is
matched fitted parity only, not a universal cutoff, dense-cell policy or
unfitted-accuracy guarantee. Candidate containment is recorded separately.
The general BIPOLE cancellation and 100-bohr Ne accuracy claims are excluded.

The MANTA scope does not establish general overlap-rank stability, density
relaxation, finite fitting error where not separately covered, basis or
Brillouin-zone convergence, zero-mode/Madelung/XC convergence, external bulk
parity, production correlation, or speedup. Low-dimensional or symmetry-reduced
variants, general gradients/stress, and screened or RI exchange certificates
are outside these four frozen profiles. Additional experimental capabilities
remain separately documented and are not promoted by inclusion in this source.
Native correlation remains experimental for production-size use.

The containing MACE runtime and final-force integration acceptance remain a
separate requirement for MACE optimization claims. Native PBE acceptance alone
does not establish MACE availability or acceptance.

## Qualification identities

The source contains the continuous-grid integration and the source-matched ECP
preview, rather than relabelling the older 0.17.26 release. Existing numerical
receipts retain their original source and native identities. An eligible
containing native build can support their integration checks, but cannot be
renamed as the final candidate's full release-test receipt. Required candidate
checks and anonymous public-source verification must pass before tagging and
publication are declared complete.
