# IAO charges, spin populations and IAO-Wiberg bonds

Request an intrinsic atomic orbital (IAO) analysis of a converged molecular
RHF/RKS or UHF/UKS determinant through the normal Python input:

```python
import vibeqc as vq

mol = vq.Molecule(
    [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])],
    charge=0, multiplicity=1,
)  # coordinates in bohr
result = vq.run_job(
    mol, basis="def2-svp", method="rhf", output="h2",
    iao_analysis=True, iao_bond_threshold=0.05,
    localize=False, output_qvf=False,
)
a = result.iao_analysis
if a.available:
    print(a.charges)          # elementary charge units, input atom order
    print(a.populations)      # electrons, including all occupied core electrons
    print(a.bond_orders)      # dense, symmetric, dimensionless; zero diagonal
    print(a.spin_populations) # alpha minus beta electrons; None for RHF/RKS
else:
    print(a.unavailable_reason)
```

The [numerical comparison](iao_validation.md) reports computed charges beside
Knizia's published values, distinguishes MINI from MINAO, and supplies a
runnable input with explicit geometries and convergence settings.

Analysis is opt-in (`iao_analysis=False` by default). It needs neither orbital
localization nor dipole integrals nor a QVF archive. Localization remains a
separate option. When both are requested for a restricted state, the runner
shares the IAO construction across analysis and localization criteria.
The SCF result's existing attributes remain accessible; opting in adds the
`.iao_analysis` result through a transparent result wrapper.

The requested IAO section appears in `.out` and, when
`write_population_file` is enabled, in `.population.txt` and
`.population.json`. JSON adds an `iao` object with the complete dense matrix,
populations, charges, optional spins, conventions, reference basis,
orthogonalization and residuals. Atom indices are zero-based in input order.
With `structured_log=True`, the properties event also includes the `iao`
payload. `iao_bond_threshold` affects only displayed text pairs. A successful analysis
with no displayed pairs is distinguished from an unavailable analysis.

QVF retains the existing `atom_properties/iao_charge.bin` member. Its
`x_vibeqc.iao_analysis` vendor section contains `analysis/iao.json` with the
complete analysis, including spin populations, dense bond orders, diagnostics
or the unavailable reason. This uses the existing QVF extension mechanism;
the canonical atom-property names and companion schemas are unchanged.
Existing viewers can display the IAO charge property. The new JSON is
available to consumers without asserting that every viewer displays it.

An unavailable population request does not disable separately supported
localization. For example, an MP2 job can still export localized SCF-reference
orbitals and their legacy `iao_charge` property. The new `iao` analysis
payload remains unavailable for that MP2 job; those legacy reference charges
must not be interpreted as correlated-density populations.

## Open-shell use and reading results

For an OH doublet, set the molecular multiplicity and select UHF or UKS:

```python
oh = vq.Molecule(
    [vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.83])],
    charge=0, multiplicity=2,
)
result = vq.run_job(
    oh, basis="def2-svp", method="uhf", output="oh",
    name_molecule=False, iao_analysis=True,
    localize=False, output_qvf=False,
)
a = result.iao_analysis
if not a.available:
    raise RuntimeError(a.unavailable_reason)
print(a.spin_populations.sum())  # approximately 1 alpha-minus-beta electron
print(a.bond_orders[0, 1])       # O-H IAO-Wiberg index, dimensionless
```

For UKS, use `method="uks", functional="pbe"`. Spin populations can be
negative on an individual atom because they measure spin polarization. Their
sum is `multiplicity - 1`; they are not probabilities or magnetic moments.
For RHF/RKS, `spin_populations` is `None`, indicating the shared restricted
partition. `bond_orders[i, j]` always uses zero-based input atom indices.

Read a population sidecar without loading a wavefunction:

```python
import json
from pathlib import Path

payload = json.loads(Path("oh.population.json").read_text())["iao"]
if payload["available"]:
    print(payload["charges"], payload["bond_orders"][0][1])
else:
    print(payload["unavailable_reason"])
```

| Option or payload | Meaning |
|---|---|
| `iao_analysis=True` | Request charges, spin populations and bond orders together |
| `iao_bond_threshold=0.05` | Minimum bond index shown in text; JSON/API retain every pair |
| `write_population_file=False` | Suppress population sidecars; retain `.out`, the API result and any requested structured-log/QVF payload |
| `available=False` | Analysis did not produce numerical populations; inspect `unavailable_reason` |
| `unavailable_reason=None` | Analysis succeeded; absent displayed bonds can simply reflect the threshold |
| `diagnostics` | Count, factorization, metric, occupied-span and idempotency residuals with the applied tolerances |

## Definition and normalization

The reference is **Huzinaga MINI**, preserving the existing molecular IAO
default. We use the projector construction and symmetric metric
orthogonalization of [Knizia (2013), Appendix C and eq 3](https://doi.org/10.1021/ct400687b).
IAO construction does not require IBO localization. MINI is different from
the MINAO initial-guess reference and from PySCF's default `minao` basis.
Laikov's [zero-bond-dipole orthogonalization](https://doi.org/10.1002/qua.22767)
is an alternative partition and is not implemented by this option.

With AO overlap $S$, occupied MO columns $C_\sigma$ and IAO columns
$A_\sigma$ satisfying $A_\sigma^\dagger S A_\sigma=I$, define

$$
D_\sigma=A_\sigma^\dagger S P_\sigma S A_\sigma
=X_\sigma X_\sigma^\dagger,\qquad
X_\sigma=A_\sigma^\dagger S C_\sigma.
$$

The native determinant convention is $P_\sigma=C_\sigma C_\sigma^\dagger$.
Restricted total densities have occupation two; analysis shares the IAOs
and splits the density equally into alpha and beta. UHF/UKS builds IAOs
separately from each spin's occupied space, with identical MINI reference
function labels. An empty spin channel contributes zero electrons and zero
bonds. The union of the occupied spin spaces is not used as a reference.

For reference functions $I_a$ labeled by atom $a$:

$$
N_a=\sum_\sigma\sum_{\mu\in I_a}D_{\sigma,\mu\mu},\qquad
q_a=Z_a-N_a,\qquad
s_a=\sum_{\mu\in I_a}(D_{\alpha,\mu\mu}-D_{\beta,\mu\mu}),
$$

$$
B_{ab}=2\sum_\sigma\sum_{\mu\in I_a}\sum_{\nu\in I_b}
|D_{\sigma,\mu\nu}|^2\quad(a\ne b),\qquad B_{aa}=0.
$$

This **spin-resolved IAO-Wiberg** convention reduces to the squared
spin-summed IAO density only in the restricted closed-shell limit. A doubly
occupied equal-weight bonding orbital gives $B=1$; a singly occupied alpha
bonding orbital gives $B=0.5$. Bonding plus antibonding double occupation
gives zero. A polarized double occupation with weights $p,1-p$ gives
$B=4p(1-p)$. Real molecular indices need not be integers.

For a determinant, Wick contraction of the atom number operators gives
$\operatorname{Cov}(N_a,N_b)=-\sum_\sigma\|D_\sigma[I_a,I_b]\|_F^2$
for distinct atoms. Thus this convention is minus twice that covariance.
The spin-dependent IAO atom projectors define those number operators in
unrestricted calculations. The relationship uses determinant factorization;
a general correlated delocalization index requires a two-particle density.
See [Wiberg (1968)](https://doi.org/10.1016/0040-4020(68)88057-3) and
[de Giambiagi, Giambiagi and Jorge (1985)](https://doi.org/10.1007/BF00529054).
A user-derived atomic bond-order sum is simply $\sum_{b\ne a}B_{ab}$,
not an oxidation state or an enforced integer valence. Pair indices also do
not fully describe [multicenter bonding](https://doi.org/10.1039/C7CP07422K).

Existing `vibeqc.bond_analysis.wiberg_bond_orders` still uses the
Loewdin-orthogonalized **original AO basis** and its existing spin-summed
convention. Existing Mayer and AO-Wiberg output fields keep their identities.

## Direct API and support checks

```python
from vibeqc.iao_population import analyse_iao

# scf is the result from run_rhf/run_rks/run_uhf/run_uks;
# basis_obj is the exact BasisSet supplied to that calculation.
a = analyse_iao(scf, basis_obj, mol)
payload = a.to_dict()
```

`analyse_iao_occupied` is the matrix-level API: supply unit-occupied,
S-orthonormal AO coefficient columns, overlap matrices and a complete
reference-function-to-atom map. Omit `occupied_beta` for a restricted state;
pass an explicit `(n_ao, 0)` array for an empty beta channel. This API raises
on invalid input. `analyse_iao` converts numerical/support failures into a
typed unavailable result with `None` arrays, never physical zero charges.
The exact original `BasisSet` and its AO row order are the adapter contract;
coefficients are never transposed heuristically.

The adapter checks density/MO agreement, integer occupations, molecular
charge/multiplicity, finite values, Hermiticity, atom labels, occupied metric,
IAO orthonormality, occupied-span preservation, spin idempotency and electron
conservation. The validation tolerance is $10^{-7}$ (absolute for counts and
metric residuals, relative for AO density factorization). Explicit occupation
vectors must have zero/full values to within $10^{-10}$; fractional occupations
are rejected. Hermitian residuals are checked before removing numerical
imaginary diagonal/trace components; off-diagonal complex phases are retained.

Metric inversion uses a spectral floor of
$10^{-10}\max(1,\lambda_{\max})$. Loss of any labeled reference direction
is unavailable, including a rank-deficient projected reference. No
pseudoinverse or charge renormalization conceals missing populations.
Ordinary diffuse bases can succeed when these checks pass; near-singular
spaces are explicitly refused. Diagnostics record the actual residuals.

ECPs, ghosts, elements outside MINI's coverage (Z > 86), fractional
occupations, correlated densities, relativistic spinors and multicenter bond
indices are outside this implementation. All occupied core electrons must
be included; a frozen-core correlation subspace cannot yield total charges.
Reference-library coverage alone is not validation of every element's
chemical results. Focused chemistry tests cover H2, H2O, CH4, HCN, benzene
and OH, including an optional PySCF construction with matched MINI overlaps
and symmetric orthogonalization. The existing Knizia charge benchmarks retain
their geometry/reference tolerances. The [Senjean et al. (2021) SI](https://doi.org/10.1021/acs.jctc.0c00964)
clarifies the occupied-span conditions and broader spinor generalization;
that broader capability is not implied here.

Successful requested analysis adds the existing Knizia, Huzinaga and Wiberg
citation routes independently of localization. An unavailable analysis does
not claim that these algorithms ran successfully.

Common unavailable reasons have different remedies:

| Reason | Interpretation and next step |
|---|---|
| SCF not converged | Converge the underlying determinant before interpreting populations |
| Fractional occupations or density/MO disagreement | Supply the integer determinant and its matching native basis; a correlated density is outside scope |
| Incomplete alpha/beta pair | Supply both density and coefficient channels, including an explicit empty channel if needed |
| Rank-deficient metric or failed occupied span | The requested labeled reference is not well defined at the stated floor; investigate the orbital/reference basis and linear dependence |
| ECP, ghost, unsupported element or periodic input | This implementation does not support that case; tighter SCF tolerances do not enable it |

## Periodic integration plan

`run_periodic_job(..., iao_analysis=True)` raises an explicit unsupported
error before calculation, including Gamma-only requests. The molecular
adapter also rejects explicitly periodic input. This gate remains until the
following contracts are implemented and tested:

1. Reuse `PeriodicGaussianBlochIAOPoint` and
   `make_periodic_gaussian_bloch_iao` in
   `cpp/src/periodic_gaussian_bloch_iao.cpp` for physical $S_{12}(k)$,
   $S_{22}(k)$ and atom labels. Its native algebra in
   `cpp/src/periodic_correlation_bloch_iao.cpp` retains all occupied bands,
   including core bands, and validates rank and retained-space reconstruction.
   The current owner uses `PeriodicRestrictedMeanFieldState`; unrestricted
   spin handling needs an explicit extension. Native IAOs are **not yet
   symmetrically orthonormalized**. Their stored occupied coefficient array
   named `D` is not itself an electron density.
2. Choose and record a periodic orthogonalization/gauge convention. The
   molecular symmetric construction cannot silently replace the native
   nonorthogonal convention. [Zhu and Tew (2024)](https://doi.org/10.1021/acs.jpca.4c04555)
   explicitly discuss gauge complications from orthogonalizing Bloch IAOs;
   [Woeckinger, Rumpf and Schaefer (2025)](https://doi.org/10.1021/acs.jctc.5c00130)
   and its SI provide additional solids/localization convergence cases.
3. For an agreed orthonormal basis, construct $D_\sigma(k)$ using the actual
   occupations and nonnegative normalized weights $\sum_k w_k=1$. Validate
   weights, electron counts per primitive cell, spin counts, reference labels
   at every k point, and insulating integer occupations for the initial scope.
   Cell populations are $N_a=\sum_{k\sigma}w_k\operatorname{Tr}_{I_a}D_\sigma(k)$.
4. Define $D_\sigma(R)=\sum_k w_k e^{-ik\cdot R}D_\sigma(k)$ in the recorded
   atom/cell phase convention, then
   $B_{ab}(R)=2\sum_\sigma\|D_\sigma(R)[I_a,I_b]\|_F^2$.
   Exclude only the same atom in the same cell, specify finite-mesh translation
   aliases, and retain $B_{ab}(R)=B_{ba}(-R)$. A reported image sum must list
   its translation support. Squaring a naively k-averaged density would keep
   only one translation contribution and is not this periodic bond index.
5. Before enabling input, extend `test_periodic_correlation_bloch_iao.py` and
   `test_periodic_gaussian_bloch_iao.py` with density tests: orbital gauge
   invariance, reciprocal/atom translations, k permutation/weight equivalence,
   mesh convergence, primitive-cell versus supercell agreement, image-sum
   convergence and isolated-molecule limits. Compare with independent periodic
   overlaps/construction using matched reference and phase conventions.
