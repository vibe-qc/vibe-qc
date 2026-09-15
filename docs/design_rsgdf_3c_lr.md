---
orphan: true
---

# Design, RSGDF 3-centre long-range half (`rsgdf_lr_3c_tensor`)

**Status (2026-05-10)**: design only. Implementation is the next
slice in `docs/design_native_gdf.md` slice 3d.

**Why this matters**: the existing `rsgdf_lr_2c_metric` works (per
calibration ratio → 1.0 at high L). But the 2c LR alone isn't useful
for SCF, it produces an `M = M_SR + M_LR` periodic Coulomb metric
in the RSGDF gauge, while the 3c tensor `T = compute_3c_eri_lattice`
is in the bare-truncated gauge. **Mixing conventions gives wrong
Lpq**: an H₂/cc-pvdz/cc-pvdz-rifit smoke-test reconstructs the ERI
to 65 % error vs exact when only 2c uses RSGDF (vs 0.16 % error
with all-bare). Both M and T must be in the same convention.

This document specifies the 3c LR implementation that closes the
gap. Once it lands, `build_lpq_native(algorithm='rsgdf')` becomes
end-to-end usable (and correct), and the de-PySCF chain (slices
3d-7) can proceed.

## Math

The 3c LR contribution at Γ:

$$
T^{LR}_{P,\mu\nu}(\omega)
  \;=\; \sum_{\mathbf T} \big(P_{\mathbf 0}\,\big|\,\frac{\mathrm{erf}(\omega r)}{r}\,\big|\,\mu_{\mathbf 0}\,\nu_{\mathbf T}\big)
  \;=\; \frac{4\pi}{\Omega} \sum_{\mathbf G \neq \mathbf 0}
        \hat\xi_P^*(\mathbf G)\, \hat\rho_{\mu\nu}(\mathbf G)\,
        \frac{e^{-|\mathbf G|^2 / (4\omega^2)}}{|\mathbf G|^2}
$$

where:

- $\hat\xi_P(\mathbf G)$ is the analytic FT of aux primitive $P$
  (already implemented as `rsgdf_aux_fourier_transform`).
- $\hat\rho_{\mu\nu}(\mathbf G)$ is the **molecular** FT of the AO
  pair density $\chi_\mu(\mathbf r)\,\chi_\nu(\mathbf r)$ (no
  lattice sum, the Bloch-sum acts as a Dirac comb in $\mathbf G$
  space and selects only the reciprocal-lattice $\mathbf G$).

The $\mathbf G = \mathbf 0$ exclusion is the same charge-neutrality
convention as the 2c LR (cf. `rsgdf_lr_2c_metric` docstring).

## AO-pair Fourier transform

For two contracted shells $\mu$ at centre $\mathbf A$ with
exponents $\{\alpha_\mu^p\}$ and contraction $\{c_\mu^p\}$, and
$\nu$ at centre $\mathbf B$ analogous, the pair density expands
via the Gaussian-product theorem:

$$
\chi_{\mu,p}(\mathbf r)\,\chi_{\nu,q}(\mathbf r)
  \;=\; \mathcal{N}_{pq}\,\mathrm{P}_{l_\mu,m_\mu;l_\nu,m_\nu}(\mathbf r-\mathbf P)\,
        e^{-\gamma_{pq}\,|\mathbf r-\mathbf P|^2}
$$

with

- $\gamma_{pq} = \alpha_\mu^p + \alpha_\nu^q$
- $\mathbf P_{pq} = (\alpha_\mu^p \mathbf A + \alpha_\nu^q \mathbf B) / \gamma_{pq}$
- $\mathcal{N}_{pq} = c_\mu^p c_\nu^q \,
  e^{-\alpha_\mu^p \alpha_\nu^q / \gamma_{pq}\,|\mathbf A - \mathbf B|^2}$
- $\mathrm{P}_{l_\mu,m_\mu;l_\nu,m_\nu}(\mathbf r-\mathbf P)$ is the
  polynomial that arises from the product of the two angular factors
  $r_{\mu A}^{l_\mu} Y_{l_\mu m_\mu}(\hat r_{\mu A})$ and
  $r_{\nu B}^{l_\nu} Y_{l_\nu m_\nu}(\hat r_{\nu B})$, expanded
  about $\mathbf P$.

For the $s \times s$ case ($l_\mu = l_\nu = 0$), the polynomial
is a constant ($P = 1$ in libint's "no-Y₀₀" convention), so:

$$
\hat\rho^{ss}_{\mu\nu}(\mathbf G)
  \;=\; \mathcal{N}_{pq}\,\big(\frac{\pi}{\gamma_{pq}}\big)^{3/2}\,
        e^{-|\mathbf G|^2 / (4\gamma_{pq})}\,e^{-i \mathbf G \cdot \mathbf P_{pq}}
$$

For higher $L$, the polynomial $\mathrm{P}$ is expanded in
**Hermite Gaussians** via the McMurchie-Davidson scheme:

$$
\mathrm{P}_{l_\mu,m_\mu;l_\nu,m_\nu}(\mathbf r-\mathbf P)\,e^{-\gamma|\mathbf r-\mathbf P|^2}
  \;=\; \sum_{t,u,v} E^{l_\mu m_\mu, l_\nu m_\nu}_{tuv}\,
        \Lambda_{tuv}(\mathbf r-\mathbf P; \gamma)
$$

where $\Lambda_{tuv} = \partial_t^x \partial_u^y \partial_v^z e^{-\gamma|\mathbf r-\mathbf P|^2}$
is a Hermite Gaussian. Its FT is closed form:

$$
\hat\Lambda_{tuv}(\mathbf G; \gamma) \;=\;
  (i G_x)^t (i G_y)^u (i G_z)^v \,(\pi/\gamma)^{3/2}\,e^{-|\mathbf G|^2 / (4\gamma)}
$$

So the full AO-pair FT is:

$$
\hat\rho_{\mu\nu}(\mathbf G) \;=\; \sum_{p,q} \mathcal{N}_{pq}\,
  e^{-i\mathbf G \cdot \mathbf P_{pq}}\,
  (\pi/\gamma_{pq})^{3/2}\,e^{-|\mathbf G|^2 / (4\gamma_{pq})}\,
  \sum_{tuv} E^{(\mu\nu)}_{tuv}\,(iG_x)^t (iG_y)^u (iG_z)^v
$$

The $E$ coefficients are computed via the McMurchie-Davidson
recursion (linear in the angular momenta):

$$
E^{ij}_t \;=\; \frac{1}{2\gamma}\,E^{i-1,j}_{t-1}
            + (P_x - A_x)\,E^{i-1,j}_t
            + (t+1)\,E^{i-1,j}_{t+1}
$$

with seed $E^{00}_0 = e^{-\alpha\beta/\gamma\,(A_x-B_x)^2}$ (and 0
for $t \neq 0$). Same recursion for $j$. The 3D version is a tensor
product of three 1D recursions.

The libint convention is **real spherical harmonics** (with the
exception that $l = 0$ omits $Y_{00}$, see
`_real_sph_harm` notes in `aux_basis.py`). The McMurchie-Davidson
scheme is naturally Cartesian; we'd convert to spherical via the
standard Cartesian-to-spherical transform matrices (already
implicit in libint's basis-function ordering).

## Function API

```python
# python/vibeqc/aux_basis.py

def rsgdf_lr_3c_tensor(
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    system: PeriodicSystem,
    omega: float,
    *,
    precision: float = 1e-8,
    g_mesh: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Long-range half of the periodic 3-centre ERI tensor,
    T^LR_{P, μν}(ω).

    See docs/design_rsgdf_3c_lr.md for the math derivation.

    Returns
    -------
    T_lr : np.ndarray, shape (n_aux, n_orb, n_orb), real
    """
    # Implementation outline:
    # 1. Build / accept G-mesh (shared with rsgdf_lr_2c_metric is fine).
    # 2. ξ̂_P(G): rsgdf_aux_fourier_transform(aux_basis, G_vectors)
    #    — already implemented; returns (n_aux, n_G) complex.
    # 3. ρ̂_μν(G): _ao_pair_fourier_transform(ao_basis, G_vectors)
    #    — NEW; returns (n_orb, n_orb, n_G) complex.
    # 4. T^LR_{P, μν} = (4π/Ω) Σ_{G≠0} ξ̂_P*(G) · ρ̂_μν(G) ·
    #                   exp(-G²/4ω²) / G²
    #    Tensor contraction via einsum or direct (n_aux, n_G) · (n_G, n_orb²) matmul.
```

### `_ao_pair_fourier_transform` (new, the core new code)

```python
def _ao_pair_fourier_transform(
    ao_basis: BasisSet, G_vectors: np.ndarray,
) -> np.ndarray:
    """FT of all AO-pair densities χ_μ(r) χ_ν(r) at given G-vectors.

    Returns (n_orb, n_orb, n_G) complex.

    Implementation:
    - Walk shell pairs (sM, sN).
    - For each (sM, sN): expand the contracted pair density using
      Gaussian-product theorem + McMurchie-Davidson Hermite
      coefficients. FT each Hermite term in closed form.
    - Sum over primitive pairs (p, q) with their contraction
      coefficients.
    - Apply Cartesian → spherical transform per shell pair to match
      libint's spherical AO convention.
    - Symmetrise (μ, ν) — analytic property at Γ.
    """
    pass
```

This is where the McMurchie-Davidson / Cartesian-to-spherical
machinery lives. Estimate: ~200-400 LOC of careful Python.

## Validation strategy

1. **s × s only first** (l = 0 only on AO side; aux can be any L).
   Closed-form FT for s shells is one line; no MD recursion
   needed. Use H₂ + cc-pvdz-rifit, validate by comparing
   `M_SR + M_LR` Lpq to bare Lpq in the molecular limit
   (cutoff → 0). Both should reconstruct exact ERI to similar
   accuracy.
2. **s × p** then **p × p**: test MD recursion correctness
   against direct numerical FFT of the AO pair density.
3. **General L**: extend recursion code; validate against PySCF's
   `pyscf.pbc.df.ft_ao.ft_aopair_kpts` element-wise on a small
   system.
4. **Periodic SCF parity**: the real test. Build full Lpq via
   `algorithm='rsgdf'` on MgO/sto-3g, run SCF via
   `run_rhf_periodic_gamma_gdf` (with native Lpq override), compare
   energy to PySCF.GDF reference. Target: sub-µHa.

## Estimate

| step              | LOC  | wall (h) |
|-------------------|-----:|---------:|
| s × s             |  ~40 |    1     |
| MD E coefficients |  ~80 |    2     |
| Cartesian-to-spherical transform | ~60 | 2 |
| `_ao_pair_fourier_transform` glue | ~60 | 1.5 |
| `rsgdf_lr_3c_tensor` wrapper      | ~30 | 0.5 |
| Validation tests                  | ~100 | 2-3 |
| **TOTAL**         | ~370 | 9-11     |

So a focused day-and-a-half of work, plus the laptop CPU budget for
validation runs. No compute-reference CPU needed during development; compute-reference
only for the final SCF parity validation against PySCF.

## Alternative approaches considered

- **Use PySCF's `ft_aopair_kpts`**: works, but defeats the purpose
  of the de-PySCF milestone. Could be used in development as an
  oracle, then dropped before merging slice 3d.
- **Use libint's `Operator::erf_coulomb`** for the LR molecular
  3c, then sum over images: doesn't work, `erf(ωr)/r` doesn't
  decay at long range, so the lattice sum diverges. The whole
  point of moving LR to reciprocal space is to avoid this.
- **Bypass 3c LR entirely** by using a different 2c convention
  that matches bare 3c: this is what `algorithm='bare'` does
  today. Works for tight aux but caps the diffuse-aux usability.

## Decision

Implement option (1), pure-Python AO-pair FT via
McMurchie-Davidson, incrementally (s×s → general L) when a
focused work block is available. Until then, keep the LR machinery
documented as 2c-only, and use `algorithm='bare'` for the
production-shippable native-Lpq path (slice 3-minimal).

## References

- McMurchie, L. E. & Davidson, E. R. (1978). *J. Comput. Phys.*
  **26**, 218. DOI [10.1016/0021-9991(78)90092-X](https://doi.org/10.1016/0021-9991(78)90092-X).
  Original Hermite-Gaussian recursion paper.
- Helgaker, T., Jørgensen, P. & Olsen, J., *Molecular
  Electronic-Structure Theory* (Wiley, 2000), §9.3-9.5 for the
  modern Obara-Saika / McMurchie-Davidson treatment.
- Sun, Q. et al. (2017). *J. Chem. Phys.* **147**, 164119.
  DOI [10.1063/1.4998644](https://doi.org/10.1063/1.4998644).
  PySCF's GDF reference (also describes the AO-pair compensating
  charges and reciprocal-space recovery).
- Ye, H.-Z. & Berkelbach, T. C. (2021). *J. Chem. Phys.* **154**,
  131104. DOI [10.1063/5.0046617](https://doi.org/10.1063/5.0046617).
  RSGDF paper. §III explicitly describes the AO-pair FT route.
