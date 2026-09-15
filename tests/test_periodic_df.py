"""Periodic Γ-point density-fitted 3-index integrals — Stage 4.

Validates :func:`vibeqc.periodic_df.build_periodic_gamma_df`. The extracted
cderi ``Lpq`` must:

1. have shape ``(n_fit, nbf, nbf)`` with each ``Lpq[L]`` symmetric in its AO
   indices;
2. **reproduce the GDF SCF's Coulomb energy** — J built from ``Lpq`` and the
   converged density equals ``PBCGDFResult.e_coulomb`` (the SCF built its own J
   from the same tensor). This is the oracle: it proves the standalone extraction
   yields the exact tensor the periodic GDF uses.
3. give a symmetric exchange matrix and 4-index integrals with the expected
   ``(pq|rs) = (rs|pq) = (qp|rs)`` permutation symmetry after MO transform.

System: H₂ in a 20-bohr box / STO-3G — the known-good compcell-GDF reference
(same as ``tests/test_periodic_accelerator_uniformity.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_df import build_periodic_gamma_df

_AUX = "def2-svp-jk"
_ETA = 1.0


def _h2_box():
    sysp = vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 20.0]),
        [
            vq.Atom(1, [10.0, 10.0, 10.0 - 0.7]),
            vq.Atom(1, [10.0, 10.0, 10.0 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _opts():
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 15.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 200
    opts.use_diis = True
    return opts


@pytest.fixture(scope="module")
def gdf_ref():
    sysp, basis = _h2_box()
    opts = _opts()
    ref = vq.run_pbc_gdf_rhf(
        sysp, basis, opts, aux_basis=_AUX, exxdiv="ewald",
        compcell_eta=_ETA, progress=False,
    )
    assert ref.converged, "reference compcell-GDF RHF must converge"
    return {"system": sysp, "basis": basis, "opts": opts, "ref": ref}


@pytest.fixture(scope="module")
def df(gdf_ref):
    return build_periodic_gamma_df(
        gdf_ref["system"],
        gdf_ref["basis"],
        aux_basis=_AUX,
        lat_opts=gdf_ref["opts"].lattice_opts,
        compcell_eta=_ETA,
    )


def test_three_center_shape(gdf_ref, df):
    nbf = np.asarray(gdf_ref["ref"].overlap).shape[0]
    assert df.three_center.ndim == 3
    assert df.three_center.shape[1] == nbf
    assert df.three_center.shape[2] == nbf
    assert df.n_fit == df.three_center.shape[0]


def test_cderi_symmetric_in_ao_indices(df):
    L = df.three_center
    np.testing.assert_allclose(L, np.transpose(L, (0, 2, 1)), atol=1e-10)


def test_coulomb_energy_reconstruction(gdf_ref, df):
    """Oracle: E_J = ½·tr(D·J) from the extracted cderi == SCF e_coulomb."""
    ref = gdf_ref["ref"]
    D = np.asarray(ref.density)
    J = df.build_J(D)
    e_j = 0.5 * float(np.einsum("ij,ij->", D, J))
    # Same public builder + same params as the SCF ⇒ bit-identical tensor
    # (measured |Δ| = 0.0); 1e-10 keeps margin for cross-platform float noise.
    assert e_j == pytest.approx(float(ref.e_coulomb), abs=1e-10)


def test_build_K_symmetric(gdf_ref, df):
    D = np.asarray(gdf_ref["ref"].density)
    K = df.build_K(D)
    np.testing.assert_allclose(K, K.T, atol=1e-10)


def test_mo_transform_eri_permutation_symmetry(gdf_ref, df):
    """(pq|rs) = (rs|pq) = (qp|rs) from the MO-transformed cderi."""
    C = np.asarray(gdf_ref["ref"].mo_coeffs)
    B = df.mo_transform(C, C)  # (n_fit, nbf, nbf)
    eri = np.einsum("Lpq,Lrs->pqrs", B, B, optimize=True)
    np.testing.assert_allclose(eri, np.transpose(eri, (2, 3, 0, 1)), atol=1e-10)
    np.testing.assert_allclose(eri, np.transpose(eri, (1, 0, 2, 3)), atol=1e-10)
