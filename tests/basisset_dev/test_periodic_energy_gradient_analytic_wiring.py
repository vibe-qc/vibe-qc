"""Wiring test for the Phase-P2 skeleton ``periodic_energy_gradient_analytic``.

The full periodic analytic gradient draws dS(k)/dT(k) from Phase P1 and dV(k)/dG(k)
from an injected :class:`PeriodicCoulombNuclearDerivativeProvider` (the R13 kernel,
not yet landed). This test exercises the *wiring* with mock providers, on a real
periodic basis and a mock GDF-result (hand-built per-k P(k)/F(k)), since no
production provider exists yet:

1. **Zero provider** (dV = dG = 0): the result must equal an independent
   reference built from Phase P1 + the contraction helper directly, proving the
   spec to libint-shell mapping, the per-k P(k)/F(k)/k-weight extraction, the W(k)
   build, and the LOG chain rule are wired correctly.
2. **Known provider**: with constant dV(k)/dG(k), the increment over the
   zero-provider result must equal Σ_k w_k Re[tr(P·dV) + ½ tr(P·dG)]·(dα/dx),
   proving dV enters Hcore at coefficient 1 and dG enters at coefficient ½.

Needs a built vibe-qc; skipped otherwise. The contraction math itself is covered
build-free by ``test_periodic_energy_gradient_assembly.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")
scipy_linalg = pytest.importorskip("scipy.linalg")

for _needed in ("overlap_exponent_derivative", "compute_overlap_lattice", "bloch_sum"):
    if not hasattr(vq, _needed):
        pytest.skip(f"{_needed} not in this build", allow_module_level=True)

from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)
from vibeqc.basis_optimization.energy_gradient import (  # noqa: E402
    _chain_rule,
    _shell_atom_maps,
    _spec_target_shells,
)
from vibeqc.basis_optimization.periodic_energy_gradient import (  # noqa: E402
    bloch_summed_one_electron_exponent_derivatives,
    periodic_energy_gradient_analytic,
    _periodic_pulay_contract,
)

SRC_H = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP" / "01_H"


def _hermitian(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


class _ConstProvider:
    """dV(k)/dG(k) provider returning fixed per-k matrices (mock R13 kernel)."""

    def __init__(self, dV_k, dG_k):
        self._dV = dV_k
        self._dG = dG_k

    def nuclear_kpoint_derivative(self, target_shells, prim_idx):
        return self._dV

    def gmatrix_kpoint_derivative(self, target_shells, prim_idx):
        return self._dG


def _build_case():
    """A 1D H2-per-cell chain on pob-TZVP H, with a mock GDF result (per-k P/F
    from the real periodic integrals). Returns everything the wiring needs."""
    par = BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(SRC_H)},
        free=[FreeSpec("H", 0, 0, "exponent", transform=Transform.LOG)],
    )
    x0 = par.pack()
    lib = TempBasisLibrary().__enter__()
    name = lib.write_g94(par.unpack(x0), basis_name="pgrad-wiring")

    a = 5.0
    lattice = np.array([[a, 0.0, 0.0], [0.0, 1.0e6, 0.0], [0.0, 0.0, 1.0e6]])
    system = vq.PeriodicSystem(
        1, lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), name)
    nbf = basis.nbasis

    opts = vq.LatticeSumOptions()
    s_lat = vq.compute_overlap_lattice(basis, system, opts)
    t_lat = vq.compute_kinetic_lattice(basis, system, opts)

    kpoints = np.array([[0.0, 0.0, 0.0], [0.31, 0.0, 0.0]])
    kweights = np.array([0.5, 0.5])
    n_occ = 1  # 2 electrons / cell, closed shell
    P_k, F_k = [], []
    for k in kpoints:
        Sk = np.asarray(vq.bloch_sum(s_lat, k))
        Fk = np.asarray(vq.bloch_sum(t_lat, k))  # kinetic as a stand-in Hermitian Fock
        _evals, C = scipy_linalg.eigh(Fk, Sk)
        occ = C[:, :n_occ]
        P_k.append(2.0 * occ @ occ.conj().T)
        F_k.append(Fk)

    result = SimpleNamespace(
        density=P_k, fock=F_k, kpoints_cart=kpoints, kpoint_weights=kweights
    )
    return par, x0, system, basis, opts, result, kpoints, kweights, nbf, lib


def test_wiring_zero_provider_matches_p1_reference():
    par, x0, system, basis, opts, result, kpoints, kweights, nbf, lib = _build_case()
    try:
        nk = len(kpoints)
        zero = [np.zeros((nbf, nbf), dtype=complex) for _ in range(nk)]
        g = periodic_energy_gradient_analytic(
            par, system, basis, result, _ConstProvider(zero, zero), x0,
            lattice_opts=opts,
        )

        # Independent reference: P1 one-electron derivatives + the contraction.
        spec = par.free[0]
        shells, by_atom, atom_syms, _ = _shell_atom_maps(vq, basis, system.unit_cell_molecule())
        targets = _spec_target_shells(spec, par, shells, by_atom, atom_syms)
        dS_k, dT_k = bloch_summed_one_electron_exponent_derivatives(
            system, basis, targets, spec.prim_idx, kpoints, lattice_opts=opts
        )
        dphys = _periodic_pulay_contract(
            result.density, result.fock, kweights, dS_k, dT_k, zero
        )
        g_ref = dphys * _chain_rule(spec, x0[0])
        np.testing.assert_allclose(g[0], g_ref, atol=1e-10, rtol=1e-8)
        # the one-electron contribution is non-trivial (guards against all-zero)
        assert abs(g[0]) > 1e-8
    finally:
        lib.__exit__(None, None, None)


def test_wiring_provider_seam_dv_dg():
    par, x0, system, basis, opts, result, kpoints, kweights, nbf, lib = _build_case()
    try:
        nk = len(kpoints)
        rng = np.random.default_rng(5)
        zero = [np.zeros((nbf, nbf), dtype=complex) for _ in range(nk)]
        dV_k = [_hermitian(rng, nbf) for _ in range(nk)]
        dG_k = [_hermitian(rng, nbf) for _ in range(nk)]

        g0 = periodic_energy_gradient_analytic(
            par, system, basis, result, _ConstProvider(zero, zero), x0, lattice_opts=opts
        )
        g1 = periodic_energy_gradient_analytic(
            par, system, basis, result, _ConstProvider(dV_k, dG_k), x0, lattice_opts=opts
        )

        spec = par.free[0]
        chain = _chain_rule(spec, x0[0])
        # dV enters Hcore (coeff 1), dG enters at coeff 1/2.
        expect = chain * sum(
            kweights[k] * np.real(
                np.trace(result.density[k] @ dV_k[k])
                + 0.5 * np.trace(result.density[k] @ dG_k[k])
            )
            for k in range(nk)
        )
        np.testing.assert_allclose(g1[0] - g0[0], expect, atol=1e-10, rtol=1e-8)
    finally:
        lib.__exit__(None, None, None)


def test_wiring_requires_provider():
    par, x0, system, basis, opts, result, kpoints, kweights, nbf, lib = _build_case()
    try:
        with pytest.raises(NotImplementedError, match="R13"):
            periodic_energy_gradient_analytic(
                par, system, basis, result, None, x0, lattice_opts=opts
            )
    finally:
        lib.__exit__(None, None, None)
