"""Public-API coverage for ``run_job(..., active_space=(n_active, n_elec))``.

Background
----------
The ``active_space`` kwarg is documented in
``python/vibeqc/solvers/ACTIVE_SPACE.md`` as supported by every non-HF
solver.  The runner at ``python/vibeqc/runner.py`` now hands the
determinant solvers a properly frozen-core-dressed Hamiltonian via
:meth:`vibeqc.solvers.Hamiltonian.active_space`, which applies the
standard complete-active-space partition: the lowest
``(nelec − n_elec) // 2`` orbitals become a doubly-occupied inactive
core whose mean field dresses the active one-electron term,

    h_pq^eff = h_pq + Σ_c (2·<pc|qc> − <pc|cq>)   (p, q active)

and whose constant inactive energy ``E_core`` is folded into the
nuclear-repulsion offset so the active-space CI reproduces the
full-space CASCI total energy.

This file pins both the historically-correct full-space identity (no
frozen core) and the frozen-core truncation that audit finding 1 had
left as strict ``xfail``.  The frozen-core cases are now asserted
positively: they were ``xfail(strict=True)`` while the dressing was
missing and flipped to real assertions the moment
``Hamiltonian.active_space`` landed.  Rather than lean on a hand-tuned
HF constant, the checks use the geometry-independent variational
sandwich ``E_FCI ≤ E_CAS ≤ E_HF`` (the HF determinant is a member of any
CAS that includes the frontier orbitals, and full FCI is the global
minimum) plus cross-method consistency (``fci`` and ``casci`` on the
same active space must agree — they share the dressing).
"""

from __future__ import annotations

import pytest
from vibeqc import Atom, Molecule


@pytest.fixture
def h2_sto3g():
    return Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])


@pytest.fixture
def h2o_sto3g():
    # Bohr; HF/STO-3G geometry from tests/test_solvers_integration.py neighbours.
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )


def _run(mol, method, tmp_path, **kw):
    from vibeqc.runner import run_job

    return run_job(
        mol,
        basis="sto-3g",
        method=method,
        output=tmp_path / f"as_{method}",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        **kw,
    )


class TestActiveSpaceFullSpaceIdentity:
    """``active_space=(norb, nelec)`` must match no-truncation energy.

    This is the only currently-trustworthy active-space path: when the
    requested active space spans the full MO space there is no frozen
    core, so the runner's slice-and-rebuild approach is exact.
    """

    def test_fci_full_space(self, h2_sto3g, tmp_path):
        e_full = _run(h2_sto3g, "fci", tmp_path).energy
        e_as = _run(
            h2_sto3g, "fci", tmp_path, active_space=(2, 2)
        ).energy
        assert abs(e_as - e_full) < 1e-9, (
            f"active_space=(2,2) on H2/sto-3g must equal full FCI "
            f"({e_full:.10f}); got {e_as:.10f}"
        )

    def test_selected_ci_full_space(self, h2_sto3g, tmp_path):
        e_full = _run(h2_sto3g, "selected_ci", tmp_path).energy
        e_as = _run(
            h2_sto3g,
            "selected_ci",
            tmp_path,
            active_space=(2, 2),
        ).energy
        assert abs(e_as - e_full) < 1e-6


class TestActiveSpaceFrozenCore:
    """Genuine truncation: ``n_active < norb`` requires frozen-core
    dressing of ``h1e`` plus a constant ``E_core`` offset.

    These were ``xfail(strict=True)`` while the dressing was missing
    (audit finding 1).  :meth:`Hamiltonian.active_space` now applies the
    standard CAS partition + dressing, so they assert positively.

    The checks avoid a hand-tuned HF constant.  Instead they use the
    variational sandwich that holds for *any* geometry and basis:

        E_FCI(full)  ≤  E_CAS(n_active, n_elec)  ≤  E_HF

    The upper bound holds because the HF determinant — the inactive core
    plus the doubly-occupied frontier orbitals — is itself one of the CAS
    determinants (the CAS is built on the HF orbitals), so the variational
    CAS energy can only go lower.  The lower bound holds because full FCI
    is the global minimum in the one-electron basis.  The pre-fix bug
    reported the active-only energy (≈ −15.5 Ha for CAS(4,4)/H₂O), which
    violates the upper bound by ~60 Ha — so the sandwich is also a sharp
    regression guard.
    """

    def test_fci_cas44_h2o_sandwich(self, h2o_sto3g, tmp_path):
        # CAS(4,4) on H₂O/STO-3G: freeze the 3 lowest MOs (6 core e⁻),
        # keep 4 active orbitals with 4 active e⁻.
        e_hf = _run(h2o_sto3g, "rhf", tmp_path).energy
        e_fci_full = _run(h2o_sto3g, "fci", tmp_path).energy
        e_cas = _run(h2o_sto3g, "fci", tmp_path, active_space=(4, 4)).energy
        assert e_fci_full - 1e-7 <= e_cas <= e_hf + 1e-7, (
            f"CAS(4,4) energy {e_cas:.8f} must sit between full FCI "
            f"{e_fci_full:.8f} and HF {e_hf:.8f}"
        )
        # Frozen-core regression guard: the old active-only number was
        # ~ −15.5 Ha; a properly dressed CAS is below the HF total.
        assert e_cas < e_hf + 1e-7

    def test_fci_and_casci_active_space_agree(self, h2o_sto3g, tmp_path):
        # The determinant ``fci`` path and the dedicated ``casci`` path now
        # share one frozen-core dressing (``_frozen_core_dressing``) and the
        # same CAS partition, so they must report the same total energy on
        # the same active space — to numerical precision, not just sign.
        e_fci_cas = _run(
            h2o_sto3g, "fci", tmp_path, active_space=(4, 4)
        ).energy
        e_casci = _run(
            h2o_sto3g, "casci", tmp_path, active_space=(4, 4)
        ).energy
        assert abs(e_fci_cas - e_casci) < 1e-9, (
            f"fci CAS(4,4) {e_fci_cas:.10f} and casci CAS(4,4) "
            f"{e_casci:.10f} must agree (shared dressing)"
        )

    def test_selected_ci_cas44_h2o_sandwich(self, h2o_sto3g, tmp_path):
        e_hf = _run(h2o_sto3g, "rhf", tmp_path).energy
        e_fci_full = _run(h2o_sto3g, "fci", tmp_path).energy
        e_cas = _run(
            h2o_sto3g, "selected_ci", tmp_path, active_space=(4, 4)
        ).energy
        # Selected-CI on 4 orbitals saturates the CAS, so it lands on the
        # CAS-FCI value: inside the sandwich, below HF.
        assert e_fci_full - 1e-6 <= e_cas <= e_hf + 1e-6

    def test_dmrg_cas44_h2o_sandwich(self, h2o_sto3g, tmp_path):
        # CAS(4,4) keeps DMRG inside its ≤6 spatial-orbital ceiling; with a
        # bond dimension ≥ 2^4 the MPS is exact in the active space.
        from vibeqc.solvers import DMRGOptions

        e_hf = _run(h2o_sto3g, "rhf", tmp_path).energy
        e_fci_full = _run(h2o_sto3g, "fci", tmp_path).energy
        e_cas = _run(
            h2o_sto3g,
            "dmrg",
            tmp_path,
            active_space=(4, 4),
            dmrg_options=DMRGOptions(
                n_sweeps=6, bond_dim_schedule=[8, 16, 32, 32, 32, 32], verbose=0
            ),
        ).energy
        assert e_fci_full - 1e-3 <= e_cas <= e_hf + 1e-3


class TestActiveSpaceValidation:
    """``Hamiltonian.active_space`` rejects ill-posed partitions."""

    def _full_h2o_hamiltonian(self, mol):
        from vibeqc import BasisSet
        from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider

        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        return build_hamiltonian_mo(mol, basis, C)

    def test_odd_inactive_count_rejected(self, h2o_sto3g):
        # nelec=10, n_active_elec=3 → inactive count 7 is odd → invalid.
        H = self._full_h2o_hamiltonian(h2o_sto3g)
        with pytest.raises(ValueError, match="even"):
            H.active_space(4, 3)

    def test_active_window_too_large_rejected(self, h2o_sto3g):
        # n_core=(10-2)/2=4 plus 4 active = 8 > norb=7.
        H = self._full_h2o_hamiltonian(h2o_sto3g)
        with pytest.raises(ValueError, match="exceeds|active"):
            H.active_space(4, 2)

    def test_full_space_is_identity(self, h2o_sto3g):
        # active_space=(norb, nelec) has no frozen core: integrals unchanged,
        # nuclear_repulsion unchanged (E_core == 0).
        import numpy as np

        H = self._full_h2o_hamiltonian(h2o_sto3g)
        Hc = H.active_space(H.norb, H.nelec)
        assert Hc.norb == H.norb and Hc.nelec == H.nelec
        assert abs(Hc.nuclear_repulsion - H.nuclear_repulsion) < 1e-12
        assert np.allclose(Hc.h1e, H.h1e)
