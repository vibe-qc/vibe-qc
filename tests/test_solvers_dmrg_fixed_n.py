"""Fixed-N regression coverage for the DMRG solver.

Background — audit finding 2
----------------------------
``tests/test_solvers_integration.py`` exercises DMRG on H2/STO-3G and
similar benign molecules where the global (Fock-space) ground state
*is* the requested ``nelec``.  Such tests cannot detect a regression
that lets DMRG converge to the vacuum (or to a wrong-N sector) when
that sector has lower energy than the requested one — i.e. the
classic "wrong Fock-sector bug" for ab-initio DMRG without explicit
particle-number conservation.

The toy Hamiltonians here pin specific N (and where practical, ``ms2``)
sectors against deliberately lower wrong-N sectors.  Any implementation
that silently drops the particle-number penalty (see
``python/vibeqc/solvers/_dmrg.py::_build_full_hamiltonian``: the
``PENALTY * abs(n_elec_state - nelec)`` block) will fail these.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.solvers import DMRGOptions, Hamiltonian, solve_dmrg


def _toy_repulsive(norb: int = 2, h1_diag=(-5.0, -3.0), repulsion: float = 100.0):
    """Toy Hamiltonian with strong on-site repulsion.

    Each spatial orbital carries a negative diagonal one-body term, so
    individual electrons sit at low energy, but ``h2e[p,q,p,q] = U`` is
    very large positive — so the 2- and many-electron sectors are
    pushed to high positive energies while vacuum stays at 0.

    Energy ordering for ``norb=2``:
      * vacuum (0e) ......... E = 0
      * 1e ground ........... E ≈ min eig of h1e (slightly < h1_diag[0])
      * 2e (singlet) ........ E ≈ Σ h1_diag + U·4 ≫ 0
    """
    h1e = np.zeros((norb, norb))
    for i, v in enumerate(h1_diag[:norb]):
        h1e[i, i] = v
    # add a tiny coupling so the 1e ground state isn't degenerate
    if norb >= 2:
        h1e[0, 1] = h1e[1, 0] = 0.05
    h2e = np.zeros((norb, norb, norb, norb))
    for p in range(norb):
        for q in range(norb):
            h2e[p, q, p, q] = repulsion
    return h1e, h2e


def _dmrg_opts() -> DMRGOptions:
    return DMRGOptions(
        n_sweeps=6,
        bond_dim_schedule=[8, 16, 16, 16, 16, 16],
        verbose=0,
        random_seed=0,
    )


class TestDMRGFixedNSector:
    """Pin DMRG to a specific N sector against a lower wrong-N sector."""

    def test_two_electron_against_lower_vacuum(self):
        """nelec=2 requested; vacuum (0e) and 1e sectors are far lower.

        DMRG must report the 2-electron energy (large positive), not
        the vacuum 0 or the 1-electron negative energy.
        """
        h1e, h2e = _toy_repulsive(norb=2, h1_diag=(-5.0, -3.0), repulsion=100.0)
        ham = Hamiltonian(
            h1e=h1e, h2e=h2e, nuclear_repulsion=0.0,
            norb=2, nelec=2, ms2=0,
        )
        res = solve_dmrg(ham, _dmrg_opts())
        # Spot-check the wrong-N sectors are indeed lower so the test
        # is meaningful: vacuum is 0 and the 1e ground is min eig h1e.
        e_1e_lower = float(np.linalg.eigvalsh(h1e)[0])
        assert e_1e_lower < 0.0, "test premise: 1e sector should be < 0"
        # If DMRG fell into the vacuum it would report ≈ 0; the 1e sector
        # is < 0; the genuine 2e sector with repulsion ~100 is ≫ 0.
        assert res.energy > 10.0, (
            f"DMRG appears to have escaped the 2-electron sector: "
            f"E={res.energy:.4f}; vacuum=0, 1e≈{e_1e_lower:.4f}. "
            "Particle-number penalty in _build_full_hamiltonian may "
            "be missing."
        )

    def test_vacuum_sector_returns_zero(self):
        """nelec=0 must return the vacuum energy (= nuclear repulsion)."""
        h1e, h2e = _toy_repulsive()
        ham = Hamiltonian(
            h1e=h1e, h2e=h2e, nuclear_repulsion=1.234,
            norb=2, nelec=0, ms2=0,
        )
        res = solve_dmrg(ham, _dmrg_opts())
        # Pure vacuum: E_elec must be 0; total = nuclear repulsion only.
        assert res.energy == pytest.approx(1.234, abs=1e-6), (
            f"vacuum (nelec=0) DMRG energy must equal nuclear "
            f"repulsion ({1.234}); got {res.energy}."
        )

    def test_one_electron_ms2_one_doublet(self):
        """nelec=1, ms2=1: must land in the single-electron sector.

        With strong on-site repulsion the 2e sector lies high above,
        and the vacuum is at 0.  The 1-electron ground state is the
        lowest eigenvalue of h1e (≈ −5.0 with mixing).
        """
        h1e, h2e = _toy_repulsive(norb=2, h1_diag=(-5.0, -3.0), repulsion=100.0)
        ham = Hamiltonian(
            h1e=h1e, h2e=h2e, nuclear_repulsion=0.0,
            norb=2, nelec=1, ms2=1,
        )
        res = solve_dmrg(ham, _dmrg_opts())
        e_1e_ref = float(np.linalg.eigvalsh(h1e)[0])
        # Tolerant — the toy DMRG approximates the 2-site case but
        # must clearly be in the 1e sector (close to e_1e_ref ≈ -5).
        # If it fell into vacuum (E=0) or 2e (E ≫ 0), this fails hard.
        assert res.energy < -1.0, (
            f"DMRG energy {res.energy} not in the 1-electron sector "
            f"(reference ≈ {e_1e_ref}; vacuum = 0)."
        )
        assert abs(res.energy - e_1e_ref) < 0.5, (
            f"DMRG 1-electron energy {res.energy} too far from the "
            f"reference {e_1e_ref}; sweep schedule may be too short, "
            "or the fixed-N constraint is leaking."
        )


class TestDMRGFullHamiltonianPenalty:
    """Direct white-box check on the full Hamiltonian builder.

    The dense full-Fock-space Hamiltonian must place a large positive
    energy on diagonal entries for every wrong-N basis state.  This
    test fails if that block is removed or weakened.
    """

    def test_wrong_N_states_are_penalised(self):
        from vibeqc.solvers._dmrg import DMRGSolver

        norb_spat = 2
        h1e_spat = np.diag([-5.0, -3.0])
        h2e_spat = np.zeros((norb_spat,) * 4)
        # Build via the public solve() path so the spin-orbital
        # transformation is identical to what DMRG actually uses.
        ham = Hamiltonian(
            h1e=h1e_spat, h2e=h2e_spat, nuclear_repulsion=0.0,
            norb=norb_spat, nelec=2, ms2=0,
        )
        s = DMRGSolver()
        s._h1e, s._h2e, s._norb = s._spatial_to_spinorbital(
            ham.h1e, ham.h2e, ham.norb
        )
        s._nelec = ham.nelec
        H_full = s._build_full_hamiltonian()
        n = s._norb  # spin-orbitals
        # Vacuum state (basis index 0) has 0 electrons; with nelec=2
        # the penalty must dominate so the diagonal is large positive.
        assert H_full[0, 0] > 1e5, (
            "Wrong-N diagonal must carry the particle-number penalty "
            "(see _dmrg.py _build_full_hamiltonian PENALTY block)."
        )
        # Pick a single-occupation state — also wrong N.
        idx_1e = 1  # bit 0 set, 1 electron
        assert H_full[idx_1e, idx_1e] > 1e5, (
            "1-electron diagonal not penalised relative to nelec=2 request."
        )
        # Any correct-N state must have a much smaller diagonal:
        # state with bits 0 and 1 set is a 2-electron state.
        idx_2e = (1 << 0) | (1 << 1)
        assert H_full[idx_2e, idx_2e] < 1e3, (
            "Correct-N diagonal should not be penalised."
        )
