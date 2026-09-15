"""Default-versus-explicit-exact parity for closed-shell UHF/UKS.

The public drivers now fail closed on an explicit quartet far-field request.
These legacy-named tests verify that the default and explicit ``False`` both
select the supported exact route.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _make_h2_box(a_bohr=8.0):
    lat = np.diag([a_bohr, a_bohr, a_bohr])
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])]
    system = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


class TestUnrestrictedFarField:
    """Closed-shell unrestricted exact-route parity."""

    def test_uhf_singlet_far_field_matches_exact(self):
        """UHF mult=1: default route matches explicit exact selection."""
        system, basis = _make_h2_box(8.0)
        kmesh = vq.monkhorst_pack(system, [1, 1, 1])
        opts = vq.PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = 8.0
        opts.lattice_opts.nuclear_cutoff_bohr = 8.0
        opts.max_iter = 10
        opts.use_diis = True

        from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

        r_ff = run_pbc_bipole_uhf(
            system, basis, kmesh, opts,
            use_ewald_j_split=True,
            progress=False,
        )
        r_ex = run_pbc_bipole_uhf(
            system, basis, kmesh, opts,
            use_ewald_j_split=True,
            use_multipole_far_field=False,
            progress=False,
        )

        assert r_ff.converged and r_ex.converged
        err = abs(r_ff.energy - r_ex.energy)
        # Both calls select the supported exact route.
        assert err < 1e-4, (
            f"UHF far-field energy mismatch: {err:.2e} Ha"
        )

    def test_uks_singlet_far_field_matches_exact(self):
        """UKS mult=1 (PBE): default matches explicit exact selection."""
        system, basis = _make_h2_box(8.0)
        kmesh = vq.monkhorst_pack(system, [1, 1, 1])
        opts = vq.PeriodicKSOptions()
        opts.functional = "pbe"
        opts.lattice_opts.cutoff_bohr = 8.0
        opts.lattice_opts.nuclear_cutoff_bohr = 8.0
        opts.max_iter = 10
        opts.use_diis = True

        from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

        r_ff = run_pbc_bipole_uks(
            system, basis, kmesh, opts,
            use_ewald_j_split=True,
            progress=False,
        )
        r_ex = run_pbc_bipole_uks(
            system, basis, kmesh, opts,
            use_ewald_j_split=True,
            use_multipole_far_field=False,
            progress=False,
        )

        assert r_ff.converged and r_ex.converged
        err = abs(r_ff.energy - r_ex.energy)
        assert err < 5e-4, (
            f"UKS far-field energy mismatch: {err:.2e} Ha"
        )
