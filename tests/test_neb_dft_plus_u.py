"""``run_neb(..., dft_plus_u=[HubbardSite(...)])`` tests.

Threads the Dudarev rotationally-invariant per-spin V_U potential
through every per-image SCF in a NEB run.

Coverage matrix:

* Molecular RHF / UHF / RKS / UKS — supported via the Options-side
  ``dft_plus_u_sites`` field on the low-level
  ``run_*_scf_with_jk`` entry points.
* Periodic UHF / UKS — supported via the ``dft_plus_u=`` kwarg on
  ``run_pbc_bipole_uhf`` / ``run_pbc_bipole_uks``; the periodic
  worker (``_evaluate_image_periodic``) forwards the list per-SCF.
* Periodic RHF / RKS — *not* supported yet (BIPOLE RHF / RKS
  drivers don't take ``dft_plus_u``); ``run_neb`` raises with a
  clear pointer to the matching open-shell entry points.

Test pins:

* ``dft_plus_u`` actually changes the SCF energy (E_TS shifts vs.
  no-+U at a U > 0).
* Warm-start with ``dft_plus_u`` set is bit-exact vs. cold-start
  (the V_U Fock contribution is invariant under the SCF initial
  guess; warm-start just changes how many iterations we use to
  reach the same converged density).
* DFT+U works for RHF / UHF / RKS / UKS through the NEB driver.
* Periodic UHF + ``dft_plus_u`` runs end-to-end + shifts E_TS.
* Periodic RHF / RKS + ``dft_plus_u`` raises with a clear message
  pointing at the open-shell entry points.
* The user's options-with-no-dft_plus_u_sites case (the kwarg is
  the only DFT+U surface in the test) works cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    HubbardSite,
    Molecule,
    PeriodicSystem,
    run_neb,
)


def _h2(z2: float) -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(z2)])]
    )


def _h3_doublet(z_middle: float) -> Molecule:
    return Molecule(
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, float(z_middle)]),
            Atom(1, [0.0, 0.0, 4.4]),
        ],
        0,
        2,
    )


class TestDFTPlusURHFNEB:
    """Closed-shell H₂ with +U on H1's s-channel."""

    def _run(self, *, dft_plus_u, warm_start: bool = True):
        return run_neb(
            _h2(1.4),
            _h2(2.0),
            basis="sto-3g",
            n_images=3,
            method="RHF",
            interpolation="linear",
            max_iter=10,
            conv_tol_force=5e-3,
            n_jobs=1,
            initial_step=0.05,
            warm_start=warm_start,
            dft_plus_u=dft_plus_u,
        )

    def test_dft_plus_u_shifts_ts_energy(self):
        """A nonzero U_eff on the H 1s channel must shift the TS
        energy relative to the no-+U baseline. Concrete number: U=4
        eV on H1's s-channel raises the TS by ~26 mHa
        (~0.7 eV) on H₂/STO-3G."""
        baseline = self._run(dft_plus_u=None)
        with_u = self._run(
            dft_plus_u=[HubbardSite(atom_index=0, l=0, U_ev=4.0)]
        )
        # Same NEB-iter budget, so trajectories diverge only via the
        # +U energy contribution.
        assert with_u.n_iter == baseline.n_iter
        e_ts_baseline = float(baseline.energies[baseline.transition_state_index])
        e_ts_with_u = float(with_u.energies[with_u.transition_state_index])
        delta = e_ts_with_u - e_ts_baseline
        assert delta > 0.01, (
            f"E_TS should shift up by tens of mHa with U=4eV; got "
            f"Δ = {delta:.4e} Ha"
        )

    def test_warm_start_bit_exact_with_dft_plus_u(self):
        sites = [HubbardSite(atom_index=0, l=0, U_ev=4.0)]
        cold = self._run(dft_plus_u=sites, warm_start=False)
        warm = self._run(dft_plus_u=sites, warm_start=True)
        # Warm-start changes only the SCF initial guess; the V_U
        # Fock contribution depends only on the converged density,
        # which is identical between the two paths. Bit-exact to
        # 1e-9 Ha on the test fixture.
        assert warm.n_iter == cold.n_iter
        np.testing.assert_allclose(
            warm.energies, cold.energies, atol=1e-9
        )


class TestDFTPlusUUHFNEB:
    """Open-shell H₃ doublet — exercise the UHF / +U interaction."""

    def test_dft_plus_u_on_uhf_neb_runs_and_shifts_energy(self):
        """Small U on a doublet H₃ — confirms the (alpha, beta)
        DFT+U path through UHF works end-to-end in NEB. We use a
        small U_ev so the +U Fock perturbation doesn't destabilise
        the doublet SCF; the energy shift is correspondingly small
        but cleanly detectable."""
        sites = [HubbardSite(atom_index=1, l=0, U_ev=1.0)]
        from vibeqc import UHFOptions

        uhf = UHFOptions()
        uhf.max_iter = 400
        # Close-to-equilibrium endpoints — avoids the very stretched
        # geometries where doublet UHF SCFs become finicky with +U.
        endpoints_close = (_h3_doublet(1.4), _h3_doublet(1.8))
        common = dict(
            basis="sto-3g",
            n_images=2,
            method="UHF",
            uhf_options=uhf,
            interpolation="linear",
            max_iter=3,
            conv_tol_force=1e-2,
            n_jobs=1,
            initial_step=0.05,
            warm_start=False,
        )
        baseline = run_neb(*endpoints_close, **common)
        with_u = run_neb(*endpoints_close, **common, dft_plus_u=sites)
        assert np.all(np.isfinite(with_u.energies))
        delta = float(
            with_u.energies[with_u.transition_state_index]
            - baseline.energies[baseline.transition_state_index]
        )
        # Even a small U=1eV must shift the open-shell TS energy by
        # ≥ 1e-4 Ha (the per-spin Dudarev term is non-trivial).
        assert abs(delta) > 1e-4


class TestDFTPlusURKSNEB:
    """Closed-shell H₂ KS-DFT NEB with +U."""

    def test_dft_plus_u_shifts_rks_ts_energy(self):
        from vibeqc import RKSOptions

        rks = RKSOptions()
        rks.functional = "lda"
        sites = [HubbardSite(atom_index=0, l=0, U_ev=4.0)]
        common = dict(
            basis="sto-3g",
            n_images=3,
            method="RKS",
            functional="lda",
            rks_options=rks,
            interpolation="linear",
            max_iter=5,
            conv_tol_force=1e-2,
            n_jobs=1,
            warm_start=False,
        )
        baseline = run_neb(_h2(1.4), _h2(2.0), **common)
        with_u = run_neb(_h2(1.4), _h2(2.0), **common, dft_plus_u=sites)
        assert np.all(np.isfinite(with_u.energies))
        delta = float(
            with_u.energies[with_u.transition_state_index]
            - baseline.energies[baseline.transition_state_index]
        )
        assert delta > 1e-3


class TestDFTPlusUPeriodicGuard:
    """The historical periodic + dft_plus_u + closed-shell guard
    has been retired: as of v0.9.0 ``run_pbc_bipole_rhf`` /
    ``run_pbc_bipole_rks`` both accept ``dft_plus_u=`` (closed-shell
    BIPOLE +U landing). Periodic NEB + dft_plus_u now works for all
    four methods. What remains here is the *boundary* guard — an
    invalid HubbardSite (channel not in the basis) raises before
    any SCF runs."""

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_invalid_hubbard_site_raises(self, tmp_path, dry_run):
        """A HubbardSite pointing at an (atom_index, l) channel
        absent from the basis must raise at the boundary so the
        user gets the error before any SCF runs."""
        stem = tmp_path / f"invalid-hubbard-site-{dry_run}"
        with pytest.raises(ValueError, match="no AOs in the basis"):
            run_neb(
                _h2(1.4), _h2(2.0),
                basis="sto-3g",
                n_images=2,
                method="RHF",
                # STO-3G H only has an s-channel (l=0); p doesn't exist.
                dft_plus_u=[HubbardSite(atom_index=0, l=1, U_ev=4.0)],
                dry_run=dry_run,
                output=stem,
            )
        assert not stem.with_suffix(".system").exists()


class TestDFTPlusUPeriodicUHFNEB:
    """End-to-end periodic UHF + DFT+U through ``run_neb``.

    Exercises the full
    ``run_neb → _evaluate_image_periodic → run_pbc_bipole_uhf(..., dft_plus_u=[...])``
    dispatch path. Uses an H₂⁺ doublet in a small cubic box — minimal
    open-shell periodic fixture; converges quickly and is sensitive
    to a U_ev on the H 1s channel."""

    # Box size is deliberately moderate (12 bohr ≈ 6.35 Å). Tighter
    # boxes (e.g. 4 bohr) put neighbouring image AOs on top of each
    # other and pump the +U projector by orders of magnitude (one
    # observed pathology: U=4 eV → ΔE_TS ≈ -224 eV at L=4 bohr,
    # vs. the physically-sane +0.27 eV at L≥8 bohr). The docstring
    # of ``examples/workflows/input-periodic-h2plus-neb-plus-u.py``
    # writes this up; until the periodic +U projector is hardened
    # against small-cell image overlap, NEB tests + examples should
    # stay at L≥8 bohr.
    BOX_BOHR = 12.0

    def _h2_plus_in_box(self, z2: float) -> PeriodicSystem:
        L = np.diag([self.BOX_BOHR] * 3)
        return PeriodicSystem(
            3,
            L,
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, float(z2)]),
            ],
            charge=1,
            multiplicity=2,
        )

    def _common(self):
        from vibeqc import LatticeSumOptions, PeriodicRHFOptions

        lat = LatticeSumOptions()
        lat.cutoff_bohr = self.BOX_BOHR / 2.0
        opts = PeriodicRHFOptions()
        opts.lattice_opts = lat
        opts.max_iter = 100
        return dict(
            basis="sto-3g",
            n_images=2,
            method="UHF",
            uhf_options=opts,
            interpolation="linear",
            max_iter=2,
            conv_tol_force=1e-1,
            n_jobs=1,
            kpoints=(1, 1, 1),
            fd_step_bohr=5e-3,
            warm_start=False,
        )

    def test_periodic_uhf_dft_plus_u_runs_and_shifts_energy(self):
        """U_ev on the H 1s channel must shift the periodic UHF TS
        energy relative to the no-+U baseline. The Dudarev per-spin
        V_U Fock contribution is non-zero on this fixture."""
        reactant = self._h2_plus_in_box(1.4)
        product = self._h2_plus_in_box(1.8)
        common = self._common()
        sites = [HubbardSite(atom_index=0, l=0, U_ev=4.0)]
        baseline = run_neb(reactant, product, **common)
        with_u = run_neb(reactant, product, **common, dft_plus_u=sites)
        assert np.all(np.isfinite(with_u.energies))
        delta = float(
            with_u.energies[with_u.transition_state_index]
            - baseline.energies[baseline.transition_state_index]
        )
        # A U=4 eV on the singly-occupied H 1s channel produces a
        # ~9.5 mHa Dudarev contribution per spin on this fixture
        # (singly-occupied bonding MO → 1s projection ≈ 0.5 →
        # tr(n−n²) ≈ 0.25 → (U_eff/2) × 0.25 ≈ 9.5 mHa per spin).
        # NEB TS energy shift tracks that closely (a few mHa for
        # the 2-iter trajectory at this conv_tol_force budget).
        # Bracket the physical range rather than just asserting
        # nonzero — a future regression of the periodic +U Fock
        # path (e.g. image-overlap pumping at small box) would land
        # well outside this window.
        assert 1e-3 < delta < 0.05, (
            f"Periodic UHF E_TS shift must sit in the Dudarev "
            f"~few-mHa window for U=4eV on H 1s; got Δ = {delta:.4e} Ha"
        )


class TestDFTPlusUPeriodicRHFNEB:
    """End-to-end periodic RHF + DFT+U through ``run_neb``.

    Covers the closed-shell branch that became reachable when
    ``run_pbc_bipole_rhf(..., dft_plus_u=...)`` landed in v0.9.0.
    Uses neutral H₂ in a 12-bohr cubic box (closed-shell, 2 e⁻).
    Mirrors ``TestDFTPlusUPeriodicUHFNEB`` for the RHF branch."""

    BOX_BOHR = 12.0

    def _h2_in_box(self, z2: float) -> PeriodicSystem:
        L = np.diag([self.BOX_BOHR] * 3)
        return PeriodicSystem(
            3, L,
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, float(z2)]),
            ],
        )

    def _common(self):
        from vibeqc import LatticeSumOptions, PeriodicRHFOptions

        lat = LatticeSumOptions()
        lat.cutoff_bohr = self.BOX_BOHR / 2.0
        opts = PeriodicRHFOptions()
        opts.lattice_opts = lat
        opts.max_iter = 100
        return dict(
            basis="sto-3g",
            n_images=2,
            method="RHF",
            rhf_options=opts,
            interpolation="linear",
            max_iter=2,
            conv_tol_force=1e-1,
            n_jobs=1,
            kpoints=(1, 1, 1),
            fd_step_bohr=5e-3,
            warm_start=False,
        )

    def test_periodic_rhf_dft_plus_u_runs_and_shifts_energy(self):
        reactant = self._h2_in_box(1.4)
        product = self._h2_in_box(1.8)
        common = self._common()
        sites = [HubbardSite(atom_index=0, l=0, U_ev=4.0)]
        baseline = run_neb(reactant, product, **common)
        with_u = run_neb(reactant, product, **common, dft_plus_u=sites)
        assert np.all(np.isfinite(with_u.energies))
        delta = float(
            with_u.energies[with_u.transition_state_index]
            - baseline.energies[baseline.transition_state_index]
        )
        # Closed-shell H₂: doubly-occupied bonding MO, AO projection
        # onto H[0] 1s ≈ 0.5 → tr(n − n²) ≈ 0.25. Closed-shell
        # bookkeeping in run_pbc_bipole_rhf doubles the Dudarev
        # contribution: E_U = 2 × (U_eff/2) tr(n − n²) ≈ 18 mHa
        # for U=4eV. NEB TS shift tracks that within the same
        # Dudarev window.
        assert 1e-3 < delta < 0.1, (
            f"Periodic RHF E_TS shift must sit in the Dudarev "
            f"window for U=4eV on H 1s; got Δ = {delta:.4e} Ha"
        )
