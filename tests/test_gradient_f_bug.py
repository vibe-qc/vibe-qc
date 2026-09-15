"""Regression test for the v0.7.3 RHF gradient bug (fixed in v0.7.4).

Pre-fix (v0.7.3), ``compute_gradient`` (and its 2e kernel
``two_electron_gradient_contribution``) gave systematically wrong values
on multi-heavy-atom systems with f-functions in the basis (e.g.
glycine + def2-tzvp showed up to ~160 mHa per-element error vs PySCF /
ORCA / FD on compute-reference, even though E_RHF agreed to 1e-9). The bug did not
trigger on:

* low-l bases (def2-svp, max l=2);
* single-heavy-atom systems (HF, H2O, NH3) even with def2-tzvp;
* CH4 + def2-tzvp (Td symmetry zeroes out the buggy terms).

The smallest reproducer is **H2CO + def2-tzvp** (4 atoms, 74 spherical
basis functions, 30 MB ERI, runs in <30 s). On H2CO atom 0 (C),
component 1 (y), the true RHF gradient is ~0 by Cs symmetry — vibe-qc's
buggy kernel returned ~+11 mHa for ``compute_gradient`` and ~−18 mHa for
the standalone 2e contribution call.

Root cause: the original ``two_electron_gradient_contribution`` looped
over ALL (s1,s2,s3,s4) ∈ shells^4 (no permutational reduction). For
high-l mixed-l quartets this triggers libint2's internal
angular-momentum canonicalisation plus derivative-index unscrambling
via ``DerivMapGenerator``, which gives wrong derivative-to-atom routing
— producing both an algorithmic error (visible at OMP=1) and a
non-deterministic error across thread counts.

**Fix C (v0.7.4)** rewrites the kernel as a canonical 1/8 shell-quartet
loop with an l-canonical reorder before ``engine.compute()``. The
l-canonical reorder bypasses libint's internal
``swap_tbra / swap_tket / swap_braket`` + ``DerivMapGenerator`` path
(matching the analogous fix on the 3c DF kernel in
``cpp/src/df.cpp::compute_3c_eri_gradient_weighted``, commit 2196345).
The 2-particle density is averaged over the 8 ERI permutations:

    Γ_avg = (1/2) D_μν D_λσ − (α_HF/8)(D_μλ D_νσ + D_νλ D_μσ)

times the permutational weight ``s1234_deg``. Post-fix, H2CO/def2-tzvp
matches PySCF to ~5e-11 Ha/bohr (vs ~25 mHa pre-fix).

If this test fails, the gradient regression has come back. Run a quick
check:

    OMP_NUM_THREADS=1 pytest tests/test_gradient_f_bug.py -v

then compare with various ``OMP_NUM_THREADS`` values — they should all
give the same result.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    SCFAccelerator,
    compute_gradient,
    run_rhf,
)


def _h2co_bohr():
    """Formaldehyde, near-equilibrium, Cs symmetric in xz-plane."""
    ANG_TO_BOHR = 1.8897261339213
    coords_ang = np.array(
        [
            (0.0, 0.0, 0.000),  # C
            (0.0, 0.0, 1.205),  # O
            (0.0, 0.943, -0.587),  # H
            (0.0, -0.943, -0.587),  # H
        ]
    )
    Zs = [6, 8, 1, 1]
    return [(z, list(c * ANG_TO_BOHR)) for z, c in zip(Zs, coords_ang)]


def _vibeqc_gradient_h2co_tzvp():
    atoms_bohr = _h2co_bohr()
    mol = Molecule([Atom(z, xyz) for z, xyz in atoms_bohr])
    basis = BasisSet(mol, "def2-tzvp")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    # EDIIS+DIIS hybrid: plain DIIS plateaus above the 1e-10 gradient
    # tolerance on H2CO/def2-tzvp and never reports converged, so
    # compute_gradient would refuse to run. The accelerator only
    # changes the path to the fixed point, not the converged density —
    # this test still exercises the f-shell gradient kernel.
    opts.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    opts.max_iter = 200
    rhf = run_rhf(mol, basis, opts)
    return rhf, np.array(compute_gradient(mol, basis, rhf))


def _pyscf_gradient_h2co_tzvp():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    atoms_bohr = _h2co_bohr()
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = "def2-tzvp"
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    e = mf.kernel()
    return e, mf.nuc_grad_method().kernel()


def test_h2co_def2_tzvp_gradient_matches_pyscf():
    """The full RHF/def2-tzvp gradient on H2CO must match PySCF to <1e-6
    Ha/bohr. Pre-fix, vibe-qc was off by ~25 mHa max on this case; post-
    Fix-C (commit landing this test as passing) the disagreement is at
    ~5e-11 Ha/bohr — essentially PySCF parity at machine precision.
    """
    rhf, g_vq = _vibeqc_gradient_h2co_tzvp()
    e_ps, g_ps = _pyscf_gradient_h2co_tzvp()

    assert rhf.converged
    np.testing.assert_allclose(
        rhf.energy, e_ps, atol=1e-9, rtol=0,
        err_msg="H2CO RHF/def2-tzvp energy disagrees with PySCF (SCF/integrals)",
    )
    np.testing.assert_allclose(
        g_vq, g_ps, atol=1e-6, rtol=0,
        err_msg=(
            "H2CO RHF/def2-tzvp gradient disagrees with PySCF — the v0.7.3 "
            "two_electron_gradient_contribution f-shell bug has likely "
            "regressed. See tests/test_gradient_f_bug.py docstring."
        ),
    )


def test_h2co_atom0_y_gradient_zero_by_symmetry():
    """H2CO is Cs in the xz-plane; the y-force on the C atom (index 0) must
    vanish by symmetry.

    History: pre-fix (v0.7.3) this returned ~+11 mHa on def2-tzvp via
    the direct 4-index ERI gradient kernel. Fix D (commit 4341655)
    auto-routed through the DF gradient and brought the residual down
    to ~-4 mHa; the 3c-kernel l-canonical-reorder follow-up
    (commit 2196345) brought it to ~10⁻¹⁰. Fix C (the canonical 1/8 +
    l-canonical reorder rewrite of the direct kernel landing on this
    branch) restores the direct path to machine precision (~10⁻¹³
    Ha/bohr), and the auto-route is no longer needed.
    """
    _, g_vq = _vibeqc_gradient_h2co_tzvp()
    # Atom 0 is C, component 1 is y.
    assert abs(g_vq[0, 1]) < 1e-6, (
        f"Symmetry-forbidden gradient component is {g_vq[0, 1]:.4e}; "
        f"expected near zero (was ~+1.1e-2 in v0.7.3; ~10⁻¹³ post-Fix-C)."
    )


def test_h2co_def2_tzvp_2e_kernel_deterministic_across_threads():
    """The standalone two_electron_gradient_contribution must be
    deterministic regardless of OMP_NUM_THREADS. Pre-fix the kernel was
    non-deterministic: OMP=1 and OMP=8 gave wildly different values
    (~+34.5 mHa vs ~−0.9 mHa at H2CO atom 0, c=1).

    This test runs with the default thread count and just checks that
    repeated calls within one process give identical results — a basic
    consistency check that the per-thread accumulation isn't racing.
    """
    from vibeqc import _vibeqc_core as core

    atoms_bohr = _h2co_bohr()
    mol = Molecule([Atom(z, xyz) for z, xyz in atoms_bohr])
    basis = BasisSet(mol, "def2-tzvp")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    rhf = run_rhf(mol, basis, opts)
    D = np.asarray(rhf.density)

    g0 = np.asarray(core.two_electron_gradient_contribution(basis, mol, D, 1.0))
    for _ in range(2):
        gi = np.asarray(
            core.two_electron_gradient_contribution(basis, mol, D, 1.0)
        )
        # Allow FP non-associativity at machine epsilon (1e-12 absolute is
        # ~1e-13 relative on the largest 2e gradient elements ~1e+1 Ha/bohr).
        # Pre-fix, this kernel could differ by mHa across thread counts.
        np.testing.assert_allclose(
            g0, gi, atol=1e-12, rtol=0,
            err_msg=(
                "two_electron_gradient_contribution returned materially "
                "different values on repeated calls — race condition in "
                "the parallel reduction has likely regressed."
            ),
        )
