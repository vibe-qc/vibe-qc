"""Audit regression: analytic atomic-coordinate gradient with ECPs
agrees with finite differences on the same Hamiltonian.

Pre-fix the gradient drivers built the nuclear-repulsion piece and the
``∂V_ne/∂R`` piece with bare atomic numbers, and never added a
``∂V_ECP/∂R`` term, so the analytic gradient differentiated a
different Hamiltonian than the SCF energy. 2026-05-18 audit probe:
Zn-He at def2-SVP/ecp28mdf, analytic z-gradient on Zn was
−0.285 Ha/bohr while finite-difference gave −0.028 Ha/bohr, a factor
of ~10 difference, i.e. the bare-Z gradient was almost entirely wrong.

This file pins:

* ``compute_ecp_gradient_contribution`` is non-zero when an ECP is
  present and zero when it isn't (smoke).
* ``ecp_effective_charges`` returns Z_eff = Z − n_core on the ECP atom
  and the bare Z on all-electron atoms (smoke).
* End-to-end: closed-shell [ZnH]+ analytic gradient agrees with central
  finite differences on the full ECP-aware energy to ≤ 1e-5 Ha/bohr,
  well below the audit's ~0.26 Ha/bohr regression magnitude.

Charge convention: ``Molecule.charge`` is the physical ionic charge
and ``n_electrons()`` the full physical count; the SCF subtracts the
ECP-replaced cores itself (``ECPHcore.total_ncore``). The [ZnH]+ test
system has 30 physical electrons; the 10-electron ecp10mdf core
leaves 20 valence electrons (even, closed shell). See
``docs/user_guide/ecp.md`` § Electron count and charge convention and
``tests/test_ecp_scf.py`` for the broader context.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    ECPCenter,
    GradientOptions,
    Molecule,
    RHFOptions,
    compute_ecp_gradient_contribution,
    compute_ecp_gradient_contribution_from_primitives,
    compute_gradient,
    ecp_effective_charges,
    run_rhf,
)


def _zn_h(z_zn: float = 0.0, z_h: float = 3.0):
    # charge=+1 physical ([ZnH]+): 30 electrons, of which the SCF
    # fills 20 valence orbital-electrons after removing the
    # 10-electron ecp10mdf core.
    return Molecule(
        [Atom(30, [0.0, 0.0, z_zn]), Atom(1, [0.0, 0.0, z_h])], 1, 1
    )


def _scf(mol, *, z_zn: float = 0.0):
    basis = BasisSet(mol, "6-31g")
    opts = RHFOptions()
    opts.ecp_centers = [ECPCenter(Z=30, xyz=[0.0, 0.0, z_zn])]
    opts.ecp_library = "ecp10mdf"
    opts.max_iter = 200
    opts.damping = 0.5
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return basis, opts, run_rhf(mol, basis, opts)


def test_ecp_effective_charges_reports_z_minus_core_on_ecp_atom():
    """ecp10mdf removes 10 core electrons from Zn, leaving Z_eff = 20.
    The all-electron H atom keeps Z_eff = 1."""
    mol = _zn_h()
    ecp_centers = [ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    Z_eff = list(ecp_effective_charges(mol, ecp_centers, "ecp10mdf", ""))
    assert Z_eff == pytest.approx([20.0, 1.0])


def test_ecp_effective_charges_empty_list_returns_bare_z():
    mol = _zn_h()
    Z_eff = list(ecp_effective_charges(mol, [], "ecp10mdf", ""))
    assert Z_eff == pytest.approx([30.0, 1.0])


def test_compute_ecp_gradient_contribution_is_zero_without_ecp():
    """No ECP centers ⇒ the ECP gradient piece is identically zero
    (audit smoke, ensures non-ECP runs are bit-for-bit unchanged)."""
    mol = _zn_h()
    basis, _, r = _scf(mol)
    g_ecp = compute_ecp_gradient_contribution(
        basis, mol, [], r.density, "ecp10mdf", "",
    )
    assert np.asarray(g_ecp).max() == 0.0
    assert np.asarray(g_ecp).shape == (2, 3)


def test_compute_ecp_gradient_contribution_is_nonzero_with_ecp():
    """With ECP present, ∂V_ECP/∂R is non-zero on the ECP atom (by
    translation invariance equal and opposite on the partner atom)."""
    mol = _zn_h()
    basis, opts, r = _scf(mol)
    g_ecp = np.asarray(compute_ecp_gradient_contribution(
        basis, mol, opts.ecp_centers, r.density, opts.ecp_library, "",
    ))
    # Z-direction component on Zn must be non-trivial.
    assert abs(g_ecp[0, 2]) > 1e-3, g_ecp


def test_ecp_analytic_gradient_matches_finite_difference():
    """End-to-end audit regression: closed-shell [ZnH]+ analytic
    gradient = central FD on the full ECP-aware energy to ≤ 1e-5
    Ha/bohr. Pre-fix the audit measured ~0.26 Ha/bohr disagreement,
    so the 1e-5 threshold has 4 orders of magnitude headroom."""
    mol0 = _zn_h(z_zn=0.0)
    basis0, opts0, r0 = _scf(mol0, z_zn=0.0)

    go = GradientOptions()
    go.ecp_centers = opts0.ecp_centers
    go.ecp_library = opts0.ecp_library
    g_analytic = np.asarray(compute_gradient(mol0, basis0, r0, go))

    h = 5e-4

    def energy_at(z_zn: float) -> float:
        mol_ = _zn_h(z_zn=z_zn)
        _, _, r_ = _scf(mol_, z_zn=z_zn)
        return r_.energy

    g_fd_zn_z = (energy_at(+h) - energy_at(-h)) / (2 * h)
    diff = abs(g_analytic[0, 2] - g_fd_zn_z)
    assert diff < 1e-5, (
        f"ECP analytic vs FD on [ZnH]+: |Δ|={diff:.3e} Ha/bohr "
        f"(analytic {g_analytic[0,2]:.7f}, FD {g_fd_zn_z:.7f}); audit "
        f"regression target ≤ 1e-5"
    )


def test_gradient_options_must_carry_ecp_into_the_compute_path():
    """An ECP reference cannot silently fall back to a bare-Z gradient."""
    mol = _zn_h()
    basis, opts, r = _scf(mol)

    go_with = GradientOptions()
    go_with.ecp_centers = opts.ecp_centers
    go_with.ecp_library = opts.ecp_library
    g_with = np.asarray(compute_gradient(mol, basis, r, go_with))
    assert np.all(np.isfinite(g_with))

    go_without = GradientOptions()
    # ecp_centers left empty: the exact provenance contract rejects the
    # pre-audit bare-Z fallback surface.
    with pytest.raises(
        ValueError,
        match="GradientOptions ECP route does not match",
    ):
        compute_gradient(mol, basis, r, go_without)


# ---------------------------------------------------------------------------
# Inline-primitive ECP path (#574)
# ---------------------------------------------------------------------------
#
# The tests above cover ECPs supplied as libecpint XML-library centres. Bases
# whose per-element cores have no matching XML library -- vDZP, def2-mSVP, and
# CRYSTAL/pob data -- reach the SCF through a different route entirely:
# ``attach_inline_ecp_options_from_basis_sidecar`` fills
# ``ecp_primitive_blocks`` and leaves ``ecp_centers`` EMPTY, and
# ``run_rhf`` prefers the inline path whenever those blocks are present.
#
# Pre-#574 ``GradientOptions`` had no inline fields at all, so an SCF that ran
# with inline ECPs produced a gradient that took the bare-Z all-electron
# branch: wrong nuclear charges in the nuclear-repulsion and V_ne derivatives,
# no dV_ECP/dR term, and an occupied block sized with the full electron count
# the SCF never filled. Measured on the probe below: analytic +0.33306 vs
# FD +0.17223 Ha/bohr, i.e. ~1.9x the true force, matching the ~0.1 Ha/bohr
# scale docs/user_guide/ecp.md records for this mismatch class.

# vDZP carries an ncore=2 ECP on carbon, so methane exercises the inline path
# cheaply. The carbon is displaced off the tetrahedral centre on purpose: at
# exact Td symmetry dE/dz on carbon vanishes by symmetry and would agree
# between the two Hamiltonians, hiding the defect.
_VDZP_CH_BOND = 1.19
_VDZP_H_POS = (
    (_VDZP_CH_BOND, _VDZP_CH_BOND, _VDZP_CH_BOND),
    (-_VDZP_CH_BOND, -_VDZP_CH_BOND, _VDZP_CH_BOND),
    (-_VDZP_CH_BOND, _VDZP_CH_BOND, -_VDZP_CH_BOND),
    (_VDZP_CH_BOND, -_VDZP_CH_BOND, -_VDZP_CH_BOND),
)
_VDZP_C_Z0 = 0.30  # bohr off the tetrahedral centre -> non-zero dE/dz on C


def _ch4_inline(z_c: float = _VDZP_C_Z0):
    atoms = [Atom(6, [0.0, 0.0, z_c])]
    atoms += [Atom(1, list(p)) for p in _VDZP_H_POS]
    return Molecule(atoms, 0, 1)


def _scf_inline(z_c: float = _VDZP_C_Z0):
    """run_rhf on vDZP; the wrapper auto-attaches the inline ECP blocks."""
    mol = _ch4_inline(z_c)
    basis = BasisSet(mol, "vdzp")
    opts = RHFOptions()
    opts.max_iter = 300
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return mol, basis, opts, run_rhf(mol, basis, opts)


def _inline_gradient_options(scf_opts) -> GradientOptions:
    go = GradientOptions()
    go.ecp_primitive_blocks = list(scf_opts.ecp_primitive_blocks)
    go.ecp_primitive_centers = [
        list(c) for c in scf_opts.ecp_primitive_centers
    ]
    go.ecp_effective_charges = list(scf_opts.ecp_effective_charges)
    go.ecp_total_ncore = int(scf_opts.ecp_total_ncore)
    return go


def test_inline_ecp_scf_actually_takes_the_primitive_path():
    """Premise guard. If vDZP ever gains a bundled XML library the probe
    would silently switch to the ecp_centers path and the regression below
    would stop testing anything."""
    _, _, opts, _ = _scf_inline()
    assert not list(opts.ecp_centers), (
        "vDZP unexpectedly resolved to XML-library ECP centres; the inline "
        "regression below no longer exercises the primitive path"
    )
    assert len(opts.ecp_primitive_blocks) == 1, opts.ecp_primitive_blocks
    assert opts.ecp_total_ncore == 2, opts.ecp_total_ncore
    # per-ATOM effective charges: C keeps Z_eff = 6 - 2, the four H bare.
    assert list(opts.ecp_effective_charges) == [4.0, 1.0, 1.0, 1.0, 1.0]


def test_inline_ecp_analytic_gradient_matches_finite_difference():
    """#574 regression: with the inline fields threaded onto
    ``GradientOptions``, the analytic gradient agrees with central FD on the
    same ECP-aware energy. Pre-fix this disagreed by 0.16 Ha/bohr, so the
    1e-5 threshold has four orders of magnitude of headroom."""
    mol, basis, opts, res = _scf_inline()
    go = _inline_gradient_options(opts)
    g_analytic = np.asarray(compute_gradient(mol, basis, res, go))

    h = 5e-4
    e_plus = _scf_inline(_VDZP_C_Z0 + h)[3].energy
    e_minus = _scf_inline(_VDZP_C_Z0 - h)[3].energy
    g_fd_c_z = (e_plus - e_minus) / (2 * h)

    diff = abs(g_analytic[0, 2] - g_fd_c_z)
    assert diff < 1e-5, (
        f"inline-ECP analytic vs FD on CH4/vDZP: |Δ|={diff:.3e} Ha/bohr "
        f"(analytic {g_analytic[0, 2]:.7f}, FD {g_fd_c_z:.7f}); #574 "
        f"regression target ≤ 1e-5"
    )


def test_gradient_options_carry_inline_ecp_into_the_compute_path():
    """An inline-ECP reference cannot silently fall back to bare-Z."""
    mol, basis, opts, res = _scf_inline()

    g_with = np.asarray(
        compute_gradient(mol, basis, res, _inline_gradient_options(opts))
    )
    assert np.all(np.isfinite(g_with))

    # Inline fields left empty: the exact provenance contract rejects the
    # pre-#574 bare-Z fallback surface.
    with pytest.raises(
        ValueError,
        match="GradientOptions ECP route does not match",
    ):
        compute_gradient(mol, basis, res, GradientOptions())


def test_inline_ecp_gradient_contribution_is_nonzero_and_zero_without_ecp():
    """``compute_ecp_gradient_contribution_from_primitives`` smoke: the
    dV_ECP/dR term is real on an ECP-bearing atom, and an empty block list
    returns a zero matrix."""
    mol, basis, opts, res = _scf_inline()
    D = np.asarray(res.density)

    g_ecp = np.asarray(compute_ecp_gradient_contribution_from_primitives(
        basis, mol, [list(c) for c in opts.ecp_primitive_centers],
        list(opts.ecp_primitive_blocks), D,
    ))
    assert g_ecp.shape == (5, 3), g_ecp.shape
    assert abs(g_ecp[0, 2]) > 1e-4, g_ecp

    g_none = np.asarray(compute_ecp_gradient_contribution_from_primitives(
        basis, mol, [], [], D,
    ))
    assert np.allclose(g_none, 0.0), g_none


# ---------------------------------------------------------------------------
# FD Hessian on ECP systems (#576)
# ---------------------------------------------------------------------------
#
# ``compute_hessian_fd`` finite-differences the analytic gradient. Pre-fix it
# (a) default-constructed ``GradientOptions`` -- the bare-Z all-electron
# gradient -- whenever the caller passed none, which the molecular runner's
# frequency path and ``run_irc``'s TS Hessian always did, and (b) left the
# ECP centres, which are ABSOLUTE coordinates on the SCF options
# (``ecp_centers[i].xyz``, ``ecp_primitive_centers[i]``), where they were
# while it moved the nucleus. cpp/src/ecp.cpp attaches a centre to an atom
# only within 1e-6 bohr, so at the first 0.005 bohr displacement of the ECP
# atom that atom reverted to bare Z while V_ECP kept acting: the SCF diverged
# (E = -7.4e10 Ha on [ZnH]+/6-31G/ecp10mdf, -1.3e13 Ha on CH4/vDZP) and
# ``compute_hessian_fd`` raised "SCF failed to converge". Displacing only an
# all-electron atom converged and silently produced the bare-Z column:
# d2E/dz_H^2 on [ZnH]+ came out 0.6473 instead of 0.1186 Ha/bohr^2 (5.5x).
#
# Post-fix the Hessian element equals the second central difference of the
# ECP-aware SCF energy to 1.1e-6 ([ZnH]+) and 4.1e-6 (CH4/vDZP) Ha/bohr^2,
# FD truncation at step 0.005 bohr. The 1e-4 tolerance leaves ~25x headroom
# over that and sits four orders of magnitude below the bare-Z error.

from vibeqc.ecp_metadata import (  # noqa: E402
    attach_inline_ecp_options_from_basis_sidecar,
    ecp_centre_atom_indices,
    ecp_centres_follow_atoms,
)
from vibeqc.gradient_options import gradient_options_from_scf  # noqa: E402
from vibeqc.hessian import HessianFDOptions, compute_hessian_fd  # noqa: E402

_FD_STEP = 0.005                # bohr, compute_hessian_fd's default
_HESSIAN_TOL = 1e-4             # Ha/bohr^2, see the section comment
_ZN_H_BOND = 3.0                # bohr, the H position in _zn_h()


def _second_derivative(energy_at, x0: float, h: float = _FD_STEP) -> float:
    return (energy_at(x0 + h) - 2.0 * energy_at(x0) + energy_at(x0 - h)) / h**2


def _inline_scf_options_without_scf():
    """CH4/vDZP inline ECP options as the drivers attach them, no SCF."""
    mol = _ch4_inline()
    opts = RHFOptions()
    attach_inline_ecp_options_from_basis_sidecar(
        opts, mol, BasisSet(mol, "vdzp")
    )
    assert opts.ecp_primitive_blocks and not list(opts.ecp_centers)
    return mol, opts


def test_gradient_options_from_scf_mirrors_xml_ecp_route():
    """The shared helper copies the XML-library route and clears the inline
    one; a plain all-electron options struct yields the bare default."""
    mol = _zn_h()
    opts = RHFOptions()
    opts.ecp_centers = [ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts.ecp_library = "ecp10mdf"
    go = gradient_options_from_scf(opts)
    assert [(c.Z, list(c.xyz)) for c in go.ecp_centers] == [(30, [0.0, 0.0, 0.0])]
    assert go.ecp_library == "ecp10mdf"
    assert not list(go.ecp_primitive_blocks) and go.ecp_total_ncore == 0

    go_ae = gradient_options_from_scf(RHFOptions())
    assert not list(go_ae.ecp_centers) and go_ae.ecp_library == ""
    assert not list(go_ae.ecp_primitive_blocks)
    del mol


def test_gradient_options_from_scf_mirrors_inline_ecp_route():
    _, opts = _inline_scf_options_without_scf()
    go = gradient_options_from_scf(opts)
    assert len(go.ecp_primitive_blocks) == 1
    assert [list(c) for c in go.ecp_primitive_centers] == [
        list(c) for c in opts.ecp_primitive_centers
    ]
    assert list(go.ecp_effective_charges) == [4.0, 1.0, 1.0, 1.0, 1.0]
    assert go.ecp_total_ncore == 2
    assert not list(go.ecp_centers) and go.ecp_library == ""


def test_ecp_centres_follow_atoms_repositions_and_restores_both_routes():
    """Inside the block the centres sit on the displaced atoms (and a given
    GradientOptions is synchronised to them); on exit every field is back,
    by value -- pybind hands the list elements out by reference, so a naive
    saved list would read the moved coordinates."""
    # XML route, [ZnH]+: move Zn by +0.005 along z.
    ref = _zn_h()
    disp = _zn_h(z_zn=0.005)
    opts = RHFOptions()
    opts.ecp_centers = [ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts.ecp_library = "ecp10mdf"
    go = GradientOptions()
    assert ecp_centre_atom_indices(opts, ref) == ([0], [])
    with ecp_centres_follow_atoms(opts, ref, disp, go):
        assert list(opts.ecp_centers[0].xyz) == [0.0, 0.0, 0.005]
        assert list(go.ecp_centers[0].xyz) == [0.0, 0.0, 0.005]
        assert go.ecp_library == "ecp10mdf"
    assert list(opts.ecp_centers[0].xyz) == [0.0, 0.0, 0.0]
    assert not list(go.ecp_centers) and go.ecp_library == ""

    # Inline route, CH4/vDZP: move C by +0.005 along z.
    ref_i, opts_i = _inline_scf_options_without_scf()
    disp_i = _ch4_inline(_VDZP_C_Z0 + 0.005)
    assert ecp_centre_atom_indices(opts_i, ref_i) == ([], [0])
    with ecp_centres_follow_atoms(opts_i, ref_i, disp_i):
        assert [list(c) for c in opts_i.ecp_primitive_centers] == [
            [0.0, 0.0, _VDZP_C_Z0 + 0.005]
        ]
        assert len(opts_i.ecp_primitive_blocks) == 1
    assert [list(c) for c in opts_i.ecp_primitive_centers] == [
        [0.0, 0.0, _VDZP_C_Z0]
    ]
    assert len(opts_i.ecp_primitive_blocks) == 1
    assert list(opts_i.ecp_effective_charges) == [4.0, 1.0, 1.0, 1.0, 1.0]
    assert opts_i.ecp_total_ncore == 2

    # All-electron options: a no-op.
    ae = RHFOptions()
    with ecp_centres_follow_atoms(ae, ref, disp):
        assert not list(ae.ecp_centers)


def test_fd_hessian_refuses_ecp_centres_that_sit_on_no_atom(monkeypatch):
    """A centre off every nucleus means the SCF already runs an inconsistent
    Hamiltonian (bare Z on the atom, V_ECP still acting). Fail closed before
    the first SCF instead of differentiating it."""
    import vibeqc as public_api

    def no_scf(*args, **kwargs):
        raise AssertionError("no SCF may run for a rejected ECP setup")

    monkeypatch.setattr(public_api, "run_rhf", no_scf)
    mol = _zn_h()
    opts = RHFOptions()
    opts.ecp_centers = [ECPCenter(Z=30, xyz=[0.0, 0.0, 0.01])]  # 0.01 bohr off Zn
    opts.ecp_library = "ecp10mdf"
    with pytest.raises(ValueError, match="coincides with no atom"):
        compute_hessian_fd(mol, "6-31g", method="RHF", scf_options=opts)


def test_fd_hessian_default_sidecar_attaches_before_first_displacement():
    """A basis-only direct call must establish ECP provenance up front."""
    mol = Molecule(
        [
            Atom(16, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.815, 1.425]),
            Atom(1, [0.0, -1.815, 1.425]),
        ]
    )
    hres = compute_hessian_fd(
        mol,
        "lanl2dz",
        method="RHF",
        hessian_options=HessianFDOptions(
            step_bohr=0.002,
            frozen_indices=[1, 2],
        ),
    )

    assert hres.hessian.shape == (3, 3)
    assert np.all(np.isfinite(hres.hessian))
    assert float(hres.hessian[2, 2]) > 0.1


def test_fd_hessian_xml_ecp_matches_energy_second_derivative():
    """#576 regression, XML-library route: displace the ECP atom itself.
    Pre-fix the SCF at the first displaced geometry diverged (centre left
    behind) and compute_hessian_fd raised. Post-fix d2E/dz_Zn^2 from the
    FD-of-gradient Hessian equals the second difference of the ECP-aware
    energy (measured 1.1e-6 Ha/bohr^2 apart), with no gradient_options
    passed -- the runner's and run_irc's call shape."""
    mol = _zn_h()
    _, opts, _ = _scf(mol)
    hres = compute_hessian_fd(
        mol, "6-31g", method="RHF", scf_options=opts,
        hessian_options=HessianFDOptions(
            step_bohr=_FD_STEP, frozen_indices=[1]),   # displace Zn only
    )
    # The SCF options are handed back exactly as given.
    assert list(opts.ecp_centers[0].xyz) == [0.0, 0.0, 0.0]
    assert opts.ecp_library == "ecp10mdf"

    def energy_at(z_zn: float) -> float:
        return _scf(_zn_h(z_zn=z_zn), z_zn=z_zn)[2].energy

    d2e = _second_derivative(energy_at, 0.0)
    h_zz = float(hres.hessian[2, 2])
    assert abs(h_zz - d2e) < _HESSIAN_TOL, (
        f"FD Hessian d2E/dz_Zn^2 = {h_zz:.7f} vs energy second difference "
        f"{d2e:.7f} Ha/bohr^2 (|delta| = {abs(h_zz - d2e):.2e})"
    )
    assert h_zz > 0.05  # bound stretch: a real positive force constant


def test_fd_hessian_xml_ecp_all_electron_column_is_not_bare_z():
    """#576 regression, the silent surface: displace only the all-electron H
    (the centre stays on Zn, so the pre-fix SCF converged) and the pre-fix
    column was the bare-Z gradient's, 0.6473 instead of 0.1186 Ha/bohr^2.
    The fixed column matches the energy second difference. Direct bare-Z
    fallback is independently excluded by the exact provenance tests."""
    mol = _zn_h()
    _, opts, _ = _scf(mol)
    hres = compute_hessian_fd(
        mol, "6-31g", method="RHF", scf_options=opts,
        hessian_options=HessianFDOptions(
            step_bohr=_FD_STEP, frozen_indices=[0]),   # displace H only
    )

    def energy_at(z_h: float) -> float:
        return _scf(_zn_h(z_h=z_h))[2].energy

    d2e = _second_derivative(energy_at, _ZN_H_BOND)
    h_zz = float(hres.hessian[2, 2])
    assert abs(h_zz - d2e) < _HESSIAN_TOL, (
        f"FD Hessian d2E/dz_H^2 = {h_zz:.7f} vs energy second difference "
        f"{d2e:.7f} Ha/bohr^2 (|delta| = {abs(h_zz - d2e):.2e})"
    )


def test_fd_hessian_inline_ecp_matches_energy_second_derivative():
    """#576 regression, inline-primitive route (vDZP): displace the ECP atom
    (carbon, off the Td centre so the surface is not flat by symmetry).
    Pre-fix the SCF diverged at the first displacement; post-fix the
    Hessian element equals the energy second difference (measured 4.1e-6
    Ha/bohr^2 apart) with no gradient_options passed."""
    mol, _, opts, _ = _scf_inline()
    hres = compute_hessian_fd(
        mol, "vdzp", method="RHF", scf_options=opts,
        hessian_options=HessianFDOptions(
            step_bohr=_FD_STEP, frozen_indices=[1, 2, 3, 4]),   # displace C only
    )
    assert [list(c) for c in opts.ecp_primitive_centers] == [[0.0, 0.0, _VDZP_C_Z0]]

    def energy_at(z_c: float) -> float:
        return _scf_inline(z_c)[3].energy

    d2e = _second_derivative(energy_at, _VDZP_C_Z0)
    h_zz = float(hres.hessian[2, 2])
    assert abs(h_zz - d2e) < _HESSIAN_TOL, (
        f"inline-ECP FD Hessian d2E/dz_C^2 = {h_zz:.7f} vs energy second "
        f"difference {d2e:.7f} Ha/bohr^2 (|delta| = {abs(h_zz - d2e):.2e})"
    )
    assert h_zz > 0.1


# ---------------------------------------------------------------------------
# Geometry-moving drivers follow the ECP centres (#643)
# ---------------------------------------------------------------------------
#
# The shared image evaluator behind run_dimer / run_irc (neb._evaluate_image)
# and the central-difference gradient helper rebuild the molecule at new
# positions but reused the caller's SCF options unchanged, so the ECP centres
# stayed at the geometry they were built for. Pre-fix the image evaluator
# additionally ran its own all-electron warm-start Hamiltonian (Hcore with
# bare Z, full electron count) that never read the ECP fields at all:
# [ZnH]+/6-31G/ecp10mdf at the template geometry came out at E = -1777.748 Ha
# instead of -224.1265 Ha. Post-fix a displaced image equals a fresh SCF +
# gradient built for that geometry to machine precision.

from vibeqc.irc import run_irc  # noqa: E402
from vibeqc.molecular_optimize import (  # noqa: E402
    _gradient_via_central_difference,
)
from vibeqc.neb import _evaluate_image  # noqa: E402


def _fresh_ecp_energy_and_gradient(mol, z_zn):
    """SCF + ECP-aware analytic gradient with the centre rebuilt for ``mol``."""
    basis, opts, res = _scf(mol, z_zn=z_zn)
    g = np.asarray(compute_gradient(mol, basis, res, gradient_options_from_scf(opts)))
    return res.energy, g


def test_image_evaluator_follows_xml_ecp_centre_and_restores_options():
    """#643 regression, XML route: the image at Zn displaced 0.1 bohr from the
    template equals a fresh calculation whose centre sits on the displaced Zn,
    and the caller's options come back describing the template."""
    template = _zn_h()
    _, opts, _ = _scf(template)
    positions = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, _ZN_H_BOND]])
    e_img, g_img, _ = _evaluate_image(
        positions, template, "6-31g", "rhf", functional=None,
        rhf_options=opts, uhf_options=None, rks_options=None, uks_options=None,
        gradient_options=None, grid_options=None, dispersion_params=None,
    )
    e_ref, g_ref = _fresh_ecp_energy_and_gradient(_zn_h(z_zn=0.1), 0.1)
    assert abs(e_img - e_ref) < 1e-9, (e_img, e_ref)
    assert np.abs(np.asarray(g_img) - g_ref).max() < 1e-8
    assert list(opts.ecp_centers[0].xyz) == [0.0, 0.0, 0.0]
    assert opts.ecp_library == "ecp10mdf"
    # the ECP-aware surface, not the all-electron warm-start one
    assert -230.0 < e_img < -220.0, e_img


def test_image_evaluator_follows_inline_ecp_centre():
    """#643 regression, inline route (CH4/vDZP): carbon displaced 0.05 bohr."""
    template, _, opts, _ = _scf_inline()
    z_c = _VDZP_C_Z0 + 0.05
    positions = np.array([[0.0, 0.0, z_c]] + [list(p) for p in _VDZP_H_POS])
    e_img, g_img, _ = _evaluate_image(
        positions, template, "vdzp", "rhf", functional=None,
        rhf_options=opts, uhf_options=None, rks_options=None, uks_options=None,
        gradient_options=None, grid_options=None, dispersion_params=None,
    )
    mol_ref, basis_ref, opts_ref, res_ref = _scf_inline(z_c)
    g_ref = np.asarray(compute_gradient(
        mol_ref, basis_ref, res_ref, _inline_gradient_options(opts_ref)))
    assert abs(e_img - res_ref.energy) < 1e-9, (e_img, res_ref.energy)
    assert np.abs(np.asarray(g_img) - g_ref).max() < 1e-8
    assert [list(c) for c in opts.ecp_primitive_centers] == [[0.0, 0.0, _VDZP_C_Z0]]


def test_image_evaluator_refuses_ecp_centres_off_the_template():
    template = _zn_h()
    opts = RHFOptions()
    opts.ecp_centers = [ECPCenter(Z=30, xyz=[0.0, 0.0, 0.02])]  # not on Zn
    opts.ecp_library = "ecp10mdf"
    positions = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, _ZN_H_BOND]])
    with pytest.raises(ValueError, match="coincides with no atom"):
        _evaluate_image(
            positions, template, "6-31g", "rhf", functional=None,
            rhf_options=opts, uhf_options=None, rks_options=None,
            uks_options=None, gradient_options=None, grid_options=None,
            dispersion_params=None,
        )


def test_run_irc_default_sidecar_uses_the_ecp_energy_surface():
    """A direct IRC call must attach its paired ECP with no option object.

    H2S/LANL2DZ is about -10.99 Ha on the ECP surface; rebuilding bare-Z
    attraction with the same orbital basis instead gives about -89.38 Ha.
    ``max_points=0`` keeps this to the transition-state force evaluation.
    """
    transition_state = Molecule(
        [
            Atom(16, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.8, 1.3]),
            Atom(1, [0.0, -1.8, 1.3]),
        ]
    )
    basis = BasisSet(transition_state, "lanl2dz")
    reference = run_rhf(transition_state, basis)
    assert reference.ecp_operator_applied
    assert reference.ecp_total_ncore == 10

    mode = np.zeros((3, 3))
    mode[1, 2] = 1.0
    result = run_irc(
        transition_state,
        basis="lanl2dz",
        method="RHF",
        rhf_options=None,
        transition_mode=mode,
        direction="forward",
        max_points=0,
    )

    assert result.ts_energy == pytest.approx(reference.energy, abs=1e-10)
    assert -12.0 < result.ts_energy < -10.0


def test_central_difference_gradient_follows_ecp_centres():
    """The FD gradient helper (native optimiser FD paths, the ASE #571
    fallback) moves the centre with each displaced atom: it must reproduce
    the ECP-aware analytic gradient, not diverge at the first displacement."""
    mol = _zn_h()
    basis, opts, res = _scf(mol)
    g_analytic = np.asarray(compute_gradient(mol, basis, res, gradient_options_from_scf(opts)))
    g_fd = np.asarray(_gradient_via_central_difference(
        mol, "6-31g", "rhf", rhf_options=opts, step_bohr=5e-4))
    assert np.abs(g_fd - g_analytic).max() < 1e-5, (g_fd, g_analytic)
    assert list(opts.ecp_centers[0].xyz) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# GitLab #642: inline-primitive route -- Z_eff in the dipole and in the FD
# Hessian's dipole-derivative rows
# ---------------------------------------------------------------------------


def test_effective_nuclear_charges_inline_route_is_per_atom():
    """CH4/vDZP attaches an inline ECP on carbon only: Z_eff = [4, 1, 1, 1, 1]
    in atom order, taken from the per-atom vector the sidecar attach builds."""
    import numpy as np
    from vibeqc.ecp_metadata import effective_nuclear_charges

    mol, opts = _inline_scf_options_without_scf()
    np.testing.assert_allclose(effective_nuclear_charges(mol, opts), [4.0, 1.0, 1.0, 1.0, 1.0])


def test_inline_ecp_dipole_is_origin_independent():
    """Inline route counterpart of the XML-route test in test_ecp_scf.py: a
    neutral molecule's dipole must not move with the origin; bare Z moved it
    by -n_core(C) = -2 au per bohr (#642)."""
    import numpy as np
    from vibeqc.ecp_metadata import effective_nuclear_charges
    from vibeqc.properties import center_of_mass, dipole_moment

    mol, basis, opts, res = _scf_inline()
    z_eff = effective_nuclear_charges(mol, opts)
    com = center_of_mass(mol)
    d0 = dipole_moment(res, basis, mol, nuclear_charges=z_eff)
    d1 = dipole_moment(
        res,
        basis,
        mol,
        origin=com + np.array([1.0, 0.0, 0.0]),
        nuclear_charges=z_eff,
    )
    assert abs(d1.x - d0.x) < 1e-8
    b0 = dipole_moment(res, basis, mol)
    b1 = dipole_moment(res, basis, mol, origin=com + np.array([1.0, 0.0, 0.0]))
    assert (b1.x - b0.x) == pytest.approx(-2.0, abs=1e-8)


def test_fd_hessian_dipole_derivatives_obey_the_translational_sum_rule():
    """GitLab #642 closure criterion for IR intensities. For a neutral
    molecule the dipole derivatives satisfy sum_A d mu_i / d R_A,j = 0
    (a rigid translation does not change the dipole). With bare Z on the
    ECP atom each of its three rows carried a spurious n_core = 2 on the
    diagonal, so the sum was 2 * I and every IR intensity of a mode that
    moves the carbon was wrong. The FD Hessian now resolves Z_eff once from
    the SCF options it differentiates (#576) and uses it in every displaced
    dipole (#642)."""
    import numpy as np
    from vibeqc.hessian import ir_intensities

    mol, basis, opts, _ = _scf_inline()
    hres = compute_hessian_fd(
        mol, "vdzp", method="RHF", scf_options=opts,
        hessian_options=HessianFDOptions(include_dipole_derivatives=True),
    )
    dd = np.asarray(hres.dipole_derivatives).reshape(len(mol.atoms), 3, 3)
    total = dd.sum(axis=0)  # sum over atoms -> (3, 3)
    assert np.abs(total).max() < 1e-4, total
    ir = np.asarray(ir_intensities(hres))
    assert np.all(np.isfinite(ir)) and ir.min() >= 0.0
    # Measured post-fix on this fixture (OMP_NUM_THREADS=4, main d72bc3fc7):
    # the strongest stretch is 87.5 km/mol; pre-fix the same mode read 89.89.
    assert ir.max() == pytest.approx(87.49, abs=0.5)
