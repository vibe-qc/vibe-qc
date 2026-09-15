"""CIS / TDA excited-state finite-difference nuclear gradients (vibeqc.excited_gradient).

Validates the reference-agnostic excited-state gradient driver and its state
tracker on both backends of the ERIProvider seam:

* state tracking follows a CIS root through reordering + sign flips;
* the HF excited-state gradient equals an independent central difference of the
  tracked total energy (the task's FD-gradient-vs-FD-energy check), is
  translationally invariant, and points along the bond of a diatomic;
* the MSINDO excited-state gradient runs and is translationally invariant, and
  its *ground-state* (state 0) gradient reproduces the existing MSINDO
  ground-state FD gradient to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.excited_gradient import (
    CISStateSet,
    cis_state_gradients_fd,
    cis_states_energy_and_gradients_fd,
    make_hf_cis_energy_fn,
    track_state,
)


# --------------------------------------------------------------------------- #
# State tracking (root-flip guard)                                            #
# --------------------------------------------------------------------------- #


def test_track_state_follows_reordered_and_sign_flipped_root():
    """The tracker follows amplitude *character*, not the eigenvalue index, and
    is invariant to the arbitrary global sign of an eigenvector."""
    ref = np.array([0.0, 1.0, 0.0, 0.0])
    # Columns: root 0 = e0, root 1 = e2, root 2 = -e1 (the reference, flipped).
    amps = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    j, ov = track_state(ref, amps)
    assert j == 2
    assert ov == pytest.approx(1.0)


def test_track_state_picks_largest_overlap_when_mixed():
    amps = np.array([[0.8, -0.6], [0.6, 0.8]])    # columns are orthonormal
    ref = amps[:, 1].copy()                       # ref == column 1 exactly
    j, ov = track_state(ref, amps)
    assert j == 1
    assert ov == pytest.approx(1.0)
    # A reference closer to column 0 selects column 0.
    j0, _ = track_state(np.array([0.6, 0.8]), amps)
    assert j0 == 0                                 # overlap 0.96 > 0.28


# --------------------------------------------------------------------------- #
# HF excited-state gradient                                                   #
# --------------------------------------------------------------------------- #


def _hf_molecule_fn(state=1, n_states=5):
    """HF (the molecule) / STO-3G singlet CIS energy function + S1 geometry."""
    fn = make_hf_cis_energy_fn([9, 1], "sto-3g", spin="singlet", n_states=n_states)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.95]])   # Angstrom, along z
    return fn, coords


def _independent_cis_total_energy_cd(fn, coords, state, step):
    """A from-scratch central difference of the tracked CIS *total* energy
    E_SCF + ω, used to cross-check the library gradient by a different code
    path (manual tracking + separate SCF/excitation differencing)."""
    C0 = np.asarray(coords, float)
    ref = fn(C0)
    refvec = ref.amplitudes[:, state - 1].copy()
    A2B = 1.8897259886
    g = np.zeros_like(C0)
    for i in range(C0.shape[0]):
        for d in range(3):
            cp = C0.copy(); cp[i, d] += step
            cm = C0.copy(); cm[i, d] -= step
            sp, sm = fn(cp), fn(cm)
            jp, _ = track_state(refvec, sp.amplitudes)
            jm, _ = track_state(refvec, sm.amplitudes)
            # Difference E_SCF and ω separately, then sum (≠ the library's
            # single-shot difference of the total — same value, other grouping).
            d_scf = (sp.e_ground - sm.e_ground) / (2.0 * step) / A2B
            d_omega = (sp.excitation_energies[jp]
                       - sm.excitation_energies[jm]) / (2.0 * step) / A2B
            g[i, d] = d_scf + d_omega
    return g


def test_hf_cis_gradient_matches_independent_central_difference():
    """The library S1 gradient equals an independent central difference of the
    tracked CIS total energy (FD-gradient-vs-FD-energy)."""
    fn, coords = _hf_molecule_fn(state=1)
    step = 2e-3
    g_lib = cis_state_gradients_fd(fn, coords, 1, step=step)[1]
    g_ref = _independent_cis_total_energy_cd(fn, coords, 1, step)
    np.testing.assert_allclose(g_lib, g_ref, atol=1e-9)


def test_hf_cis_gradient_translationally_invariant():
    """A rigid translation does not change the energy, so the FD excited-state
    gradient sums to ~zero over the atoms."""
    fn, coords = _hf_molecule_fn(state=1)
    g = cis_state_gradients_fd(fn, coords, 1, step=2e-3)[1]
    np.testing.assert_allclose(g.sum(axis=0), np.zeros(3), atol=1e-6)


def test_hf_cis_diatomic_gradient_lies_along_bond():
    """For a diatomic on the z-axis the S1 gradient has no x/y component and is
    equal-and-opposite along z (Newton's third law on a 2-body energy)."""
    fn, coords = _hf_molecule_fn(state=1)
    g = cis_state_gradients_fd(fn, coords, 1, step=2e-3)
    g1 = g[1]
    assert np.all(np.abs(g1[:, :2]) < 1e-6)            # no transverse force
    assert g1[0, 2] == pytest.approx(-g1[1, 2], abs=1e-6)
    assert abs(g1[0, 2]) > 1e-3                        # the bond is not relaxed


def test_hf_ground_state_gradient_is_well_defined():
    """state=0 differences only E_SCF (no excitation), so it is the plain RHF
    gradient — finite, transverse-free, equal-and-opposite for the diatomic."""
    fn, coords = _hf_molecule_fn(state=0)
    g0 = cis_state_gradients_fd(fn, coords, 0, step=2e-3)[0]
    assert np.all(np.abs(g0[:, :2]) < 1e-6)
    assert g0[0, 2] == pytest.approx(-g0[1, 2], abs=1e-6)


def test_energy_and_gradient_helper_is_consistent():
    """cis_states_energy_and_gradients_fd returns the center total energies and
    the same gradients as cis_state_gradients_fd."""
    fn, coords = _hf_molecule_fn()
    energies, grads = cis_states_energy_and_gradients_fd(fn, coords, [0, 1], step=2e-3)
    center = fn(coords)
    assert energies[0] == pytest.approx(center.total_energy(0))
    assert energies[1] == pytest.approx(center.total_energy(1))
    assert energies[1] > energies[0]                  # excited state is higher
    g_only = cis_state_gradients_fd(fn, coords, [0, 1], step=2e-3)
    np.testing.assert_allclose(grads[1], g_only[1], atol=1e-12)


# --------------------------------------------------------------------------- #
# MSINDO excited-state gradient                                               #
# --------------------------------------------------------------------------- #


def _h2o():
    Z = [8, 1, 1]
    xyz = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]])
    return Z, xyz


def test_msindo_cis_gradient_runs_and_translationally_invariant():
    from vibeqc.semiempirical.methods.msindo import msindo_cis_gradient_fd
    Z, xyz = _h2o()
    g = msindo_cis_gradient_fd(Z, xyz, 1, spin="singlet", step=1e-3)
    assert g.shape == (3, 3)
    assert np.isfinite(g).all()
    np.testing.assert_allclose(g.sum(axis=0), np.zeros(3), atol=1e-5)
    assert np.linalg.norm(g) > 1e-4                   # S1 is not at its minimum


def test_msindo_state0_gradient_matches_ground_state_fd():
    """The CIS-path ground-state (state 0) gradient is the existing MSINDO
    ground-state FD gradient — E_ground in the CIS state-set equals
    run_msindo's total energy, so the two FD gradients coincide."""
    from vibeqc.semiempirical.methods.msindo import (
        msindo_cis_gradient_fd,
        msindo_gradient_fd,
    )
    Z, xyz = _h2o()
    g0 = msindo_cis_gradient_fd(Z, xyz, 0, step=1e-3, conv_tol=1e-10)
    g_ref = msindo_gradient_fd(Z, xyz, step=1e-3, conv_tol=1e-10)
    np.testing.assert_allclose(g0, g_ref, atol=1e-9)


def test_msindo_cis_stateset_total_energy():
    """The MSINDO state-set carries E_SCF and the excitation, so S1 total energy
    exceeds the ground-state energy by exactly ω_1."""
    from vibeqc.semiempirical.methods.msindo import msindo_cis_stateset
    Z, xyz = _h2o()
    ss = msindo_cis_stateset(Z, xyz, spin="singlet", n_states=4)
    assert isinstance(ss, CISStateSet)
    assert ss.total_energy(1) - ss.total_energy(0) == pytest.approx(
        float(ss.excitation_energies[0]))


# --------------------------------------------------------------------------- #
# run_job(tddft_gradient=True): which surface is differentiated (issue #570)  #
# --------------------------------------------------------------------------- #


def _h2_run_job(tmp_path, tag, **kw):
    """H2/STO-3G, one TDA root, through run_job; artefacts under tmp_path."""
    import vibeqc as vq

    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, -0.7]), vq.Atom(1, [0.0, 0.0, 0.7])])
    return vq.run_job(
        mol,
        basis="sto-3g",
        tddft=True,
        tddft_n_states=1,
        output=tmp_path / tag,
        citations=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        **kw,
    )


def _parse_gradient_rows(out: str, natom: int) -> np.ndarray:
    """The ``Atom  dE/dx  dE/dy  dE/dz`` rows of the excited-state gradient block."""
    lines = out.splitlines()
    for k, line in enumerate(lines):
        if line.split()[:4] == ["Atom", "dE/dx", "dE/dy", "dE/dz"]:
            rows = [lines[k + 1 + i].split() for i in range(natom)]
            return np.array([[float(v) for v in r[1:4]] for r in rows])
    raise AssertionError("no excited-state gradient rows in the .out")


@pytest.mark.parametrize(
    ("kw", "missing"),
    [
        ({"method": "rks", "functional": "lda"}, "RKS"),
        ({"method": "rhf", "tddft_type": "casida"}, "Casida"),
    ],
    ids=["rks-tda", "rhf-casida"],
)
def test_run_job_tddft_gradient_refuses_a_surface_it_cannot_differentiate(
    tmp_path, kw, missing
):
    """Issue #570: run_job wires one excited-state energy function into the FD
    driver, the closed-shell HF + CIS (TDA) one.  Before the fix an RKS/TDA or
    an RHF/Casida run was silently handed that HF-CIS S1 gradient (on this
    deck 0.02279028 Ha/bohr, the RHF/TDA number) under a header that named
    no surface.  Such a run is now refused before any calculation, with the
    missing capability named."""
    with pytest.raises(NotImplementedError, match=missing):
        _h2_run_job(tmp_path, "refused", tddft_gradient=True, **kw)
    # Refused before the run started: no .out was written.
    assert not (tmp_path / "refused.out").exists()


def test_run_job_tddft_gradient_hf_cis_surface_is_labelled_and_unchanged(tmp_path):
    """The one wired surface keeps its numbers and now says what it is.

    The gradient rows must equal the direct driver
    (:func:`make_hf_cis_energy_fn` + :func:`cis_state_gradients_fd`) on the
    same geometry, and the block must name the reference, the excitation
    solver and the FD step."""
    _h2_run_job(tmp_path, "hfcis", method="rhf", tddft_gradient=True)
    out = (tmp_path / "hfcis.out").read_text()

    assert "Excited-state gradient (S₁, RHF/CIS(TDA) surface, FD, step=1e-3 Å)" in out
    assert "Surface: RHF ground state + CIS (TDA) excitation energy" in out

    fn = make_hf_cis_energy_fn([1, 1], "sto-3g", spin="singlet", n_states=1)
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    direct = cis_state_gradients_fd(fn, coords, states=[1])[1]
    assert _parse_gradient_rows(out, 2) == pytest.approx(direct, abs=1e-8)


def test_run_job_tddft_without_gradient_is_unchanged(tmp_path):
    """Negative control: the same RKS/TDA route with the feature off runs,
    prints its excitation table, and prints no gradient block."""
    r = _h2_run_job(tmp_path, "gradoff", method="rks", functional="lda", tddft_gradient=False)
    out = (tmp_path / "gradoff.out").read_text()
    assert r.converged
    assert "TD-DFT excited states" in out
    assert "Excited-state gradient" not in out
