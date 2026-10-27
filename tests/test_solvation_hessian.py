"""Frequencies in solvent: the Hessian of the in-solvent surface.

``compute_hessian_fd`` differentiates an analytic gradient at displaced
geometries. Before it knew about a solvent, ``run_job(hessian=True,
solvent=...)`` printed the gas-phase Hessian under a solvated SCF: every number
smooth and plausible, on the wrong surface (#359 was the same defect for
optimizations). These tests pin that it now differentiates the in-solvent
gradient, by the one oracle that needs no second implementation: the second
derivative of the energy the solvated job itself reports.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

ANG = 1.8897261254578281

# A water that is neither symmetric nor stationary. The identity used below,
# d^2E/dlambda^2 = v^T H v along a straight line, holds at any geometry; the
# lack of symmetry keeps the COSMO FINE cavity's frame tie out of the way.
_WATER_BOHR = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 1.598, -1.159)), (1, (0.0, -1.498, -1.159))]


def _water(vq):
    return vq.Molecule([vq.Atom(z, xyz) for z, xyz in _WATER_BOHR], 0, 1)


def _model(vq, **kw):
    """The default CPCM water model (302 Lebedev points per sphere), tightly converged.

    Not the 110-point grid some codes default to: at 110 points the cavity matrix
    A has an eigenvalue that crosses zero within a few thousandths of a bohr
    (the point-charge kernel is indefinite), the solvated energy jumps by up to
    7e-4 Ha between neighbouring geometries, and the analytic gradient is off by
    1.7e-2 Ha/bohr there. A finite-difference oracle cannot be trusted on it.
    """
    base = dict(epsilon=78.39, name="water", max_macro_iter=60, tol_e_solv=1e-11)
    base.update(kw)
    return vq.SolventModel(**base)


def _energy(vq, mol, sm):
    basis = vq.BasisSet(mol, "sto-3g")
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    return float(sol.energy)


def _moved(vq, mol, v, lam):
    return vq.Molecule(
        [vq.Atom(a.Z, tuple(np.asarray(a.xyz) + lam * v[i])) for i, a in enumerate(mol.atoms)],
        mol.charge, mol.multiplicity,
    )


def _curvature(vq, mol, v, sm):
    """d^2E/dlambda^2 of the solvated energy along ``v``: second differences at
    h = 0.01 and 0.005, Richardson-extrapolated."""
    e0 = _energy(vq, mol, sm)

    def second(step):
        return (_energy(vq, _moved(vq, mol, v, step), sm)
                + _energy(vq, _moved(vq, mol, v, -step), sm) - 2.0 * e0) / step**2

    return (4.0 * second(0.005) - second(0.01)) / 3.0


def test_the_solvated_hessian_is_the_second_derivative_of_the_solvated_energy():
    """The check that would have caught the defect.

    Along the two directions in which the solvent changes the curvature most
    (the leading eigenvectors of ``H_solv - H_gas``), the curvature of the
    *reported* in-solvent energy (second differences of ``run_cpcm_scf``,
    Richardson-extrapolated) must equal ``v^T H v`` of the Hessian the solvated
    call returns, and the solvent's effect must dwarf the disagreement:
    otherwise the test would pass for a Hessian that ignored the solvent.

    Measured, RHF/STO-3G, a distorted water: agreement 1.1e-05 and 5.0e-05
    Ha/bohr^2 against solvent effects of 7.5e-03 and 3.9e-03.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.hessian import compute_hessian_fd

    mol = _water(vq)
    sm = _model(vq)
    hs = compute_hessian_fd(mol, "sto-3g", method="RHF", solvent=sm)
    hg = compute_hessian_fd(mol, "sto-3g", method="RHF")
    assert hs.solvent is not None and "CPCM" in hs.solvent and hg.solvent is None

    w, U = np.linalg.eigh(hs.hessian - hg.hessian)
    for rank in range(2):
        v = U[:, np.argsort(-np.abs(w))[rank]].reshape(3, 3)
        flat = v.reshape(-1)
        h_solv, h_gas = flat @ hs.hessian @ flat, flat @ hg.hessian @ flat
        agreement = abs(_curvature(vq, mol, v, sm) - h_solv)
        effect = abs(h_solv - h_gas)
        assert agreement < SOLVATED_CURVATURE_AGREEMENT, (rank, agreement)
        assert effect > 10.0 * agreement, (rank, effect, agreement)


#: Measured agreement of the solvated Hessian with the curvature of the energy
#: the job reports, along the two directions above: 1.1e-05 and 5.0e-05. The
#: bound keeps four times the worse.
SOLVATED_CURVATURE_AGREEMENT = 2.0e-4


@pytest.mark.parametrize(
    "method, functional, charge, multiplicity, bound",
    [
        # Measured agreement 7.9e-08 (solvent effect 9.9e-04); the bounds keep
        # at least three times the measurement.
        ("RKS", "pbe", 0, 1, 5.0e-7),         # analytic KS gradient
        ("RKS", "cam-b3lyp", 0, 1, 5.0e-7),   # incomplete analytic gradient: full-energy route (#571)
        ("UHF", None, 1, 2, 1.0e-4),          # open shell; 2.8e-05 against an effect of 3.7e-02
    ],
    ids=["rks-pbe", "rks-cam-b3lyp-full-energy", "uhf-h2-cation"],
)
def test_the_solvent_changes_the_curvature_of_the_hessian_as_it_changes_the_energy(
    method, functional, charge, multiplicity, bound
):
    """Kohn-Sham, the full-energy gradient route and open shell, by differences.

    The analytic Kohn-Sham gradient on a molecular grid is not the derivative of
    the grid energy, and that is a gas-phase property with no solvent in it:
    H2/PBE/STO-3G, ``v^T H v`` is 4.7e-04 Ha/bohr^2 off the curvature of the
    energy with or without a solvent. So the absolute comparison of the test
    above is the wrong oracle here. The *change* the solvent makes cancels that
    error: along the bond, the curvature of the solvated energy minus the gas
    energy's, against the same difference of the Hessians. Measured for PBE:
    +9.898e-04 from the energies and +9.899e-04 from the Hessians, 8e-08 apart.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.hessian import compute_hessian_fd
    from vibeqc.runner import apply_ks_grid_default

    mol = vq.Molecule([vq.Atom(1, (0.0, 0.0, 0.0)), vq.Atom(1, (0.0, 0.0, 1.5))], charge, multiplicity)
    sm = _model(vq)
    v = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]) / math.sqrt(2.0)
    flat = v.reshape(-1)
    opts = {"RKS": vq.RKSOptions, "UHF": vq.UHFOptions}[method]()
    if functional is not None:
        opts.functional = functional
        apply_ks_grid_default(opts)

    def energy(m, solvated):
        basis = vq.BasisSet(m, "sto-3g")
        if solvated:
            sol = vq.run_cpcm_scf(m, basis, method=method.lower(), solvent=sm, options=opts)
            assert sol.converged
            return float(sol.energy)
        run = {"RKS": vq.run_rks, "UHF": vq.run_uhf}[method]
        return float(run(m, basis, opts).energy)

    def curvature(solvated):
        e0 = energy(mol, solvated)

        def second(step):
            return (energy(_moved(vq, mol, v, step), solvated)
                    + energy(_moved(vq, mol, v, -step), solvated) - 2.0 * e0) / step**2

        return (4.0 * second(0.005) - second(0.01)) / 3.0

    hs = compute_hessian_fd(mol, "sto-3g", method=method, scf_options=opts, solvent=sm).hessian
    hg = compute_hessian_fd(mol, "sto-3g", method=method, scf_options=opts).hessian
    from_energies = curvature(True) - curvature(False)
    from_hessians = flat @ hs @ flat - flat @ hg @ flat
    assert abs(from_energies - from_hessians) < bound, (from_energies, from_hessians)
    # The comparison must be able to fail: the solvent's effect dwarfs the bound.
    assert abs(from_hessians) > 20.0 * bound, from_hessians


def test_the_frame_tie_guard_warns_exactly_where_a_hessian_would_straddle_a_cusp():
    """The fine cavity in the molecule-fixed frame ties at a symmetric geometry,
    so a Hessian there is wrong in the modes that break the symmetry (measured on
    water: 4381 cm^-1 for the asymmetric stretch against 4426 on the Lebedev
    cavity and 4424 on the fine cavity in the lab frame). The guard has to fire
    there and nowhere else: not off the tie, not in the lab frame, not on the
    Lebedev cavity, and not for a planar solute merely because its own plane is
    a symmetry operation, which moves no atom and ties nothing."""
    vq = pytest.importorskip("vibeqc")
    from vibeqc.hessian import _warn_if_frame_tie

    ang = ANG
    symmetric = vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 0.75655 * ang, 0.63172 * ang)),
         vq.Atom(1, (0.0, -0.75655 * ang, 0.63172 * ang))], 0, 1)
    fine = dict(epsilon=78.39, cavity="fine")
    molecular = vq.SolventModel(fine_frame="molecular", **fine)

    with pytest.warns(UserWarning, match="frame ties"):
        assert _warn_if_frame_tie(molecular, symmetric, 0.005)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Off the tie: 0.1 bohr of asymmetry, far beyond the displacement.
        assert not _warn_if_frame_tie(molecular, _water(vq), 0.005)
        # No frame, no tie.
        assert not _warn_if_frame_tie(vq.SolventModel(fine_frame="lab", **fine), symmetric, 0.005)
        assert not _warn_if_frame_tie(vq.SolventModel(epsilon=78.39), symmetric, 0.005)


def test_a_vacuum_solvent_is_the_gas_phase_hessian():
    vq = pytest.importorskip("vibeqc")
    from vibeqc.hessian import compute_hessian_fd

    mol = _water(vq)
    gas = compute_hessian_fd(mol, "sto-3g", method="RHF")
    for spec in ("vacuum", 1.0, None):
        got = compute_hessian_fd(mol, "sto-3g", method="RHF", solvent=spec)
        assert got.solvent is None
        # Two gas-phase runs differ at 1e-14 from threaded summation order, so
        # the claim is the gas-phase surface, to roundoff, not bit for bit.
        np.testing.assert_allclose(got.hessian, gas.hessian, rtol=0, atol=1e-10)


def test_the_solvated_hessian_refuses_what_it_cannot_differentiate():
    vq = pytest.importorskip("vibeqc")
    from vibeqc.hessian import compute_hessian_fd
    from vibeqc.solvation import cosmors as rs
    from vibeqc.solvation.cosmors import KLAMT_1998

    mol = _water(vq)
    with pytest.raises(NotImplementedError, match="no solvated analytic gradient"):
        compute_hessian_fd(mol, "sto-3g", method="ROHF", solvent="water")
    with pytest.raises(ValueError, match="do not compose"):
        compute_hessian_fd(mol, "sto-3g", method="RHF", solvent="water", source=object())

    # Direct COSMO-RS: its energy has a cusp at a symmetric geometry that a
    # Hessian from gradient differences would straddle.
    desc = rs.SegmentDescriptors(
        areas=np.array([12.0, 12.0, 8.0]), sigma=np.array([-0.012, 0.012, 0.0]),
        sigma_perp=np.array([0.001, -0.001, 0.0]), sigma_raw=np.zeros(3),
    )
    pot = rs.sigma_potential_profile(rs.sigma_profile(desc, rs.default_sigma_grid(), label="toy"), KLAMT_1998)
    drs = vq.SolventModel(variant="dcosmo-rs", sigma_potential=pot, epsilon=math.inf,
                          cavity="fine", radii=dict(KLAMT_1998.cavity_radii),
                          radii_scale=1.0, solvent_probe_radius_ang=0.0)
    with pytest.raises(NotImplementedError, match="Direct COSMO-RS"):
        compute_hessian_fd(mol, "sto-3g", method="RHF", solvent=drs)


def _printed_frequencies(text):
    """The ``Freq/cm-1`` column of the mode table in a ``.out``."""
    return np.array([
        float(m.group(2))
        for m in re.finditer(r"^\s+(\d+)\s+(-?\d+\.\d)\s+(\d+\.\d+)\s*$", text, flags=re.M)
    ])


def test_run_job_frequencies_in_solvent_are_the_solvated_ones(tmp_path):
    """Through ``run_job``: the output names the medium, the manifest records it,
    and the printed frequencies are the in-solvent ones, not the gas-phase ones
    the job used to print under a solvated SCF.

    Measured: 2124.7, 3592.4, 4140.3 cm^-1 in water against 2128.3, 3588.0,
    4149.9 in the gas phase (RHF/STO-3G, this distorted water).
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.hessian import compute_hessian_fd

    mol = _water(vq)
    stem = tmp_path / "w_solv"
    vq.run_job(
        mol, basis="sto-3g", method="rhf", solvent=_model(vq), hessian=True,
        output=str(stem), progress=False, write_molden_file=False,
    )
    out = stem.with_suffix(".out").read_text()
    assert "Surface: RHF/sto-3g in CPCM water (eps = 78.39)" in out
    assert "Medium: CPCM water" in out and "equilibrium-solvation surface" in out
    assert "IR intensities are" in out
    manifest = stem.with_suffix(".system").read_text()
    assert 'solvent        = "CPCM water (eps = 78.39)"' in manifest

    solvated = compute_hessian_fd(mol, "sto-3g", method="RHF", solvent=_model(vq))
    gas = compute_hessian_fd(mol, "sto-3g", method="RHF")
    printed = _printed_frequencies(out)
    assert printed.size == 3
    np.testing.assert_allclose(printed, solvated.frequencies_cm1[-3:], rtol=0, atol=0.06)
    assert np.abs(printed - gas.frequencies_cm1[-3:]).max() > 3.0


def test_a_gas_phase_run_job_hessian_has_no_medium_line(tmp_path):
    vq = pytest.importorskip("vibeqc")

    stem = tmp_path / "w_gas"
    vq.run_job(
        _water(vq), basis="sto-3g", method="rhf", hessian=True,
        output=str(stem), progress=False, write_molden_file=False,
    )
    out = stem.with_suffix(".out").read_text()
    assert "Medium:" not in out and "equilibrium-solvation" not in out
