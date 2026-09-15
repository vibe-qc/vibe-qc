"""Phase 12c: multi-k periodic RHF with real-space density matrix.

Test strategy
-------------
The strongest correctness witnesses for the multi-k machinery are
regimes where the answer is already known or invariant:

  1. Molecular-limit regime (large unit cell, cutoff isolates g=0): any
     k-mesh must reproduce molecular RHF to machine precision, because
     P(g ≠ 0) is numerically zero and all lattice sums collapse to their
     molecular forms.

  2. k-mesh invariance in the molecular limit: since P(g) carries no
     k-dependent structure when only g=0 is active, the SCF result must
     be independent of the Monkhorst–Pack mesh size.

  3. Agreement with the Γ-only 12b driver (``run_rhf_periodic_gamma``)
     in the molecular limit: proves the multi-k driver specialises to
     the 12b code path correctly when the regime conditions are met.

  4. Non-trivial SCF convergence: tight 1D H₂ chains must still produce
     a converged SCF with sensible energies, even though the absolute
     value depends on the ERI-sum truncation convention (quantitative
     benchmarking awaits 12e's Ewald splitting for 3D; 1D results are
     meaningful but cutoff-dependent).

Bloch-supercell folding equivalence is NOT asserted here: at finite
lattice-sum cutoffs, the 1-cell × K k-points vs K-cell × Γ partitionings
do not see the identical physical-lattice region unless the cutoff
captures the same set of image atoms, which is fragile to set up
symmetrically. We recover this check as a convergence-to-limit exercise
rather than a hard equality.
"""

from __future__ import annotations

import gc
import re

import numpy as np
import pytest

import vibeqc as vq


H2  = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
H2O = [vq.Atom(8, [0.0,  0.0,  0.0]),
       vq.Atom(1, [0.0,  1.43, -0.98]),
       vq.Atom(1, [0.0, -1.43, -0.98])]


def _molecular_rhf(atoms):
    mol = vq.Molecule(atoms, 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return vq.run_rhf(mol, basis, opts)


def _big_box_system(atoms, dim, box=50.0, vacuum=30.0):
    if dim == 1:   lat = np.diag([box, vacuum, vacuum])
    elif dim == 2: lat = np.diag([box, box, vacuum])
    else:          lat = np.diag([box, box, box])
    return vq.PeriodicSystem(dim, lat, atoms)


def _scf_opts(cutoff=15.0):
    o = vq.PeriodicSCFOptions()
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    o.max_iter = 100
    return o


# ---------------------------------------------------------------------------
# 1. Molecular-limit agreement with molecular RHF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,atoms,dim,mesh",
    [("H2-1D-gamma",  H2, 1, [1,1,1]),
     ("H2-1D-k2",     H2, 1, [2,1,1]),
     ("H2-1D-k4",     H2, 1, [4,1,1]),
     ("H2-3D-gamma",  H2, 3, [1,1,1]),
     ("H2-3D-k222",   H2, 3, [2,2,2]),
     ("H2O-1D-k3",   H2O, 1, [3,1,1]),
     ("H2O-3D-k222", H2O, 3, [2,2,2])],
)
def test_molecular_limit_matches_molecular_rhf(name, atoms, dim, mesh):
    """Any k-mesh on a big-box system must reproduce molecular RHF."""
    sysp = _big_box_system(atoms, dim)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, mesh)
    res = vq.run_rhf_periodic(sysp, basis, km, _scf_opts())
    assert res.converged
    assert res.has_mean_field_state is False
    assert res.mean_field_state is None
    mres = _molecular_rhf(atoms)
    diff = abs(res.energy - mres.energy)
    assert diff < 1e-10, (
        f"{name}: periodic {res.energy:.12f} vs molecular {mres.energy:.12f}, "
        f"diff = {diff:.2e}"
    )


# ---------------------------------------------------------------------------
# 2. k-mesh invariance in the molecular limit
# ---------------------------------------------------------------------------

def test_kmesh_independence_in_molecular_limit():
    """Different k-mesh sizes give identical energies when only g=0 is in
    the lattice-sum cutoff — there is no k-dependent structure to resolve."""
    sysp = _big_box_system(H2O, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    energies = []
    for mesh in [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]:
        km = vq.monkhorst_pack(sysp, mesh)
        res = vq.run_rhf_periodic(sysp, basis, km, _scf_opts())
        energies.append(res.energy)
    spread = max(energies) - min(energies)
    assert spread < 1e-10, f"energy spread across meshes = {spread:.2e}"


# ---------------------------------------------------------------------------
# 3. Agreement with the 12b Γ-only driver in molecular limit
# ---------------------------------------------------------------------------

def test_multi_k_gamma_agrees_with_periodic_gamma_driver():
    """run_rhf_periodic with mesh=[1,1,1] must match run_rhf_periodic_gamma
    in the molecular limit — both specialise to molecular RHF there, so
    they must agree at machine precision."""
    sysp = _big_box_system(H2, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    km = vq.monkhorst_pack(sysp, [1,1,1])
    res_multi = vq.run_rhf_periodic(sysp, basis, km, _scf_opts())

    opts_gamma = vq.PeriodicRHFOptions()
    opts_gamma.lattice_opts.cutoff_bohr = 15.0
    opts_gamma.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts_gamma.conv_tol_energy = 1e-12
    res_gamma = vq.run_rhf_periodic_gamma(sysp, basis, opts_gamma)

    assert abs(res_multi.energy - res_gamma.energy) < 1e-12


# ---------------------------------------------------------------------------
# 4. Non-trivial SCF convergence for a 1D system
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh", [[2,1,1], [4,1,1], [6,1,1]])
def test_1d_h_chain_scf_converges(mesh):
    """Tight-spacing 1D H₂ chain: SCF must produce a converged result at
    each mesh. The quantitative value is lattice-sum-cutoff dependent
    and not asserted — see 12e for Ewald-accurate numbers."""
    sysp = vq.PeriodicSystem(1, np.diag([4.0, 30.0, 30.0]), H2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, mesh)
    opts = _scf_opts(cutoff=10.0)
    opts.lattice_opts.nuclear_cutoff_bohr = 20.0
    res = vq.run_rhf_periodic(sysp, basis, km, opts)
    assert res.converged
    # Energy must be finite and physically plausible (less negative than
    # deep-bound atomic limit).
    assert -100 < res.energy < 0


def test_experimental_capture_retains_physical_full_k_rhf_state():
    """Capture retains the certified Hermitian F[D], not accelerator state."""
    core = vq._vibeqc_core
    h2_chain = [vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(1, [1.4, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(
        1, np.diag([12.0, 30.0, 30.0]), h2_chain
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [3, 1, 1])
    # Five real-space cells for three k points exercise repeated BvK residue
    # classes without accepting the much larger finite-domain defect of the
    # deliberately tight a=4 fixture above.  The radial domains here converge
    # the raw anti-Hermitian residual below the capture contract's fixed gate.
    opts = _scf_opts(cutoff=25.2)
    opts.lattice_opts.nuclear_cutoff_bohr = 36.0
    opts.level_shift = 0.25
    opts.level_shift_warmup_cycles = 2
    opts.fock_mixing = 0.2

    request = core._PeriodicRHFStateCaptureRequest()
    request.calculation_identity = "1" * 64
    request.minimum_band_gap_hartree = 1.0e-6
    request.maximum_retained_numerical_payload_bytes = (
        core._estimate_periodic_restricted_mean_field_resident_bytes(
            [3, 1, 1], basis.nbasis, basis.nbasis
        )
    )
    request.frozen_core_mask_per_k = [
        [0] * basis.nbasis for _ in range(3)
    ]

    result = core._run_rhf_periodic_with_state_capture(
        sysp, basis, km, opts, request
    )
    assert result.converged
    assert result.has_mean_field_state is True
    state = result.mean_field_state
    assert state is not None
    assert state.periodic_dimension == 1
    assert state.n_kpoints == 3
    assert state.mesh == [3, 1, 1]
    assert state.is_shift == [0, 0, 0]
    assert state.resident_bytes == (
        request.maximum_retained_numerical_payload_bytes
    )
    assert state.reference_energy_per_cell == pytest.approx(result.energy)
    np.testing.assert_array_equal(state.kpoint_cartesian(0), np.zeros(3))
    assert np.max(np.abs(state.fock(0) - state.fock(1))) > 1.0e-8
    assert max(
        np.max(np.abs(state.overlap(1).imag)),
        np.max(np.abs(state.fock(1).imag)),
        np.max(np.abs(state.coefficients(1).imag)),
    ) > 1.0e-10

    coefficients = [state.coefficients(ik) for ik in range(state.n_kpoints)]
    overlap_lattice = vq.compute_overlap_lattice(
        basis, sysp, opts.lattice_opts
    )
    assert len(overlap_lattice.cells) > state.n_kpoints
    density = vq.real_space_density_from_kpoints(
        coefficients, [1, 1, 1], km, overlap_lattice.cells,
    )
    fock_2e = vq.build_fock_2e_real_space(
        basis, sysp, opts.lattice_opts, density
    )
    kinetic = vq.compute_kinetic_lattice(basis, sysp, opts.lattice_opts)
    nuclear = vq.compute_nuclear_lattice(basis, sysp, opts.lattice_opts)
    raw_hermiticity_residuals = []
    for ik in range(state.n_kpoints):
        kpoint = state.kpoint_cartesian(ik)
        rebuilt_raw = (
            np.asarray(vq.bloch_sum(fock_2e, kpoint))
            + np.asarray(vq.bloch_sum(kinetic, kpoint))
            + np.asarray(vq.bloch_sum(nuclear, kpoint))
        )
        raw_hermiticity_residuals.append(
            np.max(np.abs(rebuilt_raw - rebuilt_raw.conj().T))
        )
        rebuilt = 0.5 * (rebuilt_raw + rebuilt_raw.conj().T)
        np.testing.assert_allclose(state.fock(ik), rebuilt, atol=2.0e-8, rtol=0)
        np.testing.assert_allclose(
            state.fock(ik), state.fock(ik).conj().T, atol=1.0e-12, rtol=0
        )
    assert max(raw_hermiticity_residuals) < 1.0e-8

    del result
    gc.collect()
    assert state.calculation_identity == "1" * 64
    assert state.fock(1).shape == (basis.nbasis, basis.nbasis)


def test_experimental_capture_rejects_tight_cell_domain_defect():
    """A millihartree raw-Fock projection is not numerical cleanup."""
    core = vq._vibeqc_core
    h2_chain = [vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(1, [1.4, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(
        1, np.diag([4.0, 30.0, 30.0]), h2_chain
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [3, 1, 1])
    opts = _scf_opts(cutoff=10.0)
    opts.lattice_opts.nuclear_cutoff_bohr = 20.0

    request = core._PeriodicRHFStateCaptureRequest()
    request.calculation_identity = "4" * 64
    request.minimum_band_gap_hartree = 1.0e-6
    request.maximum_retained_numerical_payload_bytes = (
        core._estimate_periodic_restricted_mean_field_resident_bytes(
            km.mesh, basis.nbasis, basis.nbasis
        )
    )
    request.frozen_core_mask_per_k = [
        [0] * basis.nbasis for _ in range(len(km))
    ]

    with pytest.raises(RuntimeError, match="finite-domain Hermiticity"):
        core._run_rhf_periodic_with_state_capture(
            sysp, basis, km, opts, request
        )


def test_experimental_capture_rejects_computed_gap_below_request():
    """The producer must apply the gap contract to its converged bands."""
    core = vq._vibeqc_core
    h2_chain = [vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(1, [1.4, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(
        1, np.diag([12.0, 30.0, 30.0]), h2_chain
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [3, 1, 1])
    opts = _scf_opts(cutoff=25.2)
    opts.lattice_opts.nuclear_cutoff_bohr = 36.0

    request = core._PeriodicRHFStateCaptureRequest()
    request.calculation_identity = "5" * 64
    request.minimum_band_gap_hartree = 2.0
    request.maximum_retained_numerical_payload_bytes = (
        core._estimate_periodic_restricted_mean_field_resident_bytes(
            km.mesh, basis.nbasis, basis.nbasis
        )
    )
    request.frozen_core_mask_per_k = [
        [0] * basis.nbasis for _ in range(len(km))
    ]

    with pytest.raises(RuntimeError, match="global band gap"):
        core._run_rhf_periodic_with_state_capture(
            sysp, basis, km, opts, request
        )


def test_capture_entrypoint_rejects_before_scf_construction():
    core = vq._vibeqc_core
    sysp = _big_box_system(H2, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    retained_bytes = core._estimate_periodic_restricted_mean_field_resident_bytes(
        km.mesh, basis.nbasis, basis.nbasis
    )

    def request_for(mesh, byte_limit=retained_bytes):
        request = core._PeriodicRHFStateCaptureRequest()
        request.calculation_identity = "2" * 64
        request.minimum_band_gap_hartree = 1.0e-6
        request.maximum_retained_numerical_payload_bytes = byte_limit
        request.frozen_core_mask_per_k = [
            [0] * basis.nbasis for _ in range(len(mesh))
        ]
        return request

    opts = _scf_opts()
    with pytest.raises(RuntimeError, match="above the explicit limit"):
        core._run_rhf_periodic_with_state_capture(
            sysp, basis, km, opts, request_for(km, retained_bytes - 1)
        )

    smearing_opts = _scf_opts()
    smearing_opts.smearing_temperature = 1.0e-3
    with pytest.raises(ValueError, match="smearing"):
        core._run_rhf_periodic_with_state_capture(
            sysp, basis, km, smearing_opts, request_for(km)
        )

    loose_opts = _scf_opts()
    loose_opts.conv_tol_grad = 2.0e-6
    with pytest.raises(ValueError, match="no looser"):
        core._run_rhf_periodic_with_state_capture(
            sysp, basis, km, loose_opts, request_for(km)
        )

    damped_opts = _scf_opts()
    damped_opts.use_diis = False
    with pytest.raises(ValueError, match="density damping requires DIIS"):
        core._run_rhf_periodic_with_state_capture(
            sysp, basis, km, damped_opts, request_for(km)
        )

    vq.attach_symmetry(sysp)
    reduced = vq.monkhorst_pack(
        sysp, [2, 2, 2], use_symmetry=True
    )
    with pytest.raises(ValueError, match="full Brillouin"):
        core._run_rhf_periodic_with_state_capture(
            sysp, basis, reduced, opts, request_for(reduced)
        )


def test_capture_propagates_nonzero_frozen_core_mask():
    core = vq._vibeqc_core
    lih = [vq.Atom(3, [0.0, 0.0, 0.0]),
           vq.Atom(1, [0.0, 0.0, 3.015])]
    sysp = _big_box_system(lih, 1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [1, 1, 1])

    request = core._PeriodicRHFStateCaptureRequest()
    request.calculation_identity = "3" * 64
    request.minimum_band_gap_hartree = 1.0e-6
    request.maximum_retained_numerical_payload_bytes = (
        core._estimate_periodic_restricted_mean_field_resident_bytes(
            km.mesh, basis.nbasis, basis.nbasis
        )
    )
    request.frozen_core_mask_per_k = [
        [1] + [0] * (basis.nbasis - 1)
    ]

    result = core._run_rhf_periodic_with_state_capture(
        sysp, basis, km, _scf_opts(), request
    )
    state = result.mean_field_state
    assert result.converged and state is not None
    assert state.n_frozen_core == 1
    assert state.n_correlated_occupied == 1
    assert state.frozen_core_mask(0) == [1] + [0] * (basis.nbasis - 1)
    assert state.correlated_occupied_mask(0)[:2] == [0, 1]


# ---------------------------------------------------------------------------
# 5. Real-space density helper correctness
# ---------------------------------------------------------------------------

def test_real_space_density_single_kpoint_reduces_to_pk():
    """With a single k-point at Γ, P(g) = P(Γ) for every cell g. This is
    the trivial case and validates the folding formula."""
    sysp = _big_box_system(H2, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 20.0
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)

    # Run molecular RHF to get a known density and MO coefficients.
    mres = _molecular_rhf(H2)
    n_occ = 1
    # At Γ the coefficients are real; broadcast to complex for the helper.
    C_gamma = np.asarray(mres.mo_coeffs).astype(np.complex128)

    km = vq.monkhorst_pack(sysp, [1,1,1])
    P_set = vq.real_space_density_from_kpoints(
        [C_gamma], [n_occ], km, cells,
    )
    # Every cell's block should equal the molecular density (the trivial
    # Γ-only fold replicates P across all cells).
    D_mol = np.asarray(mres.density)
    for b in P_set.blocks:
        assert np.max(np.abs(np.asarray(b) - D_mol)) < 1e-10


def test_real_space_density_rejects_mismatched_input_sizes():
    km = vq.monkhorst_pack(_big_box_system(H2, 3), [2, 2, 2])
    cells = vq.direct_lattice_cells(_big_box_system(H2, 3), 5.0)
    C = [np.zeros((2,2), dtype=np.complex128)] * 3   # wrong length
    with pytest.raises(RuntimeError, match="size mismatch"):
        vq.real_space_density_from_kpoints(C, [1]*3, km, cells)


def test_cutoff_cell_density_bloch_sum_is_not_a_discrete_inverse():
    """Repeated BvK residue classes make a cutoff-cell sum alias P(k)."""
    sysp = vq.PeriodicSystem(1, np.diag([4.0, 30.0, 30.0]), H2)
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    cells = vq.direct_lattice_cells(sysp, 10.0)
    assert len(cells) > len(km)

    coefficients = [
        np.eye(2, dtype=np.complex128),
        np.array(
            [[1.0, -1.0], [1.0, 1.0]], dtype=np.complex128
        ) / np.sqrt(2.0),
    ]
    folded = vq.real_space_density_from_kpoints(
        coefficients, [1, 1], km, cells
    )
    exact = [
        2.0 * matrix[:, :1] @ matrix[:, :1].conj().T
        for matrix in coefficients
    ]
    aliased = [
        np.asarray(vq.bloch_sum(folded, kpoint))
        for kpoint in km.kpoints
    ]

    assert any(
        not np.allclose(recovered, projector)
        for recovered, projector in zip(aliased, exact, strict=True)
    )


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------

def test_multi_k_rejects_odd_electron_unit_cell():
    sysp = vq.PeriodicSystem(3, np.eye(3) * 10.0,
                             [vq.Atom(1, [0,0,0])],
                             charge=0, multiplicity=2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [2,2,2])
    with pytest.raises(ValueError, match="even electron count"):
        vq.run_rhf_periodic(sysp, basis, km)


# (An explicit empty-k-mesh test is omitted — BlochKMesh has no Python
# default constructor and every path that produces one populates at
# least a Γ point. The C++ guard remains as a defensive check.)


# ---------------------------------------------------------------------------
# 7. Per-cell EDIIS / ADIIS in multi-k SCF
# ---------------------------------------------------------------------------
#
# Until v0.9.x the multi-k EDIIS / EDIIS_DIIS / ADIIS branches extrapolated
# only the Γ-folded Fock and distributed the correction uniformly across
# cells. The promoted kernel keeps a full per-cell history; the QP cross-
# term ⟨F_i | D_j⟩ sums over the real-space cell list, matching the
# periodic energy bilinear form E_elec = ½ Σ_g (P(g) ⊙ [H(g) + F(g)]).sum().
# These tests pin the contract:
#
#   * Every accelerator converges on a multi-k molecular-limit system and
#     reproduces molecular RHF — i.e., the extrapolator does not drag the
#     SCF off its fixed point.
#   * EDIIS, EDIIS_DIIS, ADIIS land at the same converged energy as plain
#     DIIS, so the per-cell extrapolation is a stable accelerator across
#     the suite (not biased by the choice of accelerator).

# Pure EDIIS / pure ADIIS are documented to plateau on the gradient-norm
# convergence flag (see docs/user_guide/scf_convergence.md: "pure EDIIS
# plateaus near convergence; prefer the hybrid"). The fixed-point energy
# still reaches molecular RHF to machine precision — that's what the
# per-cell promotion has to preserve. EDIIS_DIIS and DIIS converge fully.
_ACCEL_NEEDS_CONVERGENCE_FLAG = {
    vq.SCFAccelerator.DIIS: True,
    vq.SCFAccelerator.EDIIS: False,
    vq.SCFAccelerator.EDIIS_DIIS: True,
    vq.SCFAccelerator.ADIIS: False,
}


@pytest.mark.parametrize("accel", list(_ACCEL_NEEDS_CONVERGENCE_FLAG.keys()))
def test_multi_k_accelerator_molecular_limit_matches_molecular_rhf(accel):
    """Per-cell EDIIS / ADIIS must not drag the SCF off the molecular-limit
    fixed point: the energy must match molecular RHF to machine precision.
    EDIIS_DIIS additionally must trip the convergence flag."""
    sysp = _big_box_system(H2O, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _scf_opts()
    opts.scf_accelerator = accel
    res = vq.run_rhf_periodic(sysp, basis, km, opts)
    mres = _molecular_rhf(H2O)
    assert abs(res.energy - mres.energy) < 1e-10, (
        f"{accel}: periodic {res.energy:.12f} vs molecular "
        f"{mres.energy:.12f}, diff = {abs(res.energy - mres.energy):.2e}"
    )
    if _ACCEL_NEEDS_CONVERGENCE_FLAG[accel]:
        assert res.converged, f"{accel}: SCF failed to converge"


def test_multi_k_per_cell_accelerators_agree_with_diis_on_h_chain():
    """On a non-trivial multi-k 1D H₂ chain, per-cell EDIIS / EDIIS_DIIS /
    ADIIS must land at the same fixed-point energy as plain DIIS —
    confirming the extrapolation is consistent, not just stable."""
    sysp = vq.PeriodicSystem(1, np.diag([4.0, 30.0, 30.0]), H2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [4, 1, 1])

    def _energy(accel):
        opts = _scf_opts(cutoff=10.0)
        opts.lattice_opts.nuclear_cutoff_bohr = 20.0
        opts.scf_accelerator = accel
        r = vq.run_rhf_periodic(sysp, basis, km, opts)
        if _ACCEL_NEEDS_CONVERGENCE_FLAG[accel]:
            assert r.converged, f"{accel} should converge but didn't"
        return r.energy

    e_diis = _energy(vq.SCFAccelerator.DIIS)
    for accel in (vq.SCFAccelerator.EDIIS,
                  vq.SCFAccelerator.EDIIS_DIIS,
                  vq.SCFAccelerator.ADIIS):
        e = _energy(accel)
        assert abs(e - e_diis) < 1e-8, (
            f"{accel}: {e:.12f} vs DIIS {e_diis:.12f}, "
            f"diff = {abs(e - e_diis):.2e}"
        )


# ---------------------------------------------------------------------------
# 5. run_periodic_job fail-closed SCF convergence gate
# ---------------------------------------------------------------------------


def test_run_periodic_job_raises_and_marks_crashed_on_scf_nonconvergence(tmp_path):
    """An SCF that hits its iteration cap must fail loudly.

    Regression for the 2026-08-12 CaO PBE GDF 200-iter incident:
    ``run_periodic_job`` returned the capped (unconverged) result, the CLI
    exited 0, and the .system manifest flipped to "complete", so pipelines
    trusting exit codes / manifest status promoted an unconverged energy
    into the validation database. The gate must raise, write a FATAL line
    to the .out, and crash the manifest -- the periodic mirror of the
    molecular runner's post-SCF guard.
    """
    box = 14.0
    c = box / 2.0
    sysp = vq.PeriodicSystem(
        3,
        np.diag([box, box, box]),
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "h2_gdf_capped"

    with pytest.raises(RuntimeError, match="did not converge"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            max_iter=1,
            output=str(stem),
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            output_qvf=False,
            record_hostname=False,
        )

    out_text = stem.with_suffix(".out").read_text()
    assert "FATAL" in out_text
    assert "did not converge" in out_text

    manifest_text = stem.with_suffix(".system").read_text()
    assert re.search(r'status\s+= "crashed"', manifest_text)
