"""Public ``run_periodic_job`` dispatch for the BIPOLE restricted-open-shell
backends (ROHF and ROKS).

G-PBC-007 sub-gates: ``run_periodic_job(method="ROHF"|"ROKS",
jk_method="bipole")`` reaches the validated EWALD_3D corrected-Ewald-exchange
restricted-open-shell engine at Gamma and on full Monkhorst-Pack meshes.
These tests pin

* the AUTO heuristic (ROHF rides the open-shell BIPOLE arm),
* public-route parity with the standalone multi-k drivers (the ROHF LiH+
  ``(3,1,1)`` anchor is validated out of process against PySCF KROHF/GDF
  to 0.062 mHa in ``test_periodic_rohf_multi_k_ewald.py``; the ROKS
  numerics are pinned in ``test_periodic_roks_multi_k_ewald.py``),
* the public native-GDF ROHF route at default and explicit Gamma,
* the fail-closed surface (ROKS/GDF, ROKS screened hybrids, unsupported
  restricted-open-shell knobs, smearing, optimization),
* and the fused OpenMP multi-k Bloch/density-fold kernels against a
  direct NumPy reference (the per-iteration k-transforms of both drivers
  ride these kernels).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_jk_method import PeriodicJKMethod, pick_jk_method


def _h_atom_system(box: float = 10.0):
    c = box / 2
    system = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    system.multiplicity = 2
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _lihp_system(box: float = 12.0):
    c = box / 2
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(3, [c - 0.6, c, c]), vq.Atom(1, [c + 1.0, c, c])],
        charge=1,
        multiplicity=2,
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def test_auto_resolves_rohf_to_bipole():
    system, basis = _h_atom_system()
    resolved = pick_jk_method(
        "auto",
        lattice=np.asarray(system.lattice, dtype=float),
        basis_name="sto-3g",
        n_atoms=1,
        scf_method="ROHF",
        dim=3,
    )
    assert resolved == PeriodicJKMethod.BIPOLE


def test_public_gamma_rohf_bipole_runs_and_writes(tmp_path):
    system, basis = _h_atom_system()
    stem = tmp_path / "h-rohf"
    res = vq.run_periodic_job(
        system,
        basis,
        method="ROHF",
        output=str(stem),
        max_iter=60,
    )
    assert res.converged
    assert res.s_squared == pytest.approx(0.75, abs=1e-12)
    assert res.n_alpha == 1 and res.n_beta == 0
    assert np.allclose(np.asarray(res.mo_occupations[0])[:1], [1.0])
    assert res.runtime_backend == "bipole-rohf-multi-k-ewald"
    # Runner-facing metadata consumed by the sidecar/QVF writers.
    assert res.density is not None
    assert res.kpoints_cart is not None and res.kpoints_cart.shape == (1, 3)
    out_text = stem.with_suffix(".out").read_text()
    assert "PERIODIC ROHF" in out_text
    assert "## References" in out_text
    assert "Roothaan" in out_text
    assert stem.with_suffix(".system").exists()
    assert stem.with_suffix(".qvf").exists()


def test_public_multik_rohf_matches_standalone_driver(tmp_path):
    system, basis = _lihp_system()
    stem = tmp_path / "lihp-rohf"
    res = vq.run_periodic_job(
        system,
        basis,
        method="ROHF",
        kpoints=[3, 1, 1],
        output=str(stem),
        max_iter=100,
        conv_tol_energy=1e-8,
        bipole_cutoff_bohr=12.0,
        bipole_nuclear_cutoff_bohr=15.0,
        damping=0.3,
    )
    assert res.converged
    assert res.s_squared == pytest.approx(0.75, abs=1e-12)
    # Same fixture/options as the standalone anchor in
    # test_periodic_rohf_multi_k_ewald.py (PySCF KROHF/GDF residual
    # +0.062 mHa recorded there after the padded-SR domain fix). #651
    # bounded the EWALD_3D default alpha by the 12-bohr cutoff and moved
    # that anchor -7.5192630685 -> -7.5192632201; this copy follows it.
    assert res.energy == pytest.approx(-7.5192632201, abs=1e-8)
    assert len(res.mo_coeffs) == 3
    assert res.kpoints_cart.shape == (3, 3)
    assert stem.with_suffix(".qvf").exists()


def test_public_gamma_roks_bipole_runs_and_writes(tmp_path):
    """ROKS rides the same BIPOLE arm as ROHF, with V_xc added per spin."""
    system, basis = _h_atom_system()
    stem = tmp_path / "h-roks"
    res = vq.run_periodic_job(
        system,
        basis,
        method="ROKS",
        functional="pbe",
        jk_method="bipole",
        output=str(stem),
        max_iter=60,
    )
    assert res.converged
    assert res.s_squared == pytest.approx(0.75, abs=1e-12)
    assert res.n_alpha == 1 and res.n_beta == 0
    assert res.runtime_backend == "bipole-roks-multi-k-ewald"
    assert res.functional == "pbe"
    assert res.e_hf_exchange == 0.0  # pure functional builds no exchange
    assert res.density is not None
    assert res.kpoints_cart is not None and res.kpoints_cart.shape == (1, 3)
    out_text = stem.with_suffix(".out").read_text()
    assert "PERIODIC ROKS" in out_text
    assert "## References" in out_text
    assert stem.with_suffix(".system").exists()
    assert stem.with_suffix(".qvf").exists()


def test_public_multik_roks_bipole_matches_standalone_driver(tmp_path):
    """The public multi-k ROKS route reproduces the standalone driver."""
    from vibeqc.periodic_roks_multi_k_ewald import (
        run_roks_periodic_multi_k_ewald3d,
    )

    system, basis = _lihp_system()
    res = vq.run_periodic_job(
        system,
        basis,
        method="ROKS",
        functional="pbe",
        jk_method="bipole",
        kpoints=[3, 1, 1],
        max_iter=100,
        conv_tol_energy=1e-8,
        bipole_cutoff_bohr=12.0,
        bipole_nuclear_cutoff_bohr=12.0,
        damping=0.3,
        output=str(tmp_path / "multik-roks-bipole"),
    )
    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.damping = 0.3
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-8
    ref = run_roks_periodic_multi_k_ewald3d(
        system, basis, vq.monkhorst_pack(system, [3, 1, 1]), opts, progress=False
    )
    assert res.converged and ref.converged
    assert res.energy == pytest.approx(ref.energy, abs=1e-9)
    assert len(res.mo_coeffs) == 3


def test_roks_bipole_rejects_screened_hybrid(tmp_path):
    """hse06 needs an erfc exchange arm the BIPOLE engine does not carry."""
    system, basis = _h_atom_system()
    # Unlike the knob and backend refusals below, this one fires downstream of
    # the output writer, so it needs a tmp_path stem to stay out of the tree.
    with pytest.raises(NotImplementedError, match="erfc"):
        vq.run_periodic_job(
            system,
            basis,
            method="ROKS",
            functional="hse06",
            jk_method="bipole",
            output=str(tmp_path / "roks-bipole-hse06"),
        )


def test_roks_bipole_rejects_unsupported_knobs():
    system, basis = _h_atom_system()
    with pytest.raises(NotImplementedError, match="use_mom"):
        vq.run_periodic_job(
            system,
            basis,
            method="ROKS",
            functional="pbe",
            jk_method="bipole",
            use_mom=True,
        )


def test_roks_gdf_fails_closed():
    system, basis = _h_atom_system()
    with pytest.raises(NotImplementedError, match="gdf"):
        vq.run_periodic_job(
            system, basis, method="ROKS", functional="pbe", jk_method="gdf"
        )


@pytest.mark.parametrize(
    ("kpoints", "suffix", "convergence_kwargs", "expect_molden"),
    [
        (None, "default", {}, True),
        (
            (1, 1, 1),
            "explicit",
            {"convergence": "off", "damping": 0.0},
            True,
        ),
        # Multi-k GDF exports the Gamma block like BIPOLE. It was briefly
        # refused because its degenerate frontier orbital stayed complex
        # after global-phase removal; the writer now re-expresses degenerate
        # blocks on a real basis of their own span, which fixes both routes.
        # A mesh with no Gamma point at all still fails closed.
        (
            (2, 1, 1),
            "multik",
            {"convergence": "off", "damping": 0.0},
            True,
        ),
    ],
)
def test_public_rohf_gdf_matches_standalone_driver(
    tmp_path, kpoints, suffix, convergence_kwargs, expect_molden
):
    """The public GDF route must be the validated standalone engine."""
    from vibeqc.periodic_rohf_gdf import run_krohf_periodic_gdf

    system, basis = _h_atom_system()
    opts = vq.PeriodicRHFOptions()
    opts.damping = 0.0
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-8
    ref = run_krohf_periodic_gdf(
        system,
        basis,
        kmesh=(kpoints or (1, 1, 1)),
        options=opts,
        aux_basis="def2-svp-jk",
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0,
        progress=False,
    )

    stem = tmp_path / f"h-rohf-gdf-{suffix}"
    got = vq.run_periodic_job(
        system,
        basis,
        method="ROHF",
        jk_method="gdf",
        kpoints=kpoints,
        max_iter=60,
        conv_tol_energy=1e-8,
        aux_basis="def2-svp-jk",
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0,
        output=str(stem),
        output_qvf=(suffix == "default"),
        write_xyz_file=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
        **convergence_kwargs,
    )

    assert ref.converged and got.converged
    assert got.energy == pytest.approx(ref.energy, abs=1e-12)
    assert got.s_squared == pytest.approx(0.75, abs=1e-12)
    assert got.n_alpha == 1 and got.n_beta == 0
    assert got.backend == "native-multi-k-gdf-rohf"
    assert got.runtime_backend == "native-multi-k-gdf-rohf"
    assert got.method == "rohf"
    assert len(got.mo_coeffs) == int(np.prod(kpoints or (1, 1, 1)))
    assert np.allclose(got.occupations[0], got.mo_occupations[0])
    assert np.allclose(
        got.density[0], got.density_alpha[0] + got.density_beta[0]
    )
    out_text = stem.with_suffix(".out").read_text()
    assert "PERIODIC ROHF" in out_text
    assert "Roothaan" in out_text
    assert "Sun" in out_text
    molden_path = stem.with_suffix(".molden")
    assert molden_path.exists() is expect_molden
    if expect_molden:
        molden_text = molden_path.read_text()
        # Bloch coefficients are complex128 even at Γ; the writer must have
        # divided out the global phase rather than formatting them as
        # "a+bj", which no Molden reader can parse.
        assert not [
            line
            for line in molden_text.splitlines()
            if line.rstrip().endswith("j")
        ]
        # The singly-occupied ROHF orbital carries the result's own
        # occupation. An aufbau guess from n_electrons=1 would floor to
        # zero and report the open shell as empty.
        assert " Occup=   1.00000000" in molden_text
    assert stem.with_suffix(".qvf").exists() is (suffix == "default")
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"use_mom": True}, "use_mom"),
        ({"level_shift": 0.2}, "level_shift"),
        ({"fock_mixing": 0.2}, "fock_mixing"),
    ],
)
def test_rohf_gdf_rejects_unimplemented_knobs(tmp_path, kwargs, match):
    system, basis = _h_atom_system()
    with pytest.raises(NotImplementedError, match=match):
        vq.run_periodic_job(
            system,
            basis,
            method="ROHF",
            jk_method="gdf",
            output=str(tmp_path / f"rohf-gdf-{match}"),
            output_qvf=False,
            **kwargs,
        )


def test_rohf_bipole_rejects_unsupported_knobs():
    system, basis = _h_atom_system()
    with pytest.raises(NotImplementedError, match="use_mom"):
        vq.run_periodic_job(
            system, basis, method="ROHF", jk_method="bipole", use_mom=True
        )
    with pytest.raises(NotImplementedError, match="ewald_omega"):
        vq.run_periodic_job(
            system, basis, method="ROHF", jk_method="bipole", ewald_omega=0.3
        )


def test_rohf_bipole_rejects_smearing():
    system, basis = _h_atom_system()
    with pytest.raises(NotImplementedError, match="smearing"):
        vq.run_periodic_job(
            system,
            basis,
            method="ROHF",
            jk_method="bipole",
            smearing_temperature=0.01,
        )


def test_rohf_bipole_rejects_optimize():
    system, basis = _h_atom_system()
    with pytest.raises(NotImplementedError, match="optimization"):
        vq.run_periodic_job(
            system, basis, method="ROHF", jk_method="bipole", optimize=True
        )


def test_fused_bloch_kernels_match_python_reference():
    """Pin the OpenMP multi-k kernels the ROHF hot loop rides.

    ``bloch_sum_multi_k`` / ``assemble_fock_multi_k`` against the direct
    per-k NumPy fold, and the restructured (two-phase, cell-parallel)
    ``real_space_density_from_kpoints_fractional`` against an explicit
    double loop.
    """
    from vibeqc._vibeqc_core import (
        assemble_fock_multi_k,
        bloch_sum_multi_k,
        compute_overlap_lattice,
        real_space_density_from_kpoints_fractional,
    )

    system, basis = _lihp_system()
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    S_lat = compute_overlap_lattice(basis, system, opts.lattice_opts)
    cells = list(S_lat.cells)
    cell_r = [np.asarray(c.r_cart, dtype=float) for c in cells]
    blocks = [np.asarray(b, dtype=float) for b in S_lat.blocks]
    nbf = blocks[0].shape[0]

    kmesh = vq.monkhorst_pack(system, [3, 1, 1])
    k_list = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]

    # --- forward Bloch sum -------------------------------------------
    fused = bloch_sum_multi_k(blocks, cell_r, k_list)
    for ik, k in enumerate(k_list):
        ref = np.zeros((nbf, nbf), dtype=complex)
        for g, block in enumerate(blocks):
            ref = ref + np.exp(1j * float(np.dot(k, cell_r[g]))) * block
        assert np.allclose(np.asarray(fused[ik]), ref, atol=1e-13)

    # --- fused Fock assembly (Bloch sum + Hcore) ---------------------
    rng = np.random.default_rng(7)
    hcore = [
        (lambda m: 0.5 * (m + m.conj().T))(
            rng.standard_normal((nbf, nbf))
            + 1j * rng.standard_normal((nbf, nbf))
        )
        for _ in k_list
    ]
    asm = assemble_fock_multi_k(blocks, cell_r, k_list, hcore)
    for ik in range(len(k_list)):
        assert np.allclose(
            np.asarray(asm[ik]), np.asarray(fused[ik]) + hcore[ik], atol=1e-13
        )

    # --- inverse fold (two-phase restructure) ------------------------
    C_per_k = [
        rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))
        for _ in k_list
    ]
    occ_per_k = [
        np.asarray([2.0, 1.0] + [0.0] * (nbf - 2), dtype=float)
        for _ in k_list
    ]
    folded = real_space_density_from_kpoints_fractional(
        C_per_k, occ_per_k, kmesh, cells
    )
    weights = np.asarray(kmesh.weights, dtype=float)
    for g, cell in enumerate(cells):
        ref = np.zeros((nbf, nbf), dtype=float)
        for ik, k in enumerate(k_list):
            P_k = (C_per_k[ik] * occ_per_k[ik][None, :]) @ C_per_k[ik].conj().T
            z = np.exp(-1j * float(np.dot(k, np.asarray(cell.r_cart))))
            ref += weights[ik] * np.real(z * P_k)
        assert np.allclose(np.asarray(folded.blocks[g]), ref, atol=1e-12)
