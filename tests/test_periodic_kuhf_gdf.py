"""Open-shell multi-k GDF drivers: run_kuhf_periodic_gdf /
run_kuks_periodic_gdf.

Validation strategy (the multi-k open-shell α/β absolute energy cannot be
pinned directly against PySCF KUHF/KUKS for these tiny cells — PySCF
distributes ``cell.spin`` over the Born–von-Kármán supercell, a different
state from the physically-standard per-cell multiplicity this driver
runs; see ``test_kuhf_open_shell_path_matches_pyscf_gamma`` for the clean
PySCF anchor and the module note below). Instead the drivers are pinned by:

  * **closed-shell limit** — UHF/UKS at M=1 reproduce the trusted
    multi-k ``run_krhf/krks_periodic_gdf`` (which carry µHa PySCF
    KRHF/KRKS parity) to machine precision. This validates J, V_xc,
    exxdiv, the energy, the SCF, and the whole multi-k machinery.
  * **open-shell (α≠β) path** — at a single k-point it reproduces the
    *validated* Γ driver ``run_pbc_gdf_uhf`` (itself PySCF-pinned), and
    the Γ triplet matches PySCF UHF.density_fit() to machine precision.
  * **spin** — exact ⟨S²⟩ (0 singlet, 0.75 doublet, 2.0 triplet).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_k_gdf import (
    run_krhf_periodic_gdf,
    run_krks_periodic_gdf,
    run_kuhf_periodic_gdf,
    run_kuks_periodic_gdf,
)


def _h2_box(box_bohr: float = 12.0, sep_bohr: float = 1.4):
    half = 0.5 * sep_bohr
    system = vq.PeriodicSystem(
        3, np.diag([box_bohr, box_bohr, box_bohr]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _opts(ks: bool = False):
    o = vq.PeriodicKSOptions() if ks else vq.PeriodicRHFOptions()
    o.use_diis = True
    o.damping = 0.0
    o.max_iter = 60
    o.conv_tol_energy = 1e-9
    o.lattice_opts.cutoff_bohr = 30.0
    o.lattice_opts.nuclear_cutoff_bohr = 30.0
    return o


_COMMON = dict(aux_basis="def2-svp-jk", gdf_method="rsgdf", rsgdf_ke_cutoff=200.0,
               progress=False)


def test_kuhf_m1_equals_krhf():
    """KUHF at multiplicity=1 reproduces KRHF on H2/(2,1,1) to machine
    precision — the closed-shell-limit gate validating the open-shell
    multi-k J / per-spin-K / exxdiv / energy / SCF against the trusted
    (PySCF-µHa-parity) multi-k RHF driver."""
    system, basis = _h2_box()
    r_krhf = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), use_compcell=True, **_COMMON
    )
    r_kuhf = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), **_COMMON
    )
    assert r_krhf.converged and r_kuhf.converged
    assert abs(r_kuhf.energy - r_krhf.energy) < 1e-8, (
        f"KUHF(M=1) {r_kuhf.energy} != KRHF {r_krhf.energy}"
    )
    assert abs(r_kuhf.s_squared) < 1e-6  # singlet
    assert isinstance(r_kuhf, vq.PeriodicKUHFGDFResult)


def test_kuks_pbe_m1_equals_krks():
    """KUKS-PBE at M=1 reproduces KRKS-PBE on H2/(2,1,1) to machine
    precision — validates the open-shell multi-k native V_xc
    (build_xc_periodic_uks) against the trusted multi-k RKS driver.
    Also checks UKS(functional=None) ≡ KRHF (the no-XC fallback)."""
    system, basis = _h2_box()
    r_none = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), functional=None, **_COMMON
    )
    r_krhf = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), use_compcell=True, **_COMMON
    )
    assert abs(r_none.energy - r_krhf.energy) < 1e-8

    r_krks = run_krks_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(True), functional="pbe",
        use_compcell=True, **_COMMON
    )
    r_kuks = run_kuks_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(True), functional="pbe", **_COMMON
    )
    assert r_krks.converged and r_kuks.converged
    assert r_kuks.functional == "pbe" and r_kuks.e_xc < 0.0
    assert abs(r_kuks.energy - r_krks.energy) < 1e-6, (
        f"KUKS-PBE(M=1) {r_kuks.energy} != KRKS-PBE {r_krks.energy}"
    )
    assert abs(r_kuks.s_squared) < 1e-6  # singlet


def test_kuks_ibz_reduced_kmesh_matches_full_mesh():
    """KUKS on a symmetry-reduced (IBZ) mesh must run and reproduce the
    full mesh -- the open-shell twin of test_periodic_k_gdf.py::
    test_ibz_reduced_kmesh_dft_matches_full_mesh.

    The first IBZ expansion cut (375b6363d) left the driver's
    ``kmesh_bloch`` on the caller's wedge while the k-point arrays were
    expanded, so the per-spin XC build died on the k-list length mismatch
    exactly like the closed-shell driver. The drivers now adopt the
    expanded mesh for everything derived from the kmesh.
    """
    system, basis = _h2_box()
    system.charge = 1
    system.multiplicity = 2  # H2+ doublet: genuinely open-shell
    vq.attach_symmetry(system)
    k_ibz = vq.monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
    assert np.asarray(k_ibz.kpoints).reshape(-1, 3).shape[0] < 8  # a wedge
    k_full = vq.monkhorst_pack(system, [2, 2, 2], use_symmetry=False)

    r_ibz = run_kuks_periodic_gdf(
        system, basis, k_ibz, functional="pbe", progress=False
    )
    r_full = run_kuks_periodic_gdf(
        system, basis, k_full, functional="pbe", progress=False
    )
    assert r_ibz.converged and r_full.converged
    assert r_ibz.kpoints_cart.shape[0] == 8  # full mesh, not the wedge
    assert abs(r_ibz.energy - r_full.energy) < 1e-10


def test_kuhf_open_shell_path_matches_pyscf_gamma():
    """The genuine open-shell (α≠β) path: a triplet H2 (nα=2, nβ=0) run
    through the multi-k driver at a single k-point reproduces PySCF
    UHF.density_fit()/exxdiv='ewald' (Γ) AND the validated Γ open-shell
    driver run_pbc_gdf_uhf, both to machine precision. ⟨S²⟩ is exactly
    the triplet value 2.0.

    PySCF reference (out-of-process, §10; pyscf.pbc.scf.UHF(cell)
    .density_fit(), xc-free, H2/sto-3g/12-bohr, cell.spin=2, cell.unit='B'):
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_uhf

    PYSCF_UHF_H2_TRIPLET = -0.53585051
    system, basis = _h2_box()
    system.multiplicity = 3  # triplet

    r_k = run_kuhf_periodic_gdf(
        system, basis, kmesh=(1, 1, 1), options=_opts(), **_COMMON
    )
    r_g = run_pbc_gdf_uhf(
        system, basis, _opts(), exxdiv="ewald", **_COMMON
    )
    assert r_k.converged and r_g.converged
    assert abs(r_k.energy - PYSCF_UHF_H2_TRIPLET) < 5e-6, (
        f"KUHF triplet (1,1,1) {r_k.energy} vs PySCF UHF Γ "
        f"{PYSCF_UHF_H2_TRIPLET} (Δ={r_k.energy - PYSCF_UHF_H2_TRIPLET:.2e})"
    )
    assert abs(r_k.energy - r_g.energy) < 1e-8  # ≡ validated Γ driver
    assert abs(r_k.s_squared - 2.0) < 1e-6  # triplet ⟨S²⟩


def test_run_periodic_job_open_shell_gdf_dispatch(tmp_path):
    """run_periodic_job(method='UHF'/'UKS', jk_method='gdf') routes to the
    open-shell GDF drivers (Γ → run_pbc_gdf_{uhf,uks}; multi-k →
    run_kuhf/kuks_periodic_gdf) on the rsgdf path, and the runner's
    result-processing handles the per-spin result classes. Γ UHF on the
    triplet H2 reproduces the direct driver / PySCF UHF Γ value."""
    from vibeqc.periodic_runner import run_periodic_job

    system, basis = _h2_box()
    system.multiplicity = 3  # triplet, exercises α≠β through the runner

    r_g = run_periodic_job(
        system, basis, method="UHF", jk_method="gdf", aux_basis="def2-svp-jk",
        output=str(tmp_path / "uhf_g"), max_iter=60, conv_tol_energy=1e-9,
        write_molden_file=False, initial_guess="HCORE",
    )
    assert isinstance(r_g, vq.PBCGDFUHFResult)
    assert r_g.converged
    assert abs(r_g.energy - (-0.53585051)) < 5e-6  # ≡ direct driver / PySCF UHF Γ
    assert abs(r_g.s_squared - 2.0) < 1e-6

    r_k = run_periodic_job(
        system, basis, method="UHF", jk_method="gdf", kpoints=(2, 1, 1),
        aux_basis="def2-svp-jk", output=str(tmp_path / "uhf_k"), max_iter=60,
        conv_tol_energy=1e-9, write_molden_file=False, initial_guess="HCORE",
    )
    assert isinstance(r_k, vq.PeriodicKUHFGDFResult)
    assert r_k.converged and -5.0 < r_k.energy < 5.0

    sysm, basis2 = _h2_box()  # closed-shell singlet for UKS-PBE
    r_uks = run_periodic_job(
        sysm, basis2, method="UKS", functional="pbe", jk_method="gdf",
        aux_basis="def2-svp-jk", output=str(tmp_path / "uks_g"), max_iter=60,
        conv_tol_energy=1e-9, write_molden_file=False, initial_guess="HCORE",
    )
    assert isinstance(r_uks, vq.PBCGDFUKSResult)
    assert r_uks.converged and r_uks.e_xc < 0.0 and r_uks.functional == "pbe"


def test_kuhf_m1_smeared_equals_krhf_smeared():
    """KUHF(M=1) with Fermi-Dirac smearing reproduces KRHF with the same
    smearing on H2/(2,1,1) to machine precision.

    The open-shell per-spin-μ machinery (apply_smearing_open_shell + the
    Mermin free energy / entropy threading) must collapse EXACTLY to the
    trusted closed-shell smearing at M=1: eps_α = eps_β and n_α = n_β give
    μ_α = μ_β and occ_α = occ_β, so 2·occ_α = occ_closed. This pins the new
    smearing wiring against the PySCF-µHa-parity closed-shell driver."""
    system, basis = _h2_box()
    o_r = _opts(); o_r.smearing_temperature = 0.01
    o_u = _opts(); o_u.smearing_temperature = 0.01
    r_krhf = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=o_r, use_compcell=True, **_COMMON
    )
    r_kuhf = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=o_u, **_COMMON
    )
    assert r_krhf.converged and r_kuhf.converged
    assert abs(r_kuhf.energy - r_krhf.energy) < 1e-8, (
        f"KUHF(M=1)+smear {r_kuhf.energy} != KRHF+smear {r_krhf.energy}"
    )
    assert abs(r_kuhf.free_energy - r_krhf.free_energy) < 1e-8
    assert abs(r_kuhf.entropy - r_krhf.entropy) < 1e-8
    assert r_kuhf.smearing_temperature == pytest.approx(0.01)
    # μ_α = μ_β at M=1; the two channels are identical.
    assert abs(r_kuhf.fermi_level_alpha - r_kuhf.fermi_level_beta) < 1e-8


def test_kuhf_smearing_open_shell_conserves_per_spin_counts():
    """Triplet H2 (nα=2, nβ=0) at (2,1,1) with smearing runs end-to-end
    through the per-spin Fermi path and conserves each channel's count
    (Σ_k w_k Σ_i n^σ_i = n_σ), with the exact triplet ⟨S²⟩."""
    system, basis = _h2_box()
    system.multiplicity = 3
    o = _opts(); o.smearing_temperature = 0.01
    r = run_kuhf_periodic_gdf(system, basis, kmesh=(2, 1, 1), options=o, **_COMMON)
    assert r.converged
    na = sum(float(w) * float(o_.sum())
             for w, o_ in zip(r.kpoint_weights, r.occupations_alpha))
    nb = sum(float(w) * float(o_.sum())
             for w, o_ in zip(r.kpoint_weights, r.occupations_beta))
    assert abs(na - 2.0) < 1e-7 and abs(nb) < 1e-9
    assert abs(r.s_squared - 2.0) < 1e-6


def test_kuhf_smearing_fractional_degenerate_open_shell():
    """A genuinely fractional open-shell occupation the per-k/per-spin hard
    Aufbau CANNOT represent: a boron atom's partially-filled, 3-fold
    degenerate 2p shell.

    B (5 e⁻, doublet ⇒ nα=3, nβ=2) in a cubic box. The α channel fills
    1s + 2s fully and puts ONE electron into the three symmetry-degenerate
    2p orbitals; the per-spin Fermi distributes it fractionally (≈1/3 each,
    summing to 1) instead of arbitrarily occupying one. Pins per-spin
    particle conservation, fractional occupations, and A = E - T·S < E."""
    box = 9.0
    system = vq.PeriodicSystem(3, np.diag([box] * 3), [vq.Atom(5, [0.0, 0.0, 0.0])])
    system.multiplicity = 2  # boron ground state is a doublet
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    o = _opts(); o.smearing_temperature = 0.01; o.max_iter = 200
    r = run_kuhf_periodic_gdf(system, basis, kmesh=(1, 1, 1), options=o, **_COMMON)
    assert r.converged
    occ_a = np.asarray(r.occupations_alpha[0])
    occ_b = np.asarray(r.occupations_beta[0])
    assert abs(occ_a.sum() - 3.0) < 1e-7, f"α count {occ_a.sum()} != 3"
    assert abs(occ_b.sum() - 2.0) < 1e-7, f"β count {occ_b.sum()} != 2"
    # The partially-filled degenerate 2p shell carries fractional α occ.
    assert np.any((occ_a > 1e-3) & (occ_a < 1.0 - 1e-3)), (
        f"expected fractional α occupation; got {occ_a}"
    )
    # Entropy is real and lowers the free energy below the energy.
    assert r.entropy > 1e-6
    assert r.free_energy < r.energy - 1e-9


def test_run_periodic_job_open_shell_smearing_via_kmesh(tmp_path):
    """run_periodic_job exposes per-spin open-shell smearing through the GDF
    dispatch — both multi-k (UHF + a k-mesh → run_kuhf_periodic_gdf) and Γ-only
    (UHF, no k-mesh → run_pbc_gdf_uhf). Both route a boron 2p¹ degenerate shell
    to genuinely fractional per-spin occupations."""
    from vibeqc.periodic_runner import run_periodic_job

    # Boron: a 3-fold-degenerate 2p¹ shell ⇒ genuinely fractional smearing.
    system = vq.PeriodicSystem(3, np.diag([9.0] * 3), [vq.Atom(5, [0, 0, 0])])
    system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    # (a) With a k-mesh, smearing flows through to the smearing-capable driver.
    r = run_periodic_job(
        system, basis, method="UHF", jk_method="gdf", kpoints=(1, 1, 1),
        aux_basis="def2-svp-jk", smearing_temperature=0.01,
        output=str(tmp_path / "b_smear"), max_iter=200, conv_tol_energy=1e-9,
        write_molden_file=False, initial_guess="HCORE",
    )
    assert isinstance(r, vq.PeriodicKUHFGDFResult)
    assert r.converged
    assert r.smearing_temperature == pytest.approx(0.01)
    occ_a = np.asarray(r.occupations_alpha[0])
    assert np.any((occ_a > 1e-3) & (occ_a < 1.0 - 1e-3)), (
        f"expected fractional α occupation through the runner; got {occ_a}"
    )

    # (b) Γ-only (kpoints=None) → the now-smeared Γ open-shell GDF driver.
    r_g = run_periodic_job(
        system, basis, method="UHF", jk_method="gdf",
        aux_basis="def2-svp-jk", smearing_temperature=0.01,
        output=str(tmp_path / "b_gamma"), max_iter=200, conv_tol_energy=1e-9,
        write_molden_file=False, initial_guess="HCORE",
    )
    assert isinstance(r_g, vq.PBCGDFUHFResult)
    assert r_g.converged
    assert r_g.smearing_temperature == pytest.approx(0.01)
    occ_a_g = np.asarray(r_g.occupations_alpha)
    assert np.any((occ_a_g > 1e-3) & (occ_a_g < 1.0 - 1e-3)), (
        f"expected fractional α occupation at Γ; got {occ_a_g}"
    )


def test_kuhf_multik_open_shell_converges_bounded():
    """A genuine multi-k open-shell SCF (triplet H2 at (2,1,1)) converges
    to a bounded energy with the exact triplet ⟨S²⟩. Absolute energy is
    NOT pinned against PySCF KUHF here (PySCF's BvK-supercell spin
    convention gives a different state for tiny magnetic cells — see the
    module docstring); the closed-shell-limit + Γ gates above carry the
    PySCF anchor."""
    system, basis = _h2_box()
    system.multiplicity = 3
    r = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), **_COMMON
    )
    assert r.converged
    assert -5.0 < r.energy < 5.0
    assert abs(r.s_squared - 2.0) < 1e-6
    # per-spin density traces: 2 α electrons, 0 β.
    na = sum(
        float(w) * float(np.real(np.trace(D @ S)))
        for w, D, S in zip(r.kpoint_weights, r.density_alpha, r.overlap)
    )
    nb = sum(
        float(w) * float(np.real(np.trace(D @ S)))
        for w, D, S in zip(r.kpoint_weights, r.density_beta, r.overlap)
    )
    assert abs(na - 2.0) < 1e-6 and abs(nb) < 1e-6


# --- fractional-occupation ⟨S²⟩ (smearing-consistent) -----------------------
# _multi_k_s_squared weights the α/β cross-spin overlap by the per-spin
# fractional occupations n^α_i n^β_j; at integer filling it must reduce to the
# first-n_σ hard cutoff (bit-identical), and under fractional occ it must match
# the closed-form ensemble-UHF value (Szabo & Ostlund Eq. 2.271 generalised).


def test_multi_k_s_squared_reduces_to_hardcut_at_integer_filling():
    """Integer occupations (∈ {0,1}) → the fractional branch equals the
    first-n_σ hard-cutoff value for ANY MOs / metric — the bit-identical
    guarantee for the T=0 / gapped path."""
    from vibeqc.periodic_k_gdf import _multi_k_s_squared

    rng = np.random.default_rng(0)
    nbf = 4
    S = np.eye(nbf)  # orthonormal metric ⇒ S-orthonormal MOs are unitary
    for _ in range(5):
        # Unitary α/β MO sets so ⟨S²⟩ is O(1) (physical MO scaling).
        Ca = np.linalg.qr(
            rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))
        )[0]
        Cb = np.linalg.qr(
            rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))
        )[0]
        na, nb = 3, 2
        s2_hard = _multi_k_s_squared(na, nb, [Ca], [Cb], [S], [1.0])
        oa = np.array([1.0, 1.0, 1.0, 0.0])
        ob = np.array([1.0, 1.0, 0.0, 0.0])
        s2_frac = _multi_k_s_squared(
            na, nb, [Ca], [Cb], [S], [1.0], occ_alpha_k=[oa], occ_beta_k=[ob]
        )
        assert abs(s2_hard - s2_frac) < 1e-12


def test_multi_k_s_squared_fractional_weighting_matches_closed_form():
    """Fractional occupations weight the overlap by n^α_i n^β_j; pin against a
    hand-computed closed form on a 2-orbital toy (S = I) and confirm it
    genuinely differs from the integer hard cutoff."""
    from vibeqc.periodic_k_gdf import _multi_k_s_squared

    theta = 0.6
    c, s = np.cos(theta), np.sin(theta)
    S = np.eye(2)
    Ca = np.eye(2, dtype=complex)  # α MOs = e0, e1
    Cb = np.array([[c, -s], [s, c]], dtype=complex)  # β MOs rotated by θ
    # M_ij = ⟨α_i|S|β_j⟩ = Cb; |M|² = [[c², s²], [s², c²]]
    na, nb = 1, 1
    oa = np.array([0.7, 0.3])
    ob = np.array([0.6, 0.4])
    s2 = _multi_k_s_squared(
        na, nb, [Ca], [Cb], [S], [1.0], occ_alpha_k=[oa], occ_beta_k=[ob]
    )
    overlap = 0.7 * 0.6 * c * c + 0.7 * 0.4 * s * s + 0.3 * 0.6 * s * s + 0.3 * 0.4 * c * c
    s2_ref = 0.25 * (na - nb) * (na - nb + 2) + nb - overlap
    assert abs(s2 - s2_ref) < 1e-12
    # The integer hard cutoff (occ → {1,0}) would subtract only c²:
    s2_hard = 0.25 * (na - nb) * (na - nb + 2) + nb - c * c
    assert abs(s2 - s2_hard) > 1e-3


# --- genuine metallic-band validation (uniform H-chain, band crossing E_F) ---
# The open-shell tests above use a gapped H2; this chain has NO Peierls gap
# (uniform 1.4-bohr H spacing) → a half-filled band crossing E_F, so smearing is
# essential and the BZ occupations are genuinely fractional. The heavy multi-k
# rsgdf metal SCFs are @slow (nightly/full gate), like the µHa compcell tests.


def _metal_h_chain(d_hh: float = 1.4, vac: float = 14.0):
    """Uniform H chain along z (period 2·d_hh, no dimerisation → metallic),
    vac-bohr vacuum in x,y. The half-filled band makes it a genuine metal."""
    c = vac / 2.0
    sysp = vq.PeriodicSystem(
        3, np.diag([vac, vac, 2.0 * d_hh]),
        [vq.Atom(1, [c, c, 0.0]), vq.Atom(1, [c, c, d_hh])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _metal_opts(temperature: float):
    o = vq.PeriodicRHFOptions()
    o.use_diis = True
    o.damping = 0.3  # metals oscillate without damping
    o.max_iter = 120
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-6
    o.lattice_opts.cutoff_bohr = 20.0
    o.lattice_opts.nuclear_cutoff_bohr = 20.0
    o.smearing_temperature = temperature
    return o


@pytest.mark.slow
def test_kuhf_m1_equals_krhf_on_metallic_chain():
    """Open-shell smearing reproduces closed-shell on a genuine METAL.

    Extends the M=1≡KRHF closed-shell-limit gate from the gapped H2 to a uniform
    H-chain whose band crosses E_F (fractional occupations, smearing essential),
    and pins the fractional-occupation ⟨S²⟩ = Σ_k w_k Σ_i n_i(1−n_i) — the spin
    contamination of a smeared single determinant (the pre-fix hard-cutoff
    formula wrongly reported 0 here, since α≡β at M=1).

    Exxdiv-convention-INDEPENDENT: it compares the open-shell driver to the
    closed-shell one under the SAME exxdiv and checks the smearing machinery —
    not an absolute reference. (Absolute PySCF parity on the metal is pinned by
    test_metallic_chain_krhf_smear_matches_pyscf below, now that the multi-k
    Coulomb-J conjugate over-binding is fixed.)"""
    sysp, basis = _metal_h_chain()
    r_krhf = run_krhf_periodic_gdf(
        sysp, basis, kmesh=(1, 1, 4), options=_metal_opts(0.01),
        use_compcell=True, **_COMMON,
    )
    r_kuhf = run_kuhf_periodic_gdf(
        sysp, basis, kmesh=(1, 1, 4), options=_metal_opts(0.01), **_COMMON,
    )
    assert r_krhf.converged and r_kuhf.converged
    # Genuinely metallic: fractional band occupations + finite entropy.
    occ = np.concatenate([np.asarray(o).ravel() for o in r_kuhf.occupations_alpha])
    assert np.any((occ > 1e-3) & (occ < 1.0 - 1e-3)), f"not metallic: {occ}"
    assert r_kuhf.entropy > 1e-3
    # Open-shell limit on the metal — machine precision (same exxdiv both sides).
    assert abs(r_kuhf.energy - r_krhf.energy) < 1e-7
    assert abs(r_kuhf.free_energy - r_krhf.free_energy) < 1e-7
    # Fractional-occupation spin contamination Σ_k w_k Σ_i n_i(1−n_i) (α≡β @ M=1).
    s2_expected = sum(
        float(w) * float(np.sum(np.asarray(o) * (1.0 - np.asarray(o))))
        for w, o in zip(r_kuhf.kpoint_weights, r_kuhf.occupations_alpha)
    )
    assert r_kuhf.s_squared > 1e-3
    assert abs(r_kuhf.s_squared - s2_expected) < 1e-6


# Out-of-process PySCF reference (§10): pyscf.pbc.scf.KRHF(cell, kpts)
# .density_fit(auxbasis='def2-svp-jkfit'); exxdiv='ewald';
# scf.addons.smearing_(mf, sigma=0.01, method='fermi'); the _metal_h_chain cell
# at kmesh (1,1,8); cell.unit='B'. mf.e_free (Mermin A = E − T·S) = −0.96320158.
_PYSCF_METAL_CHAIN_FREE_118 = -0.96320158


@pytest.mark.slow
def test_metallic_chain_krhf_smear_matches_pyscf():
    """Direct PySCF parity on a genuine metal — multi-k GDF free energy.

    Regression guard for the multi-k GDF Coulomb-J over-binding (§7): the
    per-k Hartree assembly ``ρ_P = Σ_k w_k tr(L(k,k)·D(k))`` was
    **conjugating** the diagonal cderi ``L(k,k)``. That is a no-op for the
    real cderi of vacuum-box / cubic insulators (so the H2 (2,1,1) gate and
    all cubic cells passed), but mis-contracts the Coulomb for the genuinely
    *complex* diagonal cderi of tight cells (inter-cell R≠0 overlap),
    over-binding E_J by Madelung-scale (~0.54 Ha here). It was NOT an exxdiv
    nor a smearing bug — exxdiv-K and the fractional-occupation energy
    expression both matched PySCF to <0.3 mHa. Fixed by dropping the spurious
    ``L_jj.conj()`` in run_krhf/kuhf_periodic_gdf._build_j_from_lpq, which
    reproduces PySCF ``get_j_kpts`` exactly. See handovers/HANDOVER_GDF_OUTSTANDING.md
    §9.2."""
    sysp, basis = _metal_h_chain()
    r = run_krhf_periodic_gdf(
        sysp, basis, kmesh=(1, 1, 8), options=_metal_opts(0.01),
        use_compcell=True, **_COMMON,
    )
    assert r.converged
    # Machine-precision match post-fix (Δ ≈ 3e-9); 1e-4 tolerates µHa AO-pair
    # FT thread-noise while still pinning the 0.54 Ha bug as fixed.
    assert abs(r.free_energy - _PYSCF_METAL_CHAIN_FREE_118) < 1e-4


# --- open-shell rsgdf high-|G| tail plumbing --------------------------------
# The closed-shell drivers have carried rsgdf_tail_ke_cutoff since the P01
# dense-core audit; the open-shell route was recorded as "no tail plumbing
# yet -- held unconditionally on the class" (HANDOVER_OPEN_BUGS_V015,
# production-k-sampling item). These pin the plumbed knob: the exact
# same-support identity, the parity-hold lift, and the fail-closed guard.


def test_kuhf_tail_completion_identity():
    """E(ke_cutoff=a, tail_ke_cutoff=b) == E(ke_cutoff=b) by construction
    (same total shifted support), now on the open-shell multi-k route.
    Triplet H2 at (2,1,1) exercises off-diagonal q tails through the
    per-spin exchange pairs."""
    system, basis = _h2_box()
    system.multiplicity = 3
    common = dict(aux_basis="def2-svp-jk", gdf_method="rsgdf", progress=False)
    r_tail = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(),
        rsgdf_ke_cutoff=20.0, rsgdf_tail_ke_cutoff=60.0, **common,
    )
    r_full = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(),
        rsgdf_ke_cutoff=60.0, **common,
    )
    assert r_tail.converged and r_full.converged
    assert abs(r_tail.energy - r_full.energy) < 1e-8
    assert abs(r_tail.s_squared - 2.0) < 1e-6


def test_kuhf_dense_core_hold_lifts_with_parity_tail(monkeypatch):
    """Dense-core cells warn + tag on the open-shell route at the untailed
    default, and a parity-sized rsgdf_tail_ke_cutoff lifts the hold --
    mirroring the closed-shell test. A sentinel replaces the cderi cache
    build so no MgO SCF runs; the warning must fire before the build."""
    import vibeqc.periodic_k_gdf as kgdf

    class _Sentinel(Exception):
        pass

    def _boom(*a, **k):
        raise _Sentinel()

    monkeypatch.setattr(kgdf, "_build_rsgdf_lpq_cache_shared_q", _boom)

    h = 3.979
    lat = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [h, h, h])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.warns(UserWarning, match="parity-held"):
        with pytest.raises(_Sentinel):
            run_kuhf_periodic_gdf(
                system, basis, kmesh=(1, 1, 2), progress=False
            )

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        with pytest.raises(_Sentinel):
            run_kuhf_periodic_gdf(
                system,
                basis,
                kmesh=(1, 1, 2),
                rsgdf_tail_ke_cutoff=3300.0,
                progress=False,
            )


def test_kuhf_tail_requires_rsgdf():
    """The tail is an rsgdf-fit concept; other gdf_methods fail closed."""
    system, basis = _h2_box()
    with pytest.raises(NotImplementedError, match="rsgdf"):
        run_kuhf_periodic_gdf(
            system,
            basis,
            kmesh=(1, 1, 1),
            gdf_method="mdf",
            rsgdf_tail_ke_cutoff=100.0,
            progress=False,
        )


def test_run_periodic_job_forwards_open_shell_multik_tail(monkeypatch, tmp_path):
    """run_periodic_job forwards rsgdf_tail_ke_cutoff to the open-shell
    multi-k drivers (it used to fail closed with 'not yet wired'). A
    sentinel captures the driver kwargs so no SCF runs."""
    import vibeqc.periodic_runner as prun

    class _Sentinel(Exception):
        pass

    captured: dict = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        raise _Sentinel()

    system, basis = _h2_box()
    system.multiplicity = 3

    monkeypatch.setattr(prun, "run_kuhf_periodic_gdf", _capture)
    with pytest.raises(_Sentinel):
        prun.run_periodic_job(
            system, basis, method="UHF", jk_method="gdf",
            kpoints=(2, 1, 1), aux_basis="def2-svp-jk",
            rsgdf_tail_ke_cutoff=60.0, write_molden_file=False,
            output=str(tmp_path / "multik_tail_uhf"),
        )
    assert captured.get("rsgdf_tail_ke_cutoff") == 60.0

    captured.clear()
    sysm, basis2 = _h2_box()
    monkeypatch.setattr(prun, "run_kuks_periodic_gdf", _capture)
    with pytest.raises(_Sentinel):
        prun.run_periodic_job(
            sysm, basis2, method="UKS", functional="pbe",
            jk_method="gdf", kpoints=(2, 1, 1),
            aux_basis="def2-svp-jk", rsgdf_tail_ke_cutoff=60.0,
            write_molden_file=False,
            output=str(tmp_path / "multik_tail_uks"),
        )
    assert captured.get("rsgdf_tail_ke_cutoff") == 60.0
