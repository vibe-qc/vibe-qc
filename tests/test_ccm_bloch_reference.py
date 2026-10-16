"""The Bloch producer's SCF as the real supercell-Γ correlation reference.

Speed-up prompt 4 of the AICCM speed-up series (2026-09-25),
Theorem 1 of the Γ-CCM paper made executable
(``python/vibeqc/periodic/ccm/bloch_reference.py``): the Bloch producer
``run_ccm_rhf_gdf`` (``N_c`` eigenproblems of size ``n``) is folded onto the
real supercell -- block-circulant fold of overlap, core Hamiltonian, Fock
and density, real S-orthonormal canonical orbitals from each time-reversal
pair or self-paired mesh point -- and handed to the real-space correlated
methods as the same object the real-Γ producer ``run_ccm_rhf_direct`` (one
eigenproblem of size ``N_c·n``) returns.

Pinned here, per the prompt: the correlation energies from the two
references agree to 1e-9 Ha/cell on LiH and MgO STO-3G at (2,1,1) and
(2,2,1) and on the H2 6-bohr cube (MP2, DLPNO-MP2 at zero truncation and,
on the cube, CCSD(T) and UMP2), with the direct reference converged to
``conv_tol_grad=1e-9`` (its default 1e-6 leaves canonical MP2 1e-8 off on
LiH (2,2,1), measured in the paper campaign). The real-Γ producer stays the
independent validation producer of ruling R1; nothing here replaces it.

The transform's own checks (fold realness, orthonormality, Fock diagonality,
eigenvalues, density, electron count, integer occupations) run inside
``ccm_reference_from_bloch`` and raise on failure, so every reference used
below passed them; the synthetic test at the top exercises the linear
algebra alone, with non-TRIM ``±k`` pairs and exact degeneracies across k,
in milliseconds and independently of the machine's load.
"""

from __future__ import annotations

import copy
import dataclasses
import tomllib

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.bloch_reference import (
    BLOCH_REFERENCE_TOLERANCES,
    as_real_reference,
    ccm_reference_from_bloch,
    fold_block_circulant,
    real_supercell_orbitals,
)
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.direct import (
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
)
from vibeqc.periodic.ccm.dlpno import ccm_dlpno_mp2
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rks_gdf, run_ccm_uhf_gdf
from vibeqc.periodic.ccm.scf import CCMSCFResult
from vibeqc.periodic.ccm.uhf import CCMUHFResult
from vibeqc.periodic.ccm.ump2 import run_ccm_ump2

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

ANG = 1.0 / 0.529177210903
#: The prompt's rule: correlation energies from the two references, per unit cell.
TOL_CORR = 1e-9
#: Theorem 1's measured class on the SCF energy per unit cell (campaign 4.4).
TOL_SCF = 1e-10


def _h2_cube():
    """Compact H2 in a 6-bohr cube, STO-3G: the fold's historical fixture."""
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _rocksalt(a_ang, z_a, z_b):
    """fcc primitive cell of a rocksalt AB crystal (lattice vectors as COLUMNS)."""
    a = a_ang * ANG
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return PeriodicSystem(
        3, lat, [Atom(z_a, [0.0, 0.0, 0.0]),
                 Atom(z_b, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1)


#: The paper's fixed CRYSTAL23 deck geometries (Computational Details).
SYSTEMS = {"LiH": (4.0840, 3, 1), "MgO": (4.2239, 12, 8)}


def _mesh_id(m):
    return "x".join(map(str, m))


def _bloch_options(ks=False):
    """The multi-k driver converged tighter than its defaults, so the Bloch
    side is at least as converged as the direct side below."""
    opts = PeriodicKSOptions() if ks else PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 200
    return opts


#: The direct producer at the orbital convergence the identity needs.
DIRECT_KW = dict(conv_tol=1e-11, conv_tol_grad=1e-9, max_iter=200)


def _bloch(ccm, **kw):
    return run_ccm_rhf_gdf(ccm, retain_fit_cache=True, options=_bloch_options(), **kw)


def _direct(ccm, **kw):
    return run_ccm_rhf_direct(ccm, **DIRECT_KW, **kw)


def _per_cell(total, ccm):
    return float(total) / int(ccm.n_cells)


# ---------------------------------------------------------------------------
# The linear algebra alone: a synthetic block-circulant operator pair.
# ---------------------------------------------------------------------------

def _synthetic_pair(ccm, n, rng, coupling):
    """Random real symmetric block-circulant ``(F_sc, S_sc)`` on the torus
    (``S_sc`` positive definite) and their Bloch transforms per mesh point.

    ``coupling=0`` switches every inter-cell block off, so all ``N_c`` copies
    of the cell are exactly degenerate: the degeneracy-across-k case.
    """
    nrep = tuple(int(x) for x in ccm.nrep)
    n1, n2, n3 = nrep
    cells = [(i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)]
    n_c = len(cells)

    def blocks(symmetric_shift):
        out = {}
        for g in cells:
            mg = tuple((-x) % N for x, N in zip(g, nrep))
            if mg in out:
                out[g] = out[mg].T
                continue
            A = rng.standard_normal((n, n))
            if g == (0, 0, 0):
                A = 0.5 * (A + A.T) + symmetric_shift * np.eye(n)
            elif mg == g:
                A = 0.5 * (A + A.T)
            if g != (0, 0, 0):
                A = coupling * 0.3 * A
            out[g] = A
        return out

    def supercell(bl):
        M = np.zeros((n_c * n, n_c * n))
        for ci, c in enumerate(cells):
            for cj, d in enumerate(cells):
                g = tuple((y - x) % N for x, y, N in zip(c, d, nrep))
                M[ci * n:(ci + 1) * n, cj * n:(cj + 1) * n] = bl[g]
        return M

    F_sc = supercell(blocks(0.0))
    B = supercell(blocks(0.0))
    S_sc = B @ B.T / n + 1.5 * np.eye(n_c * n)      # block-circulant, SPD
    a_rows = np.asarray(ccm.unit_vectors, float)
    b_rows = 2.0 * np.pi * np.linalg.inv(a_rows).T   # a_i · b_j = 2π δ_ij
    kpts = np.array([(np.array(c) / np.array(nrep)) @ b_rows for c in cells])
    r_c = np.array(cells) @ a_rows

    def bloch(M_sc):
        out = []
        for k in kpts:
            Mk = np.zeros((n, n), dtype=complex)
            for cj, d in enumerate(cells):
                Mk += np.exp(1j * (k @ r_c[cj])) * M_sc[0:n, cj * n:(cj + 1) * n]
            out.append(Mk)
        return out
    return F_sc, S_sc, kpts, bloch(F_sc), bloch(S_sc)


def _generalised_eigh(F, S):
    w, U = np.linalg.eigh(S)
    X = U / np.sqrt(w)
    e, V = np.linalg.eigh(X.conj().T @ F @ X)
    return e, X @ V


@pytest.mark.parametrize("coupling", [1.0, 0.0], ids=["coupled", "degenerate-across-k"])
@pytest.mark.parametrize("mesh", [(3, 1, 1), (2, 2, 1), (3, 2, 1)], ids=_mesh_id)
def test_fold_and_realification_on_a_synthetic_operator(mesh, coupling):
    ccm = CCMSystem(_h2_cube(), mesh, "sto-3g")
    n = 4
    F_sc, S_sc, kpts, F_k, S_k = _synthetic_pair(ccm, n, np.random.default_rng(7), coupling)
    # The fold of the Bloch transforms is the supercell operator.
    S_fold, imag_s = fold_block_circulant(ccm, kpts, S_k)
    F_fold, imag_f = fold_block_circulant(ccm, kpts, F_k)
    assert imag_s < 1e-12 and imag_f < 1e-12
    assert np.max(np.abs(S_fold - S_sc)) < 1e-12
    assert np.max(np.abs(F_fold - F_sc)) < 1e-12
    # Per-k eigenpairs, as the driver would produce them.
    eps_k, C_k = zip(*(_generalised_eigh(F, S) for F, S in zip(F_k, S_k)))
    C, eps, eps_src, labels, blocks = real_supercell_orbitals(
        ccm, kpts, list(C_k), list(eps_k), S_sc, F_sc)
    n_c = int(ccm.n_cells)
    assert C.dtype == float and C.shape == (n_c * n, n_c * n)
    assert np.max(np.abs(C.T @ S_sc @ C - np.eye(n_c * n))) < 1e-11
    fmo = C.T @ F_sc @ C
    assert np.max(np.abs(fmo - np.diag(np.diag(fmo)))) < 1e-11
    assert np.max(np.abs(np.diag(fmo) - eps)) < 1e-11
    e_ref, _ = _generalised_eigh(F_sc, S_sc)
    assert np.max(np.abs(np.sort(eps) - e_ref)) < 1e-11
    assert np.max(np.abs(eps - eps_src)) < 1e-9
    kinds = {k: sum(1 for lab in labels if lab[1] == k) for k in ("trim", "pair")}
    n_trim = sum(1 for m in np.ndindex(*mesh)
                 if all((2 * x) % N == 0 for x, N in zip(m, mesh)))
    assert kinds == {"trim": n_trim * n, "pair": (n_c - n_trim) * n}
    if coupling == 0.0:
        # every level is N_c-fold degenerate across the mesh
        assert max(blocks) >= 1 and np.max(np.abs(np.diff(np.sort(eps)).reshape(-1)[:n_c - 1])) < 1e-9


# ---------------------------------------------------------------------------
# The H2 cube: every treatment, the fit-cache and refit cderis, the hook.
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _h2_bloch(mesh):
    key = ("h2", mesh)
    if key not in _CACHE:
        ccm = CCMSystem(_h2_cube(), mesh, "sto-3g")
        _CACHE[key] = (ccm, _bloch(ccm))
    return _CACHE[key]


@pytest.mark.parametrize("mesh", [(2, 1, 1), (3, 1, 1), (2, 2, 1)], ids=_mesh_id)
def test_h2_cube_correlation_identity_between_the_two_references(mesh):
    ccm, g = _h2_bloch(mesh)
    n_c = int(ccm.n_cells)
    ref = ccm_reference_from_bloch(ccm, g)
    assert isinstance(ref, CCMSCFResult)
    rep = ref.bloch_transform
    assert rep.cderi_build == "driver-fit-cache"
    assert rep.n_k == n_c and rep.n_orbitals == int(ccm.nbf)
    n_ao = int(ccm.nbf) // n_c
    if mesh == (3, 1, 1):
        # k = ±1/3 b1 is a genuine ±k pair; only Γ is self-paired.
        assert rep.kinds == {"trim": n_ao, "pair": 2 * n_ao}
    else:
        assert rep.kinds == {"trim": n_c * n_ao, "pair": 0}
    for key in ("fock_offdiag_max", "orthonormality_max_dev", "density_max_dev"):
        assert getattr(rep, key) < 1e-10, (key, getattr(rep, key))
    assert ref.backend.endswith("+bloch-fold")
    assert ref.fit_source == "classic" and ref.exchange_q0

    d = _direct(ccm)
    assert abs(ref.energy - d.energy) / n_c < TOL_SCF
    assert np.max(np.abs(ref.mo_energies - d.mo_energies)) < 1e-7

    m_b, m_d = run_ccm_mp2(ccm, ref), run_ccm_mp2(ccm, d)
    assert abs(m_b.e_correlation - m_d.e_correlation) / n_c < TOL_CORR
    l_b, l_d = ccm_dlpno_mp2(ccm, ref), ccm_dlpno_mp2(ccm, d)
    assert abs(l_b.e_corr - l_d.e_corr) / n_c < TOL_CORR
    c_b, c_d = run_ccm_ccsd(ccm, ref), run_ccm_ccsd(ccm, d)
    assert abs(c_b.e_correlation - c_d.e_correlation) / n_c < TOL_CORR
    assert abs(c_b.e_t - c_d.e_t) / n_c < TOL_CORR

    # The entry points take the Bloch result directly and transform it
    # themselves: the same numbers, bit for bit.
    m_h = run_ccm_mp2(ccm, g)
    assert m_h.e_correlation == pytest.approx(m_b.e_correlation, abs=1e-13)
    assert m_h.scf.bloch_transform is not None
    assert m_h.e_hf == pytest.approx(ref.energy, abs=1e-13)


def test_h2_cube_refit_fold_equals_the_fit_cache_fold_and_scf_only_transform():
    ccm, g = _h2_bloch((2, 1, 1))
    ref = ccm_reference_from_bloch(ccm, g)
    # Without a retained fit the fold refits at the driver's operator.
    g2 = run_ccm_rhf_gdf(ccm, options=_bloch_options())
    assert g2.fit_cache is None
    ref2 = ccm_reference_from_bloch(ccm, g2)
    assert ref2.bloch_transform.cderi_build == "fold"
    # The fit-cache fold carries the pairs in the driver's auxiliary frame
    # (more rows), the refit in the fold's canonical frame: one operator,
    # compared on the four-index contraction it represents.
    assert ref2.cderi.shape[1:] == ref.cderi.shape[1:]
    g_cache = np.einsum("Pmn,Pls->mnls", ref.cderi, ref.cderi)
    g_refit = np.einsum("Pmn,Pls->mnls", ref2.cderi, ref2.cderi)
    assert np.max(np.abs(g_cache - g_refit)) < 1e-10
    assert abs(ref2.energy - ref.energy) < 1e-9
    # An SCF-only transform carries no cderi, and the entry points then
    # cannot correlate on it without one.
    ref3 = ccm_reference_from_bloch(ccm, g, build_cderi=False)
    assert ref3.cderi is None and ref3.bloch_transform.cderi_build is None
    # A caller's cderi is stamped as such.
    ref4 = ccm_reference_from_bloch(ccm, g, cderi=ref.cderi)
    assert ref4.cderi_info["build"] == "caller"


def test_h2_cube_open_shell_transform_and_ump2_identity():
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    n_c = int(ccm.n_cells)
    gu = run_ccm_uhf_gdf(ccm, retain_fit_cache=True, options=_bloch_options())
    refu = ccm_reference_from_bloch(ccm, gu)
    assert isinstance(refu, CCMUHFResult)
    assert refu.bloch_transform.n_occupied == (2, 2)
    assert refu.bloch_transform.cderi_build == "driver-fit-cache"
    assert abs(refu.s_squared) < 1e-8
    du = run_ccm_uhf_direct(ccm, **DIRECT_KW)
    assert abs(refu.energy - du.energy) / n_c < TOL_SCF
    u_b, u_d = run_ccm_ump2(ccm, refu), run_ccm_ump2(ccm, du)
    assert abs(u_b.e_correlation - u_d.e_correlation) / n_c < TOL_CORR
    # and UMP2 on the singlet is the RHF MP2
    _, g = _h2_bloch((2, 1, 1))
    m = run_ccm_mp2(ccm, g)
    assert abs(u_b.e_correlation - m.e_correlation) / n_c < TOL_CORR


def test_refusals_and_pass_through():
    ccm, g = _h2_bloch((2, 1, 1))
    ref = ccm_reference_from_bloch(ccm, g, build_cderi=False)
    # a real-Γ reference passes through the hook untouched
    assert as_real_reference(ccm, ref) is ref
    assert as_real_reference(ccm, None) is None
    # another fit source is another operator: refused before any fold
    other = dataclasses.replace(g, fit_source="sr")
    with pytest.raises(ValueError, match="fit_source"):
        ccm_reference_from_bloch(ccm, other)
    assert ccm_reference_from_bloch(ccm, other, build_cderi=False).cderi is None
    # the producer keeps the fold's classic pair support, and another
    # support is another operator, refused before any fold (#466)
    source = g.raw.executed_fit_source
    support = dict(source.numerical_support)
    assert support["classic_support_policy"] == "historical_index_ball"
    # executed_fit_source is attached after construction, not a field
    raw = copy.copy(g.raw)
    raw.executed_fit_source = dataclasses.replace(source, numerical_support=tuple(
        (key, "physical_pairs" if key == "classic_support_policy" else value)
        for key, value in source.numerical_support))
    physical = dataclasses.replace(g, raw=raw)
    with pytest.raises(ValueError, match="AO-pair support"):
        ccm_reference_from_bloch(ccm, physical)
    assert ccm_reference_from_bloch(ccm, physical, build_cderi=False).cderi is None
    # the result must belong to this torus
    with pytest.raises(ValueError, match="k-points"):
        ccm_reference_from_bloch(CCMSystem(_h2_cube(), (2, 2, 1), "sto-3g"), g)
    # not a Bloch result at all
    with pytest.raises(TypeError, match="Bloch producer"):
        ccm_reference_from_bloch(ccm, ref)
    # an unconverged SCF is refused
    with pytest.raises(ValueError, match="did not converge"):
        ccm_reference_from_bloch(ccm, dataclasses.replace(
            g, raw=dataclasses.replace(g.raw, converged=False)))
    # a tolerance key that does not exist
    with pytest.raises(ValueError, match="tolerance"):
        ccm_reference_from_bloch(ccm, g, build_cderi=False, tolerances={"nope": 1.0})
    assert set(BLOCH_REFERENCE_TOLERANCES) >= {"fock_offdiag", "density", "occupation"}


# ---------------------------------------------------------------------------
# The paper's cells: MP2 and DLPNO-MP2 on LiH and MgO at (2,1,1) and (2,2,1).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh", [(2, 1, 1), (2, 2, 1)], ids=_mesh_id)
@pytest.mark.parametrize("name", ["LiH", "MgO"])
def test_rocksalt_correlation_identity_between_the_two_references(name, mesh):
    a_ang, z_a, z_b = SYSTEMS[name]
    ccm = CCMSystem(_rocksalt(a_ang, z_a, z_b), mesh, "sto-3g")
    n_c = int(ccm.n_cells)
    g = _bloch(ccm)
    ref = ccm_reference_from_bloch(ccm, g)
    rep = ref.bloch_transform
    assert rep.cderi_build == "driver-fit-cache"
    # The transform already refused anything above its tolerances; the
    # Fock diagonality is at the 1e-15 class, the density deviation is the
    # SCF residual of the driver's mixed density (1.9e-9 on LiH (2,1,1)).
    assert rep.fock_offdiag_max < 1e-9
    assert rep.density_max_dev < BLOCH_REFERENCE_TOLERANCES["density"]
    d = _direct(ccm)
    assert abs(ref.energy - d.energy) / n_c < TOL_SCF
    m_b, m_d = run_ccm_mp2(ccm, ref), run_ccm_mp2(ccm, d)
    assert abs(m_b.e_correlation - m_d.e_correlation) / n_c < TOL_CORR
    l_b, l_d = ccm_dlpno_mp2(ccm, ref), ccm_dlpno_mp2(ccm, d)
    assert abs(l_b.e_corr - l_d.e_corr) / n_c < TOL_CORR


# ---------------------------------------------------------------------------
# Closed-shell KS: the transform covers RKS, and MP2 on the KS orbitals agrees.
# ---------------------------------------------------------------------------

def test_lih_pbe_reference_transforms_and_agrees_with_the_direct_ks():
    a_ang, z_a, z_b = SYSTEMS["LiH"]
    ccm = CCMSystem(_rocksalt(a_ang, z_a, z_b), (2, 1, 1), "sto-3g")
    n_c = int(ccm.n_cells)
    gk = run_ccm_rks_gdf(ccm, "pbe", retain_fit_cache=True, options=_bloch_options(ks=True))
    refk = ccm_reference_from_bloch(ccm, gk)
    # A pure functional fits only the diagonal (k, k) pairs, so its retained
    # cache cannot serve the fold and the cderi is refitted at the operator.
    assert refk.bloch_transform.cderi_build == "fold"
    assert refk.exchange_q0 is None  # pure functional: no exact exchange, no seam
    dk = run_ccm_rks_direct(ccm, "pbe", **DIRECT_KW)
    assert abs(refk.energy - dk.energy) / n_c < TOL_SCF
    assert np.max(np.abs(refk.mo_energies - dk.mo_energies)) < 1e-7
    # The same MP2 functional of the two KS references (what a double hybrid
    # evaluates), on the same folded cderi.
    m_b, m_d = run_ccm_mp2(ccm, refk), run_ccm_mp2(ccm, dk)
    assert abs(m_b.e_correlation - m_d.e_correlation) / n_c < TOL_CORR


# ---------------------------------------------------------------------------
# The front door: variant="neutral-bloch" with correlation=.
# ---------------------------------------------------------------------------

def test_runner_neutral_bloch_arm_correlates_on_the_folded_reference(tmp_path):
    from vibeqc import BasisSet, run_periodic_job

    ccm, g = _h2_bloch((2, 1, 1))
    stem = tmp_path / "h2_neutral_bloch_mp2"
    system = _h2_cube()
    res = run_periodic_job(
        system, BasisSet(system.unit_cell_molecule(), "sto-3g"),
        method="aiccm", variant="neutral-bloch",
        aiccm_lattice_extension=(2, 1, 1), correlation="mp2", output=str(stem),
        progress=False, citations=True, record_hostname=False,
        write_xyz_file=False, write_cif_file=False,
        write_xsf_structure_file=False, write_molden_file=False)
    assert res.converged and res.correlation is not None
    assert res.correlation.backend == "aiccm2026dev-a-ri-mp2"
    assert res.bloch_transform.cderi_build == "driver-fit-cache"
    # the library number, per cell, from the same producer
    lib = run_ccm_mp2(ccm, g)
    n_c = int(ccm.n_cells)
    assert res.e_correlation == pytest.approx(lib.e_correlation / n_c, abs=1e-8)
    assert res.e_total_correlated == pytest.approx(
        (lib.e_hf + lib.e_correlation) / n_c, abs=1e-8)
    out = (tmp_path / "h2_neutral_bloch_mp2.out").read_text()
    assert "AICCM correlation" in out
    assert "real-Gamma orbitals folded from the Bloch SCF (Theorem 1)" in out
    manifest = tomllib.loads((tmp_path / "h2_neutral_bloch_mp2.system").read_text())
    run = manifest["run"]
    assert run["aiccm_correlation"] == "mp2"
    assert run["aiccm_correlation_reference"].startswith("bloch-fold")
    assert run["aiccm_bloch_fold_cderi_build"] == "driver-fit-cache"
    assert float(run["aiccm_bloch_fold_fock_offdiag_max"]) < 1e-10
    assert run["rsgdf_fit_source_executed"] == "classic"
