"""Converged one-electron image domains for periodic GDF.

A positive truncated Bloch overlap does not certify image convergence.
The automatic 3D path bounds omitted S/T integrals, while preserving an
explicit flat finite domain and projecting genuine near-null directions.
Independent NaCl matrices also cover the positive-but-inaccurate case.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import LatticeSumOptions, monkhorst_pack
from vibeqc.eigs_preflight import eigs_preflight
from vibeqc.linear_dependence import (
    LinearDependenceError,
    scf_preflight_overlap_check,
)
from vibeqc.periodic_k_gdf import (
    _PERIODIC_OVERLAP_HINT,
    _gdf_overlap_preflight,
    _is_pyscf_auto,
    _oneel_lattice_opts,
)
from vibeqc.periodic_rhf_multi_k_ewald import _canonical_orthogonalizer_complex

ANG2BOHR = 1.0 / 0.529177210903


def _mgo_primitive():
    """MgO rocksalt 2-atom primitive (FCC), a = 4.211 Å."""
    return _rocksalt_primitive(4.211, 12, 8)


def _rocksalt_primitive(a_ang: float, z_cation: int, z_anion: int, scale: float = 1.0):
    """Rocksalt 2-atom primitive (FCC), with ``a_ang`` in Å."""
    a = float(a_ang) * float(scale) * ANG2BOHR
    prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    o = 0.5 * (prim[0] + prim[1] + prim[2])
    return vq.PeriodicSystem(
        3,
        prim,
        [vq.Atom(int(z_cation), [0.0, 0.0, 0.0]), vq.Atom(int(z_anion), o.tolist())],
    )


def _si_primitive():
    """Si diamond 2-atom primitive (FCC), a = 5.431 Å."""
    a = 5.431 * ANG2BOHR
    prim = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    return vq.PeriodicSystem(
        3,
        prim,
        [
            vq.Atom(14, [0.0, 0.0, 0.0]),
            vq.Atom(14, [a / 4.0, a / 4.0, a / 4.0]),
        ],
    )


def _kpts(sysp, mesh):
    return np.asarray(monkhorst_pack(sysp, list(mesh)).kpoints, dtype=float)


class _NullLog:
    def info(self, *a, **k): pass
    def write_raw(self, *a, **k): pass
    def warning(self, *a, **k): pass


# ---------------------------------------------------------------------------
# _is_pyscf_auto strategy detection
# ---------------------------------------------------------------------------

def test_is_pyscf_auto_accepts_string_enum_and_rejects_flat():
    from vibeqc.lattice_screening import RcutStrategy

    assert _is_pyscf_auto("pyscf_auto")
    assert _is_pyscf_auto(RcutStrategy.PYSCF_AUTO)
    assert _is_pyscf_auto(str(RcutStrategy.PYSCF_AUTO))
    assert not _is_pyscf_auto("flat")
    assert not _is_pyscf_auto(RcutStrategy.FLAT)
    assert not _is_pyscf_auto(None)


# ---------------------------------------------------------------------------
# Converged tight bases; flat strategy opts out; user cutoff never lowered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis_name", ["sto-3g", "pob-tzvp-rev2"])
def test_tight_basis_cutoff_converges_overlap_and_kinetic(basis_name):
    """A larger independently summed domain cannot change S/T materially."""
    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), basis_name)
    base = LatticeSumOptions()  # cutoff_bohr == 15.0
    out = _oneel_lattice_opts(
        sysp, basis, base,
        rcut_strategy="pyscf_auto", k_points_cart=_kpts(sysp, (2, 2, 2)),
    )
    assert out.cutoff_bohr >= base.cutoff_bohr
    larger = LatticeSumOptions()
    larger.cutoff_bohr = 1.25*out.cutoff_bohr
    for builder in (vq.compute_overlap_lattice, vq.compute_kinetic_lattice):
        actual = builder(basis, sysp, out)
        reference = builder(basis, sysp, larger)
        for k in _kpts(sysp, (2, 2, 2)):
            np.testing.assert_allclose(vq.bloch_sum(actual, k), vq.bloch_sum(reference, k),
                                       atol=2e-11, rtol=0)


def test_flat_strategy_is_noop():
    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "def2-svp")
    base = LatticeSumOptions()
    out = _oneel_lattice_opts(
        sysp, basis, base,
        rcut_strategy="flat", k_points_cart=_kpts(sysp, (2, 2, 2)),
    )
    assert out is base  # untouched, no optimisation run


def test_explicit_larger_cutoff_not_lowered():
    """A user cutoff already large enough is never reduced."""
    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    base = LatticeSumOptions()
    base.cutoff_bohr = 40.0
    out = _oneel_lattice_opts(
        sysp, basis, base,
        rcut_strategy="pyscf_auto", k_points_cart=_kpts(sysp, (2, 2, 2)),
    )
    assert out.cutoff_bohr >= 40.0


# ---------------------------------------------------------------------------
# Diffuse-but-usable basis: grown until non-critical at all k
# ---------------------------------------------------------------------------

def test_diffuse_usable_basis_grown_to_psd():
    """def2-SVP with redundant diffuse primitives filtered is usable but
    non-PSD at 15 bohr on the zone boundary; the S/T cutoff is grown until
    S(k) is no longer critical at every k-point."""
    sysp = _mgo_primitive()
    mol = sysp.unit_cell_molecule()
    basis = vq.make_basis(mol, "def2-svp", exp_to_discard=0.1)
    kc = _kpts(sysp, (2, 2, 2))
    base = LatticeSumOptions()

    # Non-PSD at the flat default (worst at a zone-boundary k, not Γ).
    rep_flat = eigs_preflight(sysp, basis, kc, lattice_opts=base)
    assert rep_flat.worst_severity == "critical"

    out = _oneel_lattice_opts(
        sysp, basis, base, rcut_strategy="pyscf_auto", k_points_cart=kc
    )
    assert out.cutoff_bohr > base.cutoff_bohr + 1.0
    # base not mutated
    assert base.cutoff_bohr == pytest.approx(15.0)

    rep_grown = eigs_preflight(sysp, basis, kc, lattice_opts=out)
    assert rep_grown.worst_severity in ("ok", "warn", "error")
    assert sum(r.n_negative for r in rep_grown.per_k_reports) == 0


@pytest.mark.parametrize("mesh", [(2, 2, 2), (4, 4, 4), (6, 6, 6)])
def test_si_def2_svp_release_paper_meshes_grown_to_psd(mesh):
    """P04/P15 Si/def2-SVP meshes hit a false non-PSD S(k) at 15 bohr;
    the auto-grown one-electron cutoff must make them PSD before SCF."""
    sysp = _si_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "def2-svp")
    kc = _kpts(sysp, mesh)
    base = LatticeSumOptions()

    rep_flat = eigs_preflight(sysp, basis, kc, lattice_opts=base)
    assert rep_flat.worst_severity == "critical"
    assert rep_flat.worst_min_eigenvalue < 0.0

    out = _oneel_lattice_opts(
        sysp, basis, base, rcut_strategy="pyscf_auto", k_points_cart=kc
    )
    assert out.cutoff_bohr > base.cutoff_bohr

    rep_grown = eigs_preflight(sysp, basis, kc, lattice_opts=out)
    assert rep_grown.worst_severity in ("ok", "warn", "error")
    assert sum(r.n_negative for r in rep_grown.per_k_reports) == 0


def test_near_redundant_full_basis_grown_to_noncritical():
    """Full def2-SVP on MgO is near-linearly-dependent, but the original
    negative S(k) eigenvalue is a cutoff artefact. The cutoff must grow until
    S(k) is non-critical so the GDF canonical orthogonalizer can handle the
    near-null space."""
    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "def2-svp")
    base = LatticeSumOptions()
    out = _oneel_lattice_opts(
        sysp, basis, base,
        rcut_strategy="pyscf_auto", k_points_cart=_kpts(sysp, (2, 2, 2)),
    )
    assert out.cutoff_bohr > base.cutoff_bohr
    rep = eigs_preflight(sysp, basis, _kpts(sysp, (2, 2, 2)), lattice_opts=out)
    assert rep.worst_severity == "error"
    assert sum(r.n_negative for r in rep.per_k_reports) == 0


def test_p01k_mgo_def2_svp_full_basis_supported_after_cutoff_growth():
    """Release-paper P01k guard: full def2-SVP on tight MgO at 4x4x4 starts
    critical at 15 bohr, but automatic one-electron cutoff growth removes the
    non-PSD artefact and leaves only canonical-orthogonalizer work."""
    sysp = _mgo_primitive()
    mol = sysp.unit_cell_molecule()
    kc = _kpts(sysp, (4, 4, 4))
    base = LatticeSumOptions()

    full = vq.BasisSet(mol, "def2-svp")
    rep_full = eigs_preflight(sysp, full, kc, lattice_opts=base)
    assert rep_full.worst_severity == "critical"
    assert rep_full.worst_min_eigenvalue < 0.0

    opt_full = _oneel_lattice_opts(
        sysp,
        full,
        base,
        rcut_strategy="pyscf_auto",
        k_points_cart=kc,
    )
    assert opt_full.cutoff_bohr > base.cutoff_bohr

    rep_full_grown = eigs_preflight(sysp, full, kc, lattice_opts=opt_full)
    assert rep_full_grown.worst_severity == "error"
    assert sum(r.n_negative for r in rep_full_grown.per_k_reports) == 0

    S_lat = vq.compute_overlap_lattice(full, sysp, opt_full)
    worst_idx = next(
        i for i, rep in enumerate(rep_full_grown.per_k_reports)
        if rep.severity == "error"
    )
    # The GDF-specific preflight accepts error-level near-null S(k), because
    # the driver immediately canonical-orthogonalizes. The generic SCF
    # preflight still fails on the same matrix, preserving fail-closed behavior
    # outside this recovered route.
    S0 = np.asarray(vq.bloch_sum(S_lat, kc[worst_idx]))
    S0 = 0.5 * (S0 + S0.conj().T)
    assert _gdf_overlap_preflight(
        S0, plog=_NullLog(), label="S(k=0)", basis=full
    ).severity == "error"
    with pytest.raises(LinearDependenceError):
        scf_preflight_overlap_check(S0, plog=_NullLog(), label="S(k=0)")

    n_occ = sysp.n_electrons() // 2
    n_kept = []
    for k in kc:
        Sk = np.asarray(vq.bloch_sum(S_lat, k))
        Sk = 0.5 * (Sk + Sk.conj().T)
        _, kept = _canonical_orthogonalizer_complex(
            Sk, threshold=1e-7, normalize_diag_first=True
        )
        n_kept.append(kept)
    assert min(n_kept) >= n_occ
    assert min(n_kept) < full.nbasis


def test_p12_mgo_mdf_overlap_failure_is_superseded_by_dense_core_gate():
    """The archived P12 overlap abort is fixed; MDF physics still fails closed.

    P12 used rocksalt MgO at 4.212 Angstrom, full def2-SVP, RKS/PBE,
    ``gdf_method="mdf"``, and a Gamma-centred ``(4,4,4)`` mesh.  Its flat
    15-bohr S/T lattice sum is genuinely non-PSD, reproducing the v0.15.1
    failure, but the basis-aware cutoff growth removes every critical negative
    S(k) direction.  The exact MDF deck must then stop at the independent
    compact-dense-core validity gate, never regress to ``LinearDependenceError``
    and never enter an unstable SCF.
    """
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    sysp = _rocksalt_primitive(4.212, 12, 8)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "def2-svp")
    kc = _kpts(sysp, (4, 4, 4))
    base = LatticeSumOptions()

    rep_flat = eigs_preflight(sysp, basis, kc, lattice_opts=base)
    assert rep_flat.worst_severity == "critical"
    assert rep_flat.worst_min_eigenvalue < 0.0

    grown = _oneel_lattice_opts(
        sysp,
        basis,
        base,
        rcut_strategy="pyscf_auto",
        k_points_cart=kc,
    )
    rep_grown = eigs_preflight(sysp, basis, kc, lattice_opts=grown)
    assert grown.cutoff_bohr > base.cutoff_bohr
    assert sum(report.n_negative for report in rep_grown.per_k_reports) == 0

    options = vq.PeriodicKSOptions()
    options.max_iter = 1
    with pytest.raises(
        NotImplementedError,
        match=r"gdf_method='mdf'.*compact dense-core",
    ):
        run_krks_periodic_gdf(
            sysp,
            basis,
            (4, 4, 4),
            options,
            functional="pbe",
            gdf_method="mdf",
            progress=False,
        )


@pytest.mark.parametrize(
    "label,a_ang,z_cation,z_anion,scale",
    [
        ("MgO", 4.212, 12, 8, 0.96),
        ("MgO", 4.212, 12, 8, 1.00),
        ("MgO", 4.212, 12, 8, 1.04),
        ("LiF", 4.0351, 3, 9, 0.96),
        ("LiF", 4.0351, 3, 9, 1.00),
        ("LiF", 4.0351, 3, 9, 1.04),
    ],
)
def test_p14_eos_rocksalt_def2_svp_scaled_cells_grown_to_noncritical(
    label, a_ang, z_cation, z_anion, scale
):
    """Release-paper P14 EOS guard: scaled MgO/LiF def2-SVP cells are
    critical at the flat cutoff but heal under basis-aware S/T cutoff growth."""
    sysp = _rocksalt_primitive(a_ang, z_cation, z_anion, scale=scale)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "def2-svp")
    kc = _kpts(sysp, (4, 4, 4))
    base = LatticeSumOptions()

    rep_flat = eigs_preflight(sysp, basis, kc, lattice_opts=base)
    assert rep_flat.worst_severity == "critical", (label, scale)
    assert rep_flat.worst_min_eigenvalue < 0.0

    opt = _oneel_lattice_opts(
        sysp, basis, base, rcut_strategy="pyscf_auto", k_points_cart=kc
    )
    assert opt.cutoff_bohr > base.cutoff_bohr

    rep_grown = eigs_preflight(sysp, basis, kc, lattice_opts=opt)
    assert rep_grown.worst_severity in ("ok", "warn", "error")
    assert sum(r.n_negative for r in rep_grown.per_k_reports) == 0


# ---------------------------------------------------------------------------
# Diagnosis: periodic remediation hint replaces "upstream bug" framing
# ---------------------------------------------------------------------------

def test_periodic_remediation_hint_appended_on_abort():
    S = np.array([[1.0, 1.05], [1.05, 1.0]])  # min eig < 0
    with pytest.raises(LinearDependenceError) as ei:
        scf_preflight_overlap_check(
            S, plog=_NullLog(), label="S(k=0)",
            remediation_hint=_PERIODIC_OVERLAP_HINT,
        )
    msg = str(ei.value)
    assert "exp_to_discard" in msg
    assert "pob-tzvp-rev2" in msg


def test_periodic_hint_replaces_the_upstream_bug_framing():
    """A periodic caller must not be told to go hunting for a code defect.

    A truncated Bloch sum ``S(k) = sum_g e^{ik.R_g} S(g)`` keeps ``g`` and
    ``-g`` together, so it stays Hermitian --- but the Gram structure that
    forces positive-definiteness holds only for the *converged* sum. A
    negative eigenvalue at a loose cutoff is therefore an expected
    truncation artefact, and the first thing to try is a larger cutoff.
    The molecular path keeps the "upstream bug" wording, where it is
    correct because no lattice sum is involved.
    """
    S = np.array([[1.0, 1.05], [1.05, 1.0]])  # min eig < 0
    with pytest.raises(LinearDependenceError) as ei:
        scf_preflight_overlap_check(
            S, plog=_NullLog(), label="S(k=0)",
            remediation_hint=_PERIODIC_OVERLAP_HINT,
        )
    msg = str(ei.value)
    assert "upstream bug" not in msg
    assert "lattice cutoff being" in msg
    # It must still say what to suspect if raising the cutoff does not help.
    assert "stays negative as the cutoff grows" in msg


def test_truncated_bloch_overlap_recovers_positive_definiteness():
    """The physics behind that message, measured rather than asserted.

    ``min eig S(k)`` must climb back to a positive plateau as the lattice
    cutoff grows. This is what distinguishes a truncation artefact from a
    genuine defect in the image summing, and it is what the error message
    now tells users to check. CRYSTAL documents the same behaviour for its
    EIGS check: the more severe the computational conditions, the closer
    to zero an eigenvalue may sit without numerical risk, and negative
    values signal numerical linear dependence (CRYSTAL23 manual, EIGS).

    LiH/6-31G in a 12-bohr cube on a (4,4,2) mesh is a case that actually
    trips the guard at the default cutoff.
    """
    box = 12.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(3, [c, c, c - 1.55]), vq.Atom(1, [c, c, c + 1.55])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "6-31g")
    kmesh = vq.monkhorst_pack(sysp, [4, 4, 2])

    def _min_eig(cutoff):
        opts = vq.PeriodicKSOptions()
        opts.lattice_opts.cutoff_bohr = cutoff
        opts.lattice_opts.nuclear_cutoff_bohr = cutoff
        S_lat = vq.compute_overlap_lattice(basis, sysp, opts.lattice_opts)
        worst = np.inf
        for k in kmesh.kpoints:
            k_arr = np.asarray(k, dtype=float)
            S_k = np.zeros(
                (basis.nbasis, basis.nbasis), dtype=complex
            )
            for cell, blk in zip(S_lat.cells, S_lat.blocks):
                R = np.asarray(cell.r_cart, dtype=float)
                S_k += np.exp(1j * float(np.dot(k_arr, R))) * np.asarray(
                    blk, dtype=float
                )
            worst = min(worst, float(np.linalg.eigvalsh(S_k).min()))
        return worst

    tight = _min_eig(12.0)
    grown = _min_eig(20.0)
    converged = _min_eig(32.0)
    # Trips the guard at the tight cutoff...
    assert tight < -1e-6, tight
    # ...recovers once the sum is converged, and stays put.
    assert grown > 0.0, grown
    assert converged > 0.0, converged
    assert abs(converged - grown) < 5e-3, (grown, converged)


def test_generic_message_unchanged_without_hint():
    """Molecular callers (no hint) keep the original wording."""
    S = np.array([[1.0, 1.05], [1.05, 1.0]])
    with pytest.raises(LinearDependenceError) as ei:
        scf_preflight_overlap_check(S, plog=_NullLog(), label="S(Γ)")
    msg = str(ei.value)
    assert "exp_to_discard" not in msg
    assert "upstream bug" in msg


@pytest.mark.parametrize('geometry_index', [0, 1])
def test_positive_overlap_still_requires_converged_images_against_external_reference(geometry_index):
    import hashlib
    import json
    import tomllib
    from pathlib import Path
    from vibeqc import _vibeqc_core as core

    root = Path(__file__).parent / 'data/periodic_gdf'
    record = json.loads((root/'nacl_ecp_reference.json').read_text())
    path = root/'nacl_ecp_reference.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['arrays_sha256']
    manifest = tomllib.loads((root/'nacl_ecp_reference.system').read_text())
    assert manifest['program'] == {'name': 'PySCF', 'version': '2.6.2'}
    geometry = record['geometries'][geometry_index]
    system = vq.PeriodicSystem(3, np.asarray(geometry['lattice']).T, [
        vq.Atom(11 if symbol == 'Na' else 17, center) for symbol, center in geometry['atoms']
    ])
    shells, permutation, offset = [], [], 0
    for atom, (symbol, center) in enumerate(geometry['atoms']):
        for row in record['provenance']['basis'][symbol]:
            angular = int(row[0])
            primitives = np.asarray(row[1:])
            for contraction in range(1, primitives.shape[1]):
                shells.append(core.ShellInfo(
                    atom, angular, True, primitives[:, 0].tolist(),
                    primitives[:, contraction].tolist(), center,
                ))
                order = [1, 2, 0] if angular == 1 else list(range(2*angular+1))
                permutation.extend(offset+i for i in order)
                offset += len(order)
    basis = core.BasisSet(system.unit_cell_molecule(), shells, 'lanl2dz', False)
    assert basis.nbasis == 16
    with np.load(path, allow_pickle=False) as stored:
        reference = {name: stored[f'geometry_{geometry_index}_{name}'][0][np.ix_(permutation, permutation)]
                     for name in ('S', 'T')}
    base = LatticeSumOptions()
    # The old native PSD search stopped at 24.375 bohr for this cell.
    # PySCF's rcut uses a different image enumeration and its positive
    # 15 bohr witness cannot be substituted for this native cutoff.
    base.cutoff_bohr = 24.375
    truncated = np.asarray(vq.bloch_sum(vq.compute_overlap_lattice(basis, system, base), np.zeros(3)))
    if geometry_index == 0:
        assert np.linalg.eigvalsh(truncated)[0] > 0
    assert np.max(np.abs(truncated-reference['S'])) > .01
    options = _oneel_lattice_opts(system, basis, base, rcut_strategy='pyscf_auto',
                                  k_points_cart=np.zeros((1, 3)))
    for name, builder in (('S', vq.compute_overlap_lattice), ('T', vq.compute_kinetic_lattice)):
        actual = np.asarray(vq.bloch_sum(builder(basis, system, options), np.zeros(3)))
        np.testing.assert_allclose(actual, reference[name], atol=2e-11, rtol=0)


def test_oneel_memory_rejects_before_image_enumeration(monkeypatch):
    from vibeqc import memory, _vibeqc_core as core
    from vibeqc.periodic_k_gdf import _preflight_gdf_oneel_memory

    system = _mgo_primitive()
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    options = LatticeSumOptions()
    def unexpected(*args, **kwargs):
        pytest.fail('image enumeration preceded the one-electron reservation')
    monkeypatch.setattr(core, 'direct_lattice_cells', unexpected)
    monkeypatch.setattr(memory, 'available_memory_bytes', lambda: 1)
    with pytest.raises(MemoryError, match='AO images were not allocated'):
        _preflight_gdf_oneel_memory(system, basis, options)
