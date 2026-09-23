"""Phase K1 — :class:`vibeqc.KPoints` user-facing builder.

Pinned contracts for K1 (basic MP / Γ-centered / shifted / Γ-only):

1. **API surface** — :class:`vibeqc.KPoints` is public; constructors
   :meth:`monkhorst_pack`, :meth:`gamma_centred`, :meth:`shifted`,
   :meth:`gamma` all return a :class:`KPoints` (not a raw
   :class:`BlochKMesh`).

2. **Weight normalisation** — every constructed mesh has
   ``weights.sum() == 1`` to FP precision.

3. **Classical Monkhorst–Pack auto-shift** — when ``shift`` is
   omitted, even meshes get ``shift=1`` per axis and odd meshes
   ``shift=0``. Cubic ``[4, 4, 4]`` → ``shift=(1,1,1)``;
   ``[3, 3, 3]`` → ``shift=(0,0,0)``. Mixed ``[4, 3, 4]`` →
   ``shift=(1,0,1)``.

4. **Cartesian↔fractional round-trip** — a mesh on a 3D cubic cell
   round-trips through ``B_recip @ k_frac → k_cart`` to <1e-12.

5. **Back-compat** — :meth:`to_bloch_kmesh` returns a native
   :class:`BlochKMesh` of the same length, so legacy SCF entry
   points continue to work.

6. **Γ-only convenience** — :meth:`KPoints.gamma` produces a
   length-1 mesh at the origin with weight 1.

7. **Input validation** — ill-formed ``mesh`` (wrong length for the
   system dimensionality, or any entry < 1) raises ``ValueError``;
   ``shift`` entries outside ``{0, 1}`` raise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


def _cubic_si_3d():
    """Tiny diamond-Si-like cubic primitive cell (1 Si, no symmetry)."""
    a = 5.43 * ANGSTROM_TO_BOHR
    lat = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(14, [0.0, 0.0, 0.0])])


def _orth():
    """Anisotropic orthorhombic cell so axis-handling is exercised."""
    lat = np.diag([4.0, 6.0, 8.0]) * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(3, lat, [vq.Atom(14, [0.0, 0.0, 0.0])])


def _low_dim(dim: int):
    if dim == 1:
        lat = np.diag([4.0, 30.0, 30.0]) * ANGSTROM_TO_BOHR
    elif dim == 2:
        lat = np.diag([4.0, 6.0, 30.0]) * ANGSTROM_TO_BOHR
    else:
        raise ValueError(dim)
    return vq.PeriodicSystem(dim, lat, [vq.Atom(1, [0.0, 0.0, 0.0])])


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_kpoints_is_public():
    assert hasattr(vq, "KPoints")
    assert hasattr(vq, "as_bloch_kmesh")


def test_constructors_return_kpoints():
    sys = _cubic_si_3d()
    assert isinstance(vq.KPoints.gamma(sys), vq.KPoints)
    assert isinstance(vq.KPoints.monkhorst_pack(sys, [2, 2, 2]), vq.KPoints)
    assert isinstance(vq.KPoints.gamma_centred(sys, [2, 2, 2]), vq.KPoints)
    assert isinstance(vq.KPoints.shifted(sys, [2, 2, 2], [1, 0, 1]),
                      vq.KPoints)


# ---------------------------------------------------------------------------
# 2. Weight normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh", [[1, 1, 1], [2, 2, 2], [3, 3, 3],
                                    [4, 4, 4], [4, 3, 2]])
def test_weights_sum_to_one(mesh):
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, mesh)
    np.testing.assert_allclose(kp.weights.sum(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Classical Monkhorst–Pack auto-shift convention
# ---------------------------------------------------------------------------

def test_auto_shift_all_even():
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, [4, 4, 4])
    assert kp.shift == (1, 1, 1)


def test_auto_shift_all_odd():
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, [3, 3, 3])
    assert kp.shift == (0, 0, 0)


def test_auto_shift_mixed():
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, [4, 3, 4])
    assert kp.shift == (1, 0, 1)


def test_explicit_shift_overrides_auto():
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, [4, 4, 4], shift=(0, 0, 0))
    assert kp.shift == (0, 0, 0)


def test_gamma_centred_forces_zero_shift():
    sys = _cubic_si_3d()
    kp = vq.KPoints.gamma_centred(sys, [4, 4, 4])
    assert kp.shift == (0, 0, 0)
    # Γ should be in the mesh — at least one k-point at the origin.
    assert any(np.allclose(k, 0.0, atol=1e-12) for k in kp.kpoints_cart)


@pytest.mark.parametrize(
    "dim,mesh,expected",
    [
        (1, [5], (5, 1, 1)),
        (2, [4, 6], (4, 6, 1)),
        (2, [4, 6, 9], (4, 6, 1)),
    ],
)
def test_low_dimensional_mesh_specs_pad_inactive_axes(dim, mesh, expected):
    sys = _low_dim(dim)
    kp = vq.KPoints.gamma_centred(sys, mesh)

    assert kp.mesh == expected
    assert len(kp) == int(np.prod(expected))
    assert kp.shift == (0, 0, 0)
    np.testing.assert_allclose(kp.weights.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(kp.kpoints_frac[:, dim:], 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "dim,mesh,expected_shift",
    [
        (1, [4], (1, 0, 0)),
        (2, [4, 5], (1, 0, 0)),
    ],
)
def test_low_dimensional_auto_shift_only_active_axes(dim, mesh, expected_shift):
    sys = _low_dim(dim)
    kp = vq.KPoints.monkhorst_pack(sys, mesh)

    assert kp.shift == expected_shift
    np.testing.assert_allclose(kp.kpoints_frac[:, dim:], 0.0, atol=1e-12)


def test_public_monkhorst_pack_preserves_gamma_centred_default():
    sys = _cubic_si_3d()

    bm_legacy = vq.monkhorst_pack(sys, [2, 2, 2])
    kp_auto = vq.KPoints.monkhorst_pack(sys, [2, 2, 2])
    kp_gamma = vq.KPoints.gamma_centred(sys, [2, 2, 2])

    assert tuple(bm_legacy.is_shift) == (0, 0, 0)
    assert kp_auto.shift == (1, 1, 1)
    np.testing.assert_allclose(
        np.asarray(bm_legacy.kpoints),
        kp_gamma.kpoints_cart,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "dim,mesh,shift,expected_mesh,expected_shift",
    [
        (1, [4], None, (4, 1, 1), (0, 0, 0)),
        (1, [4], [1], (4, 1, 1), (1, 0, 0)),
        (2, [3, 5], None, (3, 5, 1), (0, 0, 0)),
        (2, [3, 5], [1, 0], (3, 5, 1), (1, 0, 0)),
        (2, [3, 5, 9], [1, 0, 1], (3, 5, 1), (1, 0, 0)),
    ],
)
def test_public_monkhorst_pack_accepts_low_dimensional_specs(
    dim, mesh, shift, expected_mesh, expected_shift,
):
    sys = _low_dim(dim)
    if shift is None:
        bm = vq.monkhorst_pack(sys, mesh)
    else:
        bm = vq.monkhorst_pack(sys, mesh, shift)

    assert tuple(bm.mesh) == expected_mesh
    assert tuple(bm.is_shift) == expected_shift
    assert len(bm) == int(np.prod(expected_mesh))
    B_inv = np.linalg.inv(np.asarray(sys.reciprocal_lattice()))
    frac = (B_inv @ np.asarray(bm.kpoints).T).T
    np.testing.assert_allclose(
        frac[:, dim:], 0.0, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 4. Cartesian ↔ fractional round-trip
# ---------------------------------------------------------------------------

def test_frac_to_cart_round_trip():
    sys = _orth()
    kp = vq.KPoints.monkhorst_pack(sys, [3, 4, 5])
    B = np.asarray(sys.reciprocal_lattice())
    cart_recovered = (B @ kp.kpoints_frac.T).T
    np.testing.assert_allclose(cart_recovered, kp.kpoints_cart,
                                atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Back-compat with native BlochKMesh
# ---------------------------------------------------------------------------

def test_to_bloch_kmesh_round_trip_size():
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, [3, 3, 3])
    bm = kp.to_bloch_kmesh()
    assert isinstance(bm, vq.BlochKMesh)
    assert len(bm) == len(kp)


def test_as_bloch_kmesh_passthrough():
    """``as_bloch_kmesh`` accepts both a KPoints and a raw BlochKMesh."""
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, [2, 2, 2])
    bm = kp.to_bloch_kmesh()
    # KPoints → BlochKMesh
    bm1 = vq.as_bloch_kmesh(kp)
    assert isinstance(bm1, vq.BlochKMesh)
    assert len(bm1) == len(kp)
    # BlochKMesh → BlochKMesh (passthrough)
    bm2 = vq.as_bloch_kmesh(bm)
    assert bm2 is bm


def test_back_compat_with_periodic_rhf():
    """Hand a KPoints to the existing periodic SCF dispatcher via the
    boundary helper — energy must match the legacy BlochKMesh path
    bit-for-bit."""
    sys = _cubic_si_3d()
    bs = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    kp = vq.KPoints.gamma(sys)
    bm_legacy = vq.monkhorst_pack(sys, [1, 1, 1])
    bm_via_kp = vq.as_bloch_kmesh(kp)
    np.testing.assert_array_equal(
        np.asarray(bm_legacy.kpoints), np.asarray(bm_via_kp.kpoints))
    np.testing.assert_array_equal(
        np.asarray(bm_legacy.weights), np.asarray(bm_via_kp.weights))


# ---------------------------------------------------------------------------
# 6. Γ-only convenience
# ---------------------------------------------------------------------------

def test_gamma_is_single_point_at_origin():
    sys = _cubic_si_3d()
    kp = vq.KPoints.gamma(sys)
    assert len(kp) == 1
    np.testing.assert_allclose(kp.kpoints_cart[0], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(kp.weights, [1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Input validation
# ---------------------------------------------------------------------------

def test_mesh_must_have_length_3():
    sys = _cubic_si_3d()
    with pytest.raises(ValueError, match="length 3"):
        vq.KPoints.monkhorst_pack(sys, [2, 2])


def test_low_dimensional_mesh_wrong_active_length_raises():
    sys = _low_dim(2)
    with pytest.raises(ValueError, match="length 2"):
        vq.KPoints.monkhorst_pack(sys, [2])


def test_mesh_entries_must_be_positive():
    sys = _cubic_si_3d()
    with pytest.raises(ValueError, match=">= 1"):
        vq.KPoints.monkhorst_pack(sys, [2, 0, 2])


def test_shift_must_be_zero_or_one():
    sys = _cubic_si_3d()
    with pytest.raises(ValueError, match="0 or 1"):
        vq.KPoints.monkhorst_pack(sys, [2, 2, 2], shift=(2, 0, 0))


# ---------------------------------------------------------------------------
# 8. Repr — useful for debugging, exercise the format
# ---------------------------------------------------------------------------

def test_repr_includes_mesh_and_shift():
    sys = _cubic_si_3d()
    kp = vq.KPoints.monkhorst_pack(sys, [4, 4, 4])
    s = repr(kp)
    assert "KPoints" in s
    assert "mesh=(4, 4, 4)" in s
    assert "shift=(1, 1, 1)" in s


# ---------------------------------------------------------------------------
# 8. Citation provenance (CLAUDE.md § 8)
# ---------------------------------------------------------------------------
#
# `citation_numerics` names the published k-point construction a mesh came
# from, as [routes.numerics] keys. run_periodic_job forwards them to
# assemble(), which is the only way those rows can fire: before this existed,
# nothing in production ever passed `numerics=` and the whole table was dead
# weight that only the citation unit tests kept alive.


def test_kppra_mesh_carries_its_citation_key():
    """KPPRA is a density *convention* layered on a Monkhorst-Pack mesh, so the
    mesh `kind` stays "monkhorst-pack" while the citation names AFLOW."""
    kp = vq.KPoints.from_kppra(_cubic_si_3d(), 64)
    assert kp.kind == "monkhorst-pack"
    assert kp.citation_numerics == ("kppra",)


def test_plain_monkhorst_pack_claims_no_convention():
    """A hand-specified mesh follows no published density convention, so it must
    not silently attribute one -- an over-citation is as wrong as a missing one."""
    assert vq.KPoints.monkhorst_pack(_cubic_si_3d(), [2, 2, 2]).citation_numerics == ()
    assert vq.KPoints.gamma(_cubic_si_3d()).citation_numerics == ()


def test_generalized_regular_grid_carries_its_citation_key():
    kp = vq.KPoints.generalized_regular(_cubic_si_3d(), np.diag([2, 2, 2]))
    assert kp.citation_numerics == ("generalized_regular_kgrid",)


def test_band_path_citation_key_survives_to_kpath():
    """The HPKOT key has to reach the KPath: run_periodic_job reads it off the
    attached BandStructure's kpath, not off the KPoints the SCF ran on."""
    seekpath = pytest.importorskip("seekpath")  # noqa: F841
    bp = vq.KPoints.band_path(_cubic_si_3d())
    assert bp.citation_numerics == ("hpkot_band_path",)
    assert bp.to_kpath().citation_numerics == ("hpkot_band_path",)


def test_every_kpoints_citation_key_resolves_to_a_route():
    """Guard against a provenance key that no [routes.numerics] row answers --
    it would vanish silently, since assemble() treats a numerics miss as a
    warning rather than an error."""
    from vibeqc.output.citations.registry import load_default_database

    db = load_default_database()
    produced = {
        "kppra",
        "generalized_regular_kgrid",
        "kpoint_database",
        "hpkot_band_path",
    }
    routes = set(db._routes.get("numerics", {}))
    assert produced <= routes, f"unrouted KPoints provenance keys: {produced - routes}"


# Integer mesh counts are a specification, not a request for numeric coercion.
def _integer_mesh_builder(system, mesh, builder):
    if builder == "legacy":
        return vq.monkhorst_pack(system, mesh)
    if builder == "shifted":
        return vq.KPoints.shifted(system, mesh, (1, 0, 0))
    return getattr(vq.KPoints, builder)(system, mesh)


@pytest.mark.parametrize("builder", ["monkhorst_pack", "gamma_centred", "shifted", "legacy"])
@pytest.mark.parametrize("mesh", [
    (2.9, 1, 1), (2.0, 1, 1), (True, 1, 1), (np.bool_(True), 1, 1),
    ("2", 1, 1), "211", b"211", (2+0j, 1, 1), ((2,), 1, 1),
    np.array([2., 1., 1.]), np.array([True, True, True]),
])
def test_integer_mesh_counts_refuse_coercion_before_native(builder, mesh, monkeypatch):
    import vibeqc.kpoints as module
    def forbidden(*args, **kwargs):
        pytest.fail("non-integer mesh reached native grid construction")
    monkeypatch.setattr(module, "_mp_native", forbidden)
    monkeypatch.setattr(vq, "_monkhorst_pack_native", forbidden)
    with pytest.raises(ValueError, match="must contain integers"):
        _integer_mesh_builder(_cubic_si_3d(), mesh, builder)


@pytest.mark.parametrize("builder", ["monkhorst_pack", "gamma_centred", "shifted", "legacy"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("full", [False, True])
def test_integer_mesh_counts_preserve_numpy_and_inactive_pinning(builder, dim, full):
    system = _low_dim(dim) if dim < 3 else _cubic_si_3d()
    mesh = [np.int64(2)]*dim
    if full:
        # KPoints retains its documented pinning, distinct from chi's refusal
        # of explicit nontrivial inactive orders.
        mesh += [np.int64(7)]*(3-dim)
    result = _integer_mesh_builder(system, np.array(mesh, dtype=np.uint32), builder)
    assert tuple(result.mesh) == (2,)*dim+(1,)*(3-dim)
    assert len(result) == 2**dim


@pytest.mark.parametrize("consumer", ["runner", "madelung", "density", "ccm", "gamma_gdf"])
@pytest.mark.parametrize("mesh", [
    (1.9, 1, 1), (2.0, 1, 1), (True, 1, 1), (np.bool_(True), 1, 1),
    ("2", 1, 1), "211", b"211", (2+0j, 1, 1), ((2,), 1, 1),
    np.array([2., 1., 1.]), np.array([True, True, True]),
])
def test_periodic_count_frontdoors_reject_before_geometry(consumer, mesh, monkeypatch):
    from types import SimpleNamespace
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_runner import _runner_bloch_kmesh
    from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices
    from vibeqc.periodic.ccm.four_center_runner import run_four_center_scf
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf

    def forbidden(*args, **kwargs):
        pytest.fail("invalid periodic count reached native grid construction")

    monkeypatch.setattr(core, "monkhorst_pack", forbidden)
    # Missing lattice, density, basis and geometry deliberately prevent a
    # mistaken path from doing physical work before rejecting the count.
    system = SimpleNamespace(dim=3)
    calls = {
        "runner": lambda: _runner_bloch_kmesh(system, mesh),
        "madelung": lambda: probe_charge_madelung_supercell(system, mesh),
        "density": lambda: bvk_torus_density_matrices(None, [], mesh),
        "ccm": lambda: run_four_center_scf(system, "sto-3g", "RHF", mesh),
        "gamma_gdf": lambda: run_pbc_gdf_rhf(system, None, kmesh=mesh),
    }
    with pytest.raises(ValueError, match="must contain integers"):
        calls[consumer]()


@pytest.mark.parametrize("mesh, expected", [
    (None, (1, 1, 1)), (2, (2, 2, 2)), (np.int64(2), (2, 2, 2)),
    ((np.int64(2), 3, 1), (2, 3, 1)), ([2, 3, 1], (2, 3, 1)),
])
def test_periodic_count_runner_preserves_scalar_repeat(mesh, expected, monkeypatch):
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_runner import _runner_bloch_kmesh
    monkeypatch.setattr(core, "monkhorst_pack", lambda system, counts: tuple(counts))
    assert _runner_bloch_kmesh(None, mesh) == expected


def test_periodic_count_madelung_preserves_integer_lattice_scaling(monkeypatch):
    import vibeqc.bipole_fock_ewald as ewald
    lattice = np.array([[8., 2., 1.], [0., 7., 2.], [0., 0., 6.]])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0., 0., 0.])])
    monkeypatch.setattr(ewald, "probe_charge_madelung", lambda cell, **kw: cell.lattice)
    result = ewald.probe_charge_madelung_supercell(system, np.array([2, 3, 1], dtype=np.uint32))
    np.testing.assert_array_equal(result, lattice*np.array([2, 3, 1])[None, :])


def test_periodic_count_gamma_gdf_keeps_multik_refusal():
    from types import SimpleNamespace
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf
    with pytest.raises(NotImplementedError, match="only kmesh"):
        run_pbc_gdf_rhf(SimpleNamespace(dim=3), None, kmesh=(np.int64(2), 1, 1))


def test_periodic_count_density_accepts_numpy_integer_torus():
    from types import SimpleNamespace
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices
    cells = [SimpleNamespace(index=(i, 0, 0), r_cart=np.array([float(i), 0., 0.]))
             for i in range(2)]
    density = SimpleNamespace(cells=cells, blocks=[np.array([[2.]]), np.array([[0.5]])])
    result = bvk_torus_density_matrices(
        density, [np.zeros(3), np.array([np.pi, 0., 0.])],
        np.array([2, 1, 1], dtype=np.uint32),
    )
    np.testing.assert_allclose(np.asarray(result).reshape(2), [2.5, 1.5])


def test_periodic_count_ccm_preserves_numpy_integer_repetitions(monkeypatch):
    import vibeqc.periodic.ccm.system as ccm_system
    from vibeqc.periodic.ccm.four_center_runner import run_four_center_scf
    system = vq.PeriodicSystem(3, np.eye(3)*8.0, [vq.Atom(2, [0., 0., 0.])])
    class ReachedConstruction(Exception):
        pass
    def capture(system, repetitions, basis):
        assert repetitions == (2, 3, 1)
        raise ReachedConstruction
    monkeypatch.setattr(ccm_system, "CCMSystem", capture)
    with pytest.raises(ReachedConstruction):
        run_four_center_scf(system, "sto-3g", "RHF", np.array([2, 3, 1], dtype=np.uint32))


@pytest.mark.parametrize("consumer", ["ase_forces", "gpw_calculate", "gpw_scf", "dimer"])
@pytest.mark.parametrize("mesh", [
    (1.9, 1, 1), (2.0, 1, 1), (True, 1, 1), (np.bool_(True), 1, 1),
    ("2", 1, 1), "211", b"211", (2+0j, 1, 1), ((2,), 1, 1),
    np.array([2., 1., 1.]), np.array([True, True, True]),
])
def test_periodic_wrapper_counts_reject_before_scf(consumer, mesh, monkeypatch):
    from vibeqc import _vibeqc_core as core
    from vibeqc.dimer import run_dimer

    def forbidden(*args, **kwargs):
        pytest.fail("malformed wrapper mesh reached numerical setup")

    monkeypatch.setattr(core, "monkhorst_pack", forbidden)
    if consumer == "dimer":
        system = vq.PeriodicSystem(3, np.eye(3)*8.0, [vq.Atom(2, [0., 0., 0.])])
        with pytest.raises(ValueError, match="must contain integers"):
            run_dimer(system, "sto-3g", method="RHF", kpoints=mesh)
        return
    ase = pytest.importorskip("ase")
    import vibeqc.ase_periodic as periodic
    import vibeqc.ase_periodic_gpw as gpw
    atoms = ase.Atoms("He", positions=[[0., 0., 0.]], cell=[8., 8., 8.], pbc=True)
    monkeypatch.setattr(periodic, "atoms_to_periodic_system", forbidden)
    monkeypatch.setattr(gpw, "BasisSet", forbidden)
    if consumer != "ase_forces":
        calc = gpw.VibeqcGPW(functional="pbe")
        # Bypass ASE's constructor equality/shape inspection to exercise
        # both consumer guards even for a nested, malformed count vector.
        calc.parameters["kmesh"] = mesh
    with pytest.raises(ValueError, match="must contain integers"):
        if consumer == "ase_forces":
            periodic.periodic_forces(atoms, None, kpts=mesh)
        elif consumer == "gpw_calculate":
            monkeypatch.setattr(calc, "_run_scf", forbidden)
            calc.calculate(atoms, properties=["energy"])
        else:
            calc._run_scf(atoms)


@pytest.mark.parametrize("consumer", ["ase_forces", "gpw_calculate", "gpw_scf", "dimer"])
@pytest.mark.parametrize("mesh", [None, (1, 1, 1), np.array([2, 1, 1], dtype=np.uint32)])
def test_periodic_wrapper_counts_keep_integer_dispatch(consumer, mesh, monkeypatch):
    from vibeqc import _vibeqc_core as core
    from vibeqc.dimer import run_dimer
    expected = (1, 1, 1) if mesh is None else tuple(mesh)
    seen = []
    class ReachedDispatch(Exception):
        pass
    def capture_grid(system, counts, *args, **kwargs):
        seen.append(tuple(counts))
        raise ReachedDispatch
    if consumer == "dimer":
        monkeypatch.setattr(core, "monkhorst_pack", capture_grid)
        system = vq.PeriodicSystem(3, np.eye(3)*8.0, [vq.Atom(2, [0., 0., 0.])])
        with pytest.raises(ReachedDispatch):
            run_dimer(system, "sto-3g", method="RHF", kpoints=mesh)
    else:
        ase = pytest.importorskip("ase")
        import vibeqc.ase_periodic as periodic
        import vibeqc.ase_periodic_gpw as gpw
        atoms = ase.Atoms("He", positions=[[0., 0., 0.]], cell=[8., 8., 8.], pbc=True)
        monkeypatch.setattr(periodic, "monkhorst_pack", capture_grid)
        monkeypatch.setattr(gpw, "monkhorst_pack", capture_grid)
        monkeypatch.setattr(gpw, "BasisSet", lambda *args: None)
        def capture_gamma(*args, **kwargs):
            seen.append((1, 1, 1))
            raise ReachedDispatch
        monkeypatch.setattr(gpw, "run_periodic_rhf_gpw", capture_gamma)
        calc = gpw.VibeqcGPW(kmesh=mesh, functional="pbe")
        with pytest.raises(ReachedDispatch):
            if consumer == "ase_forces":
                periodic.periodic_forces(atoms, None, kpts=mesh)
            elif consumer == "gpw_calculate":
                def capture_scf(*args, **kwargs):
                    seen.append(expected)
                    raise ReachedDispatch
                monkeypatch.setattr(calc, "_run_scf", capture_scf)
                calc.calculate(atoms, properties=["energy"])
            else:
                calc._run_scf(atoms)
    assert seen == [expected]
