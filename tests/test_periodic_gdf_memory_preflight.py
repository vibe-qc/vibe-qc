"""Fail-early memory preflight for the dense multi-k periodic GDF Lpq cache.

Regression for prompt 75 (NiO multi-k KUKS/GDF killed before SCF artifacts):
the per-pair ``Lpq`` density-fitting cache is held dense in RAM (streaming is
future work), so a paper-grade cell (NiO/def2-SVP KUKS at a ``(4,4,4)`` mesh,
180 AOs, ~1500 aux) allocates tens of GB and gets OOM-killed (exit 137) before
SCF iter 1, leaving only a header. The driver now estimates the cache peak and
raises :class:`~vibeqc.memory.InsufficientMemoryError` early with route-specific
remedies instead.

These tests exercise the estimator + gate directly with injected ``available``
so they are deterministic and need no built core / no SCF run (the real NiO
route is un-runnable by construction -- it OOMs).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.memory import (
    InsufficientMemoryError,
    check_periodic_gdf_memory,
    estimate_periodic_multik_gdf,
)

_GB = 1024**3


def _corundum_al2o3_system():
    """Corundum Al2O3 rhombohedral primitive cell (10 atoms)."""
    a, c = 4.759, 12.991  # angstrom
    ang = 1.0 / 0.529177210903
    a_b, c_b = a * ang, c * ang
    ax = np.array([a_b / 2, -a_b * np.sqrt(3) / 6, c_b / 3])
    ay = np.array([0, a_b * np.sqrt(3) / 3, c_b / 3])
    az = np.array([-a_b / 2, -a_b * np.sqrt(3) / 6, c_b / 3])
    lat = np.column_stack([ax, ay, az])
    al = [(0, 0, 0.35216), (0, 0, -0.35216), (0, 0, 0.85216), (0, 0, 0.14784)]
    o = [
        (0.3063, 0, 0.25),
        (0, 0.3063, 0.25),
        (-0.3063, -0.3063, 0.25),
        (-0.3063, 0, -0.25),
        (0, -0.3063, -0.25),
        (0.3063, 0.3063, -0.25),
    ]
    atoms = []
    for f in al:
        atoms.append(vq.Atom(13, list(lat @ np.array(f))))
    for f in o:
        atoms.append(vq.Atom(8, list(lat @ np.array(f))))
    return vq.PeriodicSystem(3, lat, atoms)


class TestEstimate:
    def test_source_admission_reserves_retained_oneel_and_xc(self, monkeypatch):
        from types import SimpleNamespace
        import vibeqc.memory as memory
        import vibeqc.periodic_k_gdf as driver

        monkeypatch.setattr(vq._vibeqc_core, "periodic_xc_domain_counts",
                            lambda *args: (20, 10))
        monkeypatch.setattr(vq._vibeqc_core, "gdf_short_range_workspace_bytes",
                            lambda *args: 128 * 1024**2)
        captured = []
        monkeypatch.setattr(memory, "available_memory_bytes", lambda: 2 * _GB)
        monkeypatch.setattr(driver, "_build_range_separated_lpq_cache",
                            lambda *a, **kw: captured.append(kw))
        options = SimpleNamespace(use_diis=True, diis_subspace_size=8)
        system = SimpleNamespace(unit_cell=[None, None])
        basis, auxiliary = SimpleNamespace(nbasis=24), SimpleNamespace(nbasis=80)
        args = dict(
            omega=.6, raw_integral_error=1e-10, ke_cutoff=20.,
            linear_dep_thr=1e-10, lat_opts=SimpleNamespace(cutoff_bohr=10.),
            fit_screen_threshold=1e-12, options=options, open_shell=True,
            progress=None,
        )
        driver._build_scf_range_separated_lpq_cache(
            system, basis, auxiliary, np.zeros((2, 3)), True, **args,
        )
        # A .blocks access is deliberately unavailable: estimating native
        # lattice storage must not materialize every AO matrix in Python.
        lattice = SimpleNamespace(nbf=24, cells=[None] * 19)
        grid = SimpleNamespace(n_points=5000)
        driver._build_scf_range_separated_lpq_cache(
            system, basis, auxiliary, np.zeros((2, 3)), True, **args,
            oneel_lattices=(lattice, None), xc_grid=grid,
            xc_cells=[None] * 27, functional=SimpleNamespace(kind='GGA'),
        )
        assert captured[0]['native_workspace_byte_cap'] == 128 * 1024**2
        xc = memory.estimate_periodic_xc_value(
            n_basis=24, n_atoms=2, n_grid_points=5000, n_cells=27,
            functional_kind='GGA', open_shell=True,
            n_active_cells=20, n_bra_cells=10,
        )
        xc_bytes = sum(v for k, v in xc.by_category.items()
                       if k != 'Python runtime + NumPy overhead')
        assert (captured[0]['memory_byte_cap'] - captured[1]['memory_byte_cap']
                == 19 * (8 * 24**2 + 256) + xc_bytes)

    def test_hybrid_histories_are_full_mesh_spin_resolved_and_depth_dependent(self):
        args = dict(n_basis=30, n_aux=90, n_kpoints=8, need_k_pairs=True)
        shallow = estimate_periodic_multik_gdf(**args, diis_subspace_size=4)
        deep = estimate_periodic_multik_gdf(**args, diis_subspace_size=12)
        wedge = estimate_periodic_multik_gdf(**args, n_ibz_kpoints=2, diis_subspace_size=12)
        spin = estimate_periodic_multik_gdf(**args, open_shell=True, diis_subspace_size=12)
        disabled = estimate_periodic_multik_gdf(**args, diis_subspace_size=0)
        key = "SCF accelerator history peak"
        block_bytes = np.empty((8, 30, 30), dtype=np.complex128).nbytes
        # Increasing history by eight retains eight additional F/error
        # pairs and eight additional F/density pairs on the default hybrid.
        assert deep.by_category[key] - shallow.by_category[key] == 8 * 4 * block_bytes
        assert wedge.by_category[key] == deep.by_category[key]
        assert spin.by_category[key] == 2 * deep.by_category[key]
        assert disabled.by_category[key] == 0
        assert disabled.by_category["GDF Lpq factor cache"] == deep.by_category["GDF Lpq factor cache"]

    @pytest.mark.parametrize("bad", [-1, .5])
    def test_invalid_accelerator_depth_is_rejected(self, bad):
        with pytest.raises(ValueError, match="diis_subspace_size"):
            estimate_periodic_multik_gdf(
                n_basis=2, n_aux=6, n_kpoints=2, need_k_pairs=True,
                diis_subspace_size=bad,
            )

    @pytest.mark.parametrize("enabled,depth", [(True, 4), (True, 12), (False, 12)])
    def test_driver_preflight_consumes_resolved_accelerator_options(self, monkeypatch, enabled, depth):
        from types import SimpleNamespace
        import vibeqc.memory as memory
        from vibeqc.periodic_k_gdf import _preflight_gdf_lpq_memory

        captured = []
        monkeypatch.setattr(memory, "check_periodic_gdf_memory", lambda est, **kw: captured.append(est))
        args = dict(n_basis=2, n_aux=6, n_kpoints=2, need_k_pairs=True, open_shell=False)
        _preflight_gdf_lpq_memory(
            SimpleNamespace(info=lambda message: None), **args,
            route_label="SR/LR memory regression",
            options=SimpleNamespace(use_diis=enabled, diis_subspace_size=depth),
        )
        assert captured[0].by_category == estimate_periodic_multik_gdf(
            **args, diis_subspace_size=depth if enabled else 0,
        ).by_category

    def test_exact_bytes_small_case_diagonal(self):
        """Diagonal (pure-DFT) cache: n_k complex128 (n_aux, nao, nao) blocks."""
        est = estimate_periodic_multik_gdf(
            n_basis=10, n_aux=30, n_kpoints=4, need_k_pairs=False, open_shell=False
        )
        # Lpq dominant term: 4 pairs * 30 * 10 * 10 * 16 bytes.
        assert est.by_category["GDF Lpq factor cache"] == 4 * 30 * 10 * 10 * 16

    def test_hybrid_scales_as_k_squared(self):
        """HF/hybrid needs every (k_i, k_j) pair -> n_k**2, not n_k."""
        diag = estimate_periodic_multik_gdf(
            n_basis=20, n_aux=60, n_kpoints=8, need_k_pairs=False, open_shell=False
        )
        pairs = estimate_periodic_multik_gdf(
            n_basis=20, n_aux=60, n_kpoints=8, need_k_pairs=True, open_shell=False
        )
        ratio = (
            pairs.by_category["GDF Lpq factor cache"]
            / diag.by_category["GDF Lpq factor cache"]
        )
        assert ratio == pytest.approx(8.0)  # n_k**2 / n_k = n_k

    def test_open_shell_adds_only_subdominant_buffers(self):
        """Open-shell doubles the per-k Fock/density buffers but the Lpq cache
        (spin-independent) is unchanged -- it stays the dominant term."""
        closed = estimate_periodic_multik_gdf(
            n_basis=50, n_aux=150, n_kpoints=8, need_k_pairs=False, open_shell=False
        )
        opened = estimate_periodic_multik_gdf(
            n_basis=50, n_aux=150, n_kpoints=8, need_k_pairs=False, open_shell=True
        )
        assert (
            opened.by_category["GDF Lpq factor cache"]
            == closed.by_category["GDF Lpq factor cache"]
        )
        assert (
            opened.by_category["per-k complex buffers"]
            > closed.by_category["per-k complex buffers"]
        )

    def test_nio_scale_is_tens_of_gb(self):
        """The reported NiO/def2-SVP KUKS (4,4,4) config: 180 AOs, ~1500 aux,
        64 k-points, pure PBE (diagonal pairs) -> clearly beyond a laptop."""
        est = estimate_periodic_multik_gdf(
            n_basis=180, n_aux=1500, n_kpoints=64, need_k_pairs=False, open_shell=True
        )
        assert est.total_gb > 40.0  # dozens of GB, not runnable on localhost


class TestIBZReduction:
    """The space-group-reduced exchange build stores n_IBZ x n_k Lpq blocks.

    Charging a reduced run at n_k^2 aborts calculations that fit and makes
    the printed estimate contradict the cache actually allocated -- the
    reason the demonstration could not be driven from run_periodic_job.
    """

    @pytest.mark.parametrize(
        "n_k, n_ibz", [(8, 3), (64, 8), (64, 10), (27, 4)]
    )
    def test_lpq_cache_counts_exchange_wedge_and_remaining_hartree_diagonals(self, n_k, n_ibz):
        """Keep full exchange ket sums and every full-mesh Hartree diagonal."""
        full = estimate_periodic_multik_gdf(
            n_basis=40, n_aux=120, n_kpoints=n_k, need_k_pairs=True
        )
        reduced = estimate_periodic_multik_gdf(
            n_basis=40,
            n_aux=120,
            n_kpoints=n_k,
            need_k_pairs=True,
            n_ibz_kpoints=n_ibz,
        )
        key = "GDF Lpq factor cache"
        assert reduced.by_category[key] == (n_ibz * n_k + n_k - n_ibz) * 120 * 40 * 40 * 16
        assert full.by_category[key] / reduced.by_category[key] == pytest.approx(
            n_k**2 / (n_ibz * n_k + n_k - n_ibz)
        )
        # Only the Lpq cache reduces; per-k buffers are still built at every k.
        assert (
            reduced.by_category["per-k complex buffers"]
            == full.by_category["per-k complex buffers"]
        )

    def test_no_wedge_argument_is_the_full_mesh(self):
        """Default None must be byte-identical to the pre-existing n_k^2."""
        a = estimate_periodic_multik_gdf(
            n_basis=30, n_aux=90, n_kpoints=16, need_k_pairs=True
        )
        b = estimate_periodic_multik_gdf(
            n_basis=30,
            n_aux=90,
            n_kpoints=16,
            need_k_pairs=True,
            n_ibz_kpoints=16,
        )
        assert a.by_category == b.by_category

    def test_diagonal_cache_ignores_the_wedge(self):
        """Pure DFT / J-only keeps only the n_k diagonal pairs; there is no
        bra/ket product to reduce, and the driver passes no bra_rows there."""
        diag = estimate_periodic_multik_gdf(
            n_basis=30, n_aux=90, n_kpoints=16, need_k_pairs=False
        )
        diag_wedge = estimate_periodic_multik_gdf(
            n_basis=30,
            n_aux=90,
            n_kpoints=16,
            need_k_pairs=False,
            n_ibz_kpoints=2,
        )
        assert diag.by_category == diag_wedge.by_category

    @pytest.mark.parametrize("bad", [0, -1, 65])
    def test_out_of_range_wedge_is_refused(self, bad):
        """A wedge larger than the mesh (or empty) is a caller bug; report
        it rather than silently under- or over-charging the preflight."""
        with pytest.raises(ValueError, match="n_ibz_kpoints"):
            estimate_periodic_multik_gdf(
                n_basis=10,
                n_aux=30,
                n_kpoints=64,
                need_k_pairs=True,
                n_ibz_kpoints=bad,
            )


class TestGate:
    def _nio_estimate(self):
        return estimate_periodic_multik_gdf(
            n_basis=180, n_aux=1500, n_kpoints=64, need_k_pairs=False, open_shell=True
        )

    def test_raises_when_over_budget(self):
        est = self._nio_estimate()
        with pytest.raises(InsufficientMemoryError) as exc:
            check_periodic_gdf_memory(
                est, n_kpoints=64, route_label="KUKS pbe", available=16 * _GB
            )
        msg = str(exc.value)
        # Route-specific remedies, not the molecular check_memory text.
        assert "GPW" in msg and "Monkhorst-Pack" in msg
        assert "VIBEQC_GDF_MEMORY_OVERRIDE" in msg
        assert "KUKS pbe" in msg

    def test_passes_when_fits(self):
        est = self._nio_estimate()
        # Huge machine: no raise.
        check_periodic_gdf_memory(
            est, n_kpoints=64, route_label="KUKS pbe", available=4096 * _GB
        )

    def test_override_allows_exceed(self):
        est = self._nio_estimate()
        check_periodic_gdf_memory(
            est,
            n_kpoints=64,
            route_label="KUKS pbe",
            available=16 * _GB,
            allow_exceed=True,
        )

    def test_unknown_ram_is_fail_open(self):
        """available=0 means the probe failed; the gate must not block."""
        est = self._nio_estimate()
        check_periodic_gdf_memory(
            est, n_kpoints=64, route_label="KUKS pbe", available=0
        )


class TestRangeSeparatedSourcePreflight:
    """Model the streamed source without constructing the old dense mesh."""

    def _corundum_estimate(self, monkeypatch, cutoff=200.0):
        import vibeqc.aux_basis as auxiliary_module
        from vibeqc.periodic_jk_method import PeriodicJKMethod
        from vibeqc.periodic_runner import _periodic_gdf_estimate

        def forbidden(*args, **kwargs):
            pytest.fail("memory preflight tried to allocate a reciprocal mesh")
        monkeypatch.setattr(auxiliary_module, 'rsgdf_dense_g_mesh', forbidden)
        monkeypatch.setattr(auxiliary_module, '_rsgdf_bounded_reciprocal_sphere', forbidden)
        system = _corundum_al2o3_system()
        basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
        return _periodic_gdf_estimate(
            system, basis, resolved_jk=PeriodicJKMethod.GDF,
            method_upper="RKS", functional="pbe", kpoints=None,
            aux_basis=None, rsgdf_ke_cutoff=cutoff,
        )

    def test_corundum_has_bounded_source_instead_of_ao_pair_bundle(self, monkeypatch):
        estimate = self._corundum_estimate(monkeypatch).estimate
        assert 'GDF dense AO-pair FT bundle' not in estimate.by_category
        assert estimate.by_category['GDF SR/LR source and whitening workspace'] >= 64*1024**2
        assert estimate.by_category['Periodic-XC retained data and phase workspace'] > 0
        assert estimate.total_bytes < 16*_GB

    def test_corundum_fit_preflight_allows_16gb(self, monkeypatch):
        state = self._corundum_estimate(monkeypatch)
        check_periodic_gdf_memory(
            state.estimate, n_kpoints=state.n_kpoints,
            route_label=state.route_label, available=16*_GB,
        )

    @pytest.mark.parametrize("lattice", [
        np.eye(3) * 2.7,
        np.array([[2.7, 1.8, -.6], [0., 1.1, .8], [0., 0., 2.3]]),
        np.diag([.9, 3.1, 8.0]),
    ])
    @pytest.mark.parametrize("pair_complete", [False, True])
    def test_oneel_sphere_bound_covers_allocated_matrices_and_box_capacity(
        self, lattice, pair_complete,
    ):
        import math
        from vibeqc.periodic_k_gdf import _gdf_oneel_memory_estimate

        system = vq.PeriodicSystem(3, lattice, [
            vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [1.2, .7, -.3]),
        ])
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        options = vq.LatticeSumOptions()
        options.cutoff_bohr = 6.0
        options.pair_complete_1e = pair_complete
        radius = options.cutoff_bohr + (np.linalg.norm([1.2, .7, -.3]) if pair_complete else 0.)
        actual_cells = len(vq._vibeqc_core.direct_lattice_cells(system, radius))
        capacity = math.prod(2*math.ceil(np.linalg.norm(row)*radius)+3
                             for row in np.linalg.inv(lattice))
        estimate = _gdf_oneel_memory_estimate(system, basis, options)
        matrix_and_metadata = estimate.by_category['GDF one-electron lattice matrices and metadata']
        assert matrix_and_metadata >= 8*(actual_cells*8*basis.nbasis**2 + capacity*256)

    def test_unaffordable_reciprocal_cutoff_is_counted_without_allocation(self, monkeypatch):
        state = self._corundum_estimate(monkeypatch, cutoff=1.7e6)
        assert state.estimate.by_category['GDF reciprocal vectors and binding copy'] > 100*_GB
        with pytest.raises(InsufficientMemoryError):
            check_periodic_gdf_memory(
                state.estimate, n_kpoints=state.n_kpoints,
                route_label=state.route_label, available=16*_GB,
            )


class TestAuxDimensionResolution:
    """The preflight sizes the aux set the driver will allocate (issue #92).

    ``aux_basis=None`` is the runner's ``<auto>``; every GDF driver resolves
    it as ``default_aux_for(basis.name)`` before building the cderi. The
    preflight used to fall back to ``3 x n_ao`` for ``None`` -- 396 against
    the 960 def2-svp-jkfit functions of the P05 NaCl conventional cell -- and
    printed 452 GB for a 1096 GB dense Lpq cache: a 500 GB node would have
    accepted a 1.1 TB build.
    """

    def _nacl_conventional(self):
        a = 5.640 / 0.529177210903
        lat = a * np.eye(3)
        frac = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
        atoms = [vq.Atom(11, list(np.array(f) * a)) for f in frac]
        atoms += [vq.Atom(17, list(((np.array(f) + 0.5) % 1.0) * a)) for f in frac]
        return vq.PeriodicSystem(3, lat, atoms)

    def test_auto_aux_is_the_drivers_default_aux(self):
        from vibeqc.aux_basis import default_aux_for, make_aux_basis_set
        from vibeqc.periodic_runner import _periodic_gdf_aux_basis_size

        system = self._nacl_conventional()
        basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
        n_aux = _periodic_gdf_aux_basis_size(system, basis, None)
        real = make_aux_basis_set(
            system.unit_cell_molecule(), aux_name=default_aux_for("def2-svp")
        ).nbasis
        assert n_aux == real
        assert n_aux > 3 * basis.nbasis  # the old floor under-counted 2.4x

    def test_p05_nacl_conventional_krhf_444_is_terabyte_scale_unreduced(self):
        """The filed deck (KRHF, (4,4,4), no symmetry kwargs) is a 4096-pair
        dense cache above any fleet node; the space-group wedge (n_IBZ = 10
        for Fm-3m at (4,4,4)) brings it to a compute-managed / compute-large job."""
        from vibeqc.periodic_jk_method import PeriodicJKMethod
        from vibeqc.periodic_runner import _periodic_gdf_estimate

        system = self._nacl_conventional()
        basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
        full = _periodic_gdf_estimate(
            system,
            basis,
            resolved_jk=PeriodicJKMethod.GDF,
            method_upper="RHF",
            functional=None,
            kpoints=(4, 4, 4),
            aux_basis=None,
            rsgdf_ke_cutoff=200.0,
        )
        reduced = _periodic_gdf_estimate(
            system,
            basis,
            resolved_jk=PeriodicJKMethod.GDF,
            method_upper="RHF",
            functional=None,
            kpoints=(4, 4, 4),
            aux_basis=None,
            rsgdf_ke_cutoff=200.0,
            n_ibz_kpoints=10,
        )
        key = "GDF Lpq factor cache"
        assert full.estimate.by_category[key] > 1000 * 1e9
        assert full.estimate.by_category[key] / reduced.estimate.by_category[key] == pytest.approx(64**2 / (10 * 64 + 54))
        # Full retained Lpq plus the bounded SR/LR source, x1.5 headroom: a compute-large
        # (504 GB) or compute-managed (1 TB) job, not a laptop one and not a refusal.
        assert 250 * 1e9 < reduced.estimate.total_bytes < 450 * 1e9
