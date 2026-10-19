"""Fast, actionable errors on periodic routes that have no 1-D/2-D Coulomb model.

A stream-A benchmark blocker: ``jk_method="bipole"`` on a vacuum-padded 1-D/2-D
cell. The bipolar-expansion Fock build uses a 3-D Ewald/Madelung lattice sum
undefined for a low-dimensional cell; previously this surfaced only deep in the
SCF setup as a terse ``ValueError`` (``probe_charge_madelung_supercell requires
dim=3``). ``run_periodic_job`` now fails fast at the API level with an actionable
``NotImplementedError`` naming the low-D alternatives, before any setup.

(The companion GDF rsgdf low-D mesh blowup is handled separately by the
dimension-aware ``rsgdf_dense_g_mesh``, which meshes only the periodic
directions — covered by the GDF test suite, not here.)

``jk_method="auto"`` on a 1-D wire is refused the same way: no public J/K
builder has a consistent 1-D Coulomb gauge, so AUTO no longer selects GDF and
leaves the refusal to the one-electron setup. Both refusals name the ``dim=3``
vacuum-box model (https://github.com/vibe-qc/vibe-qc/issues/1).

The Γ-only closed-shell GDF driver refuses a wire too. It paired the bare 1-D
``V_ne``/``E_nn`` lattice sums with a J/K that does not cancel them, so its
total fell by about ln 2 Ha per doubling of the lattice-sum cutoff (H2 /
sto-3g, a = 4 bohr: E_nuclear = 2.29 -> 5.08 Ha as the cutoff went
10 -> 160 bohr).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, run_periodic_job

from tests.refusal_receipt import assert_preflight_refusal


def _h_chain_1d(a=7.9376582, vac=40.0):
    """A 1-D H₂ chain in a vacuum box (dim=1) — the stream-A h-chain shape."""
    return PeriodicSystem(
        1, np.diag([a, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [0.0933333 * a, 0, 0])],
        charge=0, multiplicity=1)


def test_bipole_rejects_low_dim_at_api(tmp_path):
    """run_periodic_job(jk_method='bipole') on a 1-D cell fails fast and clearly."""
    sysp = _h_chain_1d()
    b = BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    t0 = time.time()
    with pytest.raises(NotImplementedError) as exc:
        run_periodic_job(
            sysp, b, method="RHF", jk_method="bipole", kpoints=(2, 1, 1),
            max_iter=5, write_molden_file=False, write_density=False,
            write_cif_file=False, write_xsf_structure_file=False,
            write_poscar_file=False, write_xyz_file=False,
            write_population_file=False, output=tmp_path / "refused")
    assert time.time() - t0 < 5.0                      # before any SCF setup
    msg = str(exc.value)
    assert "dim=1" in msg and "bipole" in msg
    # A wire has no public route, so the remedy is the dim=3 vacuum box.
    assert "A 1-D wire has no public route" in msg
    assert "PeriodicSystem(dim=3" in msg


_QUIET = dict(
    max_iter=5, write_molden_file=False, write_density=False,
    write_cif_file=False, write_xsf_structure_file=False,
    write_poscar_file=False, write_xyz_file=False,
    write_population_file=False,
)


@pytest.mark.parametrize("kpoints", [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize(
    "method, functional",
    [("RHF", None), ("RKS", "pbe"), ("UHF", None), ("UKS", "pbe")],
)
def test_auto_rejects_dim1_at_api(tmp_path, method, functional, kpoints):
    """AUTO refuses a wire before any setup and names the vacuum-box model."""
    sysp = _h_chain_1d()
    b = BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kw = {} if functional is None else {"functional": functional}
    stem = tmp_path / "refused"
    t0 = time.time()
    with pytest.raises(NotImplementedError) as exc:
        run_periodic_job(
            sysp, b, method=method, jk_method="auto", kpoints=kpoints,
            output=stem, **kw, **_QUIET)
    assert time.time() - t0 < 5.0                      # before any SCF setup
    msg = str(exc.value)
    assert "jk_method='auto' has no route for a dim=1 (wire) system" in msg
    assert "PeriodicSystem(dim=3" in msg
    # The deep one-electron guard never ran: the refusal is AUTO's own.
    assert "_gauge_lat_opts_for_v_ne_and_e_nuc" not in msg
    assert_preflight_refusal(stem, exc.value)


_GAMMA_WIRE_REFUSAL = "no consistent Coulomb gauge for a dim = 1 cell"


@pytest.mark.parametrize("functional", [None, "pbe", "b3lyp"])
def test_gamma_gdf_driver_refuses_a_wire(functional):
    """The Γ driver refuses dim=1 before any integral work, naming the remedy."""
    from vibeqc import run_rhf_periodic_gamma_gdf

    sysp = _h_chain_1d()
    b = BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    t0 = time.time()
    with pytest.raises(NotImplementedError, match=_GAMMA_WIRE_REFUSAL) as exc:
        run_rhf_periodic_gamma_gdf(sysp, b, functional=functional,
                                   progress=False)
    assert time.time() - t0 < 5.0
    assert "PeriodicSystem(dim=3" in str(exc.value)


@pytest.mark.parametrize("functional", [None, "pbe"])
def test_k_gdf_gamma_mesh_refuses_a_wire(functional):
    """A Γ mesh on a chain delegates to the Γ driver, which refuses it."""
    from vibeqc import run_krhf_periodic_gdf, run_krks_periodic_gdf

    sysp = _h_chain_1d()
    b = BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match=_GAMMA_WIRE_REFUSAL):
        if functional is None:
            run_krhf_periodic_gdf(sysp, b, (1, 1, 1), progress=False)
        else:
            run_krks_periodic_gdf(sysp, b, (1, 1, 1), functional=functional,
                                  progress=False)


@pytest.mark.parametrize("kpoints", [None, (1, 1, 1)])
@pytest.mark.parametrize("method, functional", [("RHF", None), ("RKS", "pbe")])
def test_explicit_gdf_gamma_refuses_a_wire_at_api(
    tmp_path, method, functional, kpoints
):
    """An explicit jk_method='gdf' at Γ no longer returns a wire energy."""
    sysp = _h_chain_1d()
    b = BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kw = {} if functional is None else {"functional": functional}
    if kpoints is not None:
        kw["kpoints"] = kpoints
    with pytest.raises(NotImplementedError, match=_GAMMA_WIRE_REFUSAL):
        run_periodic_job(
            sysp, b, method=method, jk_method="gdf",
            output=tmp_path / "refused", **kw, **_QUIET)
