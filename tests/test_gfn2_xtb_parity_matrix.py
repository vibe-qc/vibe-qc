"""GFN2-xTB molecular parity matrix against xtb 6.7.0, channel by channel.

``tests/data/gfn2_xtb_parity_xtb670.json`` carries 49 molecules, ions and
radicals (period 2 to 5 main group, Zn/Cd/Hg, Cu2, TiCl4) with the energy
channels a live xtb 6.7.0 printed for the same geometries (``--gfn 2 --acc
0.0001``). This module is the closed external-parity gate for the molecular
GFN2-xTB electronic model: on every closed-shell row vibe-qc's SCC electronic
energy plus repulsion must agree with xtb's ``SCC energy - dispersion +
repulsion`` to the tolerance below, and the repulsion channel to 1e-6 Ha.

The D4 channel is pinned on both molecular paths. The public default route
(``GFN2Model``: faithful AES, self-consistent D4 built from the pinned
published reference data with SCC charges and covalent CN) must reproduce
xtb's dispersion channel to 1e-6 Ha and xtb's total to the electronic
tolerance on every closed-shell row, transition metals included. The
explicit ``legacy_eeq`` control's post-SCF native D4 must land within
0.6 mHa on rows made only of elements its own reference data covers (H-Ar
and Cu at the time of writing) and must warn and return zero D4 on rows
with other elements (the documented scope of that control, issue #132).

This complements ``tests/test_gfn2_xtb.py::test_faithful_energy8_with_native_d4_reference_data``,
the 0.1 mHa total-energy gate on the eight archived #184 geometries at
300 K: that gate checks totals on organic molecules, this one checks the
channels across the periodic table.

Open-shell rows are present in the fixture for completeness; the molecular
GFN2-xTB route is closed-shell only and the route plan refuses them, which the
last test pins.

History: before 2026-10-12 the faithful-AES multipole tables (R0_A, N_val,A of
Eq. 29) stopped at Ne, the H0 atomic radii differed from xtb's CRC table from
In onward, and Sr and Ba had no shell reference occupation. The rows that
exposed each defect are named in the tolerance notes below.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

try:
    from vibeqc import Atom, Molecule
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods import gfn2 as _gfn2_method
    from vibeqc.semiempirical.methods.gfn2 import GFN2D4UnsupportedWarning
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
    from vibeqc.semiempirical.routes import SemiempiricalRoutePlan

    _gfn2_available = True
except Exception:  # pragma: no cover - import guard
    _gfn2_available = False

requires_gfn2 = pytest.mark.skipif(not _gfn2_available, reason="GFN2-xTB unavailable")

_FIXTURE = Path(__file__).with_name("data") / "gfn2_xtb_parity_xtb670.json"
_ANGSTROM_TO_BOHR = 1.8897259885789233
_SYMBOL_TO_Z = {
    s: z
    for z, s in enumerate(
        "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
        "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
        "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
        "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
    )
    if z
}

# Electronic + repulsion agreement, Ha. Period-2 rows sit at 1-10 uHa; the
# heaviest closed-shell rows (TiCl4 -12 uHa, C6H6 -10 uHa) set the bound.
_ELECTRONIC_TOL = 2.0e-5
_REPULSION_TOL = 1.0e-6
# Legacy control: native D4 (vibe-qc's own reference data, EEQ charges)
# against xtb's self-consistent GFN2-D4 with the SCC charges: BeH2 is the
# largest of the H-Ne rows at 0.51 mHa. The supported element set is read
# from the loader, so a reference-data extension widens the gate instead of
# failing it.
_D4_TOL = 6.0e-4
# Default route: the coupled D4 term built from the pinned published
# reference data. Measured 2026-10-26 on all 45 closed-shell rows: the
# dispersion channel agrees with xtb to 0.05 uHa or better and the total to
# 12 uHa (TiCl4 -11.6, AlCl3 -7.1, C6H6 -6.4), i.e. the residual is the
# electronic channel bounded by _ELECTRONIC_TOL.
_COUPLED_D4_TOL = 1.0e-6
_TOTAL_TOL = _ELECTRONIC_TOL


def _rows():
    data = json.loads(_FIXTURE.read_text())
    return data["systems"]


_ROWS = _rows()
_CLOSED = sorted(name for name, row in _ROWS.items() if row["uhf"] == 0)
_OPEN = sorted(name for name, row in _ROWS.items() if row["uhf"] != 0)


def _molecule(row) -> "Molecule":
    atoms = [
        Atom(_SYMBOL_TO_Z[sym], [x * _ANGSTROM_TO_BOHR, y * _ANGSTROM_TO_BOHR, z * _ANGSTROM_TO_BOHR])
        for sym, x, y, z in row["atoms"]
    ]
    return Molecule(atoms, charge=int(row["charge"]), multiplicity=int(row["uhf"]) + 1)


@pytest.fixture(scope="module")
def params():
    try:
        return load_gfn2_params()
    except Exception as exc:  # pragma: no cover - offline host without a cache
        pytest.skip(f"GFN2 parameters unavailable: {exc}")


def _run(mol, params):
    opts = _xtb.XTBSccOptions()
    opts.charge_mixing = 0.2
    opts.max_iter = 3000
    opts.conv_tol_charge = 1e-9
    # Exercise the actual default, not an opt-in that could mask a regression.
    assert opts.aes_faithful is True
    return _xtb.run_gfn2_xtb(mol, params, opts)


def test_fixture_is_the_documented_xtb_run():
    data = json.loads(_FIXTURE.read_text())
    meta = data["_meta"]
    assert meta["oracle"].startswith("xtb 6.7.0")
    assert "--gfn 2" in meta["command"] and "--acc 0.0001" in meta["command"]
    assert len(data["systems"]) == 49
    for name, row in data["systems"].items():
        # total = SCC (electronic, with D4) + repulsion, as xtb prints it
        assert row["total"] == pytest.approx(row["scc"] + row["rep"], abs=2e-12), name


@requires_gfn2
@pytest.mark.parametrize("name", _CLOSED)
def test_closed_shell_electronic_and_repulsion_match_xtb(name, params):
    row = _ROWS[name]
    result = _run(_molecule(row), params)
    assert result.converged, f"{name}: SCC did not converge in {result.n_iter} iterations"
    xtb_electronic = row["scc"] - row["disp"]
    electronic = result.energy - result.e_repulsive
    assert abs(result.e_repulsive - row["rep"]) < _REPULSION_TOL, (
        f"{name}: repulsion {result.e_repulsive:.9f} vs xtb {row['rep']:.9f}"
    )
    assert abs(electronic - xtb_electronic) < _ELECTRONIC_TOL, (
        f"{name}: electronic {electronic:.8f} vs xtb {xtb_electronic:.8f} "
        f"({(electronic - xtb_electronic) * 1e6:+.1f} uHa)"
    )


@requires_gfn2
@pytest.mark.parametrize("name", _CLOSED)
def test_closed_shell_d4_channel_is_pinned_to_its_reference_scope(name, params):
    row = _ROWS[name]
    mol = _molecule(row)
    zs = {int(atom.Z) for atom in mol.atoms}
    _, supported, _ = _gfn2_method._gfn2_d4_reference_support(mol)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e_d4 = float(_gfn2_method._compute_gfn2_d4(mol, params))
    unsupported = [w for w in caught if issubclass(w.category, GFN2D4UnsupportedWarning)]
    if zs <= set(supported):
        assert not unsupported, f"{name}: unexpected D4 scope warning"
        assert abs(e_d4 - row["disp"]) < _D4_TOL, (
            f"{name}: native D4 {e_d4 * 1e3:.4f} mHa vs xtb {row['disp'] * 1e3:.4f} mHa"
        )
    else:
        assert unsupported, f"{name}: D4 outside the reference-data scope must warn"
        assert e_d4 == 0.0


@requires_gfn2
@pytest.mark.parametrize("name", _CLOSED)
def test_closed_shell_public_total_matches_xtb(name, params):
    """The public molecular route's total energy is at parity with xtb 6.7.0.

    ``GFN2Model.energy()`` runs the default model: faithful AES with the
    self-consistent D4 term built from the pinned published reference data
    (SCC charges in the zeta scaling, electronegativity-weighted covalent CN,
    neutral-reference ATM). Its ``e_dispersion`` is compared with xtb's
    dispersion channel and the total with xtb's total, so a regression in
    the SCC coupling or in the reference conversion shows up as such rather
    than as a cancelling pair. No element-scope warning is allowed on this
    route: the published table covers every row.
    """
    row = _ROWS[name]
    model = _gfn2_method.GFN2Model(_molecule(row), params, warn=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GFN2D4UnsupportedWarning)
        total = float(model.energy())
    result = model._last_result
    assert result.converged, f"{name}: public route did not converge"
    assert result.dispersion_sha256, f"{name}: the default route must run coupled D4"
    assert abs(result.e_dispersion - row["disp"]) < _COUPLED_D4_TOL, (
        f"{name}: coupled D4 {result.e_dispersion * 1e6:.2f} uHa "
        f"vs xtb {row['disp'] * 1e6:.2f} uHa"
    )
    assert abs(total - row["total"]) < _TOTAL_TOL, (
        f"{name}: total {total:.8f} vs xtb {row['total']:.8f} "
        f"({(total - row['total']) * 1e6:+.1f} uHa)"
    )


@requires_gfn2
@pytest.mark.parametrize("name", _OPEN)
def test_open_shell_rows_are_refused_by_the_molecular_route(name):
    row = _ROWS[name]
    with pytest.raises(NotImplementedError, match="closed-shell only"):
        SemiempiricalRoutePlan.from_request(
            "gfn2_xtb",
            boundary="molecule",
            charge=int(row["charge"]),
            multiplicity=int(row["uhf"]) + 1,
        )


@requires_gfn2
def test_cu2_dimer_is_at_parity_with_the_multipole_tables(params):
    """On the faithful-AES path Cu2 at 2.2 A carried a +4.2 mHa residual
    before the Z > 10 multipole tables landed (issue #520); it is 2 uHa now.
    The +19 to +48 mHa remainder that TestGFN2DFirstParameterProjection
    bounds is the default ad-hoc AES kernel at 300 K, which does not consume
    these tables and is unchanged."""
    row = _ROWS["Cu2"]
    result = _run(_molecule(row), params)
    assert result.converged
    electronic = result.energy - result.e_repulsive
    assert abs(electronic - (row["scc"] - row["disp"])) < 1.0e-5
