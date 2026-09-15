"""Selection of the Γ-CCM construction and the 2014 four-centre lineage.

The user-facing contract follows ruling R1 (2026-08-21): Paper-1 Γ-CCM is the
NEUTRAL finite-BvK-torus construction, whose admissible producers are the
``neutral-bloch`` and ``real-gamma`` routes; the union-and-weight four-centre
is the 2014 lineage (diagnosed non-variational in Paper 1). The paper-facing
``gamma`` / ``gamma-ccm`` spellings therefore fail closed naming both
producers. This covers both selector surfaces:

* the library-layer :func:`~vibeqc.periodic.ccm.run_ccm_scf` ``route=`` keyword
  where ``aiccm2026dev-a`` / ``aiccm-hf`` select the four-centre lineage and
  ``neutral-bloch`` / ``real-gamma`` select the two neutral construction
  producers, and
* the :class:`~vibeqc.periodic_jk_method.PeriodicJKMethod` enum, where the
  corresponding unwired selectors fail closed with precise pointers.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, run_periodic_job
from vibeqc.periodic.ccm import CCM_ROUTES, CCMSystem, resolve_ccm_route, run_ccm_scf
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic_jk_method import resolve_aiccm_correlation
from vibeqc.periodic_jk_method import (
    PeriodicJKMethod,
    describe_jk_method,
    resolve_jk_method_string,
    validate_jk_method,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2(nrep=(2, 1, 1)):
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _li(nrep=(1, 1, 1)):
    """Odd-electron open-shell control: Li atom (3e doublet) in an 8-bohr cube."""
    cell = PeriodicSystem(3, np.diag([8.0, 8.0, 8.0]),
                          [Atom(3, [4.0, 4.0, 4.0])], 0, 2)
    return CCMSystem(cell, nrep, "sto-3g")


def _h2_triplet():
    """Even-electron OPEN-shell control: triplet H₂ (2e, multiplicity 3) —
    the fixture class the closed-shell loops used to swallow silently."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 3)
    return CCMSystem(cell, (1, 1, 1), "sto-3g")


# --- library-layer route keyword -------------------------------------------

@pytest.mark.parametrize("alias,canon", [
    ("real-gamma", "real-gamma"), ("direct", "real-gamma"),
    ("aiccm-hf-direct", "real-gamma"),
    ("non-k", "real-gamma"),
    ("neutral-bloch", "neutral-bloch"), ("aiccm-ri", "neutral-bloch"),
    ("gdf", "neutral-bloch"), ("bloch-control", "neutral-bloch"),
    ("aiccm2026dev-a", "four-center"),
    ("four-center", "four-center"), ("4c", "four-center"), ("aiccm-hf", "four-center"),
])
def test_route_aliases_resolve(alias, canon):
    assert resolve_ccm_route(alias) == canon
    assert canon in CCM_ROUTES


def test_bad_route_raises():
    with pytest.raises(ValueError, match="unknown CCM route"):
        resolve_ccm_route("bogus")


@pytest.mark.parametrize(
    "route", ["aiccm2026dev-a-real-gamma", "aiccm2026dev-a-direct"]
)
def test_a_prefixed_real_gamma_route_aliases_fail_closed(route):
    with pytest.raises(ValueError, match="prefix denotes the union-and-weight"):
        resolve_ccm_route(route)


@pytest.mark.parametrize("route", ["gamma", "gamma-ccm", "gamma_ccm"])
def test_paper_facing_gamma_aliases_fail_closed(route):
    """IID 222 / ruling R1: the paper-facing Γ-CCM spelling denotes the
    NEUTRAL construction. With two admissible producers it must not silently
    select the diagnosis-only four-centre lineage -- it fails closed naming
    both producers (and the lineage route)."""
    with pytest.raises(ValueError, match="ruling R1") as exc:
        resolve_ccm_route(route)
    msg = str(exc.value)
    assert "NEUTRAL" in msg
    assert "route='real-gamma'" in msg and "route='neutral-bloch'" in msg
    assert "route='four-center'" in msg
    # The full dispatch path goes through the same rejection.
    with pytest.raises(ValueError, match="ruling R1"):
        run_ccm_scf(_h2(), route=route)


@pytest.mark.parametrize(
    "route", ["aiccm2026dev-a", "aiccm-hf"]
)
def test_gamma_ccm_selector_rejects_a_different_four_center_weight(route):
    with pytest.raises(ValueError, match="requires method='aiccm2026dev-a'"):
        run_ccm_scf(_h2(), route=route, method="union12")


def test_dev_a_alias_canonicalizes_case_insensitive_construction_method():
    ccm = _h2()
    got = run_ccm_scf(
        ccm,
        route="aiccm2026dev-a",
        method="AICCM2026DEV-A",
    )
    expected = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    assert got.energy == pytest.approx(expected.energy, abs=1e-12)


@pytest.mark.parametrize(
    "route",
    ["neutral-bloch", "gdf", "aiccm-ri", "real-gamma", "aiccm-hf-direct"],
)
@pytest.mark.parametrize("method", ["aiccm2026dev-a", "union12"])
def test_control_routes_reject_explicit_construction_method(route, method):
    with pytest.raises(ValueError, match="applies only to route='four-center'"):
        run_ccm_scf(_h2(), route=route, method=method)


def test_run_ccm_scf_dispatches_to_each_driver():
    """The default is Γ-CCM four-center; controls keep their own drivers."""
    ccm = _h2()
    assert run_ccm_scf(ccm, route="real-gamma").energy == pytest.approx(
        run_ccm_rhf_direct(ccm).energy, abs=1e-12)
    assert run_ccm_scf(ccm, route="neutral-bloch").energy == pytest.approx(
        run_ccm_rhf_gdf(ccm).energy, abs=1e-12)
    assert run_ccm_scf(ccm).energy == pytest.approx(
        run_ccm_rhf(ccm, method="aiccm2026dev-a").energy, abs=1e-12)


def test_neutral_bloch_and_real_gamma_agree_via_route_keyword():
    """Bloch and real-Gamma controls agree for the same neutral Hamiltonian."""
    ccm = _h2()
    e_direct = run_ccm_scf(ccm, route="real-gamma").energy / ccm.n_cells
    e_bloch = run_ccm_scf(ccm, route="neutral-bloch").energy
    assert e_direct == pytest.approx(e_bloch, abs=1e-8)


def test_real_gamma_records_exchange_q0():
    assert run_ccm_scf(_h2(), route="real-gamma").exchange_q0 == "BvK-ewald"


def test_route_ks_dispatch_and_gating():
    """KS-DFT dispatches on both neutral fitted-torus controls."""
    ccm = _h2()
    assert run_ccm_scf(ccm, route="neutral-bloch", functional="pbe").converged
    r = run_ccm_scf(ccm, route="real-gamma", functional="pbe")
    assert r.converged
    assert r.exchange_q0 == "BvK-ewald"


def test_route_four_center_dispatches_by_shell():
    """run_ccm_scf(route='four-center') reaches run_ccm_uhf / run_ccm_uks for
    open-shell clusters — the sibling of the real-gamma gate
    (test_ccm_uks_direct.py::test_route_real_gamma_dispatches_by_shell), keyed
    on multiplicity AND electron parity. Load-bearing detail: an EVEN-electron
    open-shell cluster (triplet H₂) previously fell through to the
    closed-shell loop SILENTLY (n_occ = n_elec // 2 ignores multiplicity) and
    returned a wrong closed-shell number."""
    # Odd-electron doublet → UHF / UKS four-center.
    li = _li()
    hf = run_ccm_scf(li, route="four-center")
    assert (hf.n_alpha, hf.n_beta) == (2, 1)            # CCMUHFResult
    ks = run_ccm_scf(li, route="four-center", functional="pbe")
    assert ks.open_shell is True                        # CCMKSResult, UKS path

    # Even-electron triplet → still the open-shell drivers (the former trap).
    trip = _h2_triplet()
    hf3 = run_ccm_scf(trip, route="four-center")
    assert (hf3.n_alpha, hf3.n_beta) == (2, 0)
    assert hf3.s_squared == pytest.approx(2.0, abs=1e-6)
    ks3 = run_ccm_scf(trip, route="four-center", functional="pbe")
    assert ks3.open_shell is True
    assert np.max(np.abs(ks3.density_alpha - ks3.density_beta)) > 1e-3

    # Closed shell keeps the closed-shell drivers (no behaviour change).
    h2 = _h2((1, 1, 1))
    assert run_ccm_scf(h2, route="four-center", functional="pbe").open_shell is False
    assert not hasattr(run_ccm_scf(h2, route="four-center"), "n_alpha")


def test_route_dev_a_alias_dispatches_open_shell_four_center():
    """The dev-a alias retains the four-center open-shell dispatch."""
    hf = run_ccm_scf(_h2_triplet(), route="aiccm2026dev-a")
    assert (hf.n_alpha, hf.n_beta) == (2, 0)
    assert not hasattr(hf, "raw")


def test_route_neutral_bloch_dispatches_by_shell():
    """run_ccm_scf(route='neutral-bloch') reaches KUHF/KUKS GDF wrappers
    (run_ccm_uhf_gdf / run_ccm_uks_gdf) for open-shell clusters; closed shells
    keep the KRHF/KRKS wrappers. Every result stays a CCMGDFResult — the
    dispatch is visible in ``.raw`` (per-spin multi-k fields + ⟨S²⟩, absent on
    the closed-shell result). Gates stay at (1,1,1), where the multi-k
    per-unit-cell spin convention coincides with the CCM supercell
    multiplicity (see run_ccm_uhf_gdf's spin-bookkeeping warning)."""
    # Odd-electron doublet → KUHF / KUKS via GDF.
    li = _li()
    hf = run_ccm_scf(li, route="neutral-bloch")
    assert hf.converged
    assert hf.raw.s_squared == pytest.approx(0.75, abs=0.05)   # KUHF doublet
    ks = run_ccm_scf(li, route="neutral-bloch", functional="pbe")
    assert ks.converged
    assert hasattr(ks.raw, "mo_energies_beta")                 # KUKS, per-spin

    # Even-electron triplet → still the open-shell driver (the former trap).
    trip = _h2_triplet()
    hf3 = run_ccm_scf(trip, route="neutral-bloch")
    assert hf3.converged
    assert hf3.raw.s_squared == pytest.approx(2.0, abs=0.05)   # aligned spins

    # Closed shell keeps the closed-shell drivers (no behaviour change).
    r = run_ccm_scf(_h2((1, 1, 1)), route="neutral-bloch")
    assert r.converged and not hasattr(r.raw, "s_squared")     # KRHF unchanged


# --- jk_method enum peers (surface shared with chi) ------------------------

@pytest.mark.parametrize("s,val,legacy", [
    ("gdf", "gdf", False),
    ("aiccm2026dev-a", "aiccm2026dev-a", True),
    ("real-gamma", "real-gamma", True),
    ("real_gamma", "real-gamma", True),
    ("chi", "aiccm2026dev-b", True),
    ("chi-ccm", "aiccm2026dev-b", True),
    ("aiccm2026dev-b", "aiccm2026dev-b", True),
])
def test_jk_method_string_aliases(s, val, legacy):
    """Dev-era spellings keep resolving, each with a DeprecationWarning that
    names the front door (M1, D-2); the runner-side contract is pinned in
    tests/test_periodic_feature_input_guards.py."""
    if legacy:
        with pytest.warns(DeprecationWarning, match="method='aiccm'"):
            assert resolve_jk_method_string(s).value == val
    else:
        assert resolve_jk_method_string(s).value == val


@pytest.mark.parametrize("s", ["gamma", "gamma-ccm", "gamma_ccm"])
def test_jk_method_bare_gamma_spellings_fail_closed_with_r1(s):
    """D-2: the runner resolver now agrees with resolve_ccm_route: one word
    cannot pick a producer (ruling R1), so the bare spelling is refused, not
    bound to the four-centre lineage."""
    with pytest.raises(ValueError, match="ruling R1") as exc:
        resolve_jk_method_string(s)
    msg = str(exc.value)
    assert "variant='real-gamma'" in msg and "variant='neutral-bloch'" in msg


@pytest.mark.parametrize(
    "s", ["neutral-bloch", "neutral_bloch", "bloch-control", "gdf-control", "aiccm-ri"]
)
def test_jk_method_gdf_colliding_aliases_fail_closed(s):
    """D-2b: the words the CCM library uses for the neutral-Bloch producer
    no longer mean plain unit-cell GDF on the runner; they point at the
    front door and at the library entry that runs today."""
    with pytest.raises(ValueError, match="variant='neutral-bloch'") as exc:
        resolve_jk_method_string(s)
    assert "route='neutral-bloch'" in str(exc.value)


@pytest.mark.parametrize(
    "name", ["aiccm2026dev-a-real-gamma", "aiccm2026dev-a-direct"]
)
def test_a_prefixed_real_gamma_jk_strings_fail_closed(name):
    with pytest.raises(ValueError, match="prefix denotes the union-and-weight"):
        resolve_jk_method_string(name)


def test_jk_method_enum_has_ccm_selectors():
    vals = {m.value for m in PeriodicJKMethod}
    assert {"aiccm2026dev-a", "real-gamma", "neutral-bloch", "aiccm2026dev-b"} <= vals
    assert "aiccm2026dev-a-real-gamma" not in vals  # rejected input, not a value


def test_jk_method_describe_distinguishes_routes():
    a = describe_jk_method(PeriodicJKMethod.AICCM2026DEV_A)
    g = describe_jk_method(PeriodicJKMethod.GDF)
    r = describe_jk_method(PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA)
    b = describe_jk_method(PeriodicJKMethod.AICCM2026DEV_B)
    assert "union-and-weight" in a.lower() and "four-center" in a.lower()
    # M3a: the description stopped advertising a fail-closed runner surface.
    assert "fails closed" not in a.lower()
    # The maintainer-required experimental label: -a and its real-gamma
    # control stay marked in the .out-visible description, like -b.
    assert "experimental" in a.lower()
    assert "experimental" in r.lower()
    # GDF is not an AICCM route; the neutral Bloch producer is a variant.
    assert "neutral fitted-torus" in g.lower() and "not an aiccm route" in g.lower()
    # Ruling R1: real-gamma IS a producer of the neutral Γ-CCM torus and is
    # distinguished from the four-centre construction, not from Γ-CCM.
    assert "neutral fitted-torus" in r.lower() and "not the four-center" in r.lower()
    assert all(method in b for method in ("RHF", "RKS", "UHF", "UKS"))
    assert "translation-group-character" in b.lower()
    assert "same specified χ hamiltonian" in b.lower()
    # M1: every AICCM description opens with the front-door selector and
    # keeps the trailing experimental token (the .out "J/K method" line).
    n = describe_jk_method(PeriodicJKMethod.NEUTRAL_BLOCH)
    assert a.startswith("aiccm (four-center)")
    assert r.startswith("aiccm (real-gamma)")
    assert n.startswith("aiccm (neutral-bloch)")
    assert b.startswith("aiccm (chi)")
    for text in (a, r, n, b):
        assert "experimental" in text.rsplit(";", 1)[-1].lower(), text


def test_every_ccm_selector_validates_now():
    """Every Γ-CCM selector is wired. This test used to pin the four-centre
    construction as fail-closed with a library pointer; M3a wired its runner
    arm (tests/test_periodic_ccm_four_center_adapter.py), which was the last
    one, so ``_UNWIRED_CCM_ROUTES`` is now empty. The mechanism stays for the
    next formulation: a member left out of ``_IMPLEMENTED`` still raises a
    NotImplementedError naming its library entry rather than crashing."""
    from vibeqc.periodic_jk_method import _IMPLEMENTED, _UNWIRED_CCM_ROUTES

    lattice = np.diag([6.0, 6.0, 6.0])
    for member in (PeriodicJKMethod.AICCM2026DEV_A,
                   PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
                   PeriodicJKMethod.AICCM2026DEV_B):
        assert member in _IMPLEMENTED, member
        validate_jk_method(member, lattice=lattice, basis_name="sto-3g")
    assert _UNWIRED_CCM_ROUTES == frozenset()


def test_run_periodic_job_a_routes_raise_pointer_not_crash():
    """run_periodic_job(jk_method=<construction route>) fails closed with
    the pointer rather than crashing — the enum peer is reachable
    end-to-end. 'real-gamma' runs since 2026-07-26 (gated in
    test_periodic_ccm_real_gamma_adapter.py)."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        # Legacy construction selector: since M3a it RUNS (the four-centre
        # arm is wired), and still warns with the front-door spelling first.
        with pytest.warns(DeprecationWarning, match="variant='four-center'"):
            legacy = run_periodic_job(cell, basis, method="RHF",
                                      jk_method="aiccm2026dev-a",
                                      kpoints=(2, 1, 1),
                                      initial_guess="HCORE",
                                      output=os.path.join(d, "t.out"))
        assert legacy.converged
        # D-2: the bare word is the R1 refusal, not a pointer.
        with pytest.raises(ValueError, match="ruling R1"):
            run_periodic_job(cell, basis, method="RHF", jk_method="gamma",
                             kpoints=(2, 1, 1), output=os.path.join(d, "t.out"))
        # M3a: the four-centre variant runs too, so every AICCM formulation
        # is now reachable from the front door. Its energy is the library
        # driver's total cyclic-cluster energy divided by N_c.
        fc = run_periodic_job(cell, basis, method="aiccm",
                              variant="four-center", kpoints=(2, 1, 1),
                              initial_guess="HCORE",
                              output=os.path.join(d, "fc.out"))
        assert fc.converged
        # M1b: the neutral-Bloch variant runs;
        nb = run_periodic_job(cell, basis, method="aiccm",
                              variant="neutral-bloch", kpoints=(2, 1, 1),
                              output=os.path.join(d, "nb.out"))
        assert nb.converged


# --- M4b (#778): the correlation= selector ---------------------------------
#
# ``correlation`` rides beside ``variant`` and is optional: None means an
# SCF-only run, which is the ordinary complete request. These cover the
# guards only -- every one of them fires before any SCF, so they are cheap.


@pytest.mark.parametrize("spelling", ["mp2", "MP2", " Mp2 "])
def test_resolve_aiccm_correlation_normalises(spelling):
    """Case and surrounding space fold, matching resolve_aiccm_variant."""

    assert resolve_aiccm_correlation(spelling) == "mp2"


def test_resolve_aiccm_correlation_none_is_scf_only():
    """None passes through: unlike variant, this selector is optional."""

    assert resolve_aiccm_correlation(None) is None


def test_resolve_aiccm_correlation_unknown_fails_closed():
    with pytest.raises(ValueError, match="unknown AICCM correlation") as exc:
        resolve_aiccm_correlation("ccsd(t)")
    # The message must name the vocabulary AND the SCF-only escape, so a
    # caller who guessed a spelling is not left to guess again.
    assert "'mp2'" in str(exc.value)
    assert "None" in str(exc.value)


def test_correlation_requires_method_aiccm():
    """correlation= is an AICCM selector; it fails closed on other methods."""

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="require method='aiccm'"):
            run_periodic_job(cell, basis, method="RHF", correlation="mp2",
                             output=os.path.join(d, "t.out"))


def _h2_cell_and_basis():
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return cell, BasisSet(cell.unit_cell_molecule(), "sto-3g")


def test_correlation_on_chi_refuses_as_unwired():
    """chi refuses rather than quietly running SCF-only.

    A job that asked for correlation and got an SCF number back would be
    indistinguishable from one that never asked, so this fails loudly and
    names the library entry point that does exist.
    """

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(NotImplementedError, match="four-center") as exc:
            run_periodic_job(cell, basis, method="aiccm", variant="chi",
                             correlation="mp2", kpoints=(2, 1, 1),
                             output=os.path.join(d, "t.out"))
        assert "chi.posthf" in str(exc.value)


def test_correlation_on_neutral_bloch_names_the_exact_workaround():
    """neutral-bloch's refusal is a representation fact, not a to-do.

    Its reference is per-k Bloch while the correlation drivers work in the
    real-Gamma supercell space. Ruling R1 makes real-gamma the SAME neutral
    Hamiltonian in that representation, so the refusal must point there --
    an exact substitute, not an approximation the caller has to assess.
    """

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(NotImplementedError,
                           match="real-gamma") as exc:
            run_periodic_job(cell, basis, method="aiccm",
                             variant="neutral-bloch",
                             correlation="mp2", kpoints=(2, 1, 1),
                             output=os.path.join(d, "t.out"))
        msg = str(exc.value)
        assert "Bloch" in msg and "real-Gamma" in msg
        # It must not be described as merely not-yet-wired.
        assert "library-only" not in msg


def test_correlation_refuses_a_ks_reference():
    """MP2 needs an HF determinant; a KS one has no defined meaning here."""

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(NotImplementedError, match="Hartree-Fock"):
            run_periodic_job(cell, basis, method="aiccm",
                             variant="four-center", functional="pbe",
                             correlation="mp2", kpoints=(2, 1, 1),
                             output=os.path.join(d, "t.out"))


def test_four_center_correlation_runs_and_records():
    """End to end: the wired arm runs, reports and cites the bare lineage.

    H2/STO-3G at a (1,1,1) torus -- the smallest cluster that exercises the
    whole path, because the dense n_ref_ao**4 tensor is what makes this a
    small-cluster tool.
    """

    import tomllib

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        stem = os.path.join(d, "fcmp2")
        res = run_periodic_job(
            cell, basis, method="aiccm", variant="four-center",
            correlation="mp2", aiccm_lattice_extension=(1, 1, 1),
            initial_guess="HCORE", output=stem)
        assert res.converged
        # The correlation actually ran and is attached.
        assert res.correlation is not None
        # MP2 correlation is variationally negative.
        assert res.e_correlation < 0.0
        assert res.e_total_correlated == pytest.approx(
            res.energy + res.e_correlation, abs=1e-12)
        # The citation stamp is the BARE lineage, never the neutral-RI one:
        # this SCF is the union-and-weight construction (ruling R1).
        assert res.correlation.backend == "aiccm2026dev-a-mp2"
        # The .out carries the block, next to the SCF energy it relates to.
        out = open(stem + ".out", encoding="utf-8").read()
        assert "AICCM correlation" in out
        assert "aiccm2026dev-a-mp2" in out
        # ... and the manifest records the request.
        run = tomllib.loads(
            open(stem + ".system", encoding="utf-8").read())["run"]
        assert run["aiccm_correlation"] == "mp2"
        assert run["method_status"] == "experimental"


def test_scf_only_manifest_has_no_correlation_key():
    """The manifest contract is additive: an SCF-only run is unchanged."""

    import tomllib

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        stem = os.path.join(d, "fconly")
        res = run_periodic_job(
            cell, basis, method="aiccm", variant="four-center",
            aiccm_lattice_extension=(1, 1, 1),
            initial_guess="HCORE", output=stem)
        assert res.correlation is None
        run = tomllib.loads(
            open(stem + ".system", encoding="utf-8").read())["run"]
        assert "aiccm_correlation" not in run


def test_real_gamma_correlation_cites_the_neutral_ri_lineage():
    """The real-Γ arm is the NEUTRAL construction, so it owes the RI row.

    The two wired arms must never share a citation route: four-center is the
    union-and-weight lineage, real-gamma the neutral fitted torus (ruling
    R1). A regression that collapsed them would be invisible in any energy
    test -- the numbers would still be right, attributed to the wrong papers.
    """

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        stem = os.path.join(d, "rgmp2")
        res = run_periodic_job(
            cell, basis, method="aiccm", variant="real-gamma",
            correlation="mp2", aiccm_lattice_extension=(1, 1, 1),
            output=stem)
        assert res.converged
        assert res.correlation is not None
        assert res.correlation.backend == "aiccm2026dev-a-ri-mp2"
        assert res.e_correlation < 0.0
        out = open(stem + ".out", encoding="utf-8").read()
        # The printed lineage follows the stamp, not a hard-coded word.
        assert "neutral-RI (fitted torus)" in out
        assert "bare four-centre" not in out


def test_retain_cderi_does_not_move_the_scf_energy():
    """Pre-building L for the correlation must be exactly energy-neutral.

    The correlation arm hands the SCF an L the adapter built instead of
    letting the driver build its own. Same helper, same arguments, so this
    has to be bit-identical -- if it ever is not, the correlation arm is
    silently changing the SCF number it reports beside its own.
    """

    from vibeqc.periodic.ccm.real_gamma_runner import run_real_gamma_scf

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    for mesh in [(1, 1, 1), (2, 1, 1)]:
        off = run_real_gamma_scf(cell, "sto-3g", "RHF", mesh,
                                 retain_cderi=False)
        on = run_real_gamma_scf(cell, "sto-3g", "RHF", mesh,
                                retain_cderi=True)
        assert off.energy == on.energy, mesh
        # And the retention itself behaves: freed by default, held on demand.
        assert off.cderi is None
        assert on.cderi is not None


# --- M4b second vocabulary entry: correlation="ccsd" ------------------------


@pytest.mark.parametrize("variant,route", [
    ("four-center", "aiccm2026dev-a-ccsd(t)"),
    ("real-gamma", "aiccm2026dev-a-ri-ccsd(t)"),
])
def test_ccsd_runs_on_both_wired_arms_with_its_own_lineage(variant, route):
    """Each arm's CCSD cites its own construction, as its MP2 does."""

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        stem = os.path.join(d, "cc")
        res = run_periodic_job(
            cell, basis, method="aiccm", variant=variant, correlation="ccsd",
            aiccm_lattice_extension=(1, 1, 1), initial_guess="HCORE",
            output=stem)
        assert res.converged
        assert res.correlation is not None
        assert res.correlation.backend == route
        assert res.e_correlation < 0.0
        out = open(stem + ".out", encoding="utf-8").read()
        assert "treatment           = CCSD(T)" in out


def test_triples_that_ran_are_cited_and_mp2_does_not_cite_them():
    """The (T) half of the key tracks what the call actually computed.

    compute_triples is a runtime flag, so a class-level or request-level key
    would cite Raghavachari for a run with no triples. This is the behavioural
    end of that: the reference list itself must differ.
    """

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        cc = os.path.join(d, "cc")
        mp = os.path.join(d, "mp")
        run_periodic_job(cell, basis, method="aiccm", variant="four-center",
                         correlation="ccsd",
                         aiccm_lattice_extension=(1, 1, 1),
                         initial_guess="HCORE", output=cc)
        run_periodic_job(cell, basis, method="aiccm", variant="four-center",
                         correlation="mp2",
                         aiccm_lattice_extension=(1, 1, 1),
                         initial_guess="HCORE", output=mp)
        assert "Raghavachari" in open(cc + ".out", encoding="utf-8").read()
        assert "Raghavachari" not in open(mp + ".out", encoding="utf-8").read()


def test_ccsd_without_triples_drops_the_triples_key():
    """Library level, because the runner always takes the driver default.

    Guards the other side of the per-call key: with compute_triples off the
    stamp must lose its (T), or a run that computed no triples would still
    cite them.
    """

    from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
    from vibeqc.periodic.ccm.scf import run_ccm_rhf

    ccm = _h2((1, 1, 1))
    scf = run_ccm_rhf(ccm, method="aiccm2026dev-a", initial_guess="HCORE")
    with_t = run_ccm_ccsd(ccm, scf, method="aiccm2026dev-a",
                          compute_triples=True)
    no_t = run_ccm_ccsd(ccm, scf, method="aiccm2026dev-a",
                        compute_triples=False)
    assert with_t.backend == "aiccm2026dev-a-ccsd(t)"
    assert no_t.backend == "aiccm2026dev-a-ccsd"


def test_ccsd_refuses_an_open_shell_reference_on_four_center_only():
    """The four-centre arm has no open-shell CCSD; real-gamma does.

    run_ccm_uccsd takes no ``method=``, so it cannot build the
    union-and-weight reference this variant converged on. That same
    neutral-only nature is what makes it correct on real-gamma, so the
    refusal is arm-specific and must point there rather than reading as a
    blanket "CCSD is closed-shell".
    """

    cell = PeriodicSystem(3, np.diag([8.0, 8.0, 8.0]),
                          [Atom(3, [4.0, 4.0, 4.0])], 0, 2)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(NotImplementedError, match="closed-shell") as exc:
            run_periodic_job(cell, basis, method="aiccm",
                             variant="four-center", correlation="ccsd",
                             aiccm_lattice_extension=(1, 1, 1),
                             output=os.path.join(d, "t.out"))
        # It must name the escape that does work open-shell.
        assert "correlation='mp2'" in str(exc.value)


# --- M4b third vocabulary entry: the DLPNO routes ---------------------------


@pytest.mark.parametrize("canonical,dlpno,route", [
    ("mp2", "dlpno-mp2", "aiccm2026dev-a-dlpno-mp2"),
    ("ccsd", "dlpno-ccsd", "aiccm2026dev-a-dlpno-ccsd(t)"),
])
def test_dlpno_reproduces_the_canonical_route_at_zero_truncation(
    canonical, dlpno, route
):
    """The correctness anchor for the DLPNO wiring.

    At the default zero truncations DLPNO IS the canonical RI correlation --
    same reference, same cderi, no pairs dropped -- so any disagreement here
    means the driver was handed the wrong reference or the wrong L, which is
    the one failure the on-reference dispatch exists to prevent. The two must
    nonetheless carry DIFFERENT citation routes, because a DLPNO run owes the
    local-correlation papers a canonical one does not.
    """

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        ref = run_periodic_job(
            cell, basis, method="aiccm", variant="real-gamma",
            correlation=canonical, aiccm_lattice_extension=(1, 1, 1),
            output=os.path.join(d, "a"))
        loc = run_periodic_job(
            cell, basis, method="aiccm", variant="real-gamma",
            correlation=dlpno, aiccm_lattice_extension=(1, 1, 1),
            output=os.path.join(d, "b"))
        assert ref.converged and loc.converged
        assert loc.e_correlation == pytest.approx(
            ref.e_correlation, rel=0, abs=1e-12)
        assert loc.correlation.backend == route
        assert loc.correlation.backend != ref.correlation.backend


def test_open_shell_dlpno_mp2_runs_and_cites_its_own_route():
    """dlpno-mp2 has an open-shell driver, unlike ccsd."""

    cell = PeriodicSystem(3, np.diag([8.0, 8.0, 8.0]),
                          [Atom(3, [4.0, 4.0, 4.0])], 0, 2)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        res = run_periodic_job(
            cell, basis, method="aiccm", variant="real-gamma",
            correlation="dlpno-mp2", aiccm_lattice_extension=(1, 1, 1),
            output=os.path.join(d, "o"))
        assert res.converged
        assert res.correlation.backend == "aiccm2026dev-a-dlpno-ump2"


@pytest.mark.parametrize("corr", ["dlpno-mp2", "dlpno-ccsd"])
def test_dlpno_refuses_on_the_four_centre_arm(corr):
    """There is no four-centre DLPNO, and the reason is structural.

    DLPNO screens on a fitted reference and every driver takes the neutral
    cderi. The bare four-centre operator has no RI decomposition -- which is
    exactly why that arm is dense and small-cluster only -- so nothing could
    be dispatched and nothing stamps such a label.
    """

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(NotImplementedError, match="real-gamma") as exc:
            run_periodic_job(cell, basis, method="aiccm",
                             variant="four-center", correlation=corr,
                             aiccm_lattice_extension=(1, 1, 1),
                             output=os.path.join(d, "t.out"))
        assert "RI decomposition" in str(exc.value)


def test_open_shell_ccsd_runs_on_real_gamma_with_the_uccsd_lineage():
    """The other half of the arm-specific refusal above.

    run_ccm_uccsd is neutral-only, which is exactly what the real-Gamma
    reference is, so open-shell CCSD runs here and owes the open-shell
    neutral-RI row -- a route with deliberately no bare four-centre
    counterpart. Checked against its DLPNO sibling at zero truncation, where
    the two are the same correlation.
    """

    cell = PeriodicSystem(3, np.diag([8.0, 8.0, 8.0]),
                          [Atom(3, [4.0, 4.0, 4.0])], 0, 2)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with tempfile.TemporaryDirectory() as d:
        canon = run_periodic_job(
            cell, basis, method="aiccm", variant="real-gamma",
            correlation="ccsd", aiccm_lattice_extension=(1, 1, 1),
            output=os.path.join(d, "c"))
        loc = run_periodic_job(
            cell, basis, method="aiccm", variant="real-gamma",
            correlation="dlpno-ccsd", aiccm_lattice_extension=(1, 1, 1),
            output=os.path.join(d, "l"))
        assert canon.converged and loc.converged
        assert canon.correlation.backend == "aiccm2026dev-a-ri-uccsd(t)"
        assert loc.correlation.backend == "aiccm2026dev-a-dlpno-uccsd(t)"
        assert canon.e_correlation == pytest.approx(
            loc.e_correlation, rel=0, abs=1e-12)


# --- M8 provenance: the manifest records what was CITED, not just asked ----


@pytest.mark.parametrize("variant,corr,route", [
    ("real-gamma", "ccsd", "aiccm2026dev-a-ri-ccsd(t)"),
    ("four-center", "mp2", "aiccm2026dev-a-mp2"),
])
def test_manifest_records_the_correlation_citation_route(variant, corr, route):
    """``aiccm_correlation`` is the request; this is what was actually cited.

    The two differ in exactly what a consumer cannot re-derive from the
    request: the lineage (ruling R1) and whether the perturbative triples
    ran, since compute_triples is a runtime flag. A provenance reader that
    only had the request could not tell a neutral-RI number from a
    union-and-weight one.
    """

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        stem = os.path.join(d, "m")
        run_periodic_job(cell, basis, method="aiccm", variant=variant,
                         correlation=corr, aiccm_lattice_extension=(1, 1, 1),
                         initial_guess="HCORE", output=stem)
        manifest = open(stem + ".system", encoding="utf-8").read()
        assert f'aiccm_correlation = "{corr}"' in manifest
        assert f'aiccm_correlation_route = "{route}"' in manifest


def test_scf_only_run_claims_no_correlation_provenance():
    """An absent key must never read as a claim.

    The negative control for the guard above: a run that asked for no
    correlation writes neither key, rather than an empty or "none" value a
    reader might mistake for a recorded route.
    """

    cell, basis = _h2_cell_and_basis()
    with tempfile.TemporaryDirectory() as d:
        stem = os.path.join(d, "s")
        run_periodic_job(cell, basis, method="aiccm", variant="real-gamma",
                         aiccm_lattice_extension=(1, 1, 1),
                         initial_guess="HCORE", output=stem)
        manifest = open(stem + ".system", encoding="utf-8").read()
        assert "aiccm_correlation" not in manifest
