"""``run_job`` route for double hybrids, and the construction it reports.

GitLab #365. Double hybrids used to be reachable only through
``vibeqc.run_double_hybrid`` and its four named wrappers, while the release
paper named ``run_job`` as the molecular entry point. ``method="b2plyp"``
and friends now dispatch to the same dispatcher.

Two properties matter and are pinned here:

* the two entry points return the *same number* for the same request --
  run_job spells several defaults as ``None`` where the dispatcher spells
  them ``True``, and coercing those with ``bool()`` silently turned density
  fitting off;
* ``method="rks", functional="b2plyp"`` still raises. That spelling asks for
  the SCF half alone, which is not the method it names, and
  ``tests/test_b2plyp.py`` pins the refusal itself.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    DoubleHybridResult,
    Functional,
    Molecule,
    MP2Options,
    run_b2plyp,
    run_double_hybrid,
    run_job,
)
from vibeqc.runner import _DOUBLE_HYBRID_METHODS


ANGSTROM_TO_BOHR = 1.8897261254578281

DOUBLE_HYBRID_METHODS = ("b2plyp", "dsd-pbep86", "revdsd-pbep86", "pwpb95")


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.1173]),
            Atom(1, [0.0, 1.4311, -0.9379]),
            Atom(1, [0.0, -1.4311, -0.9379]),
        ],
        0,
        1,
    )


def _oh_doublet() -> Molecule:
    return Molecule(
        [Atom(8, [0, 0, 0]), Atom(1, [0, 0, 0.97 * ANGSTROM_TO_BOHR])],
        0,
        2,
    )


# ---------------------------------------------------------------------
# The method keys
# ---------------------------------------------------------------------


def test_every_dispatched_method_key_is_a_double_hybrid():
    """The runner's dispatch set and the functional resolver agree.

    A key here that did not resolve to an ``is_double_hybrid`` functional
    would be routed to ``run_double_hybrid``, which raises ValueError for
    anything else -- a confusing failure for a name the Method literal
    advertises.
    """
    assert set(DOUBLE_HYBRID_METHODS) == set(_DOUBLE_HYBRID_METHODS)
    for name in DOUBLE_HYBRID_METHODS:
        assert Functional(name).is_double_hybrid, name


# ---------------------------------------------------------------------
# Route parity
# ---------------------------------------------------------------------


@pytest.mark.parametrize("multiplicity", [1, 2])
def test_run_job_matches_the_standalone_dispatcher(tmp_path, multiplicity):
    """Same request, same number, to the last bit.

    The closed-shell leg would have drifted by ~1e-6 Ha if run_job's
    ``density_fit=None`` were coerced with ``bool()``: the dispatcher's
    documented default is DF on both steps, and ``bool(None)`` is False.
    """
    mol = _h2o() if multiplicity == 1 else _oh_doublet()
    basis_name = "cc-pvdz"

    via_run_job = run_job(
        mol,
        basis=basis_name,
        method="b2plyp",
        output=tmp_path / "rj",
        progress=False,
    )
    via_wrapper = run_b2plyp(mol, BasisSet(mol, basis_name))

    assert isinstance(via_run_job, DoubleHybridResult)
    assert via_run_job.e_total == pytest.approx(via_wrapper.e_total, abs=1e-12)
    assert via_run_job.rks.energy == pytest.approx(
        via_wrapper.rks.energy, abs=1e-12
    )
    assert via_run_job.mp2.e_correlation == pytest.approx(
        via_wrapper.mp2.e_correlation, abs=1e-12
    )


def test_run_job_writes_the_standard_sidecars(tmp_path):
    """An unspecified sidecar is written because the route produces one.

    run_job spells "not specified" as None for the Molden and population
    files; the double-hybrid route has a Gaussian AO wavefunction (its SCF
    half), so None resolves to written, as it does for every other route
    that exposes one.
    """
    run_job(
        _h2o(),
        basis="cc-pvdz",
        method="b2plyp",
        output=tmp_path / "dh",
        progress=False,
    )
    produced = {p.name for p in tmp_path.iterdir()}
    assert {
        "dh.out",
        "dh.system",
        "dh.bibtex",
        "dh.references",
        "dh.molden",
        "dh.xyz",
        "dh.population.txt",
        "dh.population.json",
        "dh.qvf",
    } <= produced


def test_run_job_honours_mp2_options_aux_and_df(tmp_path):
    """``mp2_options`` selects the correction's DF choice, nothing else.

    The scaling coefficients belong to the functional on this route, so
    only ``density_fit`` / ``aux_basis`` are read from the options object.
    """
    mp2_opts = MP2Options()
    mp2_opts.density_fit = False

    canonical = run_job(
        _h2o(),
        basis="cc-pvdz",
        method="b2plyp",
        mp2_options=mp2_opts,
        output=tmp_path / "canon",
        progress=False,
    )
    ri = run_double_hybrid(
        _h2o(), BasisSet(_h2o(), "cc-pvdz"), "b2plyp", density_fit_mp2=True
    )
    # RI-MP2 and canonical MP2 agree to sub-microhartree on this basis but
    # are not the same calculation, so the option must have been honoured.
    assert canonical.e_total != ri.e_total
    assert canonical.e_total == pytest.approx(ri.e_total, abs=1e-5)


# ---------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------


@pytest.mark.parametrize("method", DOUBLE_HYBRID_METHODS)
def test_plain_ks_spelling_still_raises(tmp_path, method):
    """The new route does not weaken the old guard.

    ``method="rks"`` with a double-hybrid functional still returns only the
    SCF component, so it still fails closed -- and still before any output.
    """
    with pytest.raises(
        NotImplementedError,
        match=r"omit the required perturbative correlation",
    ):
        run_job(
            _h2o(),
            basis="sto-3g",
            method="rks",
            functional=method,
            output=tmp_path / "nope",
            progress=False,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimize": True},
        {"hessian": True},
        {"tddft": True},
        {"solvent": "water"},
        {"frozen_core": True},
        {"atomization": True},
        {"active_space": (2, 2)},
        {"cosx": True},
    ],
)
def test_unsupported_run_job_features_refuse_before_output(tmp_path, kwargs):
    """A feature the dispatcher does not carry refuses; it is not dropped.

    Silently ignoring ``optimize=True`` would report a double-hybrid energy
    for a geometry nobody optimized -- the shape of defect the double-hybrid
    guard exists to prevent (compare GitLab #357).
    """
    with pytest.raises(NotImplementedError, match=r"not carried by the double-hybrid"):
        run_job(
            _h2o(),
            basis="sto-3g",
            method="b2plyp",
            output=tmp_path / "nope",
            progress=False,
            **kwargs,
        )
    assert list(tmp_path.iterdir()) == []


def test_functional_kwarg_conflicting_with_the_method_raises(tmp_path):
    """A double-hybrid method already names its functional."""
    with pytest.raises(ValueError, match=r"already names its functional"):
        run_job(
            _h2o(),
            basis="sto-3g",
            method="b2plyp",
            functional="pbe0",
            output=tmp_path / "nope",
            progress=False,
        )
    assert list(tmp_path.iterdir()) == []


def test_matching_functional_kwarg_is_accepted(tmp_path):
    """Naming the same functional twice is redundant, not wrong."""
    result = run_job(
        _h2o(),
        basis="cc-pvdz",
        method="b2plyp",
        functional="b2plyp",
        output=tmp_path / "dh",
        progress=False,
    )
    assert isinstance(result, DoubleHybridResult)


# ---------------------------------------------------------------------
# The open-shell construction, on all three surfaces
# ---------------------------------------------------------------------


def test_open_shell_out_names_roks_and_the_rohf_mp2_correction(tmp_path):
    """The ``.out`` describes the calculation that ran.

    Before #365 an open-shell double hybrid reported ``RKS reference`` for a
    ROKS SCF and ``MP2 correlation`` for a semicanonical RI-ROHF-MP2
    doubles-only term; neither ROKS nor ROHF appeared anywhere in the file.
    """
    run_job(
        _oh_doublet(),
        basis="cc-pvdz",
        method="b2plyp",
        output=tmp_path / "os",
        progress=False,
    )
    text = (tmp_path / "os.out").read_text()

    assert "ROKS reference" in text
    assert "RI-ROHF-MP2 doubles" in text
    assert "RKS reference" not in text
    assert "spin-pure ROKS SCF + semicanonical RI-ROHF-MP2" in text
    assert "doubles only" in text


def test_closed_shell_out_still_names_rks(tmp_path):
    """The closed-shell labels were already correct and do not move."""
    run_job(
        _h2o(),
        basis="cc-pvdz",
        method="b2plyp",
        output=tmp_path / "cs",
        progress=False,
    )
    text = (tmp_path / "cs.out").read_text()

    assert "RKS reference" in text
    assert "MP2 correlation" in text
    assert "ROKS" not in text
    assert "hybrid RKS SCF + scaled RI-MP2" in text


@pytest.mark.parametrize(
    ("multiplicity", "reference", "correlation"),
    [(1, "RKS", "mp2"), (2, "ROKS", "rohf-mp2")],
)
def test_system_manifest_records_the_reference(
    tmp_path, multiplicity, reference, correlation
):
    """A consumer can tell the two branches apart without parsing prose."""
    mol = _h2o() if multiplicity == 1 else _oh_doublet()
    run_job(
        mol,
        basis="cc-pvdz",
        method="b2plyp",
        output=tmp_path / "dh",
        progress=False,
    )
    manifest = tomllib.loads((tmp_path / "dh.system").read_text())

    assert manifest["run"]["scf_reference"] == reference
    assert manifest["run"]["correlation_method"] == correlation
    assert manifest["run"]["open_shell"] is (multiplicity > 1)


def test_open_shell_bibtex_cites_roks_and_semicanonical_rohf_mp2(tmp_path):
    """Citing an open-shell double hybrid needs the components it ran.

    The ROKS and ROHF-MP2 routes already existed in ``database.toml``; the
    dispatcher just never fired them, so a user publishing an open-shell
    B2PLYP number had no route to the method actually used.
    """
    run_job(
        _oh_doublet(),
        basis="cc-pvdz",
        method="b2plyp",
        output=tmp_path / "os",
        progress=False,
    )
    keys = (tmp_path / "os.bibtex").read_text()

    assert "grimme_b2plyp_2006" in keys
    assert "roothaan_rohf_1960" in keys
    assert "knowles_rmp2_1991" in keys
    assert "feyereisen_rimp2_1993" in keys


def test_closed_shell_bibtex_does_not_gain_open_shell_references(tmp_path):
    """Truthful citation runs both ways: a closed-shell run used neither."""
    run_job(
        _h2o(),
        basis="cc-pvdz",
        method="b2plyp",
        output=tmp_path / "cs",
        progress=False,
    )
    keys = (tmp_path / "cs.bibtex").read_text()

    assert "grimme_b2plyp_2006" in keys
    assert "roothaan_rohf_1960" not in keys
    assert "knowles_rmp2_1991" not in keys
