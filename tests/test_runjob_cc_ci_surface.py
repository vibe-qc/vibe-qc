"""run_job CC/CI ladder surface: reachability, .method, citation firing.

v0.13.0 release audit: the canonical CCSD(T) / triples / AutoCI
``citype=`` selectors that landed in [Unreleased] must run end-to-end
through run_job AND their citation routes must fire into all three
user-facing destinations: the ``.out`` ``## References`` block, the
``.bibtex`` sidecar, and the ``.system`` manifest (full citation view).

This file also pins a release-gating regression found during that audit:
every post-HF result *proxy* (``_CCSDAugmented`` / ``_MP2Augmented`` /
``_DLPNO*`` / ``_OVGFAugmented`` / ``_DispersionAugmented``) forwarded an
unknown ``.method`` to the wrapped C++ ``RHFResult``, which has no such
attribute: so ``run_job(method="ccsd").method`` raised AttributeError,
and the default ``citations=True`` path crashed intermittently
(hash-order-dependent) on these results.  The proxies now carry their
own ``.method`` label.

cc-pVDZ is used because the DF-CCSD path needs an RI-aux basis (STO-3G
has no registered default).
"""

from __future__ import annotations

import re

import pytest
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job

H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])


def _run(tmp_path, **kw):
    return run_job(
        H2, basis="cc-pvdz", output=tmp_path / "j", citations=True,
        write_molden_file=False, write_xyz_file=False,
        write_population_file=False, **kw,
    )


def _bibtex_keys(tmp_path):
    f = tmp_path / "j.bibtex"
    return set(re.findall(r"@\w+\{([^,]+),", f.read_text())) if f.exists() else set()


def _system_keys(tmp_path):
    # .system manifest carries the full citation view as [[citations.entries]]
    # with `key = "..."` (every entry, regardless of the print flag).
    f = tmp_path / "j.system"
    return set(re.findall(r'key\s*=\s*"([^"]+)"', f.read_text())) if f.exists() else set()


def _out_references_block(tmp_path):
    # The human-readable '## References' block in the .out log renders each
    # cited work by author/title (not by bibtex key).
    f = tmp_path / "j.out"
    if not f.exists():
        return ""
    lines = f.read_text().splitlines()
    i = next((k for k, l in enumerate(lines)
              if l.strip().startswith("## References")), None)
    return "\n".join(lines[i:]) if i is not None else ""


class TestCCLadderReachable:
    @pytest.mark.parametrize(
        "kw,want_method,want_cite,want_author",
        [
            (dict(method="ccsd"), "ccsd", "purvis_bartlett_ccsd_1982", "Purvis"),
            (dict(method="ccsd(t)"), "ccsd(t)", "raghavachari_ccsdt_1989",
             "Raghavachari"),
            # triples selector overrides the method spelling for recording
            (dict(method="ccsd", triples="(t)"), "ccsd(t)",
             "raghavachari_ccsdt_1989", "Raghavachari"),
            (dict(method="ccsd", triples="A-CCSD(T)"), "a-ccsd(t)",
             "kucharski_bartlett_accsdt_1998", "Kucharski"),
            (dict(method="ccsd(t)", triples="none"), "ccsd",
             "purvis_bartlett_ccsd_1982", "Purvis"),
            # AutoCI citype selector
            (dict(method="ci", citype="cisd"), "cisd", "pople_cisd_1977",
             "Pople"),
            (dict(method="ci", citype="ccsd"), "ccsd",
             "purvis_bartlett_ccsd_1982", "Purvis"),
            (dict(method="ci", citype="ccsd(t)"), "ccsd(t)",
             "raghavachari_ccsdt_1989", "Raghavachari"),
            (dict(method="ci", citype="ccsdt"), "ccsdt",
             "noga_bartlett_ccsdt_1987", "Noga"),
            (dict(method="ci", citype="cc3"), "cc3",
             "koch_cc3_1997", "Koch"),
            (dict(method="bccd"), "bccd", "handy_bccd_1989", "Handy"),
            (dict(method="bccd(t)"), "bccd(t)", "raghavachari_ccsdt_1989",
             "Raghavachari"),
            (dict(method="ci", citype="bccd"), "bccd", "handy_bccd_1989",
             "Handy"),
            (dict(method="ci", citype="qcisd"), "qcisd",
             "pople_qcisd_1987", "Pople"),
            (dict(method="ci", citype="qcisd(t)"), "qcisd(t)",
             "raghavachari_ccsdt_1989", "Raghavachari"),
        ],
    )
    def test_runs_records_and_cites(
        self, tmp_path, kw, want_method, want_cite, want_author
    ):
        res = _run(tmp_path, **kw)
        assert res.energy < 0.0
        # CC proxies report the bare label ("ccsd"); the CISD solver path
        # reports the solver form ("cisd(ndet=4)").  Accept both.
        assert res.method == want_method or res.method.startswith(want_method + "(")
        # The citation route must fire into ALL THREE user-facing
        # destinations, not just .bibtex (the v0.13.0 freeze contract):
        assert want_cite in _bibtex_keys(tmp_path)          # .bibtex sidecar
        assert want_cite in _system_keys(tmp_path)          # .system manifest
        assert want_author in _out_references_block(tmp_path)  # .out block

    def test_raghavachari_only_when_triples(self, tmp_path):
        # CCSD must NOT pull in the (T) citation; CCSD(T) must.
        _run(tmp_path, method="ccsd")
        assert "raghavachari_ccsdt_1989" not in _bibtex_keys(tmp_path)

    def test_qcisd_t_cites_qcisd_and_triples(self, tmp_path):
        _run(tmp_path, method="qcisd(t)")
        keys = _bibtex_keys(tmp_path)
        assert "pople_qcisd_1987" in keys
        assert "raghavachari_ccsdt_1989" in keys


class TestProxyMethodAttribute:
    """The post-HF proxies expose .method (the release-gating fix)."""

    @pytest.mark.parametrize(
        "kw,want",
        [
            (dict(method="ccsd"), "ccsd"),
            (dict(method="ccsd(t)"), "ccsd(t)"),
            (dict(method="bccd"), "bccd"),
            (dict(method="bccd(t)"), "bccd(t)"),
            (dict(method="qcisd"), "qcisd"),
            (dict(method="qcisd(t)"), "qcisd(t)"),
            (dict(method="mp2"), "mp2"),
            (dict(method="scs-mp2"), "scs-mp2"),
            (dict(method="sos-mp2"), "sos-mp2"),
            (dict(method="dlpno-mp2"), "dlpno-mp2"),
        ],
    )
    def test_proxy_has_method(self, tmp_path, kw, want):
        # citations=False isolates the deterministic core bug: .method on
        # the returned proxy must not fall through to the SCF RHFResult.
        res = run_job(
            H2, basis="cc-pvdz", output=tmp_path / "j", citations=False,
            write_molden_file=False, write_xyz_file=False,
            write_population_file=False, **kw,
        )
        assert res.method == want
        # energy access still forwards/works (proxy stays transparent)
        assert res.energy < 0.0
        assert res.energy_total < 0.0


class TestSelectorGuards:
    """Unimplemented ladder entries raise instead of silently falling back."""

    def test_ci_requires_citype(self, tmp_path):
        with pytest.raises(ValueError, match="citype"):
            _run(tmp_path, method="ci")

    # A-CCSD(T) and CCSD[T] / CCSD+T(CCSD) are implemented and must not
    # silently fall back to ordinary CCSD(T).
    @pytest.mark.parametrize(
        "good,want",
        [("A-CCSD(T)", "a-ccsd(t)"), ("CCSD[T]", "ccsd[t]"),
         ("CCSD+T(CCSD)", "ccsd[t]")],
    )
    def test_bracket_triples_now_supported(self, tmp_path, good, want):
        res = _run(tmp_path, method="ccsd", triples=good)
        assert res.method == want

    def test_qcisd_rejects_bracket_triples(self, tmp_path):
        with pytest.raises(NotImplementedError, match="QCISD\\(T\\)"):
            _run(tmp_path, method="qcisd", triples="[t]")

    @pytest.mark.parametrize(
        "good", ["bccd", "bccd(t)", "cc2", "ccd", "lccd", "lccsd", "cepa(0)",
                 "cepa(1)", "cepa(2)", "cepa(3)", "qcisd", "qcisd(t)"]
    )
    def test_coupled_pair_citype_variants_supported(self, tmp_path, good):
        res = _run(tmp_path, method="ci", citype=good)
        assert res.method == good
