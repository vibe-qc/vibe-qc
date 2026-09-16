"""Golden-file snapshots that freeze the ``.out`` *format*.

The maintainer's decision on the output logger (2026-07-10) was "one clean
break, then freeze": pick the format once, then lock it so it cannot drift
again silently. This module is the lock. Every writer migration and every
future format edit must either keep these snapshots byte-identical or
regenerate them deliberately -- there is no third, silent path.

What is frozen is the **scaffold**, not the physics. Before comparing, the
``.out`` is canonicalized: run-to-run volatile fields (wall timings,
available memory, the banner's branch/SHA/dirty flag, absolute paths) are
scrubbed, and every numeric value has its digits zeroed while its shape is
kept. So ``-76.0261400000 Ha`` becomes ``-00.0000000000 Ha``. That means:

* a label / separator / alignment / unit change  -> caught (text differs);
* a field-width change (``16.10f`` -> ``18.10f``) -> caught (length differs);
* a precision change (``.10f`` -> ``.6f``)        -> caught (zero count differs);
* a legitimate energy change (e.g. an exxdiv convention flip) -> ignored,
  because the digits were zeroed. The snapshot is a format gate, not a
  numerical regression gate -- those live in the per-method runner tests.

Zeroing the digits also makes the golden files independent of the compiled
core, so a stale ``_vibeqc_core`` cannot false-fail a format snapshot.

One residual coupling to physics: the *number* of SCF iteration rows is a
convergence property, not a format one, so an SCF-algorithm change that
alters the iteration count on these fixtures would trip the snapshot. The
fixtures are chosen to be maximally stable (tiny molecular jobs -- no
periodic exxdiv sensitivity, no accelerator borderline cases), so in
practice that only happens on a deliberate convergence change, which is a
reasonable thing to require a reviewed golden regen for.

Regenerate after a deliberate format change::

    VIBEQC_REGEN_GOLDEN=1 pytest tests/test_out_format_snapshot.py

then review the diff in ``tests/golden/*.out`` before committing it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import Atom, Molecule, run_job


_GOLDEN_DIR = Path(__file__).parent / "golden"

# Order matters: scrub the volatile *lines* first (they contain numbers we
# do not want the digit-zeroing to touch differently run to run), then zero
# the digits everywhere else.
_VOLATILE = [
    # banner: "... (branch @ 0a1b2c3, dirty) ..."  -> "... (<GIT>) ..."
    (re.compile(r"\([^()]*@ [0-9a-f]{7}[^()]*\)"), "(<GIT>)"),
    (re.compile(r"Available on this machine: [\d.]+ GB"),
     "Available on this machine: <MEM> GB"),
    (re.compile(r"/[^\s]*?/(tmp|Users|home)/[^\s]*"), "<PATH>"),
    # thread counts track the machine / OMP_NUM_THREADS cap, and the digit
    # zeroing preserves width -- "Threads: 8" vs "Threads: 18" would differ.
    # Scrub to a fixed placeholder like the timings below.
    (re.compile(r"(Threads:\s*)\d+"), r"\1<NT>"),
    # The footer pluralises: "Used 1 OpenMP thread." vs "Used 8 OpenMP
    # threads.". A 1-thread run (conftest pins OMP_NUM_THREADS=1 on macOS)
    # must canonicalise identically to a multi-thread one, so the word is
    # normalised along with the count.
    (re.compile(r"Used\s+\d+\s+OpenMP threads?\."),
     "Used <NT> OpenMP threads."),
    # The SCF state fingerprint is a hex digest of the converged numerical
    # state: digit-zeroing leaves its letters, which track the numbers, not
    # the format. Scrub the digest, keep the label.
    (re.compile(r"(state fingerprint:\s*)[0-9a-f]{16,}"), r"\1<FPR>"),
    # any wall-clock the writers emit
    (re.compile(r"\(\s*\d+\.\d+\s*ms\)"), "(<T>ms)"),
    (re.compile(r"\b\d+\.\d+\s*s\b"), "<T>s"),
    # Exact-density phase scalars use the document's header-unit policy.
    # Their elapsed time has no suffix; normalize only this complete row,
    # so concatenated following output cannot be hidden by the scrubber.
    (re.compile(r"^(\s*Wall time)\s+\d+\.\d{3}$"), r"\1 <T>"),
    (re.compile(
        r"(Job total|SCF total|SCF stability analysis"
        r"|SCF avg\. per iteration)\b.*"), r"\1 <T>"),
]

# A signed decimal / scientific number. Zero the digits, keep the mantissa
# sign, the point and the overall length so width/precision changes still
# show. The exponent sign is normalised with the digits (see _zero_digits):
# once the exponent reads "00" its sign describes the original magnitude,
# not the format.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# A genuinely-zero-magnitude value (all digits zero). Its SIGN is
# numerically unstable: a charge sum or residual that rounds to zero can
# render "-0.000000" in one build and "0.000000" in the next as the last
# bit flips -- not a format change. Normalise the leading minus to a space
# (preserving field width) BEFORE zeroing the other digits, so signed and
# unsigned zeros canonicalise identically. Matched on the original text so
# a clearly-negative value like "-76.02" (real sign, format-relevant) is
# untouched.
_SIGNED_ZERO = re.compile(
    r"(?<![A-Za-z0-9_.])-(0+(?:\.0+)?(?:[eE][-+]?[0-9]+)?)(?![0-9])"
)

# The version banner is a box drawn with these glyphs. It comes from
# banner.py (provenance), not the .out writers this module freezes, and its
# line lengths encode the pre-scrub git-string / version length -- so its
# trailing padding is not stable run to run. Collapse the whole box to one
# placeholder rather than trying to freeze a moving target.
_BOX = frozenset("╔╗╚╝║═")


def _zero_digits(match: re.Match) -> str:
    zeroed = re.sub(r"\d", "0", match.group(0))
    # GitLab #112: normalise the EXPONENT sign too, for the same reason
    # _SIGNED_ZERO normalises the mantissa sign. Once the digits are zeroed
    # the exponent reads "00" whatever it was, so its sign no longer carries
    # any format information -- it only records whether the original
    # magnitude was >= 1, which is a *number* property this module is
    # documented to ignore. Leaving it in meant a residual that is exactly
    # 0.0 on one build ("0.000e+00") and subnormal-tiny on the next
    # ("4.710e-16" -> "0.000e-00") failed a gate that is supposed to catch
    # format changes only. Width is untouched, since the sign character
    # stays, so a precision or field-width change still shows.
    return re.sub(r"([eE])[-+](?=\d)", r"\1+", zeroed)


def canonicalize(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        # The references block ("## References" to end of file) is citation
        # *content* -- owned by the citation database (CLAUDE.md Sec. 8) and
        # exercised by tests/test_citations.py. It changes whenever a chat
        # adds a reference, which is not a change to the SCF .out *format*
        # this module freezes. Collapse it to one marker so the snapshot
        # still verifies the block is present (and would catch it
        # vanishing) without pinning which papers are in it.
        if line.startswith("## References"):
            lines.append("<REFERENCES>")
            break
        if any(ch in _BOX for ch in line):
            if not lines or lines[-1] != "<BANNER>":
                lines.append("<BANNER>")
            continue
        for pat, repl in _VOLATILE:
            line = pat.sub(repl, line)
        line = _SIGNED_ZERO.sub(r" \1", line)
        line = _NUMBER.sub(_zero_digits, line)
        lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def _compare_or_regen(name: str, out_text: str) -> None:
    golden = _GOLDEN_DIR / f"{name}.out"
    canon = canonicalize(out_text)
    if os.environ.get("VIBEQC_REGEN_GOLDEN", "").strip().lower() in ("1", "true", "yes"):
        _GOLDEN_DIR.mkdir(exist_ok=True)
        golden.write_text(canon)
        pytest.skip(f"regenerated golden {golden.name}")
    if not golden.is_file():
        pytest.fail(
            f"missing golden {golden}. Generate it with "
            f"VIBEQC_REGEN_GOLDEN=1 pytest {Path(__file__).name} and commit it."
        )
    expected = golden.read_text()
    if canon != expected:
        # A focused diff so the reviewer sees exactly which scaffold line moved.
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(), canon.splitlines(),
                fromfile=f"golden/{name}.out", tofile="produced", lineterm="",
            )
        )
        pytest.fail(
            f"{name}: .out format changed vs the committed snapshot.\n"
            f"If deliberate, regenerate with VIBEQC_REGEN_GOLDEN=1 and commit "
            f"tests/golden/{name}.out. Diff (digits are zeroed -- a diff here "
            f"is a *format* change, not a number change):\n{diff}"
        )


def test_phase_timing_canonicalization_preserves_record_boundaries():
    expected = "  Wall time <T>\n"
    for value in ("0.003", "12.345", "120.001"):
        assert canonicalize(f"  Wall time    {value}\n") == expected
    assert canonicalize("  Wall time    0.003  iter energy\n") != expected
    assert canonicalize("  Wall time    0.0030\n") != expected


def _h2o() -> Molecule:
    return Molecule([
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])


def test_out_format_molecular_rhf(tmp_path):
    stem = tmp_path / "mol_rhf"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem)
    _compare_or_regen("mol_rhf", stem.with_suffix(".out").read_text())


def test_out_format_molecular_rks_pbe(tmp_path):
    stem = tmp_path / "mol_rks"
    run_job(_h2o(), basis="sto-3g", method="rks", functional="pbe", output=stem)
    _compare_or_regen("mol_rks", stem.with_suffix(".out").read_text())


def _oh_doublet() -> Molecule:
    # OH radical (9 electrons, doublet): the smallest stable open-shell case
    # that exercises the UHF-specific .out format -- the spin-resolved
    # alpha/beta orbital tables (HOAMO/LUAMO, HOBMO/LUBMO markers) and the
    # spin-density population column.
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.83])],
        multiplicity=2,
    )


def test_out_format_molecular_uhf(tmp_path):
    stem = tmp_path / "mol_uhf"
    run_job(_oh_doublet(), basis="sto-3g", method="uhf", output=stem)
    _compare_or_regen("mol_uhf", stem.with_suffix(".out").read_text())


def test_out_format_double_hybrid(tmp_path):
    stem = tmp_path / "dh"
    vq.run_double_hybrid(_h2o(), "cc-pvdz", "b2plyp", output=stem)
    _compare_or_regen("double_hybrid", stem.with_suffix(".out").read_text())


def test_out_format_molecular_hessian(tmp_path):
    # Freezes the vibrational-frequency + RRHO-thermochemistry .out blocks
    # (hessian=True): the Mode / Freq(cm-1) / IR(km/mol) table and the
    # thermochemistry lines (T in K, ZPE / thermal corrections). These blocks
    # have no other golden, so this is the safety net for the `frequency`
    # (cm-1) and `temperature` (K) FormatPolicy kinds the runner renders them
    # through. Only possible now that the "## References" footer is emitted
    # last: the canonicalizer truncates at "## References", so with the old
    # references-before-frequencies ordering these blocks fell past the cut.
    stem = tmp_path / "mol_hess"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem, hessian=True)
    _compare_or_regen("mol_hessian", stem.with_suffix(".out").read_text())


def test_out_format_tddft_excited_states(tmp_path):
    # Freezes the TD-DFT excited-states .out table (now the document-layer
    # Table primitive: State / E(eV) / λ(nm) / f_osc / dominant, rule sized
    # to content). tddft_spectrum=False on purpose -- the Gaussian
    # stick-spectrum bar chart below the table is physics-coupled (bar length
    # tracks intensity), which is not a *format* the snapshot should pin.
    stem = tmp_path / "mol_tddft"
    run_job(
        _h2o(), basis="sto-3g", method="rhf", output=stem,
        tddft=True, tddft_n_states=3, tddft_spectrum=False,
    )
    _compare_or_regen("mol_tddft", stem.with_suffix(".out").read_text())


def _h2_periodic():
    length = 12.0
    c = length / 2.0
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * length,
        [Atom(1, [c, c, c - 0.7]), Atom(1, [c, c, c + 0.7])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def test_out_format_periodic_rhf(tmp_path):
    # Freezes the *unified* periodic .out: since MS2 the periodic SCF trace,
    # convergence line, and energy-component breakdown come from the same
    # format_scf_trace as the molecular path, so this golden and the
    # molecular ones share a scaffold. A drift on either side trips.
    sysp, basis = _h2_periodic()
    stem = tmp_path / "per_rhf"
    vq.run_periodic_job(
        sysp, basis, method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
        max_iter=20, output=stem, output_qvf=False, write_density=False,
        citations=False, progress=False,
    )
    _compare_or_regen("periodic_rhf", stem.with_suffix(".out").read_text())


def test_out_format_periodic_geomopt(tmp_path):
    # Freezes the periodic geometry-optimisation .out. Since MS3 the
    # relaxation trace (relax_atoms' "Atomic relaxation: ..." summary) and the
    # "Optimized geometry" block both emit through the ambient .out channel
    # under a "Geometry optimization" header. Before MS3 the trace was
    # stdout-only, so a batch .out record showed only the final geometry with
    # no trace of how it got there.
    # The compact cell keeps both executed BIPOLE image sets periodic under
    # the public Gamma preflight. Fold diagnostics have their own focused
    # coverage; this snapshot freezes the successful optimization scaffold.
    sysp, basis = _h2_periodic()
    stem = tmp_path / "per_opt"
    vq.run_periodic_job(
        sysp, basis, method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
        max_iter=20, output=stem, output_qvf=False, write_density=False,
        citations=False, progress=False, optimize=True, optimize_max_iter=2,
    )
    _compare_or_regen("periodic_geomopt", stem.with_suffix(".out").read_text())


# --------------------------------------------------------------------
# The canonicalizer's own behaviour -- these must not depend on a job run,
# so the freeze mechanism itself is pinned.
# --------------------------------------------------------------------


def test_canonicalize_zeroes_digits_but_keeps_shape():
    assert canonicalize("  Total energy:  -76.0261400000 Ha\n") == \
        "  Total energy:  -00.0000000000 Ha\n"


def test_canonicalize_catches_a_width_change():
    # 16.10f vs 18.10f -> different length -> different canonical form.
    a = canonicalize(f"  E = {-76.02614:16.10f}\n")
    b = canonicalize(f"  E = {-76.02614:18.10f}\n")
    assert a != b


def test_canonicalize_catches_a_precision_change():
    a = canonicalize(f"  E = {-76.02614:.10f}\n")
    b = canonicalize(f"  E = {-76.02614:.6f}\n")
    assert a != b


def test_canonicalize_ignores_a_pure_value_change():
    # Same format, different physics -> identical canonical form.
    a = canonicalize(f"  E = {-76.02614:18.10f} Ha\n")
    b = canonicalize(f"  E = {-80.99999:18.10f} Ha\n")
    assert a == b


def test_canonicalize_catches_a_label_change():
    assert canonicalize("  Total energy: -1.0\n") != \
        canonicalize("  E_total: -0.0\n")


def test_canonicalize_scrubs_thread_counts_width_independently():
    # Thread counts follow the machine / OMP_NUM_THREADS, and digit zeroing
    # keeps width -- an 18-thread box and a 2-thread CI lane must canonicalize
    # identically or the golden pins the generating machine's core count.
    wide = canonicalize(
        "  Threads: 18  (OpenMP shared-memory parallelism)\n"
        "  Used 18 OpenMP threads.\n"
    )
    narrow = canonicalize(
        "  Threads: 2  (OpenMP shared-memory parallelism)\n"
        "  Used 2 OpenMP threads.\n"
    )
    assert wide == narrow
    assert "<NT>" in wide


def test_canonicalize_scrubs_singular_thread_footer():
    # A 1-thread run prints "Used 1 OpenMP thread." (singular). It must
    # canonicalize identically to the plural form, or goldens pin whether
    # the generating machine ran the suite single-threaded (conftest pins
    # OMP_NUM_THREADS=1 on macOS; other platforms run wide).
    single = canonicalize("  Used 1 OpenMP thread.\n")
    plural = canonicalize("  Used 8 OpenMP threads.\n")
    assert single == plural
    assert "<NT>" in single


def test_canonicalize_scrubs_state_fingerprint():
    # The state fingerprint is a hex digest of the converged numerical
    # state; its hex letters survive digit-zeroing but track the numbers.
    # Two different digests must canonicalize identically.
    a = canonicalize("  state fingerprint: " + "ab12" * 16 + "\n")
    b = canonicalize("  state fingerprint: " + "9f3e" * 16 + "\n")
    assert a == b
    assert "<FPR>" in a


def test_canonicalize_collapses_the_whole_banner_box():
    # The banner is provenance from banner.py, not the .out format; its box
    # width moves with version/git, so the whole box collapses to one line.
    banner = (
        "╔══════╗\n"
        '║ dev 0.15.34 "Cheetah" (feature/x @ 0a1b2c3, dirty) -- QC ║\n'
        "║ linked: libint 2.11.0 . libxc 7.0.0                      ║\n"
        "╚══════╝\n"
        "\n  Job: RHF  basis=sto-3g\n"
    )
    out = canonicalize(banner)
    assert "0a1b2c3" not in out
    assert "libint" not in out
    assert out.count("<BANNER>") == 1  # a contiguous box is one marker
    assert "Job: RHF  basis=sto-0g" in out  # content below the box survives


def test_canonicalize_normalizes_signed_zero():
    # A near-zero value that flips between -0.000000 and 0.000000 between
    # builds is numerical noise, not a format change; both must canonicalize
    # the same (field width preserved).
    assert canonicalize("  sum  -0.000000\n") == canonicalize("  sum   0.000000\n")
    assert canonicalize("dE -0.0e+00\n") == canonicalize("dE  0.0e+00\n")


def test_canonicalize_normalizes_the_exponent_sign_of_a_zeroed_number():
    # GitLab #112: the final ||[F,DS]|| of the H2/Gamma fixture is exactly
    # 0.0 on some builds and subnormal-tiny on others. After digit zeroing
    # those render "0.000e+00" and "0.000e-00", which differ only in a sign
    # that no longer describes anything -- the exponent digits are "00"
    # either way. Both spellings must canonicalize the same, or a 1e-16
    # numerical difference fails a format-only gate.
    assert canonicalize("  r 0.000e+00\n") == canonicalize("  r 4.710e-16\n")
    assert canonicalize("  r 1.000e+05\n") == canonicalize("  r 4.710e-16\n")


def test_canonicalize_still_catches_a_precision_change_in_the_exponent_form():
    # The normalisation above must not blunt the gate: mantissa precision
    # and field width are still format, and still caught.
    assert canonicalize("  r 1.234e-05\n") != canonicalize("  r 1.23e-05\n")
    assert canonicalize("  r 1.234e-05\n") != canonicalize("  r  1.234e-05\n")
    # An exponent with no sign at all is a different format, not noise.
    assert canonicalize("  r 1.234e05\n") != canonicalize("  r 1.234e-05\n")


def test_canonicalize_keeps_a_real_negative_sign():
    # A genuinely-negative value's sign is format-relevant (e.g. %+.3e vs
    # %.3e) and must survive.
    assert canonicalize("E = -76.02\n") != canonicalize("E =  76.02\n")
