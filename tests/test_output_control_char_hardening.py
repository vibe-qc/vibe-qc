"""Regression guard: dangerous Unicode must never reach a user-facing file raw.

vibe-qc's output writers interpolate user-controlled free text (the
``run_job(output=...)`` basename, ``basis`` / ``functional`` / ``method``
names, SCF exception messages) into file headers and the ``.system`` /
``.dump`` manifests. A bidirectional-format control such as U+202E
RIGHT-TO-LEFT OVERRIDE written raw silently reorders how the file renders
in terminals, editors, ``grep`` / diff viewers, and log-parsing agents —
the "Trojan Source" display-deception class (CVE-2021-42574). This module
pins that the reachable sinks (the Gaussian-cube header, the ``.system``
manifest, the crash ``.dump``, and the molden ``[Title]``) neutralise the
dangerous class, and that the shared helper does so without mangling
legitimate text.

The dangerous characters are constructed via ``chr()`` and asserted on,
never embedded as literals or printed raw — embedding one would corrupt
this very source file and any tool that reads it, which is the whole
point of the guard.

See :mod:`vibeqc.output._text_safety`.
"""

from __future__ import annotations

import io
import tomllib
from types import SimpleNamespace

import pytest

from vibeqc.output._text_safety import (
    is_unsafe_output_char,
    scrub_output_text,
    toml_escape_str,
)

# A representative char from every guarded range (built, never literal).
DANGER = {
    0x00: "NUL",
    0x07: "BEL",
    0x1B: "ESC",
    0x7F: "DEL",
    0x80: "C1-PAD",
    0x9F: "C1-APC",
    0x200B: "ZERO WIDTH SPACE",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x2060: "WORD JOINER",
    0x2064: "INVISIBLE PLUS",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
    0xFEFF: "BOM / ZWNBSP",
}
DANGER_CHARS = [chr(cp) for cp in DANGER]
RLO = chr(0x202E)  # the headline case
ZWSP = chr(0x200B)

# Legitimate text that must pass through untouched — including non-ASCII
# letters and typographic punctuation vibe-qc already emits (Greek, µ,
# em-dash, accents). Over-scrubbing these would corrupt real output.
SAFE_TEXT = "café Ω ω µ — H2O/def2-SVP r=1.4\tcol"


def _has_unsafe(text: str) -> list[str]:
    """Codepoints from the dangerous class present raw in ``text``."""
    return [f"U+{ord(c):04X}" for c in text if is_unsafe_output_char(c)]


# --------------------------------------------------------------------------- #
# Helper: the dangerous-character predicate                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cp", list(DANGER), ids=[DANGER[c] for c in DANGER])
def test_predicate_flags_every_danger_char(cp: int) -> None:
    assert is_unsafe_output_char(chr(cp)) is True


@pytest.mark.parametrize("ch", list(SAFE_TEXT) + ["\t", "\n", "\r"])
def test_predicate_allows_legitimate_text(ch: str) -> None:
    # Tab/newline/CR are benign whitespace (the TOML emitter handles them
    # structurally); printable + legitimate non-ASCII must pass.
    assert is_unsafe_output_char(ch) is False


# --------------------------------------------------------------------------- #
# Helper: scrub_output_text (plain-text sinks)                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ch", DANGER_CHARS, ids=list(DANGER.values()))
def test_scrub_neutralises_every_danger_char(ch: str) -> None:
    out = scrub_output_text(f"job{ch}name")
    assert _has_unsafe(out) == [], out
    assert ch not in out
    assert out.isascii()  # collapses to inspectable ASCII tokens


def test_scrub_preserves_legitimate_text() -> None:
    # Only a literal embedded newline is rewritten (it would line-inject a
    # single-line header); the rest of the legitimate text is untouched.
    assert scrub_output_text(SAFE_TEXT) == SAFE_TEXT
    assert "\n" not in scrub_output_text("line1\nline2")


# --------------------------------------------------------------------------- #
# Helper: toml_escape_str (manifest / dump sinks) — safe AND lossless         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ch", DANGER_CHARS, ids=list(DANGER.values()))
def test_toml_escape_is_safe_and_round_trips(ch: str) -> None:
    s = f"basis{ch}set"
    quoted = toml_escape_str(s)
    # No raw dangerous byte survives into the emitted TOML...
    assert _has_unsafe(quoted) == [], quoted
    assert ch not in quoted
    # ...yet it is lossless: stdlib tomllib decodes the escape back.
    assert tomllib.loads(f"v = {quoted}")["v"] == s


def test_toml_escape_preserves_legitimate_text() -> None:
    quoted = toml_escape_str(SAFE_TEXT)
    assert tomllib.loads(f"v = {quoted}")["v"] == SAFE_TEXT
    assert "é" in quoted and "ω" in quoted  # legit non-ASCII not escaped


def test_toml_escape_still_handles_structural_chars() -> None:
    # Pre-existing behaviour the manifests rely on must be preserved.
    assert tomllib.loads(f'v = {toml_escape_str(chr(92))}')["v"] == chr(92)  # backslash
    assert tomllib.loads(f'v = {toml_escape_str(chr(34))}')["v"] == chr(34)  # quote
    assert tomllib.loads(f'v = {toml_escape_str("a\nb")}')["v"] == "a\nb"


# --------------------------------------------------------------------------- #
# The three confirmed-reachable writer sinks, end to end                      #
# --------------------------------------------------------------------------- #

def test_cube_header_scrubs_tainted_title() -> None:
    from vibeqc.cube import _write_cube_header

    buf = io.StringIO()
    mol = SimpleNamespace(atoms=[])
    grid = SimpleNamespace(origin=(0.0, 0.0, 0.0), shape=(2, 2, 2),
                           spacing=(0.1, 0.1, 0.1))
    _write_cube_header(buf, title=f"vibe-qc density / out{RLO}put",
                       comment="rho(r)", mol=mol, grid=grid)
    assert _has_unsafe(buf.getvalue()) == []
    assert RLO not in buf.getvalue()


def test_system_manifest_scrubs_tainted_basename(tmp_path) -> None:
    from vibeqc.output.formats.system_info import write_system_manifest

    p = write_system_manifest(tmp_path / "x", 0.01,
                              f"job{RLO}name{ZWSP}", record_hostname=False)
    text = p.read_text(encoding="utf-8")
    assert _has_unsafe(text) == []
    assert RLO not in text
    tomllib.loads(text)  # still valid, machine-readable TOML


def test_crash_dump_scrubs_tainted_exception(tmp_path) -> None:
    from vibeqc.output.formats.crash_dump import dump_on_failure, load_dump

    exc = ValueError(f"linear dependence in basis {RLO}evil")
    p = dump_on_failure(tmp_path / "job", exc, {},
                        phase=f"scf_iter_5{ZWSP}", options=None, molecule=None)
    text = p.read_text(encoding="utf-8")
    assert _has_unsafe(text) == []
    assert RLO not in text
    # Round-trips, and the (now-escaped) exception text decodes back intact.
    parsed = load_dump(p)
    assert RLO in parsed["crash"]["exception"]


def test_molden_header_scrubs_tainted_title() -> None:
    # The molden [Title] is user-controlled: run_job / run_periodic_job pass
    # the output basename into it. The writer was widened from ASCII to
    # UTF-8 so a legitimate accented basename no longer raises
    # UnicodeEncodeError mid-write and aborts the artifact — scrubbing keeps
    # that widening from leaking a bidi / control byte into the file.
    from vibeqc.output.formats.molden import _write_header

    buf = io.StringIO()
    _write_header(buf, title=f"café{RLO}evil{ZWSP}")
    out = buf.getvalue()
    assert _has_unsafe(out) == []
    assert RLO not in out and ZWSP not in out
    assert "café" in out  # legitimate accent passes through untouched


def test_molden_file_accepts_accented_title_as_utf8(tmp_path) -> None:
    # End to end: a non-ASCII output basename — the crash trigger, where the
    # ascii-encoded molden file raised UnicodeEncodeError on the accented
    # [Title] and aborted the whole job after the SCF had converged — now
    # writes a valid UTF-8 file with the bidi override neutralised.
    from vibeqc import Atom, BasisSet, Molecule, run_rhf
    from vibeqc.output.formats.molden import write_molden

    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    result = run_rhf(mol, basis)
    assert result.converged

    out = tmp_path / "café.molden"
    write_molden(out, mol, basis, result, title=f"café{RLO}molecule")

    text = out.read_bytes().decode("utf-8")  # valid UTF-8, no crash
    assert _has_unsafe(text) == []
    assert "café" in text  # accent preserved on disk
    assert RLO not in text  # bidi override scrubbed
    assert "[Molden Format]" in text and "[MO]" in text


# --------------------------------------------------------------------------- #
# Deferred plain-text writer sinks (2026-06-01 audit follow-up). molden is     #
# covered above; qvf is json.dumps-safe. POSCAR / CIF comments ARE            #
# user-reachable — run_periodic_job embeds the functional + basis name        #
# (periodic_runner.py); xyz / extended-xyz / trajectory comment params are    #
# not runner-routed today but share the class. All scrubbed via _text_safety. #
# --------------------------------------------------------------------------- #

def _fake_mol():
    return SimpleNamespace(atoms=[SimpleNamespace(Z=1, xyz=(0.0, 0.0, 0.0))])


def _fake_periodic():
    import numpy as np
    return SimpleNamespace(lattice=np.eye(3) * 5.0,
                           unit_cell=[SimpleNamespace(Z=1, xyz=(0.0, 0.0, 0.0))])


def test_xyz_comment_scrubbed() -> None:
    from vibeqc.output.formats.xyz import format_xyz
    out = format_xyz(_fake_mol(), comment=f"geom café{RLO}evil{ZWSP}")
    assert _has_unsafe(out) == []
    assert RLO not in out and ZWSP not in out
    assert "café" in out  # legitimate accent preserved


def test_extended_xyz_comment_scrubbed() -> None:
    from vibeqc.output.formats.extended_xyz import format_extended_xyz
    out = format_extended_xyz(_fake_periodic(), comment=f"cell{RLO}x")
    assert _has_unsafe(out) == []
    assert RLO not in out
    assert 'Lattice="' in out  # structural ext-XYZ tags intact


def test_poscar_comment_scrubbed() -> None:
    from vibeqc.output.formats.poscar import format_poscar
    out = format_poscar(_fake_periodic(), comment=f"slab café{RLO}evil")
    assert _has_unsafe(out) == []
    assert RLO not in out
    assert "café" in out
    assert out.splitlines()[1].strip() == "1.0"  # POSCAR scale line intact


def test_cif_comment_scrubbed() -> None:
    from vibeqc.output.formats.cif import format_cif
    out = format_cif(_fake_periodic(), comment=f"struct{RLO}x{ZWSP}")
    assert _has_unsafe(out) == []
    assert RLO not in out and ZWSP not in out
    assert "data_vibeqc" in out  # CIF body intact


def test_trajectory_comment_scrubbed() -> None:
    from vibeqc.output.formats.trajectory import _write_one_xyz_frame
    buf = io.StringIO()
    _write_one_xyz_frame(buf, _fake_mol(), comment=f"frame{RLO}3{ZWSP}")
    out = buf.getvalue()
    assert _has_unsafe(out) == []
    assert RLO not in out and ZWSP not in out


def test_safe_json_bytes_neutralises_and_round_trips() -> None:
    # The QVF zip writer serialises every JSON member via safe_json_bytes
    # (ensure_ascii=False to keep legit non-ASCII readable, but the
    # dangerous class re-escaped). Pin that helper directly. (A full QVF
    # archive integration test lives in tests/test_qvf_writer.py.)
    import json as _json
    from vibeqc.output._text_safety import escape_unsafe, safe_json_bytes

    payload = {"label": f"dens{RLO}x", f"k{ZWSP}": "v", "desc": "café ω", "n": [1, 2.5]}
    for indent in (None, 2):
        b = safe_json_bytes(payload, indent=indent)
        text = b.decode("utf-8")
        assert _has_unsafe(text) == []  # no raw danger byte in the file
        assert RLO not in text and ZWSP not in text
        d = _json.loads(b)  # still valid JSON
        assert d["desc"] == "café ω"  # legitimate non-ASCII preserved raw
        assert d["n"] == [1, 2.5]  # numbers untouched
    # escape_unsafe keeps tab / newline + accents, neutralises only the class
    assert escape_unsafe("a\tb\né ω") == "a\tb\né ω"
    assert _has_unsafe(escape_unsafe(f"x{RLO}y")) == []
