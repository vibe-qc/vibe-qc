"""``AssembledCitations.printable`` filter + TOML ``print`` parsing.

Pins the contract from CLAUDE.md § 8: each ``[entries.<key>]`` block
may carry a ``print = true/false`` field. When ``false``, the entry
is recorded in the ``.system`` manifest as internal provenance only
and is hidden from the user-facing reference list (``.out`` ``##
References`` block, ``.bibtex``, ``.references``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vibeqc.output.citations.registry import (
    AssembledCitations,
    Citation,
    _entry_from_toml,
    load_database,
)


def _make_citation(key: str, *, print_flag: bool = True) -> Citation:
    return Citation(
        key=key,
        kind="article",
        bibtex_key=key,
        authors=("Doe, J.",),
        title=f"Paper {key}",
        **{"print": print_flag},
    )


def test_printable_filters_print_visible_entries() -> None:
    c1 = _make_citation("a", print_flag=True)
    c2 = _make_citation("b", print_flag=False)
    c3 = _make_citation("c", print_flag=True)
    assembled = AssembledCitations(citations=(c1, c2, c3))
    assert assembled.printable == (c1, c3)
    # Full ``.citations`` view still has all three (manifest path).
    assert assembled.citations == (c1, c2, c3)


def test_printable_default_includes_all_when_no_flag_set() -> None:
    c1 = _make_citation("a")
    c2 = _make_citation("b")
    assembled = AssembledCitations(citations=(c1, c2))
    assert assembled.printable == (c1, c2)


def test_printable_empty_returns_empty_tuple() -> None:
    assembled = AssembledCitations(citations=())
    assert assembled.printable == ()


def test_entry_from_toml_parses_print_false() -> None:
    blob = {
        "kind": "misc",
        "bibtex_key": "internal_2024",
        "authors": ["Internal, A."],
        "title": "Internal helper",
        "print": False,
    }
    c = _entry_from_toml("internal", blob)
    assert getattr(c, "print") is False


def test_entry_from_toml_parses_print_true_explicit() -> None:
    blob = {
        "kind": "article",
        "bibtex_key": "x_2024",
        "authors": ["X, Y."],
        "title": "Paper",
        "print": True,
    }
    c = _entry_from_toml("x", blob)
    assert getattr(c, "print") is True


def test_entry_from_toml_print_defaults_to_true_when_absent() -> None:
    blob = {
        "kind": "article",
        "bibtex_key": "y_2024",
        "authors": ["Y, Z."],
        "title": "Paper without explicit print",
    }
    c = _entry_from_toml("y", blob)
    assert getattr(c, "print") is True


def test_database_load_honours_print_false_in_toml(tmp_path: Path) -> None:
    db_path = tmp_path / "with_hidden.toml"
    db_path.write_text(
        'schema_version = "1"\n'
        "[entries.user_visible]\n"
        'kind = "article"\n'
        'bibtex_key = "user_visible_2024"\n'
        'authors = ["A, B"]\n'
        'title = "Visible"\n'
        "\n"
        "[entries.hidden]\n"
        'kind = "misc"\n'
        'bibtex_key = "hidden_2024"\n'
        'authors = ["C, D"]\n'
        'title = "Hidden"\n'
        "print = false\n"
        "\n"
        "[routes.software]\n"
        'always = ["user_visible", "hidden"]\n',
        encoding="utf-8",
    )
    db = load_database(db_path)
    assembled = db.assemble()
    keys_all = [c.key for c in assembled.citations]
    keys_printable = [c.key for c in assembled.printable]
    assert "user_visible" in keys_all
    assert "hidden" in keys_all
    assert "user_visible" in keys_printable
    assert "hidden" not in keys_printable


# ---------------------------------------------------------------------------
# write_references_block -- the citation printer emits through the channel
# ---------------------------------------------------------------------------


def test_write_references_block_emits_through_the_channel():
    import io

    from vibeqc.output import Level, OutputChannel
    from vibeqc.output.citations import (
        format_references_block,
        load_default_database,
        write_references_block,
    )

    refs = load_default_database().assemble(method="rhf", basis="sto-3g")
    block = format_references_block(refs)

    # Byte-identical to the hand-spliced "\n" + block + "\n" the runners used.
    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        assert write_references_block(refs) is True
    assert buf.getvalue() == "\n" + block + "\n"

    # A pre-built block string is accepted too (the runners pass this).
    buf2 = io.StringIO()
    with OutputChannel.to_stream(buf2):
        write_references_block(block=block, leading="", trailing="")
    assert buf2.getvalue() == block

    # Respects the channel level like every other write.
    buf3 = io.StringIO()
    with OutputChannel.to_stream(buf3, level=Level.QUIET):
        write_references_block(refs, level=Level.STANDARD)
    assert buf3.getvalue() == ""  # STANDARD suppressed on a QUIET channel


def test_write_references_block_returns_false_with_no_citations():
    from vibeqc.output.citations import write_references_block

    assert write_references_block(None) is False
    assert write_references_block(block="") is False
