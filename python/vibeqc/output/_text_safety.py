"""Neutralise dangerous Unicode in text written to user-facing files.

vibe-qc's output writers interpolate user-controlled free text -- the
``run_job(output=...)`` basename, the ``basis`` / ``functional`` /
``method`` names, SCF exception messages -- into file headers and the
``.system`` / ``.dump`` manifests. Left raw, a bidirectional-format
control such as U+202E RIGHT-TO-LEFT OVERRIDE silently reverses how the
surrounding text renders in terminals, editors, ``grep`` / ``cat``
output, code-review and diff viewers, and log-parsing agents: the byte is
invisible but reorders everything around it. That is the "Trojan Source"
class of display deception (Boucher & Anderson, "Trojan Source: Invisible
Vulnerabilities", 2021; CVE-2021-42574 -- CWE-150 / CWE-116). Zero-width
characters, the BOM / ZWNBSP (U+FEFF), and C0/C1 control bytes are the
same family of parser-corrupting, visually-silent input.

This module is the single definition of that dangerous class plus the three
neutralisation strategies the writers use:

* :func:`toml_escape_str` -- for the hand-rolled TOML manifests
  (``.system``, crash ``.dump``): emit the dangerous char as a ``\\uXXXX``
  escape, the native and lossless TOML basic-string representation
  (``tomllib.loads`` round-trips it back to the original character).
* :func:`scrub_output_text` -- for plain-text headers with no escape
  convention (Gaussian-cube comment lines): replace the dangerous char
  with a visible ``\\uXXXX`` token so the field stays inspectable ASCII
  and cannot reorder or line-inject in a single-line header.
* :func:`safe_json_bytes` -- for the QVF zip archive's JSON members, which
  serialise ``ensure_ascii=False`` to keep legitimate non-ASCII readable:
  re-escape only the dangerous class so it never lands as a raw byte.

Pure stdlib by design -- no vibeqc imports -- so it stays import-cheap and
unit-testable in isolation.
"""

from __future__ import annotations

import json


def is_unsafe_output_char(ch: str) -> bool:
    """True for a character that must never reach a user-facing file raw.

    The dangerous class -- all non-printing, yet able to reorder, hide, or
    inject into rendered / parsed output:

    * C0 controls U+0000-U+001F, except TAB / LF / CR
    * DEL + C1 controls U+007F-U+009F
    * zero-width + bidi marks U+200B-U+200F (ZWSP, ZWNJ, ZWJ, LRM, RLM)
    * bidi embeddings / overrides U+202A-U+202E and isolates U+2066-U+2069
    * word-joiner + invisible operators U+2060-U+2064
    * BOM / ZWNBSP U+FEFF
    """
    cp = ord(ch)
    if cp < 0x20:
        return ch not in "\t\n\r"
    if cp <= 0x7E:
        return False  # printable ASCII fast path
    return (
        0x7F <= cp <= 0x9F  # DEL + C1 controls
        or 0x200B <= cp <= 0x200F  # zero-width + bidi marks
        or 0x202A <= cp <= 0x202E  # bidi embeddings / overrides
        or 0x2060 <= cp <= 0x2064  # word-joiner + invisible operators
        or 0x2066 <= cp <= 0x2069  # bidi isolates
        or cp == 0xFEFF  # BOM / ZWNBSP
    )


def _u_escape(ch: str) -> str:
    # Every dangerous codepoint is <= 0xFFFF, so the 4-hex-digit ``\uXXXX``
    # form -- a valid TOML basic-string escape and an unambiguous ASCII
    # token in plain text -- is always sufficient. Uppercase hex matches
    # the pre-existing manifest emitters' round-trip-tested output.
    return f"\\u{ord(ch):04X}"


def toml_escape_str(s: str) -> str:
    """Quote ``s`` as a TOML 1.0 basic string (returns the ``"..."`` form).

    Escapes ``\\`` and ``"``, the whitespace controls TAB / LF / CR, every
    C0 control, and -- the hardening this adds -- the full
    :func:`is_unsafe_output_char` class (bidi / zero-width / BOM / C1) as
    ``\\uXXXX``. The escaped form is lossless: ``tomllib.loads`` decodes it
    back to the original character, so machine-readability is preserved
    while the raw, terminal-corrupting byte never lands in the file.
    """
    out: list[str] = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif is_unsafe_output_char(ch):
            out.append(_u_escape(ch))
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def scrub_output_text(s: str) -> str:
    """Neutralise dangerous Unicode for a plain-text, single-line header
    field that has no escape convention (e.g. a Gaussian-cube comment
    line).

    Each :func:`is_unsafe_output_char` character -- plus any embedded
    newline, which would otherwise inject extra lines into a single-line
    header -- is replaced with a visible ``\\uXXXX`` token, so the field
    stays inspectable ASCII and structurally intact. Ordinary text passes
    through unchanged.
    """
    return "".join(
        _u_escape(ch) if (is_unsafe_output_char(ch) or ch in "\n\r") else ch
        for ch in s
    )


def escape_unsafe(s: str) -> str:
    """Replace each dangerous-class character (:func:`is_unsafe_output_char`
    -- bidi / zero-width / BOM / C1 / C0-except-TAB-LF-CR) with a visible
    ``\\uXXXX`` token, leaving everything else -- TAB / LF / CR, accents,
    Greek, ordinary text -- untouched.

    For structured formats whose serializer already quotes strings but
    emits the dangerous class raw (e.g. ``json.dumps(ensure_ascii=False)``):
    neutralise only that class, without forcing legitimate non-ASCII to
    escapes or disturbing structural whitespace.
    """
    return "".join(_u_escape(ch) if is_unsafe_output_char(ch) else ch for ch in s)


def safe_json_bytes(obj: object, *, indent: int | None = None) -> bytes:
    """Serialise ``obj`` to UTF-8 JSON bytes with the dangerous Unicode
    class neutralised in every string value and key.

    ``ensure_ascii=False`` keeps legitimate non-ASCII (accents, Greek)
    human-readable in the file; :func:`escape_unsafe` then replaces only
    the bidi / zero-width / BOM / C1 class -- which ``ensure_ascii=False``
    would otherwise write as a raw, terminal-/parser-corrupting byte -- with
    a visible ``\\uXXXX`` token. Numeric payloads pass through untouched.
    """
    def _walk(o: object) -> object:
        if isinstance(o, str):
            return escape_unsafe(o)
        if isinstance(o, dict):
            return {(escape_unsafe(k) if isinstance(k, str) else k): _walk(v)
                    for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_walk(x) for x in o]
        return o

    return json.dumps(_walk(obj), ensure_ascii=False, indent=indent).encode("utf-8")
