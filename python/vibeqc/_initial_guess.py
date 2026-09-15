"""Canonical normalisation for user-supplied ``initial_guess`` values.

Every options surface that carries an ``initial_guess`` field accepts two
spellings: the pybind11 :class:`~vibeqc._vibeqc_core.InitialGuess` enum
(``InitialGuess.SAP``) and the canonical string (``"sap"``, plus the
documented aliases).  :func:`coerce_initial_guess` maps both onto the enum
and raises on anything else, so a mis-typed or wrong-typed guess fails
loudly instead of being silently replaced by the default.

The helper lives in its own module (rather than in ``runner``) so the
pure-Python ROHF/ROKS drivers can share it without importing the molecular
runner, which imports them.
"""

from __future__ import annotations

from ._vibeqc_core import InitialGuess

__all__ = ["coerce_initial_guess"]

# Accepted spellings that are not enum member names.
_ALIASES = {
    "CORE": "HCORE",
    "HUCKEL": "HUECKEL",
    "MOREAD": "READ",
    "COREAD": "READ",
}


def coerce_initial_guess(guess: object) -> InitialGuess:
    """Normalise ``guess`` to an :class:`InitialGuess` member.

    Accepts an :class:`InitialGuess` member, any of its member names
    (case-insensitive, ``-``/space tolerated, optionally dotted such as
    ``"InitialGuess.SAP"``), or one of the aliases in :data:`_ALIASES`.

    Raises
    ------
    ValueError
        If ``guess`` is not a recognised guess. Never falls back to a
        default: a guess the caller pinned either takes effect or errors.
    """
    if isinstance(guess, InitialGuess):
        if guess in InitialGuess.__members__.values():
            return guess
        raise ValueError(f"unknown initial_guess={guess!r}")
    name = getattr(guess, "name", None)
    text = str(name if name is not None else guess).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    key = text.upper().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    try:
        return InitialGuess.__members__[key]
    except KeyError as exc:
        valid = ", ".join(InitialGuess.__members__.keys())
        raise ValueError(
            f"unknown initial_guess={guess!r} (valid: {valid}; aliases: "
            "huckel, moread, coread)"
        ) from exc
