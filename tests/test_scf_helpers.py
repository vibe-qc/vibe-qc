"""Track D — SCFAccelerator string parsing + is_scf_converged contract.

The helper consolidation in this track exposes:

* ``vibeqc.scf_accelerator_from_string("kdiis")`` — runner / config /
  YAML-input parsing → ``SCFAccelerator`` enum.
* ``vibeqc.SCFAccelerator.from_string("kdiis")`` — same, attached to
  the enum for ergonomic call sites.
* ``vibeqc::is_scf_converged`` (C++, header-only in
  `cpp/include/vibeqc/scf_convergence.hpp`) — refactored out of 7
  duplicated driver-loop sites; covered indirectly via the existing
  driver tests (any of which would fail if the helper's contract
  drifted from the inline pattern it replaces).
"""

from __future__ import annotations

import pytest

import vibeqc as vq
from vibeqc import SCFAccelerator, scf_accelerator_from_string


# ---------------------------------------------------------------------------
# from_string: round-trips for every enum value.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("diis",        SCFAccelerator.DIIS),
    ("DIIS",        SCFAccelerator.DIIS),
    ("Diis",        SCFAccelerator.DIIS),
    ("  diis  ",    SCFAccelerator.DIIS),
    ("kdiis",       SCFAccelerator.KDIIS),
    ("KDIIS",       SCFAccelerator.KDIIS),
    ("ediis",       SCFAccelerator.EDIIS),
    ("ediis_diis",  SCFAccelerator.EDIIS_DIIS),
    ("EDIIS_DIIS",  SCFAccelerator.EDIIS_DIIS),
    ("ediis+diis",  SCFAccelerator.EDIIS_DIIS),
])
def test_scf_accelerator_from_string_round_trip(name, expected):
    assert scf_accelerator_from_string(name) == expected


def test_scf_accelerator_from_string_attached_to_enum():
    """The same helper is available as ``SCFAccelerator.from_string``
    for the ergonomic call site."""
    assert SCFAccelerator.from_string("kdiis") == SCFAccelerator.KDIIS
    assert SCFAccelerator.from_string("ediis_diis") == SCFAccelerator.EDIIS_DIIS


def test_scf_accelerator_from_string_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown SCFAccelerator"):
        scf_accelerator_from_string("trah")        # roadmap, not yet implemented
    with pytest.raises(ValueError, match="Unknown SCFAccelerator"):
        scf_accelerator_from_string("")
    with pytest.raises(ValueError, match="Unknown SCFAccelerator"):
        scf_accelerator_from_string("newton")      # finalizer, not accelerator


def test_scf_accelerator_from_string_error_lists_options():
    """The error message names the accepted enum members so users
    can correct their input from the error alone."""
    with pytest.raises(ValueError) as exc:
        scf_accelerator_from_string("nonexistent")
    msg = str(exc.value)
    assert "DIIS" in msg
    assert "KDIIS" in msg
    assert "EDIIS" in msg
    assert "EDIIS_DIIS" in msg


@pytest.mark.parametrize("name", ["anderson", "broyden", "Anderson", "BROYDEN", " broyden "])
def test_scf_accelerator_from_string_unwired_mixers_raise(name):
    """``"anderson"`` / ``"broyden"`` name *density-space* mixers, a different
    axis from the Fock-space ``SCFAccelerator`` enum (they have no enum value).
    They used to resolve *silently* to DIIS (a wrong-method-without-warning
    trap); they must now raise a clear NotImplementedError — distinct from the
    ValueError typo path — that points the user at the real ``density_mixer=``
    entry point and still names the Fock options, so the substitution can't
    happen unnoticed."""
    with pytest.raises(NotImplementedError) as exc:
        scf_accelerator_from_string(name)
    msg = str(exc.value)
    assert "density_mixer" in msg           # points at the real density-mixer path
    assert "silently treated as DIIS" in msg
    assert "DIIS" in msg                     # names the Fock-space options


def test_scf_accelerator_unwired_mixer_is_not_valueerror():
    """Regression guard: a known-but-unwired mixer is a NotImplementedError,
    NOT the generic ValueError used for typos — the two are different
    contracts (roadmap-gap vs unrecognised name). NotImplementedError is a
    RuntimeError subclass, so it is *not* caught by ``except ValueError``."""
    with pytest.raises(NotImplementedError):
        scf_accelerator_from_string("anderson")
    try:
        scf_accelerator_from_string("anderson")
    except ValueError:
        pytest.fail("unwired mixer must not raise ValueError")
    except NotImplementedError:
        pass
