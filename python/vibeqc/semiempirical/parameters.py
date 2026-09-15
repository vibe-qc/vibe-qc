"""Python-side parameter access for semiempirical methods.

Wraps the C++ ``semiempirical.SemiempiricalParameters`` class,
providing convenient access to the built-in DFTB0 parameter set
and element-level queries.

Shipped: 91 elements across the periodic table (H-U except Po/Z=84,
including lanthanides and early actinides).
All parameters are in-house estimates; see
``docs/roadmap.md`` for provenance and quality notes."""

from __future__ import annotations

# Import the C extension submodule as a plain attribute access to avoid
# triggering vibeqc.__init__ while it's still loading.
from vibeqc._vibeqc_core import semiempirical as _se_cxx

SemiempiricalParameters = _se_cxx.SemiempiricalParameters


def default_parameters():
    """Return the built-in DFTB0 parameter set (91 elements, H-U except Po)."""
    return SemiempiricalParameters.dftb0_default()


def supports_element(params, Z: int) -> bool:
    """Check whether element Z has parameters."""
    return params.has_element(Z)
