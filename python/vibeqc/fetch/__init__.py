"""External structure fetcher for vibe-qc benchmarking.

Pulls crystal & molecular structures from open external databases
(OPTIMADE federation primary; Materials Project, NOMAD, COD as
secondaries) and emits two artefacts per structure:

  1. a ``PeriodicSpec`` / ``MoleculeSpec`` instance compatible with
     ``examples/regression/systems/`` (consumed by
     ``examples.regression.run_suite``);
  2. an executable ``input-<id>-<basis>.py`` script in the same style
     as ``examples/input-mgo-pob-tzvp.py`` (consumed directly by
     ``python -m examples.input-...``).

See ``docs/tutorial/external_data_fetcher.md`` for the design contract.
"""
from __future__ import annotations

# Fetcher version. Bumped manually when the heuristics / Provenance
# emission contract changes -- every fetched SPEC stamps this into its
# Provenance record so downstream readers know which fetcher wrote it.
__version__ = "0.1.0"

# The schema dataclasses come from the top-level ``vibeqc_spec`` package,
# which the wheel ships, so the fetcher needs no ``examples/`` tree on
# ``sys.path``.

from .cache import FetchCache, FetchCacheMiss
from .client_cod import fetch_cod
from .client_optimade import fetch_optimade
from .emit_input import emit_input_script
from .emit_spec import emit_spec_module
from .heuristics import (
    open_shell_default,
    pick_damping,
    pick_initial_guess,
    pick_kmesh,
    pick_recommended_basis,
)


def __getattr__(name):
    """Lazy re-exports from ``vibeqc.fetch.references`` for the natural
    user-facing import path ``from vibeqc.fetch import fetch_cccbdb``.

    Phase 2 (references) optionally depends on ``beautifulsoup4`` /
    ``lxml`` -- lazy-loading keeps the structures-only path
    (``vibeqc.fetch.fetch_optimade``) usable when those aren't
    installed.
    """
    if name == "fetch_cccbdb":
        from .references.client_cccbdb import fetch_cccbdb as f
        return f
    if name == "experimental_geometry_to_molecule_spec":
        from .references.geometry_bridge import experimental_geometry_to_molecule_spec as f
        return f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "FetchCache",
    "FetchCacheMiss",
    "fetch_cod",
    "fetch_optimade",
    "fetch_cccbdb",                                # lazy via __getattr__
    "experimental_geometry_to_molecule_spec",      # lazy via __getattr__
    "emit_input_script",
    "emit_spec_module",
    "open_shell_default",
    "pick_damping",
    "pick_initial_guess",
    "pick_kmesh",
    "pick_recommended_basis",
]
