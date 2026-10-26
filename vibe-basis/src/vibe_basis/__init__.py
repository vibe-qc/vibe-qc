"""vibe-basis — basis-set optimization driver.

A standalone Python package that drives quantum-chemistry programs
under NLopt BOBYQA / iminuit MIGRAD / scipy optimization loops.
vibe-qc is the primary engine; CRYSTAL23 and GPAW are fallbacks and
independent cross-checks (see ``ROADMAP.md`` § 2). The modern
successor to the CRYSTAL09 + MINUIT2 + python pipeline that produced
pob-TZVP (Peintinger et al. 2013).

See ``README.md`` for scope, and the parent vibe-qc repository's
``docs/basisset_dev/GOAL8_MPEI_TZVP.md`` for the first production
use case (the ``mpei-TZVP`` Hartree-Fock-optimized basis set).

Top-level surface
-----------------
``vibe_basis.backends``
    One module per external SCF program. Each defines an emitter
    (parameter vector → input file), a runner-adapter (subprocess
    or vq), and an output parser.
``vibe_basis.transports``
    How external-SCF jobs reach a CPU: ``local`` (subprocess) or
    ``vq`` (vibe-queue → compute-host).
``vibe_basis.recipes``
    End-to-end driver scripts for published basis-set optimization
    recipes.
``vibe_basis.engine`` / ``vibe_basis.engines``
    The one energy-engine abstraction (``EnergyEngine``,
    ``EngineEnergy``) plus the external implementations. The primary
    engine, ``VibeQcEngine``, lives on the vibe-qc side because it
    needs the native build.
``vibe_basis.pipeline``
    Cohesive (atomization) energies: relax → single point → ZPE →
    isolated atoms → assemble, engine-neutral and resumable.
``vibe_basis.cache``
    Content-addressed energy cache with an append-only journal. Free-
    atom energies are shared across every system in a campaign.
``vibe_basis.objective``
    The L1 objective: cohesive energies scored against a plane-wave-limit
    reference, with weights, four losses and a held-out validation split.
    References and penalties are injected by the caller, so the module
    stays engine-neutral. Not to be confused with
    ``vibeqc.basis_optimization.objective``, which is the L2 layer: a
    parameter vector to one SCF energy.
``vibe_basis.topology``
    The topology search: an immutable ``BasisTopology`` with add / drop /
    decontract operators, the add-high / add-low / leave-one-out scan, a
    Pareto front over accuracy, cost and conditioning, and a hill-climb
    driver with a plateau rule. Candidate evaluation is injected.
"""

# Keep in lockstep with pyproject.toml's [project] version — ``vb
# --version`` reads this one, so a drift makes the CLI lie about what is
# installed. Pinned by tests/test_versioning.py; scheme in VERSIONING.md.
__version__ = "0.12.0"
