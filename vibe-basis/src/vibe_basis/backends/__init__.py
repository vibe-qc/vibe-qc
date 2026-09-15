"""External SCF-program backends.

Each module implements the same interface triple:

* ``emit_input(params, ...) -> str``
    Serialize a basis + system into the program's native input
    format (e.g. CRYSTAL .d12, ORCA .inp).
* ``run(input_path, transport, ...) -> RunResult``
    Drive the program via the chosen transport (local subprocess
    or vq submit). Wait for completion. Return paths to stdout /
    stderr / structured output.
* ``parse_output(text_or_path) -> EnergyResult``
    Extract total energy + convergence flags. Tags failure modes
    distinctly so the optimizer can route around bad evaluations.

vibe-basis treats every supported program as an **external** code
per the same discipline as vibe-qc's CLAUDE.md § 10: we drive them
via subprocess + file I/O, never as a Python library.
"""
