# CODEX.md

Codex reads [`AGENTS.md`](AGENTS.md) on its own. This file exists so anyone
looking for Codex guidance finds the pointer: all project rules live in
`AGENTS.md`.

## Codex notes

- **The native build is long and needs the vendored dependencies.** A fresh
  clone spends 15-40 minutes in `scripts/setup_native_deps.sh`, and a sandbox
  without network access may not fetch them. If you cannot build, say so and
  name the tests you could not run, rather than reporting the change as tested.
- **Pure-Python changes still import the native core.** Most tests need a built
  `_vibeqc_core`; check it imports before running a lane.
- **Codex sessions share the maintainer's git identity** with every other
  agent. In a shared checkout, stage files by name and leave changes you didn't
  make alone (`AGENTS.md`, "Working alongside other sessions").
