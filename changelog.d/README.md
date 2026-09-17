# Changelog fragments

One file per user-visible change. **Do not edit `CHANGELOG.md` directly.**

## Why

Every branch that appends to `CHANGELOG.md`'s `[Unreleased]` section inserts at
the same anchor. Two branches doing that conflict as an add/add with an empty
base region, which no content edit resolves -- only moving the merge base does.

With a single merge queue and a serialised `native-build-test` resource group,
that turns draining N ready merge requests into N(N+1)/2 pipeline runs instead
of N: each merge re-conflicts everything queued behind it, every one of those
needs a rebase, and every rebase means another run through a one-at-a-time
queue. Five ready merge requests measured at roughly 2.5 hours instead of 50
minutes.

Fragments remove the shared anchor. Two branches never touch the same path, so
they never conflict.

## How

Add one file here named for the issue or a short slug:

    changelog.d/284-multi-k-gdf-override.md
    changelog.d/fix-euler-pole.md

Start it with a `### ` heading or a `- ` bullet, matching how the entry should
read in `CHANGELOG.md`:

```markdown
### Fixed: multi-k GDF reports when it overrides use_compcell (#284)

On the shipped dim-3 `gdf_method="rsgdf"` route the Hartree always comes from
the cached-Lpq fit, so a caller asking for the analytic Ewald-3D J was given
the fitted one without being told. It now logs.
```

Tests-only and internal changes may skip the changelog, as before.

## Release

`scripts/assemble_changelog.py` folds every fragment into `[Unreleased]` and,
with `--clean`, removes them:

    python scripts/assemble_changelog.py --clean

Run it when cutting a release, before the promotion commit that turns
`[Unreleased]` into a `## [vX.Y.Z]` section. It never touches released sections,
so `scripts/test_gate/changelog_pins.toml` and `changelog_guard.py` are
unaffected -- the guard pins released bodies only and never pins `[Unreleased]`.

`--check` validates fragment formatting without writing anything.
