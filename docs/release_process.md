# Release process

vibe-qc is independently versioned. Development changes land on the canonical
GitLab main branch through the existing contributor and review process.
Released tags are immutable. Do not rewrite published history or move a
release tag to repair documentation or packaging metadata.

## Contributor evidence

Follow the repository's CONTRIBUTING.md and developer test guidance. A change
needs the relevant tests and validation comparisons, with commands, selected
source revision and results recorded for review. A release blocker or a fix to
an incorrect numerical result requires an independent check before closure.
Add ordinary release notes as individual Markdown fragments under
`changelog.d/`; do not edit `CHANGELOG.md` directly. Validate them with
`python scripts/assemble_changelog.py --check`. The release owner assembles
them into `[Unreleased]` when preparing a release, following
`changelog.d/README.md`. Preserve released changelog sections and their pins.

AI-assisted changes follow the same disclosure, attribution, licensing and
validation requirements as other contributions. Assistance does not replace
review or evidence. See the existing contributor policy for the full contract.

## Pre-flight environment integrity

A gate is evidence about the tree the interpreter actually loaded, not the tree
you mean to tag, so the release lane checks the environment before it runs the
blocking tier. Four failures have each produced a false result here, and none
of them announces itself:

- **An editable install silently replaced by a copy.** A wheel-based reinstall
  can leave `import vibeqc` resolving into `site-packages` rather than
  `python/vibeqc/`, so later edits are invisible to pytest and a green suite
  against a stale copy looks like any other green suite. Restore with
  `pip install -e . --no-build-isolation`, and confirm `vibeqc.__file__`
  resolves inside the checkout being gated.
- **A stale `build/basis_library/` overlay.** The overlay is preferred over the
  committed library, so a basis missing from it does not resolve at all, and
  the error names the basis rather than the overlay. Regenerate it with
  `bash scripts/setup_basis_library.sh`.
- **A stale `.pyc` that survived an edit made inside the same second.**
  CPython's timestamp invalidation has one-second granularity, so a checkout or
  rebase that rewrites a source file within the same second the cache was
  written leaves the cache looking current and the old code running. Suspect it
  when a value read back does not match the file in front of you, and clear it
  with `find python -name __pycache__ -type d -prune -exec rm -rf {} +`.
- **A compiled core built from another tree.** The installed distribution
  version records the release line the core was built from, so it catches a
  core from an earlier line; it cannot tell two builds apart inside one `.dev0`
  cycle. Compare the extension's timestamp with the newest `cpp/` source as
  well, and rebuild when the core is older.

The candidate pipeline also runs the changelog guard.
`changelog_guard.py check` compares every released `CHANGELOG.md` section with
its SHA-256 pin, and `changelog_guard.py audit --require-tagged 1` compares each
pin with the same section at its own git tag. The audit needs the tags, so it
runs in the candidate pipeline rather than in the blocking tier, whose clone may
be shallow, and `--require-tagged` makes a clone without tags fail instead of
passing without having compared anything. An entry for unreleased work belongs
under `[Unreleased]`; a deliberate change to an already released section needs
`changelog_guard.py pin vX.Y.Z --amended "<reason>"` in the same commit.

The blocking gate is the **T1 tier**, the same invocation the candidate
pipeline runs, so pre-flight and CI cannot disagree about what green means.
That equivalence holds only while the local environment matches CI's: the
viewer installed into the release venv, for example, makes the local gate run
files CI skips. A new red on a loaded machine is not yet a regression; re-run
it in isolation with `scripts/test_gate/gate_verdict.py --reverify` before
reading it as one.

Candidate CI explicitly removes `VIBEQC_ALLOW_STALE_CORE`, records the loaded
package, source SHA, core hash and linked-library versions in
`test-results/native-runtime.json`, and checks the fresh core before T1.
Candidate artifacts also retain the matching core and CMake cache under
`test-results/native-core/`. The receipt records native source and linked
dependency paths. This is not a standalone runtime: any independent reuse
must qualify the interpreter, dependencies and data before calculations.
After T1 it reports the release-impact selection against `VIBEQC_RELEASE_BASE`;
these informational calls do not feed T2 results to the T0/T1 verdict. Update
that CI variable to the previous released tag when preparing a patch.
The release-core sentinel is already included in T1.

### Viewer producer/consumer gate

After the candidate's T1 gate, CI installs the pinned viewer and runs the
producer/consumer selection as an unprivileged user under Xvfb. The selection
includes the original seven contract files plus `test_relocalize.py`,
`test_wavefunction_gradient.py`, `test_manifest_root_blocks.py`,
`test_plot_renderers.py` and the pending-QVF submission case in
`test_controller_state.py` (#363).

`VIBEQC_RELOCALIZE_PYTHON` points to the candidate's installed interpreter so
the six external-worker relocalization cases execute. A native import
preflight must succeed before pytest. Review the retained
`test-results/viewer-contract.xml` and the job's skip summary: a passing
protocol-only run with a missing QC core is not integration acceptance.

## Maintainer publication

The release owner selects scope and version only after the required gates
pass. The agentic loop coordinates the source tag and supported publication
jobs. Site-specific commands, accounts, destinations and gate-operation
runbooks are maintained in the private operations repository. Contributor
source archives do not contain or install those operational scripts.

A public source snapshot has its own commit identity and records the original
source revision and per-file checksums. It contains no private development
ancestry. A correction is published as a new snapshot or patch release;
existing source tags remain unchanged. Public availability is established by
successful anonymous access and the publication verification report.

The [GitHub publication policy](github_publication.md) explains which releases
and qualified main revisions are exported, the checks before each update,
and the current automation status. Main publication is separate from cutting
a release.

The core release lane retains its T1 and candidate validation requirements.
Ordinary main pushes do not replace release-gate evidence. The manual publishes
from the qualified release; companion versions are validated at their selected
pins. Moving operational configuration does not authorize a tag or deployment.

## Documentation cadence

Review documentation with each behavior change. Contributor guidance and
reference examples should match the selected source revision. The public
manual is published through the existing qualified-release gate; the private
operator runbook records the publication procedure and periodic review schedule.

## Platform packages

See [distribution builds](distribution.md) for Linux supporter packages and paid
Apple/Microsoft Store editions, runtime requirements and channel acceptance.
