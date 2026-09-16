# GitHub source publication

The [vibe-qc organization](https://github.com/vibe-qc) hosts public source
snapshots for vibe-qc, vibe-view, vibe-queue and QVF. GitLab remains the
development source of truth. The same publication policy applies to all four.

## Current status

As of 2026-09-16, all four initial public repositories and their selected
release tags are available, with successful anonymous clone verification.
The one-way transport from the separate publication repositories to GitHub
is configured. The unattended snapshot producer and its schedule are still
being completed. New source pushes are therefore not yet automatically
exported. Update this status when deployment and its end-to-end checks pass.

## Which changes will be published?

| Source event | Publication policy |
| --- | --- |
| A new stable `vX.Y.Z` release | Publish its clean source snapshot under the same version tag after the exact source commit passes the project's release gate and the publication checks. A tag alone is insufficient. |
| An update to `main` | Publish selected, qualified current heads after the required validation and the same publication checks. Several development commits may become one public snapshot. |
| A feature, work, release-candidate or deployment branch | Do not publish the branch. |
| A fork pull request | Do not publish or execute it through the publisher. Contributions use a separate review and validation route. |

The loop submits the exact commit and its qualification evidence to one shared
publisher. It does not send an unrestricted stream of development commits.
Core main pushes ordinarily have no test pipeline; an absent or skipped
pipeline is not qualification. The loop must supply accepted test or gate
evidence for that exact revision. Main publication does not declare a release
or replace release validation.

The deployment schedule has not yet been set. The intended trigger is successful
qualification, with retries for incomplete publication. There is no promise
that every main commit will appear on GitHub, or that a GitLab push becomes
public immediately. If main advances before its candidate is accepted for
publication, the stale request stops and the new head needs its own evidence.

## Checks before every publication

```text
Qualified GitLab revision
    -> byte-identical source snapshot (no private ancestry)
    -> privacy, secret and public-history checks
    -> separate publication repository
    -> one-way GitLab push mirror
    -> anonymous GitHub verification
```

Every candidate, whether main or a release, goes through these checks:

1. Verify the selected commit, allowed ref and successful qualification
   evidence. Evidence from another revision cannot qualify this one.
2. Export every committed source file with identical bytes and modes. The
   clean-source gate rejects content rewrites, omissions and generated guide
   replacements; only publication provenance and inventory files are added.
   Private operations and cluster configuration must already live outside
   the product repository. Development ancestry stays outside the export. The publisher never runs source hooks, build scripts
   or contributor code with its credentials.
3. Scan the complete candidate public history reachable from all approved
   branches and tags. Check secrets, private identifiers and paths, filenames,
   commit metadata, archives, embedded metadata, images and large files.
   Repository-controlled scanner exceptions cannot disable this gate.
4. Verify the exported file inventory, provenance and Git objects. Validation
   must cover the exact source revision; a clean scan alone does not establish
   that a program works. Historical bootstrap exports had separately reviewed
   sanitation rules. They are not a fallback for future dirty source.
5. Push only approved public refs, then verify the actual GitHub result without
   credentials. A successful staging push alone does not mean publication
   succeeded.

The scans run for every candidate. Previously reviewed findings can be reused
only when their path and exact matching context still agree with the review.
Binary and visual reviews require identical file bytes. New paths, changed
images, new findings and unknown payloads stop publication for review.
Scanning does not silently redact new findings or rewrite source history.

If a check fails before staging, GitHub keeps its last accepted snapshot. If
delivery or anonymous verification fails after staging, publication remains
incomplete until the actual remote result is verified. The loop records the
blocker privately and retries the same mapping after the cause is resolved.
Private scan matches and operational logs are not copied to public issues.

## What "without development history" means

GitHub has its own small history of accepted public snapshots. Each main
update appends a public snapshot commit; it does not recreate the repository
or force-push a new root. A release tag identifies its own reviewed snapshot
without moving public main backwards. Existing release tags never move.

The original GitLab commits and private ancestry are never copied into that
history. Public commit IDs consequently differ from source commit IDs.
Provenance and per-file checksums record the relationship. Original GitLab
tags remain unchanged, including the source releases described by the paper.

## Private configuration and runtime state

Keep host aliases, cluster accounts, deployment destinations, private tracker
identifiers and operator evidence outside all product checkouts, worktrees and
Git databases. Select an absolute external root with `VIBE_PRIVATE_ROOT`; the
private tooling defaults to `$XDG_STATE_HOME/vibe-private`, or
`~/.local/state/vibe-private`. Private directories use mode `0700` and files
`0600`. Credentials remain in their existing secret stores.

Public source contains generic interfaces and safe example profiles. Core study
launchers accept an absolute external `AICCM_SITE_CONFIG`; the optional hook
policy uses `VIBE_PRIVACY_TERMS_FILE` or a clone-local `privacy.termsFile` pointer
to an external file. Missing study profiles do not enable remote submissions.
GitLab deployment selects its private include using `VIBEQC_CI_PROJECT`,
`VIBEQC_CI_REF` and `VIBEQC_CI_FILE`. All three unset means portable source-only
CI; partial configuration fails validation. Private access recipes live in
maintainer operations documentation.

Ordinary virtual environments, native build caches and installer metadata may
remain in a checkout. Existing calculation bundles and frozen release evidence
are not moved by publication. Their owner must preserve hashes, references and
reader compatibility before a recorded migration.

## Responsibilities and contributions

Release owners prepare and validate releases. The agentic loop qualifies
revisions and submits publication requests. The shared publisher owns export,
scanning, staging and verification; individual release tasks do not push
directly to GitHub. Credentials and operational configuration stay in private
infrastructure, not in product source repositories.

Only the four product repositories are public publication targets. The agentic
loop, paper repositories, library and operator records remain private. Every
stable release cut is submitted automatically once the unattended publisher
is deployed, but publication still requires the exact release's qualification
and scan results. The current deployment status above applies.

GitHub issues and pull requests are enabled. Automatic issue routing and the
PR-to-MR bridge are not yet active as of 2026-09-16. The proposed contribution
route is a private-side importer that reads public pull requests and transfers
their patches into GitLab merge requests, preserving attribution and the public
PR description and link. It binds each import to the exact PR head and base;
new pushes invalidate prior validation. It does not merge unrelated public
snapshot history into GitLab or allow GitHub to overwrite canonical refs.

Fork code is untrusted. The publisher never executes it with credentials, and
the importer does not run it while importing. Validation requires a separate,
unprivileged, secret-free runner context and maintainer admission. A configured
mirror does not provide that isolation. The bridge stays disabled until this
boundary and an end-to-end fork contribution have been qualified.

Public replies contain sanitized status and public links only, never private
GitLab MR URLs, runner details or raw CI logs. After GitLab review and green CI,
a merge follows the normal qualification and publication gate. The importer
closes the PR as merged upstream only after the corresponding public result is
verified. A declined private MR produces a reviewed public reason. No reverse
mirror writes to GitLab main. Tracker routing is a separate maintainer decision;
until it is enabled, maintainers explicitly connect public reports to their
private issue queue.

See the [release process](release_process.md) for release ownership and the
[contributor policy](contributing.md) for required evidence and AI assistance
disclosure. Maintainers keep the detailed publisher contract, receipts and
deployment procedure in the private operations repository.
