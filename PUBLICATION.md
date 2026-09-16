# Public source snapshots

GitLab is the canonical development source. GitHub contains independently
committed snapshots of accepted source revisions, with public snapshot history
only. Original development history is not mirrored.

Each future main or release candidate must have qualified source evidence and
pass the privacy gate. Every committed source file is exported with identical
bytes and modes; private profiles and operational records must already be
outside the product repository. Publication adds only source provenance and a
file inventory. Existing source and public release tags remain immutable.

See [the publication mechanism](docs/github_publication.md) for the main and
release policy, checks, private configuration boundaries and current automation
status. A configured GitLab push mirror is transport, not evidence that an
unattended snapshot publisher is active.
