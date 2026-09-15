# Study site configuration

The study directory contains scientific inputs and portable launchers. Keep
machine assignments, deployment paths and historic operator identifiers in a
private configuration directory outside the source checkout.

Copy `site-profile.example.json` there, edit the example aliases to match your
queue configuration, and select it with `AICCM_SITE_CONFIG`. The file is JSON
only: it cannot contain executable configuration. A profile inside the checkout,
unknown keys, duplicate keys and malformed values are rejected.

`allow_submission` defaults to `false`. Leave it disabled when reading historical
records. Enable it deliberately for remote default targets after confirming the
host assignments. Without an enabled profile, choose a remote target explicitly
with the launcher's `--host` or `--machine` option, or use `--local`. The full
molecular batch uses `molecular_hosts`; its dry-run mode renders local commands.
Local queue aliases and loopback addresses are refused as remote targets.

`tier_hosts`, `crystal_tier_hosts`, `paper_host`, `coverage_hosts` and
`posthf_host` select job destinations. `system_hosts` supplies descriptive
metadata only; it does not alter geometries, basis sets, routes or meshes.
`local_host_aliases` extends the local-host exclusions. No source-controlled
profile contains real machine assignments.

For a remote extracted job payload, set `VIBEQC_PYTHON` to the executable
interpreter in the private job environment. The wrapper otherwise uses the
checkout's `.venv/bin/python`, then a PATH Python that imports vibeqc. An empty
or invalid explicit override fails immediately. Numerical preflight and
same-process producer attestations still run after interpreter selection.

The comparison producer additionally requires an enabled profile whose
`comparison_host` matches its explicit `--vq-host`. VQ, scheduler and source
qualification checks remain mandatory. Configure the profile on that remote
host; do not copy private profiles into job payloads intended for publication.

The default bundle identifier is `aiccm-bundle-attestation/v1`. For private
historical records, `bundle_attestation_mode` can retain a legacy identifier;
the reader accepts that selected identifier and the generic identifier. This
compatibility setting does not authorize a frozen campaign or enable submission.
Existing archived producers and installed campaign bundles are not updated by
these source changes.
