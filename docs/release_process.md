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
Keep new release notes under Unreleased; preserve released changelog sections.

AI-assisted changes follow the same disclosure, attribution, licensing and
validation requirements as other contributions. Assistance does not replace
review or evidence. See the existing contributor policy for the full contract.

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
