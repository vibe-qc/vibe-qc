# Legacy companion artifacts

The qvf-writer tarball and vibeview wheel in this directory came from the
pre-split monorepo. They are retained for reproducibility and existing
version-pinned repair URLs, not advertised as current companion releases.
Core builds do not regenerate them.

The QVF and vibe-view release owners must validate and publish replacement
artifacts, checksums, and bootstrap URLs in their respective repositories:

- https://github.com/vibe-qc/qvf/releases
- https://github.com/vibe-qc/vibe-view/releases

Do not overwrite an old artifact with different bytes under the same version.
The documentation's current installation route is a separate source clone
until an appropriate companion artifact is available.
