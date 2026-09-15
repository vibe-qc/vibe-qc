#!/bin/bash
# Build the vibe-qc.com docs site into ./public/.
#
# Used by .gitlab-ci.yml's docs-build stage and by anyone who wants to
# reproduce the live site locally. Reads:
#
#   docs/                          — Sphinx source
#
# Writes:
#
#   public/                        — built HTML, ready to rsync
#   public/robots.txt              — search-engine directive
#   public/sitemap.xml             — every content URL with lastmod
#   public/.build-info             — commit SHA / branch / pipeline ID
#
# Environment:
#
#   CANONICAL                      — base URL for sitemap (default
#                                    https://vibe-qc.com)
#   OUT                            — output directory (default public)
#   CI_COMMIT_REF_NAME             — branch name; auto from GitLab CI,
#                                    falls back to `git rev-parse` for
#                                    local builds
#   CI_PIPELINE_ID                 — recorded in .build-info if set
set -euo pipefail

CANONICAL="${CANONICAL:-https://vibe-qc.com}"
OUT="${OUT:-public}"

SITE_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Companion artifacts retained here predate the split. Do not regenerate or
# silently advertise them as current releases from this core checkout.
# Their owners publish validated replacements in their own repositories.
echo "warning: docs/_static/downloads contains legacy companion artifacts; see its README.md" >&2

rm -rf "$OUT"
# --keep-going (no -W): warnings don't fail the build. The CI is for
# publishing; local sphinx-build runs (with -W) catch real warnings
# during development. Sphinx 9.x emits 79 autodoc.mocked_object
# warnings about our intentional vibeqc._vibeqc_core mock; suppressing
# them properly would need a tweak to docs/conf.py.
sphinx-build -b html --keep-going -q \
    -D autodoc_mock_imports=vibeqc._vibeqc_core \
    docs/ "$OUT"

# robots.txt
cat > "$OUT/robots.txt" <<EOF
User-agent: *
Allow: /

Sitemap: $CANONICAL/sitemap.xml
EOF

# sitemap.xml — every content .html, excluding indexes / sources / errors
LASTMOD=$(date -u +%Y-%m-%d)
{
    printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
    printf '%s\n' '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ( cd "$OUT" && find . -type f -name '*.html' | sed 's|^\./||' | sort ) \
        | while IFS= read -r path; do
            case "$path" in
                _modules/*|_sources/*|error/*|genindex.html|py-modindex.html|search.html)
                    continue
                    ;;
            esac
            printf '  <url><loc>%s/%s</loc><lastmod>%s</lastmod></url>\n' \
                "$CANONICAL" "$path" "$LASTMOD"
        done
    printf '%s\n' '</urlset>'
} > "$OUT/sitemap.xml"

# .build-info — what produced this site
{
    git log -1 --format='%H %ci %s'
    branch="${CI_COMMIT_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
    printf 'branch %s\n' "$branch"
    [ -n "${CI_PIPELINE_ID:-}" ] && printf 'pipeline %s\n' "$CI_PIPELINE_ID"
} > "$OUT/.build-info"

# Strip Sphinx internals we don't serve
rm -rf "$OUT/.doctrees" "$OUT/_sources" "$OUT/.buildinfo"

printf 'built %s files into %s\n' \
    "$(find "$OUT" -type f | wc -l | tr -d ' ')" "$OUT"
