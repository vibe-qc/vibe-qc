# Security Policy

vibe-qc takes security seriously. This document describes how to
report a vulnerability and what's in scope.

See also: [CONTRIBUTING.md](CONTRIBUTING.md) for non-security bugs.

## Supported versions

vibe-qc is pre-1.0 software. Only the latest commit on `main` receives
security fixes — we do not backport. A supported-version table will
appear here once a 1.0 release is tagged.

## Reporting a vulnerability

Please email **mpei@vibe-qc.com** directly. Do not open a public
issue for security-relevant reports — that includes any bug
you believe could be exploited for code execution, data leakage, or
resource exhaustion beyond what the test suite would surface.

What to include in your report:

- A description of the issue and its potential impact.
- Steps to reproduce, ideally with a minimal script or input.
- The version / commit hash you observed the issue on
  (`git rev-parse HEAD` or the `VIBEQC_VERSION` from
  `vibe-qc`'s banner).

We aim to acknowledge your report within **72 hours** and will
coordinate a disclosure timeline with you privately. Public disclosure
happens after a fix is available on `main`, unless the reporter
requests otherwise.

### Encrypting your report (optional but recommended for sensitive details)

If your report contains exploit details, proof-of-concept code, or
anything you'd rather not transmit in cleartext, encrypt it to the
project author's PGP key.

**Fingerprint** — `CC6D 30BB DF96 F694 C615  FBDE 4CD5 65CF 26B1 E7E5`

(no-space form for `gpg` and URLs:
``CC6D30BBDF96F694C615FBDE4CD565CF26B1E7E5``)

**Get the key:**

```sh
# Direct from vibe-qc.com:
curl -O https://vibe-qc.com/docs/_static/pgp/mpei.asc
gpg --import mpei.asc

# Or from a keyserver:
gpg --keyserver hkps://keys.openpgp.org \
    --recv-keys CC6D30BBDF96F694C615FBDE4CD565CF26B1E7E5
```

After importing, **always verify the fingerprint matches the
canonical value listed above** before trusting the key — paste-jacking
and MITM at HTTP fetch time are real concerns. The fingerprint is a
hash; any tampered-with key would produce a different one.

```sh
gpg --fingerprint CC6D30BBDF96F694C615FBDE4CD565CF26B1E7E5
# Should print: CC6D 30BB DF96 F694 C615  FBDE 4CD5 65CF 26B1 E7E5
```

**Encrypt and send:**

```sh
gpg --encrypt --armor --recipient mpei@vibe-qc.com \
    --output report.asc report.txt
# Then attach report.asc to an email to mpei@vibe-qc.com
```

The same fingerprint is also published in the project's
[CONTRIBUTING.md](https://github.com/vibe-qc/vibe-qc/blob/main/CONTRIBUTING.md)
"Where to report what" section. If the two ever disagree, that's
itself a security signal worth flagging — email the address above
(unencrypted is fine for that meta-report).

## Scope

**In scope** — bugs in the vibe-qc code under this repository:

- `cpp/` (C++ core and pybind11 bindings)
- `python/vibeqc/` (Python frontend)
- `scripts/` (build helpers)

**Out of scope** — bugs in upstream dependencies should be reported to
those projects directly:

- [libint](https://github.com/evaleev/libint) — see the repo's
  `SECURITY.md` if present, or open an issue.
- [libxc](https://libxc.gitlab.io/) — upstream tracker on GitLab.
- [spglib](https://github.com/spglib/spglib) — GitHub issues.
- [Eigen](https://eigen.tuxfamily.org/) — their bug tracker.
- [pybind11](https://github.com/pybind/pybind11) — GitHub issues.

If you're unsure whether a finding is in scope, email it to the
address above and we'll triage together.
