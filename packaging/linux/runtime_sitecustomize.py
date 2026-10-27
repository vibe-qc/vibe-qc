"""Use bundled CA certificates only when a standalone host has no trust store.

Installed as sitecustomize.py in the optional runtime, including for direct
Python and vq use. Explicit SSL settings and populated system stores win.
"""

import os
from pathlib import Path
import ssl


def configure_trust():
    if "SSL_CERT_FILE" in os.environ or "SSL_CERT_DIR" in os.environ:
        return
    paths = ssl.get_default_verify_paths()
    if paths.cafile and Path(paths.cafile).is_file():
        return
    if paths.capath and any(Path(paths.capath).glob("*.0")):
        return
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()


if __name__ == "sitecustomize":
    configure_trust()
