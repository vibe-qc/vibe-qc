"""``python -m vibeqc`` — the ``vibeqc`` CLI as a runnable module.

Same entry point as the ``vibeqc`` console script
(:func:`vibeqc._cli.main`). This exists for callers that hold a Python
interpreter rather than a bin directory — notably vq's first-class QVF
container dispatch, which runs ``$VENV/bin/python -m vibeqc run
job.qvf`` on the managed runtime it resolved.
"""

from __future__ import annotations

import sys

from vibeqc._cli import main

if __name__ == "__main__":
    sys.exit(main())
