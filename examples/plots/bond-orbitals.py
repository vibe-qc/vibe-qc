#!/usr/bin/env python
"""Regenerate ``docs/_static/plots/bond-orbitals.png`` for tutorial 11.

Renders the bonding molecular orbitals that distinguish single, double, and
triple bonds, captured from the *real* vibe-view viewport (offscreen):

  * H2        -> the sigma single bond (one lobe, single sign),
  * ethylene  -> the C=C pi of the double bond (HOMO),
  * acetylene -> one of the two perpendicular C=C... C-C pi orbitals of the
                 triple bond (HOMO).

For each molecule it runs an RHF job, writes a ``.qvf`` with the orbital
volume section, opens it in headless vibe-view (``PYVISTA_OFF_SCREEN``),
activates the "Molecular Orbitals" section, screenshots the canvas, and
composes the three crops side by side. The crop is the 3D canvas only, so no
run-info / provenance host is baked into the committed image.

This is a one-off docs tool, not a runtime/test/CI dependency. It needs
the separately installed vibe-view executable on PATH (see
``docs/tutorial/vibe_view_getting_started.md``). To use another location, set
``VIBE_VIEW_EXECUTABLE`` to its absolute path. Only the capture tooling must
be installed into the core environment::

    .venv/bin/pip install playwright pillow && .venv/bin/playwright install chromium

Then::

    PYVISTA_OFF_SCREEN=True .venv/bin/python examples/plots/bond-orbitals.py
"""

from __future__ import annotations

import io
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

import vibeqc as vq

ROOT = Path(__file__).resolve().parents[2]
VIEWER = shutil.which(os.environ.get("VIBE_VIEW_EXECUTABLE", "vibe-view"))
OUT = ROOT / "docs" / "_static" / "plots" / "bond-orbitals.png"
VIEWPORT = {"width": 1440, "height": 880}

# (molecule, label, base port). Geometries in bohr; the relevant bonding
# orbital is the HOMO in every case, so write_cube=["homo"] is enough.
MOLECULES = [
    ("h2", vq.Molecule([vq.Atom(1, [0, 0, 0.7]), vq.Atom(1, [0, 0, -0.7])]), 8152),
    ("ethylene", vq.Molecule([
        vq.Atom(6, [1.265, 0.0, 0.0]), vq.Atom(6, [-1.265, 0.0, 0.0]),
        vq.Atom(1, [2.340, 1.740, 0.0]), vq.Atom(1, [2.340, -1.740, 0.0]),
        vq.Atom(1, [-2.340, 1.740, 0.0]), vq.Atom(1, [-2.340, -1.740, 0.0]),
    ]), 8153),
    ("acetylene", vq.Molecule([
        vq.Atom(6, [0, 0, 1.136]), vq.Atom(6, [0, 0, -1.136]),
        vq.Atom(1, [0, 0, 3.141]), vq.Atom(1, [0, 0, -3.141]),
    ]), 8154),
]


def _wait_port(port: int, timeout: float = 60.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.4)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"vibe-view never came up on :{port}")


def _wait_ready(page) -> None:
    page.wait_for_selector(".v-list-item", timeout=30_000)
    for _ in range(40):
        ready = page.evaluate("() => !!document.querySelector('canvas')")
        loading = page.evaluate("() => /Loading|Awaiting/.test(document.body.innerText||'')")
        if ready and not loading:
            break
        time.sleep(0.5)
    time.sleep(2.0)


def _capture(qvf: Path, port: int) -> Image.Image:
    """Open the QVF in offscreen vibe-view, render its orbital, return a crop."""
    proc = subprocess.Popen(
        [str(VIEWER), "open", str(qvf), "--no-browser",
         "--host", "127.0.0.1", "--port", str(port)],
        env={**os.environ, "PYVISTA_OFF_SCREEN": "True"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
            _wait_ready(page)
            page.locator(".v-list-item", has_text="Molecular Orbitals").first.click()
            time.sleep(4)
            # nudge the camera so a fresh WebGL frame paints before the shot
            page.mouse.move(720, 440)
            page.mouse.down()
            page.mouse.move(731, 448, steps=4)
            page.mouse.up()
            time.sleep(0.8)
            box = page.locator("canvas").first.bounding_box()
            raw = page.screenshot()
            browser.close()
        crop = (int(box["x"]), int(box["y"]),
                int(box["x"] + box["width"]), int(box["y"] + box["height"]))
        return Image.open(io.BytesIO(raw)).convert("RGB").crop(crop)
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(1)


def main() -> None:
    if not VIEWER:
        raise SystemExit(
            "vibe-view not found; install the separate viewer and put it on PATH, "
            "or set VIBE_VIEW_EXECUTABLE to its absolute executable path"
        )
    work = Path(os.environ.get("BOND_VIZ_WORKDIR", "/tmp/bond-orbitals"))
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)

    panels = []
    for name, mol, port in MOLECULES:
        vq.run_job(mol, basis="6-31g*", method="rhf",
                   write_cube=["homo"], output_qvf=True, output=name)
        panels.append(_capture(work / f"{name}.qvf", port))

    h = min(p.height for p in panels)
    panels = [p.resize((int(p.width * h / p.height), h)) for p in panels]
    gap, bg = 14, (26, 26, 42)
    total = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (total, h), bg)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    target_w = 1500
    canvas = canvas.resize((target_w, int(h * target_w / total)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
