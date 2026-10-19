#!/usr/bin/env python3
"""Check relocated commands and a small native calculation in an installed app."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CALCULATION = """
import json, pathlib, sys
import numpy as np
import vibeqc
from vibeqc import Atom, Molecule, BasisSet, run_rhf, run_rks, RKSOptions
from rdkit import Chem
from optimade.client import OptimadeClient
from dftd3.interface import DispersionModel, RationalDampingParam
from dftd4.interface import DispersionModel as D4, DampingParam

prefix = pathlib.Path(sys.prefix).resolve()
import ctypes
fftw = ctypes.CDLL(str(next((prefix / 'lib/vibe-native').glob('libfftw3.*.dylib'))))
fftw.fftw_plan_dft_1d.restype = ctypes.c_void_p
fftw.fftw_plan_dft_1d.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_uint]
fftw.fftw_execute.argtypes = [ctypes.c_void_p]
fftw.fftw_destroy_plan.argtypes = [ctypes.c_void_p]
x = np.arange(8, dtype=np.complex128)
y = np.zeros(8, dtype=np.complex128)
plan = fftw.fftw_plan_dft_1d(8, x.ctypes.data, y.ctypes.data, -1, 64)
assert plan
try:
    fftw.fftw_execute(plan)
    assert np.allclose(y, np.fft.fft(x))
finally:
    fftw.fftw_destroy_plan(plan)
assert pathlib.Path(vibeqc.__file__).resolve().is_relative_to(prefix)
basis_root = vibeqc._resolve_basis_library()
assert basis_root.resolve().is_relative_to(prefix)
assert (basis_root / 'basis/def2-svp.g94').is_file()
assert pathlib.Path(vibeqc._resolve_bundled_ecp_share_dir()).resolve().is_relative_to(prefix)
from vibeqc.fetch.client_cod import PeriodicSpec
assert PeriodicSpec.__module__ == 'examples.regression.core.spec'
mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
basis = BasisSet(mol, 'sto-3g')
hf = run_rhf(mol, basis)
assert hf.converged and abs(hf.energy - (-1.1167143250625702)) < 1e-8
options = RKSOptions()
options.functional = 'LDA'
dft = run_rks(mol, basis, options)
assert dft.converged and abs(dft.energy - (-1.121200704006311)) < 1e-8
assert Chem.MolFromSmiles('CCO').GetNumAtoms() == 3
numbers = np.array([1, 1])
positions = np.array([[0., 0., 0.], [0., 0., 1.4]])
d3 = DispersionModel(numbers, positions).get_dispersion(RationalDampingParam(method='pbe'), grad=True)
d4 = D4(numbers, positions).get_dispersion(DampingParam(method='pbe'), grad=True)
assert np.isfinite(d3['energy']) and np.isfinite(d4['energy'])
print(json.dumps({'rhf': hf.energy, 'lda': dft.energy,
                  'd3': float(d3['energy']), 'd4': float(d4['energy'])}))
"""


def snapshot(app: Path) -> dict[str, str]:
    result = {}
    for path in app.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                result[str(path.relative_to(app))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return result


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="qc-installed-") as temp:
        root = Path(temp)
        app = root / "Installed Apps With Spaces" / source.name
        shutil.copytree(source, app, symlinks=True)
        home = root / "empty-home"
        home.mkdir()
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PYTHON", "DYLD_", "VIBEQC_", "LIBINT_"))
        }
        env.update(
            HOME=str(home),
            PATH="/usr/bin:/bin:/usr/sbin:/sbin",
            XDG_CONFIG_HOME=str(home / "config"),
            XDG_DATA_HOME=str(home / "data"),
            XDG_CACHE_HOME=str(home / "cache"),
            OMP_NUM_THREADS="2",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
            VIRTUAL_ENV="/nonexistent",
        )
        before = snapshot(app)
        resources = app / "Contents/Resources"
        manifest = json.loads((resources / "runtime.json").read_text())
        for command in manifest["entrypoints"]:
            subprocess.run(
                [app / "Contents/MacOS" / command, "--help"],
                env=env,
                cwd=home,
                check=True,
                capture_output=True,
                timeout=60,
            )
        calculation = root / "calculation.py"
        calculation.write_text(CALCULATION)
        result = subprocess.run(
            [resources / "python/bin/python3", "-I", "-B", calculation],
            env=env,
            cwd=home,
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
        print(result.stdout.strip())
        assert snapshot(app) == before, "Calculation modified application resources"
        print(
            "PASS: relocated commands, bundled basis/native RHF and LDA, RDKit, D3/D4; "
            "isolated home and PATH; application unchanged"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Installed command failed:\n{error.stdout}\n{error.stderr}"
        ) from error
