#!/usr/bin/env python3
"""Run with packaged Python, an empty HOME/XDG cache, and a writable working directory.

Run once online and again with --offline using the same cache. The second run
blocks Python network connections and verifies the same energy, forces and stress.
No model weights belong in the application payload.
"""

import hashlib
import json
import socket
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

if "--offline" in sys.argv:
    def blocked(*args, **kwargs):
        raise RuntimeError("Network access during cached MACE qualification")
    socket.socket.connect = blocked
    socket.create_connection = blocked

import vibeqc
from vibeqc import Atom, BasisSet, Molecule, run_job, run_rhf
from vibeqc.ase_periodic import atoms_to_periodic_system
from vibeqc.mlip import MLIPOptions
from vibeqc.mlip.mace import MACEModel, PeriodicMACEEvaluator, mace_cache_root
from ase.build import bulk

prefix = Path(sys.prefix).resolve()
assert sys.version_info[:2] == (3, 14)
assert Path(vibeqc.__file__).resolve().is_relative_to(prefix)
assert not list(prefix.rglob("*.model")), "Model weights must download on first use"
assert not mace_cache_root().resolve().is_relative_to(prefix)
if "--offline" not in sys.argv:
    assert not mace_cache_root().exists() or not list(mace_cache_root().iterdir())

def hf():
    molecule = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
    result = run_rhf(molecule, BasisSet(molecule, "sto-3g"))
    assert result.converged and abs(result.energy + 1.1167143250625702) < 1e-8
    return result.energy

before = hf()
Path("water.xyz").write_text("3\nwater\nO 0 0 0\nH 0.758602 0 0.504284\nH -0.758602 0 0.504284\n")
molecule = Molecule.from_xyz("water.xyz")
options = MLIPOptions(model="medium-mpa-0", device="cpu", dtype="float64")
model = MACEModel(molecule, options)
energy, gradient = model.energy(), model.gradient()
assert abs(energy - (-0.49841867929111155)) < 1e-9
assert np.isfinite(energy) and gradient.shape == (3, 3) and np.isfinite(gradient).all()
run_job(molecule, method="mace", mlip_options=options, output="water-mace")
for suffix in ("out", "qvf", "bibtex", "references"):
    assert Path("water-mace." + suffix).stat().st_size > 0
assert "MACE" in Path("water-mace.out").read_text()
silicon = PeriodicMACEEvaluator(options).run(atoms_to_periodic_system(bulk("Si", "diamond", a=5.43)))
assert np.isfinite(silicon.gradient()).all() and np.isfinite(silicon.stress()).all()
assert abs(silicon.energy - (-0.39780276599773545)) < 1e-9
assert np.allclose(np.diag(silicon.stress()), -6.29615473995103e-5, atol=1e-10)
assert model._weight_identity.sha256 == "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
assert abs(hf() - before) < 1e-12
record = {"python": sys.version.split()[0],
          "packages": {name: version(name) for name in ("vibe-qc", "mace-torch", "torch", "e3nn", "setuptools")},
          "water_energy_Ha": energy, "water_gradient_Ha_bohr": gradient.tolist(),
          "silicon_energy_Ha": silicon.energy, "silicon_stress_Ha_bohr3": silicon.stress().tolist(),
          "checkpoint_sha256": model._weight_identity.sha256,
          "native_sha256": hashlib.sha256(Path(sys.modules["vibeqc._vibeqc_core"].__file__).read_bytes()).hexdigest()}
if "--offline" in sys.argv:
    prior = json.loads(Path("mace-online.json").read_text())
    for key in ("water_energy_Ha", "water_gradient_Ha_bohr", "silicon_energy_Ha", "silicon_stress_Ha_bohr3"):
        assert np.allclose(record[key], prior[key], atol=1e-12, rtol=0), key
Path("mace-offline.json" if "--offline" in sys.argv else "mace-online.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
