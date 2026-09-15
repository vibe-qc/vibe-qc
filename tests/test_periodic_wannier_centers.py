"""Γ-point Wannier centres in periodic QVF output.

``qvf_wannier_centers=True`` used to be implemented only for
``jk_method="aiccm2026dev-b"``, on the grounds that its finite-torus
localization convention is explicit. A Γ-sampled run has an equally explicit
convention -- at Γ the occupied Bloch functions span the whole occupied space
of the cell -- so it is now allowed there too.

**Descriptors only.** No localized coefficients are emitted for periodic
output. Home-cell coefficients with no image sum are the home-cell truncation
of the Γ crystalline orbital and clip wherever an orbital straddles a cell
face (``periodic_runner.py``, the Molden note), and a Wannier function is
*defined* by its image sum -- Zicovich-Wilson, J. Chem. Phys. 115, 9708
(2001), doi:10.1063/1.1415745, eq 1. Centres and spreads carry no such
problem. See handovers/HANDOVER_IBO.md § Milestone 3.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import vibeqc as vq


def _lih_cell():
    """A two-electron periodic cell: cheapest system with a core + a bond."""
    system = vq.PeriodicSystem(
        3,
        8.0 * np.eye(3),
        [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [4.0, 4.0, 4.0])],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _run(tmp_path, **kwargs):
    system, basis = _lih_cell()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = vq.run_periodic_job(
            system=system, basis=basis, method="rhf", **kwargs
        )
    finally:
        os.chdir(cwd)
    return result


def _wannier_entries(tmp_path):
    qvf_path = next(Path(tmp_path).glob("*.qvf"))
    with zipfile.ZipFile(qvf_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        section = next(
            (s for s in manifest["sections"] if "wannier" in s["id"]), None
        )
        if section is None:
            return None, None
        member = section["members"]["centers"]["path"]
        payload = json.loads(archive.read(member))
    entries = payload["centers"] if isinstance(payload, dict) else payload
    return section, entries


def test_gamma_run_emits_wannier_centres(tmp_path):
    result = _run(tmp_path, output_qvf=True, qvf_wannier_centers=True)
    assert result.converged

    section, entries = _wannier_entries(tmp_path)
    assert section is not None, "no x_ccm.wannier_centers section was written"
    assert section["kind"] == "x_ccm.wannier_centers"
    assert "Γ-point Wannier centres" in section["label"]

    # LiH/STO-3G has two doubly-occupied orbitals: a tight Li core and a
    # much more diffuse bonding orbital.
    assert len(entries) == 2
    spreads = sorted(float(e["spread"]) for e in entries)
    assert spreads[0] < spreads[1], "core and bond should not have equal spread"
    assert all(len(e["center"]) == 3 for e in entries)
    # Å, not bohr: the renderer plots these in the structure's own frame.
    assert all(abs(c) < 10.0 for e in entries for c in e["center"])


def test_the_viewer_can_read_what_the_runner_wrote(tmp_path):
    """Guard the producer/consumer contract, not just the producer."""
    pytest.importorskip("vibeview")
    from vibeview.qvf import QVFReader
    from vibeview.renderers.wannier import read_wannier_centres

    _run(tmp_path, output_qvf=True, qvf_wannier_centers=True)
    reader = QVFReader(str(next(Path(tmp_path).glob("*.qvf"))))
    centres = read_wannier_centres(reader)
    assert len(centres) == 2
    assert all(c.spread > 0 for c in centres)
    assert all("Wannier" in (c.label or "") for c in centres)


def test_multi_k_is_refused_rather_than_mislabelled(tmp_path):
    """The whole point of the gate.

    Localizing only the Γ orbitals of a multi-k calculation samples one k
    point; calling the result a Wannier function would be wrong. The multi-k
    scheme (Zicovich-Wilson) is not implemented, so this must refuse.
    """
    with pytest.raises(NotImplementedError, match="localization convention"):
        _run(
            tmp_path,
            output_qvf=True,
            qvf_wannier_centers=True,
            kpoints=(2, 1, 1),
        )


def test_requires_a_container_to_write_into(tmp_path):
    with pytest.raises(ValueError, match="requires output_qvf=True"):
        _run(tmp_path, output_qvf=False, qvf_wannier_centers=True)


def test_not_requested_means_not_emitted(tmp_path):
    _run(tmp_path, output_qvf=True)
    section, _ = _wannier_entries(tmp_path)
    assert section is None


def test_no_localized_coefficients_are_emitted_for_periodic(tmp_path):
    """Periodic QVF must not gain a wavefunction.gto section of localized
    orbitals: home-cell coefficients clip at the cell face."""
    _run(tmp_path, output_qvf=True, qvf_wannier_centers=True)
    qvf_path = next(Path(tmp_path).glob("*.qvf"))
    with zipfile.ZipFile(qvf_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    ids = {s["id"] for s in manifest["sections"]}
    assert not any(i.startswith("wf_localized") for i in ids)
    assert not any(i.startswith("wf_relocalized") for i in ids)
