"""External-reference controls are explicit and sidecar case is immaterial."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).parents[1]
SCRIPT=ROOT/'examples/regression/periodic_ecp_reference.py'


def test_generated_mixed_case_ecp_headers_are_read_without_changing_terms():
    spec=importlib.util.spec_from_file_location('ecp_reference',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=ROOT/'python/vibeqc/basis_library/basis/lanl2dz.ecp'
    for symbol,ncore in [('Na',10),('Cl',10)]:
        text=module._sidecar_nwchem(path,symbol)
        assert text.startswith(f'{symbol} nelec {ncore}\n{symbol} ul\n')
        assert f'{symbol} s\n' in text and f'{symbol} p\n' in text


@pytest.mark.parametrize('omega',['0','-1','nan','inf'])
def test_invalid_omega_refused_before_external_import_or_output_creation(tmp_path,omega):
    output=tmp_path/'unused'
    p=subprocess.run([sys.executable,str(SCRIPT),'--output',str(output),'--omega',omega],capture_output=True,text=True)
    assert p.returncode==2 and 'omega must be finite and positive' in p.stderr
    assert not output.exists()
