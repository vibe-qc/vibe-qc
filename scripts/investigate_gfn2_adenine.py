#!/usr/bin/env python3
"""Investigate GFN2-xTB energy: test aes_faithful path."""
from __future__ import annotations

from vibeqc import Molecule
from vibeqc._vibeqc_core import Atom
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.methods.gfn2 import _compute_gfn2_d4

_SYM = dict(H=1,He=2,Li=3,Be=4,B=5,C=6,N=7,O=8,F=9,Ne=10,
            Na=11,Mg=12,Al=13,Si=14,P=15,S=16,Cl=17,Ar=18)

ADENINE_XYZ = """\
15
adenine C5H5N5 0 1
N     -1.4526600000    1.1333700000   -0.0191600000
C     -0.0068000000    1.1413400000   -0.0103700000
N      0.5805500000   -0.0368800000    0.0028500000
C     -0.2951200000   -1.1383700000   -0.0079800000
C     -1.6309200000   -0.9946700000    0.0041800000
C     -2.2955900000    0.2466100000   -0.0109400000
N      0.2262200000   -2.4097700000    0.0161900000
C      1.5619900000   -2.5532700000    0.0391500000
N      2.3165500000   -1.4386200000   -0.0111800000
N     -2.5602300000   -2.0064100000   -0.0232100000
H     -2.0027200000    2.0116400000   -0.0327300000
H      0.7734100000    2.0822000000   -0.0080700000
H      2.0146900000   -3.5280400000    0.0775300000
H     -3.5690400000   -1.7647200000   -0.0236600000
H     -3.3768800000    0.2853700000   -0.0201000000
"""

XTB_REF = {
    "total": -27.593506373901,
    "scc": -27.946306919238,
    "iso_es": 0.048303138564,
    "aniso_es": 0.009043051785,
    "aniso_xc": 0.035014265098,
    "dispersion": -0.013063293986,
    "repulsion": 0.352800545337,
}

def make_adenine():
    ang2bohr = 1.8897259886
    lines = ADENINE_XYZ.strip().split("\n")
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4: continue
        Z = _SYM.get(parts[0], 0)
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(Atom(Z, (x * ang2bohr, y * ang2bohr, z * ang2bohr)))
    return Molecule(atoms, 0, 1)

def run_with_opts(label, mol, params, opts):
    print(f"\n--- {label} ---")
    try:
        result = _xtb.run_gfn2_xtb(mol, params, opts)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None
    print(f"  converged={result.converged}  n_iter={result.n_iter}")
    e_band0 = float(result.e_band0)
    e_es = float(result.e_scc)
    e_aes = float(result.e_aes)
    e_3rd = float(result.e_3rd)
    e_rep = float(result.e_repulsive)
    total = float(result.energy)
    print(f"  E_band0  = {e_band0:16.10f} Ha")
    print(f"  E_es     = {e_es:16.10f} Ha")
    print(f"  E_aes    = {e_aes:16.10f} Ha")
    print(f"  E_3rd    = {e_3rd:16.10f} Ha")
    print(f"  E_rep    = {e_rep:16.10f} Ha")
    print(f"  C++ total= {total:16.10f} Ha")
    if hasattr(result, 'vmp_rms'):
        print(f"  V_mp RMS = {float(result.vmp_rms):.6e} Ha")
        print(f"  dip RMS  = {float(result.shell_dip_rms):.6e} ea0")
    e_d4 = _compute_gfn2_d4(mol, params)
    print(f"  +D4      = {e_d4:16.10f} Ha")
    total_d4 = total + e_d4
    print(f"  total+D4 = {total_d4:16.10f} Ha = {total_d4*627.509:.4f} kcal/mol")
    if total_d4 is not None:
        delta = total_d4 - XTB_REF["total"]
        print(f"  Δ vs xtb = {delta:+.10f} Ha = {delta*627.509:+.4f} kcal/mol")
    return result

def main():
    mol = make_adenine()
    params = load_gfn2_params()

    print("=" * 72)
    print("GFN2-xTB Adenine: Default vs Faithful AES vs xtb 6.7.1")
    print("=" * 72)

    # 1. Default (ad-hoc shell AES)
    opts_def = _xtb.XTBSccOptions()
    opts_def.max_iter = 3600
    run_with_opts("1. Default (ad-hoc shell AES)", mol, params, opts_def)

    # 2. Faithful AES
    opts_aes = _xtb.XTBSccOptions()
    opts_aes.max_iter = 3600
    opts_aes.aes_faithful = True
    run_with_opts("2. Faithful AES (atom-resolved)", mol, params, opts_aes)

    # 3. xtb reference
    print(f"\n--- 3. xtb 6.7.1 reference ---")
    for k, v in XTB_REF.items():
        print(f"  {k:14s} = {v:16.10f} Ha")


if __name__ == "__main__":
    main()
