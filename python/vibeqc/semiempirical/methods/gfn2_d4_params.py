"""Pinned, external LGPL GFN2 D4 reference data; no upstream code is executed.

The published polarizabilities are converted with the partner subtraction and
Casimir-Polder quadrature of Caldeweyher et al., JCP 150, 154122 (2019), Eq. 5.
GFN2 reference charges and covalent CNs follow Bannwarth et al., JCTC 15, 1652
(2019), section 2.2.6. The source and both license texts remain in the cache.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
import math
import os
from pathlib import Path
import re
import tempfile
from functools import lru_cache

import numpy as np

REVISION = "26b28010e805f7d1aeeef39813feb473e69cc4be"
SOURCES = {
    "include/param_ref.fh": "c3ff85a51151aeb9488aa0d3e833d483a1d50aa0885433befe6f8a8067bbb238",
    "src/disp/ncoord.f90": "c63839e89e5cca2f981af23058f888acf1377e5eb5cad005cd8d86f7004f0eb0",
    "COPYING": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "COPYING.LESSER": "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118",
}


def cache_directory():
    override = os.environ.get("VIBEQC_GFN2_D4_CACHE_DIR")
    return (Path(override).expanduser() if override else
            Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            / "vibeqc" / "gfn2-d4" / REVISION)


def _source_bytes(allow_fetch):
    root = cache_directory()
    flag = os.environ.get("VIBEQC_GFN2_OFFLINE")
    offline = (flag is not None and flag.lower() in ("1", "true", "yes")) or (
        flag is None and bool(os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID")))
    records = {}
    for relative, digest in SOURCES.items():
        path = root / Path(relative).name
        data = path.read_bytes() if path.is_file() else b""
        if hashlib.sha256(data).hexdigest() != digest:
            if not allow_fetch or offline:
                raise RuntimeError(f"GFN2 D4 cache missing or invalid: {path.name}; "
                                   "seed load_gfn2_d4_reference_data(allow_fetch=True) "
                                   "on a connected host before offline execution")
            from urllib.request import urlopen
            url = f"https://raw.githubusercontent.com/grimme-lab/xtb/{REVISION}/{relative}"
            try:
                with urlopen(url, timeout=30) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("upstream checksum mismatch")
            except Exception as exc:
                raise RuntimeError(f"GFN2 D4 refresh failed for {path.name}; "
                                   "existing cache preserved") from exc
            root.mkdir(parents=True, exist_ok=True)
            name = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                    name = stream.name
                    stream.write(data)
                Path(name).replace(path)
            finally:
                if name is not None:
                    Path(name).unlink(missing_ok=True)
        records[relative] = data
    return records


def _numbers(text):
    return [float(x.replace("D", "e").replace("d", "e")) for x in
            re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?",
                       text.replace("_wp", ""))]


@lru_cache(maxsize=2)
def _convert(raw):
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset, _PerElement
    from vibeqc.dispersion_d4_model import zeta_charge_scaling, zeta_hardness, effective_nuclear_charge

    text = re.sub(r"!.*", "", raw.decode())
    fields = {}
    for name, indices, values in re.findall(r"\bdata\s+(\w+)\s*\(([^)]*)\)\s*/([^/]*)/", text, re.I):
        key = tuple(int(i) for i in indices.replace(":", "").split(",") if i.strip())
        fields[name.lower(), key] = np.array(_numbers(values))
    def value(name, *key):
        return fields[name, key]
    freq = np.array([1e-6,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.,1.2,1.4,1.6,1.8,2.,2.5,3.,4.,5.,7.5,10.])
    widths = np.diff(freq)
    weights = (np.r_[widths, 0.] + np.r_[0., widths]) * (1.5/np.pi)
    elements = {}
    for z in sorted(key[0] for name, key in fields if name == "refn"):
        n = int(value("refn", z)[0])
        if not n:
            continue
        nominal = [int(math.floor(value("refcn", i, z)[0]+.5)) for i in range(1,n+1)]
        counts = [nominal.count(c)+(c==0) for c in nominal]
        alpha, charges, cns = [], [], []
        for i in range(1,n+1):
            partner = int(value("refsys", i, z)[0])
            zeff = effective_nuclear_charge(partner)
            # GFN2's secondary references are neutral (not their stored secq).
            scale = zeta_charge_scaling(3., 2.*zeta_hardness(partner),
                                        zeff, value("refh", i, z)[0]+zeff)
            secondary = value("sscale", partner)[0]*value("secaiw", partner)*scale
            alpha.append(np.maximum(value("ascale", i, z)[0]*(
                value("alphaiw", i, z)-value("hcount", i, z)[0]*secondary), 0.))
            charges.append(value("refq", i, z)[0])
            cns.append(value("refcovcn", i, z)[0])
        a = np.array(alpha)
        elements[z] = _PerElement(z, cns, charges, a, (a*weights)@a.T,
                                   [c*(c+1)//2 for c in counts])
    cross = {(a,b):(ea.alpha_iw*weights)@eb.alpha_iw.T
             for a,ea in elements.items() for b,eb in elements.items() if a<b}
    return D4ReferenceDataset(elements=elements, c6_cross=cross)


def load_gfn2_d4_reference_data(*, allow_fetch=True):
    """Validate every pinned source/license before returning parsed data."""
    records = _source_bytes(allow_fetch)
    dataset = deepcopy(_convert(records["include/param_ref.fh"]))
    dataset._vibeqc_source_sha256 = SOURCES["include/param_ref.fh"]
    # Portable input identity, distinct from the exact converted float records.
    # Include the CN source and license texts, and hash the validated bytes.
    manifest = {
        "revision": REVISION,
        "sources": {name: hashlib.sha256(raw).hexdigest()
                    for name, raw in records.items()},
    }
    dataset._vibeqc_source_revision = REVISION
    dataset._vibeqc_sources_sha256 = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return dataset


@lru_cache(maxsize=2)
def _cn_constants(raw):
    text = re.sub(r"!.*", "", raw.decode())
    def array(name):
        body = re.search(r"\b"+name+r"\(max_elem\)\s*=\s*\(/(.*?)/\)", text, re.S).group(1)
        return np.array([0.] + _numbers(body))
    return array("rad")*(4./3.)/.52917726, array("en")


def coordination_numbers_and_jacobian(mol):
    """Published electronegativity-weighted erf CN and exact derivative.

    Caldeweyher et al. (2019), Eq. 6; no EEQ logarithmic CN saturation.
    The Jacobian has shape (3*N, N), matching the existing SCC operands.
    """
    records = _source_bytes(False)
    radii, en = _cn_constants(records["src/disp/ncoord.f90"])
    xyz = np.array([a.xyz for a in mol.atoms]); zs = [a.Z for a in mol.atoms]
    cn = np.zeros(len(zs)); jac = np.zeros((len(zs),3,len(zs)))
    for a,za in enumerate(zs):
        for b,zb in enumerate(zs[:a]):
            vector = xyz[a]-xyz[b]; r = float(np.linalg.norm(vector))
            if r <= 1e-10:
                raise ValueError("GFN2 D4 requires distinct atom positions")
            if r > 40.:
                continue
            rc = radii[za]+radii[zb]; t=7.5*(r/rc-1.)
            factor = 4.10451*math.exp(-(abs(en[za]-en[zb])+19.08857)**2/(2.*11.28174**2))
            count = .5*factor*math.erfc(t)
            derivative = -factor*7.5/(math.sqrt(math.pi)*rc)*math.exp(-t*t)*vector/r
            cn[a]+=count;cn[b]+=count
            for owner in (a,b):
                jac[a,:,owner]+=derivative; jac[b,:,owner]-=derivative
    return cn, jac.reshape(-1,len(zs))
