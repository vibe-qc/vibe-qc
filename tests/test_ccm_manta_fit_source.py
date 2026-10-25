"""The MANTA fit source of the neutral real-Gamma torus (#519, RG-8a).

``MANTAFitSource`` builds the per-channel fits of the fold from MANTA's
periodic GDF tiles and its fixed RI-Q factor, on the domains MANTA's own
fitted RI HF builds for the same BvK mesh. These rows pin, on the H2 6-bohr
cube STO-3G at (2,1,1) and (3,1,1):

* the block storage on those fits gives J and K equal to MANTA's whole-mesh
  ``PeriodicRIFockBuilder`` on the same domains, to rounding, for arbitrary
  PSD spin densities (the (3,1,1) mesh keeps one representative of each +-q
  pair, so the block's transpose reconstruction of the partner channel is
  exercised against MANTA's explicit pair sources);
* the dense fold of the same fits gives the block storage's J and K (one
  auxiliary frame per channel, the property the fold's cross-bra products
  and the correlated consumers need);
* the source records its engine, its channels and their finite supports,
  and the real-Gamma SCF converges on it.

The finite source here (pair cutoff 6 bohr, auxiliary images to 12 bohr,
a 12 Ha reciprocal sphere, MANTA's default fitting omega) is chosen so the
fitting metric is positive definite; it is not an accuracy setting. The
budget is explicit and uncapped because MANTA's domain factory admits a
whole RI HF run's upper bound before building the domains.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

manta = pytest.importorskip("vibeqc.manta")

from vibeqc._vibeqc_core import BasisSet  # noqa: E402
from vibeqc.aux_basis import default_aux_for, make_aux_basis_set  # noqa: E402
from vibeqc.periodic.ccm.fit_source import (  # noqa: E402
    CLASSIC_ENGINE,
    MANTA_ENGINE,
    MANTAFitSource,
)
from vibeqc.periodic.ccm.neutral import (  # noqa: E402
    ccm_neutral_cderi_blocks,
    ccm_neutral_cderi_fold,
)
from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral  # noqa: E402

CUTOFF_HA = 12.0


def _cell():
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _options():
    o = manta.PeriodicRIRHFOptions()
    o.pair_cutoff_bohr = o.fitting.pair_cutoff_bohr = 6.0
    o.fitting.auxiliary_cutoff_bohr = 12.0
    return o


def _budget():
    b = manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets",
                 "scalar_terms", "kernel_terms"):
        setattr(b, name, 2 ** 62)
    return b


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module", params=[(2, 1, 1), (3, 1, 1)],
                ids=lambda m: "x".join(map(str, m)))
def torus(request):
    nrep = request.param
    ccm = CCMSystem(_cell(), nrep, "sto-3g")
    block = ccm_neutral_cderi_blocks(
        ccm, fit_source=MANTAFitSource(_options(), CUTOFF_HA, budget=_budget()))
    return ccm, block


def _psd_torus_densities(block, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        c = rng.normal(size=(block.n_sc, 2))
        out.append(0.3 * (c @ c.T))
    return out


def _manta_whole_mesh(ccm, block):
    """MANTA's own whole-mesh RI Fock on the domains its RHF builds."""
    cell = ccm.unit_system
    mol = cell.unit_cell_molecule()
    orb = BasisSet(mol, "sto-3g")
    aux = make_aux_basis_set(mol, aux_name=default_aux_for("sto-3g"), drop_eta=0.0)
    nrep = tuple(int(n) for n in ccm.nrep)
    mesh = manta.RegularKMesh(nrep, (0, 0, 0))
    nk = int(np.prod(nrep))
    options, budget = _options(), _budget()
    domains = manta.build_periodic_ri_rhf_domains(
        orb, aux, cell, mesh, CUTOFF_HA, options, budget)
    recip = np.asarray(cell.reciprocal_lattice(), dtype=float)
    frac = np.array([mesh.fractional(mesh.address(i)) for i in range(nk)])
    fock = manta.PeriodicRIFockBuilder(
        orb, aux, cell, momenta=np.ascontiguousarray(frac @ recip.T),
        weights=np.full(nk, 1.0 / nk),
        gamma_vectors=np.asarray(domains.reciprocal_vectors),
        pair_vectors=np.asarray(domains.pair_vectors),
        transfer_offsets=np.asarray(domains.transfer_offsets),
        options=options.fitting, ri_options=options.ri, budget=budget)
    nvec = np.asarray(nrep)
    manta_labels = [tuple(int(x) for x in np.round(f * nvec).astype(int) % nvec)
                    for f in frac]
    fold_labels = [tuple(int(x) for x in block.frac[i] % nvec) for i in range(nk)]
    to_manta = [manta_labels.index(label) for label in fold_labels]
    return fock, to_manta, int(orb.nbasis)


def test_block_jk_on_manta_fits_is_mantas_whole_mesh_fock(torus):
    ccm, block = torus
    fock, to_manta, n = _manta_whole_mesh(ccm, block)
    nk = block.n_c
    d_alpha, d_beta = _psd_torus_densities(block, 519)
    j_block = block.bloch(block.torus_blocks(block.coulomb(d_alpha + d_beta)))
    k_block = [block.bloch(block.torus_blocks(block.exchange(d)))
               for d in (d_alpha, d_beta)]
    spin = np.zeros((2, nk, n, n), dtype=complex)
    for s, d in enumerate((d_alpha, d_beta)):
        dk = block.density_bloch(d)
        for i in range(nk):
            spin[s, to_manta[i]] = dk[i]
    ref = fock.build(hcore=np.zeros((nk, n, n), dtype=complex),
                     spin_densities=np.ascontiguousarray(spin),
                     exchange_scale=1.0)
    J = np.asarray(ref.J).reshape(nk, n, n)
    K = np.asarray(ref.K).reshape(2, nk, n, n)
    for i in range(nk):
        m = to_manta[i]
        assert np.max(np.abs(j_block[i] - J[m])) < 1e-12 * max(1.0, np.max(np.abs(J[m])))
        for s in (0, 1):
            assert np.max(np.abs(k_block[s][i] - K[s, m])) < 1e-12 * max(1.0, np.max(np.abs(K[s, m])))


def test_dense_fold_of_manta_fits_is_the_block_storage(torus):
    """One auxiliary frame per channel: the fold's cross-bra products of the
    MANTA fits give the same J and K as the block storage."""
    ccm, block = torus
    dense = ccm_neutral_cderi_fold(
        ccm, fit_source=MANTAFitSource(_options(), CUTOFF_HA, budget=_budget()))
    np.testing.assert_allclose(block.to_dense(), dense, atol=1e-13, rtol=0)
    d, _ = _psd_torus_densities(block, 4)
    # The circulant projection is what the block storage contracts.
    d = block.embed(block.torus_blocks(d))
    j = ccm_ri_j_neutral(dense, d)
    k = ccm_ri_k_neutral(dense, d)
    # Two summation orders of one contraction: rounding, relative 1e-11.
    for a, b in ((j, block.coulomb(d)), (k, block.exchange(d))):
        assert np.max(np.abs(a - b)) < 1e-11 * max(1.0, np.max(np.abs(a)))


def test_the_source_records_its_engine_and_channels(torus):
    ccm, block = torus
    receipt = block.info["fit_source"]
    assert receipt["provider"] == "manta" and receipt["engine"] == MANTA_ENGINE
    assert receipt["engine"] != CLASSIC_ENGINE
    assert receipt["reciprocal_energy_cutoff_ha"] == CUTOFF_HA
    assert receipt["fitting"]["auxiliary_cutoff_bohr"] == 12.0
    labels = [tuple(c["label"]) for c in receipt["channels"]]
    assert labels == sorted(tuple(int(x) for x in block.frac[ch.qi])
                            for ch in block.channels)
    for record in receipt["channels"]:
        assert record["rank"] > 0 and record["vectors"] > 0
        assert record["source_identity"] and record["projector_identity"]


def test_the_real_gamma_scf_converges_on_manta_fits(torus):
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct

    ccm, block = torus
    res = _quiet(run_ccm_rhf_direct, ccm, cderi=block)
    assert res.converged and res.cderi_info["storage"] == "block"
    assert res.energy / ccm.n_cells < 0.0


def test_the_source_refuses_what_it_cannot_build():
    ccm = CCMSystem(_cell(), (2, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="reciprocal_energy_cutoff"):
        MANTAFitSource(_options(), 0.0)
    with pytest.raises(TypeError, match="MANTA periodic RI HF options"):
        MANTAFitSource(object(), CUTOFF_HA)
    with pytest.raises(ValueError, match="star plan"):
        ccm_neutral_cderi_fold(
            ccm, fit_source=MANTAFitSource(_options(), CUTOFF_HA, budget=_budget()),
            symmetry=True)
    with pytest.raises(NotImplementedError, match="screened"):
        ccm_neutral_cderi_fold(
            ccm, fit_source=MANTAFitSource(_options(), CUTOFF_HA, budget=_budget()),
            omega_screen=0.11)
