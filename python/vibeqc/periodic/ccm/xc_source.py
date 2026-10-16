"""XC sources of the neutral real-Gamma torus (#519, RG-8c).

The real-Gamma KS drivers evaluate ``E_xc`` and ``V_xc`` with vibe-qc's
periodic Becke-grid builders folded onto the torus. A MANTA-sourced torus
(fits from :class:`~vibeqc.periodic.ccm.fit_source.MANTAFitSource`, the
one-electron side from
:class:`~vibeqc.periodic.ccm.one_electron_source.MANTAOneElectronSource`)
also needs MANTA's semilocal XC quadrature, or it mixes two finite sources.
:class:`MANTAXCSource` supplies it: MANTA's ``compute_periodic_semilocal_xc``
on the XC image domain of MANTA's fitted KS for the same BvK mesh, at the
torus k-points and the caller's quadrature, mapped onto the torus.

Conventions, read from MANTA's reference implementation
(``tests/test_manta_periodic_semilocal_xc.py``): the Bloch AO
``phi^k_i(r) = sum_T e^{ik.T} phi_i(r - T)``, the cell density
``rho = sum_k w_k phi^k D(k) phi^k^H`` with ``w_k = 1/N_c``, the potential
``V_ij(k) = <phi^k_i| v_xc |phi^k_j>`` and the cell energy ``E_cell``. With
the torus block row ``D(g) = (1/N_c) sum_c D[(c), (c+g)]`` this is
``D(k) = sum_g e^{+ik.R_g} D(g)``, the convention of the block storage, and
``dE_torus / dD`` is the torus matrix of
``V(g) = Re (1/N_c) sum_k e^{-ik.R_g} V(k)``. The total torus energy is
``N_c E_cell``.

An XC source answers ``closed_shell(ccm, functional)`` with the RKS drivers'
closure ``D -> (E_xc, V_xc)`` and ``open_shell(ccm, functional)`` with the UKS
closure ``(Da, Db) -> (E_xc, Va, Vb)``, both on supercell matrices, and
``record()`` with the JSON-able record that joins the neutral-source identity
(RG-2) under ``xc``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["MANTA_XC_ENGINE", "MANTAXCSource"]

#: Engine identity of MANTA's semilocal XC quadrature.
MANTA_XC_ENGINE = "manta-periodic-semilocal-xc"


def _torus_phases(lattice, nrep, kpts_frac):
    """``E[k, g] = exp(+i k.R_g)`` over the CCM cell-major cell order."""
    n1, n2, n3 = (int(n) for n in nrep)
    cells = np.array([(i, j, k) for i in range(n1) for j in range(n2)
                      for k in range(n3)], dtype=float)
    # k.R = 2 pi frac . n for R = A n and k = 2 pi A^{-T} frac.
    return np.exp(2j * np.pi * (np.asarray(kpts_frac, dtype=float) @ cells.T))


def _block_row(nrep, n_mu, M):
    """``M(g) = (1/N_c) sum_c M[(c), (c+g)]`` over the cell-major order."""
    n1, n2, n3 = (int(n) for n in nrep)
    nc = n1 * n2 * n3
    cells = np.array([(i, j, k) for i in range(n1) for j in range(n2)
                      for k in range(n3)])
    plus = (((cells[:, None, 0] + cells[None, :, 0]) % n1) * n2
            + (cells[:, None, 1] + cells[None, :, 1]) % n2) * n3 \
        + (cells[:, None, 2] + cells[None, :, 2]) % n3
    m4 = np.asarray(M).reshape(nc, n_mu, nc, n_mu)
    rows = m4[np.arange(nc)[:, None], :, plus, :]          # (c, g, mu, nu)
    return rows.mean(axis=0)


def _embed(nrep, blocks):
    """Block-circulant torus matrix ``M[(c), (c')] = M(c' - c)``."""
    n1, n2, n3 = (int(n) for n in nrep)
    nc = n1 * n2 * n3
    cells = np.array([(i, j, k) for i in range(n1) for j in range(n2)
                      for k in range(n3)])
    diff = (((cells[None, :, 0] - cells[:, None, 0]) % n1) * n2
            + (cells[None, :, 1] - cells[:, None, 1]) % n2) * n3 \
        + (cells[None, :, 2] - cells[:, None, 2]) % n3
    blocks = np.asarray(blocks)
    n_mu = blocks.shape[-1]
    return np.ascontiguousarray(
        blocks[diff].transpose(0, 2, 1, 3).reshape(nc * n_mu, nc * n_mu))


class MANTAXCSource:
    """MANTA's semilocal XC on the torus, for the real-Gamma KS drivers.

    ``options`` (``manta.PeriodicRIKSOptions``) and
    ``reciprocal_energy_cutoff`` are those of the matching MANTA fitted KS
    run (``manta.run_periodic_ri_ks_from_cutoffs`` / its UKS-capable
    sibling), ``xc_image_cutoff_bohr`` its XC image cutoff and ``points`` /
    ``quadrature_weights`` its caller-supplied quadrature (Cartesian bohr,
    weights summing to the cell volume). The XC image domain is MANTA's
    (``build_periodic_ri_ks_domains``). Semilocal functionals only: a hybrid
    keeps its exact exchange on the torus fits, and MANTA's semilocal
    component is the XC part.
    """

    engine = MANTA_XC_ENGINE

    def __init__(self, options, reciprocal_energy_cutoff, xc_image_cutoff_bohr,
                 points, quadrature_weights, *, auxiliary_basis=None,
                 budget=None, grid_batch=64):
        cutoff = float(reciprocal_energy_cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "MANTAXCSource: reciprocal_energy_cutoff must be finite and "
                f"positive, got {reciprocal_energy_cutoff!r}")
        reach = float(xc_image_cutoff_bohr)
        if not np.isfinite(reach) or reach <= 0.0:
            raise ValueError(
                "MANTAXCSource: xc_image_cutoff_bohr must be finite and "
                f"positive, got {xc_image_cutoff_bohr!r}")
        for name in ("fitting", "ri", "ks"):
            if not hasattr(options, name):
                raise TypeError(
                    "MANTAXCSource: options must be MANTA periodic RI KS "
                    f"options (missing {name!r}); got {type(options).__name__}")
        pts = np.ascontiguousarray(points, dtype=float)
        qw = np.ascontiguousarray(quadrature_weights, dtype=float).ravel()
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] != qw.size:
            raise ValueError(
                "MANTAXCSource: points must be (n, 3) and quadrature_weights "
                f"(n,); got {pts.shape} and {qw.shape}")
        self.options = options
        self.reciprocal_energy_cutoff = cutoff
        self.xc_image_cutoff_bohr = reach
        self.points = pts
        self.quadrature_weights = qw
        self.auxiliary_basis = auxiliary_basis
        self.budget = budget
        self.grid_batch = int(grid_batch)
        self._record = None

    # ------------------------------------------------------------------
    def _bind(self, ccm, functional, spin):
        from vibeqc import manta
        from vibeqc._vibeqc_core import BasisSet, Functional
        from vibeqc.aux_basis import default_aux_for, make_aux_basis_set

        unit = ccm.unit_system
        if int(unit.dim) != 3:
            raise NotImplementedError(
                "MANTAXCSource: MANTA's periodic XC sources are 3-D")
        name = functional if isinstance(functional, str) else functional.name
        func = Functional(str(name), int(spin))
        if float(func.hf_exchange_fraction) != 0.0 or bool(
                getattr(func, "is_range_separated", False)):
            raise NotImplementedError(
                "MANTAXCSource: semilocal functionals only; a hybrid's exact "
                "exchange and its semilocal remainder are not split here")
        mol = unit.unit_cell_molecule()
        orbital = BasisSet(mol, ccm.basis_name)
        aux_name = str(self.auxiliary_basis or default_aux_for(ccm.basis_name))
        auxiliary = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=0.0)
        budget = self.budget if self.budget is not None else manta.ResourceBudget()
        nrep = tuple(int(n) for n in ccm.nrep)
        mesh = manta.RegularKMesh(nrep, (0, 0, 0))
        domains = manta.build_periodic_ri_ks_domains(
            orbital, auxiliary, unit, mesh, func, self.reciprocal_energy_cutoff,
            self.xc_image_cutoff_bohr, int(self.points.shape[0]), self.options,
            budget)
        images = np.ascontiguousarray(domains.xc_images, dtype=float)
        nk = int(np.prod(nrep))
        frac = np.array([mesh.fractional(mesh.address(i)) for i in range(nk)],
                        dtype=float)
        lattice = np.ascontiguousarray(unit.lattice, dtype=float)
        momenta = np.ascontiguousarray(
            2.0 * np.pi * (frac @ np.linalg.inv(lattice)), dtype=float)
        weights = np.full(nk, 1.0 / nk)
        phases = _torus_phases(lattice, nrep, frac)            # (k, g)
        n_mu = int(orbital.nbasis)
        if n_mu * nk != int(ccm.basis.nbasis):
            raise ValueError(
                "MANTAXCSource: the unit-cell basis does not tile the torus "
                f"basis ({n_mu} x {nk} != {int(ccm.basis.nbasis)})")
        self._record = {
            "provider": "manta",
            "engine": self.engine,
            "functional": str(func.name),
            "reciprocal_energy_cutoff_ha": float(self.reciprocal_energy_cutoff),
            "xc_image_cutoff_bohr": float(self.xc_image_cutoff_bohr),
            "xc_images": int(len(images)),
            "quadrature_points": int(self.points.shape[0]),
            "quadrature_weight_sum": float(np.sum(self.quadrature_weights)),
        }

        def evaluate(per_spin_torus):
            densities = np.stack([
                np.tensordot(phases, _block_row(nrep, n_mu, d), axes=(1, 0))
                for d in per_spin_torus])                     # (s, k, mu, nu)
            result = manta.compute_periodic_semilocal_xc(
                orbital, func, lattice, images, momenta, weights,
                self.points, self.quadrature_weights,
                np.ascontiguousarray(densities, dtype=complex),
                self.grid_batch, budget)
            values = np.asarray(result.values)                # (s, k, mu, nu)
            out = []
            for v in values:
                blocks = np.tensordot(np.conj(phases).T, v, axes=(1, 0)) / nk
                out.append(_embed(nrep, np.ascontiguousarray(blocks.real)))
            return float(result.receipt.energy) * nk, out, result

        return evaluate

    # ------------------------------------------------------------------
    def closed_shell(self, ccm, functional):
        """The RKS closure ``D -> (E_xc, V_xc)`` on the torus."""
        evaluate = self._bind(ccm, functional, 1)

        def xc_of_D(D):
            e_xc, (v,), _ = evaluate([np.asarray(D, dtype=float)])
            return e_xc, 0.5 * (v + v.T)

        xc_of_D.xc_lattice_cutoff_bohr = float(self.xc_image_cutoff_bohr)
        xc_of_D.torus_grouping = False
        xc_of_D.torus_grouping_used = False
        xc_of_D.xc_source = self
        return xc_of_D

    def open_shell(self, ccm, functional):
        """The UKS closure ``(Da, Db) -> (E_xc, Va, Vb)`` on the torus."""
        evaluate = self._bind(ccm, functional, 2)

        def xc_of_spins(Da, Db):
            e_xc, (va, vb), _ = evaluate([np.asarray(Da, dtype=float),
                                          np.asarray(Db, dtype=float)])
            return e_xc, 0.5 * (va + va.T), 0.5 * (vb + vb.T)

        xc_of_spins.xc_lattice_cutoff_bohr = float(self.xc_image_cutoff_bohr)
        xc_of_spins.torus_grouping = False
        xc_of_spins.torus_grouping_used = False
        xc_of_spins.xc_source = self
        return xc_of_spins

    def record(self) -> dict:
        """The JSON-able record of the bound source (after a bind)."""
        if self._record is None:
            raise RuntimeError("MANTAXCSource.record: the source is not bound")
        return dict(self._record)
