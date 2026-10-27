"""Finite Gamma pair-density Hartree operator, explicitly opted in.

Lippert/Hutter/Parrinello, Mol. Phys. 92, 477 (1997), density/Poisson/
projection construction; Sharma/Beylkin (2021) physical pair support.
Density and potential use an adjoint pair transform on the same FFT grid.
No default migration, infinite-lattice convergence or external parity claim.
"""
import hashlib
import numpy as np
from . import _vibeqc_core as core
from .bipole_domain import OptionSnapshot, LATTICE_FIELDS


def resolve_options(options, *, functional=None, omega=0.0, cache=None):
    if options is None:
        return None
    saved = OptionSnapshot.capture(options, LATTICE_FIELDS)
    result = saved.restore(core.LatticeSumOptions)
    if not np.isfinite(result.cutoff_bohr) or result.cutoff_bohr <= 0:
        raise ValueError("GPW pair cutoff must be finite and positive")
    if result.pair_complete_1e:
        if omega != 0 or cache is not None:
            raise NotImplementedError("Physical pair-domain GPW requires matched range-separated projectors and cannot use a historical AO collocation cache")
        if functional is not None:
            func = core.Functional(functional, 1)
            if func.is_external or func.kind == core.XCKind.MGGA or float(func.rsh_omega) != 0.0:
                raise NotImplementedError("Physical pair-domain GPW currently supports local/GGA/global-hybrid XC; external, meta-GGA and RSH need matched projectors")
    return result


class PairDomainJBuilder:
    """Bounded-memory pair-FT density and its discrete adjoint projection."""
    _mpi_aware = False

    def __init__(self, basis, system, grid, options, *, exchange=False, exchange_alpha=.5):
        self.basis, self.grid, self.system = basis, grid, system
        self.options = OptionSnapshot.capture(options, LATTICE_FIELDS)
        self.cutoff = float(options.cutoff_bohr)
        overlap = core.compute_overlap_lattice(basis, system, options)
        self.shifts = np.asarray([list(c.r_cart) for c in overlap.cells], dtype=float).reshape(-1, 3)
        if not len(self.shifts):
            raise ValueError("Physical GPW requires nonempty AO-pair cells")
        self.shifts.setflags(write=False)
        self.exchange = None
        if exchange:
            from .periodic_corrected_exchange import CorrectedEwaldExchange
            self.overlap = np.asarray(core.bloch_sum(overlap, np.zeros(3))).real.copy()
            self.overlap.setflags(write=False)
            self.exchange = CorrectedEwaldExchange.build(
                basis, system, self.shifts, core.monkhorst_pack(system, [1, 1, 1], [0, 0, 0], False),
                float(exchange_alpha), where="physical Gamma GPW", lattice_opts=options)
            # Populate the only Gamma channel before capturing the source.
            self.exchange.k_space_terms_at_k(0, self.overlap, [np.zeros_like(self.overlap)])

    def source_identity(self):
        return dict(kind="finite-physical-pair-ft", options=self.options.values,
                    cutoff=self.cutoff, shifts=hashlib.sha256(self.shifts.tobytes()).hexdigest(),
                    exchange=self.exchange_identity())

    def exchange_identity(self):
        if self.exchange is None:
            return None
        x = self.exchange
        cache = x.k_cache.j_cache
        digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        return dict(kernel="corrected-ewald", alpha=x.alpha, g0=x.exchange_g0,
            mesh=x.mesh, weights=tuple(float(v) for v in x.weights),
            points=[digest(v) for v in x.k_points], overlap=digest(self.overlap),
            reciprocal=digest(cache.K_vectors), kernel_values=digest(cache.kernel),
            pair_ft=digest(cache.ft_per_cell),
            bloch_cache=sorted((k.hex(), digest(v)) for k, v in cache.ft_bloch_cache.items()))

    def build_K(self, density):
        """Reuse convergent SR + reciprocal LR + probe-charge q=0 correction.

        A Gamma density repeats on every image: bare real-space Coulomb
        exchange diverges even when its physical quartet domain is exact.
        Gygi/Baldereschi (1986); Carrier/Rohra/Goerling (2007). Reuse the
        existing periodic implementation, never subtract an empirical shift.
        """
        if self.exchange is None:
            raise ValueError("Physical GPW exchange was not prepared")
        options = self.options.restore(core.LatticeSumOptions)
        short = np.asarray(core.build_jk_gamma_molecular_limit(
            self.basis, self.system, options, density, omega=self.exchange.alpha,
            adjoint_symmetric=True).K)
        result = short + self.exchange.k_space_terms_at_k(0, self.overlap, [density])
        if np.max(np.abs(np.asarray(result).imag)) > 1e-10:
            raise ValueError("Gamma corrected exchange has a non-real residual")
        return .5 * (result.real + result.real.T)

    def _panels(self):
        from ._aopair_ft import ao_pair_fourier_transform_bloch
        from .aux_basis import _ao_scales_for_rsgdf
        g = self.grid.reciprocal_vectors().reshape(-1, 3)
        scales = _ao_scales_for_rsgdf(self.basis)
        pair_scales = np.outer(scales, scales)
        chunk = max(1, (16 * 1024 * 1024) // (16 * max(1, self.basis.nbasis ** 2)))
        for start in range(0, len(g), chunk):
            stop = min(start + chunk, len(g))
            pair = ao_pair_fourier_transform_bloch(self.basis, g[start:stop],
                self.shifts, np.zeros(3), pair_cutoff_bohr=self.cutoff)
            yield start, stop, pair * pair_scales[:, :, None]

    def density_on_grid(self, density):
        d = np.asarray(density)
        if np.iscomplexobj(d) or d.shape != (self.basis.nbasis, self.basis.nbasis) or not np.isfinite(d).all():
            raise ValueError("Physical Gamma GPW requires a finite real AO density")
        if not np.allclose(d, d.T, rtol=0, atol=1e-12):
            raise ValueError("Physical Gamma GPW requires symmetric density")
        rho_ft = np.empty(self.grid.n_points, dtype=complex)
        for start, stop, pair in self._panels():
            rho_ft[start:stop] = np.einsum("mn,mng->g", d, pair)
        # Real projection treats even-grid Nyquist modes identically on both
        # sides; no unmatched analytic reciprocal contraction is introduced.
        return np.fft.ifftn(rho_ft.reshape(self.grid.shape)).real / self.grid.voxel_volume_bohr3

    def project_potential(self, potential):
        potential = np.asarray(potential)
        if potential.shape != self.grid.shape or np.iscomplexobj(potential) or not np.isfinite(potential).all():
            raise ValueError("Physical GPW requires a finite real potential on its grid")
        coefficients = np.fft.fftn(potential).reshape(-1) / self.grid.n_points
        j = np.zeros((self.basis.nbasis, self.basis.nbasis), dtype=complex)
        for start, stop, pair in self._panels():
            j += np.einsum("g,mng->mn", coefficients[start:stop], pair.conj())
        return .5 * (j.real + j.real.T)

    def build_J(self, density):
        rho = self.density_on_grid(density)
        return self.project_potential(core.solve_poisson_coulomb(rho, self.grid.lattice_bohr))

    def scalar_xc(self, density, functional):
        """LDA/GGA energy and derivative on the same finite pair density.

        GGA integration by parts is the adjoint of the real FFT gradient.
        The existing libxc energy clips rho at zero; apply its local chain
        rule before projection. Sigma still uses the unclipped FFT gradient,
        matching _evaluate_xc_on_grid rather than silently changing energy.
        """
        from .periodic_gapw_j import _evaluate_xc_on_grid
        if functional.is_external or functional.kind == core.XCKind.MGGA or float(functional.rsh_omega) != 0.0:
            raise NotImplementedError("Physical pair scalar XC excludes external, meta-GGA and RSH")
        rho = self.density_on_grid(density)
        energy, vrho, vsigma, gradient, vtau = _evaluate_xc_on_grid(rho, self.grid, functional)
        if vtau is not None:
            raise NotImplementedError("Physical pair scalar XC cannot project kinetic density")
        potential = np.where(rho > 0., vrho, 0.)
        if vsigma is not None:
            flux = 2. * vsigma[..., None] * gradient
            reciprocal = self.grid.reciprocal_vectors()
            for axis in range(3):
                potential -= np.fft.ifftn(1j * reciprocal[..., axis] * np.fft.fftn(flux[..., axis])).real
        return energy, self.project_potential(potential)
