"""Explicit periodic BUPO job policies, independent of radial Ewald inputs."""
from dataclasses import dataclass, fields
import math
import numbers

from . import _vibeqc_core as core


@dataclass(frozen=True)
class PeriodicBupoIntegralOptions:
    """Fixed matched AO/Ewald cubes and evaluated-density HF accuracy targets.

    Cube radii are integer lattice labels. The physical AO pair radius is
    separately in bohr; reciprocal kinetic energy is in hartree. Extents,
    moments and hierarchy use CoulombBuildOptions.bupo. No automatic accuracy
    profile, source adaptation, RI fitting or symmetry authority is implied.
    """
    reciprocal_energy_cutoff: float
    pair_cube_radius: int = 1
    real_cube_radius: int = 4
    reciprocal_cube_radius: int = 10
    near_cube_radius: int = 0
    eta: float = .6
    pair_cutoff_bohr: float | None = None
    reciprocal_batch: int = 64
    madelung_alpha: float | None = None
    madelung_tolerance: float = 1e-12
    charge_tolerance: float = 1e-10
    maximum_electrostatic_energy_error: float = 1e-8
    maximum_electrostatic_matrix_error: float = 1e-8
    maximum_overlap_image_error: float = 1e-10
    maximum_kinetic_image_error: float = 1e-8
    maximum_kinetic_energy_error: float = 1e-8
    maximum_exchange_matrix_error: float = 1e-8
    maximum_exchange_energy_error: float = 1e-8
    confirmation_fock_tolerance: float = 1e-8
    confirmation_energy_tolerance: float = 1e-10

    def __post_init__(self):
        cubes = ('pair_cube_radius', 'real_cube_radius', 'reciprocal_cube_radius', 'near_cube_radius')
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if name in cubes or name == 'reciprocal_batch':
                minimum = 1 if name == 'reciprocal_batch' else 0
                if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
                        or not minimum <= value < 2**31-1):
                    raise ValueError(f'PeriodicBupoIntegralOptions.{name} requires a supported integer extent')
            else:
                if value is None and name in ('pair_cutoff_bohr', 'madelung_alpha'):
                    continue
                if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                        or not math.isfinite(value) or value < 0
                        or (name != 'pair_cutoff_bohr' and value == 0)):
                    raise ValueError(f'PeriodicBupoIntegralOptions.{name} requires a finite '
                                     + ('nonnegative value' if name == 'pair_cutoff_bohr' else 'positive value'))
        if self.near_cube_radius > self.real_cube_radius:
            raise ValueError('PeriodicBupoIntegralOptions near cube must lie inside the real Ewald cube')
        if self.madelung_tolerance >= 1:
            raise ValueError('PeriodicBupoIntegralOptions.madelung_tolerance must be below one')

    def _hf_options(self, method):
        options = core._manta.PeriodicBupoHFOptions()
        options.spin_mode = 'restricted' if method == 'RHF' else 'unrestricted'
        options.bound_exchange = True
        for field in fields(self):
            if hasattr(options, field.name):
                setattr(options, field.name, getattr(self, field.name))
        return options

    def _source_options(self, bupo):
        if not isinstance(bupo, core._manta.BupoOptions):
            raise TypeError('Periodic BUPO requires coulomb.bupo as manta.BupoOptions')
        if bupo.near_backend != 'manta' or bupo.collect_timings:
            raise NotImplementedError('Periodic BUPO requires near_backend=manta and collect_timings=False')
        source = core._manta.PeriodicBupoHFSourceOptions()
        for name in ('pair_cube_radius', 'real_cube_radius', 'reciprocal_cube_radius', 'near_cube_radius', 'eta'):
            setattr(source, name, getattr(self, name))
        source.pairs.order = bupo.order
        source.pairs.extents = bupo.extents
        source.pairs.pair_cutoff_bohr = self.pair_cutoff_bohr
        source.bubbles = bupo.bubbles
        source.bupo.opening = bupo.opening
        source.bupo.integral_tolerance = bupo.integral_tolerance
        return source
