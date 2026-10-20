"""Selected periodic MANTA jobs over admitted native HF and KS sources."""
from __future__ import annotations

import copy
import inspect
import math
import numbers
import os
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import _vibeqc_core as core
from .manta_scf import IntegralEngineOptions, CoulombBuildOptions, ExchangeBuildOptions
from . import manta_periodic
from .manta_periodic_bupo_job import PeriodicBupoIntegralOptions
from .manta_periodic_bupo import periodic_bupo_hf_provenance
from .manta_periodic import _periodic_hf_configuration


@dataclass(frozen=True)
class PeriodicFittingOptions:
    """Finite GDF source controls, independent of nuclear Ewald and physical K.

    Omega is the numerical inverse-bohr split, not a physical exchange range.
    The auxiliary image radius is in bohr. The AO pair radius comes from
    PeriodicIntegralOptions; metric controls come from CoulombBuildOptions.
    Coupling-only screening leaves that finite metric unchanged and is
    mutually exclusive with common integral screening; RI K is unsupported.
    """
    omega: float
    auxiliary_cutoff_bohr: float
    integral_screen_error: float = 0.
    image_candidate_cap: int = 1000000
    coupling_screen_error: float = 0.
    coupling_multipole_error: float = 0.
    coupling_multipole_order: int = 6
    coupling_multipole_hierarchy: bool = False
    coupling_multipole_reuse: bool = True
    coupling_multipole_compress: bool = False
    coupling_multipole_global_actions: bool = False
    coupling_multipole_partition_bytes: int = 64*1024*1024
    coupling_multipole_extents: object | None = None
    coupling_multipole_pair_tree: bool = False
    accuracy: object | None = None

    def __post_init__(self):
        if self.accuracy is not None:
            if not isinstance(self.accuracy,core._manta.PeriodicRIAccuracyOptions):
                raise TypeError('PeriodicFittingOptions.accuracy must be PeriodicRIAccuracyOptions or None')
            for name in ('j_error','energy_error','density_l1_limit'):
                value=getattr(self.accuracy,name)
                if not math.isfinite(value) or value<0 or (name!='density_l1_limit' and value==0):
                    raise ValueError('Automatic RI accuracy requires positive finite targets and nonnegative density_l1_limit')
            entries=self.accuracy.density_entry_limits
            if any(not math.isfinite(value) or value<0 for value in entries):
                raise ValueError('Automatic RI density entry limits must be finite nonnegative')
            if entries and self.accuracy.density_l1_limit==0:
                raise ValueError('Derived RI density limits cannot override entry limits')
            if not 1<=self.accuracy.maximum_attempts<=32 or not self.coupling_multipole_pair_tree:
                raise ValueError('Automatic RI accuracy requires one through 32 attempts and a pair-tree source')
        for name in ('omega', 'auxiliary_cutoff_bohr', 'integral_screen_error', 'coupling_screen_error', 'coupling_multipole_error'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value) or value < 0 or (name == 'omega' and value == 0)):
                raise ValueError(f'PeriodicFittingOptions.{name} must be finite '
                                 + ('positive' if name == 'omega' else 'nonnegative'))
        if self.coupling_multipole_error and (self.coupling_screen_error or self.integral_screen_error):
            raise ValueError('coupling_multipole_error cannot combine with screening')
        if (isinstance(self.coupling_multipole_order, bool)
                or not isinstance(self.coupling_multipole_order, numbers.Integral)
                or not 0 <= self.coupling_multipole_order <= 12):
            raise ValueError('coupling_multipole_order must be an integer from zero through twelve')
        for name in ('coupling_multipole_hierarchy', 'coupling_multipole_reuse', 'coupling_multipole_compress', 'coupling_multipole_global_actions', 'coupling_multipole_pair_tree'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} must be boolean')
        if self.coupling_multipole_pair_tree and (not self.coupling_multipole_reuse or self.coupling_multipole_error <= 0):
            raise ValueError('coupling_multipole_pair_tree requires positive multipole error and preparation reuse')
        if (isinstance(self.coupling_multipole_partition_bytes, bool)
                or not isinstance(self.coupling_multipole_partition_bytes, numbers.Integral)
                or not 0 <= self.coupling_multipole_partition_bytes <= 2**64-1):
            raise ValueError('coupling_multipole_partition_bytes must be a nonnegative uint64 integer')
        if self.coupling_multipole_extents is not None and not isinstance(self.coupling_multipole_extents, core._manta.ExtentOptions):
            raise TypeError('coupling_multipole_extents must be manta.ExtentOptions or None')
        if self.coupling_screen_error and self.integral_screen_error:
            raise ValueError('coupling_screen_error cannot combine with integral_screen_error')
        if (isinstance(self.image_candidate_cap, bool)
                or not isinstance(self.image_candidate_cap, numbers.Integral)
                or not 0 < self.image_candidate_cap <= 2**64-1):
            raise ValueError('PeriodicFittingOptions.image_candidate_cap must be a positive uint64 integer')


@dataclass(frozen=True)
class PeriodicIntegralOptions:
    """Finite 3D integral domains, independent of the selected J/K algorithms.

    All three cutoffs are explicit. Pair/nuclear radii are in bohr and the
    reciprocal kinetic energy is in hartree. The job consumers are
    fixed-occupation RHF, UHF, RKS and UKS. Madelung controls apply to HF and
    the resolved long-range Coulomb part of hybrid KS exchange, independently
    of the semilocal XC grid. Finite erfc zero modes receive no correction.
    """
    reciprocal_energy_cutoff: float
    pair_cutoff_bohr: float
    nuclear_cutoff_bohr: float
    alpha: float = .7
    reciprocal_batch: int = 64
    madelung_alpha: float | None = None
    madelung_tolerance: float = 1e-12
    one_electron_accuracy: object | None = None
    exchange_accuracy: object | None = None
    nuclear_accuracy: object | None = None
    support_selection: object | None = None

    def __post_init__(self):
        if self.support_selection is not None:
            if not isinstance(self.support_selection, core._manta.PeriodicSupportSelectionOptions):
                raise TypeError('PeriodicIntegralOptions.support_selection must be PeriodicSupportSelectionOptions or None')
            if self.one_electron_accuracy is None or self.nuclear_accuracy is None:
                raise ValueError('Automatic supports require both one-electron and nuclear accuracy targets')
            policy = self.support_selection
            if not 1 <= policy.maximum_attempts <= 1024 or not 1 <= policy.maximum_completion_cube_radius <= 2147483647:
                raise ValueError('Automatic supports require 1..1024 attempts and a positive supported cube limit')
            if not math.isfinite(policy.growth_factor) or not 1 < policy.growth_factor <= 4:
                raise ValueError('Automatic support growth factor must be in (1,4]')
            for name in ('maximum_pair_cutoff_bohr', 'maximum_nuclear_cutoff_bohr', 'maximum_reciprocal_energy_cutoff'):
                value = getattr(policy, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('Automatic support limits must be finite positive')
        if self.nuclear_accuracy is not None:
            if not isinstance(self.nuclear_accuracy, core._manta.PeriodicNuclearAccuracyOptions):
                raise TypeError('PeriodicIntegralOptions.nuclear_accuracy must be PeriodicNuclearAccuracyOptions or None')
            for name in ('matrix_error', 'attraction_energy_error', 'nuclear_energy_error'):
                value = getattr(self.nuclear_accuracy, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'PeriodicIntegralOptions.nuclear_accuracy.{name} must be finite positive')
        if self.exchange_accuracy is not None:
            if not isinstance(self.exchange_accuracy, core._manta.PeriodicExchangeAccuracyOptions):
                raise TypeError('PeriodicIntegralOptions.exchange_accuracy must be PeriodicExchangeAccuracyOptions or None')
            for name in ('matrix_error', 'energy_error'):
                value = getattr(self.exchange_accuracy, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'PeriodicIntegralOptions.exchange_accuracy.{name} must be finite positive')
        if self.one_electron_accuracy is not None:
            if not isinstance(self.one_electron_accuracy, core._manta.PeriodicOneElectronAccuracyOptions):
                raise TypeError('PeriodicIntegralOptions.one_electron_accuracy must be PeriodicOneElectronAccuracyOptions or None')
            for name in ('overlap_error', 'kinetic_error', 'kinetic_energy_error'):
                value = getattr(self.one_electron_accuracy, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'PeriodicIntegralOptions.one_electron_accuracy.{name} must be finite positive')
        for name in ('reciprocal_energy_cutoff', 'pair_cutoff_bohr', 'nuclear_cutoff_bohr',
                     'alpha', 'madelung_tolerance', 'madelung_alpha'):
            value = getattr(self, name)
            if name == 'madelung_alpha' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'PeriodicIntegralOptions.{name} must be finite positive')
        if self.madelung_tolerance >= 1:
            raise ValueError('PeriodicIntegralOptions.madelung_tolerance must be in (0,1)')
        if isinstance(self.reciprocal_batch, bool) or not isinstance(self.reciprocal_batch, numbers.Integral) or self.reciprocal_batch <= 0:
            raise ValueError('PeriodicIntegralOptions.reciprocal_batch must be a positive integer')


@dataclass(frozen=True)
class PeriodicXCOptions:
    """Explicit finite XC images and periodic atomic quadrature for KS.

    Radii are in bohr. Grid counts and the partition radius are independent
    convergence controls, not an accuracy profile. XC images are independent
    of the physical shell-pair support in ``PeriodicIntegralOptions``.
    """
    xc_image_cutoff_bohr: float
    radial_points: int = 50
    lebedev_order: int = 29
    partition_radius_bohr: float = 8.
    stratmann: bool = False
    grid_batch: int = 64

    def __post_init__(self):
        for name in ('xc_image_cutoff_bohr', 'partition_radius_bohr'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f'PeriodicXCOptions.{name} must be finite nonnegative')
        if not 0 < self.partition_radius_bohr <= 1e4:
            raise ValueError('PeriodicXCOptions.partition_radius_bohr must be in (0,10000]')
        for name in ('radial_points', 'lebedev_order', 'grid_batch'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f'PeriodicXCOptions.{name} must be a positive integer')
        if self.radial_points > 10000:
            raise ValueError('PeriodicXCOptions.radial_points must be at most 10000')
        if not isinstance(self.stratmann, bool):
            raise TypeError('PeriodicXCOptions.stratmann must be a bool')

    def _quadrature_options(self):
        options = core._manta.PeriodicQuadratureOptions()
        for name in ('radial_points', 'lebedev_order', 'partition_radius_bohr', 'stratmann'):
            setattr(options, name, getattr(self, name))
        return options


@dataclass(frozen=True)
class PeriodicMANTAResult:
    """Job result retaining the native evaluated density/Fock snapshot.

    Arrays returned by properties are independent owned exports. Orbitals
    diagonalize the terminal Fock and need not reproduce an unconverged D.
    Equal alpha/beta density exports let the existing QVF spin-density format
    retain the exact restricted SCF state without an orbital reconstruction.
    """
    native: object
    mesh: tuple[int, int, int]
    is_shift: tuple[int, int, int]
    _provenance: dict
    _run_fields: dict

    @property
    def energy(self):
        return self.native.receipt.energy

    @property
    def converged(self):
        return self.native.converged

    @property
    def iterations(self):
        return self.native.solver_snapshot.evaluated_iteration

    @property
    def n_iter(self):
        return self.iterations

    @property
    def scf_history(self):
        """Owned QVF-compatible records of evaluated solver iterations."""
        rows = []
        for step in self.native.scf_history:
            row = dict(iter=step.evaluated_iteration, energy_eh=step.energy_per_cell)
            if step.has_energy_change:
                row['delta_e'] = step.energy_change
            rows.append(row)
        return rows

    @property
    def run_fields(self):
        return copy.deepcopy(self._run_fields)

    @property
    def exchange_q0(self):
        return self.native.receipt.exchange_q0

    @property
    def integral_provenance(self):
        return copy.deepcopy(self._provenance)

    @property
    def density_k(self):
        return list(self.native.density)

    @property
    def density_alpha_k(self):
        return list(.5*self.native.density)

    @property
    def density_beta_k(self):
        return list(.5*self.native.density)

    @property
    def fock_k(self):
        return list(self.native.fock)

    @property
    def overlap_k(self):
        return list(self.native.overlap)

    @property
    def hcore_k(self):
        return list(self.native.hcore)

    @property
    def mo_coeffs_k(self):
        return list(self.native.coefficients)

    @property
    def mo_energies_k(self):
        return list(self.native.orbital_energies)

    @property
    def occupations_k(self):
        memory = self.native.solver_memory
        out = np.zeros((memory.n_kpoints, memory.retained_rank))
        out[:, :memory.n_occupied] = 2.
        return list(out)

    @property
    def kpoint_weights(self):
        return np.full(self.native.plan.momenta, 1/self.native.plan.momenta)

    @property
    def kpoints_fractional(self):
        mesh = core._RegularKMesh(self.mesh, self.is_shift)
        return np.array([mesh.fractional(mesh.address(i)) for i in range(len(mesh))])


def _spin_history(step, native):
    populations = [native.plan.n_alpha, native.plan.n_beta]
    return dict(spin_electrons_per_cell=populations,
                spin_homo=[float(x) if count else None for x, count in zip(step.spin_homo, populations)],
                spin_lumo=[float(x) for x in step.spin_lumo],
                spin_band_gaps=[float(gap) if count else None
                                for gap, count in zip(step.spin_band_gap, populations)])


@dataclass(frozen=True)
class PeriodicMANTAUHFResult(PeriodicMANTAResult):
    """Fixed-spin job state; total density and separately owned spin exports.

    There is no single UHF Fock or orbital set. Use the alpha/beta properties.
    QVF retains the evaluated spin densities, including nonconverged states.
    """
    @property
    def density_k(self):
        return list(self.native.density.sum(axis=0))

    @property
    def density_alpha_k(self):
        return list(self.native.density[0])

    @property
    def density_beta_k(self):
        return list(self.native.density[1])

    @property
    def fock_alpha_k(self):
        return list(self.native.fock[0])

    @property
    def fock_beta_k(self):
        return list(self.native.fock[1])

    @property
    def mo_coeffs_alpha_k(self):
        return list(self.native.coefficients[0])

    @property
    def mo_coeffs_beta_k(self):
        return list(self.native.coefficients[1])

    @property
    def mo_energies_alpha_k(self):
        return list(self.native.orbital_energies[0])

    @property
    def mo_energies_beta_k(self):
        return list(self.native.orbital_energies[1])

    def _occupations(self, count):
        memory = self.native.solver_memory
        out = np.zeros((memory.n_kpoints, memory.retained_rank))
        out[:, :count] = 1.
        return list(out)

    @property
    def occupations_alpha_k(self):
        return self._occupations(self.native.plan.n_alpha)

    @property
    def occupations_beta_k(self):
        return self._occupations(self.native.plan.n_beta)

    @property
    def fock_k(self):
        raise AttributeError('Unrestricted results have separate fock_alpha_k and fock_beta_k')

    @property
    def mo_coeffs_k(self):
        raise AttributeError('Unrestricted results have separate mo_coeffs_alpha_k and mo_coeffs_beta_k')

    @property
    def mo_energies_k(self):
        raise AttributeError('Unrestricted results have separate mo_energies_alpha_k and mo_energies_beta_k')

    @property
    def occupations_k(self):
        raise AttributeError('Unrestricted results have separate occupations_alpha_k and occupations_beta_k')


@dataclass(frozen=True)
class PeriodicMANTARKSResult(PeriodicMANTAResult):
    """Restricted KS job; the native singleton spin axis is exported as k."""
    @property
    def density_k(self):
        return list(self.native.density[0])

    @property
    def density_alpha_k(self):
        return list(.5*self.native.density[0])

    @property
    def density_beta_k(self):
        return list(.5*self.native.density[0])

    @property
    def fock_k(self):
        return list(self.native.fock[0])

    @property
    def mo_coeffs_k(self):
        return list(self.native.coefficients[0])

    @property
    def mo_energies_k(self):
        return list(self.native.orbital_energies[0])


@dataclass(frozen=True)
class PeriodicMANTABupoRHFResult(PeriodicMANTARKSResult):
    """Restricted BUPO HF with the native singleton spin axis exported as k."""


@dataclass(frozen=True)
class PeriodicMANTAUKSResult(PeriodicMANTAUHFResult):
    """Fixed-spin KS job retaining the evaluated spin/k snapshot."""


def _same_default(value, default):
    if value is default:
        return True
    if isinstance(default, (str, int, float, bool)) and isinstance(value, type(default)):
        return value == default
    return False


def _ks_option_method(functional):
    # A composite can cancel aggregate CAM metadata while still containing
    # range components. Select the native admission route from its descriptor.
    if any(kernel != 'coulomb' for kernel, _, _ in functional.exchange_components):
        return 'RangeSeparatedKS'
    return 'GlobalHybridKS' if functional.is_hybrid else 'KS'


def _controls(arguments, entrypoint):
    # An explicit inventory makes additions to the large runner signature fail
    # closed here unless they are unchanged defaults or deliberately admitted.
    accepted = {
        'system', 'basis', 'integral_engine', 'coulomb', 'exchange', 'periodic_integrals', 'periodic_xc', 'periodic_fitting',
        'method', 'functional', 'jk_method', 'use_diis', 'convergence', 'initial_guess', 'max_iter',
        'conv_tol_energy', 'conv_tol_grad', 'exchange_exxdiv', 'kpoints',
        'write_molden_file', 'write_population_file', 'write_density', 'output',
        'write_xyz_file', 'write_poscar_file', 'write_xsf_structure_file', 'write_cif_file',
        'citations', 'output_qvf', 'dry_run', 'record_hostname', 'structured_log', 'progress',
    }
    for name, parameter in inspect.signature(entrypoint).parameters.items():
        if name not in accepted and not _same_default(arguments[name], parameter.default):
            raise NotImplementedError(f"Selected MANTA periodic jobs do not support {name}")
    a = arguments
    if not isinstance(a['method'], str) or a['method'].upper() not in ('RHF', 'UHF', 'RKS', 'UKS'):
        raise NotImplementedError('Selected MANTA periodic jobs require method=RHF, UHF, RKS or UKS')
    ks = a['method'].upper() in ('RKS', 'UKS')
    if ks:
        if not isinstance(a['functional'], str) or not a['functional'].strip():
            raise ValueError('Selected MANTA KS method requires a nonempty functional name')
        if not isinstance(a['periodic_xc'], PeriodicXCOptions):
            raise TypeError('Selected MANTA KS requires periodic_xc=PeriodicXCOptions(...)')
    elif a['functional'] is not None or a['periodic_xc'] is not None:
        raise NotImplementedError('Selected MANTA HF does not use functional or periodic_xc')
    if a['jk_method'] != 'auto':
        raise ValueError("Selected MANTA periodic jobs require jk_method='auto'; select J/K with coulomb and exchange")
    if a['use_diis'] is not False:
        raise NotImplementedError('Selected MANTA periodic jobs require use_diis=False')
    if a['convergence'] not in (None, 'off'):
        raise NotImplementedError("Selected MANTA periodic jobs require convergence=None or 'off'")
    guess = a['initial_guess']
    guess = guess.upper() if isinstance(guess, str) else getattr(guess, 'name', '')
    if guess not in ('AUTO', 'HCORE'):
        raise NotImplementedError('Selected MANTA periodic jobs currently support HCORE/AUTO guesses')
    for name in ('write_molden_file', 'write_population_file', 'write_density'):
        if a[name] is not None and a[name] is not False:
            raise NotImplementedError(f'Selected MANTA periodic jobs do not yet support {name}=True')
    if a['progress'] is not None and a['progress'] is not False:
        raise NotImplementedError('Selected MANTA periodic jobs do not yet expose per-iteration progress callbacks')
    for name in ('write_xyz_file', 'write_poscar_file', 'write_xsf_structure_file', 'write_cif_file',
                 'citations', 'output_qvf', 'dry_run', 'record_hostname'):
        if not isinstance(a[name], bool):
            raise TypeError(f'{name} must be a bool')
    if a['exchange_exxdiv'] not in ('ewald', 'none'):
        raise ValueError("Selected MANTA exchange_exxdiv must be 'ewald' or 'none'")
    if isinstance(a['max_iter'], bool) or not isinstance(a['max_iter'], numbers.Integral) or a['max_iter'] <= 0:
        raise ValueError('max_iter must be a positive integer')
    for name in ('conv_tol_energy', 'conv_tol_grad'):
        x = a[name]
        if name == 'conv_tol_grad' and x is None:
            continue
        if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) or x <= 0:
            raise ValueError(f'{name} must be finite positive')
    functional = core.Functional(a['functional'], spin=2 if a['method'].upper() == 'UKS' else 1) if ks else None
    # Dispatch only. Complete component/range admission and the exchange
    # coefficient belong to the native method planner, not an alias table.
    option_method = _ks_option_method(functional) if ks else a['method'].upper()
    hybrid = ks and option_method != 'KS'
    for name, kind in [('coulomb', CoulombBuildOptions), ('exchange', ExchangeBuildOptions)]:
        required = 'none' if ks and not hybrid and name == 'exchange' else 'exact'
        if a[name] is None:
            a[name] = kind(algorithm=required)
        if not isinstance(a[name], kind):
            raise TypeError(f'{name} must be {kind.__name__}')
    bupo = a['coulomb'].algorithm == 'bupo'
    fitted = a['coulomb'].algorithm == 'ri' or a['exchange'].algorithm == 'ri'
    if bupo:
        if ks or fitted or a['exchange'].algorithm != 'exact':
            raise NotImplementedError('Selected periodic BUPO currently requires RHF/UHF and exact reciprocal K')
        if a['periodic_fitting'] is not None or a['coulomb'].auxiliary_basis is not None:
            raise ValueError('Selected periodic BUPO does not use a fitting source')
        if not isinstance(a['periodic_integrals'], PeriodicBupoIntegralOptions):
            raise TypeError('Selected periodic BUPO requires periodic_integrals=PeriodicBupoIntegralOptions(...)')
        a['periodic_integrals']._source_options(a['coulomb'].bupo)
        option_method = 'BupoHF'
    elif fitted:
        allowed_k = ('none',) if ks and not hybrid else ('ri', 'exact')
        if a['coulomb'].algorithm not in ('ri', 'exact') or a['exchange'].algorithm not in allowed_k:
            raise NotImplementedError('Selected periodic fitting requires exact/RI J and K, or RI J with no K for pure KS')
        if not isinstance(a['periodic_fitting'], PeriodicFittingOptions):
            raise TypeError('Selected periodic RI requires periodic_fitting=PeriodicFittingOptions(...)')
        if not isinstance(a['coulomb'].auxiliary_basis, core.BasisSet):
            raise TypeError('Selected periodic RI requires coulomb.auxiliary_basis as a resolved BasisSet')
        if not isinstance(a['coulomb'].metric, core._manta.RIOptions):
            raise TypeError('Selected periodic RI requires coulomb.metric as RIOptions')
        option_method = 'RI'+option_method
    else:
        for name in ('coulomb', 'exchange'):
            required = 'none' if ks and not hybrid and name == 'exchange' else 'exact'
            if a[name].algorithm != required:
                raise NotImplementedError(f'Selected MANTA periodic {name} requires algorithm={required} for {a["method"]}')
        if a['periodic_fitting'] is not None:
            raise ValueError('periodic_fitting requires an explicitly selected RI J or RI K')
    if not fitted and a['coulomb'].auxiliary_basis is not None:
        raise ValueError('Selected exact reciprocal MANTA J/K does not use an auxiliary basis')
    domains = a['periodic_integrals']
    if not isinstance(domains, PeriodicBupoIntegralOptions if bupo else PeriodicIntegralOptions):
        raise TypeError('Selected periodic jobs require periodic_integrals=PeriodicIntegralOptions(...)')
    if getattr(domains, 'support_selection', None) is not None and (bupo or not fitted):
        raise NotImplementedError('Automatic physical supports currently require a periodic RI method')
    if getattr(domains, 'nuclear_accuracy', None) is not None and (bupo or not fitted):
        raise NotImplementedError('Nuclear accuracy currently requires a periodic RI method with midpoint-sphere nuclear sources')
    if not bupo and domains.exchange_accuracy is not None and not fitted:
        raise NotImplementedError('Exchange accuracy currently requires a periodic RI method with exact K')
    if not bupo and domains.one_electron_accuracy is not None and not fitted:
        raise NotImplementedError('One-electron image accuracy currently requires a periodic RI method')
    options = domains._hf_options(a['method'].upper()) if bupo else getattr(core._manta, f'Periodic{option_method}Options')()
    physical = options.ks if ks and (hybrid or fitted) else options
    if not bupo:
        for name in ('pair_cutoff_bohr', 'nuclear_cutoff_bohr', 'alpha', 'reciprocal_batch'):
            setattr(physical, name, getattr(domains, name))
    if fitted:
        if not ks or hybrid:
            options.coulomb_algorithm = a['coulomb'].algorithm
            options.exchange_algorithm = a['exchange'].algorithm
        options.ri = a['coulomb'].metric
        if domains.nuclear_accuracy is not None:
            options.nuclear_accuracy = domains.nuclear_accuracy
        if domains.one_electron_accuracy is not None:
            options.one_electron_accuracy = domains.one_electron_accuracy
        if domains.exchange_accuracy is not None:
            options.exchange_accuracy = domains.exchange_accuracy
        if a['periodic_fitting'].accuracy is not None:
            options.ri_accuracy = a['periodic_fitting'].accuracy
        options.fitting.pair_cutoff_bohr = domains.pair_cutoff_bohr
        for name in ('omega', 'auxiliary_cutoff_bohr', 'integral_screen_error', 'coupling_screen_error',
                     'coupling_multipole_error', 'coupling_multipole_order', 'coupling_multipole_hierarchy', 'coupling_multipole_reuse', 'coupling_multipole_compress', 'coupling_multipole_global_actions', 'coupling_multipole_pair_tree', 'coupling_multipole_partition_bytes', 'image_candidate_cap'):
            setattr(options.fitting, name, getattr(a['periodic_fitting'], name))
        if a['periodic_fitting'].coupling_multipole_extents is not None:
            options.fitting.coupling_multipole_extents = a['periodic_fitting'].coupling_multipole_extents
    if ks:
        if not hybrid and (domains.madelung_alpha is not None or domains.madelung_tolerance != 1e-12):
            raise ValueError('Selected pure KS does not use Madelung controls')
        physical.grid_batch = int(a['periodic_xc'].grid_batch)
    if not ks or hybrid:
        options.madelung_alpha = domains.madelung_alpha
        options.madelung_tolerance = domains.madelung_tolerance
        options.exchange_q0 = 'BvK-ewald' if a['exchange_exxdiv'] == 'ewald' else 'strict-zero-mode'
    physical.scf.maximum_iterations = int(a['max_iter'])
    physical.scf.energy_change_tolerance = float(a['conv_tol_energy'])
    if a['conv_tol_grad'] is not None:
        physical.scf.commutator_tolerance = float(a['conv_tol_grad'])
    return options, functional


def _mesh_metadata(kpoints):
    from .kpoints import KPoints, _integer_counts
    if isinstance(kpoints, core._RegularKMesh):
        return kpoints, None
    if kpoints is None:
        return core._RegularKMesh((1, 1, 1), (0, 0, 0)), None
    if isinstance(kpoints, core.BlochKMesh):
        if np.size(kpoints.ir_mapping):
            raise NotImplementedError('Selected MANTA requires a full unreduced BlochKMesh')
        return core._RegularKMesh(kpoints.mesh, kpoints.is_shift), kpoints
    if isinstance(kpoints, KPoints):
        if kpoints.mesh is None or kpoints.shift is None or np.size(kpoints.ir_mapping):
            raise NotImplementedError('Selected MANTA requires a full unreduced regular KPoints mesh')
        if kpoints.smearing is not None or kpoints.bz_integration is not None:
            raise NotImplementedError('Selected MANTA does not support recommended smearing/BZ metadata')
        if kpoints.grid_matrix is not None or kpoints.citation_numerics:
            raise NotImplementedError('Selected MANTA currently supports ordinary regular KPoints meshes')
        return core._RegularKMesh(kpoints.mesh, kpoints.shift), kpoints
    if isinstance(kpoints, (list, tuple)):
        shape = _integer_counts(kpoints, name='kpoints')
    else:
        shape = _integer_counts(kpoints, name='kpoints', repeat=3)
    if len(shape) != 3:
        raise ValueError('Selected MANTA kpoints requires three divisions')
    return core._RegularKMesh(shape, (0, 0, 0)), None


def _validate_kpoints(system, mesh, source):
    fractions = np.array([mesh.fractional(mesh.address(i)) for i in range(len(mesh))])
    if source is not None:
        if isinstance(source, core.BlochKMesh):
            cart = np.asarray(source.kpoints, dtype=float)
            actual = cart @ np.asarray(system.lattice)/(2*np.pi)
        else:
            actual = np.asarray(source.kpoints_frac, dtype=float)
            cart = np.asarray(source.kpoints_cart, dtype=float)
        weights = np.asarray(source.weights, dtype=float)
        if actual.shape != fractions.shape or cart.shape != fractions.shape or weights.shape != (len(mesh),):
            raise ValueError('Selected MANTA KPoints payload does not match its full mesh metadata')
        delta = actual-fractions
        if (not np.all(np.isfinite(actual)) or not np.all(np.isfinite(cart)) or not np.all(np.isfinite(weights))
            or not np.allclose(delta, np.rint(delta), atol=1e-12, rtol=0)
            or not np.allclose(weights, 1/len(mesh), atol=1e-12, rtol=0)
            or not np.allclose(cart, actual @ (2*np.pi*np.linalg.inv(system.lattice)), atol=1e-12, rtol=1e-12)):
            raise ValueError('Selected MANTA KPoints must match the lattice, canonical address order and uniform full-mesh weights')
    return fractions


def _basis_image_shifts(basis, system):
    """Integer AO-center gauge relative to the QVF nuclear-center basis."""
    inverse = np.linalg.inv(system.lattice)
    positions = np.array([atom.xyz for atom in system.unit_cell])
    shifts = []
    for shell in basis.shells():
        # Native source admission already proved integer image correspondence.
        fractional = inverse @ (np.asarray(shell.origin)-positions[shell.atom_index])
        label = np.rint(fractional).astype(np.int64)
        count = 2*shell.l+1 if shell.pure else (shell.l+1)*(shell.l+2)//2
        shifts.extend([label]*count)
    return np.asarray(shifts, dtype=np.int64)


def _qvf_evaluated_density(result, basis, system, fractions, image_shifts):
    from .output.formats.qvf import qvf_bloch_wf_data
    density = np.array([result.density_alpha_k, result.density_beta_k])
    if np.any(image_shifts):
        # Source S/D/F transform as U M U^H for a displaced AO center,
        # U_mu=exp(+2 pi i k_frac.n_mu). QVF's shell centers are the nuclei.
        phase = np.exp(2j*np.pi*(fractions @ image_shifts.T))
        density *= phase[:, :, None].conj()*phase[:, None, :]
    physical = SimpleNamespace(density_alpha_k=list(density[0]), density_beta_k=list(density[1]))
    return qvf_bloch_wf_data(physical, basis, system.unit_cell_molecule(),
        k_points=fractions, k_weights=result.kpoint_weights)


def run_selected_periodic_job(arguments, entrypoint):
    """Called only by the public job front door with its complete bound kwargs."""
    from .banner import banner, enforce_runtime_pin_from_env
    from .periodic_runner import _resolve_ecp_data, _PERIODIC_OUTPUT_WRITER
    from .output import (OutputPlan, OutputWriter, OutputChannel, write, record,
                         dry_run_manifest, is_dry_run_requested)
    from .output._stem_paths import stem_sibling
    from .output.formats.structured_log import structured_log, _resolve_structured_log_target
    from .output.citations import load_default_database, format_references_block, citation_manifest_rows, format_bibtex

    a = dict(arguments)
    options, functional = _controls(a, entrypoint)
    method = a["method"].upper()
    profile = method.lower()
    ks = method in ('RKS', 'UKS')
    fitted = isinstance(options, (core._manta.PeriodicRIRHFOptions, core._manta.PeriodicRIUHFOptions,
                                  core._manta.PeriodicRIKSOptions, core._manta.PeriodicRIGlobalHybridKSOptions,
                                  core._manta.PeriodicRIRangeSeparatedKSOptions))
    bupo = isinstance(options, core._manta.PeriodicBupoHFOptions)
    option_method = 'BupoHF' if bupo else _ks_option_method(functional) if ks else method
    hybrid = ks and option_method != 'KS'
    range_separated = option_method == 'RangeSeparatedKS'
    mixed = fitted and (a['coulomb'].algorithm != 'ri' or ((not ks or hybrid) and a['exchange'].algorithm != 'ri'))
    source_profile = 'ri_'+profile if fitted else profile
    if fitted: option_method = 'RI'+option_method
    unrestricted = method in ('UHF', 'UKS')
    physical = options.ks if ks and (hybrid or fitted) else options
    ks_source = 'range_separated_ks' if range_separated else 'global_hybrid_ks' if hybrid else 'ks'
    if fitted: ks_source = 'ri_'+ks_source
    provenance = periodic_bupo_hf_provenance if bupo else getattr(manta_periodic, f"periodic_{'ks' if ks else source_profile}_provenance")
    engine = a['integral_engine']
    # Validate selection before consulting basis metadata or allocating arrays.
    _periodic_hf_configuration(options, engine, option_method)
    system, basis = a['system'], a['basis']
    if not isinstance(system, core.PeriodicSystem) or not isinstance(basis, core.BasisSet):
        raise TypeError('Selected MANTA periodic jobs require PeriodicSystem and BasisSet objects')
    enforce_runtime_pin_from_env()
    if _resolve_ecp_data(system, basis)[3]:
        raise NotImplementedError('Selected MANTA periodic jobs currently require an all-electron basis')
    auxiliary = a['coulomb'].auxiliary_basis if fitted else None
    if fitted and _resolve_ecp_data(system, auxiliary)[3]:
        raise NotImplementedError('Selected MANTA periodic fitting requires an all-electron auxiliary basis')
    ks_basis_args = (basis, auxiliary) if fitted else (basis,)
    mesh, source_kpoints = _mesh_metadata(a['kpoints'])
    domains = a['periodic_integrals']
    selection_policy = getattr(domains, 'support_selection', None)
    selection_result = selection_evidence = None
    bupo_source_options = domains._source_options(a['coulomb'].bupo) if bupo else None
    grid_options = a['periodic_xc']._quadrature_options() if ks else None

    def admit(native_budget):
        if bupo:
            # Count-only primitive census; source construction occurs once,
            # after job/output reservations and the dry-run short circuit.
            return core._manta.plan_periodic_pairs(basis, (2*domains.pair_cube_radius+1)**3,
                len(mesh), bupo_source_options.pairs, native_budget)
        if ks:
            return getattr(core._manta, f'plan_periodic_{ks_source}_with_quadrature')(
                *ks_basis_args, system, mesh, functional, domains.reciprocal_energy_cutoff,
                a['periodic_xc'].xc_image_cutoff_bohr, grid_options, options, native_budget)
        if fitted:
            return getattr(core._manta, f'plan_periodic_{source_profile}_from_cutoffs')(
                basis, auxiliary, system, mesh, domains.reciprocal_energy_cutoff, options, native_budget)
        return getattr(core._manta, f'plan_periodic_{source_profile}_from_cutoffs')(
            basis, system, mesh, domains.reciprocal_energy_cutoff, options, native_budget)

    def configure(selected):
        if fitted and ks:
            if range_separated:
                return manta_periodic._periodic_ri_range_ks_configuration(options, selected, mesh, functional)
            return manta_periodic._periodic_ri_ks_configuration(options, selected, mesh, hybrid)
        if fitted:
            return getattr(manta_periodic, f'_periodic_{source_profile}_configuration')(options, selected, mesh)
        return _periodic_hf_configuration(options, selected, option_method)

    # Numerical exports, density-format conversion/validation, serialization
    # and native/output overlap. This is a reservation, not measured RSS.
    nk, n = len(mesh), basis.nbasis
    # Iteration records, QVF dictionaries and serialized copies are bounded
    # by the requested maximum, even when convergence happens much earlier.
    spin_factor = 2 if unrestricted else 1
    job_bytes = 2**20 + spin_factor*2048*nk*n*n + 4096*nk + 4096*basis.nshells + 16384*int(a['max_iter'])
    if fitted and options.ri_accuracy.j_error>0 and options.ri_accuracy.density_l1_limit==0:
        job_bytes += 32768 + 1024*nk  # overlap-envelope receipts, owned exports and serialization copies
    if fitted and options.nuclear_accuracy.matrix_error > 0:
        job_bytes += 49152  # nuclear component receipts and serialization
    if fitted and options.exchange_accuracy.matrix_error > 0:
        job_bytes += 49152  # fixed-space exchange/correction receipts and serialization
    if fitted and options.one_electron_accuracy.overlap_error > 0:
        job_bytes += 32768  # scalar image-certificate output and serialization
    if bupo:
        job_bytes += 49152  # nested source policy, plan and provenance copies
    job_scalar_terms = job_kernel_terms = 0
    if fitted:
        job_bytes += 4096*auxiliary.nshells + 64*len(auxiliary.name)
    if range_separated:
        # An erf entry can resolve into two canonical kernels. Reserve nested
        # component receipts, owned result copies and serialized output before
        # generation, even when coefficients later merge or cancel.
        job_bytes += 32768*2*len(functional.exchange_components)
    budget = core._manta.ResourceBudget()
    for name in ('memory_bytes', 'shell_calls', 'primitive_quartets', 'scalar_terms',
                 'kernel_terms', 'other_live_bytes', 'backend_margin_bytes'):
        setattr(budget, name, getattr(engine.budget, name))
    if job_bytes+budget.other_live_bytes > budget.memory_bytes:
        raise ValueError('MANTA periodic job output memory budget exceeded')
    budget.other_live_bytes += job_bytes
    if bupo:
        # Count-only planning, option capture, output conversion and records.
        # The primitive-dependent serialization charge follows its census.
        job_scalar_terms = 32768+1024*basis.nshells+8192*spin_factor*nk*n*n+16384*int(a['max_iter'])
        job_kernel_terms = 32*job_scalar_terms
        if job_scalar_terms > budget.scalar_terms or job_kernel_terms > budget.kernel_terms:
            raise ValueError('MANTA BUPO job preparation/output work budget exceeded')
        budget.scalar_terms -= job_scalar_terms
        budget.kernel_terms -= job_kernel_terms
    selected = IntegralEngineOptions(backend='manta', budget=budget)
    _, native_budget, python_bytes = configure(selected)
    admission = admit(native_budget)
    # The overlap census is cells*(sum shell primitive counts)^2. Its square
    # root therefore bounds the primitive entries copied to QVF's shell JSON.
    # Charge those Python owners before calling basis.shells() or a writer.
    source_plan = None if bupo else admission.domains.ks if ks else getattr(admission, profile)
    primitive_bytes = 2048*math.isqrt(admission.primitive_pairs if bupo else source_plan.overlap.primitive_pairs)
    job_bytes += primitive_bytes
    if primitive_bytes+budget.other_live_bytes > budget.memory_bytes:
        raise ValueError('MANTA periodic job basis/output memory budget exceeded')
    budget.other_live_bytes += primitive_bytes
    if bupo:
        if primitive_bytes > budget.scalar_terms or 32*primitive_bytes > budget.kernel_terms:
            raise ValueError('MANTA BUPO job basis serialization work budget exceeded')
        budget.scalar_terms -= primitive_bytes
        budget.kernel_terms -= 32*primitive_bytes
        job_scalar_terms += primitive_bytes
        job_kernel_terms += 32*primitive_bytes
    _, native_budget, python_bytes = configure(selected)
    if not bupo:
        admission = admit(native_budget)
    fractions = _validate_kpoints(system, mesh, source_kpoints)
    image_shifts = _basis_image_shifts(basis, system) if a['output_qvf'] else None
    stem = Path(os.fspath(a['output']))
    structured_target = _resolve_structured_log_target(a['structured_log'], stem)
    plan = OutputPlan.from_run_job_kwargs(
        output=stem, method=method, basis=basis.name, functional=a['functional'],
        write_molden_file=False, write_population=False, write_density_xsf=False,
        write_xyz=a['write_xyz_file'], write_poscar=a['write_poscar_file'],
        write_xsf_structure=a['write_xsf_structure_file'], write_cif=a['write_cif_file'],
        citations=a['citations'], structured_log=structured_target or False,
        crash_dump=False, output_qvf=a['output_qvf'], job_kind='periodic_scf')
    stem.parent.mkdir(parents=True, exist_ok=True)
    if a['dry_run'] or is_dry_run_requested():
        # BUPO construction is executing scientific work, so output discovery
        # uses the explicit complete-job model cap as a conservative upper
        # reservation. It does not pretend that the partial census is an HF estimate.
        estimate = engine.budget.memory_bytes if bupo or selection_policy is not None else admission.modeled_peak_bytes
        dry_run_manifest(plan, record_hostname=a['record_hostname'], estimate_bytes=estimate)
        if selection_policy is not None:
            write('  Automatic support dry run: declared complete-job memory cap; numerical selection is deferred.\n')
        if bupo:
            write('  BUPO dry run: memory reservation is the declared complete-job cap; source construction is deferred.\n')
        return None
    if selection_policy is not None:
        from .manta_periodic_support_selection import support_request, support_selection_provenance
        request = support_request(domains, options, source_plan, ks=ks, hybrid=hybrid, unrestricted=unrestricted)
        selection_result = core._manta.select_periodic_supports(basis, system, mesh, request, selection_policy, native_budget)
        if not selection_result.qualified:
            raise ValueError(f'MANTA automatic supports did not meet component targets: {selection_result.termination}')
        choice = selection_result.selected
        domains = replace(domains, reciprocal_energy_cutoff=choice.reciprocal_energy_cutoff,
                          pair_cutoff_bohr=choice.pair_cutoff_bohr, nuclear_cutoff_bohr=choice.nuclear_cutoff_bohr,
                          one_electron_accuracy=choice.one_electron_accuracy, nuclear_accuracy=choice.nuclear_accuracy,
                          exchange_accuracy=choice.exchange_accuracy if request.exchange_accuracy.matrix_error > 0 else None)
        for name in ('pair_cutoff_bohr', 'nuclear_cutoff_bohr'):
            setattr(physical, name, getattr(choice, name))
        options.fitting.pair_cutoff_bohr = choice.pair_cutoff_bohr
        for name in ('one_electron_accuracy', 'nuclear_accuracy', 'exchange_accuracy'):
            setattr(options, name, getattr(choice, name))
        # Search and final SCF share one job budget. Retained trial metadata
        # stays live; all admitted trial work, including rejected candidates,
        # is deducted before planning the final physical method.
        job_bytes += selection_result.retained_bytes
        budget.other_live_bytes += selection_result.retained_bytes
        for field, charged in (('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
                               ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')):
            setattr(budget, field, getattr(budget, field) - getattr(selection_result, charged))
        _, native_budget, python_bytes = configure(selected)
        admission = admit(native_budget)
        source_plan = admission.domains.ks if ks else getattr(admission, profile)
        selection_evidence = support_selection_provenance(selection_result)
    active_exchange_q0 = not ks or (hybrid and source_plan.long_range_exchange_fraction != 0.)
    fields = dict(integral_backend_requested='manta', integral_backend_executed='manta',
                  coulomb_algorithm=a['coulomb'].algorithm, exchange_algorithm=a['exchange'].algorithm,
                  jk_method_requested='auto', jk_method_resolved='ri' if fitted else 'exact-reciprocal',
                  jk_method_executed=f'finite-3d-ri-{profile}' if fitted else f'finite-3d-reciprocal-{profile}',
                  exchange_q0=options.exchange_q0 if active_exchange_q0 else 'not-applicable',
                  initial_guess_executed='HCORE', symmetry_reduced=False,
                  reciprocal_energy_cutoff=domains.reciprocal_energy_cutoff,
                  pair_cutoff_bohr=domains.pair_cutoff_bohr, periodic_dimension=3,
                  scf_trace_scope='evaluated-iterations', job_reserved_bytes=job_bytes)
    if selection_evidence is not None:
        fields.update(support_selection_identity=selection_evidence['certificate_identity'],
                      support_selection_attempts=len(selection_evidence['trials']))
    if bupo:
        fields.update(jk_method_resolved='bupo-exact-reciprocal', jk_method_executed=f'fixed-3d-bupo-{profile}',
                      ewald_eta=domains.eta, pair_cube_radius=domains.pair_cube_radius,
                      real_cube_radius=domains.real_cube_radius, reciprocal_cube_radius=domains.reciprocal_cube_radius,
                      near_cube_radius=domains.near_cube_radius,
                      accuracy_qualification='evaluated-density-electrostatic-one-electron-and-full-exchange',
                      job_scalar_terms=job_scalar_terms, job_kernel_terms=job_kernel_terms)
    else:
        fields.update(nuclear_cutoff_bohr=domains.nuclear_cutoff_bohr, ewald_alpha=domains.alpha)
    if mixed:
        fields.update(jk_method_resolved='mixed-exact-ri', jk_method_executed=f'finite-3d-mixed-ri-{profile}')
    if fitted:
        fields.update(auxiliary_basis=auxiliary.name, auxiliary_basis_functions=auxiliary.nbasis,
                      fitting_omega=options.fitting.omega, auxiliary_cutoff_bohr=options.fitting.auxiliary_cutoff_bohr,
                      fitting_integral_screen_error=options.fitting.integral_screen_error,
                      fitting_coupling_screen_error=options.fitting.coupling_screen_error,
                      fitting_coupling_multipole_error=options.fitting.coupling_multipole_error,
                      fitting_coupling_multipole_order=options.fitting.coupling_multipole_order,
                      fitting_coupling_multipole_hierarchy=options.fitting.coupling_multipole_hierarchy,
                      fitting_coupling_multipole_reuse=options.fitting.coupling_multipole_reuse,
                      fitting_coupling_multipole_compress=options.fitting.coupling_multipole_compress,
                      fitting_coupling_multipole_global_actions=options.fitting.coupling_multipole_global_actions,
                      fitting_coupling_multipole_pair_tree=options.fitting.coupling_multipole_pair_tree,
                      fitting_coupling_multipole_partition_bytes=options.fitting.coupling_multipole_partition_bytes,
                      fitting_image_candidate_cap=options.fitting.image_candidate_cap,
                      fitting_metric_absolute_cutoff=options.ri.absolute_cutoff,
                      fitting_metric_relative_cutoff=options.ri.relative_cutoff,
                      fitting_negative_relative_tolerance=options.ri.negative_relative_tolerance,
                      fitting_mode_block=options.ri.mode_block,
                      accuracy_qualification='finite-fitted-source-only')
    if ks:
        xc = a['periodic_xc']
        fields.update(xc_algorithm='finite-bloch-semilocal', functional=a['functional'],
                      exact_exchange_fraction=source_plan.exact_exchange_fraction,
                      exchange_q0_applicability='active' if active_exchange_q0 else 'inactive',
                      xc_image_cutoff_bohr=xc.xc_image_cutoff_bohr,
                      grid_radial_points=xc.radial_points, grid_lebedev_order=xc.lebedev_order,
                      grid_partition_radius_bohr=xc.partition_radius_bohr, grid_stratmann=xc.stratmann,
                      accuracy_qualification='finite-source-only')
        if range_separated:
            fields.update(range_separated=True, exchange_q0_requested=options.exchange_q0,
                          long_range_exchange_fraction=source_plan.long_range_exchange_fraction,
                          short_range_exchange_fraction=source_plan.short_range_exchange_fraction)
    writer = OutputWriter(plan, record_hostname=a['record_hostname'], extra_run_fields=fields)
    _PERIODIC_OUTPUT_WRITER.set(writer)
    out_path = stem_sibling(stem, '.out')
    with writer.context():
        with structured_log(structured_target), OutputChannel.to_file(out_path):
            write(banner()+'\n\n')
            source_summary = 'XC='+a['functional'] if ks else 'K=exact reciprocal'
            if bupo:
                source_summary += '; matched fixed BUPO source; all evaluated-density integral targets enforced'
            j_label = 'BUPO' if bupo else 'RI' if fitted and a['coulomb'].algorithm == 'ri' else 'exact reciprocal'
            if fitted:
                k_label = {'ri': 'RI', 'exact': 'exact reciprocal', 'none': 'none'}[a['exchange'].algorithm]
                source_summary = ('XC='+a['functional']+'; ' if ks else '')
                source_summary += f'K={k_label}; auxiliary={auxiliary.name}; exchange {fields["exchange_q0"]}'
                if range_separated:
                    source_summary += f'; physical range-separated; long-range fraction={fields["long_range_exchange_fraction"]}'
                elif hybrid:
                    source_summary += f'; fraction={fields["exact_exchange_fraction"]}'
            elif range_separated:
                source_summary += (f'; K=exact reciprocal range-separated; '
                                   f'long-range fraction={fields["long_range_exchange_fraction"]}; '
                                   f'exchange {fields["exchange_q0"]}')
            elif hybrid:
                source_summary += (f'; K=exact reciprocal; fraction={fields["exact_exchange_fraction"]}'
                                   f'; exchange {options.exchange_q0}')
            record('job_start', f'  Job: PERIODIC {method}  basis={basis.name}\n'
                   f'  Integral engine: manta; J={j_label}; {source_summary}.\n'
                   f'  Full regular mesh: {tuple(mesh.mesh)}; shifts: {tuple(mesh.is_shift)}.\n', **fields)
            if bupo:
                prepared = core._manta.prepare_periodic_bupo_hf(basis, system, mesh,
                    domains.reciprocal_energy_cutoff, bupo_source_options, options, native_budget)
                native = prepared.run()
                options = prepared.options
            elif ks:
                native = getattr(core._manta, f'run_periodic_{ks_source}_with_quadrature')(
                    *ks_basis_args, system, mesh, functional, domains.reciprocal_energy_cutoff,
                    a['periodic_xc'].xc_image_cutoff_bohr, grid_options, options, native_budget)
            elif fitted:
                native = getattr(manta_periodic, f'run_periodic_{source_profile}_from_cutoffs')(
                    basis, auxiliary, system, mesh, domains.reciprocal_energy_cutoff,
                    options=options, integral_engine=selected)
            else:
                native = getattr(manta_periodic, f'run_periodic_{source_profile}_from_cutoffs')(
                    basis, system, mesh, domains.reciprocal_energy_cutoff, options=options, integral_engine=selected)
            evidence = provenance(native, mesh, options, python_bytes)
            if selection_result is not None:
                from .manta_periodic_support_selection import verify_selected_supports
                verify_selected_supports(native, selection_result)
                evidence['support_selection'] = selection_evidence
                evidence['modeled_peak_bytes'] = max(evidence['modeled_peak_bytes'], selection_result.modeled_peak_bytes)
                evidence['scalar_terms'] += selection_result.scalar_terms
                evidence['kernel_terms'] += selection_result.kernel_terms
                evidence['job_shell_calls_upper_bound'] = native.plan.shell_calls + selection_result.shell_calls
                primitive_work = native.plan.primitive_work if ks else native.plan.primitive_pairs
                evidence['job_primitive_pairs_upper_bound'] = primitive_work + selection_result.primitive_pairs
            if bupo:
                evidence['preparation_admission'] = {name: getattr(prepared.plan, name) for name in (
                    'modeled_peak_bytes', 'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms')}
                evidence['preparation_admission']['scalar_terms'] += job_scalar_terms
                evidence['preparation_admission']['kernel_terms'] += job_kernel_terms
                evidence.update(job_scalar_terms=job_scalar_terms, job_kernel_terms=job_kernel_terms,
                                scalar_terms=evidence['scalar_terms']+job_scalar_terms,
                                kernel_terms=evidence['kernel_terms']+job_kernel_terms)
                record('manta_periodic_bupo_hf', f'  Periodic BUPO {method}: manta; '
                       f'converged={native.converged}; E={native.receipt.energy:.12f}; '
                       'all evaluated-density integral targets passed.\n', integral_provenance=evidence)
            evidence.update(job_reserved_bytes=job_bytes, coulomb_algorithm=fields['coulomb_algorithm'], exchange_algorithm=fields['exchange_algorithm'],
                            method=method, functional=a['functional'],
                            max_iter=int(a['max_iter']), conv_tol_energy=physical.scf.energy_change_tolerance,
                            conv_tol_grad=physical.scf.commutator_tolerance, initial_guess='HCORE',
                            restart_density_encoding=('spin-resolved-evaluated-density' if unrestricted
                                else 'equal-alpha-beta-evaluated-density'))
            if fitted:
                evidence.update(auxiliary_basis=auxiliary.name, auxiliary_basis_functions=auxiliary.nbasis)
            if unrestricted:
                evidence['spin_scf_history'] = [dict(iteration=step.evaluated_iteration,
                    **_spin_history(step, native)) for step in native.scf_history]
            if image_shifts is not None:
                evidence.update(qvf_density_gauge='nuclear-centers',
                                source_ao_image_shifts=image_shifts.tolist())
            fields.update(integral_source_identity=native.receipt.source_identity,
                integral_domain_identity=native.receipt.domain_identity,
                density_identity=native.receipt.final_density_identity,
                fock_identity=native.receipt.final_fock_identity)
            if fitted:
                if range_separated:
                    # J and each physical K have distinct fixed projectors.
                    # K-only fitting has no parent J fitting receipt.
                    fits = ([native.receipt.fitting_info] if native.receipt.fitting_info else [])
                    fits += [c.fitting_info for c in native.receipt.exchange_components if c.fitting_info]
                    fields.update(fitting_source_identities=[f.source_identity for f in fits],
                                  fitting_frame_identities=[f.frame_identity for f in fits])
                else:
                    fields.update(fitting_source_identity=native.receipt.fitting_info.source_identity,
                                  fitting_frame_identity=native.receipt.fitting_info.frame_identity,
                                  exchange_source_identity=native.receipt.exchange_identity)
                fields.update(coulomb_source_identity=native.receipt.coulomb_identity,
                              exchange_correction_identity=native.receipt.exchange_correction_identity)
            if ks:
                fields.update(quadrature_source_identity=native.receipt.quadrature.source_identity,
                              xc_source_identity=native.receipt.xc.source_identity)
                if range_separated:
                    fields.update(exchange_source_identities=[(c.fitted_fock if c.fitted_fock else c.exchange).source_identity
                                                              for c in native.receipt.exchange_components],
                                  exchange_correction_identity=native.receipt.exchange_correction_identity)
                elif hybrid:
                    fields.update(exchange_source_identity=native.receipt.exchange_identity,
                                  exchange_correction_identity=native.receipt.exchange_correction_identity)
                energy_fields = ('nuclear_energy', 'one_electron_energy', 'coulomb_energy', 'xc_energy',
                                 'uncorrected_exchange_energy', 'exchange_correction_energy', 'exchange_energy')
                fields.update({name: getattr(native.receipt, name) for name in energy_fields})
                record('manta_periodic_ks', f'  Finite periodic {method}: manta; '
                       f'functional={a["functional"]}; converged={native.converged}; '
                       f'E={native.receipt.energy:.12f}.\n', integral_provenance=evidence)
                terms = [('nuclear_energy', 'Nuclear'), ('one_electron_energy', 'One-electron'),
                         ('coulomb_energy', 'Hartree'), ('xc_energy', 'Semilocal XC')]
                if hybrid:
                    k_term = 'RI exchange' if fitted and a['exchange'].algorithm == 'ri' else 'Exact exchange'
                    terms += [('uncorrected_exchange_energy', k_term+' (scaled, uncorrected)'),
                              ('exchange_correction_energy', 'Exchange zero-mode correction (scaled)')]
                record('scf_energy_components', '  Evaluated energy components (hartree per cell):\n'+
                       ''.join(f'    {label}: {fields[name]:.12f}\n' for name, label in terms),
                       iteration=native.solver_snapshot.evaluated_iteration,
                       density_identity=native.receipt.final_density_identity,
                       **{name: fields[name] for name in energy_fields})
            result_type = {'RHF': PeriodicMANTAResult, 'UHF': PeriodicMANTAUHFResult,
                           'RKS': PeriodicMANTARKSResult, 'UKS': PeriodicMANTAUKSResult}[method]
            if bupo and not unrestricted:
                result_type = PeriodicMANTABupoRHFResult
            result = result_type(native, tuple(mesh.mesh), tuple(mesh.is_shift), evidence, dict(fields))
            terminal = native.solver_snapshot
            status = 'converged' if result.converged else 'not_converged'
            writer.set_result_status(status)
            writer.update_run_fields(dict(fields, scf_converged=result.converged, scf_energy=result.energy,
                                          n_scf_iterations=result.iterations))
            # Native snapshots are scalar, owned and collected within the
            # admitted run. Delivery follows the solve; these are not live
            # callbacks and the final independent confirmation is not a row.
            for step in native.scf_history:
                record('scf_iteration', f'  SCF {step.evaluated_iteration:4d}: '
                       f'E={step.energy_per_cell:.12f}  '
                       f'commutator RMS={step.commutator_frobenius_rms:.6e}\n',
                       iteration=step.evaluated_iteration, energy=step.energy_per_cell,
                       energy_change=step.energy_change if step.has_energy_change else None,
                       commutator_frobenius_rms=step.commutator_frobenius_rms,
                       density_closure_frobenius=step.maximum_density_closure_frobenius,
                       global_band_gap=step.global_band_gap,
                       solver_converged=step.converged, delivery='after-solve',
                       **(_spin_history(step, native) if unrestricted else {}))
            record('scf_terminal', f'  SCF status: {status}; iterations={result.iterations}; '
                   f'energy={result.energy:.12f} hartree.\n'
                   f'  Commutator RMS: {terminal.commutator_frobenius_rms:.6e}; '
                   f'density closure: {terminal.maximum_density_closure_frobenius:.6e}.\n',
                   converged=result.converged, iteration=result.iterations, energy=result.energy,
                   status=terminal.status.name, integral_provenance=evidence)
            bibtex = ''
            if a['citations']:
                refs = load_default_database().assemble(method=method, basis=basis.name, functional=a['functional'],
                    periodic=True, uses_integrals=False, scf_guess='hcore',
                    extra_methods=evidence['citation_methods'], numerics=evidence.get('citation_numerics', []))
                writer.set_citations(citation_manifest_rows(refs))
                writer.dispatch_role('citations', citations=refs, raise_on_error=True)
                write(format_references_block(refs))
                bibtex = format_bibtex(refs)
            writer.dispatch_role('geometry', system=system, molecule=system.unit_cell_molecule(), raise_on_error=True)
            if a['output_qvf']:
                bloch = _qvf_evaluated_density(result, basis, system, fractions, image_shifts)
                if bloch is None:
                    raise RuntimeError('Selected MANTA QVF could not encode the evaluated periodic density')
                writer.dispatch_role('qvf', system=system, basis=basis.name, result=result,
                    method=method, functional=a['functional'], job_kind='periodic', record_hostname=a['record_hostname'],
                    run_status='converged' if result.converged else 'failed',
                    run_fields=fields, exchange_q0=result.exchange_q0,
                    bloch_wf_data=bloch, extensions={'x_vibeqc': {'version': '1.0', 'critical': False}},
                    vendor_json_sections=[dict(id='periodic_integral_engine',
                        kind='x_vibeqc.periodic_integral_engine', payload=evidence)],
                    scf_history_data=result.scf_history,
                    bibtex_content=bibtex,
                    raise_on_error=True)
            record('job_end', '  Periodic MANTA job complete.\n', converged=result.converged)
        writer.record(out_path)
        if structured_target is not None:
            writer.record(structured_target)
    return result
