"""General velocity-Verlet molecular dynamics on an energy+gradient callable.

This is a **method-agnostic** Born-Oppenheimer MD driver: it integrates
Newton's equations for a callable ``force_fn(coords) -> (energy, gradient)``
and knows nothing about *how* the energy and gradient are produced. Any
vibe-qc method that exposes a potential-energy surface -- HF/DFT via the
molecular runner, or the MSINDO semiempirical engine via
:func:`vibeqc.md.msindo_force_provider` -- drives it through the same
interface, exactly like :mod:`vibeqc.neb` drives a band over an
energy+gradient seam.

Everything here works in **atomic units**: positions in bohr, energies in
Hartree, gradients in Ha/bohr, masses in electron masses, time in atomic
time units (ħ/E_h ≈ 0.0241888 fs). The public entry points accept the
timestep and the thermostat relaxation time in femtoseconds and convert
internally; trajectory velocities are stored in bohr / atomic-time-unit.

Ensembles
=========
* **NVE** (microcanonical) -- ``thermostat=None``; the symplectic
  velocity-Verlet integrator conserves the total energy with no secular
  drift.
* **NVT** (canonical) via either
  * a :class:`BerendsenThermostat` (weak-coupling velocity rescaling), or
  * a :class:`NoseHooverThermostat` (a single extended-system thermostat
    with a genuine conserved quantity).

References
==========
Velocity-Verlet integrator:
    Swope, Andersen, Berens & Wilson, "A computer simulation method for
    the calculation of equilibrium constants ...", J. Chem. Phys. 76,
    637 (1982). doi:10.1063/1.442716. (Velocity form of L. Verlet,
    Phys. Rev. 159, 98 (1967), doi:10.1103/PhysRev.159.98.)
Berendsen weak-coupling thermostat:
    Berendsen, Postma, van Gunsteren, DiNola & Haak, "Molecular dynamics
    with coupling to an external bath", J. Chem. Phys. 81, 3684 (1984).
    doi:10.1063/1.448118.
Nosé-Hoover thermostat:
    Nosé, "A unified formulation of the constant temperature molecular
    dynamics methods", J. Chem. Phys. 81, 511 (1984), doi:10.1063/1.447334;
    Hoover, "Canonical dynamics: Equilibrium phase-space distributions",
    Phys. Rev. A 31, 1695 (1985), doi:10.1103/PhysRevA.31.1695.

The driver's loop structure (velocity-Verlet bracketed by thermostat
half-steps; Berendsen rescale after the second half-kick) mirrors the
reference MSINDO BO-MD driver ``dynamics.f`` (algorithm only -- read for
structure, not copied): the position/half-velocity then second-half
update at ``dynamics.f:196,217`` (VELOCITY_VERLET_I / _II), the Berendsen
rescale at ``dynamics.f:200-203,231``, and the Nosé half-step brackets at
``dynamics.f:195,219``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .properties import _ATOMIC_MASSES

__all__ = [
    "ForceProvider",
    "MDTrajectory",
    "Thermostat",
    "BerendsenThermostat",
    "NoseHooverThermostat",
    "kinetic_energy",
    "temperature_from_kinetic",
    "maxwell_boltzmann_velocities",
    "remove_com_velocity",
    "run_md",
    "run_nve",
    "run_nvt",
]

# ----------------------------------------------------------------------
# Physical constants (CODATA 2018; values match python/vibeqc/thermo.py)
# ----------------------------------------------------------------------

_KB_J_PER_K: float = 1.380649e-23            # Boltzmann (J/K, exact since SI 2019)
_HARTREE_TO_J: float = 4.3597447222071e-18   # 1 Hartree in Joules
_KB_HARTREE_PER_K: float = _KB_J_PER_K / _HARTREE_TO_J  # ≈ 3.166811e-6 Ha/K
_AMU_TO_ELECTRON_MASS: float = 1822.888486209           # m_u / m_e
# Atomic unit of time (ħ/E_h) in femtoseconds.
_AU_TIME_IN_FS: float = 0.024188843265857


# ``force_fn(coords_bohr) -> (energy_Ha, gradient_Ha_per_bohr)``.
# ``coords`` is (n_atoms, 3) in bohr; the returned gradient has the same
# shape. Forces are the negative gradient -- the driver negates internally,
# so providers return the *gradient* dE/dR (matching run_neb's convention).
ForceProvider = Callable[[np.ndarray], Tuple[float, np.ndarray]]


# ----------------------------------------------------------------------
# Mass / kinetic helpers
# ----------------------------------------------------------------------


def _resolve_masses_electron(
    n_atoms: int,
    atomic_numbers: Optional[Sequence[int]],
    masses_amu: Optional[np.ndarray],
) -> np.ndarray:
    """Per-atom masses in electron-mass units, shape ``(n_atoms,)``.

    Either ``masses_amu`` (explicit, amu) or ``atomic_numbers`` (looked up
    in the standard atomic-weight table) must be given.
    """
    if masses_amu is not None:
        m = np.asarray(masses_amu, dtype=float).reshape(-1)
    elif atomic_numbers is not None:
        out = []
        for z in atomic_numbers:
            zi = int(z)
            if zi <= 0 or zi >= len(_ATOMIC_MASSES) or _ATOMIC_MASSES[zi] <= 0:
                raise ValueError(
                    f"no tabulated atomic mass for Z={zi}; pass masses_amu="
                    "[...] explicitly."
                )
            out.append(_ATOMIC_MASSES[zi])
        m = np.array(out, dtype=float)
    else:
        raise ValueError("provide either atomic_numbers or masses_amu")
    if m.shape[0] != n_atoms:
        raise ValueError(
            f"masses length {m.shape[0]} != number of atoms {n_atoms}"
        )
    if np.any(m <= 0.0):
        raise ValueError("atomic masses must be positive")
    return m * _AMU_TO_ELECTRON_MASS


def kinetic_energy(velocities: np.ndarray, masses_electron: np.ndarray) -> float:
    """Kinetic energy (Hartree) for velocities (bohr/aut) + masses (m_e).

    ``KE = 1/2 S_i m_i |v_i|^2``.
    """
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses_electron, dtype=float)
    return 0.5 * float(np.sum(m[:, None] * v * v))


def temperature_from_kinetic(ke_hartree: float, n_dof: int) -> float:
    """Instantaneous temperature (K) from kinetic energy via equipartition.

    ``T = 2.KE / (N_dof . k_B)``.
    """
    if n_dof <= 0:
        return 0.0
    return 2.0 * ke_hartree / (n_dof * _KB_HARTREE_PER_K)


def remove_com_velocity(
    velocities: np.ndarray, masses_electron: np.ndarray
) -> np.ndarray:
    """Subtract the centre-of-mass velocity (zero total linear momentum)."""
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses_electron, dtype=float)
    v_com = np.sum(m[:, None] * v, axis=0) / np.sum(m)
    return v - v_com[None, :]


def maxwell_boltzmann_velocities(
    masses_electron: np.ndarray,
    temperature_K: float,
    *,
    rng: Optional[np.random.Generator] = None,
    remove_com: bool = True,
    n_dof: Optional[int] = None,
    exact_temperature: bool = True,
) -> np.ndarray:
    """Sample velocities (bohr/aut) from a Maxwell-Boltzmann distribution.

    Each Cartesian component is drawn from a Gaussian with variance
    ``k_B T / m_i``. With ``remove_com`` the centre-of-mass velocity is
    projected out; with ``exact_temperature`` the result is rescaled so
    the instantaneous temperature equals ``temperature_K`` exactly (using
    ``n_dof`` degrees of freedom, defaulting to ``3N - 3`` when COM is
    removed, else ``3N``).
    """
    if rng is None:
        rng = np.random.default_rng()
    m = np.asarray(masses_electron, dtype=float)
    n_atoms = m.shape[0]
    if temperature_K <= 0.0:
        return np.zeros((n_atoms, 3), dtype=float)
    sigma = np.sqrt(_KB_HARTREE_PER_K * temperature_K / m)  # (n,)
    v = rng.standard_normal((n_atoms, 3)) * sigma[:, None]
    if remove_com and n_atoms > 1:
        v = remove_com_velocity(v, m)
    if n_dof is None:
        n_dof = max(1, 3 * n_atoms - (3 if remove_com and n_atoms > 1 else 0))
    if exact_temperature:
        t_now = temperature_from_kinetic(kinetic_energy(v, m), n_dof)
        if t_now > 0.0:
            v *= np.sqrt(temperature_K / t_now)
    return v


# ----------------------------------------------------------------------
# Thermostats
# ----------------------------------------------------------------------


class Thermostat:
    """Base class -- a no-op (NVE / microcanonical).

    A thermostat is invoked twice per MD step: :meth:`pre` immediately
    before the velocity-Verlet position update, and :meth:`post`
    immediately after the second half-kick. The base implementation does
    nothing (NVE). :meth:`bath_energy` returns the extended-system energy
    that must be *added* to ``KE + PE`` to form the conserved quantity
    (zero for NVE / Berendsen).
    """

    name = "nve"

    def pre(
        self, v: np.ndarray, masses_electron: np.ndarray, dt_au: float, n_dof: int
    ) -> np.ndarray:
        return v

    def post(
        self, v: np.ndarray, masses_electron: np.ndarray, dt_au: float, n_dof: int
    ) -> np.ndarray:
        return v

    def bath_energy(self) -> float:
        return 0.0


class BerendsenThermostat(Thermostat):
    """Berendsen weak-coupling thermostat (Berendsen et al. 1984).

    After each step the velocities are rescaled by

        l = sqrt(1 + (Δt / t_T) . (T_target / T - 1))

    (J. Chem. Phys. 81, 3684 (1984), eq. 11), which relaxes the kinetic
    temperature toward ``temperature_K`` with a time constant
    ``tau_fs``. This is a robust, strongly-damping thermostat that does
    **not** sample the exact canonical ensemble (no fluctuation theorem)
    and has no conserved quantity -- use :class:`NoseHooverThermostat`
    when a rigorous NVT distribution matters.
    """

    name = "berendsen"

    def __init__(self, temperature_K: float, tau_fs: float = 50.0):
        if tau_fs <= 0.0:
            raise ValueError("Berendsen tau_fs must be positive")
        self.temperature_K = float(temperature_K)
        self.tau_au = float(tau_fs) / _AU_TIME_IN_FS

    def post(
        self, v: np.ndarray, masses_electron: np.ndarray, dt_au: float, n_dof: int
    ) -> np.ndarray:
        t_now = temperature_from_kinetic(
            kinetic_energy(v, masses_electron), n_dof
        )
        if t_now <= 0.0:
            return v
        # l^2 = 1 + (Δt/t)(T0/T - 1); clamp the radicand so a large cold
        # excursion can't request an imaginary rescale.
        lam2 = 1.0 + (dt_au / self.tau_au) * (self.temperature_K / t_now - 1.0)
        lam2 = max(lam2, 1e-8)
        return v * np.sqrt(lam2)


class NoseHooverThermostat(Thermostat):
    """Single Nosé-Hoover extended-system thermostat (Nosé 1984 / Hoover 1985).

    A friction variable ζ obeys ``Q ζ̇ = 2.KE - N_dof.k_B.T`` and damps the
    velocities (``v̇ = a - ζ v``); the thermostat "mass"
    ``Q = N_dof . k_B . T . t^2`` sets the coupling time ``tau_fs``. The
    equations of motion are integrated with the symmetric Trotter
    half-step below (the single-thermostat case of the measure-preserving
    Nosé-Hoover-chain integrator), applied as :meth:`pre` / :meth:`post`
    brackets around the plain velocity-Verlet update -- the same operator
    split the reference driver uses (``dynamics.f:195,219``).

    Unlike Berendsen this generates the true canonical distribution and
    carries a conserved quantity ``H_NH = KE + PE + 1/2 Q ζ^2 + N_dof k_B T η``
    (:meth:`bath_energy` returns the last two terms), which a correct
    integration keeps constant up to ``O(Δt^2)``.
    """

    name = "nose_hoover"

    def __init__(self, temperature_K: float, tau_fs: float = 50.0):
        if tau_fs <= 0.0:
            raise ValueError("Nosé-Hoover tau_fs must be positive")
        self.temperature_K = float(temperature_K)
        self.tau_au = float(tau_fs) / _AU_TIME_IN_FS
        self.zeta = 0.0   # friction (thermostat "velocity")
        self.eta = 0.0    # thermostat coordinate (for the conserved energy)
        self._kT = _KB_HARTREE_PER_K * self.temperature_K
        self._Q: Optional[float] = None  # set lazily once n_dof is known

    def _ensure_q(self, n_dof: int) -> None:
        if self._Q is None:
            self._Q = max(1, n_dof) * self._kT * self.tau_au * self.tau_au

    def _half(
        self, v: np.ndarray, masses_electron: np.ndarray, dt_au: float, n_dof: int
    ) -> np.ndarray:
        # Symmetric half-step (M=1 Nosé-Hoover chain): two quarter-steps on
        # ζ bracketing a half-step velocity scale + a half-step on η.
        self._ensure_q(n_dof)
        Q = self._Q
        kT = self._kT
        ke = kinetic_energy(v, masses_electron)
        g = (2.0 * ke - n_dof * kT) / Q
        self.zeta += 0.25 * dt_au * g
        scale = float(np.exp(-0.5 * dt_au * self.zeta))
        v = v * scale
        self.eta += 0.5 * dt_au * self.zeta
        ke *= scale * scale
        g = (2.0 * ke - n_dof * kT) / Q
        self.zeta += 0.25 * dt_au * g
        return v

    def pre(
        self, v: np.ndarray, masses_electron: np.ndarray, dt_au: float, n_dof: int
    ) -> np.ndarray:
        return self._half(v, masses_electron, dt_au, n_dof)

    def post(
        self, v: np.ndarray, masses_electron: np.ndarray, dt_au: float, n_dof: int
    ) -> np.ndarray:
        return self._half(v, masses_electron, dt_au, n_dof)

    def bath_energy(self) -> float:
        if self._Q is None:
            return 0.0
        n_dof_kt_eta = (self._Q / (self.tau_au * self.tau_au)) * self.eta
        return 0.5 * self._Q * self.zeta * self.zeta + n_dof_kt_eta


def _make_thermostat(
    thermostat: Union[None, str, Thermostat],
    temperature_K: float,
    tau_fs: float,
) -> Thermostat:
    if thermostat is None:
        return Thermostat()
    if isinstance(thermostat, Thermostat):
        return thermostat
    key = str(thermostat).strip().lower().replace("-", "_")
    if key in ("nve", "none"):
        return Thermostat()
    if key == "berendsen":
        return BerendsenThermostat(temperature_K, tau_fs)
    if key in ("nose_hoover", "nosehoover", "nose"):
        return NoseHooverThermostat(temperature_K, tau_fs)
    raise ValueError(
        f"unknown thermostat {thermostat!r}; use None, 'berendsen', "
        "'nose_hoover', or a Thermostat instance"
    )


# ----------------------------------------------------------------------
# Trajectory container
# ----------------------------------------------------------------------


@dataclass
class MDTrajectory:
    """Recorded MD trajectory (one entry per recorded frame).

    All arrays have the recorded frames along axis 0. Positions and
    velocities are in atomic units (bohr, bohr/aut); energies in Hartree;
    temperature in Kelvin; times in femtoseconds.
    """

    times_fs: np.ndarray                 # (F,)
    positions: np.ndarray                # (F, n_atoms, 3) bohr
    velocities: np.ndarray               # (F, n_atoms, 3) bohr/aut
    potential_energy: np.ndarray         # (F,) Ha
    kinetic_energy: np.ndarray           # (F,) Ha
    total_energy: np.ndarray             # (F,) Ha (KE + PE)
    temperature: np.ndarray              # (F,) K
    conserved_energy: np.ndarray         # (F,) Ha (total + thermostat bath)
    n_dof: int
    timestep_fs: float
    thermostat: str

    def energy_drift(self) -> float:
        """Max absolute deviation of the conserved quantity from its
        initial value (Ha). For NVE this is the total-energy drift; for
        Nosé-Hoover it is the extended-system (H_NH) drift."""
        c = self.conserved_energy
        return float(np.max(np.abs(c - c[0]))) if c.size else 0.0

    def mean_temperature(self, skip_fraction: float = 0.5) -> float:
        """Mean temperature over the last ``1 - skip_fraction`` of frames
        (discarding an initial equilibration segment)."""
        n = self.temperature.size
        if n == 0:
            return 0.0
        start = int(np.clip(skip_fraction, 0.0, 0.99) * n)
        return float(np.mean(self.temperature[start:]))


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_md(
    force_fn: ForceProvider,
    coords: np.ndarray,
    *,
    atomic_numbers: Optional[Sequence[int]] = None,
    masses_amu: Optional[np.ndarray] = None,
    timestep_fs: float = 0.5,
    n_steps: int = 100,
    temperature_K: float = 300.0,
    thermostat: Union[None, str, Thermostat] = None,
    thermostat_tau_fs: float = 50.0,
    velocities: Optional[np.ndarray] = None,
    remove_com: bool = True,
    n_dof: Optional[int] = None,
    seed: Optional[int] = None,
    record_stride: int = 1,
    callback: Optional[Callable[[int, dict], None]] = None,
) -> MDTrajectory:
    """Run Born-Oppenheimer MD on an energy+gradient callable.

    Parameters
    ----------
    force_fn
        ``force_fn(coords_bohr) -> (energy_Ha, gradient_Ha_per_bohr)``.
        ``coords`` is ``(n_atoms, 3)`` in bohr; the gradient has the same
        shape. (Forces are ``-gradient``; the driver negates internally.)
    coords
        Initial positions ``(n_atoms, 3)`` in bohr.
    atomic_numbers, masses_amu
        Supply exactly one to set the per-atom masses (Z-table lookup, or
        explicit amu). ``masses_amu`` wins if both are given.
    timestep_fs
        MD timestep in femtoseconds.
    n_steps
        Number of integration steps.
    temperature_K
        Target temperature for the thermostat, and the temperature used to
        sample initial velocities when ``velocities is None``.
    thermostat
        ``None`` -> NVE; ``"berendsen"`` / ``"nose_hoover"`` -> that NVT
        thermostat (constructed with ``thermostat_tau_fs``); or a
        :class:`Thermostat` instance.
    thermostat_tau_fs
        Thermostat relaxation/coupling time (fs) when ``thermostat`` is a
        string.
    velocities
        Optional initial velocities ``(n_atoms, 3)`` in bohr/aut. ``None``
        -> Maxwell-Boltzmann sample at ``temperature_K``.
    remove_com
        Project out the centre-of-mass velocity (every step), so the COM
        stays at rest and ``N_dof = 3N - 3``. Set ``False`` for an
        externally anchored potential (then ``N_dof = 3N``).
    n_dof
        Override the degrees-of-freedom count used for the temperature and
        the thermostat. Defaults as above.
    seed
        Seed for the Maxwell-Boltzmann RNG (reproducible velocities).
    record_stride
        Record a frame every ``record_stride`` steps (frame 0 and the
        final frame are always recorded).
    callback
        Optional ``callback(step, info)`` invoked after every step with a
        dict of ``{"coords", "velocities", "e_pot", "e_kin", "temperature"}``
        -- used by the metadynamics driver to inject bias forces / deposit
        hills without re-implementing the integrator.

    Returns
    -------
    MDTrajectory
    """
    if (
        isinstance(record_stride, (bool, np.bool_))
        or not isinstance(record_stride, (int, np.integer))
        or record_stride < 1
    ):
        raise ValueError("run_md: record_stride must be a positive integer")
    record_stride = int(record_stride)

    x = np.array(coords, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    n_atoms = x.shape[0]
    masses_e = _resolve_masses_electron(n_atoms, atomic_numbers, masses_amu)

    if n_dof is None:
        n_dof = max(1, 3 * n_atoms - (3 if remove_com and n_atoms > 1 else 0))

    thermo = _make_thermostat(thermostat, temperature_K, thermostat_tau_fs)

    rng = np.random.default_rng(seed)
    if velocities is None:
        v = maxwell_boltzmann_velocities(
            masses_e, temperature_K, rng=rng, remove_com=remove_com,
            n_dof=n_dof,
        )
    else:
        v = np.array(velocities, dtype=float)
        if v.shape != x.shape:
            raise ValueError("velocities must match coords shape")
        if remove_com and n_atoms > 1:
            v = remove_com_velocity(v, masses_e)

    dt_au = float(timestep_fs) / _AU_TIME_IN_FS

    # Initial energy + acceleration (a = -gradE / m).
    e_pot, grad = _eval(force_fn, x)
    accel = -np.asarray(grad, dtype=float) / masses_e[:, None]

    # Frame buffers.
    times: list[float] = []
    pos_frames: list[np.ndarray] = []
    vel_frames: list[np.ndarray] = []
    epot_frames: list[float] = []
    ekin_frames: list[float] = []
    etot_frames: list[float] = []
    temp_frames: list[float] = []
    cons_frames: list[float] = []

    def _record(step: int, e_pot_now: float, v_now: np.ndarray) -> None:
        ke = kinetic_energy(v_now, masses_e)
        temp = temperature_from_kinetic(ke, n_dof)
        times.append(step * timestep_fs)
        pos_frames.append(x.copy())
        vel_frames.append(v_now.copy())
        epot_frames.append(float(e_pot_now))
        ekin_frames.append(ke)
        etot_frames.append(float(e_pot_now) + ke)
        temp_frames.append(temp)
        cons_frames.append(float(e_pot_now) + ke + thermo.bath_energy())

    _record(0, e_pot, v)

    for step in range(1, n_steps + 1):
        # Thermostat half-step (Nosé-Hoover); no-op for NVE / Berendsen.
        v = thermo.pre(v, masses_e, dt_au, n_dof)
        # Velocity-Verlet (Swope 1982): half-kick, drift, recompute force,
        # second half-kick. Mirrors dynamics.f:196 (_I) / :217 (_II).
        v = v + 0.5 * dt_au * accel
        x = x + dt_au * v
        e_pot, grad = _eval(force_fn, x)
        accel = -np.asarray(grad, dtype=float) / masses_e[:, None]
        v = v + 0.5 * dt_au * accel
        # Thermostat half-step (Nosé-Hoover) / rescale (Berendsen).
        v = thermo.post(v, masses_e, dt_au, n_dof)
        if remove_com and n_atoms > 1:
            v = remove_com_velocity(v, masses_e)

        if callback is not None:
            callback(
                step,
                {
                    "coords": x,
                    "velocities": v,
                    "e_pot": float(e_pot),
                    "e_kin": kinetic_energy(v, masses_e),
                    "temperature": temperature_from_kinetic(
                        kinetic_energy(v, masses_e), n_dof
                    ),
                },
            )

        if step % record_stride == 0 or step == n_steps:
            _record(step, e_pot, v)

    return MDTrajectory(
        times_fs=np.array(times, dtype=float),
        positions=np.array(pos_frames, dtype=float),
        velocities=np.array(vel_frames, dtype=float),
        potential_energy=np.array(epot_frames, dtype=float),
        kinetic_energy=np.array(ekin_frames, dtype=float),
        total_energy=np.array(etot_frames, dtype=float),
        temperature=np.array(temp_frames, dtype=float),
        conserved_energy=np.array(cons_frames, dtype=float),
        n_dof=n_dof,
        timestep_fs=float(timestep_fs),
        thermostat=thermo.name,
    )


def _eval(force_fn: ForceProvider, x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Call the provider and validate its (energy, gradient) shape."""
    e, g = force_fn(x)
    g = np.asarray(g, dtype=float)
    if g.shape != x.shape:
        raise ValueError(
            f"force_fn gradient shape {g.shape} != coords shape {x.shape}"
        )
    return float(e), g


def run_nve(force_fn: ForceProvider, coords: np.ndarray, **kwargs) -> MDTrajectory:
    """Shorthand for microcanonical (NVE) MD -- :func:`run_md` with
    ``thermostat=None``."""
    kwargs.pop("thermostat", None)
    return run_md(force_fn, coords, thermostat=None, **kwargs)


def run_nvt(
    force_fn: ForceProvider,
    coords: np.ndarray,
    *,
    temperature_K: float = 300.0,
    thermostat: Union[str, Thermostat] = "nose_hoover",
    **kwargs,
) -> MDTrajectory:
    """Shorthand for canonical (NVT) MD -- :func:`run_md` with a thermostat
    (default Nosé-Hoover)."""
    return run_md(
        force_fn, coords, temperature_K=temperature_K,
        thermostat=thermostat, **kwargs,
    )
