"""Born-Oppenheimer molecular dynamics for the GAPW periodic route.

Uses the :class:`VibeqcGAPW` ASE calculator (:mod:`vibeqc.ase_periodic_gapw`)
under the hood, driving ASE's built-in MD engines (VelocityVerlet,
Langevin, NPTBerendsen) for NVE, NVT, and NPT ensembles.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from ase import units
    from ase.atoms import Atoms
    from ase.md.langevin import Langevin
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
    )
    from ase.md.verlet import VelocityVerlet
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "vibeqc.periodic_gapw_md requires ASE. Install it with "
        "`pip install ase` into your vibe-qc venv, or "
        "`pip install -e '.[ase]'` from the repo checkout."
    ) from exc

from .ase_periodic_gapw import VibeqcGAPW
from .output import active_channel, render_energy_labeled, write
from .progress import resolve_progress

_log = logging.getLogger("vibeqc.periodic_gapw_md")

__all__ = [
    "run_md",
    "run_nve",
    "run_nvt",
    "run_npt",
]

# Default Langevin friction in inverse ASE time units.
# 0.01 / fs ≈ 10 ps⁻¹ -- typical condensed-phase NVT damping rate,
# corresponding to a relaxation time of ~100 fs.
_LANGEVIN_FRICTION_DEFAULT = 0.01 / units.fs

# Default NPTBerendsen coupling time constants (ASE internal units).
_NPT_TAUT_DEFAULT = 50.0 * units.fs  # temperature coupling (≈50 fs)
_NPT_TAUP_DEFAULT = 100.0 * units.fs  # pressure coupling (≈100 fs)
# Default isothermal compressibility (~4.5e-5 bar⁻¹, water-like).
_NPT_COMPRESSIBILITY_DEFAULT = 4.5e-5 / units.GPa


def _format_md_status(
    step: int,
    e_pot_ev: float,
    e_tot_ev: float,
    temperature_k: float,
) -> str:
    """Render one ASE-native MD status row through the output policy."""
    e_pot_ha = e_pot_ev / units.Hartree
    e_tot_ha = e_tot_ev / units.Hartree
    e_pot = render_energy_labeled(e_pot_ha, width=0, precision=6, sign=True)
    e_tot = render_energy_labeled(e_tot_ha, width=0, precision=6, sign=True)
    return (
        f"step {step:4d}:  E_pot = {e_pot}  E_tot = {e_tot}  T = {temperature_k:6.1f} K"
    )


def _emit_md_status(plog: Any, status: str) -> None:
    """Emit status persistently when a job channel is installed."""
    if active_channel() is not None:
        write(f"  {status}\n")
    else:
        plog.info(status)


def run_md(
    atoms: Atoms,
    *,
    md_engine: str = "nvt",
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    n_steps: int = 100,
    log_interval: int = 10,
    trajectory_file: Optional[str] = None,
    **calculator_kwargs: Any,
) -> Atoms:
    """Run Born-Oppenheimer molecular dynamics via the GAPW route.

    Attaches a :class:`VibeqcGAPW` calculator to *atoms*, initialises
    velocities with a Maxwell-Boltzmann distribution at *temperature_K*,
    removes centre-of-mass drift, and runs an ASE MD engine for
    *n_steps* steps. The same *atoms* object (with updated positions,
    velocities, and attached calculator) is returned.

    Parameters
    ----------
    atoms
        The atomic system (must be 3D-periodic). Modified in place.
    md_engine
        ``"nve"`` -> :class:`~ase.md.verlet.VelocityVerlet`;
        ``"nvt"`` -> :class:`~ase.md.langevin.Langevin`;
        ``"npt"`` -> :class:`~ase.md.nptberendsen.NPTBerendsen`
        (ambient pressure).
    temperature_K
        Target temperature for the thermostat (NVT / NPT), or the
        initial temperature for velocity sampling (NVE). [K]
    timestep_fs
        MD timestep. [fs]
    n_steps
        Number of MD steps to run.
    log_interval
        Print energy and temperature diagnostics every *log_interval*
        steps. Set to 0 to suppress logging.
    trajectory_file
        Path to an ASE trajectory file (``.traj`` extension). Frames
        are written every *log_interval* steps. ``None`` disables
        trajectory output.

    **calculator_kwargs
        Forwarded verbatim to :class:`VibeqcGAPW` -- e.g. *basis*,
        *functional*, *kmesh*, *cutoff_ha*, *gapw_kwargs*, etc.

    Returns
    -------
    :class:`~ase.atoms.Atoms`
        The same *atoms* object, modified in place.
    """
    atoms.calc = VibeqcGAPW(**calculator_kwargs)

    # Initialise velocities at target temperature.
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)  # remove centre-of-mass momentum

    dt = timestep_fs * units.fs  # ASE internal time units

    _log.info(
        "GAPW MD: engine=%s  T=%.1f K  dt=%.1f fs  steps=%d  log_interval=%d",
        md_engine,
        temperature_K,
        timestep_fs,
        n_steps,
        log_interval,
    )
    plog = resolve_progress(log_interval > 0, verbose=2)

    engine_lower = md_engine.lower()

    if engine_lower == "nve":
        dyn = VelocityVerlet(
            atoms,
            timestep=dt,
            trajectory=trajectory_file,
            logfile=None,
            loginterval=log_interval,
        )
    elif engine_lower == "nvt":
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=temperature_K,
            friction=_LANGEVIN_FRICTION_DEFAULT,
            trajectory=trajectory_file,
            logfile=None,
            loginterval=log_interval,
        )
    elif engine_lower == "npt":
        dyn = NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature_K,
            pressure=0.0,  # ambient (0 GPa) by default
            taut=_NPT_TAUT_DEFAULT,
            taup=_NPT_TAUP_DEFAULT,
            compressibility=_NPT_COMPRESSIBILITY_DEFAULT,
            trajectory=trajectory_file,
            logfile=None,
            loginterval=log_interval,
        )
    else:
        raise ValueError(
            f"Unknown md_engine={md_engine!r}. Expected 'nve', 'nvt', or 'npt'."
        )

    # ---- Per-step diagnostics -----------------------------------
    def _log_status() -> None:
        step = dyn.get_number_of_steps()
        e_kin = atoms.get_kinetic_energy()
        e_pot = atoms.get_potential_energy(force_consistent=False)
        e_tot = e_kin + e_pot
        # Degrees of freedom: 3N - 3 (subtract COM translation).
        n_dof = 3 * len(atoms) - 3
        T = 2.0 * e_kin / (n_dof * units.kB) if n_dof > 0 else 0.0
        _emit_md_status(plog, _format_md_status(step, e_pot, e_tot, T))

    if log_interval > 0:
        dyn.attach(_log_status, interval=log_interval)

    dyn.run(n_steps)

    return atoms


def run_nve(
    atoms: Atoms,
    *,
    timestep_fs: float = 1.0,
    n_steps: int = 100,
    **calculator_kwargs: Any,
) -> Atoms:
    """Shorthand for NVE (microcanonical) MD via :func:`run_md`.

    Parameters
    ----------
    atoms
        The atomic system (must be 3D-periodic). Modified in place.
    timestep_fs
        MD timestep. [fs]
    n_steps
        Number of MD steps to run.

    **calculator_kwargs
        Forwarded to :class:`VibeqcGAPW`.

    Returns
    -------
    :class:`~ase.atoms.Atoms`
        The same *atoms* object, modified in place.
    """
    return run_md(
        atoms,
        md_engine="nve",
        temperature_K=300.0,
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        log_interval=10,
        trajectory_file=None,
        **calculator_kwargs,
    )


def run_nvt(
    atoms: Atoms,
    *,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    n_steps: int = 100,
    **calculator_kwargs: Any,
) -> Atoms:
    """Shorthand for NVT (canonical) MD via :func:`run_md`.

    Uses the Langevin thermostat with a default friction of
    0.01 fs⁻¹ (≈10 ps⁻¹).

    Parameters
    ----------
    atoms
        The atomic system (must be 3D-periodic). Modified in place.
    temperature_K
        Target temperature for the Langevin thermostat. [K]
    timestep_fs
        MD timestep. [fs]
    n_steps
        Number of MD steps to run.

    **calculator_kwargs
        Forwarded to :class:`VibeqcGAPW`.

    Returns
    -------
    :class:`~ase.atoms.Atoms`
        The same *atoms* object, modified in place.
    """
    return run_md(
        atoms,
        md_engine="nvt",
        temperature_K=temperature_K,
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        log_interval=10,
        trajectory_file=None,
        **calculator_kwargs,
    )


def run_npt(
    atoms: Atoms,
    *,
    temperature_K: float = 300.0,
    pressure_GPa: float = 0.0,
    timestep_fs: float = 1.0,
    n_steps: int = 100,
    **calculator_kwargs: Any,
) -> Atoms:
    """Shorthand for NPT (isothermal-isobaric) MD.

    Uses the Berendsen barostat + thermostat
    (:class:`~ase.md.nptberendsen.NPTBerendsen`).  Pressure is
    specified in GPa and converted internally to ASE units.

    Parameters
    ----------
    atoms
        The atomic system (must be 3D-periodic). Modified in place.
    temperature_K
        Target temperature. [K]
    pressure_GPa
        Target pressure. [GPa]
    timestep_fs
        MD timestep. [fs]
    n_steps
        Number of MD steps to run.

    **calculator_kwargs
        Forwarded to :class:`VibeqcGAPW`.

    Returns
    -------
    :class:`~ase.atoms.Atoms`
        The same *atoms* object, modified in place.
    """
    atoms.calc = VibeqcGAPW(**calculator_kwargs)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)

    dt = timestep_fs * units.fs
    pressure_ase = pressure_GPa * units.GPa  # GPa -> eV/Å^3

    _log.info(
        "GAPW MD: engine=npt  T=%.1f K  P=%.4f GPa  dt=%.1f fs  steps=%d",
        temperature_K,
        pressure_GPa,
        timestep_fs,
        n_steps,
    )
    plog = resolve_progress(True, verbose=2)

    dyn = NPTBerendsen(
        atoms,
        timestep=dt,
        temperature_K=temperature_K,
        pressure=pressure_ase,
        taut=_NPT_TAUT_DEFAULT,
        taup=_NPT_TAUP_DEFAULT,
        compressibility=_NPT_COMPRESSIBILITY_DEFAULT,
        trajectory=None,
        logfile=None,
        loginterval=10,
    )

    # ---- Per-step diagnostics -----------------------------------
    def _log_status() -> None:
        step = dyn.get_number_of_steps()
        e_kin = atoms.get_kinetic_energy()
        e_pot = atoms.get_potential_energy(force_consistent=False)
        e_tot = e_kin + e_pot
        n_dof = 3 * len(atoms) - 3
        T = 2.0 * e_kin / (n_dof * units.kB) if n_dof > 0 else 0.0
        _emit_md_status(plog, _format_md_status(step, e_pot, e_tot, T))

    dyn.attach(_log_status, interval=10)
    dyn.run(n_steps)

    return atoms
