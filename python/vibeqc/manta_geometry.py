"""Geometry-local evaluations of an explicitly selected molecular HF source."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import _vibeqc_core as core
from .manta_scf import IntegralEngineOptions, CoulombBuildOptions, ExchangeBuildOptions


def _shell_template(molecule, basis):
    atoms = molecule.atoms
    shells = basis.shells()
    for shell in shells:
        owner = shell.atom_index
        if owner < 0 or owner >= len(atoms) or list(shell.origin) != list(atoms[owner].xyz):
            raise ValueError("Geometry source basis centers/atom map differ from the initial molecule")
    return shells


def _move_basis(molecule, shells, name):
    moved = []
    atoms = molecule.atoms
    for template in shells:
        shell = core.ShellInfo()
        shell.atom_index, shell.l, shell.pure = template.atom_index, template.l, template.pure
        shell.exponents, shell.coefficients = template.exponents, template.coefficients
        shell.origin = atoms[shell.atom_index].xyz
        moved.append(shell)
    return core.BasisSet(molecule, moved, name, True)


class MolecularHFGeometrySource:
    """Callable energy/gradient source for ``vibeqc.geomopt.run_geomopt``.

    Snapshot orbital/auxiliary shell content and atom ownership at the initial
    molecule, then rebuild their centers at each geometry. RHF/UHF support
    independently selected exact/RI J and K with MANTA, or explicit exact
    libint reference integrals. Each geometry starts from the shared guess the
    options select (HCORE when no options are given); no projected restart
    or unqualified derivative source is substituted. An optional
    ``dispersion_params`` (D3(BJ) parameters or a D4 specification) adds the
    dispersion energy and gradient to the reported surface, so the optimizer
    walks the same surface whose total ``run_job`` reports.

    Effective core potentials (an XML library selected on the options or the
    basis sidecar's inline primitives) are admitted with exact or RI J and K.
    The centres are absolute coordinates on the options, so they follow their
    atoms at every geometry. The gradient of a MANTA ECP geometry is the
    selected Lagrangian gradient, which differentiates V_ne(Z_eff) and the
    effective nuclear repulsion with the SCF's effective charges and adds the
    ECP provider's own operator derivative; a libint-reference ECP result
    takes the ordinary native path, which carries the same terms.

    The original integral budget applies per SCF build and, for MANTA, to
    aggregate gradient admission. As for the ordinary SCF interface, retained
    Python data and optimizer allocations are outside that integral budget.
    """

    def __init__(self, molecule, basis, *, method="rhf", options=None,
                 integral_engine=None, coulomb=None, exchange=None, dispersion_params=None,
                 dft_plus_u=None):
        self._method = str(method).lower()
        self._dispersion_params = dispersion_params
        # Hubbard sites are applied by the shared hosts at every geometry and
        # their explicit overlap-derivative term is added to the gradient on
        # the executed one-electron provider (vibeqc.compute_gradient).
        self._dft_plus_u = list(dft_plus_u) if dft_plus_u else None
        if self._method not in ("rhf", "uhf"):
            raise ValueError("Selected geometry sources require molecular RHF or UHF")
        engine = IntegralEngineOptions(backend="manta") if integral_engine is None else integral_engine
        j = CoulombBuildOptions() if coulomb is None else coulomb
        k = ExchangeBuildOptions() if exchange is None else exchange
        if not isinstance(engine, IntegralEngineOptions):
            raise TypeError("integral_engine must be IntegralEngineOptions")
        if not isinstance(j, CoulombBuildOptions) or not isinstance(k, ExchangeBuildOptions):
            raise TypeError("coulomb/exchange must use CoulombBuildOptions/ExchangeBuildOptions")
        if j.algorithm not in ("exact", "ri") or k.algorithm not in ("exact", "ri"):
            raise ValueError("Selected geometry sources require qualified exact/RI HF derivatives")
        if engine.backend not in ("manta", "libint") or engine.fallback != "error":
            raise ValueError("Selected geometry sources require manta/libint with fallback='error'")
        fitted = "ri" in (j.algorithm, k.algorithm)
        if engine.backend == "libint" and fitted:
            raise ValueError("The selected libint geometry reference requires exact J and K")
        if fitted != (j.auxiliary_basis is not None):
            raise ValueError("An auxiliary basis is required exactly when J or K selects RI")
        cls = core.RHFOptions if self._method == "rhf" else core.UHFOptions
        if options is not None and not isinstance(options, cls):
            raise TypeError(f"{self._method} options must be {cls.__name__}")
        self._options = cls() if options is None else cls(options)
        if options is None:
            self._options.initial_guess = core.InitialGuess.HCORE
        if self._options.initial_guess in (core.InitialGuess.READ, core.InitialGuess.FRAGMO):
            raise ValueError("Selected geometry sources take the shared HCORE/SAD/SAP/PATOM/HUECKEL/MINAO/AUTO "
                             "guesses at every geometry; READ projection and FRAGMO remain unqualified")

        # Native assignment copies these mutable structs. The frozen Python
        # configuration then refers only to this source's owned copies.
        self._context = core._manta._MolecularHFEngine()
        self._context.budget, self._context.ri = engine.budget, j.metric
        self._engine = replace(engine, budget=self._context.budget)
        self._j = CoulombBuildOptions(j.algorithm, metric=self._context.ri)
        self._k = ExchangeBuildOptions(k.algorithm)
        self._composition = (tuple(a.Z for a in molecule.atoms), molecule.charge, molecule.multiplicity)
        orbital = core.BasisSet(molecule, basis) if isinstance(basis, str) else basis
        if not isinstance(orbital, core.BasisSet):
            raise TypeError("basis must be a basis name or BasisSet")
        from . import _attach_inline_ecp_from_basis_sidecar
        from .ecp_metadata import ecp_centre_atom_indices, options_carry_ecp
        _attach_inline_ecp_from_basis_sidecar(self._options, molecule, orbital)
        self._ecp = options_carry_ecp(self._options)
        if self._ecp:
            # Exact-J/K ECP results take the ordinary native gradient path;
            # fitted ones the selected RI Lagrangian gradient with the SCF's
            # effective charges and the ECP provider's derivative. Both read
            # the ECP from the gradient options derived from the SCF options.
            # Every centre must sit on an atom of the reference geometry so
            # it can follow that atom; this is the SCF's own consistency rule.
            ecp_centre_atom_indices(self._options, molecule)
        self._reference = molecule
        self._orbital = _shell_template(molecule, orbital)
        self._orbital_name = orbital.name
        auxiliary = j.auxiliary_basis
        if isinstance(auxiliary, str):
            auxiliary = core.BasisSet(molecule, auxiliary)
        if auxiliary is not None and not isinstance(auxiliary, core.BasisSet):
            raise TypeError("auxiliary_basis must be a basis name or BasisSet")
        self._auxiliary = None if auxiliary is None else _shell_template(molecule, auxiliary)
        self._auxiliary_name = "" if auxiliary is None else auxiliary.name
        self._last_result = self._last_basis = None

    @property
    def options(self):
        """The source's own SCF options (ECP fields attached); read them, do not mutate."""
        return self._options

    @property
    def last_result(self):
        """Native result of the last successful energy/gradient evaluation."""
        return self._last_result

    @property
    def last_basis(self):
        """Orbital basis belonging to ``last_result``, or None before success."""
        return self._last_basis

    def _check_geometry(self, molecule):
        composition = (tuple(a.Z for a in molecule.atoms), molecule.charge, molecule.multiplicity)
        if composition != self._composition:
            raise ValueError("Geometry source atom order, elements, charge and multiplicity must stay fixed")

    def basis_at(self, molecule):
        """Owned orbital basis with the original content at a new geometry."""
        self._check_geometry(molecule)
        return _move_basis(molecule, self._orbital, self._orbital_name)

    def scf_keywords(self, molecule):
        """Independent engine/J/K configuration for a final same-source SCF."""
        self._check_geometry(molecule)
        auxiliary = (None if self._auxiliary is None else
                     _move_basis(molecule, self._auxiliary, self._auxiliary_name))
        context = core._manta._MolecularHFEngine()
        context.budget, context.ri = self._engine.budget, self._j.metric
        return dict(integral_engine=replace(self._engine, budget=context.budget),
                    coulomb=replace(self._j, auxiliary_basis=auxiliary, metric=context.ri), exchange=self._k)

    def __call__(self, molecule):
        from contextlib import nullcontext
        import vibeqc as vq
        from .ecp_metadata import ecp_centres_follow_atoms
        from .gradient_options import gradient_options_from_scf

        basis = self.basis_at(molecule)
        driver = vq.run_rhf if self._method == "rhf" else vq.run_uhf
        gradient = vq.compute_gradient if self._method == "rhf" else vq.compute_gradient_uhf
        follow = (ecp_centres_follow_atoms(self._options, self._reference, molecule)
                  if self._ecp else nullcontext())
        with follow:
            result = driver(molecule, basis, self._options, dft_plus_u=self._dft_plus_u,
                            **self.scf_keywords(molecule))
            if not result.converged:
                raise RuntimeError("Selected HF geometry source did not converge; no gradient is admitted")
            if self._ecp:
                # The derivative options carry the ECP route the SCF solved;
                # the gradient entry dispatches MANTA results to the selected
                # gradient with the ECP provider's term.
                values = gradient(molecule, basis, result, gradient_options_from_scf(self._options),
                                  dft_plus_u=self._dft_plus_u)
            else:
                values = gradient(molecule, basis, result, dft_plus_u=self._dft_plus_u)
        values = np.asarray(values, dtype=float).ravel()
        energy = float(result.energy)
        if self._dispersion_params is not None:
            from .dispersion import compute_dispersion
            dispersion = compute_dispersion(molecule, self._dispersion_params, with_gradient=True)
            energy += float(dispersion.energy)
            values = values+np.asarray(dispersion.gradient, dtype=float).ravel()
        self._last_result, self._last_basis = result, basis
        return energy, values
