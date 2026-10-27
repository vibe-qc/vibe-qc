"""Scalar ECP Hamiltonians on the physical BIPOLE pair domain.

Hay and Wadt, J. Chem. Phys. 82, 270 (1985), doi:10.1063/1.448799, Sec. II, Eqs. (5), (10)-(11):
the explicit valence Hamiltonian uses effective ionic Coulomb charges plus
local and angular-projector ECP terms. Bare species still define the basis
and XC partition. Projector support uses the shared midpoint sphere (#652).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BipoleECPRequest:
    context: tuple
    contract: dict
    guess_context: object

    @property
    def ionic_system(self):
        return self.context[4]

    @property
    def ncore(self):
        return self.context[3]


def prepare_bipole_ecp(options, system, basis, *, driver,
                       historical_domain=False, exchange_split=None,
                       symmetry=False, experimental_source=False, derivatives=False):
    """Validate a complete scalar request before allocation, or retain AE."""
    from .pbc_bipole_common import reject_bipole_ecp_options
    if not getattr(options, "ecp_primitive_blocks", None):
        reject_bipole_ecp_options(options, driver=driver, basis=basis, system=system)
        return None

    from .periodic_k_gdf import _periodic_ecp_context
    from .guess import guess_ecp_context
    from .periodic.chi.ecp import ecp_contract

    context = _periodic_ecp_context(options, system, driver)
    if context is None:
        reject_bipole_ecp_options(options, driver=driver, basis=basis, system=system)
        return None
    # The inert shared scalar metadata validator preserves zero-coefficient
    # local-channel placeholders and binds every removed core to an atom.
    contract = ecp_contract(system, options)
    if (historical_domain or exchange_split is False or symmetry
            or getattr(getattr(system, "symmetry", None), "operations", None)
            or experimental_source or derivatives):
        raise NotImplementedError(
            f"{driver}: BIPOLE ECP requires neutral 3D corrected physical pair "
            "support without symmetry, experimental source adapters or derivatives"
        )
    return BipoleECPRequest(context, contract,
        guess_ecp_context(options, molecule=system.unit_cell_molecule()))


def add_bipole_ecp(request, basis, system, kmesh, lattice_options, ionic_potential):
    """Add the unmodified native ECP blocks by cell key and retain their source."""
    from .periodic_k_gdf import _ecp_lattice_blocks, _ecp_projector_options
    from ._vibeqc_core import make_lattice_matrix_set
    from .periodic_ecp_source import _frame, _array_identity
    from .bipole_domain import LATTICE_FIELDS, OptionSnapshot
    from .fock_state import SourceProvenance

    ecp = _ecp_lattice_blocks(basis, system, lattice_options, request.context)
    blocks = {tuple(map(int, c.index)): np.asarray(b, dtype=float)
              for c, b in zip(ecp.cells, ecp.blocks)}
    residual = 0.0
    for key, block in blocks.items():
        reverse = blocks.get(tuple(-i for i in key))
        if reverse is None or not np.isfinite(block).all():
            raise ValueError("BIPOLE ECP lacks finite reverse-translation blocks")
        error = float(np.max(abs(block-reverse.T)))
        residual = max(residual, error)
        if error > 1e-10 * max(1., float(np.max(abs(block)))):
            raise ValueError("BIPOLE ECP raw operator violates translation Hermiticity")
    cells = list(ionic_potential.cells)
    keys = [tuple(map(int, c.index)) for c in cells]
    if set(keys) != set(blocks):
        raise ValueError("BIPOLE ionic and ECP operators have different physical pair domains")
    potential = make_lattice_matrix_set(basis.nbasis, cells,
        [np.asarray(v)+blocks[k] for k, v in zip(keys, ionic_potential.blocks)])
    actual_options, physical = _ecp_projector_options(system, lattice_options)
    source = SourceProvenance.capture(
        "bipole-scalar-ecp-v1", frame=_frame(system, basis, kmesh),
        scalar_operator=request.contract,
        projector_options=OptionSnapshot.capture(actual_options, LATTICE_FIELDS).values,
        physical_pair_support=physical, projector_center="AO-pair midpoint",
        raw_hermiticity_residual=residual,
        cells=[list(key) for key in keys],
        operator_blocks=[_array_identity(blocks[key]) for key in keys],
    )
    return potential, source


def reject_bipole_ecp_derivative(result):
    """ECP source retention does not admit an unimplemented force consumer."""
    if (getattr(result, "ecp_operator", False)
            or getattr(result, "ecp_total_ncore", 0)
            or getattr(result, "ecp_source", None) is not None):
        raise NotImplementedError("BIPOLE ECP gradients/stress are not implemented")
