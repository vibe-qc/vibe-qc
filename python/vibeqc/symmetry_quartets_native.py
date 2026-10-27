"""Opt-in compiled transport/contraction over admitted molecular tile plans.

The integral producer and complete source audit are unchanged. The native
accumulator is algebra, not an independent source or symmetry certificate.
"""
from . import _vibeqc_core as core
from .symmetry_shared import Budget, _array, _integer
from .symmetry_quartets import QuartetJKResult, ShellQuartetPlan
from .symmetry_quartets_streaming import StreamingShellQuartetPlan


def _source_requirements(plan, budget, reserved_bytes, reserved_work):
    """Optional native producer bridge; never a generic protocol shortcut."""
    from .symmetry_quartets import _MantaSource
    if (not isinstance(plan, StreamingShellQuartetPlan) or type(plan.source) is not _MantaSource
            or getattr(core, '_SymmetryQuartetSourceAccumulator', None) is None
            or getattr(core._SymmetryQuartetAccumulator, 'accumulate_orbit', None) is None):
        return None
    # Count only: no workspace or source payload is allocated by the planner.
    # An optional second workspace must not reject a previously valid native
    # request. The checked Python reader uses the audit session when it will
    # not fit alongside the admitted plan, targets and screening metadata.
    source = core._plan_symmetry_quartet_source(plan.source._provider,
        Budget(2**63-1, 2**63-1).native())
    if (reserved_bytes+source.bytes > budget.maximum_bytes
            or reserved_work+source.work > budget.maximum_work):
        return None
    return source


def _requirements(plan, budget, density_count=1, targets=None, overhead=(0, 0)):
    if not isinstance(plan, (ShellQuartetPlan, StreamingShellQuartetPlan)):
        raise TypeError('an admitted molecular quartet plan is required')
    if not getattr(plan, 'equivariance_qualified', True):
        raise ValueError('ordered-source qualification does not admit reduced execution')
    density_count = _integer(density_count, 'density count', positive=True)
    from .symmetry_quartets_batch import _target_counts
    nj, nk = _target_counts(density_count, targets)
    budget.admit(plan.required_bytes, plan.required_work)
    n, ns, ng = int(plan.offsets[-1]), len(plan.offsets)-1, plan.group.order
    maximum = max(m.shape[0] for m in plan.inverse_blocks[0])
    elements = sum(m.size for m in plan.inverse_blocks[0])
    planner = getattr(core, '_plan_symmetry_quartet_accumulator', None)
    if planner is None:
        raise RuntimeError('native quartet executor is unavailable; use executor="numpy"')
    native = planner(n, ns, ng, elements, maximum, budget.native(), density_count, nj, nk)
    size, work = plan.required_bytes+native.bytes+overhead[0], plan.required_work+native.work+overhead[1]
    budget.admit(size, work)
    source = _source_requirements(plan, budget, size, work)
    total = Budget(size+(0 if source is None else source.bytes), work+(0 if source is None else source.work))
    budget.admit(total.maximum_bytes, total.maximum_work)
    return total, native, maximum


def native_quartet_requirements(plan, *, budget: Budget, density_count=1, targets=None):
    """Count-only combined live plan/native reservation before packing payload.

    Returned exact limits admit one native build. Plan construction/admission
    remains a separate mandatory step. This model is not an RSS guarantee.
    """
    return _requirements(plan, budget, density_count, targets)[0]


def native_quartet_jk(plan, density, *, budget: Budget):
    _array(density, 'complex128', 2)
    from .symmetry_quartets_batch import quartet_jk_batch
    result = quartet_jk_batch(plan, density[None], budget=budget, executor='native')
    return QuartetJKResult(result.j[0], result.k[0], result.work, result.scalar_products,
        result.block_contractions, result.transport_scalar_products, float(result.error_bounds[0]),
        executor=result.executor)
