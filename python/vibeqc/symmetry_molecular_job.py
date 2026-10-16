"""Declarative molecular symmetry requests, never serialized audit seals.

An executable job reconstructs and qualifies a fresh producer. A process-local
prepared source can be used by run_job, but its audit/cache is not portable.
"""
from dataclasses import asdict

import numpy as np

from .symmetry_molecular_scf import (MolecularSymmetryOptions,
    MolecularRISymmetryOptions, PreparedMolecularSymmetry)
from .symmetry_shared import Budget


def configuration(value):
    if type(value) is PreparedMolecularSymmetry:
        return value.options
    if type(value) not in (MolecularSymmetryOptions, MolecularRISymmetryOptions):
        raise TypeError('symmetry_reduction requires MolecularSymmetryOptions, MolecularRISymmetryOptions or PreparedMolecularSymmetry')
    return value


def source_request(value):
    """JSON request for fresh qualification; contains no source authority."""
    config = configuration(value)
    common = dict(rotations=config.rotations.tolist(), translations=config.translations.tolist(),
        source_tolerance=config.source_tolerance, budget=asdict(config.budget), executor=config.executor,
        expected_builds=config.expected_builds)
    if type(config) is MolecularRISymmetryOptions:
        return dict(common, source_kind='abelian-ri', auxiliary_basis=config.auxiliary_basis,
            factor_export=config.factor_export,
            characters_real=config.characters.real.tolist(), characters_imag=config.characters.imag.tolist(),
            **{k: getattr(config, k) for k in
               ('absolute_cutoff', 'relative_cutoff', 'negative_relative_tolerance', 'mode_block')})
    return dict(common, maximum_omitted_norm=config.maximum_omitted_norm, screening=config.screening)


def hydrate_source_request(value):
    """Strict fixed schema with resource admission before numeric allocation."""
    from ._qvf_options import _convert, _fail
    path = 'options.symmetry_reduction'
    if type(value) is not dict:
        _fail(path, 'expected a JSON object')
    is_ri = value.get('source_kind') == 'abelian-ri'
    required = {'rotations', 'translations', 'source_tolerance', 'budget'}
    optional = {'executor', 'expected_builds'}
    if is_ri:
        required |= {'source_kind', 'auxiliary_basis', 'characters_real', 'characters_imag'}
        optional |= {'absolute_cutoff', 'relative_cutoff', 'negative_relative_tolerance', 'mode_block', 'factor_export'}
    else:
        required |= {'maximum_omitted_norm'}
        optional |= {'screening'}
    if value.keys() - required - optional:
        _fail(path, 'unknown fields: ' + ', '.join(sorted(value.keys() - required - optional)))
    if required - value.keys():
        _fail(path, 'missing fields: ' + ', '.join(sorted(required - value.keys())))
    budget_record = value['budget']
    if type(budget_record) is not dict or budget_record.keys() != {'maximum_bytes', 'maximum_work'}:
        _fail(path+'.budget', 'requires maximum_bytes and maximum_work only')
    try:
        budget = Budget(**{k: _convert(v, int, path+'.budget.'+k) for k, v in budget_record.items()})
        rotations, translations = value['rotations'], value['translations']
        if type(rotations) is not list or type(translations) is not list:
            _fail(path, 'operations must be JSON arrays')
        ng = len(rotations)
        if not ng or len(translations) != ng:
            _fail(path, 'operation counts must be equal and positive')
        # Include JSON-to-array staging, constructor snapshots and validation.
        budget.admit(4096+512*ng+(128*ng*ng if is_ri else 0), 128*ng*(ng if is_ri else 1))
        for rotation, translation in zip(rotations, translations):
            if (type(rotation) is not list or len(rotation) != 3
                    or any(type(row) is not list or len(row) != 3 for row in rotation)
                    or type(translation) is not list or len(translation) != 3):
                _fail(path, 'expected rotations (ng,3,3) and translations (ng,3)')
        r = _convert(rotations, list[list[list[float]]], path+'.rotations')
        t = _convert(translations, list[list[float]], path+'.translations')
        kwargs = {k: _convert(value[k], annotation, path+'.'+k)
                  for k, annotation in dict(source_tolerance=float, maximum_omitted_norm=float,
                      screening=str, executor=str, expected_builds=int | None).items() if k in value}
        if is_ri:
            panels = []
            for name in ('characters_real', 'characters_imag'):
                panel = value[name]
                if type(panel) is not list or len(panel) != ng or any(type(row) is not list or len(row) != ng for row in panel):
                    _fail(path+'.'+name, 'expected a complete (ng,ng) character table')
                panels.append(np.asarray(_convert(panel, list[list[float]], path+'.'+name)))
            kwargs.update({k: _convert(value[k], annotation, path+'.'+k)
                for k, annotation in dict(auxiliary_basis=str, absolute_cutoff=float, relative_cutoff=float,
                    negative_relative_tolerance=float, mode_block=int, factor_export=str).items() if k in value})
            return MolecularRISymmetryOptions(np.asarray(r, dtype=np.float64),
                np.asarray(t, dtype=np.float64), np.ascontiguousarray(panels[0]+1j*panels[1]),
                budget=budget, **kwargs)
        return MolecularSymmetryOptions(np.asarray(r, dtype=np.float64),
            np.asarray(t, dtype=np.float64), budget=budget, **kwargs)
    except (TypeError, ValueError, OverflowError) as exc:
        _fail(path, str(exc))
