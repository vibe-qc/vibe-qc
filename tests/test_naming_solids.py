"""Solid names depend on composition and explicit periodic metadata, not cell size."""

import importlib
import json
import sys
import zipfile

import numpy as np
import pytest

from vibeqc_naming import Confidence, name_solid_from_atoms
from vibeqc_naming._solid_elements import ELEMENTS


CELL = np.diag([8.0, 8.0, 8.0])
NACL = [(11, 0, 0, 0), (17, 4, 4, 4)]


@pytest.fixture(params=['vibeqc_naming.iupac_name', 'vibeqc.naming.iupac_name'])
def api(request):
    return importlib.import_module(request.param)


def test_salt_name_is_invariant_under_supercells_rotation_and_atom_order(api):
    unit = api.name_solid_from_atoms(NACL, CELL)
    supercell = [
        (z, x + 8 * image, y, cz)
        for image in range(3) for z, x, y, cz in NACL
    ]
    cell = CELL.copy()
    cell[0] *= 3
    angle = .43
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transformed = [(z, *(np.array([x, y, cz]) @ rotation)) for z, x, y, cz in reversed(supercell)]
    result = api.name_solid_from_atoms(transformed, cell @ rotation)
    assert result.name == unit.name == 'sodium chloride'
    assert result.reduced_formula == unit.reduced_formula == 'ClNa'
    assert result.formula == 'Cl3Na3'
    assert result.dimensionality == 3
    assert isinstance(result, api.NamedResult)


@pytest.mark.parametrize('z', range(1, 119))
def test_all_elements_have_full_names(z):
    symbol, name = ELEMENTS[z]
    result = name_solid_from_atoms([(z, 0, 0, 0)], CELL)
    assert result.name == name
    assert result.reduced_formula == result.formula == symbol
    assert result.confidence == Confidence.MEDIUM


@pytest.mark.parametrize('atoms, expected', [
    ([(29, 0, 0, 0), (30, 2, 0, 0)], 'copper zincide'),
    ([(56, 0, 0, 0), (22, 2, 0, 0), (8, 0, 2, 0), (8, 0, 0, 2), (8, 2, 2, 2)], 'barium titanate'),
    ([(3, 0, 0, 0), (26, 2, 0, 0), (15, 0, 2, 0), (8, 0, 0, 2)], 'FeLiOP solid'),
    ([(92, 0, 0, 0), (8, 2, 0, 0), (8, 0, 2, 0)], 'uranium dioxide'),
])
def test_binary_multinary_and_mixed_metal_compositions(atoms, expected):
    assert name_solid_from_atoms(atoms, CELL).name == expected


@pytest.mark.parametrize('pbc, suffix, dimensions', [
    ((True, True, True), '', 3),
    ((False, True, True), ' slab', 2),
    ((False, True, False), ' chain', 1),
])
def test_periodicity_is_explicit_not_inferred_from_vacuum(api, pbc, suffix, dimensions):
    # An elongated cell is still 3D if all boundary conditions are periodic.
    result = api.name_from_atoms_with_lattice([(6, 0, 0, 0)], np.diag([40, 8, 8]), pbc=pbc)
    assert result.name == 'carbon' + suffix
    assert result.pbc == pbc
    assert result.dimensionality == dimensions


def test_bulk_metal_with_many_bonds_is_not_called_a_slab():
    atoms = [(29, 2*x, 2*y, 2*z) for x in range(2) for y in range(2) for z in range(2)]
    assert name_solid_from_atoms(atoms, CELL).name == 'copper'


def test_no_polymorph_or_miller_plane_is_guessed():
    result = name_solid_from_atoms([(6, 0, 0, 0)], CELL)
    assert result.name == 'carbon'
    assert result.phase is result.miller_indices is result.space_group is None


def test_explicit_phase_and_surface_metadata_have_provenance(api):
    result = api.name_solid_from_atoms(
        [(6, 0, 0, 0)], CELL, pbc=(True, True, False),
        phase='graphite', miller_indices=(0, 0, 0, 1),
    )
    assert result.name == 'carbon (graphite) (0 0 0 1) slab'
    assert result.phase == 'graphite'
    assert result.miller_indices == (0, 0, 0, 1)
    assert any('not inferred or verified' in note for note in result.notes)


def test_partial_occupancy_preserves_average_composition_and_scales():
    atoms = [(26, 0, 0, 0), (8, 2, 0, 0)]
    result = name_solid_from_atoms(atoms, CELL, occupancies=[.9, 1], analyze_symmetry=True)
    assert result.name == result.formula + ' solid' == 'Fe0.9O solid'
    assert result.reduced_formula == 'Fe9O10'
    assert dict(result.composition) == {'Fe': .9, 'O': 1}
    assert result.space_group is None
    assert any('skipped' in note for note in result.notes)
    supercell = atoms + [(z, x+8, y, cz) for z, x, y, cz in atoms]
    assert name_solid_from_atoms(supercell, np.diag([16, 8, 8]), occupancies=[.9, 1, .9, 1]).name == result.name


def test_shared_site_disorder_and_vacancies():
    result = name_solid_from_atoms(
        [(29, 0, 0, 0), (30, 0, 0, 0), (8, 2, 0, 0)],
        CELL, occupancies=[.5, .5, 0],
    )
    assert result.name == 'Cu0.5Zn0.5 solid'
    assert result.formula == 'Cu0.5Zn0.5'
    assert result.reduced_formula == 'CuZn'


def test_spglib_analysis_is_optional_and_uses_full_structure(api):
    pytest.importorskip('spglib')
    result = api.name_solid_from_atoms([(14, 0, 0, 0)], CELL, analyze_symmetry=True)
    assert result.space_group_number == 221
    assert result.space_group == 'Pm-3m'
    assert result.crystal_system == 'cubic'
    assert result.name == 'silicon'
    assert result.phase is None
    assert result._replace(name='label').space_group_number == 221


def test_missing_spglib_keeps_composition_name(monkeypatch):
    monkeypatch.setitem(sys.modules, 'spglib', None)
    result = name_solid_from_atoms(NACL, CELL, analyze_symmetry=True)
    assert result.name == 'sodium chloride'
    assert result.space_group is None
    assert any('unavailable' in note for note in result.notes)


def test_slab_does_not_use_a_three_dimensional_space_group():
    result = name_solid_from_atoms(NACL, CELL, pbc=(True, False, True), analyze_symmetry=True)
    assert result.space_group is None
    assert any('skipped' in note for note in result.notes)


def test_explicit_adsorbate_partition(api):
    result = api.name_solid_from_atoms(
        [(29, 0, 0, 0), (6, 0, 0, 4), (8, 0, 0, 5.13)],
        CELL, pbc=(True, True, False), adsorbate_groups=[(1, 2)], prefer_trivial=True,
    )
    assert result.name == 'carbon monoxide on copper'
    assert result.formula == 'CCuO'
    assert any('partition supplied' in note for note in result.notes)


@pytest.mark.parametrize('options', [
    {'pbc': (False, False, False)}, {'pbc': (True, True)},
    {'occupancies': [1]}, {'occupancies': [-.1, 1]}, {'occupancies': [0, 0]},
    {'occupancies': [float('nan'), 1]}, {'symprec': 0}, {'phase': ''},
    {'miller_indices': (0, 0, 1)}, {'adsorbate_groups': [(0,)]},
])
def test_invalid_metadata_is_explicit(options):
    with pytest.raises(ValueError):
        name_solid_from_atoms(NACL, CELL, **options)


@pytest.mark.parametrize('atoms, cell', [
    ([], CELL), ([(119, 0, 0, 0)], CELL), ([(1.5, 0, 0, 0)], CELL),
    ([(6, float('nan'), 0, 0)], CELL), (NACL, np.zeros((3, 3))),
    ([(6, 0, 0, 0), (6, 8, 0, 0)], CELL),
])
def test_invalid_geometry_is_explicit(atoms, cell):
    with pytest.raises(ValueError):
        name_solid_from_atoms(atoms, cell)


def test_qvf_preserves_pbc_occupancy_and_solid_metadata(api, tmp_path):
    path = tmp_path / 'disordered.qvf'
    structure = {'pbc': [True, False, True], 'lattice_vectors': CELL.tolist(), 'atoms': [
        {'atomic_number': 26, 'symbol': 'Fe', 'position': [0, 0, 0], 'occupancy': .9},
        {'atomic_number': 8, 'symbol': 'O', 'position': [2, 0, 0]},
    ]}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'sections': [{'id': 'structure',
            'members': {'structure': {'path': 'structure.json'}}}]}))
        archive.writestr('structure.json', json.dumps(structure))
    result = api.name_from_qvf(path)
    assert result.name == 'Fe0.9O slab'
    assert result.formula == 'Fe0.9O'
    assert result.reduced_formula == 'Fe9O10'
    assert result.pbc == (True, False, True)


def test_formula_names_and_benchmark_helpers_share_the_solid_policy():
    from vibeqc_naming import name_solid_from_formula
    from vibeqc_naming.name_systems import _name_periodic as standalone
    from vibeqc.naming.name_systems import _name_periodic as shim

    for formula, name in [('C8', 'carbon'), ('Na4Cl4', 'sodium chloride'),
                          ('SiO2', 'silicon dioxide'), ('Fe0.9O', 'Fe0.9O solid')]:
        assert name_solid_from_formula(formula).name == name
        assert standalone(formula).name == name
        assert shim('ignored-system-label', formula).name == name


@pytest.mark.parametrize('formula', ['', 'Xx2', 'H0', 'Fe-2', 'Ca(OH)2', 'Fe1-xO'])
def test_formula_naming_rejects_unresolved_or_invalid_compositions(formula):
    from vibeqc_naming import name_solid_from_formula

    with pytest.raises(ValueError):
        name_solid_from_formula(formula)


def test_real_rocksalt_symmetry_is_not_inferred_from_just_the_formula():
    pytest.importorskip('spglib')
    fcc = np.array([(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)])
    atoms = [(11, *(r*5.64)) for r in fcc]
    atoms += [(17, *((r + [.5, 0, 0]) % 1 * 5.64)) for r in fcc]
    rock = name_solid_from_atoms(atoms, np.eye(3)*5.64, analyze_symmetry=True)
    cubic = name_solid_from_atoms(NACL, CELL, analyze_symmetry=True)
    assert rock.name == cubic.name == 'sodium chloride'
    assert rock.space_group_number == 225
    assert cubic.space_group_number == 221
    assert rock.phase_descriptor == 'NaCl(cF8)'
    assert cubic.phase_descriptor == 'NaCl(cP2)'
    assert rock.phase is cubic.phase is None


def test_default_naming_avoids_molecular_graph_construction(monkeypatch):
    import vibeqc_naming.iupac_name as module

    def unexpected(*args, **kwargs):
        raise AssertionError('periodic composition naming must not build a molecular graph')

    monkeypatch.setattr(module, 'from_atoms_list', unexpected)
    assert module.name_from_atoms_with_lattice(NACL, CELL).name == 'sodium chloride'


def test_no_native_core_or_spglib_is_needed_for_composition():
    import subprocess

    code = '''
import sys
from vibeqc_naming import name_solid_from_atoms
r = name_solid_from_atoms([(92, 0, 0, 0)], [(5,0,0),(0,5,0),(0,0,5)])
assert r.name == 'uranium'
assert 'vibeqc' not in sys.modules
assert 'spglib' not in sys.modules
assert not any('_vibeqc_core' in name for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('pbc', [(False, False), (0, 0, 0), (False, False, False, False)])
def test_atoms_entry_point_validates_pbc_even_on_molecular_path(api, pbc):
    with pytest.raises(ValueError, match='three booleans'):
        api.name_from_atoms_with_lattice(NACL, CELL, pbc=pbc)


@pytest.mark.parametrize('groups', [[(1, 1)], [(1,), (1,)], [()], [(0, 1, 2)], [(3,)]])
def test_invalid_adsorbate_partitions(groups):
    with pytest.raises(ValueError):
        name_solid_from_atoms([(29, 0, 0, 0), (6, 0, 0, 3), (8, 0, 0, 4.13)],
                              CELL, pbc=(True, True, False), adsorbate_groups=groups)


@pytest.mark.parametrize('indices', [(0, 0, 0), (1, 0, 0, 1), (0, 0, 1.5)])
def test_invalid_miller_indices(indices):
    with pytest.raises(ValueError):
        name_solid_from_atoms(NACL, CELL, pbc=(True, True, False), miller_indices=indices)


def test_uncertain_adsorbate_does_not_gain_confidence(monkeypatch):
    from vibeqc_naming import NamedResult, NamingSource
    import vibeqc_naming.iupac_name as module

    monkeypatch.setattr(module, 'name_from_atoms_detailed', lambda *a, **kw: NamedResult(
        name='uncertain fragment', source=NamingSource.ORGANIC_SYSTEMATIC,
        confidence=Confidence.LOW,
    ))
    result = name_solid_from_atoms([(29, 0, 0, 0), (6, 0, 0, 4)], CELL,
                                  pbc=(True, True, False), adsorbate_groups=[(1,)])
    assert result.name == 'uncertain fragment on copper'
    assert result.confidence == Confidence.LOW


def test_public_shim_exports_the_solid_api():
    import vibeqc.naming as shim
    import vibeqc_naming as standalone

    for symbol in ('SolidResult', 'SolidCompositionResult', 'name_solid_from_atoms', 'name_solid_from_formula',
                   'name_from_atoms_with_lattice', 'name_periodic_system'):
        assert symbol in shim.__all__
        assert getattr(shim, symbol) is getattr(standalone, symbol)


def test_parallel_formula_api_preserves_result_types(api):
    result = api.name_solid_from_formula('UO2')
    assert isinstance(result, api.NamedResult)
    assert result.name == 'uranium dioxide'


# Red Book 2005 IR-5.2, pp. 69-70. Names express empirical composition;
# the molecular N2O4 example is therefore tested separately below.
@pytest.mark.parametrize('formula, expected', [
    ('HCl', 'hydrogen chloride'), ('NO', 'nitrogen oxide'),
    ('NO2', 'nitrogen dioxide'), ('OCl2', 'oxygen dichloride'),
    ('O2Cl', 'dioxygen chloride'), ('Fe3O4', 'triiron tetraoxide'),
    ('SiC', 'silicon carbide'), ('SiCl4', 'silicon tetrachloride'),
    ('Ca3P2', 'tricalcium diphosphide'), ('NiSn', 'nickel stannide'),
    ('Cu5Zn8', 'pentacopper octazincide'), ('Cr23C6', 'tricosachromium hexacarbide'),
    ('MgGe', 'magnesium germide'), ('NaAg', 'sodium argentide'),
    ('LiAu', 'lithium auride'), ('LaH3', 'lanthanum trihydride'),
])
def test_red_book_binary_examples_on_both_paths(api, formula, expected):
    result = api.name_solid_from_formula(formula)
    assert result.name == expected
    assert result.name_kind == 'systematic'
    assert isinstance(result, api.SolidCompositionResult)
    assert result.display_formula == formula
    assert result.display_formula_order == 'compositional'


@pytest.mark.parametrize('pbc, suffix', [((True, True, True), ''), ((True, False, True), ' slab')])
def test_lattice_uses_red_book_order_for_bulk_and_slab(api, pbc, suffix):
    result = api.name_from_atoms_with_lattice(
        [(8, 0, 0, 0), (17, 3, 0, 0), (17, 0, 3, 0)], CELL, pbc=pbc,
    )
    assert result.name == 'oxygen dichloride' + suffix
    assert result.display_formula == 'OCl2'
    assert result.formula == 'Cl2O'
    assert result.name_kind == ('descriptive' if suffix else 'systematic')


@pytest.mark.parametrize('formula, display, hill', [
    ('Na4Cl4', 'NaCl', 'ClNa'), ('SiC', 'SiC', 'CSi'),
    ('BaTiO3', 'BaTiO3', 'BaO3Ti'), ('N2O4', 'NO2', 'NO2'),
])
def test_chemical_display_is_separate_from_empirical_hill_formula(api, formula, display, hill):
    result = api.name_solid_from_formula(formula)
    assert result.display_formula == display
    assert result.formula == hill


@pytest.mark.parametrize('formula', ['Fe0.9O', 'Fe1.1O', 'Fe2.2O2', 'Cu0.5Zn0.5'])
def test_decimal_formulas_preserve_the_callers_composition_basis(api, formula):
    result = api.name_solid_from_formula(formula)
    assert result.name == formula + ' solid'
    assert result.display_formula == formula
    assert result.name_kind == 'formula_label'


def test_interstitial_excess_and_explicit_occupancy_normalization(api):
    atoms = [(26, 0, 0, 0), (26, 2, 2, 0), (8, 2, 0, 0)]
    result = api.name_solid_from_atoms(atoms, CELL, occupancies=[1, .1, 1])
    assert result.name == 'Fe1.1O solid'
    assert result.display_formula == 'Fe1.1O'
    twice = atoms + [(z, x+8, y, cz) for z, x, y, cz in atoms]
    repeated = api.name_solid_from_atoms(twice, np.diag([16, 8, 8]), occupancies=[1, .1, 1]*2)
    assert repeated.name == result.name
    assert repeated.formula == 'Fe2.2O2'
    explicit = api.name_solid_from_atoms(twice, np.diag([16, 8, 8]),
                                        occupancies=[1, .1, 1]*2, formula_units=1)
    assert explicit.name == 'Fe2.2O2 solid'
    assert any('caller-supplied formula units' in note for note in explicit.notes)


@pytest.mark.parametrize('units', [0, -1, float('nan'), float('inf'), True])
def test_invalid_formula_units(units):
    with pytest.raises(ValueError, match='formula_units'):
        name_solid_from_atoms(NACL, CELL, occupancies=[.9, 1], formula_units=units)


@pytest.mark.parametrize('formula, kind, order', [
    ('Na2SO4', 'formula_label', 'hill'), ('OgF2', 'formula_label', 'hill'),
    ('Cr10000C', 'formula_label', 'compositional'),
])
def test_unsupported_rules_produce_explicit_labels(api, formula, kind, order):
    result = api.name_solid_from_formula(formula)
    assert result.name
    assert result.name_kind == kind
    assert result.display_formula_order == order


def test_common_names_are_distinguished(api):
    result = api.name_solid_from_formula('Al2O3', prefer_trivial=True)
    assert result.name == 'alumina'
    assert result.name_kind == 'common'


@pytest.mark.parametrize('count, prefix', [
    (11, 'undeca'), (13, 'trideca'), (20, 'icosa'), (21, 'henicosa'),
    (22, 'docosa'), (23, 'tricosa'), (31, 'hentriaconta'), (35, 'pentatriaconta'),
    (48, 'octatetraconta'), (52, 'dopentaconta'), (100, 'hecta'), (200, 'dicta'),
    (231, 'hentriacontadicta'), (500, 'pentacta'), (1000, 'kilia'),
    (1002, 'dokilia'), (2000, 'dilia'), (9999, 'nonanonacontanonactanonalia'),
])
def test_published_multipliers_in_composition_names(count, prefix):
    from vibeqc_naming import name_solid_from_formula

    assert name_solid_from_formula(f'Cr{count}C').name == f'{prefix}chromium carbide'


def test_table_vi_and_ix_cover_the_published_element_set():
    from vibeqc_naming._solid_rules import ELEMENT_SEQUENCE, IDE_NAMES

    published = {ELEMENTS[z][0] for z in range(1, 112)}
    assert len(ELEMENT_SEQUENCE) == len(published)
    assert set(ELEMENT_SEQUENCE) == set(IDE_NAMES) == published


def test_pearson_uses_conventional_cell_for_primitive_and_repeated_inputs(api):
    pytest.importorskip('spglib')
    # One atom in a primitive fcc cell represents four conventional-cell atoms.
    cell = np.array([(0, 2, 2), (2, 0, 2), (2, 2, 0)], dtype=float)
    primitive = api.name_solid_from_atoms([(29, 0, 0, 0)], cell, analyze_symmetry=True)
    supercell = cell.copy()
    supercell[0] *= 2
    repeated = api.name_solid_from_atoms([(29, 0, 0, 0), (29, *cell[0])],
                                         supercell, analyze_symmetry=True)
    assert primitive.pearson_symbol == repeated.pearson_symbol == 'cF4'
    assert primitive.phase_descriptor == repeated.phase_descriptor == 'Cu(cF4)'
    assert primitive.name == repeated.name == 'copper'
    assert repeated._replace(name='label').pearson_symbol == 'cF4'


def test_rhombohedral_pearson_counts_the_hexagonal_conventional_cell(api):
    pytest.importorskip('spglib')
    # Generic primitive rhombohedron, away from cubic special metrics.
    lattice = np.linalg.cholesky(np.full((3, 3), 3.0) + np.eye(3)*6)
    result = api.name_solid_from_atoms([(29, 0, 0, 0)], lattice, analyze_symmetry=True)
    assert result.space_group_number == 166
    assert result.pearson_symbol == 'hR3'


def test_legacy_spglib_dataset_centering_notation(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, 'spglib', SimpleNamespace(
        get_symmetry_dataset=lambda *a, **kw: {'number': 12, 'international': 'C2/m',
                                              'std_types': [29, 29]},
    ))
    result = name_solid_from_atoms([(29, 0, 0, 0)], CELL, analyze_symmetry=True)
    assert result.pearson_symbol == 'mS2'


def test_adsorbate_substrate_keeps_explicit_normalization(api):
    result = api.name_solid_from_atoms(
        [(26, 0, 0, 0), (8, 2, 0, 0), (6, 0, 0, 4), (8, 0, 0, 5.13)],
        CELL, pbc=(True, True, False), occupancies=[.9, 1, 1, 1],
        adsorbate_groups=[(2, 3)], prefer_trivial=True, formula_units=2,
    )
    assert result.name == 'carbon monoxide on Fe0.45O0.5'
    assert result.name_kind == 'descriptive'
