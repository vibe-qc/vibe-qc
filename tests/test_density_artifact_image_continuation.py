"""Artifact image refinement preserves charge and convergence refusals."""
import ast
from pathlib import Path

import numpy as np
import pytest


def certify():
    path=Path(__file__).parents[1]/'python/vibeqc/periodic_runner.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_converged_periodic_density_grid_for_artifact')
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ns={'np':np};exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
    return ns[node.name]


def run(values,**kwargs):
    radii=[]
    def evaluate(radius):
        radii.append(radius)
        return np.full((2,2,2),values[radius-1]/10),{'radius':radius}
    result=certify()(evaluate,lattice_bohr=np.diag([10.,1.,1.]),grid_shape=(2,2,2),analytic_electrons=8.,label='diffuse density',**kwargs)
    return result,radii


def test_diffuse_density_continues_without_relaxing_original_criteria():
    values=[46365.,1085.,9.9814,8.00574,8.00000516,8.0000000005]
    (rho,meta),radii=run(values)
    assert radii==list(range(1,7)) and meta['radius']==6
    assert abs(rho.sum()*10/8-8)<1e-8
    with pytest.raises(ValueError,match='did not converge.*radius 3'):
        run(values,max_image_radius=3)


def test_compact_density_stops_at_original_radius_three():
    (_,meta),radii=run([8.,8.,8.])
    assert radii==[1,2,3] and meta['radius']==3


def test_early_equal_pair_cannot_hide_later_images():
    (_,meta),radii=run([8.,8.,9.,8.,8.])
    assert radii==[1,2,3,4,5] and meta['radius']==5


def test_no_convergence_at_bound_refuses():
    with pytest.raises(ValueError,match='did not converge.*radius 8'):
        run([8.+1/r for r in range(1,9)])


def test_converged_images_do_not_normalize_bad_charge():
    with pytest.raises(ValueError,match='decrease density_spacing_bohr'):
        run([9.,9.,9.])


@pytest.mark.parametrize('kwargs',[{'image_l1_tolerance':0},{'image_max_tolerance':float('nan')},{'charge_tolerance':-1},{'max_image_radius':1}])
def test_invalid_certification_controls_refuse(kwargs):
    with pytest.raises(ValueError):run([8.]*8,**kwargs)
