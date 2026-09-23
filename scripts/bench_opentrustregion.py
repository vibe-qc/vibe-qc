#!/usr/bin/env python3
"""Compare molecular orbital optimizers by work counts and physical state.

Run with the enabled native build. JSON goes to stdout; timings are descriptive,
never CI assertions. Every JK call (including response) is counted, so an SCF
iteration is not incorrectly treated as one equally expensive unit of work.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from scipy.optimize import minimize_scalar


class CountingJK(core.JKBuilder):
    def __init__(self, delegate):
        super().__init__()
        self.delegate = delegate
        self.j_calls = 0
        self.k_calls = 0

    def build_J(self, density):
        self.j_calls += 1
        return self.delegate.build_J(density)

    def build_K(self, density):
        self.k_calls += 1
        return self.delegate.build_K(density)


def axial_density_alignment(basis, densities, reference):
    """Diagnose linear-molecule degeneracy by a single spatial rotation.

    Fit the AO representation of a rigid rotation about the molecular z axis
    on an overdetermined point set, then compare both spin densities under
    that same rotation. This is a benchmark diagnostic, not SCF state repair.
    """
    points = np.random.default_rng(83).normal(size=(max(200, 5*basis.nbasis), 3))*2
    ao = np.asarray(vq.evaluate_ao(basis, points))
    inverse = np.linalg.pinv(ao)

    def difference(theta):
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        transform = inverse @ np.asarray(vq.evaluate_ao(basis, points @ rot))
        return max(np.linalg.norm(transform @ d @ transform.T - r)
                   for d, r in zip(densities, reference))

    angles = np.linspace(0, 2*np.pi, 73)[:-1]
    best = angles[np.argmin([difference(t) for t in angles])]
    fit = minimize_scalar(difference, bounds=(best-np.pi/36, best+np.pi/36),
                          method="bounded", options={"xatol": 1e-12})
    return float(fit.x), float(fit.fun)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--functional", default="PBE")
    parser.add_argument("--case", choices=("water", "oh", "o2-quintet"), default="water")
    args = parser.parse_args()
    if not vq.has_opentrustregion():
        parser.error("rebuild with VIBEQC_ENABLE_OPENTRUSTREGION=ON")
    if args.case == "water":
        mol = vq.Molecule([vq.Atom(8,[0,0,0]),vq.Atom(1,[0,1.43,1.1]),vq.Atom(1,[0,-1.43,1.1])])
        methods = ("rhf", "rks")
    elif args.case == "oh":
        mol = vq.Molecule([vq.Atom(8,[0,0,0]),vq.Atom(1,[0,0,1.8])],multiplicity=2)
        methods = ("uhf", "uks")
    else:
        mol = vq.Molecule([vq.Atom(8,[0,0,0]),vq.Atom(8,[0,0,2.28])],multiplicity=5)
        methods = ("uhf", "uks")
    basis = vq.BasisSet(mol,args.basis)
    s = vq.compute_overlap(basis)
    h = vq.compute_kinetic(basis) + vq.compute_nuclear(basis,mol)
    nuclear = mol.nuclear_repulsion()
    n = mol.n_electrons()
    na, nb = (n+mol.multiplicity-1)//2, (n-mol.multiplicity+1)//2
    grid_options = vq.GridOptions()
    grid_options.n_radial=30;grid_options.n_theta=14;grid_options.n_phi=28
    grid = vq.build_grid(mol,grid_options)
    records=[]
    for method in methods:
        references=[]
        for backend in ("native","opentrustregion"):
            jk = CountingJK(core.make_four_index_jk_builder(basis))
            opts=getattr(vq,method.upper()+"Options")()
            opts.initial_guess=vq.InitialGuess.HCORE
            opts.max_iter=200;opts.conv_tol_energy=1e-9;opts.conv_tol_grad=1e-7
            opts.orbital_optimizer=backend
            if method.endswith("ks"):opts.functional=args.functional
            opts.stability_check=True
            start=time.perf_counter()
            positional=[basis,n,s,h,nuclear,jk] if method in ("rhf","rks") else [basis,na,nb,s,h,nuclear,jk]
            if method.endswith("ks"):positional.append(grid)
            result=getattr(core,"run_"+method+"_scf_with_jk")(*positional,options=opts,molecule=mol)
            elapsed=time.perf_counter()-start
            densities=[np.asarray(result.density)] if method in ("rhf","rks") else [np.asarray(result.density_alpha),np.asarray(result.density_beta)]
            record=dict(case=args.case,method=method,basis=args.basis,backend=backend,
                        converged=result.converged,energy=result.energy,accepted_rows=result.n_iter,
                        j_calls=jk.j_calls,k_calls=jk.k_calls,wall_seconds=elapsed)
            if backend=="opentrustregion":
                report=result.opentrustregion
                record.update(termination=report.termination,fock_evaluations=report.fock_evaluations,
                              response_evaluations=report.response_evaluations,trial_evaluations=report.trial_evaluations,
                              stability_converged=report.stability_converged,stable=report.stable)
                record["energy_difference"] = result.energy-references[0][0]
                record["density_difference"] = max(np.linalg.norm(d-e) for d,e in zip(densities,references[0][1]))
                record["same_solution"] = bool(result.converged and records[-1]["converged"]
                    and abs(record["energy_difference"])<1e-7 and record["density_difference"]<1e-5)
                if not record["same_solution"] and args.case != "water":
                    angle, difference = axial_density_alignment(basis, densities, references[0][1])
                    record["axial_rotation_radians"] = angle
                    record["axial_aligned_density_difference"] = difference
                    record["same_solution_after_axial_rotation"] = bool(
                        result.converged and records[-1]["converged"]
                        and abs(record["energy_difference"])<1e-7 and difference<1e-5)
            references.append((result.energy,densities))
            records.append(record)
    print(json.dumps(records,indent=2))


if __name__ == "__main__":
    main()
