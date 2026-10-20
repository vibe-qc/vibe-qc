"""Measure exact finite support preparation for global periodic RI pair bubbles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(repeats):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from vibeqc import manta,_vibeqc_core as core
    from tests.test_manta_periodic_gdf_pair_tree import pair_tree_case
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[]
    for order in (0,2,4):
        a,source=pair_tree_case(order=order)
        start=time.perf_counter();tree=manta.build_periodic_gdf_pair_tree(source,a['budget']);cold=time.perf_counter()-start
        counts=dict(pairs=len(tree.entries),nodes=len(tree.tree.nodes),images=0,included_pairs=0,
                    visited_nodes=0,refined_nodes=0,aggregated_blocks=0,aggregated_pairs=0)
        samples=[]
        for repeat in range(repeats):
            start=time.perf_counter();warm=manta.build_periodic_gdf_pair_tree(source,a['budget']);build=time.perf_counter()-start
            start=time.perf_counter();unions=[warm.auxiliary_images(i) for i in range(a['auxiliary'].nshells)];enumerate_time=time.perf_counter()-start
            start=time.perf_counter()
            partitions=[warm.partition(i,n) for i,images in enumerate(unions) for n in images]
            walk=time.perf_counter()-start
            samples.append(dict(tree_seconds=build,union_seconds=enumerate_time,partition_seconds=walk))
            if repeat==0:
                counts['images']=sum(map(len,unions));nodes=warm.tree.nodes
                for part in partitions:
                    counts['included_pairs']+=part.included_pairs
                    counts['visited_nodes']+=part.visited_nodes;counts['refined_nodes']+=part.refined_nodes
                    for id in part.included_nodes:
                        count=nodes[id].end-nodes[id].begin
                        if count>1:counts['aggregated_blocks']+=1;counts['aggregated_pairs']+=count
        rows.append(dict(order=order,cold_seconds=cold,samples=samples,modeled_peak_bytes=tree.plan.modeled_peak_bytes,
                         pair_capacity=tree.plan.pair_capacity,auxiliary_image_capacity=tree.plan.auxiliary_image_capacity,**counts))
    files=('cpp/src/periodic_gdf_short_range.cpp','cpp/include/vibeqc/manta_periodic_gdf_pair_tree.hpp',
           'cpp/src/manta_bubbles.cpp','tests/test_manta_periodic_gdf_pair_tree.py','benchmarks/manta_periodic_gdf_pair_tree.py')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
                native_sha256=sha(native),source_sha256={f:sha(root/f) for f in files},rows=rows,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);p.add_argument('--repeats',type=int,default=3);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.workers,a.repeats)<1:p.error('workers/repeats must be positive')
    if a.worker:print(json.dumps(worker(a.repeats),allow_nan=False));return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
    workers=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker','--repeats',str(a.repeats)],env=env,text=True)) for _ in range(a.workers)]
    print(json.dumps(dict(workers=workers),indent=2,allow_nan=False))


if __name__=='__main__':main()
