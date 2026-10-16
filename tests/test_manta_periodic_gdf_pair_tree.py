"""Ordered GDF pair bubbles: exact capsule masks and source-owned Bloch frames."""
from concurrent.futures import ThreadPoolExecutor
import gc
import itertools

import numpy as np
import pytest
from scipy.special import hyp1f1

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_ri_multipole import multipole_case
from tests.test_manta_periodic_ri_compressed import angular_case
from tests.test_manta_periodic_fitting import CAPS,libint_reference
from tests.test_manta_periodic_gdf import independent,gaussian_fourier
from tests.test_manta_periodic_gdf_actions import inputs


def pair_tree_case(gamma=False,shift=False,order=2):
    a=multipole_case(gamma,shift,order=order)
    return a,manta.PeriodicGDFTiles(**a)


def expand(tree,partition):
    nodes,order=tree.tree.nodes,tree.tree.pair_order
    return [order[k] for i in partition.included_nodes for k in range(nodes[i].begin,nodes[i].end)]


def frames(a):
    inverse=np.linalg.inv(a['system'].lattice)
    return [np.floor(np.array([s.origin for s in a[name].shells()])@inverse.T+1e-12).astype(np.int64)
            for name in ('orbital','auxiliary')]


def expected_membership(a,tree,auxiliary,image):
    orbital,aux=a['orbital'].shells(),a['auxiliary'].shells();of,af=frames(a);l=a['system'].lattice
    out=[]
    for e in tree.entries:
        A=np.array(orbital[e.bra_shell].origin);B=np.array(orbital[e.ket_shell].origin)+l@e.ket_image
        P=np.array(aux[auxiliary].origin)+l@(np.asarray(image)+of[e.bra_shell]-af[auxiliary])
        segment=B-A;fraction=np.clip(np.dot(P-A,segment)/np.dot(segment,segment),0,1) if np.dot(segment,segment) else 0
        out.append(np.linalg.norm(P-A-fraction*segment)<=a['options'].auxiliary_cutoff_bohr)
    return np.array(out)


@pytest.mark.parametrize('shift',[False,True])
@pytest.mark.parametrize('order',[0,2,4])
def test_ordered_inventory_matches_independent_integer_spheres(shift,order):
    a,source=pair_tree_case(shift=shift,order=order);t=manta.build_periodic_gdf_pair_tree(source,a['budget'])
    l=a['system'].lattice;shells=a['orbital'].shells();expected=[]
    for i,A in enumerate(shells):
        for j,B in enumerate(shells):
            ordinal=0
            for n in itertools.product(range(-5,6),repeat=3):
                if np.linalg.norm(np.array(A.origin)-B.origin-l@n)<=a['options'].pair_cutoff_bohr:
                    expected.append((i,j,ordinal,n));ordinal+=1
    assert [(e.bra_shell,e.ket_shell,e.pair_ordinal,tuple(e.ket_image)) for e in t.entries]==expected
    assert t.source_identity==source.three_center_source_identity
    assert len(t.entries)<=t.plan.pair_capacity and len(t.tree.nodes)<=t.plan.node_capacity
    assert source.preparation.pair_extents==len(t.entries) and source.preparation.moment_panels==0
    for e,extent in zip(t.entries,t.tree.pairs):
        np.testing.assert_allclose(extent.center,(np.array(e.bra_center)+e.ket_center)/2-e.frame_translation,atol=2e-14)
    assert sorted(t.tree.pair_order)==list(range(len(t.entries)))


@pytest.mark.parametrize('shift',[False,True])
@pytest.mark.parametrize('cutoff',[1.7,5.,8.])
def test_mask_nodes_and_partition_match_every_exact_capsule(shift,cutoff):
    a,_=pair_tree_case(shift=shift);a['options'].auxiliary_cutoff_bohr=cutoff
    source=manta.PeriodicGDFTiles(**a);t=manta.build_periodic_gdf_pair_tree(source,a['budget'])
    saw_mixed=saw_aggregate=False;nodes=t.tree.nodes;permutation=t.tree.pair_order
    for auxiliary in range(a['auxiliary'].nshells):
        for image in itertools.product(range(-2,3),repeat=3):
            expected=expected_membership(a,t,auxiliary,image)
            part=t.partition(auxiliary,image);actual=expand(t,part)
            assert len(set(actual))==len(actual)==part.included_pairs
            assert set(actual)==set(np.flatnonzero(expected))
            assert part.included_pairs+part.excluded_pairs==len(t.entries)
            assert part.source_identity==source.three_center_source_identity
            for id,node in enumerate(nodes):
                mask=t.classify(id,auxiliary,image);members=expected[np.asarray(permutation[node.begin:node.end])]
                assert mask.pairs==len(members)
                if mask.membership=='inside':assert np.all(members)
                elif mask.membership=='outside':assert not np.any(members)
                else:assert mask.membership=='mixed' and len(node.children)>0;saw_mixed=True
            saw_aggregate|=any(nodes[i].end-nodes[i].begin>1 for i in part.included_nodes)
    assert saw_mixed and saw_aggregate


@pytest.mark.parametrize('shift',[False,True])
@pytest.mark.parametrize('gamma',[False,True])
def test_factored_phases_equal_original_source_frames(shift,gamma):
    a,s=pair_tree_case(gamma,shift);t=manta.build_periodic_gdf_pair_tree(s,a['budget'])
    of,af=frames(a);l=a['system'].lattice
    for i,e in enumerate(t.entries):
        assert tuple(e.bra_frame)==tuple(of[e.bra_shell])
        for auxiliary in range(a['auxiliary'].nshells):
            for image in [(-2,1,0),(0,0,0),(1,-1,2)]:
                original=np.array(image)+of[e.bra_shell]-af[auxiliary]
                assert tuple(t.source_auxiliary_image(i,auxiliary,image))==tuple(original)
                phase=np.asarray(t.pair_phase(i))*t.auxiliary_phase(auxiliary,image)
                expected=np.exp(1j*(a['ket_kpoints']@(l@e.ket_image)-a['q']@(l@original)))
                np.testing.assert_allclose(phase,expected,atol=3e-14,rtol=0)


def tree_short_range(a,tree):
    """Use native masked tree/phase traversal and independent s-Gaussian integrals."""
    orbital,aux=a['orbital'].shells(),a['auxiliary'].shells();l=a['system'].lattice;o=a['options'];entries=tree.entries
    result=np.zeros((len(a['ket_kpoints']),len(aux),len(orbital),len(orbital)),complex)
    for ip,sp in enumerate(aux):
        c=sp.exponents[0]
        for image in tree.auxiliary_images(ip):
            partition=tree.partition(ip,image)
            for index in expand(tree,partition):
                e=entries[index];sm,sn=orbital[e.bra_shell],orbital[e.ket_shell]
                A=np.array(sm.origin);B=np.array(sn.origin)+l@e.ket_image
                P=np.array(sp.origin)+l@tree.source_auxiliary_image(index,ip,image)
                alpha,beta=sm.exponents[0],sn.exponents[0];p=alpha+beta;rho=c*p/(c+p)
                X=(alpha*A+beta*B)/p;u=rho*np.sum((P-X)**2);eta=o.omega/np.sqrt(o.omega**2+rho)
                integral=2*np.pi**2.5/(c*p*np.sqrt(c+p))*np.exp(-alpha*beta/p*np.sum((A-B)**2))*(
                    hyp1f1(.5,1.5,-u)-eta*hyp1f1(.5,1.5,-eta*eta*u))
                result[:,ip,e.bra_shell,e.ket_shell]+=np.asarray(tree.pair_phase(index))*tree.auxiliary_phase(ip,image)*integral
    return result


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('shift',[False,True])
def test_complete_short_range_coupling_from_tree_against_independent_libint(gamma,shift):
    a,s=pair_tree_case(gamma,shift);t=manta.build_periodic_gdf_pair_tree(s,a['budget'])
    np.testing.assert_allclose(tree_short_range(a,t),libint_reference(a)[1],atol=5e-12,rtol=3e-13)


@pytest.mark.parametrize('pure',[False,True])
def test_signed_angular_extent_tree_and_original_ownership(pure):
    a=angular_case(pure);s=manta.PeriodicGDFTiles(**a);t=manta.build_periodic_gdf_pair_tree(s,a['budget'])
    old=t.partition(0,[0,0,0]);del s;gc.collect()
    assert t.partition(0,[0,0,0]).included_nodes==old.included_nodes
    assert t.tree.pairs[0].full_moment_bounds[0]>0
    a['options'].auxiliary_cutoff_bohr=.01
    assert t.partition(0,[1,0,0]).included_pairs==len(t.entries)


@pytest.mark.parametrize('field,quantity',CAPS)
def test_pair_tree_budget_admission_precedes_preparation(field,quantity):
    a,s=pair_tree_case();p=manta.plan_periodic_gdf_pair_tree(s,a['budget']);b=a['budget']
    for name,value in CAPS:setattr(b,name,getattr(p,value))
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.build_periodic_gdf_pair_tree(s,b)
    assert s.preparation.pair_extents==0
    setattr(b,field,getattr(p,quantity));manta.build_periodic_gdf_pair_tree(s,b)


def test_concurrent_queries_and_source_lifetime_are_immutable():
    a,s=pair_tree_case();t=manta.build_periodic_gdf_pair_tree(s,a['budget']);images=list(itertools.product(range(-1,2),repeat=3))
    expected=[t.partition(0,n).included_nodes for n in images];del s;gc.collect()
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual=list(pool.map(lambda n:t.partition(0,n).included_nodes,images))
    assert actual==expected
    with pytest.raises((ValueError,IndexError)):t.partition(20,[0,0,0])
    with pytest.raises((ValueError,IndexError)):t.classify(10**6,0,[0,0,0])
    with pytest.raises(ValueError,match='coordinates'):t.partition(0,[2*10**12,0,0])


@pytest.mark.parametrize('route',['exact','unprepared','reciprocal'])
def test_unqualified_sources_do_not_supply_a_pair_tree(route):
    a,_=pair_tree_case()
    if route=='exact':a['options'].coupling_multipole_error=0
    elif route=='unprepared':a['options'].coupling_multipole_reuse=False
    else:
        k=manta.PeriodicGDFKernelOptions();k.kernel='erf';k.omega=.3;a['physical_kernel']=k
    s=manta.PeriodicGDFTiles(**a)
    with pytest.raises(ValueError,match='prepared real short-range'):manta.build_periodic_gdf_pair_tree(s,a['budget'])



@pytest.mark.parametrize('direction',[-1,0,1])
def test_capsule_cutoff_equality_uses_exact_source_predicate(direction):
    a=angular_case(False);radius=6.
    if direction:radius=np.nextafter(radius,-np.inf if direction<0 else np.inf)
    a['options'].auxiliary_cutoff_bohr=radius
    source=manta.PeriodicGDFTiles(**a);t=manta.build_periodic_gdf_pair_tree(source,a['budget'])
    assert len(t.entries)==1
    part=t.partition(0,[1,0,0])
    assert part.included_pairs==int(direction>=0)


def test_canonical_tree_and_masks_are_covariant_under_atom_images():
    a,s=pair_tree_case();moved,ms=pair_tree_case(shift=True)
    t=manta.build_periodic_gdf_pair_tree(s,a['budget']);mt=manta.build_periodic_gdf_pair_tree(ms,moved['budget'])
    assert t.tree.pair_order==mt.tree.pair_order
    np.testing.assert_allclose([n.center for n in t.tree.nodes],[n.center for n in mt.tree.nodes],atol=2e-14,rtol=0)
    for ip in range(a['auxiliary'].nshells):
        for image in itertools.product(range(-2,3),repeat=3):
            assert set(expand(t,t.partition(ip,image)))==set(expand(mt,mt.partition(ip,image)))



def test_independent_ao_auxiliary_images_and_reciprocal_relabeling():
    a,s=pair_tree_case();t=manta.build_periodic_gdf_pair_tree(s,a['budget']);l=a['system'].lattice
    moved=dict(a)
    shifts={'orbital':np.array([[0,0,0],[2,-1,0]]),'auxiliary':np.array([[-1,0,1],[0,2,-1]])}
    for name,images in shifts.items():
        shells=a[name].shells()
        for shell,image in zip(shells,images):shell.origin=(np.array(shell.origin)+l@image).tolist()
        molecule=vq.Molecule([vq.Atom(1,shell.origin) for shell in shells],0,1)
        moved[name]=vq.BasisSet(molecule,shells,'pair-tree-image-gauge',True)
    mt=manta.build_periodic_gdf_pair_tree(manta.PeriodicGDFTiles(**moved),a['budget'])
    reciprocal=a['system'].reciprocal_lattice()@np.array([1,-1,2])
    relabel=dict(a,q=a['q']+reciprocal,ket_kpoints=a['ket_kpoints']+reciprocal)
    rt=manta.build_periodic_gdf_pair_tree(manta.PeriodicGDFTiles(**relabel),a['budget'])
    np.testing.assert_allclose([e.center for e in t.tree.pairs],[e.center for e in mt.tree.pairs],atol=3e-14,rtol=0)
    for i,e in enumerate(t.entries):
        for ip in range(a['auxiliary'].nshells):
            image=[1,-1,0];base=np.asarray(t.pair_phase(i))*t.auxiliary_phase(ip,image)
            factor=np.exp(1j*((a['ket_kpoints']-a['q'])@(l@shifts['orbital'][e.bra_shell])-
                a['ket_kpoints']@(l@shifts['orbital'][e.ket_shell])+a['q']@(l@shifts['auxiliary'][ip])))
            np.testing.assert_allclose(np.asarray(mt.pair_phase(i))*mt.auxiliary_phase(ip,image),base*factor,atol=5e-14,rtol=0)
            np.testing.assert_allclose(np.asarray(rt.pair_phase(i))*rt.auxiliary_phase(ip,image),base,atol=5e-14,rtol=0)



@pytest.mark.parametrize('shift',[False,True])
@pytest.mark.parametrize('radius',[1.7,8.])
def test_complete_auxiliary_union_matches_independent_capsules(shift,radius):
    a,_=pair_tree_case(shift=shift);a['options'].auxiliary_cutoff_bohr=radius
    t=manta.build_periodic_gdf_pair_tree(manta.PeriodicGDFTiles(**a),a['budget'])
    for auxiliary in range(a['auxiliary'].nshells):
        images=[tuple(n) for n in t.auxiliary_images(auxiliary)]
        assert images==sorted(set(images)) and len(images)<=t.plan.auxiliary_image_capacity
        expected=[n for n in itertools.product(range(-4,5),repeat=3) if np.any(expected_membership(a,t,auxiliary,n))]
        assert images==expected
