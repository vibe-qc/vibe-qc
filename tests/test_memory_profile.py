"""Pure-Python scoped residual calibration contract; no scientific validation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).parents[1] / 'python/vibeqc/memory_profile.py'
if not path.exists():
    path = Path(__file__).with_name('memory_profile.py')
spec = importlib.util.spec_from_file_location('memory_profile', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.domain = dict(runtime_source='a'*40, native_sha256='b'*64,
                           hardware_class='x86-test', harness_sha256='c'*64,
                           method='rhf', route='dense_conventional',
                           threads_per_rank=1, nbf={'min': 2, 'max': 8})
        self.rows = [dict(observation_sha256='d'*64, peak_bytes=1200, allocation_bytes=200, nbf=2),
                     dict(observation_sha256='e'*64, peak_bytes=1600, allocation_bytes=400, nbf=8)]
        self.profile = m.fit_runtime_memory_profile(self.domain, self.rows)

    def estimate(self, n=2, route='dense_conventional'):
        # A monotone allocation fixture; runtime baseline is NOT included.
        term = dict(name='ERI', route=route, scope='rank', value_bytes=n**4*8, units='bytes', inputs={'nbf':n}, valid_domain={'nbf':{'min':1}})
        return SimpleNamespace(by_category={'ERI':term['value_bytes'], 'Python runtime + NumPy overhead':100},
                               resolved_features=dict(nbf=n, algorithm=route, threads_per_rank=1),
                               headroom_factor=1.5,
                               analytical_memory=dict(scope='rank', complete=False,
                                                      terms_json=json.dumps([term]), unknown_json='["unqualified"]'))

    def apply(self, e, **kwargs):
        with m.runtime_memory_profile(self.profile, self.domain):
            m.apply_runtime_memory_profile(e, method='rhf', **kwargs)
        return e.analytical_memory

    def test_rule_and_headroom(self):
        self.assertEqual(self.profile['value_bytes'],1200)
        x=self.apply(self.estimate());self.assertEqual(x['raw_peak_bytes'],1328)
        self.assertEqual(x['estimated_peak_bytes'],1992)
        self.assertNotIn('empirical',x['terms_json'])
        self.assertEqual(json.loads(x['empirical_baseline_json'])['profile_sha256'],m._digest(self.profile))

    def test_monotonic_domain(self):
        peaks=[self.apply(self.estimate(n))['raw_peak_bytes'] for n in range(2,9)]
        self.assertEqual(peaks,sorted(peaks))
        self.assertFalse(self.apply(self.estimate(9))['complete'])

    def test_unknown_routes_and_response(self):
        self.assertFalse(self.apply(self.estimate(route='direct_scf'))['complete'])
        self.assertFalse(self.apply(self.estimate(),response=True)['complete'])
        e=self.estimate();e.by_category['missing']=10
        self.assertFalse(self.apply(e)['complete'])

    def test_runtime_mismatch_and_default(self):
        e=self.estimate();m.apply_runtime_memory_profile(e,method='rhf');self.assertFalse(e.analytical_memory['complete'])
        with m.runtime_memory_profile(self.profile,{**self.domain,'native_sha256':'f'*64}):
            m.apply_runtime_memory_profile(e,method='rhf')
        self.assertFalse(e.analytical_memory['complete'])

    def test_no_refit_or_false_bounds(self):
        with self.assertRaises(ValueError):
            with m.runtime_memory_profile({**self.profile,'value_bytes':1201},self.domain): pass
        with self.assertRaises(ValueError):
            m.fit_runtime_memory_profile({**self.domain,'nbf':{'min':1,'max':8}},self.rows)
        with self.assertRaises(ValueError):
            m.fit_runtime_memory_profile(self.domain,self.rows+self.rows)

    def test_same_nbf_routes(self):
        direct_domain={**self.domain,'route':'direct_scf'}
        direct=m.fit_runtime_memory_profile(direct_domain,self.rows)
        e=self.estimate(route='direct_scf');term=json.loads(e.analytical_memory['terms_json'])[0]
        term['value_bytes']=64;e.analytical_memory['terms_json']=json.dumps([term]);e.by_category['ERI']=64
        with m.runtime_memory_profile(direct,direct_domain):m.apply_runtime_memory_profile(e,method='rhf')
        self.assertEqual(e.analytical_memory['raw_peak_bytes'],1264)
        self.assertNotEqual(e.analytical_memory['raw_peak_bytes'],self.apply(self.estimate())['raw_peak_bytes'])

if __name__=='__main__': unittest.main()
