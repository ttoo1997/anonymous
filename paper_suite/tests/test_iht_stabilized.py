import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from paper_suite.algorithms.ldp_iht import fit_ldp_iht
from paper_suite.algorithms.iht_stabilized_profile import stabilized_parameters
from paper_suite.runner import run_algorithm
from paper_suite.data import make_synthetic
from paper_suite.config import SyntheticSetting


class StabilizedIHTTests(unittest.TestCase):
    def test_production_matches_preexisting_diagnostic(self):
        import sys
        path=Path(__file__).resolve().parents[2]/'validation/iht_stabilized/diagnostic_reference.py'
        spec=importlib.util.spec_from_file_location('_iht_diagnostic_reference',path)
        ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
        rng=np.random.default_rng(818)
        x=rng.normal(size=(2103,16));y=x[:,0]*.125+rng.normal(size=len(x))*.25
        for eps in (.5,1.,2.,4.,8.,16.,32.):
            p=stabilized_parameters(8,len(x),eps)
            cfg=ref.Config('locked',mode=p['radius_mode'],**{k:v for k,v in p.items() if k not in ('radius_mode','vectorized')})
            actual,meta=fit_ldp_iht(x,y,eps,8,222,**p)
            expected,diag=ref.fit(x,y,eps,222,cfg)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual([t['report_radius'] for t in meta['iteration_diagnostics']], [t['gradient_radius'] for t in diag['trace']])
            self.assertEqual(meta['epsilon_ldp_target'],eps)
            self.assertEqual(meta['delta_ldp_target'],0.)
            self.assertFalse(meta['original_utility_theorem_claimed'])

    def test_adversarial_inputs_are_clipped_before_randomizer(self):
        rng=np.random.default_rng(6)
        x=rng.choice([-1e12,1e12],size=(100,12));y=np.full(100,1e12)
        calls=[]
        def record(v,r,e,rng):
            self.assertLessEqual(np.linalg.norm(v,axis=1).max(),r*(1+1e-12))
            self.assertLessEqual(r,4.)
            calls.append((len(v),r,e))
            return v
        with patch('paper_suite.algorithms.ldp_iht.privunit_batch',side_effect=record):
            beta,meta=fit_ldp_iht(x,y,4.,8,45,**stabilized_parameters(8,len(x),4.))
        self.assertEqual(sum(t[0] for t in calls),len(x))
        self.assertTrue(all(t[2]==4. for t in calls))
        self.assertLessEqual(np.linalg.norm(beta),1.+1e-12)
        self.assertLessEqual(np.count_nonzero(beta),8)
        self.assertTrue(any(t['gradient_clip_fraction']>0 for t in meta['iteration_diagnostics']))

    def test_dispatch_preserves_all_rows_and_epsilon(self):
        for eps in (.5,2.,4.,32.):
            st=SyntheticSetting(n=20,m=3,d=16,k=8,epsilon=eps)
            data=make_synthetic(st,12,test_samples=4)
            with patch('paper_suite.runner.fit_ldp_iht',return_value=(np.zeros(16),{})) as fn:
                _,meta=run_algorithm('ldpiht',data,st,20)
            self.assertEqual(fn.call_args.args[2],eps)
            np.testing.assert_array_equal(fn.call_args.args[0],data.user_x.reshape(-1,16))
            self.assertEqual(fn.call_args.kwargs,stabilized_parameters(8,60,eps))
            self.assertIn('gradient_clipped',meta['parameter_profile'])

    def test_invalid_radius_fails_closed(self):
        x=np.zeros((10,4));y=np.zeros(10)
        for cap in (-1.,0.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):fit_ldp_iht(x,y,4.,1,2,gradient_clip=cap)

if __name__=='__main__':unittest.main()
