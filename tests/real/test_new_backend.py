import tempfile,unittest
from unittest.mock import patch
import numpy as np
from exp5 import common
from exp5.algorithms import fit
from exp5.parameter_policy import resolve_parameters
from exp5.runtime_checks import validate_fit
from dp_safety.parameters import APPROXIMATE_DP_DELTA
from lasso_threshold_v1.mechanism.proposed import PROTOCOL
from paper_suite.algorithms.iht_stabilized_profile import stabilized_parameters

class NewBackendTests(unittest.TestCase):
    def test_delta_one_source(self):
        self.assertEqual(APPROXIMATE_DP_DELTA,1e-5)
        self.assertEqual(PROTOCOL.delta,1e-5)

    def test_ldpiht_real_adapter_matches_latest_estimator(self):
        from paper_suite.algorithms.ldp_iht import fit_ldp_iht
        rng=np.random.default_rng(12);X=rng.normal(size=(1000,16));y=X[:,0]*.2+rng.normal(size=1000)*.1
        with tempfile.TemporaryDirectory() as cache:
            for eps in (1.,4.):
                p=resolve_parameters('ldpiht',{'parameter_policy':'frozen_stabilized_epsilon_v1','k':8},eps,10,'native',len(X))
                a,meta=fit('ldpiht',X,y,10,eps,'native',p,17,cache,'fake',{})
                ref,_=fit_ldp_iht(X,y,eps,8,17,**stabilized_parameters(8,len(X),eps))
                np.testing.assert_array_equal(a,ref)
                self.assertEqual(meta['budget']['record_epsilon'],eps)
                self.assertEqual(meta['budget']['record_delta'],0.)
                validate_fit('ldpiht',a,meta,len(X),16)
                meta['iteration_diagnostics'][0]['report_radius']/=2
                with self.assertRaisesRegex(ValueError,'radii differ'):validate_fit('ldpiht',a,meta,len(X),16)

    def test_sample_delta_passed_to_noise_backend_without_epsilon_division(self):
        X=np.zeros((100,4));y=np.zeros(100)
        with tempfile.TemporaryDirectory() as cache,patch('exp5.algorithms.fit_sample',return_value=(np.zeros(4),{})) as f:
            _,meta=fit('sample_sdp',X,y,10,4.,'native',{},9,cache,'fake',{})
            self.assertEqual(f.call_args.args[2:4],(4.,1e-5))
            self.assertEqual(meta['budget']['record_delta'],1e-5)

    def test_no_hidden_parameter_overrides(self):
        with self.assertRaisesRegex(ValueError,'Unexpected overrides'):
            resolve_parameters('ldpiht',{'parameter_policy':'frozen_stabilized_epsilon_v1','k':8,'step_size':99},4.,10,'native',100)

if __name__=='__main__':unittest.main()
