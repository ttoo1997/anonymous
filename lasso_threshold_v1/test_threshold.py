import unittest
import math
import numpy as np
from unittest.mock import patch
from .mechanism.local import candidates
from .mechanism.proposed import select_support,fit,PROTOCOL
from .dataset import synthetic,public_signal_lower_bound,public_parameter_l2_norm

class ThresholdTests(unittest.TestCase):
    def test_algorithm3_threshold_is_rho_over_two(self):
        beta=np.array([.299999,.3,-.3,0.,.4])
        with patch('lasso_threshold_v1.mechanism.local._fit_local',return_value=(beta,'coordinate_descent',None)):
            supports,meta=candidates(np.ones((1,10,5)),np.ones((1,10)),2,total_users=80,rho=.6)
        np.testing.assert_array_equal(supports[0],[1,2,4])
        self.assertEqual(meta['coefficient_threshold'],.3)

    def test_appendix_b_stage1_parameters(self):
        x,y,_,_,_=synthetic(512,50,16,2,42,1)
        rho=public_signal_lower_bound(2)
        s,meta=select_support(x,y,4.,2,123,rho=rho)
        self.assertEqual(meta['coefficient_threshold'],rho/2)
        self.assertEqual(meta['rho'],rho)
        self.assertEqual(meta['protocol']['version'],'appendix_b_alg3_alg4_v6_private_release_boundary')
        self.assertGreaterEqual(meta['gamma'],32*math.log(1e6)/(256*16))
        self.assertLessEqual(meta['stage1_privacy_certificate']['certified_total_epsilon_bound'],4.)
        self.assertEqual(meta['hh_threshold'],32)

    def test_stage2_requires_public_parameter_norm(self):
        x,y,_,_,_=synthetic(512,50,16,2,42,1)
        with self.assertRaisesRegex(ValueError,'parameter_l2_norm'):
            fit(x,y,4.,2,123,rho=public_signal_lower_bound(2))
        beta,meta=fit(x,y,4.,2,123,rho=public_signal_lower_bound(2),
                      parameter_l2_norm=public_parameter_l2_norm(2))
        self.assertEqual(beta.shape,(16,))
        self.assertAlmostEqual(meta['coarse_radius'],public_parameter_l2_norm(2)+meta['tau_beta'])

if __name__=='__main__':unittest.main()
