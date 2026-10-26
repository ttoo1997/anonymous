import unittest
from unittest.mock import patch
import numpy as np
from paper_suite.runner import run_algorithm
from paper_suite.config import SyntheticSetting
from paper_suite.data import make_synthetic


class UnscaledIHTTests(unittest.TestCase):
    def test_main_synthetic_calls_zhu24_estimator_directly(self):
        s=SyntheticSetting(n=80,m=5,d=8,k=2)
        d=make_synthetic(s,7,test_samples=1)
        with patch('paper_suite.runner.fit_ldp_iht',return_value=(np.zeros(8),{})) as f:
            run_algorithm('ldpiht',d,s,9)
        np.testing.assert_array_equal(f.call_args.args[0],d.user_x.reshape(-1,8))
        np.testing.assert_array_equal(f.call_args.args[1],d.user_y.ravel())
        self.assertNotIn('public_l2_bound',f.call_args.kwargs)

    def test_frozen_truth_is_inside_unit_l2_ball(self):
        s=SyntheticSetting(n=20,m=3,d=8,k=8)
        d=make_synthetic(s,7,test_samples=1)
        self.assertLessEqual(np.linalg.norm(d.beta_true),1.0)
        self.assertAlmostEqual(np.linalg.norm(d.beta_true,1),1.0)

if __name__=='__main__':unittest.main()
