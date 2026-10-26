import math
import unittest

from lasso_threshold_v1.mechanism.privacy_accountant import (
    appendix_gamma_floor,
    certify_stage1_gamma,
    cheu_stream_epsilon_bound,
    local_vector_epsilon,
)
from paper_suite.config import EPSILON_GRID, synthetic_grid


class Stage1PrivacyAccountantTests(unittest.TestCase):
    def test_main_exp1_grid_is_certified(self):
        n1=7500; delta=1e-6
        for eps in EPSILON_GRID:
            cert=certify_stage1_gamma(n1,float(eps),delta)
            self.assertGreaterEqual(cert.gamma,cert.appendix_gamma_floor)
            self.assertGreaterEqual(cert.gamma,cert.certified_gamma_floor)
            self.assertLess(cert.gamma,.5)
            self.assertLessEqual(cert.certified_total_epsilon_bound,float(eps)*(1+1e-9))
            self.assertLessEqual(cert.certified_total_delta_bound,delta)

    def test_certificate_is_data_independent(self):
        a=certify_stage1_gamma(7500,4.,1e-6)
        b=certify_stage1_gamma(7500,4.,1e-6)
        self.assertEqual(a,b)

    def test_cheu_certificate_directly_rechecks(self):
        cert=certify_stage1_gamma(7500,4.,1e-6)
        self.assertEqual(cert.certificate_mode,'cheu_corollary_4_7_two_stream')
        bound=cheu_stream_epsilon_bound(cert.gamma,7500,5e-7)
        self.assertLessEqual(2*bound,4.)
        self.assertAlmostEqual(cert.certified_total_delta_bound,1e-6)

    def test_large_epsilon_can_use_pure_local_fallback(self):
        cert=certify_stage1_gamma(7500,16.,1e-6)
        self.assertEqual(cert.certificate_mode,'pure_local_vector')
        self.assertLessEqual(local_vector_epsilon(cert.gamma),16.)
        self.assertEqual(cert.certified_total_delta_bound,0.)

    def test_appendix_reference_floor_is_preserved(self):
        cert=certify_stage1_gamma(7500,.5,1e-6)
        expected=appendix_gamma_floor(7500,.5,1e-6,32.)
        self.assertAlmostEqual(cert.appendix_gamma_floor,expected)
        self.assertAlmostEqual(cert.gamma,expected)

    def test_every_formal_synthetic_setting_has_feasible_certificate(self):
        for experiment in ('exp1_epsilon','exp2_dimension','exp3_users','exp4a_recall_m','exp4b_recall_n'):
            for st in synthetic_grid(experiment):
                cert=certify_stage1_gamma(st.n//2,st.epsilon,st.delta)
                self.assertTrue(0 < cert.gamma < .5)
                self.assertLessEqual(cert.certified_total_epsilon_bound,st.epsilon*(1+1e-9))
                self.assertLessEqual(cert.certified_total_delta_bound,st.delta)


if __name__=='__main__':
    unittest.main()
