import importlib.util
import unittest
import numpy as np
from exp5.algorithms import cap_u2r_support
from exp5.runtime_checks import validate_fit
from exp5.two_solver import _certified_conic_lasso, certificate
from exp5 import two_local
from tests.real.synthetic_cases import case


class SupportAndSolverChecks(unittest.TestCase):
    def test_eleven_features_ten_samples_has_fixed_public_cap(self):
        chosen=np.arange(11);scores=np.arange(11,dtype=float)
        result,meta=cap_u2r_support(chosen,scores,10,8,'min_k_m_minus_2')
        np.testing.assert_array_equal(result,np.arange(3,11))
        self.assertEqual(meta['threshold_selected_count'],11)
        self.assertEqual(meta['public_support_cap'],8)
        self.assertTrue(meta['support_cap_dp_postprocessing'])

    def test_no_change_below_cap_and_deterministic_ties(self):
        selected=np.array([1,3,6]);scores=np.zeros(15)
        result,meta=cap_u2r_support(selected,scores,10,8,'min_k_m_minus_2')
        np.testing.assert_array_equal(result,selected)
        self.assertFalse(meta['support_cap_applied'])
        result,_=cap_u2r_support(np.arange(11),scores,10,8,'min_k_m_minus_2')
        np.testing.assert_array_equal(result,np.arange(8))

    def test_undefined_ols_can_no_longer_be_marked_valid(self):
        detail=dict(budget={'epsilon':1.,'delta':1e-5},algorithm_status='ok',
                    local_lasso_totalization_count=0,local_ols_fallback_count=5000,
                    appendix_local_inverse_applicable=False)
        with self.assertRaisesRegex(ValueError,'OLS inverse inapplicable'):
            validate_fit('u2r',np.zeros(90),detail,100000,90)

    @unittest.skipUnless(importlib.util.find_spec('clarabel'), 'requires pinned clarabel dependency')
    def test_seeded_synthetic_lasso_certificate(self):
        X,y=case(31,n=70,p=8)
        A=X-X.mean(0);t=y-y.mean();alpha=.02
        beta,cert=_certified_conic_lasso(A,t,alpha)
        self.assertTrue(cert['certified'])
        self.assertTrue(certificate(A,t,beta,alpha)['certified'])

    def test_seeded_synthetic_user_produces_finite_candidate_set(self):
        X,y=case(32,n=50,p=12)
        support,diag=two_local.candidates(X,y,2,64)
        self.assertTrue(np.isfinite(support).all())
        self.assertGreaterEqual(len(support),2)
        self.assertLessEqual(len(support),4)
        self.assertGreaterEqual(diag['robust_checked_fits'],0)
