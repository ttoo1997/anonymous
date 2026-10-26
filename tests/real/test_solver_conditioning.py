import importlib.util
import unittest
import numpy as np
from exp5 import two_local
from exp5.two_solver import _certified_conic_lasso,certificate
from tests.real.synthetic_cases import case


class NumericalConditioningChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('clarabel'), 'requires pinned clarabel dependency')
    def test_correlated_synthetic_fold_certificate(self):
        X,y=case(41,n=80,p=10,correlated=True)
        A=X-X.mean(0);t=y-y.mean();alpha=.04
        beta,meta=_certified_conic_lasso(A,t,alpha)
        self.assertTrue(certificate(A,t,beta,alpha)['certified'])
        self.assertIn(meta['conic_objective_scale'],(1.,1e4,1e6))

    def test_correlated_synthetic_user_candidates(self):
        X,y=case(42,n=70,p=10,correlated=True)
        support,diag=two_local.candidates(X,y,2,64)
        self.assertGreater(len(support),0)
        self.assertTrue(np.isfinite(support).all())
        self.assertGreaterEqual(diag['robust_checked_fits'],0)
