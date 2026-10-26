import importlib.util
import unittest
import numpy as np
from exp5 import two_local
from exp5.two_solver import practical_certificate,_certified_conic_lasso
from tests.real.synthetic_cases import case


class NumericalTolerance(unittest.TestCase):
    def test_materially_wrong_and_nonfinite_solutions_rejected(self):
        X=np.array([[-1.],[1.]]);y=X[:,0]
        self.assertFalse(practical_certificate(X,y,np.array([0.]),.1)['numerical_accepted'])
        self.assertFalse(practical_certificate(X,y,np.array([np.nan]),.1)['numerical_accepted'])
        self.assertTrue(practical_certificate(X,y,np.array([.9]),.1)['numerical_accepted'])

    @unittest.skipUnless(importlib.util.find_spec('clarabel'), 'requires pinned clarabel dependency')
    def test_synthetic_fold_meets_numerical_accuracy(self):
        X,y=case(51,n=70,p=8,correlated=True)
        A=X-X.mean(0);t=y-y.mean();alpha=.03
        b,meta=_certified_conic_lasso(A,t,alpha)
        self.assertTrue(practical_certificate(A,t,b,alpha)['numerical_accepted'])
        self.assertLess(meta['dual_gap'],1e-7)

    def test_synthetic_user_finishes(self):
        X,y=case(52,n=70,p=12)
        support,diag=two_local.candidates(X,y,2,64)
        self.assertTrue(len(support)>0)
        self.assertGreaterEqual(diag['accuracy_warning_count'],0)
