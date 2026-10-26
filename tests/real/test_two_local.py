import unittest
from unittest.mock import patch
import warnings
import numpy as np
from tests.real.synthetic_cases import case
from exp5 import common
from sklearn.exceptions import ConvergenceWarning
from exp5 import two_local
from exp5.two_solver import LocalSolveError,certificate
from paper_suite.algorithms.two_slr_paper import _lasso_candidates


class LocalSolverChecks(unittest.TestCase):
    def test_converged_author_path_unchanged(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(100,12));y=X[:,0]*.2+rng.normal(size=100)*.05
        expected=_lasso_candidates(X,y,2,12)
        actual,diag=two_local.candidates(X,y,2,12)
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(diag['retry_count'],0)

    def test_warning_forces_same_objective_retry(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(60,8));y=X[:,0]*.3+rng.normal(size=60)*.05
        original=two_local.Lasso;seen=[]
        def factory(**kw):
            model=original(**kw);fit=model.fit;seen.append(kw)
            def wrapped(a,b):
                result=fit(a,b)
                if kw['max_iter']==1000:warnings.warn('forced slow solver',ConvergenceWarning)
                return result
            model.fit=wrapped;return model
        with patch.object(two_local,'Lasso',side_effect=factory):
            model,alpha,diag=two_local.fit_checked(X,y,alpha=.01)
        self.assertEqual([r['max_iter'] for r in seen],[1000,8000])
        self.assertEqual(alpha,.01);self.assertEqual(diag['retry_count'],1)
        reference=original(alpha=.01,max_iter=8000).fit(X,y)
        np.testing.assert_array_equal(model.coef_,reference.coef_)

    def test_three_seeded_synthetic_users_produce_finite_candidates(self):
        for seed in (61,62,63):
            X,y=case(seed,n=60,p=12,correlated=seed!=61)
            support,diag=two_local.candidates(X,y,2,64)
            self.assertTrue(np.isfinite(support).all())
            self.assertGreaterEqual(len(support),2)
            self.assertLessEqual(len(support),4)
            self.assertGreaterEqual(diag['robust_fallback_count'],0)

    def test_unresolved_failure_does_not_emit_index_zero(self):
        from paper_suite.algorithms.two_slr_paper import TwoSLRProfile
        profile=TwoSLRProfile(2,1,4,1.,16)
        with patch.object(two_local,'candidates',side_effect=LocalSolveError('forced certificate failure')):
            with self.assertRaisesRegex(LocalSolveError,'Local user 0'):
                two_local.local_votes(np.ones((1,10,4)),np.arange(10.)[None,:],profile,np.random.default_rng(5))

    def test_wrong_solution_rejected_by_independent_certificate(self):
        X=np.array([[-1.],[1.]]);y=X[:,0]
        self.assertFalse(certificate(X,y,np.zeros(1),.1)['certified'])
        self.assertTrue(certificate(X,y,np.array([.9]),.1)['certified'])
