import warnings
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from paper_suite.algorithms.nonprivate_lasso import fit_lasso
from paper_suite.data import RegressionData


class LassoWarningPolicyTest(unittest.TestCase):
    def test_convergence_warning_does_not_fail_even_when_environment_promotes_warnings(self):
        x=np.arange(24,dtype=float).reshape(6,4)/10
        y=np.arange(6,dtype=float)
        data=RegressionData(
            user_x=x.reshape(3,2,4), user_y=y.reshape(3,2),
            x_test=x[:2], y_test=y[:2], beta_true=None,
            lasso_x=x, lasso_y=y)

        def fake_fit(model, X, Y):
            warnings.warn('fixture did not fully converge', ConvergenceWarning)
            model.coef_=np.array([1.,0.,0.,0.])
            model.alpha_=0.01
            return model

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with patch('paper_suite.algorithms.nonprivate_lasso.LassoCV.fit', new=fake_fit):
                beta,meta=fit_lasso(data)
        np.testing.assert_array_equal(beta,np.array([1.,0.,0.,0.]))
        self.assertTrue(meta['convergence_warning'])
        self.assertEqual(meta['convergence_warning_count'],1)
        self.assertIn('fixture did not fully converge',meta['convergence_warning_messages'][0])

    def test_nonfinite_solution_still_fails(self):
        x=np.zeros((6,4)); y=np.zeros(6)
        data=RegressionData(
            user_x=x.reshape(3,2,4), user_y=y.reshape(3,2),
            x_test=x[:2], y_test=y[:2], beta_true=None,
            lasso_x=x, lasso_y=y)
        def fake_fit(model, X, Y):
            model.coef_=np.array([np.nan,0.,0.,0.])
            model.alpha_=0.01
            return model
        with patch('paper_suite.algorithms.nonprivate_lasso.LassoCV.fit', new=fake_fit):
            with self.assertRaisesRegex(ValueError,'non-finite'):
                fit_lasso(data)

if __name__=='__main__':unittest.main()
