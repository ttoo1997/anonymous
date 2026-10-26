import unittest
import numpy as np
from exp5.runtime_checks import validate_fit


class RuntimeChecks(unittest.TestCase):
    def test_underbudget_and_overspend_are_distinguished(self):
        details=dict(budget={'record_epsilon':1.,'record_delta':1e-6},
                     privacy_certificate={'epsilon_noise_certified':.9,'delta_total_bound':8e-7},
                     estimator_variant='SampleSDP-Ridge',privacy_postprocessing_only=True,regularized_min_eigenvalue=.1)
        validate_fit('sample_sdp',np.array([.1,0.]),details,100,2)
        details['privacy_certificate']['epsilon_noise_certified']=1.1
        with self.assertRaises(ValueError):validate_fit('sample_sdp',np.array([.1,0.]),details,100,2)

    def test_old_totalized_baseline_is_rejected(self):
        details=dict(budget={'epsilon':1.,'delta':0.},local_lasso_totalization_count=1)
        with self.assertRaisesRegex(ValueError,'Unresolved 2-SLR'):
            validate_fit('2slr',np.array([.2]),details,100,1)

    def test_unresolved_u2r_local_fit_is_rejected(self):
        details=dict(budget={'epsilon':1.,'delta':1e-6},local_lasso_totalization_count=1)
        with self.assertRaisesRegex(ValueError,'Unresolved U2R'):
            validate_fit('u2r',np.array([.2]),details,100,1)
