import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from dp_safety.parameters import APPROXIMATE_DP_DELTA
from lasso_threshold_v1.mechanism.proposed import PROTOCOL
from paper_suite.algorithms.sample_sdp_core import Config, sdp2_table_parameters
from paper_suite.config import SyntheticSetting, synthetic_grid, validate_protocol_grid
from paper_suite.data import make_synthetic
from paper_suite.runner import run_algorithm, run_recall
from paper_suite.run_synthetic import EXPERIMENTS


class DeltaProtocolTests(unittest.TestCase):
    def test_all_grids_and_mechanisms_share_public_delta(self):
        self.assertEqual(APPROXIMATE_DP_DELTA, 1e-5)
        self.assertEqual(PROTOCOL.delta, APPROXIMATE_DP_DELTA)
        self.assertEqual(Config().delta, APPROXIMATE_DP_DELTA)
        for experiment in EXPERIMENTS:
            grid = synthetic_grid(experiment)
            validate_protocol_grid(experiment, grid)
            self.assertTrue(all(s.delta == APPROXIMATE_DP_DELTA for s in grid))

    def test_runner_passes_actual_delta_and_refuses_mismatches(self):
        setting = SyntheticSetting(n=64, m=20, d=4, k=2, epsilon=4.)
        data = make_synthetic(setting, 91, test_samples=16)
        with patch('paper_suite.runner.fit_sample_sdp', return_value=(np.zeros(4), {})) as fit:
            run_algorithm('sample_sdp', data, setting, 92)
            self.assertEqual(fit.call_args.kwargs['delta'], setting.delta)
        old = replace(setting, delta=1e-6)
        for entry in (run_algorithm, run_recall):
            with self.assertRaisesRegex(ValueError, 'delta disagrees'):
                entry('proposed', data, old, 92)

    def test_small_sample_fit_certificate_uses_new_delta(self):
        setting = SyntheticSetting(n=64, m=20, d=4, k=2, epsilon=4.)
        data = make_synthetic(setting, 93, test_samples=16)
        beta, detail = run_algorithm('sample_sdp', data, setting, 94)
        self.assertTrue(np.isfinite(beta).all())
        self.assertLessEqual(detail['privacy_certificate']['delta_total_bound'], setting.delta)
        old = sdp2_table_parameters(setting.n * setting.m, setting.d, setting.epsilon, 1e-6)
        new = sdp2_table_parameters(setting.n * setting.m, setting.d, setting.epsilon, setting.delta)
        self.assertNotEqual(old['total_trials'], new['total_trials'])
