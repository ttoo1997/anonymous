import json
import unittest
from pathlib import Path

from paper_suite.algorithms.iht_selected_profile import selected_parameters
from paper_suite.algorithms.ldp_iht import (
    ma24_empirical_parameter_grid,
    qualification_parameter_grid,
)
from paper_suite.validate_ldpiht import _grid


class LDPIHTQualificationGridTests(unittest.TestCase):
    def test_frozen_qualification_grid_is_exact_and_source_bounded(self):
        grid = qualification_parameter_grid()
        expected = [
            dict(iterations=2, step_size=.01, x_clip=2., y_clip=2., sparsity=10, vectorized=True),
            dict(iterations=5, step_size=.01, x_clip=2., y_clip=2., sparsity=10, vectorized=True),
            dict(iterations=10, step_size=.01, x_clip=2., y_clip=2., sparsity=10, vectorized=True),
            dict(iterations=20, step_size=.01, x_clip=2., y_clip=2., sparsity=10, vectorized=True),
            dict(iterations=2, step_size=.1, x_clip=2., y_clip=2., sparsity=10, vectorized=True),
            dict(iterations=10, step_size=.1, x_clip=2., y_clip=2., sparsity=10, vectorized=True),
            dict(iterations=20, step_size=.1, x_clip=2., y_clip=2., sparsity=10, vectorized=True),
            dict(iterations=2, step_size=.01, x_clip=2., y_clip=2., sparsity=5, vectorized=True),
        ]
        self.assertEqual(grid, expected)
        reported = ma24_empirical_parameter_grid()
        self.assertTrue(all(cfg in reported for cfg in grid))

    def test_formal_profile_is_in_qualification_grid(self):
        self.assertIn(selected_parameters(8, 750000), qualification_parameter_grid())

    def test_validation_dispatch_exposes_qualification_grid(self):
        self.assertEqual(_grid('qualification', 750000, 8), qualification_parameter_grid())

    def test_selected_profile_matches_fixed_configuration(self):
        root = Path(__file__).resolve().parents[2]
        data = json.loads((root / 'paper_suite/algorithms/iht_paper_profile.json').read_text())
        values = data['fixed_selected_values']
        expected = dict(iterations=values['iterations'], step_size=values['step_size'],
                        x_clip=values['x_clip_tau1'], y_clip=values['y_clip_tau2'],
                        sparsity=values['sparsity_k_prime'], vectorized=True)
        self.assertEqual(selected_parameters(8, 750000), expected)
        self.assertEqual(len(data['qualification_grid']['profiles']), 8)


if __name__ == '__main__':
    unittest.main()
