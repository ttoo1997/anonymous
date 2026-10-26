import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from paper_suite.config import SyntheticSetting
from paper_suite.data import load_tabular_csv, make_synthetic
from paper_suite.runner import evaluate, fit_lasso, run_algorithm, write_outputs
from paper_suite.run_synthetic import run
from paper_suite.run_real import run as run_real


class AuditRegressions(unittest.TestCase):
    def test_epsilon_changes_do_not_change_data(self):
        s = SyntheticSetting(n=20, m=3, d=6, k=2)
        a, b = make_synthetic(s, 7), make_synthetic(replace(s, epsilon=32), 7)
        for key in ("user_x", "user_y", "x_test", "y_test", "beta_true"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))

    def test_ldpiht_uses_all_rows(self):
        s = SyntheticSetting(n=20, m=3, d=6, k=2)
        data = make_synthetic(s, 7, test_samples=10)
        with patch("paper_suite.runner.fit_ldp_iht", return_value=(np.zeros(6), {})) as fit:
            run_algorithm("ldpiht", data, s, 8)
        np.testing.assert_array_equal(fit.call_args.args[0], data.user_x.reshape(-1, 6))
        self.assertEqual(len(fit.call_args.args[1]), 60)

    def test_paper_baseline_cannot_be_silently_substituted(self):
        s = SyntheticSetting(n=120, m=10, d=6, k=2)
        data = make_synthetic(s, 7, test_samples=10)
        row = evaluate("2slr", data, s, 8)
        self.assertEqual(row["status"], "ok")
        self.assertIn("ma24_algorithm1_alg7to9_paper_aligned_v4_hh_alpha0p05_exp5_context", row['details_json'])

    def test_no_output_is_written_for_invalid_metric(self):
        rows = [{"experiment": "exp1", "algorithm": "x", "repeat": 0,
                 "test_mse": np.inf, "support_recall": np.nan}]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"bad"
            with self.assertRaises(AssertionError):
                write_outputs(out, rows, {})
            self.assertFalse(out.exists())


    def test_existing_output_rejected_before_data_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("paper_suite.run_synthetic.make_synthetic") as generator:
                with self.assertRaises(FileExistsError):
                    run(["exp1_epsilon"], ["lasso"], 1, Path(directory))
                generator.assert_not_called()

    def test_real_lasso_uses_full_training_split_for_every_m(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(503, 3))
        frame = pd.DataFrame(x, columns=["a", "b", "c"])
        frame["target"] = x[:, 0] + rng.normal(size=503)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.csv"
            frame.to_csv(path, index=False)
            a = load_tabular_csv(path, "target", 10, 11)
            b = load_tabular_csv(path, "target", 50, 11)
            np.testing.assert_array_equal(a.lasso_x, b.lasso_x)
            np.testing.assert_array_equal(a.lasso_y, b.lasso_y)
            self.assertEqual(a.lasso_x.shape[0], 402)
            self.assertEqual(a.user_x.shape[0], 40)
            self.assertEqual(b.user_x.shape[0], 8)
            np.testing.assert_allclose(fit_lasso(a)[0], fit_lasso(b)[0])
            out = Path(directory)/"real"
            run_real("fixture", path, "target", ["lasso"], 1, out)
            result = pd.read_csv(out/"results.csv")
            self.assertEqual(len(result), 6)
            np.testing.assert_allclose(result.mse_ratio, 1)
            self.assertEqual(result.lasso_test_mse.nunique(), 1)
            self.assertIn("lasso_denominator_convergence_warning", result.columns)
            self.assertIn("lasso_denominator_convergence_warning_count", result.columns)
            self.assertFalse(result.lasso_denominator_convergence_warning.astype(bool).any())

    def test_exp5_rejects_algorithms_without_real_public_bounds(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(1200,4))
        frame=pd.DataFrame(x,columns=list('abcd'))
        frame['target']=x[:,0]-x[:,1]+rng.normal(size=len(x))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.csv'; frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Exp5 supports only'):
                run_real('fixture',path,'target',['proposed'],1,Path(directory)/'bad')

    def test_real_setting_and_ldpiht_metadata_do_not_claim_synthetic_sigma(self):
        from paper_suite.config import RealSetting
        rng=np.random.default_rng(3)
        user_x=rng.normal(size=(12,4,3)); user_y=rng.normal(size=(12,4))
        data=__import__('paper_suite.data',fromlist=['RegressionData']).RegressionData(
            user_x,user_y,rng.normal(size=(20,3)),rng.normal(size=20),None,metadata={'preprocessing':'fixture'})
        setting=RealSetting(n=12,m=4,d=3,k=2,epsilon=4)
        with patch('paper_suite.runner.fit_ldp_iht',return_value=(np.zeros(3),{})):
            _,meta=run_algorithm('ldpiht',data,setting,7)
        self.assertEqual(meta['data_regime'],'real_standardized')
        self.assertIn('Exp5 loader z-scores',meta['scale_adaptation'])
        self.assertFalse(hasattr(setting,'noise_std'))

    def test_real_2slr_adapter_marks_author_realdata_context(self):
        from paper_suite.config import RealSetting
        from paper_suite.data import RegressionData
        rng=np.random.default_rng(8)
        data=RegressionData(rng.normal(size=(100,3,4)),rng.normal(size=(100,3)),
                            rng.normal(size=(20,4)),rng.normal(size=20),None,metadata={'preprocessing':'fixture'})
        setting=RealSetting(n=100,m=3,d=4,k=4,epsilon=4)
        with patch('paper_suite.runner.fit_paper_two_slr',return_value=(np.zeros(4),{'support':np.array([0])})) as fit:
            run_algorithm('2slr',data,setting,7)
        self.assertEqual(fit.call_args.kwargs['data_regime'],'real_standardized')


if __name__ == "__main__":
    unittest.main()
