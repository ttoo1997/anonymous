import unittest
from unittest.mock import patch
import numpy as np
from paper_suite.full_experiment import plan
from paper_suite.data import RegressionData
from paper_suite.config import SyntheticSetting
from paper_suite.runner import evaluate


class CompletePdfContract(unittest.TestCase):
    def test_finished_status_requires_every_expected_result(self):
        import tempfile,json
        from pathlib import Path
        from paper_suite.full_experiment import job_is_complete
        job=dict(job_id=0,kind='synthetic',experiment='exp4a_recall_m',repeat=0,
                 setting=dict(n=800,m=25,d=256,k=8,epsilon=4,delta=1e-6,noise_std=.25))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);d=p/'jobs/0000';d.mkdir(parents=True)
            (d/'status.json').write_text(json.dumps({'state':'finished'}))
            self.assertFalse(job_is_complete(p,job))
            for name in ['proposed','2slr']:
                row=dict(experiment=job['experiment'],repeat=0,seed=20260915,algorithm=name,
                         status='ok',support_recall=1.,**job['setting'])
                (d/f'{name}.row.json').write_text(json.dumps(row))
            self.assertTrue(job_is_complete(p,job))
            row['epsilon']=1
            (d/'2slr.row.json').write_text(json.dumps(row))
            self.assertFalse(job_is_complete(p,job))

    def test_sample_cache_rejects_fractional_rounded_statistics(self):
        from paper_suite.algorithms.sample_sdp import fit_sample_sdp
        prepared={'meta':{'n':20,'d':2,'D':5},'rounded':[.5]*5}
        with self.assertRaisesRegex(ValueError,'signed integer'):
            fit_sample_sdp(np.zeros((10,2,2)),np.zeros((10,2)),4,1,prepared)

    def test_all_pdf_panels_and_repeats(self):
        jobs=plan(10,'pdf_v1')
        self.assertEqual(len(jobs),270)  # supported synthetic runs; optional datasets require explicit registration
        count=sum(6 if j['kind']=='real' else 2 if j['experiment'].startswith('exp4') else 5 for j in jobs)
        self.assertEqual(count,1050)
        for name,num in [('exp1_epsilon',70),('exp2_dimension',40),('exp3_users',60),('exp4a_recall_m',40),('exp4b_recall_n',60)]:
            self.assertEqual(sum(j.get('experiment')==name for j in jobs),num)
        for j in jobs:
            if j['kind']=='synthetic' and j['experiment']!='exp1_epsilon':
                self.assertEqual(j['setting']['epsilon'],4)

    def test_cross_panel_anchor_and_user_grid_are_frozen(self):
        from paper_suite.config import synthetic_grid,USER_GRID
        self.assertEqual(USER_GRID,(8000,10000,12000,15000,18000,20000))
        anchors={
            'exp1_epsilon': next(s for s in synthetic_grid('exp1_epsilon') if s.epsilon==4),
            'exp2_dimension': next(s for s in synthetic_grid('exp2_dimension') if s.d==256),
            'exp3_users': next(s for s in synthetic_grid('exp3_users') if s.n==15000),
            'exp4a_recall_m': next(s for s in synthetic_grid('exp4a_recall_m') if s.m==50),
            'exp4b_recall_n': next(s for s in synthetic_grid('exp4b_recall_n') if s.n==15000),
        }
        tuples={(s.n,s.m,s.d,s.k,s.epsilon,s.delta,s.noise_std) for s in anchors.values()}
        self.assertEqual(tuples,{(15000,50,256,8,4.0,1e-5,.25)})

    def test_stage1_recall_does_not_depend_on_final_zero_coefficients(self):
        setting=SyntheticSetting(n=1,m=1,d=3,k=1)
        data=RegressionData(np.zeros((1,1,3)),np.zeros((1,1)),np.zeros((1,3)),np.zeros(1),np.array([.2,0,0]))
        with patch('paper_suite.runner.run_algorithm',return_value=(np.zeros(3),{'support':np.array([0,2])})):
            row=evaluate('proposed',data,setting,1)
        self.assertEqual(row['support_recall'],1)
        self.assertEqual(row['stage1_support_recall'],1)
        self.assertEqual(row['final_support_recall'],0)
        self.assertEqual(row['support_precision'],.5)
        self.assertAlmostEqual(row['support_f1'],2/3)
        self.assertEqual(row['final_support_precision'],0)
        self.assertEqual(row['final_support_f1'],0)

    def test_resume_rejects_wrong_or_nonfinite_saved_cell(self):
        from paper_suite.full_experiment import validate_saved_row
        base={'experiment':'exp1_epsilon','algorithm':'proposed','epsilon':4}
        row={**base,'status':'ok','test_mse':1.,'relative_coefficient_error':.2}
        validate_saved_row(row,base)
        for bad in [dict(row,epsilon=1),dict(row,status='failed'),dict(row,test_mse=float('nan'))]:
            with self.assertRaises(ValueError):validate_saved_row(bad,base)

    def test_recovery_checks_data_checksum(self):
        from paper_suite.full_experiment import validate_resume
        frozen={'jobs':[],'source_checksums':{},'data_checksums':{'fixture':'expected'}}
        validate_resume(frozen,dict(frozen))
        with self.assertRaisesRegex(ValueError,'resume inputs differ'):
            validate_resume(frozen,dict(frozen,data_checksums={'fixture':'changed'}))
        with self.assertRaisesRegex(ValueError,'resume source code differs'):
            validate_resume(frozen,dict(frozen,source_checksums={'code':'changed'}))

    def test_row_count_alone_cannot_mark_run_complete(self):
        import tempfile,json
        from pathlib import Path
        from paper_suite.full_experiment import collect
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);d=p/'jobs/0000';d.mkdir(parents=True)

            for i,name in enumerate(['proposed','2slr']):
                (d/f'{i}.row.json').write_text(json.dumps(dict(experiment='exp4a_recall_m',algorithm=name,repeat=0,n=800,m=25,d=256,k=8,epsilon=4,status='ok',support_recall=1.)))
            (d/'status.json').write_text(json.dumps({'state':'running'}))
            self.assertFalse(collect(p,[{'kind':'synthetic','experiment':'exp4a_recall_m'}])['all_requested_cells_ok'])

    def test_exp4_never_calls_estimation(self):
        setting=SyntheticSetting(n=1,m=1,d=3,k=1)
        data=RegressionData(np.zeros((1,1,3)),np.zeros((1,1)),np.zeros((1,3)),np.zeros(1),np.array([.2,0,0]))
        with patch('paper_suite.runner.run_recall',return_value=(np.array([0]),{})),patch('paper_suite.runner.run_algorithm',side_effect=AssertionError('stage2 called')):
            row=evaluate('2slr',data,setting,1,True)
        self.assertEqual(row['stage1_support_recall'],1)

    def test_iht_streaming_aggregation_matches_materialized_reference(self):
        from paper_suite.algorithms.ldp_iht import fit_ldp_iht
        from paper_suite.algorithms.common import project_l2,hard_threshold,privunit
        rng=np.random.default_rng(42);x=rng.normal(size=(24,5));y=rng.normal(size=24)
        stream=np.random.default_rng(77);groups=np.array_split(stream.permutation(24),2)
        theta=np.zeros(5);radius=np.sqrt(5)*4*(np.sqrt(5)*4+4)
        xc=np.clip(x,-4,4);yc=np.clip(y,-4,4)
        for group in groups:
            reports=[privunit(project_l2(xc[i]*(xc[i]@theta-yc[i]),radius),radius,4,stream) for i in group]
            theta=project_l2(hard_threshold(theta-.1*np.mean(reports,axis=0),5),1)
        actual,meta=fit_ldp_iht(x,y,4,1,77,iterations=2)
        np.testing.assert_allclose(actual,theta,atol=1e-14,rtol=1e-14)
        self.assertEqual(len(meta['iteration_diagnostics']),2)

    def test_exp5_collect_uses_six_cells_per_real_job(self):
        import tempfile, json
        from pathlib import Path
        from paper_suite.full_experiment import collect
        from paper_suite.config import REAL_FORMAL_ALGORITHMS, REAL_EPSILON_GRID
        job={'job_id':0,'kind':'real','dataset':'fixture','m':10,'repeat':0}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); d=root/'jobs'/'0000'; d.mkdir(parents=True)
            for algorithm in REAL_FORMAL_ALGORITHMS:
                for epsilon in REAL_EPSILON_GRID:
                    row=dict(experiment='exp5_real',dataset='fixture',algorithm=algorithm,
                             repeat=0,seed=20260915,n=100,m=10,d=4,k=4,delta=1e-6,
                             epsilon=epsilon,status='ok',test_mse=1.0,mse_ratio=1.0)
                    (d/f'{algorithm}_{epsilon}.row.json').write_text(json.dumps(row))
            (d/'status.json').write_text(json.dumps({'state':'finished'}))
            result=collect(root,[job])
            self.assertEqual(result['expected_cells'],6)
            self.assertEqual(result['completed_cells'],6)
            self.assertTrue(result['all_requested_cells_ok'])


if __name__=='__main__':unittest.main()
