import sys,unittest,tempfile,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exp5 import common
import numpy as np
from exp5.privacy import budgets
from exp5.sample_fast import prepare
from exp5.algorithms import fit_u2r,fit_two
from lasso_threshold_v1.mechanism import proposed as frozen_u2r
from paper_suite.algorithms import two_slr_paper as two
from threadpoolctl import threadpool_limits

class Check(unittest.TestCase):
    def test_lasso_full_objective_equivalence(self):
        from exp5.lasso_gram import fit as accelerated
        from sklearn.linear_model import LassoCV
        from sklearn.model_selection import GroupKFold
        rng=np.random.default_rng(82);X=rng.normal(size=(1600,20));days=np.repeat(np.arange(40),40)
        X[:,5]=X[:,2]*.8+rng.normal(size=1600)*.2
        y=.2*X[:,0]-.12*X[:,5]+rng.normal(size=1600)*.1
        a,meta=accelerated(X,y,days)
        ref=LassoCV(alphas=300,cv=GroupKFold(5).split(X,y,days),fit_intercept=False,max_iter=8000,tol=1e-4).fit(X,y)
        self.assertAlmostEqual(meta['alpha'],ref.alpha_,places=12)
        np.testing.assert_allclose(a,ref.coef_,rtol=1e-7,atol=1e-8)

    def test_group_privacy(self):
        for m in (10,25,50,75,100):
            for e in (1.,4.):
                b=budgets('sample_sdp',e,1e-6,m,'user')
                factor=sum(math.exp(j*b['record_epsilon']) for j in range(m))
                self.assertAlmostEqual(m*b['record_epsilon'],e)
                self.assertAlmostEqual(factor*b['record_delta'],1e-6,places=17)
                self.assertEqual(budgets('ldpiht',e,1e-6,m,'user')['record_delta'],0.)

    def test_u2r_adapter_same_seed(self):
        rng=np.random.default_rng(7);X=rng.normal(size=(400,10,6));y=X[:,:,0]*.2+rng.normal(size=(400,10))*.05
        with tempfile.TemporaryDirectory() as cache:
            a,ma=fit_u2r(X.reshape(-1,6),y.ravel(),10,4.,dict(k=1,rho=.125,norm=1.),71,cache,'test')
            b,mb=frozen_u2r.fit(X,y,4.,1,71,rho=.125,parameter_l2_norm=1.)
            np.testing.assert_allclose(a,b,rtol=0.,atol=1e-13)
            self.assertEqual(ma['support'].tolist(),list(mb['support']))

    def test_twoslr_adapter_same_seed(self):
        rng=np.random.default_rng(8);X=rng.normal(size=(100,10,6));y=X[:,:,0]*.2+rng.normal(size=(100,10))*.05
        p=dict(local_selected=2,k=2,B=1.,num_bins=4)
        with tempfile.TemporaryDirectory() as cache:
            a,_=fit_two(X.reshape(-1,6),y.ravel(),10,4.,p,72,cache,'test')
            b,_=two.fit(X,y,4.,2,72,data_regime='real_standardized')
            np.testing.assert_allclose(a,b,rtol=0.,atol=1e-13)

    def test_grouped_rounding_distribution(self):
        rng=np.random.default_rng(9);numeric=rng.normal(size=(24,2));bits=np.eye(3)[rng.integers(0,3,24)]
        low=np.array([-.2,-.3,-.4]);high=low+1
        X=np.column_stack([numeric,np.where(bits,high,low)]);y=rng.normal(size=24)
        upper=np.triu_indices(5)
        f=np.column_stack([X[:,upper[0]]*X[:,upper[1]],X*y[:,None]])
        for clip in (.7,100.):
            z=3*f*np.minimum(1,clip/np.linalg.norm(f,axis=1))[:,None]
            expected=z.sum(0);frac=z-np.floor(z);var=(frac*(1-frac)).sum(0)
            samples=np.array([prepare(X,y,clip,3,np.random.default_rng(s),2,low,high,chunk=7)['rounded'] for s in range(400)])

            error=abs(samples.mean(0)-expected)
            self.assertTrue(np.all(error<5*np.sqrt(var/400)+.02))
            self.assertTrue(np.all(abs(samples.var(0)-var)<.5*var+.1))

    def test_grouped_schema_fail_closed(self):
        X=np.ones((25,3));X[:,1]=.33
        with self.assertRaises(ValueError):prepare(X,np.ones(25),2,1,np.random.default_rng(1),1,np.zeros(2),np.ones(2))

if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
