"""Ma24 source subset: Screening only; class body unchanged."""
import numpy as np

class ScreeningSelector(object):
    def __init__(self, 
                 num_features = 2,
                 screening_num_features = None,
                 alpha = None, 
                 max_iter = None, 
                 selection = None,
                 tol = None
                 ):
        '''
        Input:
        ---------
        num_features, the number of features to be selected
        '''
        
        self.num_features = num_features
        
        
    
    def fit(self, X, y):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        self, the fitted object
        '''
        # compute the marginal correlation
        self.m, self.d = X.shape
        self.corr = (y @ X ).ravel()

        # the i th quantile value of corr
        quantile = np.quantile(self.corr, max(1 - self.num_features / self.d, 0))
        self.indexes = np.nonzero(self.corr >= quantile)[0]
        return self
    
    def get_idx(self, X = None, y = None):
        '''
        Input:
        ---------
        X, the local samples
        y, the local labels

        Output:
        ---------
        idx, the selected index
        '''

        if not hasattr(self, 'indexes'):
            assert X is not None and y is not None, "Please fit the model or input X and y"
            self.fit(X, y)
        return np.random.choice(self.indexes, 1)[0]
