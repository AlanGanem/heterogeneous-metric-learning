import warnings

import numpy as np

from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer, normalize
from sklearn.cluster import KMeans

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin, clone
from sklearn.utils._encode import _unique
from sklearn.utils import check_X_y, check_array

from scipy import sparse


#community based heterogeneous space manifold learning
class CBHML(TransformerMixin, BaseEstimator):
    
    def __init__(
        self,
        network_embedder,        
        linear_estimator = None,
        bipartite = True,
        max_archetypes = None,
        max_cumulative_membership = None,
        normalize = True,
        return_sparse = False,
        numerical_features = [],
        categorical_features = [],
        bag_features = [],
        passthrough_features = [],
        numerical_pipeline = None,
        categorical_pipeline = None,
        bag_pipeline = None,
        numerical_n_bins = 10,
        numerical_fuzzy = True,
        numerical_strategy='quantile',
        numerical_handle_nan = 'ignore',
        categorical_nan_value = np.nan,
        categorical_handle_nan = 'ignore',
        categorical_handle_unknown = 'ignore',
        bag_nan_value = np.nan,
        bag_handle_nan = 'ignore',
        bag_handle_unknown = 'ignore',        
        n_jobs = None,
        
    ):
        self.linear_estimator = linear_estimator
        self.network_embedder = network_embedder
        self.max_archetypes = max_archetypes #max number of greater than zero embedding dimensions
        self.max_cumulative_membership = max_cumulative_membership
        self.normalize = normalize
        self.return_sparse = return_sparse
        self.bipartite = bipartite #whether to perform comunity detection in kernelized feature space or in point-feature biaprtite graph
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.bag_features = bag_features
        self.passthrough_features = passthrough_features
        self.numerical_pipeline = numerical_pipeline
        self.categorical_pipeline = categorical_pipeline
        self.bag_pipeline = bag_pipeline
        self.numerical_n_bins = numerical_n_bins
        self.numerical_fuzzy = numerical_fuzzy
        self.numerical_strategy = numerical_strategy
        self.numerical_handle_nan = numerical_handle_nan
        self.categorical_handle_nan = categorical_handle_nan
        self.categorical_handle_unknown = categorical_handle_unknown
        self.categorical_nan_value = categorical_nan_value
        self.bag_handle_nan = bag_handle_nan
        self.bag_handle_unknown = bag_handle_unknown
        self.bag_nan_value = bag_nan_value
        self.n_jobs = n_jobs
        return
                
    def _make_preprocess_pipeline(self,):
        '''
        create preprocessing pipeline for features
        '''
        if self.numerical_pipeline is None:            
            numerical_pipeline = RobustKBinsDiscretizer(
                n_bins = self.numerical_n_bins,
                handle_nan = self.numerical_handle_nan,
                strategy = self.numerical_strategy, 
                encode = 'fuzzy' if self.numerical_fuzzy else 'onehot',
            )
        else:
            numerical_pipeline = self.numerical_pipeline
        
        if self.categorical_pipeline is None:
            categorical_pipeline = RobustOneHotEncoder(
                handle_unknown = self.categorical_handle_unknown,
                handle_nan = self.categorical_handle_nan
            )
        else:
            categorical_pipeline = self.categorical_pipeline
        
        if self.bag_pipeline is None:
            #TODO: define default bag_pipeline
            bag_pipeline = 'drop'#self.bag_pipeline
        else:
            bag_pipeline = self.bag_pipeline
        
        #if no features to passthrough, drop, else apply passthrough
        if self.passthrough_features == []:
            passthrough_pipe = 'drop'            
        else:
            passthrough_pipe = 'passthrough'                        
        
        preprocess_pipeline = ColumnTransformer(
            [
                ('numerical_pipeline',numerical_pipeline, self.numerical_features),
                ('caregorical_pipeline',categorical_pipeline, self.categorical_features),
                ('bag_pipeline',bag_pipeline, self.bag_features),
                ('passthrough_pipeline', passthrough_pipe, self.passthrough_features),
            ],
            n_jobs = self.n_jobs,
            sparse_threshold=1.0
        )
        return preprocess_pipeline
    
    def fit(self, X, y = None, **kwargs):
        '''
        fits linear estimator, sets wieghts and fits graph embedder
        '''
        #parse sample_weight
        if 'sample_weight' in kwargs:
            sample_weight = kwargs['sample_weight']
        else:
            sample_weight = None
            
        self.linear_estimator = clone(self.linear_estimator)
        #parse max_archetypes
        if not self.max_archetypes is None:
            if type(self.max_archetypes) == int:
                if not self.max_archetypes > 0:
                    raise ValueError(f'if int, max archetypes should be greater than 0, got {self.max_archetypes}')
                else:
                    pass
            elif type(self.max_archetypes) == float:                
                if not (self.max_archetypes > 0) and (self.max_archetypes < 1):
                    raise ValueError(
                        f'if float, max archetypes should be in range 0 < max_arcgetypes < 1, got {self.max_archetypes}'                        
                    )
                else:
                    pass
            else:
                raise ValueError(
                        f'max_archetypes should be None, float or int. got {type(self.max_archetypes)}'
                    )
                        
        #handle column attributes
        if len(sum([self.numerical_features, self.categorical_features, self.bag_features, self.passthrough_features],[])) == 0:
            self.passthrough_features = np.arange(X.shape[1]).tolist()
        
        #fit preprocess pipeline
        preprocess_pipeline_ = self._make_preprocess_pipeline().fit(X, y)
        #transform X
        Xt = preprocess_pipeline_.transform(X)
        Xt = sparse.csr_matrix(Xt)
        # fit linear estimator if passed (metric learning)
        if not self.linear_estimator is None:
            self.linear_estimator.fit(X=Xt, y=y, **kwargs)
            #get feature importances
            feature_importances_ = self.linear_estimator.coef_
        else:
            feature_importances_ = np.ones(Xt.shape[1])
        
        if feature_importances_.ndim == 1:
            #regression case
            feature_importances_ = np.abs(feature_importances_)
        else:
            #multiclass case
            feature_importances_ = np.abs(feature_importances_).sum(0)
        
        #scale feature space
        if not sample_weight is None:            
            Xt = Xt.multiply(sample_weight.reshape(-1,1)) #multiply by column matrix of sample_weight
        
        Xt = Xt.multiply(feature_importances_.reshape(1,-1)) #multiply by row matrix of feature weights
        #add a small amount of noise to make sum positive, if needed
        if Xt.sum() == 0:
            Xt+= np.abs(np.random.randn(1)*1e-8)
        #fit graph embedder
        if self.bipartite:
            Xt = sparse.csr_matrix(Xt)
            self.network_embedder.fit(Xt)            
            features_membership_matrix_ = self.network_embedder.membership_col_
            feature_labels_ = self.network_embedder.labels_col_
            
        else:
            Xt = sparse_dot_product(Xt.T, Xt, ntop = Xt.shape[1]) #flexible dot product. if sparse_dot_topn not instaled, perform scipy dot product            
            self.network_embedder.fit(Xt)            
            features_membership_matrix_ = self.network_embedder.membership_
            feature_labels_ = self.network_embedder.labels_
        
        #get topn archetyes
        total_archetypes_ = features_membership_matrix_.shape[-1]
        if not self.max_archetypes is None:
            if type(self.max_archetypes) == float:
                topn_archetypes_ = int(max(1, round(total_archetypes_*self.max_archetypes, 0)))                
            else: #int case
                topn_archetypes_ = min(total_archetypes_, self.max_archetypes)
        else:
            topn_archetypes_ = total_archetypes_
        
        if (topn_archetypes_ == total_archetypes_) and (self.max_cumulative_membership is None):
            subset_archetypes_ = False
        else:
            subset_archetypes_ = True
        
        #save only feature embeddings dims that have at least one value
        features_membership_matrix_ = features_membership_matrix_[:, (features_membership_matrix_.sum(0) > 0).A.flatten()]
        #save state
        self.subset_archetypes_ = subset_archetypes_
        self.topn_archetypes_ = topn_archetypes_
        self.preprocess_pipeline_ = preprocess_pipeline_
        self.features_membership_matrix_ = features_membership_matrix_
        self.feature_importances_ = feature_importances_
        return self
    
    def transform(self, X, return_sparse = None):
        #parse return_sparse argumento
        if return_sparse is None:
            return_sparse = self.return_sparse
        
        Xt = self.preprocess_pipeline_.transform(X)
        Xt = sparse.csr_matrix(Xt)
        Xt = Xt.multiply(self.feature_importances_.reshape(1,-1)) #multiply by row matrix of feature weights
        Xt = sparse_dot_product(Xt, self.features_membership_matrix_, ntop = self.features_membership_matrix_.shape[0]) #TODO: decide whether to normalize (non noramlization yields a "confidence" score, since rows with many NaNs will have lower norm)
        Xt = Xt.A
        if self.subset_archetypes_:                                    
            
            argsort = np.argsort(Xt, axis = 1)
                        
            #TODO: decide how to handle when both max_cumulative_membership and topn_archetypes_ are not None
            if not self.max_cumulative_membership is None:
                #indexes of flattened array
                flat_argsort = (argsort + np.arange(Xt.shape[0]).reshape(-1,1)*Xt.shape[1]).flatten()                
                
                #cumsum of softmax
                cumsum_xt = np.cumsum(
                    softmax(Xt,1).flatten()[flat_argsort].reshape(Xt.shape), # flatten and reshape in order to order array
                    axis = 1) #
                #TODO: why softmax instead of l1 norm?
                #cumsum_xt = cumsum_xt#/cumsum_xt.max(1).reshape(-1,1) #normalize to max value to 1
                zeros_idxs_msk = cumsum_xt < 1 - self.max_cumulative_membership #check columns that sum up to the complemetary of max_cumulative_membership
                flat_zeros_idxs = flat_argsort[zeros_idxs_msk.flatten()]
            else:
                #bottom_n
                zeros_idxs = argsort[:,:-self.topn_archetypes_]
                flat_zeros_idxs = (zeros_idxs + np.arange(Xt.shape[0]).reshape(-1,1)*Xt.shape[1]).flatten()
            
            #replace values using put            
            Xt.put(flat_zeros_idxs, 0)
        
        if self.normalize:            
            Xt = normalize(Xt, norm = 'l1')
            
        if return_sparse:
            Xt = sparse.csr_matrix(Xt)
        else:
            pass
        
        return Xt
    
    def _infer(self, X, inference_type, **kwargs):
        X = self.preprocess_pipeline_.transform(X)
        return getattr(self.linear_estimator, inference_type)(X, **kwargs)
    
    def predict(self, X, **kwargs):
        return self._infer(X, inference_type = 'predict', **kwargs)
    
    def predict_proba(self, X, **kwargs):
        return self._infer(X, inference_type = 'predict_proba', **kwargs)
    