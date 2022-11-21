from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_array
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from scipy.special import softmax
import numpy as np

from copy import deepcopy

from sklearn.model_selection import train_test_split
from functools import reduce

from scipy import sparse

from .utils import _parse_pipeline_fit_kws, _parse_pipeline_fit_sample_weight, _parse_pipeline_sample_weight_and_kwargs

def _log_odds_ratio_scale(X):
    X = np.clip(X, 1e-8, 1 - 1e-8)   # numerical stability
    X = np.log(X / (1 - X))  # transform to log-odds-ratio space
    return X

class FuzzyTargetClassifier(ClassifierMixin, BaseEstimator):
        
    def __init__(self, regressor):
        '''
        Fits regressor in the log odds ratio space (inverse crossentropy) of target variable.
        during transform, rescales back to probability space with softmax function
        
        Parameters
        ---------
        regressor: Sklearn Regressor
            base regressor to fit log odds ratio space. Any valid sklearn regressor can be used here.
        
        '''
        
        self.regressor = regressor
        return
    
    def fit(self, X, y=None, **kwargs):
        #ensure passed y is onehotencoded-like
        y = check_array(y, accept_sparse=True, dtype = 'numeric', ensure_min_features=1)
        
        if (y.max() > 1) or (y.min() < 0):
            raise ValueError('y contains values out of the range [0,1], please ensure inputs are valid')
        if (y.sum(1) != 1).any():
            raise ValueError("y rows don't sum up to 1, please ensure inputs are valid")
            
        self.regressors_ = [clone(self.regressor) for _ in range(y.shape[1])]
        for i in range(y.shape[1]):
            self._fit_single_regressor(self.regressors_[i], X, y[:,i], **kwargs)
        
        return self
    
    def _fit_single_regressor(self, regressor, X, ysub, **kwargs):
        ysub = _log_odds_ratio_scale(ysub)        
        regressor.fit(X, ysub, **kwargs)
        return regressor    
        
    def decision_function(self,X):
        all_results = []
        for reg in self.regressors_:
            results = reg.predict(X)
            if results.ndim < 2:
                results = results.reshape(-1,1)
            all_results.append(results)
        
        results = np.hstack(all_results)                
        return results
    
    def predict_proba(self, X):
        results = self.decision_function(X)
        results = softmax(results, axis = 1)
        return results
    
    def predict(self, X):
        results = self.decision_function(X)
        results = results.argmax(1)
        return results
    


class ResidualRegressor(BaseEstimator, RegressorMixin):
    #TODO: try to implement cross_val_predcit during fit
    def __init__(
        self,
        regressors,        
        residual_split_fraction = None,
    ):
        
        '''
        fits regressors recursively in its parents residuals
        '''
        self.regressors = regressors
        self.residual_split_fraction = residual_split_fraction
        return
    
    def fit(self,X, y = None, sample_weight = None, **kwargs):
        
        if y.ndim == 1:
            y = y.reshape(-1,1)
        
        self.regressors = [clone(i) for i in self.regressors]
        self.regressors_ = []
        
        estimator = self.regressors[0]
        if self.residual_split_fraction is None:         
            sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(estimator, sample_weight, **kwargs)            
            estimator.fit(X=X, y=y, **{**kws, **sample_weights})
            self.regressors_.append(estimator)
            if len(self.regressors) == 1:
                #end case
                return self                                                
            else:                
                self._fit_recursive(X=X, y=y, i = 1, **kwargs)
        else:
            X, Xres, y, yres = train_test_split(X, y, test_size = self.residual_split_fraction)
            sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(estimator, sample_weight, **kwargs)
            estimator.fit(X=X, y=y, **{**kws, **sample_weights})
            self.regressors_.append(estimator)                       
            if len(self.regressors) == 1:
                #end case
                return self
            else:                                
                self._fit_recursive(X=Xres, y=yres, i = 1, **kwargs)            
                
        return self
    
    def _fit_recursive(self, X, y, i, sample_weight = None, **kwargs):
                        
        estimator = self.regressors[i]

        if self.residual_split_fraction is None:         
            res = y - self._infer(X, 'predict')
            sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(estimator, sample_weight, **kwargs)
            estimator.fit(X=X, y=res, **{**kws, **sample_weights})
            self.regressors_.append(estimator)
            if i+1 >= len(self.regressors):
                #end case
                return self
            else:
                self._fit_recursive(X=X, y=y, i = i+1, **kwargs)
        else:
            X, Xres, y, yres = train_test_split(X, y, test_size = self.residual_split_fraction)
            res = y - self._infer(X, 'predict')
            sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(estimator, sample_weight, **kwargs)
            estimator.fit(X=X, y=res, **{**kws, **sample_weights})
            self.regressors_.append(estimator)            
            if i+1 >= len(self.regressors):
                #end case
                return self
            else:
                res = yres - self._infer(Xres, 'predict')
                self._fit_recursive(X=Xres, y=yres, i = i+1, **kwargs)            
        
        return self
                
    def _infer(self, X, infer_method = 'predict'):        
        predictions = [getattr(i, infer_method)(X) for i in self.regressors_]
        predictions = [i.reshape(-1,1) if i.ndim == 1 else i for i in predictions]
        predictions = reduce(lambda a1,a2: a1+a2, predictions)
        return predictions
    
    def predict(self, X):
        return self._infer(X, 'predict')
            
        

class _CustomFuzzyTargetClassifier(FuzzyTargetClassifier):
    def predict(self, X):
        return self.decision_function(X)

class ResidualClassifier(ResidualRegressor):
    
    def __init__(self, regressors, residual_split_fraction = None):
        '''
    
        '''
        self.regressors = regressors
        self.residual_split_fraction = residual_split_fraction
        return
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        self.regressors = [_CustomFuzzyTargetClassifier(clone(reg)) for reg in self.regressors]        
        super().fit(X = X, y = y, sample_weight = sample_weight, **kwargs)
        return self
    
    def decision_function(self, X):
        return self._infer(X, 'decision_function')
    
    def predict(self, X):        
        return self._infer(X, 'decision_function').argmax(1)
    
    def predict_proba(self, X):
        return self._infer(X, 'predict_proba')
        


class _SingleLabelClassifier(BaseEstimator):
    def __init__(self,):
        """
        a helper estimator to handle cases where there is only one target
        """
        return
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        classes = np.unique(y)
        if len(classes) > 1:
            raise ValueError(f"y should contain only one value, found values: {classes}")
        
        self.classes_ = classes
        return self
    
    def predict(self, X):
        return np.array(X.shape[0]*[self.classes_[0]])
    
    def predict_proba(self, X):
        return np.ones((X.shape[0], 1))
    
    
    
#####################################################
################# Archetypes ########################
#####################################################

        
class ArchetypeEnsembleClassifier(BaseEstimator):
    def __init__(
        self,
        base_embedder,
        final_transformer,
        prefit_embedder = False,
        use_membership_weights = True,
        transform_method = "transform"
    ):
        
        """
        An abstract estimator that applies some transformation
        on data that has a fuzzy membership to a given cluster (or archetype)

        The fit and transform/predict/... processes in each archetype are performed 
        only in the subset of data that has a positive probability of belonging to that
        cluster. Then, the individual weight of each data point is given by the membership score of that
        point. If user defined sample_weight is passed, the final weights during train is the product
        of both membership scores and sample_weight
        """

        self.base_embedder = base_embedder
        self.final_transformer = final_transformer
        self.prefit_embedder = prefit_embedder
        self.use_membership_weights = use_membership_weights
        self.transform_method = transform_method
        return

    def fit(self, X, y = None, sample_weight = None, **kwargs):
        
        if not self.prefit_embedder:
            base_embedder = clone(self.base_embedder)
            sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(base_embedder, sample_weight, **kwargs)
            base_embedder.fit(X, y=y, **{**sample_weights, **kws})
        else:
            base_embedder = deepcopy(self.base_embedder)
            #base_embedder = clone(self.base_embedder)
        self.base_embedder_ = base_embedder
        
        memberships = self.base_embedder_.transform(X)
        
        memberships = normalize(memberships, "l1")
        if not (np.isclose(memberships.sum(axis = 1).A.flatten(), 1, atol=1e-6)).all():
            raise ValueError(f"Some membership rows do not sum up to 1")
        
        n_archetypes = memberships.shape[-1]
        archetype_estimator_list = []
        for i in range(n_archetypes):
            estim = clone(self.final_transformer)
            X_sample, y_sample, weights, mask = self._get_subset_and_weights(
                X=X,
                y=y,
                membership=memberships[:,i].A.flatten(),
                sample_weight = sample_weight,
                use_membership_weights = self.use_membership_weights
            )
            
            #handle cases where partitions have only one class
            if len(np.unique(y_sample)) <= 1:
                estim = _SingleLabelClassifier().fit(X_sample,y_sample)
                
            else:
                if not weights is None:
                    sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(estim, weights, **kwargs)
                    estim.fit(X=X_sample, y=y_sample, **{**kws, **sample_weights})                    
                else:
                    #to ensure will work with estimators that donnot accept sample_weight parameters in fit
                    estim.fit(X=X_sample, y=y_sample)
            
            archetype_estimator_list.append(estim)
        
        #save states
        self.classes_ = np.unique(y)
        self.archetype_estimator_list_ = archetype_estimator_list
        self.n_archetypes_ = n_archetypes
        return self
    
    
    def _get_subset_and_weights(self, X, y, membership, sample_weight, use_membership_weights):
        """
        returns data instances and sample weights for membership > 0
        """
        mask = membership > 0
        
        X_sample = X[mask]
        
        if not y is None:
            y_sample = y[mask]
        else:
            y_sample = None
        

        if sample_weight is None:
            if use_membership_weights:
                weights = membership[mask]
            else:
                weights = None
        else:
            if use_membership_weights:
                weights = sample_weight[mask]*membership[mask]
            else:
                weights = sample_weight[mask]
        
        return X_sample, y_sample, weights, mask
    

    def _infer_reduce(self, infer_method, X, **kwargs):
        
        memberships = self.base_embedder_.transform(X, **kwargs)
        if not (np.isclose(memberships.sum(axis = 1), 1)).all():
            raise ValueError(f"Some membership rows do not sum up to 1")
                        
        #results  = sparse.lil_matrix((X.shape[0], self.n_archetypes_), dtype=np.float32)
        
        results = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(self.n_archetypes_):
            estim = self.archetype_estimator_list_[i]
            class_idx  = np.isin(self.classes_,estim.classes_).nonzero()[0]
            weights = memberships[:,i].A.reshape(-1,1)
            
            res = getattr(estim, infer_method)(X)
            res_placeholder = np.zeros((X.shape[0], len(self.classes_)))
            #handle when binary classifier and returns column matrix

            if res.shape[-1] <= 1:
                class_idx = class_idx[0:1]
            

            res_placeholder[:, class_idx] = res
            
            if not weights is None:
                res_placeholder = res_placeholder*weights
            else:
                pass
            
            
            results += res_placeholder
            #results[mask,i] = res
            
        results = results/memberships.sum(1)#.sum(1)
        return results
    
    def predict_proba(self, X, **kwargs):
        X = self._infer_reduce("predict_proba", X, **kwargs)
        return X
    
    
    def transform(self, X, **kwargs):
        X = self._infer_reduce(self.transform_method, X, **kwargs)
        return X
    
    def predict(self, X):
        X = self.predict_proba(X)
        X = X.argmax(1)
        return X
    

class ArchetypeEnsembleRegressor(BaseEstimator):
    def __init__(
        self,
        base_embedder,
        final_transformer,
        prefit_embedder = False,
        use_membership_weights = True,
        transform_method = "predict"
    ):
        
        """
        An abstract estimator that applies some transformation
        on data that has a fuzzy membership to a given cluster (or archetype)

        The fit and transform/predict/... processes in each archetype are performed 
        only in the subset of data that has a positive probability of belonging to that
        cluster. Then, the individual weight of each data point is given by the membership score of that
        point. If user defined sample_weight is passed, the final weights during train is the product
        of both membership scores and sample_weight
        """

        self.base_embedder = base_embedder
        self.final_transformer = final_transformer
        self.prefit_embedder = prefit_embedder
        self.use_membership_weights = use_membership_weights
        self.transform_method = transform_method
        return

    def fit(self, X, y = None, sample_weight = None, **kwargs):
        
        if not self.prefit_embedder:
            base_embedder = clone(self.base_embedder)
            sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(base_embedder, sample_weight, **kwargs)
            base_embedder.fit(X, y=y, **{**sample_weights, **kws})
        else:
            base_embedder = self.base_embedder
            #base_embedder = clone(self.base_embedder)
        
        memberships = base_embedder.transform(X)
        if not (np.isclose(memberships.sum(axis = 1), 1)).all():
            raise ValueError(f"Some membership rows do not sum up to 1")
        
        n_archetypes = memberships.shape[-1]
        archetype_estimator_list = []
        
        for i in range(n_archetypes):
            estim = clone(self.final_transformer)
            X_sample, y_sample, weights, mask = self._get_subset_and_weights(
                X=X,
                y=y,
                membership=memberships[:,i].A.flatten(),
                sample_weight = sample_weight,
                use_membership_weights = self.use_membership_weights
            )
            
            if not weights is None:                
                weights, kws = _parse_pipeline_sample_weight_and_kwargs(estim, weights, **kwargs)                    
                estim.fit(X=X_sample, y=y_sample, **{**kws, **weights})                
            else:
                #to ensure will work with estimators that donnot accept sample_weight parameters in fit
                estim.fit(X=X_sample, y=y_sample)
            
            archetype_estimator_list.append(estim)
        
        #save states
        self.archetype_estimator_list_ = archetype_estimator_list
        self.base_embedder_ = base_embedder
        self.n_archetypes_ = n_archetypes
        return self
    
    
    def _get_subset_and_weights(self, X, y, membership, sample_weight, use_membership_weights):
        """
        returns data instances and sample weights for membership > 0
        """
        mask = membership > 0
        
        X_sample = X[mask]
        
        if not y is None:
            y_sample = y[mask]
        else:
            y_sample = None
        

        if sample_weight is None:
            if use_membership_weights:
                weights = membership[mask]
            else:
                weights = None
        else:
            if use_membership_weights:
                weights = sample_weight[mask]*membership[mask]
            else:
                weights = sample_weight[mask]
        
        return X_sample, y_sample, weights, mask
    

    def _infer_reduce(self, infer_method, X, **kwargs):
        
        memberships = self.base_embedder_.transform(X, **kwargs)
        if not (np.isclose(memberships.sum(axis = 1), 1)).all():
            raise ValueError(f"Some membership rows do not sum up to 1")
                        
        #results  = sparse.lil_matrix((X.shape[0], self.n_archetypes_), dtype=np.float32)
        
        results = np.zeros((X.shape[0],))
        for i in range(self.n_archetypes_):
            
            estim = self.archetype_estimator_list_[i]
            weights = memberships[:,i].A.reshape(-1,1)
            
            res = getattr(estim, infer_method)(X)
            
            if not weights is None:
                res = res*weights
            else:
                pass
            
            
            results += res
            #results[mask,i] = res
            
        results = results/memberships.sum(1)#.sum(1)
        return results    
    
    def predict(self, X, **kwargs):
        X = self._infer_reduce("predict", X, **kwargs)
        return X
    
    
    def transform(self, X):
        X = self._infer_reduce(self.transform_method, X, **kwargs)
        return X    