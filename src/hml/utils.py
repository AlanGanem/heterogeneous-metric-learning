
# Cell
from warnings import warn
from inspect import getmembers, isfunction
import inspect

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

try:
    from sparse_dot_topn import awesome_cossim_topn
except Exception as e:
    warn(f'could not load sparse_dot_topn: {e}')


# Cell
#util funcs and classes


def get_default_args(func):
    '''THANKS TO mgilson at https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value'''
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def inherit_docstrings(cls):
    '''
    thanks to Martijn Pieters♦ at https://stackoverflow.com/questions/17393176/python-3-method-docstring-inheritance-without-breaking-decorators-or-violating
    '''
    for name, func in getmembers(cls, isfunction):
        if func.__doc__: continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__doc__ = getattr(parent, name).__doc__
    return cls

# Cell
#TODO: implement minkowski distance with sparse_dot_topn
#TODO: implement RBF distance

#export
def make_batches(arr, batch_size = 100):
    '''make batches for batch query'''
    #lst = [i for i in arr]

    if arr.shape[0] < batch_size:
        batches = [arr]
    else:
        n_bs = arr.shape[0] // batch_size
        last_batch = arr.shape[0] - batch_size * n_bs
        batches = []
        i = 0
        for i in range(n_bs):
            yield arr[i * batch_size:(i + 1) * batch_size]

        if last_batch:
            yield arr[(i + 1) * batch_size:]

def sim_matrix_to_idx_and_score(sim_matrix):
    '''
    returns list of indexes (col index of row vector) and scores (similarity value) for each row, given a similarity matrix
    '''
    scores = []
    idxs = []
    for row in sim_matrix:
        idxs.append(row.nonzero()[-1])
        scores.append(row.data)

    return idxs, scores

def cosine_similarity(A, B, topn = 30, remove_diagonal = False, **kwargs):

    A,B = sparsify(A,B)
    A = normalize(A, norm  = 'l2').astype(np.float64)
    B = normalize(B, norm  = 'l2').astype(np.float64)
    dot = awesome_cossim_topn(A, B.T, ntop = topn, **kwargs)

    if remove_diagonal:
        dot.setdiag(0)
        dot.eliminate_zeros()

    return dot


def cosine_distance(A, B, topn = 30, remove_diagonal = False, **kwargs):

    #calculate sim
    dist = cosine_similarity(A, B, topn, remove_diagonal, **kwargs)
    #calculate distance
    dist.data = 1 - dist.data
    return dist

# Cell

def sparse_dot_product(
    A,
    B,
    ntop = None,
    lower_bound=0,
    use_threads=False,
    n_jobs=1,
    return_best_ntop=False,
    test_nnz_max=-1,
):

    '''
    flexible dot product function to work with or without sparse_dot_topn. In the absence of sparse_dot_topn, naive numpy dot product will be performed

    sparse_dot_topn.awesome_cossim_topn Docs:

    This function will return a matrix C in CSR format, where
    C = [sorted top n results > lower_bound for each row of A * B].
    If return_best_ntop=True then best_ntop
    (the true maximum number of elements > lower_bound per row of A * B)
    will also be returned in a tuple together with C as (C, best_ntop).

    Input:
        A and B: two CSR matrices
        ntop: top n results
        lower_bound: a threshold that the element of A*B must be greater than
        use_threads: use multi-thread or not
        n_jobs: number of thread, must be >= 1
        return_best_ntop: (default: False) if True, will return best_ntop together
                          with C as a tuple: (C, best_ntop)

    Output:
        C: result matrix (returned alone, if return_best_ntop=False)
        best_ntop: The true maximum number of elements > lower_bound per row of
                   A * B returned together with C as a tuple: (C, best_ntop). It is
                   returned only if return_best_ntop=True.

    N.B. if A and B are not in CSR format, they will be converted to CSR
    '''

    MAX_BYTES = 100e6 #process dense arrays of maximum 100MB for dense numpy dot product
    if n_jobs is None:
        n_jobs = 1
        
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    
    if not sparse.issparse(B):
        
        B = sparse.csr_matrix(B)

    if 'awesome_cossim_topn' in globals():
        if ntop is None:
            ntop = B.shape[-1]            
        
        B = B.astype(np.float32)
        A = A.astype(np.float32)
        
        dot = awesome_cossim_topn(
            A = A,
            B = B,
            ntop = ntop,
            lower_bound=lower_bound,
            use_threads=use_threads,
            n_jobs=n_jobs,
            return_best_ntop=return_best_ntop,
            test_nnz_max=test_nnz_max,
        )
    else:
        warn('sparse_dot_topn is not installed, this may cause performance issues in dot product calculations')
        dot = A@B

    return dot

# Cell
def similarity_plot(vector, query_matrix):
    '''
    plots similarity plots like in https://gdmarmerola.github.io/forest-embeddings/
    '''
    return


# Cell
def sparsify(*arrs):
    '''
    makes input arrs sparse
    '''
    arrs = list(arrs)
    for i in range(len(arrs)):
        if not sparse.issparse(arrs[i]):
            arrs[i] = sparse.csr_matrix(arrs[i])

    return arrs

def _robust_stack(blocks, stack_method = 'stack', **kwargs):

    if any(sparse.issparse(i) for i in blocks):
        #handle sparse
        stacked = getattr(sparse, stack_method)(blocks, **kwargs)

    else:
        #handle pandas
        if all(hasattr(i, 'iloc') for i in blocks):
            if stack_method == 'hstack':
                stacked = pd.concat(blocks, axis = 1)
            else:
                stacked = pd.concat(blocks, axis = 0)

        else:
            #handle  numpy
            stacked = getattr(np, stack_method)(blocks, **kwargs)

    return stacked

def hstack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'hstack', **kwargs)

def vstack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'vstack', **kwargs)

def stack(blocks, **kwargs):
    return _robust_stack(blocks, stack_method = 'stack', **kwargs)


#prefir estimator
class PrefitEstimator(BaseEstimator):
    
    def __init__(self, prefit_estimator):
        self.prefit_estimator = prefit_estimator
        self.is_fitted_ = True
        return
    
    def __getattr__(self, attr):
        '''
        gets the attributes from prefit_estimator, except if the attribute (or method)
        is "fit".
        
        if the "transform" or "predict" method is called, it'll return self.prefit_estimator's method
        '''
        if attr == 'fit':
            return self.fit        
        elif attr == 'fit_transform':
            return self.fit_transform
        elif attr == 'fit_predict':
            return self.fit_predict            
        else:
            return getattr(self.prefit_estimator, attr)
    
    def fit(self, X, y = None, **kwargs):
        '''
        the fit method does nothing (since prefit_estimator is already fitted) and returns self.
        '''
        return self    
    
    def fit_transform(self, X, y = None, **kwargs):
        return self.transform(X) #will get "transform" method from self.prefit_estimator
    
    def fit_predict(self, X, y = None, **kwargs):
        return self.predict(X) #will get "predict" method from self.prefit_estimator
    

## Parse Pipeline kwargs in fit
def _parse_pipeline_fit_kws(pipeline, **kwargs):        
    kws = {}                
    for param_name in kwargs:
        for estim_name in pipeline.named_steps:
            signature = inspect.signature(pipeline.named_steps[estim_name].fit).parameters
            _, _, varkw, _ = inspect.getargspec(pipeline.named_steps[estim_name].fit)

            if (param_name in signature) or (not varkw is None):
                kws[f"{estim_name}__{param_name}"] = kwargs[param_name]        
    return kws
    
def _parse_pipeline_fit_sample_weight(pipeline, sample_weight):
    weight_estim_names = [k for k,v in pipeline.named_steps.items() if "sample_weight" in inspect.signature(v.fit).parameters]
    sample_weights = {f"{name}__sample_weight":sample_weight for name in weight_estim_names}                
    return sample_weights


from sklearn.pipeline import Pipeline

def _parse_pipeline_sample_weight_and_kwargs(estimator, sample_weight, **kwargs):
    """
    prases sample wieghts and kwargs. if estimator is a sklearn pipeline for all estimators in the pipeline that accepts the kwargs,
    this parser will return a dictionary containing args in the format <estimator_name>__<arg_name>, as it is accepted in sklearn pipeline fit API
    
    if estimator is not a pipeline, will return a dictionary resembling the original parametr names and their values
    
    Returns
    -------
    
    sample_weights dict, kws dict
    """
    if isinstance(estimator, Pipeline):
        kws = _parse_pipeline_fit_kws(estimator, **kwargs)
        sample_weights = _parse_pipeline_fit_sample_weight(estimator, sample_weight)
        return sample_weights, kws
    
    else:
        return {"sample_weight": sample_weight}, kwargs