import inspect
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, normalize, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble._forest import BaseForest

from joblib import effective_n_jobs, Parallel, delayed

try:
    from sknetwork.clustering import KMeans as GraphKMeans
    from sknetwork.clustering import PropagationClustering, Louvain
except Exception as e:
    warnings.warn(f"Failed do load sknetwork, some classes that depend upon it will raise errors: {e}")
    
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor, LGBMModel

from scipy import sparse
from scipy.special import softmax

import networkx as nx

from umap import UMAP

from .utils import hstack, sparse_dot_product
from .utils import _parse_pipeline_fit_kws, _parse_pipeline_fit_sample_weight, _parse_pipeline_sample_weight_and_kwargs

#lgbm custom estimators


class LGBMTransformer():
    """
    Same as LGBMClassifier, but contaions the apply method and leaf weights are leaerned as well
    """
    def __init__(self, lgbm_estimator, leaf_weight_strategy = "cumulative_unit_gain", prefit_estimator = False):
        """
        A transformer that returns the weighted leaf space of a sample
        """
        self.lgbm_estimator = lgbm_estimator
        self.leaf_weight_strategy = leaf_weight_strategy
        self.prefit_estimator = prefit_estimator
    
    def __getattr__(self,attr):
        return getattr(self.lgbm_estimator, attr)
    
    def apply(self, X):
        
        """
        maps from input space to leaf node index for each of the trees in the ensemble
        """
        return self.predict(X, pred_leaf = True)
    
    def transform(self, X):
        """
        maps from input space to weighted leaf space
        """
        X = self.apply(X)
        X = self.encoder_.transform(X)
        if hasattr(self, "leaf_weights_"):
            X = X.multiply(self.leaf_weights_)
        return sparse.csr_matrix(X)
    
    def decision_path(self, X, decision_weight = None):
        leafs = self.apply(X)
        return self._transform_decision_path(leafs)
    
    def fit(self, X, y = None, sample_weight = None, prefit_ensemble = False, **kwargs):
        
        """
        fits estimators and learns leaf weights according to leaf_weigjts_strategy.
        can be one of ["cumulative_gain","cumulative_unit_gain", "count", None]
        
        Parameters
        ----------
        
        X: Array like
            Features to train model
            
        y: array like
            Dependent variable
        
        sample_weight: Array like of shape (n_samples,)
            Weights of each sample 
        
        leaf_weight_strategy: str or None
            How to determine leaf weights. Can be one of ["cumulative_gain","cumulative_unit_gain", "count", None].
            
            cumulative_gain:
                calculates the sum of all information gains for all the splits up to the leaf node
            
            cumulative_unit_gain:
                same as cumulative gain, but divides the gain of each split by the number of points in the node of the split.
            
            count:
                simply the sum of the ammount of points in all parent nodes (recursively) from leaf up to the root node
            
            None:
                no weights are calculated, CustomLGBMClassifier will not have the leaf_weights_ attribute
        
        kwargs:
            keyword arguments passed to LGBMClassifier.fit method
            
        """        
        
        if not self.prefit_estimator:            
            self.lgbm_estimator = clone(self.lgbm_estimator)
            sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(self.lgbm_estimator, sample_weight, **kwargs)            
            self.lgbm_estimator.fit(X=X, y=y, **{**kws, **sample_weights})            
        else:
            self.lgbm_estimator = deepcopy(self.lgbm_estimator)
        
        self._set_leaf_weights(self.leaf_weight_strategy)
        self.encoder_ = OneHotEncoder().fit(self.apply(X))
        return self
    
    
    def _set_leaf_weights(self, strategy):
        """
        sets leaf_weights_ attribute according to strategy
        """
        
        VALID_STRATEGIES = ["cumulative_gain","cumulative_unit_gain", "count", None]
        if not strategy in VALID_STRATEGIES:
            raise ValueError(f"strategy should be one of {VALID_STRATEGIES}, got {strategy}")
        
        if strategy is None:
            #do not assign leaf_weights_ attribute
            return
        else:    
            model_df = self.booster_.trees_to_dataframe()

            d_cols = ["split_gain","parent_index","node_index","count"]
            parent_split_gain = pd.merge(model_df[d_cols], model_df[d_cols], left_on = "parent_index", right_on = "node_index", how = "left")

            root_nodes = model_df[model_df["parent_index"].isna()][["tree_index","node_index"]].rename(columns={"node_index":"root_node"})
            model_df = model_df.merge(root_nodes, how = "left", on = "tree_index")

            model_df["parent_gain"] = parent_split_gain["split_gain_y"]
            model_df["parent_count"] = parent_split_gain["count_y"]
            model_df["parent_fraction"] = model_df["count"]/model_df["parent_count"]            
            
            model_df["cumulative_gain"] = model_df["parent_gain"]#*model_df["parent_fraction"]
            model_df["cumulative_gain"] = model_df["cumulative_gain"].fillna(0)
            
            model_df["cumulative_unit_gain"] = model_df["cumulative_gain"]/model_df["count"]
            model_df["cumulative_unit_gain"] = model_df["cumulative_unit_gain"].fillna(0)

            model_df["inverse_count"] = 1/model_df["count"]
            model_df["unit_weight"] = model_df["weight"]/model_df["count"]

            model_df["leaf_index"] = model_df["node_index"].str.split("-").str[1].str[1:].astype(int)
            model_df["leaf_index"] = np.where(model_df["right_child"].isna(), model_df["leaf_index"], None)

            #path_indpr = np.hstack([[0], model_df.reset_index().groupby("tree_index")["index"].max().values])
            leaf_indexes = model_df.dropna(subset = ["leaf_index"]).sort_values(by = ["tree_index", "leaf_index"]).index.values
            leaf_index_map = model_df.dropna(subset = ["leaf_index"]).sort_values(by = ["tree_index", "leaf_index"])["node_index"]
            path_pairs_df = model_df.dropna(subset = ["leaf_index"])[["root_node","node_index"]]



            g = nx.Graph()
            g.add_nodes_from(model_df["node_index"])

            z = list(tuple(i) for i in tuple(model_df[["parent_index","node_index"]].dropna().values))
            g.add_edges_from(z)

            #strategy = "unit_parent_gain"
            path_pairs = list(tuple(i) for i in tuple(path_pairs_df.values))
            paths = np.array([np.array(nx.shortest_path(g, *i)) for i in path_pairs])

            weights_df = model_df.set_index("node_index")[[strategy, "tree_index", "node_depth"]]
            weights_df["network_weight"] = np.nan
            for i in np.arange(len(path_pairs_df["node_index"])):
                weights_df.loc[path_pairs_df.loc[path_pairs_df.index[i],"node_index"], "network_weight"] = weights_df.loc[paths[i], strategy].sum()
        
            leaf_weights = weights_df.loc[leaf_index_map, "network_weight"].values.reshape(1,-1)
            self.leaf_weights_ = leaf_weights
        return

    
    

##############################################
### using leafs as indexes for a NN search ###
##############################################


class ForestNeighbors(BaseEstimator):
    
    def __init__(
        self,
        ensemble_estimator,
        n_neighbors = 30,
        radius = .5,
        n_jobs = 1,
        prefit_fensemble_estimator = False,
    ):
        '''
        Kneighbors search based on terminal node co-ocurrence.
        Trees can be grown adaptatively (supervised tree ensemble) or randomly (unsupervised tree ensemble)        
        '''
        self.ensemble_estimator = ensemble_estimator
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_jobs = n_jobs
        self.prefit_ensemble_estimator = prefit_ensemble_estimator
        return
    
    def fit(self, X, y = None, **kwargs):
        '''
        fits ensemble_estimator if prefit_ensemble_estimator is set to False. Then, fits neighbors search index
        '''
        
        if not self.prefit_ensemble_estimator:
            self.ensemble_estimator.fit(X = X, y = y, **kwargs)
        else:
            #check if fitted, then pass
            check_is_fitted(self.ensemble_estimator)
            
                        
        #get terminal node indexes
        node_indexes = self.ensemble_estimator.apply(X)        
        #create node encoding pipeline
        index_encoder = make_pipeline(OrdinalEncoder(), OneHotEncoder())
        index_encoder.fit(node_indexes)
        #create node->point mapper
        point_node_mapping = self._create_node_point_mapper(node_indexes, index_encoder)        
        #correct node indexes
        node_indexes = index_encoder[0].transform(node_indexes)
        max_raw_node_values = node_indexes.max(0)
        sum_indexes = np.roll((max_raw_node_values + 1).reshape(1,-1).cumsum(), 1, )
        sum_indexes[0] = 0
        sum_indexes = sum_indexes.astype(int)    
        node_indexes = (node_indexes + sum_indexes).astype(int)        
        #save states        
        self.max_raw_node_values_ = max_raw_node_values
        self.terminal_nodes_ = node_indexes
        self.point_node_mapping_ = point_node_mapping
        self.index_encoder_ = index_encoder
        return self
    
    def sample(self, X=None, size = 30, n_jobs = None):                       
        '''
        Samples point indexes in terminal nodes for a given input value.
        if size is None, will return all the indexes for all the terminal nodes for that point.
        '''        
                
        if n_jobs is None:
            n_jobs = self.n_jobs
        
        #get effective n_jobs
        n_jobs = effective_n_jobs(n_jobs)                
                
        if X is None:
            node_indexes = self.terminal_nodes_
        
        else:
            #get terminal node indexes
            node_indexes = self.get_indexes(X)                            
        
        if n_jobs > 1:            
            results = Parallel(n_jobs=n_jobs)(
                delayed(_sample)(
                    self.point_node_mapping_[i],
                    size = size
                ) for i in np.array_split(node_indexes, n_jobs)
            )
            
            
            #concatenate parallel results
            results = sum([i for i in results], [])
        
        else:
            results = _sample(
                self.point_node_mapping_[node_indexes],   
                size = size
            )                
        
        return results
    
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, sample_size = None, n_jobs = None):                       
        '''
        Retreives the kneighbors of each point in X. The neighbors points are defined as co-ocurence in terminal leafs of a ensemble of trees
        '''        
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_jobs is None:
            n_jobs = self.n_jobs
        
        #get effective n_jobs
        n_jobs = effective_n_jobs(n_jobs)                
                
        if X is None:
            node_indexes = self.terminal_nodes_
        
        else:
            #get terminal node indexes
            node_indexes = self.get_indexes(X)                            
        
        if n_jobs > 1:            
            results = Parallel(n_jobs=n_jobs)(
                delayed(_postprocess_node_points)(
                    self.point_node_mapping_[i],                    
                    n_neighbors = n_neighbors,
                    sample_size = sample_size
                ) for i in np.array_split(node_indexes, n_jobs)
            )
            
            
            #concatenate parallel results
            results = sum([i[0] for i in results], []), sum([i[1] for i in results], [])
        
        else:
            results = _postprocess_node_points(
                self.point_node_mapping_[node_indexes],                
                n_neighbors = n_neighbors,
                sample_size = sample_size
            )
        
        if not return_distance:
            results = results[1]
        
        return results                                   
    
    def get_indexes(self, X):        
        indexes = self.ensemble_estimator.apply(X)
        indexes = self._correct_node_indexes(indexes, self.index_encoder_)
        return indexes
    
    def _create_node_point_mapper(self, node_indexes, index_encoder):
        '''
        creates terminal_node->points mapper in order to retrieve points based on terminal nodes
        '''
        point_node_mapping = index_encoder.transform(node_indexes)
        M = point_node_mapping.T.tocsr()
        mapper = np.array(np.split(M.indices, M.indptr)[1:-1])
        return mapper

    def _correct_node_indexes(self, node_indexes, index_encoder):    
        '''
        correct node indexes to reflect actual positional indexing of nodes in the forest
        '''

        node_indexes = index_encoder[0].transform(node_indexes)    
        sum_indexes = np.roll((self.max_raw_node_values_ + 1).reshape(1,-1).cumsum(), 1, )
        sum_indexes[0] = 0
        sum_indexes = sum_indexes.astype(int)    
        corrected_indexes = (node_indexes + sum_indexes).astype(int)
        return corrected_indexes
    

def _sample(points, size = None):
    '''
    sample points from terminal nodes.
    not a method in class to avoid issues with parallelism
    '''        
    if size is None:
        samples = [np.concatenate(points[i]) for i in range(len(points))]    
    else:
        samples = [np.random.choice(np.concatenate(points[i]), size = size) for i in range(len(points))]
    return samples
    
def _postprocess_node_points(points, n_neighbors, sample_size):
    '''
    postprocess queried points.
    not a method in class to avoid issues with parallelism
    '''    
    dist = []
    idx = []
    n_terminal_nodes = points.shape[1]
    #points = mapper[corrected_indexes]
    if sample_size is None:
        for i in range(len(points)):        
            idx_, count_ = np.unique(np.concatenate(points[i]), return_counts = True)        
            argsort = np.argsort(count_)[::-1][:n_neighbors]        
            dist.append(1 - (count_[argsort].reshape(1,-1)/n_terminal_nodes))
            idx.append(idx_[argsort].reshape(1,-1))
    else:        
        for i in range(len(points)):            
            idx_, count_ = np.unique(
                np.random.choice(
                    np.concatenate(points[i]),
                    size = sample_size
                ),
                return_counts = True
            )
            argsort = np.argsort(count_)[::-1][:n_neighbors]        
            dist.append(1 - (count_[argsort].reshape(1,-1)/n_terminal_nodes))
            idx.append(idx_[argsort].reshape(1,-1))                
    
    #dist, idx = np.array(dist), np.array(idx)
    
    return dist, idx



##############################################
###    Archetype encoding and embedding    ###
##############################################

class ClusterArchetypeEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        ensemble_estimator,        
        embedder = GaussianRandomProjection(50),
        clusterer = MiniBatchKMeans(10),
        boosting_leaf_weight_strategy = "cumulative_unit_gain",
        alpha = 1,
        beta = 1,  
        normalization = "l1",
        fuzzy_membership = False,
        prefit_ensemble = False,
        n_jobs = None,

    ):
        
        self.ensemble_estimator = ensemble_estimator
        self.embedder = embedder
        self.clusterer = clusterer
        self.alpha = alpha
        self.beta = beta                
        self.fuzzy_membership = fuzzy_membership
        self.n_jobs = n_jobs
        self.normalization = normalization

        self.prefit_ensemble = prefit_ensemble
        self.boosting_leaf_weight_strategy = boosting_leaf_weight_strategy
        return
        
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        
        # set estimators
        if not self.embedder is None:            
            embedder_ = clone(self.embedder)
        else:
            embedder_ = FunctionTransformer()
        
        clusterer_ = clone(self.clusterer)
        
        #set ensemble transformers
        #if sklearn forest
        #if lgbm
        if isinstance(self.ensemble_estimator, LGBMModel):            
            ensemble_estimator = LGBMTransformer(self.ensemble_estimator, leaf_weight_strategy = self.boosting_leaf_weight_strategy, prefit_estimator = self.prefit_ensemble)  
        
        elif isinstance(self.ensemble_estimator, BaseForest):
            ensemble_estimator = ForestTransformer(self.ensemble_estimator)
        else:
            raise TypeError(f"for now, only lightgbm.LGBMModel or sklearn.ensemble._forest.BaseForest instances are accepted as ensemble estimator")
        
        ## processing pipeline
        pipe = Pipeline(
            [
                ("ensemble_transformer", ensemble_estimator),
                ("embedder", embedder_),
                ("clusterer", clusterer_),                
                ("sparsifier", FunctionTransformer(sparse.csr_matrix)),
            ]
        )
        
        sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(pipe, sample_weight, **kwargs)
        pipe.fit(X, y, **{**sample_weights, **kws})
        
        self.ensemble_estimator_ = pipe.named_steps["ensemble_transformer"]
        self.processing_pipe_ = pipe
        return self
        
    def transform(self, X, alpha = None, normalization=None):
        
        if alpha is None:
            alpha = self.alpha
        if normalization is None:
            normalization = self.normalization
        
        pointwise_membership = self.processing_pipe_.transform(X)
        pointwise_membership.data = 1/pointwise_membership.data        
        if not alpha is None:
            pointwise_membership.data = pointwise_membership.data**alpha        
        
        if normalization.lower() == "l1":
            pointwise_membership = normalize(pointwise_membership, norm = "l1")
        elif normalization.lower() == "softmax":
            pointwise_membership = softmax(pointwise_membership.A, axis = 1)
            pointwise_membership = sparse.csr_matrix(pointwise_membership)
        else:
            raise ValueError(f"normalization should be one of ['l1', 'softmax'], got {normalization}")

        return pointwise_membership    



class ForestTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, forest, prefit_estimator = False):
        self.forest = forest
        self.prefit_estimator = prefit_estimator
        return
        
    def __getattr__(self, attr):
        return getattr(self.forest, attr)
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        if self.prefit_estimator:
            self.forest = deepcopy(self.forest)
        else:
            self.forest = clone(self.forest)
            self.forest.fit(X, y = y, sample_weight = sample_weight, **kwargs)
        
        leafs = self.forest.apply(X)
        self.leafs_onehotencoder_ = OneHotEncoder().fit(leafs)
        return self                
        
    def transform(self, X):
        X = self.apply(X)
        X = self.leafs_onehotencoder_.transform(X)        
        return X
    

class GraphArchetypeEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        ensemble_estimator,        
        n_archetypes = 100,
        alpha = 1,
        beta = 1,
        boosting_leaf_weight_strategy = "cumulative_information_gain",
        graph_cluster_method = "kmeans",
        use_leaf_weights = False,
        fuzzy_membership = False,
        prefit_ensemble = False,
        n_jobs = None,
        **graph_clustering_kwargs,

    ):
        
        self.ensemble_estimator = ensemble_estimator
        self.n_archetypes = n_archetypes
        self.alpha = alpha
        self.beta = beta                
        self.fuzzy_membership = fuzzy_membership
        self.n_jobs = n_jobs
        self.graph_clustering_kwargs = graph_clustering_kwargs 
        self.use_leaf_weights = use_leaf_weights
        self.prefit_ensemble = prefit_ensemble
        self.graph_cluster_method = graph_cluster_method
        self.boosting_leaf_weight_strategy = boosting_leaf_weight_strategy
        return
    
            
    def _get_graph_clusterer(self, graph_cluster_method, leaf_biadjacency_matrix, n_archetypes, beta, **graph_clustering_kwargs):
        
        if not graph_clustering_kwargs:
            graph_clustering_kwargs = self.graph_clustering_kwargs
            
        if graph_cluster_method is None:
            graph_cluster_method = self.graph_cluster_method
        
        if graph_cluster_method == 'louvain':
            method = Louvain(**graph_clustering_kwargs)
        
        elif graph_cluster_method == 'kmeans':
            method = GraphKMeans(self.n_archetypes, **graph_clustering_kwargs)
        
        elif graph_cluster_method == 'propagation':
            method = PropagationClustering(**graph_clustering_kwargs)
        
        else:
            raise ValueError(f'Suported methods are: ["louvain","propagation", "kmeans"], {graph_cluster_method} was passed.')
        
        graph_embedder = method.fit(leaf_biadjacency_matrix)
        return graph_embedder
    
    
    def _get_leaf_biadjecency_matrix(self, X, beta, sample_weight):
        
        terminal_nodes = self.ensemble_estimator.apply(X)        
        #gets biadjecency matrix
        biadjecency_matrix = self.one_hot_leaf_encoder_.transform(terminal_nodes)

        
        if self.use_leaf_weights:
            if hasattr(self.ensemble_estimator, "leaf_weights_"):
                biadjecency_matrix = biadjecency_matrix.multiply(self.ensemble_estimator.leaf_weights_)
                biadjecency_matrix = sparse.csr_matrix(biadjecency_matrix)
            else:
                raise AttributeError("ensemble_estimator should contain leaf_weights_ attribute error in order to properly use use_leaf_weights")
        
        if not beta is None:
            biadjecency_matrix.data = biadjecency_matrix.data**beta
        
        if not sample_weight is None:
            biadjecency_matrix = biadjecency_matrix.multiply(sample_weight.reshape(-1,1))
            biadjecency_matrix = sparse.csr_matrix(biadjecency_matrix)
        
        return biadjecency_matrix
    
    def _get_archetype_membership(self, X, leaf_memberships, alpha):
        
        pointwise_membership = sparse_dot_product(X,
                                                  leaf_memberships,
                                                  ntop = None,
                                                  lower_bound=0,
                                                  use_threads=False,
                                                  n_jobs=self.n_jobs,
                                                  return_best_ntop=False,
                                                  test_nnz_max=-1,
                                                 )
        
        pointwise_membership.data = pointwise_membership.data**alpha        
        pointwise_membership = normalize(pointwise_membership, norm = "l1")
        
        return pointwise_membership
            
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
                
        if isinstance(self.ensemble_estimator, LGBMModel):            
            ensemble_estimator = LGBMTransformer(self.ensemble_estimator, leaf_weight_strategy = self.boosting_leaf_weight_strategy, prefit_estimator = self.prefit_ensemble)
        
        elif isinstance(self.ensemble_estimator, BaseForest):
            ensemble_estimator = ForestTransformer(self.ensemble_estimator)
        else:
            raise TypeError(f"for now, only lightgbm.LGBMModel or sklearn.ensemble._forest.BaseForest instances are accepted as ensemble estimator")
        
        #fit estimator
        self.ensemble_estimator = clone(ensemble_estimator)            
        sample_weights, kws = _parse_pipeline_sample_weight_and_kwargs(self.ensemble_estimator, sample_weight, **kwargs)
        self.ensemble_estimator.fit(X=X, y=y, **{**kws, **sample_weights})                        
        
        # gets terminal nodes
        terminal_leafs = self.ensemble_estimator.apply(X)
        #fit one hot encoders of the nodes        
        self.one_hot_leaf_encoder_ = OneHotEncoder().fit(terminal_leafs)
        del terminal_leafs
        # get one hot representaiton of leafs
        X = self._get_leaf_biadjecency_matrix(X, self.beta, sample_weight)
        #find graph comunities
        
        com_detector = self._get_graph_clusterer(graph_cluster_method=self.graph_cluster_method,
                                                 leaf_biadjacency_matrix=X,
                                                 n_archetypes=self.n_archetypes,
                                                 beta=self.beta,
                                                 **self.graph_clustering_kwargs
                                                )
        #
        if self.fuzzy_membership:
            #gets the membership as the aerage of the point membership that falls into that leaf
            leaf_memberships = sparse_dot_product(X.T,
                                                  com_detector.membership_row_,
                                                  ntop = None,
                                                  lower_bound=0,
                                                  use_threads=False,
                                                  n_jobs=self.n_jobs,
                                                  return_best_ntop=False,
                                                  test_nnz_max=-1,
                                                 )
            
            leaf_memberships = normalize(leaf_memberships, "l1")
        else:
            leaf_memberships = com_detector.membership_col_

        
        self.graph_clusterer_ = com_detector
        self.leaf_memberships_ = leaf_memberships
        return self
        
    def transform(self, X, alpha = None):
        
        if alpha is None:
            alpha = self.alpha
        
        X = self._get_leaf_biadjecency_matrix(X, beta=self.beta, sample_weight=None)
        pointwise_membership = self._get_archetype_membership(X, self.leaf_memberships_, alpha)                    
        return pointwise_membership    

    

    
class MixedForestGraphArchetypeEncoder(GraphArchetypeEncoder):
    
    def __init__(
        self,
        estimators,        
        estimators_weights = None,
        fully_supervised = True,
        use_already_fitted = False,
        biadjecency_weights = 'uniform',
        stack_outputs = True,
        #embeddings kws
        embedding_method = 'louvain',
        alpha = 1.0,
        return_embeddings_as_sparse = True,
        ensemble_node_weights_attr = None,
        **embedding_kws
    ):
        '''
        a heterogeneous ensemble of forests with bipartitie node-point graph embedding functionality
        
        Parameters
        ----------
        
        estimators: List of forest estimators
            list of estimators to build heterogeneous ensemble
        
        estimators_weights: array like or None
            array like of weights of each passed estimator. If None, will assume not weights. beware that for graph embeddings,
            forests with more estimators will have higher weights since there will be more terminal nodes and thus more edges in the graph.
            To make the total wieght of edges uniform in the bipartite graph, set biadjecency_weights as "uniform"
        
        fully_supervised: bool
            whether all forests are supervised (takes the same y as a parameter) or mixed (some may be unsupervised, like RandomTreeEmeddings)
        
        use_laready_fitted: bool
            if True, will skip fitting those passed estiamtors that are already fitted. Usefull for building embeddings for ensemble of heterogeneous output (classifier + regressor).
            Note that the input must be the same for all estimators or the estimator may induce incosistent results.
            
        
        biadjecency_weights: "n_estimators", "weighted" or "uniform"
            how to weight edges in the resulting node-point bipartite graph. 
            if "n_estimators", forests with more estimators will yield graphs with greater sum of edge weights (more edges in general)
            if "weighted", will uniformize the total graph edges sum and then apply passed wieghts in estimator_weights for each graph
            if "uniform", will uniformize (make the sum of edge wieghts uniform) accross all graphs            
        
        stack_outputs: bool
            whether to stack the outputs after transformations. i.e. return a list of matrices or a single stacked matrix.
        
        embedding_method: {'louvain', 'kmeans', 'propagation'}
            embedding method from sknetwork. Embeddings are calculaated as the normalized linear combination of memberships of terminal nodes.
            for more information about the methods please refer to https://scikit-network.readthedocs.io/en/latest/reference/clustering.html
        
        alpha: float
            concentration parameter. Will concentrate or distribute the embedding dimensions, such that embeddings = normalize(normalize(embeddings)**alpha)
        
        return_embeddings_as_sparse: bool
            whether to return embeddings results as a sparse matrix or a dense one
        
        embedding_kws: key word arguments
            key word araguments passed to the constructor of the clustering object trained to build the embeddings. for more details please refer to https://scikit-network.readthedocs.io/en/latest/reference/clustering.html
        
        
        Returns
        -------
        MixedForestArchetypeGraphEncoder object
        
        '''
        self.estimators = estimators        
        self.estimators_weights = estimators_weights
        self.fully_supervised = fully_supervised
        self.use_already_fitted = use_already_fitted
        self.biadjecency_weights = biadjecency_weights
        self.stack_outputs = stack_outputs
        #embeddings args
        self.embedding_kws = embedding_kws
        self.embedding_method = embedding_method
        self.alpha = alpha
        self.return_embeddings_as_sparse = return_embeddings_as_sparse
        self.ensemble_node_weights_attr = ensemble_node_weights_attr
        return
    
    def __getattr__(self, attr):
        return [getattr(i, attr) for i in self.estimators]
        
    def _iter_apply(self, method, *args, **kwargs):
        
        result = []
        for estim in self.estimators:
            result.append(getattr(estim, method)(*args, **kwargs))
        return result
            
    def apply(self, X, stack = None, **kwargs):
        
        result = self._iter_apply(method = 'apply', X = X)
        
        if stack is None:
            stack = self.stack_outputs
        #handle boosting case where returned array has 3 dims
        for arr in result:            
            if len(arr.shape) >= 3:
                arr = arr.reshape(arr.shape[0], arr.shape[1]*arr.shape[2])                                    
            
        if stack:
            result = hstack(result)
        return result    
    
    def decision_path(self, X, stack = None, **kwargs):
        result = self._iter_apply(method = 'decision_path', X = X)        
        
        if stack is None:
            stack = self.stack_outputs
        
        if stack:
            result0 = sparse.csr_matrix(hstack([i[0] for i in result]))
            result1 = hstack([i[1] for i in result])
        else:
            result0,result1 = result
        
        return result0, result1
    
    def node_biadjecency_matrix(self, X, use_weights = None, stack = None):
        
        '''
        use_weights can be "n_estimators", "weighted" or "uniform"
        unweighted returns biadjecency matrix filled with ones or zeros
        uniform makes the total sum of edge weights from all estimators the same
        wieghted 
        '''
        
        if use_weights is None:
            use_weights = self.biadjecency_weights
        if stack is None:
            stack = self.stack_outputs
        
        #invert natural weights to make the sum of edges in each biadjacency uniform accross all estimators
        natural_weights = np.array([i.n_estimators for i in self.estimators])
        natural_weights = natural_weights/natural_weights.sum()            
        
        uniform_weights = (1/natural_weights)
        uniform_weights = uniform_weights/uniform_weights.sum()
        
        if use_weights == "uniform":
            weights = uniform_weights
        elif use_weights == "weighted":
            weights = uniform_weights*self.estimators_weights_
        elif use_weights == "n_estimators":
            weights = np.ones((len(self.estimators_weights_),))
        else:
            raise ValueError(f'use_weights should be one of ["uniform", "weighted", "n_estimators", None], got {use_weights}')
                            
        terminal_nodes = self.apply(X, stack=False)
        biadjs = []
        for i in range(len(terminal_nodes)):            
            biadj_i = self.one_hot_node_embeddings_encoders_[i].transform(terminal_nodes[i])
            #scale and append            
            biadjs.append(sparse.csr_matrix((biadj_i*weights[i])))
        
        if stack:
            biadjs = hstack(biadjs)
            biadjs = sparse.csr_matrix(biadjs)
        
        return biadjs


    
#export   
class HeterogeneousMixedForest(MixedForestGraphArchetypeEncoder):
    
    def fit(self, X, y = None, sample_weight = None):                        
                    
        #get "natural" weights of estimators
        natural_weights = np.array([i.n_estimators for i in self.estimators])
        natural_weights = natural_weights/natural_weights.sum()            
        #set estimator weights
        if self.estimators_weights is None:
            weights = natural_weights            
        else:
            if len(self.estimators_weights) != len(self.estimators):
                raise ValueError(f'Shape mismatch between estimators and weights ({len(self.estimators)} != {len(self.estimators_weights)})')            
            weights = np.array(self.estimators_weights)
            weights = weights/weights.sum()
                                                            
        
        self.natural_weights_ = natural_weights
        self.estimators_weights_ = weights
        self.classes_ = classes
        self.multilabel_ = multilabel
        self.output_dim_ = output_dim
        
        #fit one hot encoders of the nodes
        terminal_nodes = super().apply(X, stack = False)
        self.one_hot_node_embeddings_encoders_ = [OneHotEncoder().fit(xi) for xi in terminal_nodes]        
        #fit louvain embeddings
        self.fit_archetypes(X, embedding_method = self.embedding_method,)
        return self
    

    
#export
class MixedForestRegressor(MixedForestGraphArchetypeEncoder):
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
                
        #check if multidim output
        if len(y.shape) > 1:
            output_dim = y.shape[-1]
        else:
            output_dim = 1
                        
        #get "natural" weights of estimators
        natural_weights = np.array([i.n_estimators for i in self.estimators])
        natural_weights = natural_weights/natural_weights.sum()            
        #set estimator weights
        if self.estimators_weights is None:
            weights = natural_weights            
        else:
            if len(self.estimators_weights) != len(self.estimators):
                raise ValueError(f'Shape mismatch between estimators and weights ({len(self.estimators)} != {len(self.estimators_weights)})')                                    
            weights = np.array(self.estimators_weights)
            weights = weights/weights.sum()
        
        #fit estimators if needed
        for estim in self.estimators:
            if self.use_already_fitted:
                if not check_is_fitted(estim):
                    if isinstance(estim, Pipeline):
                        kwargs = _parse_pipeline_fit_kws(estim, **kwargs)
                        sample_weights = _parse_pipeline_fit_sample_weight(estim, sample_weight)                
                        estim.fit(X=X, y=y, **{**kwargs, **sample_weights})                        
                    else:
                        estim.fit(X=X, y=y, sample_weight=sample_weight)
                    
                else:
                    warn(f"{estim} is already fitted and use_already_fitted was set to True in the constructor. The estimator won't be fitted, so ensure compatibility of inputs and outputs.")
            else:                
                if isinstance(estim, Pipeline):
                    kwargs = _parse_pipeline_fit_kws(estim, **kwargs)
                    sample_weights = _parse_pipeline_fit_sample_weight(estim, sample_weight)                
                    estim.fit(X=X, y=y, **{**kwargs, **sample_weights})                        
                else:
                    estim.fit(X=X, y=y, sample_weight=sample_weight)
        
        self.natural_weights_ = natural_weights
        self.estimators_weights_ = weights        
        self.output_dim_ = output_dim
        
        #fit one hot encoders of the nodes
        terminal_nodes = super().apply(X, stack = False)
        self.one_hot_node_embeddings_encoders_ = [OneHotEncoder().fit(xi) for xi in terminal_nodes]
        #fit louvain embeddings
        self.fit_archetypes(X, embedding_method = self.embedding_method,)
        return self
    
    
    def predict(self, X, aggregate = True):
        '''
        predicts classes
        '''
        
        weights = self.estimators_weights_
        result = super()._iter_apply(method = 'predict', X = X)
        sum_f = lambda a,b : weights[a]*result[a] + weights[b]*result[b]
        
        if aggregate:
            result = reduce(sum_f, range(len(result)))
        else:            
            result = hstack([i.reshape(-1,1) for i in result])
                
        return result
    
#export
class MixedForestClassifier(MixedForestGraphArchetypeEncoder):            
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
                
        #check if multidim output
        if len(y.shape) > 1:
            output_dim = y.shape[-1]
        else:
            output_dim = 1
            
        if output_dim > 1:
            multilabel = True
        else:
            multilabel = False
            
        #get "natural" weights of estimators
        natural_weights = np.array([i.n_estimators for i in self.estimators])
        natural_weights = natural_weights/natural_weights.sum()            
        #set estimator weights
        if self.estimators_weights is None:
            weights = natural_weights            
        else:
            if len(self.estimators_weights) != len(self.estimators):
                raise ValueError(f'Shape mismatch between estimators and weights ({len(self.estimators)} != {len(self.estimators_weights)})')            
            weights = np.array(self.estimators_weights)
            weights = weights/weights.sum()
            
        #fit estimators if needed
        for estim in self.estimators:
            if self.use_already_fitted:
                if not check_is_fitted(estim):
                    if isinstance(estim, Pipeline):
                        kwargs = _parse_pipeline_fit_kws(estim, **kwargs)
                        sample_weights = _parse_pipeline_fit_sample_weight(estim, sample_weight)                
                        estim.fit(X=X, y=y, **{**kwargs, **sample_weights})                        
                    else:
                        estim.fit(X=X, y=y, sample_weight=sample_weight)
                else:
                    warn(f"{estim} is already fitted and use_already_fitted was set to True in the constructor. The estimator won't be fitted, so ensure compatibility of inputs and outputs.")
            else:                
                if isinstance(estim, Pipeline):
                    kwargs = _parse_pipeline_fit_kws(estim, **kwargs)
                    sample_weights = _parse_pipeline_fit_sample_weight(estim, sample_weight)                
                    estim.fit(X=X, y=y, **{**kwargs, **sample_weights})                        
                else:
                    estim.fit(X=X, y=y, sample_weight=sample_weight)
                                
        #check if classes are the same accross estimators
        if self.fully_supervised:
            classes = [estim.classes_ for estim in self.estimators]
            
            if multilabel:            
                for i in range(len(classes)-1):
                    for j in range(len(classes[i])):
                        assert np.all(classes[i][j] == classes[i+1][j]), f'different classes in estimators: {classes[i]} and {classes[i+1]}'
                classes = classes[0]

            else:
                for i in range(len(classes)-1):                
                    assert np.all(classes[i] == classes[i+1]), f'different classes in estimators: {classes[i]} and {classes[i+1]}'
                classes = classes[0]
        else:
            classes = None                                                                                        
        
        self.natural_weights_ = natural_weights
        self.estimators_weights_ = weights
        self.classes_ = classes
        self.multilabel_ = multilabel
        self.output_dim_ = output_dim
        
        #fit one hot encoders of the nodes
        terminal_nodes = super().apply(X, stack = False)
        self.one_hot_node_embeddings_encoders_ = [OneHotEncoder().fit(xi) for xi in terminal_nodes]
        #fit louvain embeddings
        self.fit_archetypes(X, embedding_method = self.embedding_method,)
        return self
    
    def predict(self, X):
        '''
        predicts classes
        '''
        if self.multilabel_:
            probas = self.predict_proba(X)
            labels = []
            for i in range(len(probas)):
                indices = np.argmax(probas[i], axis = 1)
                labels.append(self.classes_[i][indices])
            labels = np.array(labels).T
        else:
            probas = self.predict_proba(X)
            indices = np.argmax(probas, axis = 1)
            labels = self.classes_[indices]
                
        return labels
        
    
    def predict_proba(self, X):
        '''
        predicts proba of each class
        '''
        weights = self.estimators_weights_
        result = self._iter_apply(method = 'predict_proba', X = X)
        sum_f = lambda a,b : weights[a]*result[a] + weights[b]*result[b]
        if self.multilabel_:
            #reverse the order of arrays in list            
            res = [[] for _ in range(self.output_dim_)]
            for i in range(len(result)):                
                for j in range(self.output_dim_):
                    res[j].append(result[i][j])
            
            result = []
            for dim_result in res:                
                sum_f = lambda a,b : weights[a]*dim_result[a] + weights[b]*dim_result[b]
                res_i = reduce(sum_f, range(len(dim_result)))                
                result.append(res_i) 
        
        else:            
            result = reduce(sum_f, range(len(result)))
        
        return result                                            
    
    pass


##############################################
###      Target encoding and embedding     ###
##############################################




class TargetArchetypeEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        estimator,
        cluster_estimator = None,
        n_archetypes = 100,
        alpha = 1,
        fuzzy_membership = False,
        prefit_estimator = False,
        predict_method = "predict_proba",
        normalization_method = "l1",
        n_jobs = None,
        estimator_fit_kwargs = {},

    ):
        
        self.estimator = estimator        
        self.cluster_estimator=cluster_estimator
        self.n_archetypes = n_archetypes
        self.alpha = alpha
        self.normalization_method = normalization_method
        self.fuzzy_membership = fuzzy_membership
        self.n_jobs = n_jobs
        self.prefit_estimator = prefit_estimator
        self.estimator_fit_kwargs = estimator_fit_kwargs
        self.predict_method = predict_method
        return
    
    def __getattr__(self, attr):
        return getattr(self.estimator, attr)
            
    def fit(self, X, y = None, sample_weight = None, **kwargs):
                
        #fit estimator        
        if not self.prefit_estimator:
            self.estimator = clone(self.estimator)
            if isinstance(self.estimator, Pipeline):
                kwargs = _parse_pipeline_fit_kws(self.estimator, **kwargs)
                sample_weights = _parse_pipeline_fit_sample_weight(self.estimator, sample_weight)                
                self.estimator.fit(X=X, y=y, **{**kwargs, **sample_weights})                        
            else:
                self.estimator.fit(X=X, y=y, sample_weight=sample_weight, **self.estimator_fit_kwargs)
                        
        else:
            pass
        
        # gets outputs
        out = getattr(self.estimator, self.predict_method)(X)
        #fit cluster on output
        if self.cluster_estimator is None:
            clusterer = KMeans(n_clusters=self.n_archetypes).fit(out)
        else:
            clusterer = clone(self.cluster_estimator).fit(out)
        
        self.clusterer_ = clusterer
        return self
        
    def transform(self, X, alpha = None, normalization_method = None):
        out = getattr(self.estimator, self.predict_method)(X)

        if normalization_method is None:
            normalization_method=self.normalization_method
        if alpha is None:
            alpha = self.alpha
        
        if normalization_method == "softmax":
            memberships = self.clusterer_.transform(out)
            memberships = softmax(1/memberships**alpha, 1).astype(np.float32)
            
        elif normalization_method == "l1":
            memberships = self.clusterer_.transform(out)
            memberships = normalize(np.clip(1/memberships**alpha, 0, np.finfo(np.float32).max), "l1")
        else:
            raise ValueError(f"normalization_method should be one of ['softmax','l1'], got {normalization_method}")
        
        return sparse.csr_matrix(memberships)

    
    
    
    
    
from sklearn.base import TransformerMixin, BaseEstimator

class GraphArchetypeUMAP(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        n_archetypes = 100,
        alpha = 1,
        beta = 1,
        gamma = 1,
        n_neighbors=15,
        n_components=2,
        metric='cosine',
        normalize_connections = "l1",
        input_archetype_membership = False,
        fuzzy_membership = False,
        archetype_aggregate_community_graph = None,
        n_jobs = None,
        graph_clustering_kwargs = {},
        umap_kwargs = {},        
    ):
        
        self.n_archetypes = n_archetypes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric 
        self.normalize_connections = normalize_connections
        self.input_archetype_membership = input_archetype_membership
        self.archetype_aggregate_community_graph = archetype_aggregate_community_graph
        self.fuzzy_membership = fuzzy_membership
        self.n_jobs = n_jobs
        self.graph_clustering_kwargs = graph_clustering_kwargs
        self.umap_kwargs = umap_kwargs
        return
    
        
    def _get_graph_clusterer(self, X, n_archetypes, sample_weight, gamma, **graph_clustering_kwargs):
        
        X.data = X.data**gamma
        if not sample_weight is None:
            com_detector = GraphKMeans(n_clusters = n_archetypes, **graph_clustering_kwargs).fit(X.multiply(sample_weight.reshape(-1,1)))
        else:
            com_detector = GraphKMeans(n_clusters = n_archetypes, **graph_clustering_kwargs).fit(X)
        
        return com_detector
    
    def _get_archetype_membership(self, X, leaf_memberships, alpha):
        
        if not self.input_archetype_membership:
            
            pointwise_membership = sparse_dot_product(X,
                                                      leaf_memberships,
                                                      ntop = None,
                                                      lower_bound=0,
                                                      use_threads=False,
                                                      n_jobs=self.n_jobs,
                                                      return_best_ntop=False,
                                                      test_nnz_max=-1,
                                                     )

        else:
            pointwise_membership = X
        
        pointwise_membership.data = pointwise_membership.data**alpha        
        pointwise_membership = normalize(pointwise_membership, norm = "l1")
        
        return pointwise_membership
    
    def _preprocess_aggregate_graph(self, agg_network, beta, normalize_connections):
                
        #agg_network.setdiag(0)
        #agg_network.data = 1/minmax_scale(agg_network.data.reshape(-1,1), (1e-1,1e1)).flatten()
        #agg_network = normalize(agg_network, "l1")
        agg_network = agg_network.copy()
        agg_network.data = agg_network.data**beta
        
        if not normalize_connections is None:
            agg_network = normalize(agg_network, normalize_connections)

        agg_network.eliminate_zeros()        
        return agg_network
    
    def _get_community_embeddings(self, agg_network, **umap_kwargs):
        net_embs = UMAP(**umap_kwargs).fit_transform(agg_network)
        return net_embs
    
    def fit(self, X, y = None, sample_weight = None):
        
        if not self.input_archetype_membership:        
            X = sparse.csr_matrix(X)
            #find graph comunities
            com_detector = self._get_graph_clusterer(X,
                                                     n_archetypes=self.n_archetypes,
                                                     sample_weight=sample_weight,
                                                     gamma=self.gamma,
                                                     **self.graph_clustering_kwargs)
            agg_network = com_detector.aggregate_
            #
            if self.fuzzy_membership:
                leaf_memberships = sparse_dot_product(X.T,
                                                      com_detector.membership_row_,
                                                      ntop = None,
                                                      lower_bound=0,
                                                      use_threads=False,
                                                      n_jobs=self.n_jobs,
                                                      return_best_ntop=False,
                                                      test_nnz_max=-1,
                                                     )
                leaf_memberships = normalize(leaf_memberships, "l1")
            else:
                leaf_memberships = com_detector.membership_col_
                
            agg_network = self._preprocess_aggregate_graph(agg_network, beta=self.beta, normalize_connections=self.normalize_connections)
            importances = agg_network.sum(1).A            
            sizes = agg_network.diagonal()                         
            #
            
            
        else:
            del X
            agg_network = self._preprocess_aggregate_graph(self.archetype_aggregate_community_graph, beta=self.beta, normalize_connections=self.normalize_connections)
            importances = agg_network.sum(1).A
            sizes = agg_network.diagonal()
            leaf_memberships = None
        
        #fit aggregate graph embeddings
        net_embs = self._get_community_embeddings(agg_network,
                                                  metric = self.metric,
                                                  n_neighbors=self.n_neighbors,
                                                  n_components=self.n_components,
                                                  **self.umap_kwargs
                                                 )
        
        self.archetype_importances_ = importances
        self.archetype_sizes_ = sizes
        self.agg_network_ = agg_network
        self.network_embeddings_ = net_embs
        self.leaf_memberships_ = leaf_memberships
        return self
        
    def transform(self, X, alpha = None):
        
        if alpha is None:
            alpha = self.alpha
        
        pointwise_membership = self._get_archetype_membership(X, self.leaf_memberships_, alpha)
        
        embs = sparse_dot_product(pointwise_membership,
                                  self.network_embeddings_,
                                  ntop = None,
                                  lower_bound=0,
                                  use_threads=False,
                                  n_jobs=self.n_jobs,
                                  return_best_ntop=False,
                                  test_nnz_max=-1,
                                 )
        
        if sparse.issparse(embs):
            embs = embs.A    
        return embs
    
    
class ClusterArchetypeUMAP(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        n_components=2,
        *,
        embedder = GaussianRandomProjection(50),
        clusterer = MiniBatchKMeans(100),
        n_neighbors=15,
        metric='cosine',
        normalize_connections = "l1",
        input_archetype_membership = False,
        fuzzy_membership = False,
        alpha = 1,
        beta = 1,
        gamma = 1,                
        n_jobs = None,
        archetype_aggregate_community_graph = None,
        umap_kwargs = {},        
    ):
        
        self.clusterer = clusterer
        self.embedder = embedder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric 
        self.normalize_connections = normalize_connections
        self.input_archetype_membership = input_archetype_membership
        self.archetype_aggregate_community_graph = archetype_aggregate_community_graph
        self.fuzzy_membership = fuzzy_membership
        self.n_jobs = n_jobs
        self.umap_kwargs = umap_kwargs
        return
    
        
    def _get_leaf_membership_and_agg_netowrk(self, X, sample_weight, gamma):
        
        X.data = X.data**gamma        
        embs = self.embedder.fit_transform(X)
        clusters = self.clusterer.fit_predict(embs, sample_weight=sample_weight)
        
        instance_memberships = OneHotEncoder().fit_transform(clusters.reshape(-1,1))
        agg_cluster_network = sparse_dot_product(instance_memberships.T,
                                                  instance_memberships,
                                                  ntop = None,
                                                  lower_bound=0,
                                                  use_threads=False,
                                                  n_jobs=self.n_jobs,
                                                  return_best_ntop=False,
                                                  test_nnz_max=-1,
                                                 )


        leaf_memberships = sparse_dot_product(X.T,
                                              instance_memberships,
                                              ntop = None,
                                              lower_bound=0,
                                              use_threads=False,
                                              n_jobs=self.n_jobs,
                                              return_best_ntop=False,
                                              test_nnz_max=-1,
                                             )

        leaf_memberships = normalize(leaf_memberships, norm = "l1")

        return leaf_memberships, instance_memberships, agg_cluster_network
    
    def _get_archetype_membership(self, X, leaf_memberships, alpha):
        
        if not self.input_archetype_membership:
            
            pointwise_membership = sparse_dot_product(X,
                                                      leaf_memberships,
                                                      ntop = None,
                                                      lower_bound=0,
                                                      use_threads=False,
                                                      n_jobs=self.n_jobs,
                                                      return_best_ntop=False,
                                                      test_nnz_max=-1,
                                                     )

        else:
            pointwise_membership = X
        
        pointwise_membership.data = pointwise_membership.data**alpha        
        pointwise_membership = normalize(pointwise_membership, norm = "l1")
        
        return pointwise_membership
    
    def _preprocess_aggregate_graph(self, agg_network, beta, normalize_connections):
                
        #agg_network.setdiag(0)
        #agg_network.data = 1/minmax_scale(agg_network.data.reshape(-1,1), (1e-1,1e1)).flatten()
        #agg_network = normalize(agg_network, "l1")
        agg_network = agg_network.copy()
        agg_network.data = agg_network.data**beta
        
        if not normalize_connections is None:
            agg_network = normalize(agg_network, normalize_connections)

        agg_network.eliminate_zeros()        
        return agg_network
    
    def _get_community_embeddings(self, agg_network, **umap_kwargs):
        net_embs = UMAP(**umap_kwargs).fit_transform(agg_network)
        return net_embs
    
    def fit(self, X, y = None, sample_weight = None):
        
        if not self.input_archetype_membership:        
            X = sparse.csr_matrix(X)
            #find graph comunities
            
            
            leaf_memberships, instance_memberships, agg_cluster_network = self._get_leaf_membership_and_agg_netowrk(X,                                                     
                                                     sample_weight=sample_weight,
                                                     gamma=self.gamma)
            
            agg_network = agg_cluster_network
            #
            if self.fuzzy_membership:
                leaf_memberships = sparse_dot_product(X.T,
                                                      instance_memberships,
                                                      ntop = None,
                                                      lower_bound=0,
                                                      use_threads=False,
                                                      n_jobs=self.n_jobs,
                                                      return_best_ntop=False,
                                                      test_nnz_max=-1,
                                                     )
                leaf_memberships = normalize(leaf_memberships, "l1")
            else:
                leaf_memberships = leaf_memberships
                
            agg_network = self._preprocess_aggregate_graph(agg_network, beta=self.beta, normalize_connections=self.normalize_connections)
            importances = agg_network.sum(1).A            
            sizes = agg_network.diagonal()                         
            #
            
            
        else:
            del X
            agg_network = self._preprocess_aggregate_graph(self.archetype_aggregate_community_graph, beta=self.beta, normalize_connections=self.normalize_connections)
            importances = agg_network.sum(1).A
            sizes = agg_network.diagonal()
            leaf_memberships = None
        
        #fit aggregate graph embeddings
        net_embs = self._get_community_embeddings(agg_network,
                                                  metric = self.metric,
                                                  n_neighbors=self.n_neighbors,
                                                  n_components=self.n_components,
                                                  **self.umap_kwargs
                                                 )
        
        self.archetype_importances_ = importances
        self.archetype_sizes_ = sizes
        self.agg_network_ = agg_network
        self.network_embeddings_ = net_embs
        self.leaf_memberships_ = leaf_memberships
        return self
        
    def transform(self, X, alpha = None):
        
        if alpha is None:
            alpha = self.alpha
        
        pointwise_membership = self._get_archetype_membership(X, self.leaf_memberships_, alpha)
        
        embs = sparse_dot_product(pointwise_membership,
                                  self.network_embeddings_,
                                  ntop = None,
                                  lower_bound=0,
                                  use_threads=False,
                                  n_jobs=self.n_jobs,
                                  return_best_ntop=False,
                                  test_nnz_max=-1,
                                 )
        
        if sparse.issparse(embs):
            embs = embs.A    
        return embs