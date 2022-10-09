import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, normalize
from sklearn.pipeline import make_pipeline

from joblib import effective_n_jobs, Parallel, delayed

from sknetwork.clustering import KMeans as GraphKMeans
from sknetwork.clustering import PropagationClustering,Louvain

from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

from scipy import sparse

from utils import hstack, sparse_dot_product



#lgbm custom estimators
class CustomLGBMClassifier(LGBMClassifier):
    
    def apply(self, X):
        return self.predict(X, pred_leaf = True)
    
    def decision_path(self, X, decision_weight = None):
        leafs = self.apply(X)
        return self._transform_decision_path(leafs)
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight, **kwargs)
        model_df = self.booster_.trees_to_dataframe()
        node_weights = model_df[model_df["decision_type"].isna()]["weight"].values
        
        leafs = self.predict(X, pred_leaf = True)
        leaf_encoder = OneHotEncoder().fit(leafs)
        self.leaf_encoder_ = leaf_encoder
        self._fit_decision_path(leafs, model_df, decision_weight = "unit_parent_gain")
        self.node_weights_ = node_weights
        
        return self
    
    def _fit_decision_path(self, leafs, model_df, decision_weight = None):

        d_cols = ["split_gain","parent_index","node_index","count"]
        parent_split_gain = pd.merge(model_df[d_cols], model_df[d_cols], left_on = "parent_index", right_on = "node_index", how = "left")

        model_df["parent_gain"] = parent_split_gain["split_gain_y"]
        model_df["parent_count"] = parent_split_gain["count_y"]
        model_df["unit_parent_gain"] = model_df["parent_gain"]/model_df["count"]
        model_df["unit_gain"] = model_df["split_gain"]/model_df["count"]
        model_df["inverse_count"] = 1/model_df["count"]

        model_df["int_index"] = model_df["node_index"].str.split("-").str[1].str[1:].astype(int)
        model_df["leaf_index"] = np.where(model_df["right_child"].isna(), model_df["int_index"], np.nan)

        leaf_indexes = model_df.dropna(subset = ["leaf_index"]).sort_values(by = ["tree_index", "int_index"]).index.values

        g = nx.DiGraph()
        g.add_nodes_from(model_df["node_index"])
        if not decision_weight is None:
            z = list(tuple(i) for i in tuple(model_df[["parent_index","node_index",decision_weight]].dropna().values))
            g.add_weighted_edges_from(z)
        else:
            z = list(tuple(i) for i in tuple(model_df[["parent_index","node_index"]].dropna().values))
            g.add_edges_from(z)

        node_decision_paths = nx.adjacency_matrix(g)
        paths = self._transform_decision_path(leafs)        
        self._leaf_decision_paths = paths
        return 

    def _transform_decision_path(self, leafs):
        encoded_leafs = self.leaf_encoder_.transform(leafs)
        terminal_nodes = np.array(np.split(encoded_leafs.nonzero()[1], leafs.shape[0]))
        paths = node_decision_paths[terminal_nodes.flatten()]
        paths = paths.reshape(leafs.shape[0], paths.shape[0]//leafs.shape[0]*paths.shape[1]).tocsr()

        return paths
    
class CustomLGBMRegressor(LGBMRegressor):
    
    def apply(self, X):
        return self.predict(X, pred_leaf = True)
    
    def decision_path(self, X, decision_weight = None):
        leafs = self.apply(X)
        return self._transform_decision_path(leafs)
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight, **kwargs)
        model_df = self.booster_.trees_to_dataframe()
        node_weights = model_df[model_df["decision_type"].isna()]["weight"].values
        
        leafs = self.predict(X, pred_leaf = True)
        leaf_encoder = OneHotEncoder().fit(leafs)
        self.leaf_encoder_ = leaf_encoder
        self._fit_decision_path(leafs, model_df, decision_weight = "unit_parent_gain")
        self.node_weights_ = node_weights
        
        return self
    
    def _fit_decision_path(self, leafs, model_df, decision_weight = None):

        d_cols = ["split_gain","parent_index","node_index","count"]
        parent_split_gain = pd.merge(model_df[d_cols], model_df[d_cols], left_on = "parent_index", right_on = "node_index", how = "left")

        model_df["parent_gain"] = parent_split_gain["split_gain_y"]
        model_df["parent_count"] = parent_split_gain["count_y"]
        model_df["unit_parent_gain"] = model_df["parent_gain"]/model_df["count"]
        model_df["unit_gain"] = model_df["split_gain"]/model_df["count"]
        model_df["inverse_count"] = 1/model_df["count"]

        model_df["int_index"] = model_df["node_index"].str.split("-").str[1].str[1:].astype(int)
        model_df["leaf_index"] = np.where(model_df["right_child"].isna(), model_df["int_index"], np.nan)

        leaf_indexes = model_df.dropna(subset = ["leaf_index"]).sort_values(by = ["tree_index", "int_index"]).index.values

        g = nx.DiGraph()
        g.add_nodes_from(model_df["node_index"])
        if not decision_weight is None:
            z = list(tuple(i) for i in tuple(model_df[["parent_index","node_index",decision_weight]].dropna().values))
            g.add_weighted_edges_from(z)
        else:
            z = list(tuple(i) for i in tuple(model_df[["parent_index","node_index"]].dropna().values))
            g.add_edges_from(z)

        node_decision_paths = nx.adjacency_matrix(g)
        paths = self._transform_decision_path(leafs)        
        self._leaf_decision_paths = paths
        return 

    def _transform_decision_path(self, leafs):
        encoded_leafs = self.leaf_encoder_.transform(leafs)
        terminal_nodes = np.array(np.split(encoded_leafs.nonzero()[1], leafs.shape[0]))
        paths = node_decision_paths[terminal_nodes.flatten()]
        paths = paths.reshape(leafs.shape[0], paths.shape[0]//leafs.shape[0]*paths.shape[1]).tocsr()

        return paths
    
class CustomLGBMRanker(LGBMRanker):
    
    def apply(self, X):
        return self.predict(X, pred_leaf = True)
    
    def decision_path(self, X, decision_weight = None):
        leafs = self.apply(X)
        return self._transform_decision_path(leafs)
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight, **kwargs)
        model_df = self.booster_.trees_to_dataframe()
        node_weights = model_df[model_df["decision_type"].isna()]["weight"].values
        
        leafs = self.predict(X, pred_leaf = True)
        leaf_encoder = OneHotEncoder().fit(leafs)
        self.leaf_encoder_ = leaf_encoder
        self._fit_decision_path(leafs, model_df, decision_weight = "unit_parent_gain")
        self.node_weights_ = node_weights
        
        return self
    
    def _fit_decision_path(self, leafs, model_df, decision_weight = None):

        d_cols = ["split_gain","parent_index","node_index","count"]
        parent_split_gain = pd.merge(model_df[d_cols], model_df[d_cols], left_on = "parent_index", right_on = "node_index", how = "left")

        model_df["parent_gain"] = parent_split_gain["split_gain_y"]
        model_df["parent_count"] = parent_split_gain["count_y"]
        model_df["unit_parent_gain"] = model_df["parent_gain"]/model_df["count"]
        model_df["unit_gain"] = model_df["split_gain"]/model_df["count"]
        model_df["inverse_count"] = 1/model_df["count"]

        model_df["int_index"] = model_df["node_index"].str.split("-").str[1].str[1:].astype(int)
        model_df["leaf_index"] = np.where(model_df["right_child"].isna(), model_df["int_index"], np.nan)

        leaf_indexes = model_df.dropna(subset = ["leaf_index"]).sort_values(by = ["tree_index", "int_index"]).index.values

        g = nx.DiGraph()
        g.add_nodes_from(model_df["node_index"])
        if not decision_weight is None:
            z = list(tuple(i) for i in tuple(model_df[["parent_index","node_index",decision_weight]].dropna().values))
            g.add_weighted_edges_from(z)
        else:
            z = list(tuple(i) for i in tuple(model_df[["parent_index","node_index"]].dropna().values))
            g.add_edges_from(z)

        node_decision_paths = nx.adjacency_matrix(g)
        paths = self._transform_decision_path(leafs)        
        self._leaf_decision_paths = paths
        return 

    def _transform_decision_path(self, leafs):
        encoded_leafs = self.leaf_encoder_.transform(leafs)
        terminal_nodes = np.array(np.split(encoded_leafs.nonzero()[1], leafs.shape[0]))
        paths = node_decision_paths[terminal_nodes.flatten()]
        paths = paths.reshape(leafs.shape[0], paths.shape[0]//leafs.shape[0]*paths.shape[1]).tocsr()

        return paths

class ForestNeighbors(BaseEstimator):
    
    def __init__(
        self,
        forest_estimator,
        n_neighbors = 30,
        radius = .5,
        n_jobs = 1,
        prefit_forest_estimator = False,
    ):
        '''
        Kneighbors search based on terminal node co-ocurrence.
        Trees can be grown adaptatively (supervised tree ensemble) or randomly (unsupervised tree ensemble)        
        '''
        self.forest_estimator = forest_estimator
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_jobs = n_jobs
        self.prefit_forest_estimator = prefit_forest_estimator
        return
    
    def fit(self, X, y = None, **kwargs):
        '''
        fits forest_estimator if prefir_forest_estimator is set to False. Then, fits neighbors search index
        '''
        
        if not self.prefit_forest_estimator:
            self.forest_estimator.fit(X = X, y = y, **kwargs)
        else:
            #check if fitted, then pass
            check_is_fitted(self.forest_estimator)
            
                        
        #get terminal node indexes
        node_indexes = self.forest_estimator.apply(X)        
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
        indexes = self.forest_estimator.apply(X)
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


class ArchetypeEmbedder(BaseEstimator):
    #fit models and graph embeddings on terminal node space
    #save louvain terminal node embeddings for inference
    #transform method will yield louvain embeddings of the point
    #create get_label method
    
    def __init__(
        self,
        forest_estimator,
        embedding_method = 'louvain',        
        alpha = 1.0,
        use_leaf_weights = False,
        return_embeddings_as_sparse = True,
        ensemble_node_weights_attr = None,
        **embedding_kws        
    ):
        '''
        a heterogeneous ensemble of forests with bipartitie node-point graph embedding functionality
        
        Parameters
        ----------
        
        forest_estimator: forest estimator
            Forest estimator. Can be sklearns or any other implementation containing  the `apply` method
                
        
        embedding_method: {'louvain', 'kmeans', 'propagation'}
            embedding method from sknetwork. Embeddings are calculaated as the normalized linear combination of memberships of terminal nodes.
            for more information about the methods please refer to https://scikit-network.readthedocs.io/en/latest/reference/clustering.html
        
        alpha: float
            concentration parameter. Will concentrate or distribute the embedding dimensions, such that embeddings = normalize(normalize(embeddings)**alpha)
        
        return_embeddings_as_sparse: bool
            whether to return embeddings results as a sparse matrix or a dense one
        
        ensemble_node_weights_attr: str or None
            if the node of the trees in the ensemble contaion weights (like in boosters) and it is desired
            to use the node weights when computing the communities of each node, the base estimator should contaion
            an attribute that returns the weight of each node in a 1d vector, in the order they appear in OneHotEncoder().transform(ensemble.apply(X)).
            If None, no weights are used
        
        embedding_kws: key word arguments
            key word araguments passed to the constructor of the clustering object trained to build the embeddings. for more details please refer to https://scikit-network.readthedocs.io/en/latest/reference/clustering.html
            
        
        
        
        Returns
        -------
        ArchetypeEmbedder object
        
        '''
        self.forest_estimator = forest_estimator
        self.embedding_method = embedding_method
        self.return_embeddings_as_sparse = return_embeddings_as_sparse
        self.alpha = alpha
        self.embedding_kws = embedding_kws
        self.ensemble_node_weights_attr = ensemble_node_weights_attr
        self.use_leaf_weights = use_leaf_weights
        return
    
    def __getattr__(self, attr):
        '''
        returns self.forest_estimator attribute if not found in class definition
        '''
        return getattr(self.forest_estimator, attr)
    
    
    def fit(self, X, y = None, **kwargs):
        '''
        fits the estimator in supervised or unsupervised manner, according to `forest_estimator` type.
        
        Parameters
        ----------
        
        X: array like
            input independent variables
        
        y: array like
            output target variables
        
        **kwargs:
            keyword arguments passed in self.forest_estimator.fit(X = X, y = y, **kwargs)
        
        Returns
        -------
        
        ArchetypeEmbedder fitted object
        
        '''
        #fit estimator
        self.forest_estimator.fit(X = X, y = y, **kwargs)
        # gets terminal nodes
        terminal_nodes = self.apply(X)
        #fit one hot encoders of the nodes        
        self.one_hot_node_embeddings_encoders_ = OneHotEncoder().fit(terminal_nodes)
        #fits emedder
        self.graph_embedder_ = self._fit_embeddings(X, **kwargs)
        return self
        
    def apply(self, X, **kwargs):
        '''
        applies method "apply" of `forest_estimator` ensuring shape compatibility
        '''
        X = self.forest_estimator.apply(X, **kwargs)
        #handle boosting case
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        return X
    
    def node_biadjecency_matrix(self, X):
        
        terminal_nodes = self.apply(X)        
        #gets biadjecency matrix
        biadjecency_matrix = self.one_hot_node_embeddings_encoders_.transform(terminal_nodes)
        biadjecency_matrix = sparse.csr_matrix(biadjecency_matrix)
        
        if self.use_leaf_weights:
            if hasattr(self.forest_estimator, "leaf_weights_"):
                biadjecency_matrix = biadjecency_matrix.multiply(self.forest_estimator.leaf_weights_)
            else:
                raise AttributeError("forest_estimator should contain leaf_weights_ attribute error in order to properly use use_leaf_weights"):
                
        return biadjecency_matrix
    
    def fit_embeddings(self, X, embedding_method = None, **kwargs):
        '''
        fits only the embedding object. Will use the forest estimator that was fitted in the `fit` method
        
        Parameters
        ----------
        
        X: array like
            input matrix
        
        embedding_method: {"kmeans", "propagation", "louvain"}
            Clustering method for the terminal nodes. Please refer to https://scikit-network.readthedocs.io/en/latest/reference/clustering.html .
            if None is passed, will use `embedding_method` passed in the constructor.
        
        **kwargs:
            keyword arguments passed to the embedding object constructor. Please refer to https://scikit-network.readthedocs.io/en/latest/reference/clustering.html .
            
            
        '''                
        self.graph_embedder_ = self._fit_embeddings(X, embedding_method, **kwargs)
        return self
    
    def _fit_embeddings(self, X, embedding_method = None, **kwargs):
        
        '''
        fits embedder and returns fitted object
        '''
                        
        if not kwargs:
            kwargs = self.embedding_kws
            
        if embedding_method is None:
            embedding_method = self.embedding_method
        
        if embedding_method == 'louvain':
            method = Louvain(**kwargs)
        
        elif embedding_method == 'kmeans':
            method = GraphKMeans(**kwargs)
        
        elif embedding_method == 'propagation':
            method = PropagationClustering(**kwargs)
        
        else:
            raise ValueError(f'Suported methods are: ["louvain","propagation", "kmeans"], {embedding_method} was passed.')
        
        G = self.node_biadjecency_matrix(X)
        graph_embedder = method.fit(G)
        return graph_embedder
        
        
    def transform(self, X, alpha = None, return_embeddings_as_sparse = None):
        '''
        Maps X from feature space to embedding space. Embedings are a normalized linear combination of the cluster memberships of the terminal of each point in the ensemble of trees.
        It can be interpreted as a Fuzzy clustering index, in the sense that points in between clusters will have their membership shared accross their neighbour clusters.
        
        Parameters
        ----------
        X: array like
            Input data.
        
        alpha: float
            concentration parameter. Will concentrate or distribute the embedding dimensions, such that embeddings = normalize(normalize(embeddings)**alpha).
            if None is passed, will use alpha value passed in the constructor.
        
        return_embeddings_as_sparse: bool
            whether to return embeddings results as a sparse matrix or a dense one.
            if None is passed, will use return_embeddings_as_sparse value passed in the constructor.
            
        
        Returns
        -------
        
        Embeddings: Sparse or dense array
             
             Mapping of the input in the embedding space        
        
        '''
        if alpha is None:
            alpha = self.alpha
            
        if return_embeddings_as_sparse is None:
            return_embeddings_as_sparse = self.return_embeddings_as_sparse
            
        X = self.node_biadjecency_matrix(X)
        terminal_node_embs = sparse.csr_matrix(self.graph_embedder_.membership_col_)
                        
        #normalization of the normalized exponentialized embeddings
        embs = normalize(sparse_dot_product(X, terminal_node_embs, terminal_node_embs.shape[-1]), 'l1')
        if alpha != 1:
            embs.data  = embs.data**alpha
            embs = normalize(embs, 'l1')
        else:
            pass
        
        if return_embeddings_as_sparse:
            embs = sparse.csr_matrix(embs)
        else:
            if sparse.issparse(embs):
                embs = embs.A
            else:
                pass
        
        return embs    
    
    

    
class MixedForestArchetypeEmbedder(ArchetypeEmbedder):
    
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
        MixedForestArchetypeEmbedder object
        
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
class HeterogeneousMixedForest(MixedForestArchetypeEmbedder):
    
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
        self.fit_embeddings(X, embedding_method = self.embedding_method,)
        return self
    

    
#export
class MixedForestRegressor(MixedForestArchetypeEmbedder):
    
    def fit(self, X, y = None, sample_weight = None):
                
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
                    estim.fit(X, y = y, sample_weight = sample_weight)
                else:
                    warn(f"{estim} is already fitted and use_already_fitted was set to True in the constructor. The estimator won't be fitted, so ensure compatibility of inputs and outputs.")
            else:                
                estim.fit(X, y = y, sample_weight = sample_weight)                                                                                                                                
        
        self.natural_weights_ = natural_weights
        self.estimators_weights_ = weights        
        self.output_dim_ = output_dim
        
        #fit one hot encoders of the nodes
        terminal_nodes = super().apply(X, stack = False)
        self.one_hot_node_embeddings_encoders_ = [OneHotEncoder().fit(xi) for xi in terminal_nodes]
        #fit louvain embeddings
        self.fit_embeddings(X, embedding_method = self.embedding_method,)
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
class MixedForestClassifier(MixedForestEmbedder):            
    
    def fit(self, X, y = None, sample_weight = None):
                
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
                    estim.fit(X, y = y, sample_weight = sample_weight)
                else:
                    warn(f"{estim} is already fitted and use_already_fitted was set to True in the constructor. The estimator won't be fitted, so ensure compatibility of inputs and outputs.")
            else:                
                estim.fit(X, y = y, sample_weight = sample_weight)
                                
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
        self.fit_embeddings(X, embedding_method = self.embedding_method,)
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
    
class LeafEmbedder(ArchetypeEmbedder):
    """
    uses archetype embeddings to reduce dimensionality using the comunity aggregated network to anchor points in a N dimensional space
    """
    pass