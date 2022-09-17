import numpy as np

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

class RobustKBinsDiscretizer(KBinsDiscretizer):
    
    #TODO: allow extrapolation behaviour in the extremes if desired (not sure if conceptually possible)
    def __init__(
        self,
        n_bins=5,
        *,
        encode='onehot',
        strategy='quantile',
        fuzzy_alpha = 1,
        dtype=None,
        handle_nan = 'handle', #error, handle, ignore        
    ):        
        self.encode = encode
        self.strategy = strategy
        self.fuzzy_alpha = fuzzy_alpha
        self.dtype = dtype
        self.handle_nan = handle_nan      
        super().__init__(n_bins = n_bins,encode = encode,strategy = strategy,dtype = dtype)
        return
    
    
    def fit(self, X, y=None):
        """
        Fit the estimator.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        valid_handling = ["handle","error","ignore"]
        if not self.handle_nan in valid_handling:
            raise ValueError(f'handle_nan should be one of {valid_handling}, got {self.handle_nan}')
                                
            
        #make it robust to NaNs
        if self.handle_nan in ('handle','ignore'):
            X = self._validate_data(X, dtype="numeric", force_all_finite = "allow-nan")
        else:
            X = self._validate_data(X, dtype="numeric")
            
        supported_dtype = (np.float64, np.float32)
        if self.dtype in supported_dtype:
            output_dtype = self.dtype
        elif self.dtype is None:
            output_dtype = X.dtype
        else:
            raise ValueError(
                "Valid options for 'dtype' are "
                f"{supported_dtype + (None,)}. Got dtype={self.dtype} "
                " instead."
            )

        valid_encode = ("onehot", "onehot-dense", "ordinal","fuzzy")
        if self.encode not in valid_encode:
            raise ValueError(
                "Valid options for 'encode' are {}. Got encode={!r} instead.".format(
                    valid_encode, self.encode
                )
            )
        valid_strategy = ("uniform", "quantile", "kmeans")
        if self.strategy not in valid_strategy:
            raise ValueError(
                "Valid options for 'strategy' are {}. "
                "Got strategy={!r} instead.".format(valid_strategy, self.strategy)
            )

        n_features = X.shape[1]        
        if self.n_bins == 'auto':
            #nbins heursitic
            self.n_bins =int(max(4, min(10000, int(np.power(X.shape[0],0.5)))))
            n_bins = self._validate_n_bins(n_features)            
            self.n_bins = 'auto'
        else:
            n_bins = self._validate_n_bins(n_features)
        
        bin_edges = np.zeros(n_features, dtype=object)
        bin_lens = np.zeros(n_features, dtype=object)
        bin_centers = np.zeros(n_features, dtype=object)
        for jj in range(n_features):                        
            #select column to work with
            column = X[:, jj]
            #make it Robust to NaNs excluding them from fit
            if self.handle_nan in ('handle','ignore'):
                column = column[~np.isnan(column.flatten())]
            else:
                column = column
            
            col_min, col_max = column.min(), column.max()

            if col_min == col_max:
                warnings.warn(
                    "Feature %d is constant and will be replaced with 0." % jj
                )
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.strategy == "uniform":
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)

            elif self.strategy == "quantile":
                quantiles = np.linspace(0, 100, n_bins[jj] + 1)
                bin_edges[jj] = np.asarray(np.percentile(column, quantiles))

            elif self.strategy == "kmeans":
                #from ..cluster import KMeans  # fixes import loops

                # Deterministic initialization with uniform spacing
                uniform_edges = np.linspace(col_min, col_max, n_bins[jj] + 1)
                init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5

                # 1D k-means procedure
                km = KMeans(
                    n_clusters=n_bins[jj], init=init, n_init=1, algorithm="full"
                )
                centers = km.fit(column[:, None]).cluster_centers_[:, 0]
                # Must sort, centers may be unsorted even with sorted init
                centers.sort()
                bin_edges[jj] = (centers[1:] + centers[:-1]) * 0.5
                bin_edges[jj] = np.r_[col_min, bin_edges[jj], col_max]

            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy in ("quantile", "kmeans"):
                mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                bin_edges[jj] = bin_edges[jj][mask]
                if len(bin_edges[jj]) - 1 != n_bins[jj]:
                    warnings.warn(
                        "Bins whose width are too small (i.e., <= "
                        "1e-8) in feature %d are removed. Consider "
                        "decreasing the number of bins." % jj
                    )
                    n_bins[jj] = len(bin_edges[jj]) - 1
            
            #create other attributes
            #bin_sizes
            bin_edges_i = bin_edges[jj]
            bin_lens_i = np.diff(bin_edges_i)
            #midle point between bins
            bin_centers_i = bin_edges_i[:-1] + bin_lens_i/2
            #set extreme bin centers to min and max values of bin edges
            bin_centers_i[-1] = bin_edges_i[-1]
            bin_centers_i[0] = bin_edges_i[0]
            #multiply extreme lens by 2 to enforce it containing extreme points
            bin_lens_i[-1] *=2
            bin_lens_i[0] *=2
            #append to containers
            bin_lens[jj] = bin_lens_i
            bin_centers[jj] = bin_centers_i
            pass #end for

        self.bin_edges_ = np.array([np.array([i]) if not isinstance(i, np.ndarray) else i for i in bin_edges])
        self.n_bins_ = n_bins
        self.bin_lens_ = np.array([np.array([i]) if not isinstance(i, np.ndarray) else i for i in bin_lens])
        self.bin_centers_ = np.array([np.array([i]) if not isinstance(i, np.ndarray) else i for i in bin_centers])
        self.n_features_ = n_features
        #add one bin if create_nan_bin is True        
        if self.handle_nan == 'handle':
            self.n_bins_ += 1
            
        if self.encode in ("onehot", "fuzzy"):
            if self.handle_nan == 'handle':
                categories = [np.arange(-1,i-1) for i in self.n_bins_]
            elif self.handle_nan == 'ignore':
                categories = [np.arange(i) for i in self.n_bins_]
            else:
                categories = [np.arange(i) for i in self.n_bins_]
            
            self._encoder = OneHotEncoder(
                categories=categories,
                sparse=self.encode == "onehot",
                dtype=output_dtype,
                handle_unknown = "ignore" if self.handle_nan in ('ignore','handle') else "error"
            )
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KBinsDiscretizer is fitted
            self._encoder.fit(np.zeros((1, len(self.n_bins_))))

        #create index correction for transform
        extra_bin = 1 if (self.handle_nan == 'handle') else 0
        col_index_correction = np.cumsum([len(i) -1 + extra_bin for i in self.bin_edges_]) #if all handles are set to ignore. add +1 for each handle (nan, unknown)                
        col_index_correction = np.insert(col_index_correction, 0, 0)[:-1]                
        self.col_index_correction_ = col_index_correction

        return self

    def _transform_ordinal(self, X):
        '''
        returns ordinal result before firther encoding
        '''
        # check input and attribute dtypes
        dtype = (np.float64, np.float32) if self.dtype is None else self.dtype
        if self.handle_nan in ('handle','ignore'):
            Xt = self._validate_data(X, copy=True, dtype=dtype, reset=False, force_all_finite = 'allow-nan')
        else:
            Xt = self._validate_data(X, copy=True, dtype=dtype, reset=False)
        
        bin_edges = self.bin_edges_
        if self.handle_nan in ('handle','ignore'):
            for jj in range(Xt.shape[1]):
                #create empty array of same shape to populate with nans
                Xt_temp = np.empty(Xt[:,jj].shape)
                Xt_nanmsk = np.isnan(Xt[:,jj].flatten())
                column = Xt[:,jj][~Xt_nanmsk]
                # Values which are close to a bin edge are susceptible to numeric
                # instability. Add eps to X so these values are binned correctly
                # with respect to their decimal truncation. See documentation of
                # numpy.isclose for an explanation of ``rtol`` and ``atol``.
                rtol = 1.0e-5
                atol = 1.0e-8
                eps = atol + rtol * np.abs(column)
                column = np.digitize(column + eps, bin_edges[jj][1:])
                if self.handle_nan == 'handle':
                    #clip up to self.n_bins_ - 2, since theres 1 bin for NaNs
                    np.clip(column, 0, self.n_bins_[jj] - 2, out=column)
                    #create NaN Category                
                    Xt_temp[Xt_nanmsk] = -1
                else:                    
                    np.clip(column, 0, self.n_bins_[jj] -1, out=column)
                    #create NaN Category                
                    Xt_temp[Xt_nanmsk] = -1
                                
                #fill template where there are no NaNs
                Xt_temp[~Xt_nanmsk] = column                
                Xt[:, jj] = Xt_temp                        
                
        else:
            for jj in range(Xt.shape[1]):
                # Values which are close to a bin edge are susceptible to numeric
                # instability. Add eps to X so these values are binned correctly
                # with respect to their decimal truncation. See documentation of
                # numpy.isclose for an explanation of ``rtol`` and ``atol``.
                rtol = 1.0e-5
                atol = 1.0e-8
                eps = atol + rtol * np.abs(Xt[:, jj])
                Xt[:, jj] = np.digitize(Xt[:, jj] + eps, bin_edges[jj][1:])            
                
            #clip up to self.n_bins_ - 1 (zero to self.n_bins_ - 1)
            np.clip(Xt, 0, self.n_bins_ - 1, out=Xt)
        
        return Xt
    
    def transform(self, X):
        """
        Discretize the data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.
        Returns
        -------
        Xt : {ndarray, sparse matrix}, dtype={np.float32, np.float64}
            Data in the binned space. Will be a sparse matrix if
            `self.encode='onehot'` and ndarray otherwise.
        """
        check_is_fitted(self)

        # check input and attribute dtypes
        dtype = (np.float64, np.float32) if self.dtype is None else self.dtype
        if self.handle_nan in ('handle','ignore'):
            X = self._validate_data(X, copy=True, dtype=dtype, reset=False, force_all_finite = 'allow-nan')
        else:
            X = self._validate_data(X, copy=True, dtype=dtype, reset=False)        

        bin_edges = self.bin_edges_
        #transform to ordinal results
        Xt = self._transform_ordinal(X)

        if self.encode == "ordinal":
            return Xt

        dtype_init = None
        if self.encode in ("onehot","fuzzy"):
            dtype_init = self._encoder.dtype
            self._encoder.dtype = Xt.dtype        
        try:
            Xt_enc = self._encoder.transform(Xt)
        finally:
            # revert the initial dtype to avoid modifying self.
            self._encoder.dtype = dtype_init
        
        if "fuzzy" in self.encode:
            Xt_enc = sparse.csr_matrix(Xt_enc)
            Xt_enc = self._fuzzy_transform(X, Xt, Xt_enc)
        
        return Xt_enc

    def _fuzzy_transform(self, X, Xt, Xt_enc):
        #apply this only to non NaN rows
        #get index correction array to access final trnasformed columns correctly
        col_index_correction = self.col_index_correction_
        #cast to lilmatrix to append values easily
        Xt_enc = Xt_enc.tolil()
        for i in range(Xt.shape[1]):
            column = X[:,i]
            nan_msk = np.isnan(column)            
            column_ordinal = Xt[:,i].astype(int)            
            #belonging to the bin:
            #calculate fuzzy score for rows
            try:
                fuzzy_score = (column - self.bin_centers_[i][column_ordinal])/self.bin_lens_[i][column_ordinal]
            except:
                print(self.bin_centers_)
                print(self.bin_centers_[i])
                print(self.n_features_)

            fuzzy_score_extrapolation_msk = np.abs(fuzzy_score) > 1
            fuzzy_score_extrapolation = fuzzy_score[fuzzy_score_extrapolation_msk]
            
            extreme_left_msk = column_ordinal <= 0
            extreme_right_msk = column_ordinal >= (len(self.bin_edges_[i]) - 2)

            left_fuzzy_msk = fuzzy_score <= 0
            right_fuzzy_msk = fuzzy_score >= 0

            nonfuzzy_msk =  (left_fuzzy_msk & extreme_left_msk) | (right_fuzzy_msk & extreme_right_msk)
            #set nonfuzzy to zero only to facilitate defining neighbor cols
            #will set nonfuzzy to 1 after that

            #define columns that each fuzzy score will be appended in final encoded matrix
            delta_neighbor_col_idx = np.zeros(column_ordinal.shape)
            delta_neighbor_col_idx[fuzzy_score > 0] = 1
            delta_neighbor_col_idx[fuzzy_score < 0] = -1
            delta_neighbor_col_idx[nonfuzzy_msk] = 0

            corrected_column_ordinal = column_ordinal + col_index_correction[i]
            neighbor_col_idx = (corrected_column_ordinal + delta_neighbor_col_idx).astype(int)

            fuzzy_rows_mask = (~nan_msk) & (~nonfuzzy_msk)
            fuzzy_rows = np.nonzero(fuzzy_rows_mask)[0]

            correction = 1 if self.handle_nan == 'handle' else 0
            neighbor_cols = neighbor_col_idx[fuzzy_rows] + correction
            bin_cols = corrected_column_ordinal[fuzzy_rows] + correction

            
            fuzzy_score = np.abs(fuzzy_score)[fuzzy_rows]
                        
            
            if len(fuzzy_score) > 0:
                #apply alpha to 
                normalized_fuzzy = fuzzy_score.reshape(-1,1)
                normalized_fuzzy = normalize(np.hstack([normalized_fuzzy, 1 - normalized_fuzzy])**self.fuzzy_alpha, 'l1')[:, 0].flatten()
                #subtract a small value to ensure invertibility (fuzzy score for the bin th epoin belongs to is always greater)
                normalized_fuzzy[normalized_fuzzy == 0.5] = 0.5 - 1e-6
                #replace ones with complementary fuzzy scores            
                Xt_enc[fuzzy_rows, bin_cols] = 1 - normalized_fuzzy
                #fill neighbor cells with fuzzy score
                Xt_enc[fuzzy_rows, neighbor_cols] = normalized_fuzzy                                                                                
        
        #converts back to csr
        Xt_enc = Xt_enc.tocsr()
        return Xt_enc
    
    def inverse_transform(self, Xt):
        """
        Transform discretized data back to original feature space.
        Note that this function does not regenerate the original data
        due to discretization rounding.
        Parameters
        ----------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data in the binned space.
        Returns
        -------
        Xinv : ndarray, dtype={np.float32, np.float64}
            Data in the original feature space.
        """
        check_is_fitted(self)

        if self.encode in ("onehot"):
            Xt = self._encoder.inverse_transform(Xt)            

        if self.encode in ("fuzzy"):
            Xinv = self._fuzzy_inverse_transform(Xt)
            return Xinv
        
        Xinv = check_array(Xt, copy=True, dtype=(np.float64, np.float32), force_all_finite='allow-nan')
        
        if self.handle_nan in ('handle','ignore'):            
            xinv_nanmsk = (Xinv == - 1) | np.isnan(Xinv)
        elif self.handle_nan == 'ignore':
            xinv_nanmsk = np.isnan(Xinv)
        else:
            xinv_nanmsk = None
        
        n_features = self.n_bins_.shape[0]
        if Xinv.shape[1] != n_features:
            raise ValueError(
                "Incorrect number of features. Expecting {}, received {}.".format(
                    n_features, Xinv.shape[1]
                )
            )

        for jj in range(n_features):            
            bin_edges = self.bin_edges_[jj]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            if self.handle_nan in ('handle', 'ignore'):
                nanmsk = xinv_nanmsk[:, jj].flatten()
                Xinv[~nanmsk, jj] = bin_centers[np.int_(Xinv[~nanmsk, jj])]
                Xinv[nanmsk, jj] = np.nan
            else:
                Xinv[:, jj] = bin_centers[np.int_(Xinv[:, jj])]

        return Xinv
    
    def _fuzzy_inverse_transform(self, Xinv):
    
        '''
        invert fuzzy one hot values if encoding method is fuzzy.
        this method recovers the original values without information loss if the original value is between the range
        X_train[:,i].min() <= X <= X_train[:,i].max().
        beyond the left and right borders, the inverse are 
        inverse_transform(X_train[:,i].min()) and inverse_transform(X_train[:,i].max())
        respectively
        '''
        if not sparse.issparse(Xinv):
            Xinv = sparse.csr_matrix(Xinv)

        #nonzero rows and cols
        nonzero_row_col = Xinv.nonzero()
        #get cols with fuzzy between zero and one (fuzzy)
        fuzzy_row_col = (Xinv > 0).multiply((Xinv < 1)).nonzero()
        #get left conjugate index
        fuzzy_left_row_col = fuzzy_row_col[0][::2], fuzzy_row_col[1][::2]
        #get left conjugate values
        left_values = Xinv[fuzzy_left_row_col].A.flatten()
        #decide whether the center of the bin is the left or right cell
        delta_left_col = np.zeros(left_values.shape)
        delta_left_col[left_values < 0.5] = 1
        #get right conjugate (c)
        delta_right_col = np.abs(delta_left_col - 1)

        #make tuple with coordinaates of center (where the fuzzy membership is higher) cols
        main_row_cols = fuzzy_left_row_col[0], (fuzzy_left_row_col[1] + delta_left_col).astype(int)

        #calculate fuzzy scores
        fuzzy_score = 1 - Xinv[main_row_cols].A.flatten()
        #signal to define whether the inverse is to the right or to the left of the center bin
        fuzzy_score_signal = np.where(delta_left_col == 0, 1, -1)
        fuzzy_score = fuzzy_score*fuzzy_score_signal

        #get bin of each value based on inverse
        main_bin_cols = np.digitize(main_row_cols[1],self.col_index_correction_) -1

        Xord = self._encoder.inverse_transform(Xinv)
        Xord = np.where(Xord == None, -1, Xord)
        correction = self.col_index_correction_


        for i in range(Xord.shape[-1]):
            #get fuzzy scores, column and row indxs for values where dim == i
            fuzzy_score_i = np.zeros(Xord.shape[0])
            column_ordinal = Xord[:,i].astype(int)        
            dim_msk = main_bin_cols == i
            fuzzy_rows_i = main_row_cols[0][dim_msk]        
            bin_edges_i = rob.bin_edges_[i]
            fuzzy_score_i[fuzzy_rows_i] = fuzzy_score[dim_msk]
            #invert the results
            Xinv_i = fuzzy_score_i*rob.bin_lens_[i][column_ordinal]+rob.bin_centers_[i][column_ordinal]
            #fill with Nans
            Xinv_i[column_ordinal == -1] = np.nan
            #append to Xord to avoid creating another array with same shape
            Xord[:,i] = Xinv_i

        return Xord

    
class RobustOneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        categories = 'auto',
        dtype=np.float64,
        handle_unknown = 'error',# for one hot {'error', 'ignore','handle'}
        drop=None,
        sparse=True,
        nan_value = np.nan, #could be "nan" (Str) for example
        handle_nan = 'handle', #error, handle, ignore        
    ):
        '''
        A robust one hot encoder. Always return the same amount of nonzero value sin each transformed row.
        Has columns for unknown values
        '''
        
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.sparse = sparse
        self.nan_value = nan_value
        self.handle_nan = handle_nan
        return
                    
    def _nan_msk(self, X):
        """
        check NaNs in special cases like 'O' type arrays
        """
        
        
        if self.nan_value in (np.nan,):
            if X.dtype == 'O':
                nanmsk_ = X == 'nan' #'nan' that are not proper NaNs
                nanmsk = X.astype(str) == 'nan' #proper and not propper NaNs
                #exclude values that had "nan" instead of np.nan
                #if want to count them as valid NaNs, perform transformation upstream
                nanmsk = nanmsk & (~nanmsk_)
            else:
                nanmsk = np.isnan(X)
        else:
            nanmsk = (X == self.nan_value)
        
        return nanmsk
    
    def _check_nan_cols(self, X):
        '''
        returns cols with corresponding with self.nan_value
        '''        
        nanmsk = self._nan_msk(X)
        cols_w_nans = np.nonzero(nanmsk.any(0))[0]
        return cols_w_nans        
        
    def _infer_categories(self, X):
        
        categories = []
        for jj in range(X.shape[1]):            
            
            if self.handle_nan in ('handle','ignore'):
                #if nan handling, create a NaN msk
                if self.nan_value in (np.nan,):
                    if X[:, jj].dtype == 'O':
                        nanmsk_ = X[:, jj] == 'nan' #'nan' that are not proper NaNs
                        nanmsk = X[:, jj].astype(str) == 'nan' #proper and not propper NaNs
                        #exclude values that had "nan" instead of np.nan
                        #if want to count them as valid NaNs, perform transformation upstream
                        nanmsk = nanmsk & (~nanmsk_)
                    else:
                        nanmsk = np.isnan(X[:, jj])
                    
                else:
                    nanmsk = (X[:, jj] == self.nan_value)
                
                #if want to create nan dummies, should include nans do categories
                #even if not found during train
                if self.handle_nan == 'handle':
                    unq = _unique(X[:,jj])
                    
                    if (nanmsk.sum() == 0)  and (self.handle_unknown == 'handle'):
                        #append nan class even if there are no nans in traning
                        #unknown nans are treatted diffrently from unknown non-nans
                        unq = np.append(unq, self.nan_value)
                                            
                    categories.append(unq)
                
                #if nans are to be ignored in dummies, exclude nans in training set
                else:                                        
                    #exclude NaNs and append only other categories
                    #case where handle_nan is set to ignore or create_nan_dummy is set to False
                    categories.append(_unique(X[~nanmsk,jj]))
            
            #if not handle_nan, appends regular unique categories
            #nan checking is assumed to be performed upstream
            else:
                categories.append(_unique(X[:,jj]))
            
        return categories            
            
    def fit(self, X, y = None, **kwawrgs):                                
        
        #check array
        X = check_array(X, dtype = None, force_all_finite = 'allow-nan')
        #check nans
        if self.handle_nan == 'error':
            nancols = self._check_nan_cols(X).tolist()
            if nancols:
                raise ValueError(f'Columns {[i for i in nancols]} has NaN Values set to {self.nan_value} ({type(self.nan_value)})')        
        
        ordinal_handle = 'use_encoded_value' if self.handle_unknown in ('handle','ignore') else 'error'
        if self.categories == 'auto':
            ordinal_categories = self._infer_categories(X)
        else:
            ordinal_categories =self.categories
            
        self.ordinalencoder_ = OrdinalEncoder(
            handle_unknown = ordinal_handle,
            unknown_value = -1 if ordinal_handle == 'use_encoded_value' else None,
            categories = ordinal_categories,
            dtype = self.dtype,            
        ).fit(X)        
        
        X = self.ordinalencoder_.transform(X)
        
        
        if self.handle_unknown in ('ignore', 'error'):
            #ordinal encoder returns a category called nan instead of ordinal index when nans are passed in input
            onehot_categories = []
            for cats in self.ordinalencoder_.categories_:
                if self._nan_msk(cats).sum() == 0:
                    onehot_categories.append(np.arange(0, len(cats)))
                else:
                    cts = np.arange(0, len(cats)).astype(cats.dtype)
                    cts[-1] = np.nan
                    onehot_categories.append(cts)                
        else:
            #ordinal encoder returns a category called nan instead of ordinal index when nans are passed in input
            onehot_categories = []
            for cats in self.ordinalencoder_.categories_:
                if self._nan_msk(cats).sum() == 0:
                    onehot_categories.append(np.arange(-1, len(cats)))
                else:
                    cts = np.arange(-1, len(cats)).astype(cats.dtype)
                    cts[-1] = np.nan
                    onehot_categories.append(cts)                
        
        onehot_handle = 'error' if (self.handle_unknown == 'error') else 'ignore'
        self.onehotencoder_ = OneHotEncoder(
            categories = onehot_categories,
            drop = self.drop,
            sparse = self.sparse,
            dtype = self.dtype,
            handle_unknown = onehot_handle
        ).fit(np.zeros((1, X.shape[1]))) #fit with toy dataset since categories are passed
        
        return self
    
    def transform(self, X, **kwargs):
        #check array
        X = check_array(X, dtype = None, force_all_finite = 'allow-nan')
        check_is_fitted(self)
        X = self.ordinalencoder_.transform(X)
        X = self.onehotencoder_.transform(X)
        return X