from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor


#lgbm custom estimators
class CustomLGBMClassifier(LGBMClassifier):
    
    def apply(self, X):
        return self.predict(X, pred_leaf = True)
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight, **kwargs)
        model_df = self.booster_.trees_to_dataframe()
        node_weights = model_df[model_df["decision_type"].isna()]["weight"].values
        self.node_weights_ = node_weights
        return self
    
class CustomLGBMRegressor(LGBMRegressor):
    
    def apply(self, X):
        return self.predict(X, pred_leaf = True)
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight, **kwargs)
        model_df = self.booster_.trees_to_dataframe()
        node_weights = model_df[model_df["decision_type"].isna()]["weight"].values
        self.node_weights_ = node_weights
        return self

class CustomLGBMRanker(LGBMRanker):
    
    def apply(self, X):
        return self.predict(X, pred_leaf = True)
    
    def fit(self, X, y = None, sample_weight = None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight, **kwargs)
        model_df = self.booster_.trees_to_dataframe()
        node_weights = model_df[model_df["decision_type"].isna()]["weight"].values
        self.node_weights_ = node_weights
        return self

    
