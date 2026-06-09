import numpy as np
import itertools
from scipy.sparse import issparse

class CustomGridSearchCV:
    def __init__(self, estimator, param_grid, cv, scoring='f1'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        
        self.best_params_ = None
        self.best_score_ = -np.inf 
        self.best_estimator_ = None
        self.cv_results_ = []      

    def _score(self, y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if self.scoring == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
        elif self.scoring == 'accuracy':
            return np.mean(y_true == y_pred)
            
        elif self.scoring == 'f1':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            # F1 = 2*TP / (2*TP + FP + FN)
            denominator = (2 * tp + fp + fn)
            return (2 * tp) / denominator if denominator > 0 else 0.0
            
        return 0

    def fit(self, X, y):
        is_sparse = issparse(X)
        y = np.array(y) if not hasattr(y, 'iloc') else y
        
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"Bắt đầu GridSearchCV: {len(combinations)} tổ hợp tham số, {self.cv.n_splits} folds.")
        
        for idx, params in enumerate(combinations):
            self.estimator.set_params(**params)
            fold_scores = []
            
            for train_idx, val_idx in self.cv.split(X, y):
                if is_sparse:
                    X_train, X_val = X[train_idx], X[val_idx]
                else:
                    X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                    X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                
                y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
                
                self.estimator.fit(X_train, y_train)
                y_pred = self.estimator.predict(X_val)
                score = self._score(y_val, y_pred)
                fold_scores.append(score)
            
            mean_score = np.mean(fold_scores)
            self.cv_results_.append({'params': params, 'mean_test_score': mean_score})
            
            print(f"[{idx+1}/{len(combinations)}] Params: {params} --> {self.scoring}: {mean_score:.4f}")
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        
        print(f"\n-> Tham số TỐT NHẤT: {self.best_params_}")
        print(f"-> Điểm {self.scoring} TỐT NHẤT: {self.best_score_:.4f}")
        
        # Train model on all Training set
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y)
        self.best_estimator_ = self.estimator
        return self