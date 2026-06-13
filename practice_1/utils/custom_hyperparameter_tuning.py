"""
Module này dùng để thử nhiều tổ hợp siêu tham số cho mô hình phân loại,
đánh giá từng tổ hợp bằng Stratified Cross Validation, sau đó chọn ra bộ
tham số có điểm trung bình tốt nhất theo một metric được chỉ định.
"""

import copy
import itertools

import numpy as np

try:
    from scipy.sparse import issparse
except ImportError:
    def issparse(X):
        return False


class CustomGridSearchCV:
    """Tìm kiếm siêu tham số tốt nhất cho mô hình phân loại bằng Grid Search."""

    def __init__(self, estimator, param_grid, cv, scoring='f1', pos_label=None):
        """Khởi tạo bộ tìm kiếm siêu tham số.

        Args:
            estimator (object): Mô hình cần tối ưu siêu tham số. Mô hình phải
                có các phương thức `set_params`, `fit` và `predict`.
            param_grid (dict): Tập các siêu tham số cần thử.
            cv (object): Bộ chia cross validation có `n_splits` và `split`.
            scoring (str): Metric dùng để đánh giá. Các giá trị đang hỗ trợ:
                `accuracy`, `precision`, `recall`, `f1`.
            pos_label (object | None): Nhãn được xem là Positive khi tính
                `precision`, `recall`, `f1`. Nếu None, Positive được suy ra
                từ dữ liệu trong từng lần đánh giá.
        """
        if not isinstance(scoring, str):
            raise ValueError("scoring chỉ nhận một metric dạng str")

        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.pos_label = pos_label

        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        self.cv_results_ = []

    def _clone_estimator(self):
        """Tạo bản sao mới của mô hình để tránh rò rỉ trạng thái giữa các fold."""
        return copy.deepcopy(self.estimator)

    def _get_positive_label(self, y_true, y_pred):
        """Xác định lớp dương cho các metric nhị phân."""
        if self.pos_label is not None:
            return self.pos_label

        labels = np.unique(np.concatenate([y_true, y_pred]))
        if len(labels) != 2:
            raise ValueError(
                "Metric precision/recall/f1 cần bài toán nhị phân hoặc "
                "cần truyền pos_label rõ ràng."
            )

        return labels[-1]

    def _binary_confusion_counts(self, y_true, y_pred):
        """Tính TP, FP, FN theo lớp dương đã chọn hoặc suy ra."""
        pos_label = self._get_positive_label(y_true, y_pred)
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        return tp, fp, fn

    def _score(self, y_true, y_pred):
        """Tính điểm classification theo metric đã chọn."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        if self.scoring == 'accuracy':
            return np.mean(y_true == y_pred)

        if self.scoring == 'precision':
            tp, fp, _ = self._binary_confusion_counts(y_true, y_pred)
            denominator = tp + fp
            return tp / denominator if denominator > 0 else 0.0

        if self.scoring == 'recall':
            tp, _, fn = self._binary_confusion_counts(y_true, y_pred)
            denominator = tp + fn
            return tp / denominator if denominator > 0 else 0.0

        if self.scoring == 'f1':
            tp, fp, fn = self._binary_confusion_counts(y_true, y_pred)
            denominator = (2 * tp + fp + fn)
            return (2 * tp) / denominator if denominator > 0 else 0.0

        raise ValueError(f"Metric không được hỗ trợ cho classification: {self.scoring}")

    def fit(self, X, y):
        """Thực hiện Grid Search và huấn luyện mô hình tốt nhất."""
        is_sparse = issparse(X)
        y = np.array(y) if not hasattr(y, 'iloc') else y

        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        self.cv_results_ = []

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"Bắt đầu GridSearchCV: {len(combinations)} tổ hợp tham số, {self.cv.n_splits} folds.")

        for idx, params in enumerate(combinations):
            fold_scores = []

            for train_idx, val_idx in self.cv.split(X, y):
                if is_sparse:
                    X_train, X_val = X[train_idx], X[val_idx]
                else:
                    X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                    X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]

                y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

                estimator = self._clone_estimator()
                estimator.set_params(**params)
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_val)

                fold_scores.append(self._score(y_val, y_pred))

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            result = {
                'params': params,
                f'mean_test_{self.scoring}': mean_score,
                f'std_test_{self.scoring}': std_score,
            }
            self.cv_results_.append(result)

            print(
                f"[{idx + 1}/{len(combinations)}] Params: {params} "
                f"--> {self.scoring}: {mean_score:.4f}"
            )

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

        print(f"\n-> Tham số TỐT NHẤT: {self.best_params_}")
        print(f"-> Điểm {self.scoring} TỐT NHẤT: {self.best_score_:.4f}")

        self.best_estimator_ = self._clone_estimator()
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self
