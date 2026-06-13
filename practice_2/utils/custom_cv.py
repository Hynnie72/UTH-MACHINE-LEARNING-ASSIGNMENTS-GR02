import numpy as np


class CustomKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """Khởi tạo bộ chia K-Fold Cross Validation thông thường.

        Args:
            n_splits (int): Số lượng fold (thường là 5 hoặc 10)
            shuffle (bool): Có xáo trộn dữ liệu trước khi chia hay không
            random_state (int): Seed ngẫu nhiên để kết quả có thể tái lập
        """
        if n_splits <= 1:
            raise ValueError("Số lượng splits phải lớn hơn 1")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """Chia dữ liệu thành các cặp chỉ số train/validation.

        K-Fold thường không dùng giá trị của y để chia, nên phù hợp hơn cho
        bài toán hồi quy có nhãn liên tục.
        """
        n_samples = len(y)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            current = stop

            yield train_indices, val_indices
