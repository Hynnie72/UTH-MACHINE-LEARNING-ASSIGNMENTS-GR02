import numpy as np


class CustomStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """Khởi tạo bộ chia Stratified K-Fold Cross Validation.

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
        """Chia dữ liệu sao cho mỗi fold giữ tỷ lệ nhãn gần giống nhau."""
        y = np.array(y) if not hasattr(y, 'iloc') else y.values

        # Tìm các nhãn duy nhất (Ví dụ: ham/spam hoặc 0/1)
        unique_classes = np.unique(y)

        # Tạo sẵn k mảng rỗng (k thùng) để chứa index
        folds = [[] for _ in range(self.n_splits)]

        rng = np.random.RandomState(self.random_state)

        # Rải đều từng nhãn vào các thùng
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]

            if self.shuffle:
                rng.shuffle(cls_indices)

            # Rải từng index vào các thùng theo thứ tự để giữ tỷ lệ nhãn
            for i, idx in enumerate(cls_indices):
                folds[i % self.n_splits].append(idx)

        for i in range(self.n_splits):
            val_indices = np.array(folds[i])
            train_indices = np.concatenate([
                folds[j] for j in range(self.n_splits) if j != i
            ])

            yield train_indices, val_indices


# Alias này giúp notebook cũ chỉ cần đổi import tối thiểu.
CustomKFold = CustomStratifiedKFold
