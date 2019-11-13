import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_array


class StratifiedGroupKfold(_BaseKFold):
    """K-fold iterator variant with non-overlapping groups
    combination which also attempts. to evenly distribute the number of each
    classes across each fold.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    weighted : bool, default=True
        If True, the data will be distributed by its size. If False, the
        weight of each group and label combination is ignored.
    constrain_groups:  bool, default=True
        if True, groups are constrained to be in a single fold. if False, the
        same group can be in in multiple folds, but the same group and label
        combination can not be across multiple folds.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKfold
    >>> groups = np.array([3, 0, 3, 0, 3, 3, 2, 0, 1])
    >>> y =      np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    >>> X =      np.array([1 ,2 ,3 ,4 ,5 ,6 ,6 ,6, 7])
    >>> sgkf = StratifiedGroupKfold(n_splits=2)
    >>> sgkf.get_n_splits(X, y, groups)
    2
    >>> for train_index, test_index in sgkf.split(X, y, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...     print(X_train, X_test, y_train, y_test)
    ...
        TRAIN: [1 3 6 7] TEST: [0 2 4 5 8]
        [2 4 6 6] [1 3 5 6 7] [1 1 2 2] [1 1 1 2 2]
        TRAIN: [0 2 4 5 8] TEST: [1 3 6 7]
        [1 3 5 6 7] [2 4 6 6] [1 1 1 2 2] [1 1 2 2]
    See also
    --------
    LeaveOneGroupOut
        For splitting the data according to explicit domain-specific
        stratification of the dataset.
    GroupKfold
        For splitting the data according to explicit domain-specific
        stratification of the dataset.
    """

    def __init__(self, n_splits=5, constrain_groups=True, weighted=True):
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.weighted = weighted
        self.constrain_groups = constrain_groups

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        unique_y = np.unique(y)

        label_to_fold = []
        fold_groups = [[] for x in range(self.n_splits)]

        for label in unique_y:
            # Weight labels by their number of occurrences
            tmp_groups = groups[y == label]

            n_samples_per_group = np.bincount(tmp_groups)

            # Distribute the most frequent labels first
            indices = np.argsort(n_samples_per_group)[::-1]
            n_samples_per_group = n_samples_per_group[indices]

            # Total weight of each fold
            n_samples_per_fold = np.zeros(self.n_splits)

            # Mapping from group index to fold index
            groups_in_fold = {x: [] for x in range(self.n_splits)}

            # Distribute samples, add the largest weight to the lightest fold
            for group_index, weight in enumerate(n_samples_per_group):

                if not weight:
                    continue

                # calculate the lightest fold for the next to be placed
                lightest_fold = lightest_fold = np.argmin(n_samples_per_fold)

                # do a check to see if the group has already been placed
                # if it has, set the lightest_fold to be the fold_index
                if self.constrain_groups:
                    for fold_index, fold_group in enumerate(fold_groups):
                        if indices[group_index] in fold_group:
                            lightest_fold = fold_index

                if self.weighted:
                    n_samples_per_fold[lightest_fold] += weight
                else:
                    n_samples_per_fold[lightest_fold] += 1

                groups_in_fold[lightest_fold].append(indices[group_index])
                fold_groups[lightest_fold].append(indices[group_index])

            label_to_fold.append((label, groups_in_fold))

        for f in range(self.n_splits):
            indices = []
            for label, groups_in_fold in label_to_fold:
                label_indices = np.argwhere(y == label)

                # isin introduced numpy 1.13
                if hasattr(np, 'isin'):
                    group_indices = np.argwhere(
                        np.isin(groups, groups_in_fold[f]))
                else:
                    group_indices = np.argwhere(
                        np.array(
                            [item in groups_in_fold[f] for item in groups]))

                indices.extend(np.intersect1d(label_indices, group_indices))

            yield indices

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)


