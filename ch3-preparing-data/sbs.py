
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:
    """Sequential Backward Selection (SBS) feature selector.

    SBS starts with all input features and iteratively removes one feature at a
    time. At each step, it evaluates all candidate subsets of size
    ``current_dim - 1`` using the configured estimator and scoring function, and
    keeps the subset with the best score. The process stops when
    ``k_features`` remain.

    Attributes set after ``fit``:
    - ``indices_``: tuple of selected feature indices at the final step.
    - ``subsets_``: list of feature-index tuples for each SBS step.
    - ``scores_``: list of scores corresponding to ``subsets_``.
    - ``k_score_``: score of the final subset with ``k_features`` features.
    """

    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        """Initialize the SBS selector.

        Parameters
        ----------
        estimator : estimator object
            A scikit-learn compatible estimator implementing ``fit`` and
            ``predict``.
        k_features : int
            Target number of features to retain.
        scoring : callable, default=accuracy_score
            Scoring function with signature ``scoring(y_true, y_pred)``.
        test_size : float, default=0.25
            Fraction of samples used for the holdout evaluation set.
        random_state : int, default=1
            Random seed used by ``train_test_split`` for reproducibility.
        """
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """Run sequential backward selection on the provided dataset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        SBS
            Fitted selector instance.
        """
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        """Reduce input matrix to the selected feature subset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray
            Matrix containing only the columns selected by SBS.
        """
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        """Fit estimator on selected columns and return holdout score.

        Parameters
        ----------
        X_train, X_test : ndarray
            Training and test feature matrices.
        y_train, y_test : ndarray
            Training and test labels.
        indices : tuple[int, ...]
            Feature-column indices to evaluate.

        Returns
        -------
        float
            Score computed by ``self.scoring`` on holdout predictions.
        """
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score



