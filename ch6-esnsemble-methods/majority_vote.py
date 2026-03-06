from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from sklearn.base import clone 
from sklearn.pipeline import _name_estimators 
import operator 


class MajorityVoteClassifier(ClassifierMixin, BaseEstimator):
    
    """A majority vote ensemble classifier.
    
    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
        A list of classifiers for the ensemble.
        
    vote : str, {'classlabel', 'probability'}, default='classlabel'
        If 'classlabel' the prediction is based on the argmax of 
        class labels. Else if 'probability', the argmax of the sum of 
        probabilities is used to predict the class label (recommended for 
        calibrated classifiers).
        
    weights : array-like, shape = [n_classifiers], optional, default=None
        If specified, the weights are applied to the classifier votes. 
        The array should be equal to the number of classifiers.
        
    """
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote 
        self.weights = weights 

    def fit(self, X, y):

        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f'vote must be "probability'
                             f'or "classlabel"'
                             f'; got (vote={self.vote})')

        if self.weights and len(self.weights) != len(self.classifiers):
                raise ValueError(f'Number of classifiers and weights must be equal;'
                                 f' got {len(self.weights)} weights,'
                                 f' but {len(self.classifiers)} classifiers')
                # Use LabelEncoder to ensure class labels start with 0 and increase by 1
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []

        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # self.vote == 'classlabel'
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)

        maj_vote = self.lablenc_.inverse_transform(maj_vote)

        return maj_vote


    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)

        return avg_proba

    def get_proba(self, X):
        return self.predict_proba(X)

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out
  
