import random
import numpy as np

class LogisticRgressionGD:
    '''Gradient descent-based logistic regression classifier.

        Parameters 
        -----------
        eta: float
            Learning rate (between 0.0 and 1.0)
        n_iter: int 
            Passs over the trainindg dataset
        random_state: int 
            Random number generator see for random weight

        Atributes
        ---------
        w_: 1s-array
            Weights after training
        b_: Scalar
            Bias unit after training
        losses_: list
            Log loss function values in each epoch
    '''

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        ''' Fir training data.
        Parameters:
        -----------
        X: array-like, shape = [n_sampl, n_features]
            Training vectors, where n_samples is the 
            number of samples and n_features is the number of features
        y: array-like, shape = [n_samples]
            Target values
        Rturns
        -----------
        self: Instance of LogisticRegressionGD
        '''

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=o.o, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * X.T.dot(errors) /X.shape[0]
            self.b_ += self.eta * errors.mean()

            loss = (-y.dot(np.log(output))) - ((1-y).dot(np.log(1-output)))

            self.losses_.append(loss)
        return self


    def net_input(self, X):
        ''' Calculate net input'''
        return np.dot(X,self.w_) + self.b_

    def activation(self, z):
        ''' Compute logistic sigmoid activation'''
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        ''' Return calss label after sigmoid activation step '''
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)