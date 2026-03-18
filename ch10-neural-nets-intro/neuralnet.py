import numpy as np 

num_epochs = 50
minibatch_size = 100


def sigmoid(z):
    """Compute the logistic sigmoid activation element-wise."""
    return 1.0 / (1.0 + np.exp(-z))

def one_hot(y, numlabels):
    """Convert integer class labels to a one-hot encoded matrix."""
    ary = np.zeros((y.shape[0], numlabels))
    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary

def minibatch_generator(X, y, minibatch_size):
    """Yield shuffled mini-batches of features and labels."""
    indices = np.arange(X.shape[0])

    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1 , minibatch_size):

        batch_idx = indices[start_idx:start_idx+minibatch_size]
        yield X[batch_idx], y[batch_idx]

def mse_loss(targets, probas, num_labels=10):
    """Compute mean squared error between one-hot targets and predictions."""
    onehot_targets = one_hot(
            targets, numlabels=num_labels
            )

    return np.mean((onehot_targets - probas) ** 2)


def accuracy(targets, predicted_labels):
    """Return classification accuracy from true and predicted labels."""

    return np.mean(predicted_labels == targets)

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    """Evaluate model performance over mini-batches using MSE and accuracy."""

    mse, correct_pred, num_examples = 0., 0, 0

    minibatch_gen = minibatch_generator(X, y, minibatch_size=minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = one_hot(targets, numlabels=num_labels)

        loss = np.mean((onehot_targets- probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss 
    mse = mse/i 
    acc = correct_pred / num_examples
    return mse, acc


def train(model,
          X_train,
          y_train,
          X_valid,
          y_valid, num_epochs, learning_rate=0.1):
    """Train the model with mini-batch gradient descent and report metrics."""
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):

        # Iterate over minibatches

        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:

            a_h, a_out = model.forward(X_train_mini)

            d_loss__d_w_out, d_loss__d_b_out,\
            d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini, a_h, a_out, y_train_mini)

            # Update weights
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out


            # Epoch logging 
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        _, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)

        print(f'Epoch {e+1:03d}/{num_epochs:03d}'
                f'| Train MSE: {train_mse:.2f}'
                f'| Train Acc: {train_acc:.2f}'
                f'| Validation ACC: {valid_acc:.2f}'
                  )

    return epoch_loss, epoch_train_acc, epoch_valid_acc

    






class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes,
                 random_seed=123) -> None:
        """Initialize network weights and biases for a one-hidden-layer MLP."""
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden) )
        self.bias_out = np.zeros(num_classes)



    def forward(self, x):
        """Run a forward pass and return hidden and output activations."""

        # Hidden layer 

        # input dim [n_examples, num_features]
        #    dot    [num_hidden, num_features].T
        # output dim [num_examples, num_hidden]

        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer 
        # input dim [num_examples, num_hidden]
        #       dot [num_classes, num_hidden].T 
        # output dim [num_examples, num_classes]

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        """Compute gradients for output and hidden layer parameters."""

        y_onehot = one_hot(y, self.num_classes)

        d_loss__d_a_out = 2. * (a_out - y_onehot)/y.shape[0]

        d_a_out__d_z_out = a_out*(1. - a_out)

        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_z_out__dw_out = a_h

        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_z_out_a_h = self.weight_out

        d_loss__d_a_h = np.dot(delta_out, d_z_out_a_h)


        d_a_h__d_z_h = a_h * (1. - a_h)

        d_z_h__d_w_h = x 

        d_loss__d_w_h = np.dot((d_loss__d_a_h * d_a_h__d_z_h).T, 
                               d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__d_a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)









        
