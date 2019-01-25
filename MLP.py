import numpy as np
import math
from sklearn.metrics import mean_squared_error, zero_one_loss
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(0)


class MLP:
    def __init__(self, inputs, targets, num_iterations=20, learning_rate=0.001, alpha=0.9,
                 num_hidden_nodes_layer_1=5, num_output_layers=1):
        self.inputs = inputs
        self.targets = targets

        self.alpha = alpha
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.num_inputs = np.shape(inputs)[0]
        self.num_hidden_nodes_layer_1 = num_hidden_nodes_layer_1
        self.num_output_layers = num_output_layers

    def initialize_weights(self, num_of_nodes_in_layer, num_of_inputs_in_neuron):
        # Need to add one one weight for bias term
        # from the book :--->  a common trick is to set the weights in the range −1 /√n < w < 1 /√n,
        # where n is the number of nodes in the input layer

        min = -1 / math.sqrt(num_of_inputs_in_neuron)
        max = 1 / math.sqrt(num_of_inputs_in_neuron)
        return np.random.normal(min, max, size=(num_of_inputs_in_neuron, num_of_nodes_in_layer + 1))

    def train_batch(self):
        # Weights for first layer
        weights_layer_1 = self.initialize_weights(self.num_inputs, self.num_hidden_nodes_layer_1)

        # Weights for second layer
        weights_layer_2 = self.initialize_weights(self.num_hidden_nodes_layer_1, self.num_output_layers)

        delta_weights_1 = 0
        delta_weights_2 = 0

        for epoch in range(self.num_iterations):
            h_out, o_out = self.forward_pass(self.inputs, weights_layer_1, weights_layer_2)


            [loss, mse] = self.compute_error(self.targets, o_out)
            print('epoch {0} produced misclassification rate {1} and mse {1}'.format(epoch, loss, mse))

            delta_h, delta_o = self.backwards_pass(self.targets, h_out, o_out, weights_layer_2)

            weights_layer_1, weights_layer_2, delta_weights_1, delta_weights_2 = self.update_weights(self.inputs,
                            weights_layer_1, weights_layer_2, delta_weights_1, delta_weights_2, delta_h, delta_o, h_out)

            # # Make a prediction on training data with the current weights
            # _, predictions = self.forward_pass(self.inputs, weights_layer_1, weights_layer_2)
            # [loss, mse] = self.compute_error(self.targets, predictions)
            #
            # print('after epoch {0} produced loss {1} and mse {1}'.format(epoch, loss, mse))

        return [weights_layer_1, weights_layer_2]

    def transfer_function(self, inputs):
        return (2 / (1 + np.exp(-inputs))) - 1

    def transfer_function_derivative(self, inputs):
        return np.multiply((1 + self.transfer_function(inputs)), (1 - self.transfer_function(inputs))) / 2

    def forward_pass(self, inputs, weights_layer_1, weights_layer_2):
        bias = np.ones(inputs.shape[1])
        inputs = np.vstack((inputs, bias))

        # summed input signal Σxi * w1
        hidden_in = np.dot(inputs.T, weights_layer_1.T).T
        # output signal hj = φ( h ∗j )
        hidden_out = np.vstack([self.transfer_function(hidden_in), bias]).T

        # summed input signal Σxi * w2
        output_in = np.dot(hidden_out, weights_layer_2.T)
        # output signal
        output_out = self.transfer_function(output_in)

        return hidden_out, output_out

    def backwards_pass(self, targets, h_out, o_out, weights_layer_2):

        # compute output layer delta
        # δ(o) = (ok − tk) * φ′(o∗k)
        delta_o = np.multiply((o_out - targets), self.transfer_function_derivative(o_out))

        # compute hidden layer delta
        delta_h = np.dot(delta_o, weights_layer_2) * self.transfer_function_derivative(h_out)

        # remove the extra row that we previously added to the forward pass to take care of the bias term.
        delta_h = delta_h[:, :self.num_hidden_nodes_layer_1]

        return delta_h, delta_o

    def update_weights(self, inputs, weights_layer_1, weights_layer_2, delta_weights_1, delta_weights_2, delta_h,
                       delta_o, h_out):
        bias = np.ones(inputs.shape[1])
        inputs = np.vstack((inputs, bias))

        delta_weights_1 = np.multiply(delta_weights_1, self.alpha) - np.dot(inputs, delta_h) * (1 - self.alpha)
        delta_weights_2 = np.multiply(delta_weights_2, self.alpha) - np.dot(h_out.T, delta_o) * (1 - self.alpha)

        weights_layer_1 += delta_weights_1.T * self.learning_rate
        weights_layer_2 += delta_weights_2.T * self.learning_rate

        return [weights_layer_1, weights_layer_2, delta_weights_1, delta_weights_2]

    def compute_error(self, targets, predictions):
        mse = mean_squared_error(targets, predictions)

        predictions = np.where(predictions >= 0, 1, -1)

        # fraction of misclassifications
        loss = zero_one_loss(targets, predictions, normalize=True)

        return loss, mse


def create_non_linearly_separable_data(n=100):
    meanA = [1, 0.5]
    meanB = [-0.5, -1]

    sigmaA = np.diag([1, 1])
    sigmaB = np.diag([1, 1])

    classA = np.random.multivariate_normal(meanA, sigmaA, n)
    labelsA = -np.ones((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, n)
    labelsB = np.ones((classB.shape[0], 1))

    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    [X, Y] = shuffle(X, Y)

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return [X.T, Y]


def plot_initial_data(inputs, targets):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1], s=15)
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1], s=15)

    plt.ylim(-10, 10)
    plt.xlim(-6, 6)

    plt.show()


[X, Y] = create_non_linearly_separable_data()

# plot_initial_data(X.T, Y)

num_hidden_nodes_layer_1 = 30
num_output_layers = 1

mlp = MLP(inputs=X, targets=Y, num_hidden_nodes_layer_1=num_hidden_nodes_layer_1,
          num_output_layers=num_output_layers)

mlp.train_batch()
