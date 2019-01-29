from sklearn.utils import shuffle
import numpy as np
import math
import Utils

np.random.seed(0)


class MLP:
    def __init__(self, inputs, inputs_labels, input_validation=None, input_validation_labels=None, num_iterations=20,
                 binary=True, learning_rate=0.01, alpha=0.5, num_nodes_hidden_layer=5,
                 num_output_layers=1, batch_train=True, verbose=False):

        self.original_inputs = inputs
        self.inputs_labels = inputs_labels

        self.input_validation = input_validation
        self.input_validation_labels = input_validation_labels

        self.alpha = alpha
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.num_inputs = np.shape(inputs)[0]
        self.num_data = np.shape(inputs)[1]
        self.num_hidden_nodes_layer_1 = num_nodes_hidden_layer
        self.num_output_layers = num_output_layers
        self.batch_train = batch_train

        self.inputs_with_bias = np.vstack((inputs, np.ones(inputs.shape[1])))

        if input_validation_labels is not None:
            self.input_validation_with_bias = np.vstack(
                (self.input_validation, np.ones(self.input_validation.shape[1])))

        self.verbose = verbose
        self.binary = binary
        self.mse = np.zeros(num_iterations)
        self.validation_mse = np.zeros(num_iterations)

    def initialize_weights(self, num_of_nodes_in_layer, num_of_inputs_in_neuron):
        # Need to add one one weight for bias term
        # from the book :--->  a common trick is to set the weights in the range −1 /√n < w < 1 /√n,
        # where n is the number of nodes in the input layer

        min = -1 / math.sqrt(num_of_inputs_in_neuron)
        max = 1 / math.sqrt(num_of_inputs_in_neuron)
        return np.random.normal(min, max, size=(num_of_inputs_in_neuron, num_of_nodes_in_layer + 1))

    def fit(self):

        self.weights_layer_1 = self.initialize_weights(self.num_inputs, self.num_hidden_nodes_layer_1)
        self.weights_layer_2 = self.initialize_weights(self.num_hidden_nodes_layer_1, self.num_output_layers)

        self.delta_weights_1 = 0
        self.delta_weights_2 = 0

        for epoch in range(self.num_iterations):
            self.epoch = epoch
            if self.batch_train is False:
                self.inputs_with_bias, self.inputs_labels = shuffle(self.inputs_with_bias.T, self.inputs_labels)
                self.inputs_with_bias = self.inputs_with_bias.T

            for idx in range(self.num_data):
                [data, labels] = self._fetch_data(idx)

                h_out, o_out = self.forward_pass(data)

                delta_h, delta_o = self.backwards_pass(labels, h_out, o_out)

                self.update_weights(data, delta_h, delta_o, h_out)

                # if batch train all data then break
                if self.batch_train:
                    break

            o_out = self.forward_pass(self.inputs_with_bias)[1]

            if self.binary is False and epoch % 50 == 0:
                num_data = int(math.sqrt(len(o_out)))
                xx = np.arange(-5, 5, 0.5)
                yy = np.arange(-5, 5, 0.5)
                X, Y = np.meshgrid(xx, yy)
                Z = np.reshape(o_out, (num_data, num_data))
                Utils.plot_3d_data(X, Y, Z)

            self.mse[epoch] = self.evaluate(o_out, self.inputs_labels)[1]

            if self.input_validation_with_bias is not None:
                validation_pred = self.forward_pass(self.input_validation_with_bias)[1]
                self.validation_mse[epoch] = self.evaluate(validation_pred, self.input_validation_labels)[1]

            # # Make a prediction on training data with the current weights
            # _, predictions = self.forward_pass(self.inputs_with_bias, weights_layer_1, weights_layer_2)
            # [loss, mse] = Utils.compute_error(self.inputs_labels, predictions)
            #
            # print('after epoch {0} produced loss {1} and mse {1}'.format(epoch, loss, mse))

        return [self.weights_layer_1, self.weights_layer_2, self.mse]

    def predict(self, test_input):

        if test_input.shape[0] != self.inputs_with_bias.shape[0]:
            if np.ndim(test_input) == 1:
                test_input = np.reshape(test_input, (test_input.shape[0], 1))
            test_input = np.vstack((test_input, np.ones(test_input.shape[1])))

        o_out = self.forward_pass(test_input)[1]

        return o_out

    def evaluate(self, o_out, labels):

        [loss, mse] = Utils.compute_error(labels, o_out, self.binary)

        if self.verbose:
            print('epoch {0} produced misclassification rate {1} and mse {2}'.format(self.epoch, loss, mse))

        return [loss, mse]

    def _fetch_data(self, index):

        if self.batch_train:
            data = self.inputs_with_bias
            labels = self.inputs_labels
        else:
            row = self.inputs_with_bias.T[index]
            data = np.reshape(row, (len(row), 1))
            labels = np.array(self.inputs_labels[index])

        return [data, labels]

    def transfer_function(self, inputs):
        return (2 / (1 + np.exp(-inputs))) - 1

    def transfer_function_derivative(self, inputs):
        return np.multiply((1 + self.transfer_function(inputs)), (1 - self.transfer_function(inputs))) / 2

    def forward_pass(self, inputs):

        bias = np.ones(inputs.shape[1])

        # summed input signal Σxi * w1
        hidden_in = np.dot(inputs.T, self.weights_layer_1.T).T
        # output signal hj = φ( h ∗j )
        hidden_out = np.vstack([self.transfer_function(hidden_in), bias]).T

        # summed input signal Σxi * w2
        output_in = np.dot(hidden_out, self.weights_layer_2.T)
        # output signal
        output_out = self.transfer_function(output_in)

        return hidden_out, output_out

    def backwards_pass(self, targets, h_out, o_out):

        # compute output layer delta
        # δ(o) = (ok − tk) * φ′(o∗k)
        delta_o = np.multiply((o_out - targets), self.transfer_function_derivative(o_out))

        # compute hidden layer delta
        delta_h = np.dot(delta_o, self.weights_layer_2) * self.transfer_function_derivative(h_out)

        # remove the extra row that we previously added to the forward pass to take care of the bias term.
        delta_h = delta_h[:, :self.num_hidden_nodes_layer_1]

        return delta_h, delta_o

    def update_weights(self, inputs, delta_h, delta_o, h_out):

        self.delta_weights_1 = np.multiply(self.delta_weights_1, self.alpha) - np.dot(inputs, delta_h) * (
                1 - self.alpha)
        self.delta_weights_2 = np.multiply(self.delta_weights_2, self.alpha) - np.dot(h_out.T, delta_o) * (
                1 - self.alpha)

        self.weights_layer_1 += self.delta_weights_1.T * self.learning_rate
        self.weights_layer_2 += self.delta_weights_2.T * self.learning_rate


if __name__ == "__main__":

    percent_split = 0.2
    use_validation_set = True
    num_iterations = 100
    learning_rate = 0.001
    num_hidden_nodes_layer_1 = 20
    verbose = False

    [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data(
        use_validation_set=use_validation_set,
        percent_split=percent_split)

    # Utils.plot_initial_data(inputs.T, inputs_labels)



    mlp_seq = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                  input_validation_labels=input_validation_labels, num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                  num_iterations=num_iterations, learning_rate=learning_rate, batch_train=False, verbose=verbose)

    [_, _, mse_seq] = mlp_seq.fit()

    train_mse = mlp_seq.mse
    val_mse = mlp_seq.validation_mse


    # Utils.plot_decision_boundary_mlp(inputs,inputs_labels, mlp_seq)


    mse = [train_mse, val_mse]
    legend_names = ['train_mse error', 'batch val_mse']
    Utils.plot_error(mse, legend_names=legend_names, num_epochs=num_iterations, title='MLP with learning rate ')
