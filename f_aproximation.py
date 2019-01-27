import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt
import Utils

def run(X,Y,Z):

    targets = np.reshape(Z, (1, (len(X) * len(Y)))).T

    num_data = len(X) * len(Y)

    X = np.reshape(X, (1, (num_data)))
    Y = np.reshape(Y, (1, (num_data)))

    patterns = np.vstack((X, Y))

    num_hidden_nodes_layer_1 = 20
    num_iterations = 5000
    learning_rate = 0.001
    verbose = False

    mlp_batch = MLP(inputs=patterns, inputs_labels=targets, num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                    num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose,
                    binary=False, num_output_layers=1)

    [w1, w2, mse_batch] = mlp_batch.train()

    out = mlp_batch.predict()
    return out

if __name__ == "__main__":
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.exp(-(np.square(X) + np.square(Y)) / 10) - 0.5

    print(Z.shape)
    Utils.plot_3d_data(X, Y, Z, pause=False)


    n_X = len(X)
    n_Y = len(Y)
    out = run(X,Y,Z)

    Z = np.reshape(out, (n_X, n_Y))


    # Utils.plot_3d_data(X, Y, Z)




