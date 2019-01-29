import Utils
from MLP import MLP

if __name__ == "__main__":
    [data, labels] = Utils.create_one_out_of_n_dataset()

    num_hidden_nodes_layer_1 = 3
    num_output_layers = 8
    num_iterations = 10000
    learning_rate = 0.01

    mlp_batch = MLP(inputs=data, inputs_labels=data, num_output_layers=num_output_layers, learning_rate=learning_rate,
                    num_nodes_hidden_layer=num_hidden_nodes_layer_1, num_iterations=num_iterations, batch_train=True
                    ,verbose=True)

    [_, _, mse_batch] = mlp_batch.fit()

    mse = [mse_batch]

    legend_name = ['batch error']
    # Utils.plot_error(mse, legend_name, num_epochs=num_iterations)
