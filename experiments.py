import Utils
from MLP import MLP


def run_hidden_nodes_mse_plot_experiment():
    percent_split = 0.2
    use_validation_set = False

    [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data(
        use_validation_set=use_validation_set,
        percent_split=percent_split)

    Utils.plot_initial_data(inputs.T, inputs_labels)

    num_iterations = 500
    learning_rate = 0.00001
    verbose = False

    nodes = [1, 5, 10, 20, 30, 40, 50, 70, 80, 90, 100, 200]
    losses = []
    mses = []
    for node in nodes:
        mlp_batch = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                        input_validation_labels=input_validation_labels,
                        num_nodes_hidden_layer=node,
                        num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose)

        [_, _, mse_batch] = mlp_batch.fit()
        out = mlp_batch.predict(inputs)
        [loss, mse] = mlp_batch.evaluate(out)
        losses.append(loss)
        mses.append(mse)

    legend_names = ['mse', 'misclassification']
    Utils.plot_error_hidden_nodes(mses, legend_names=legend_names, hidden_nodes=nodes,
                                  title='MLP with learning rate {0}, iterations {1} '.format(learning_rate,
                                                                                             num_iterations),
                                  loss=losses)


def experiment_train_validation():
    percent_split = 0.2
    use_validation_set = True
    case = 4

    [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data(
        use_validation_set=use_validation_set,
        percent_split=percent_split,case = case)

    # Utils.plot_initial_data(inputs.T, inputs_labels)

    num_hidden_nodes_layer_1 = 20
    num_iterations = 500
    learning_rate = 0.0001
    verbose = True

    mlp_batch = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                    input_validation_labels=input_validation_labels,
                    num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                    num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose)

    [_, _, mse_batch] = mlp_batch.fit()

    # Utils.plot_decision_boundary_mlp(inputs,inputs_labels, mlp_batch)

    # mlp_seq = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
    #               input_validation_labels=input_validation_labels, num_nodes_hidden_layer=num_hidden_nodes_layer_1,
    #               num_iterations=num_iterations, learning_rate=learning_rate, batch_train=False, verbose=verbose)
    #
    # [_, _, mse_seq] = mlp_seq.fit()

    # mse = [mse_seq, mse_batch]

    train_mse = mlp_batch.mse
    val_mse = mlp_batch.validation_mse

    # Utils.plot_decision_boundary_mlp(inputs,inputs_labels, mlp_seq)

    mse = [train_mse, val_mse]
    legend_names = ['train mse error', 'validation mse error']
    Utils.plot_error(mse, legend_names=legend_names, num_epochs=num_iterations,
                     title='MLP with lr = {0}, iterations = {1} , hidden nodes = {2} '
                     .format(learning_rate, num_iterations, num_hidden_nodes_layer_1))


if __name__ == "__main__":
    # run_hidden_nodes_mse_plot_experiment()

    experiment_train_validation()
