from sklearn.model_selection import train_test_split

import Utils
from MLP import MLP
import numpy as np

from Perceptron import perceptron


def run_hidden_nodes_mse_plot_experiment():
    use_validation_set = False

    [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data_2(
        use_validation_set=use_validation_set)

    Utils.plot_initial_data(inputs.T, inputs_labels)

    num_iterations = 1000
    learning_rate = 0.002
    verbose = True

    nodes = [1, 5, 10, 20, 30, 40, 50, 70, 80, 90, 100, 200, 250]
    nodes = np.arange(1,50,1)
    losses = []
    mses = []
    for node in nodes:
        mlp_batch = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                        input_validation_labels=input_validation_labels,
                        num_nodes_hidden_layer=node,
                        num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose)

        [_, _, mse_batch] = mlp_batch.fit()
        out = mlp_batch.predict(inputs)
        [loss, mse] = mlp_batch.evaluate(out, inputs_labels)
        losses.append(loss)
        mses.append(mse)

    legend_names = ['mse', 'misclassification']
    Utils.plot_error_hidden_nodes(mses, legend_names=legend_names, hidden_nodes=nodes,
                                  title='MLP with learning rate {0}, iterations {1} '.format(learning_rate,
                                                                                             num_iterations),
                                  loss=losses)


def experiment_train_validation_error():
    use_validation_set = True
    num_hidden_nodes_layer_1 = 20
    num_iterations = 1000
    learning_rate = 0.002
    verbose = False

    cases = [1, 2, 3, 4]
    cases = [1, 2, 3, 4]
    train_MSE = []
    val_MSE = []
    mse = []
    for case in cases:
        [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data_2(
            use_validation_set=use_validation_set, case=case)

        # Utils.plot_initial_data(inputs.T, inputs_labels)



        mlp_batch = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                        input_validation_labels=input_validation_labels,
                        num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                        num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose)

        [_, _, mse_batch] = mlp_batch.fit()

        mse.append(mlp_batch.mse)
        mse.append(mlp_batch.validation_mse)


    legend_names = ['train mse error case 1','validation mse error case 1',
                    'train mse error case 2', 'validation mse error case 2',
                     'train mse error case 3', 'validation mse error case 3',
                    'train mse error case 4', 'validation mse error case 4']

    Utils.plot_error_with_epochs(mse, legend_names=legend_names, num_epochs=num_iterations,
                                 title='MLP with lr = {0}, iterations = {1} , hidden nodes = {2} '
                                 .format(learning_rate, num_iterations, num_hidden_nodes_layer_1))



def experiment_learning_curves_error():
    train_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


    use_validation_set = True
    num_hidden_nodes_layer_1 = 20
    num_iterations = 1000
    learning_rate = 0.001
    verbose = False

    cases = [1, 2, 3, 4]
    train_MSE = []
    val_MSE = []
    for case in cases:
        [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data_2(
            use_validation_set=use_validation_set, case=case)

        print(case)

        current_train = []
        current_validation = []
        for check in train_test:

            X_train, X_test, y_train, y_test = train_test_split(inputs.T, inputs_labels, test_size=check, random_state=42)

            mlp_batch = MLP(inputs=X_train.T, inputs_labels=y_train, input_validation=input_validation,
                                input_validation_labels=input_validation_labels,
                                num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                                num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True,
                                verbose=verbose)

            [_, _, mse_batch] = mlp_batch.fit()

            current_train.append(mlp_batch.mse[-1])
            current_validation.append(mlp_batch.validation_mse[-1])

        train_MSE.append(current_train)
        val_MSE.append(current_validation)


    legend_names = ['train mse error case 1', 'train mse error case 2',
                    'train mse error case 3', 'train mse error case 4',
                    'validation mse error case 1', 'validation mse error case 2',
                    'validation mse error case 3', 'validation mse error case 4']

    Utils.plot_learning_curves(train_MSE, legend_names=legend_names, train_size=train_test,
                                  title='Learning curve with lr = {0}, iterations = {1} '
                                  .format(learning_rate, num_iterations), loss=val_MSE)


def experiment_train_validation_nodes():
    use_validation_set = True

    num_iterations = 1000
    learning_rate = 0.002
    verbose = False

    nodes = [1, 5, 10, 20, 25]
    cases = [1, 2, 3, 4]
    train_MSE = []
    val_MSE = []

    for case in cases:
        print(case)
        [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data_2(
            use_validation_set=use_validation_set, case=case)

        current_mse = []
        current_val_mse = []
        for node in nodes:
            mlp_batch = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                            input_validation_labels=input_validation_labels,
                            num_nodes_hidden_layer=node,
                            num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True,
                            verbose=verbose)

            [_, _, mse_batch] = mlp_batch.fit()

            current_mse.append(mlp_batch.mse[-1])
            current_val_mse.append(mlp_batch.validation_mse[-1])

        train_MSE.append(current_mse)
        val_MSE.append(current_val_mse)

    legend_names = ['train mse error case 1', 'train mse error case 2',
                    'train mse error case 3', 'train mse error case 4',
                    'validation mse error case 1','validation mse error case 2',
                    'validation mse error case 3', 'validation mse error case 4']

    Utils.plot_error_hidden_nodes(train_MSE, legend_names=legend_names, hidden_nodes=nodes,
                                  title='MLP with learning rate {0}, iterations {1} '
                                  .format(learning_rate, num_iterations),loss=val_MSE)


def experiment_train_val_seq_batch_mlp():
    use_validation_set = False
    case = 1

    [inputs, inputs_labels, input_validation, input_validation_labels] = Utils.create_non_linearly_separable_data_2(
        use_validation_set=use_validation_set, case=case)

    num_hidden_nodes_layer_1 = 20
    num_iterations = 1000
    learning_rate = 0.002
    verbose = False

    mlp_batch = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                    input_validation_labels=input_validation_labels,
                    num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                    num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose)

    [_, _, mse_batch] = mlp_batch.fit()
    train_batch_mse_batch = mlp_batch.mse
    eval_batch_mse_batch = mlp_batch.validation_mse


    Utils.plot_decision_boundary_mlp(inputs, inputs_labels, mlp_batch,
                                     'MLP with learning rate {0}, iterations {1} , num hidden nodes {2}'
                                     .format(learning_rate,num_iterations,num_hidden_nodes_layer_1))

    mlp_seq = MLP(inputs=inputs, inputs_labels=inputs_labels, input_validation=input_validation,
                  input_validation_labels=input_validation_labels,
                  num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                  num_iterations=num_iterations, learning_rate=learning_rate, batch_train=False, verbose=verbose)

    [_, _, mse_seq] = mlp_seq.fit()
    train_seq_mse_batch = mlp_seq.mse
    eval_seq_mse_batch = mlp_seq.validation_mse

    mse = [train_batch_mse_batch, train_seq_mse_batch, eval_batch_mse_batch, eval_seq_mse_batch]
    legend_names = ['train batch', 'train seq', 'eval batch', 'eval seq']
    Utils.plot_error_with_epochs(mse, legend_names=legend_names, num_epochs=num_iterations,
                                 title='MLP with lr = {0}, iterations = {1} , hidden nodes = {2} '
                                 .format(learning_rate, num_iterations, num_hidden_nodes_layer_1))



def experiment_perceptron_delta():
    [X, Y] = Utils.create_linearly_separable_data()

    # Utils.plot_initial_data(X, Y)

    learning_rate = 0.001
    n_epochs = 40
    perceptron_learning = False

    percep = perceptron(X, Y, n_epochs=n_epochs, learning_rate=learning_rate,
                        batch_train=True, perceptron_learning=True)
    [weights_perceptron, _] = percep.train()

    delta = perceptron(X, Y, n_epochs=n_epochs, learning_rate=learning_rate,
                       batch_train=True, perceptron_learning=False)
    [weights_delta, _] = delta.train()


    Utils.plot_Perceptron_Delta(X, Y, weights_delta=weights_delta, weights_perceptron=weights_perceptron,
                                title="Batch Perceptron with epochs = {0}".format(str(n_epochs)) )


if __name__ == "__main__":
    # run_hidden_nodes_mse_plot_experiment()

    # experiment_train_validation_error()

    # experiment_train_validation_nodes()

    # experiment_train_val_seq_batch_mlp()

    # experiment_learning_curves_error()

    experiment_perceptron_delta()