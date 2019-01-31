from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, GaussianNoise
from keras import regularizers
from keras import backend as K
import keras.optimizers as optimizers
from keras.regularizers import *
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import csv
import Utils

np.random.seed(0)


class ErrorsCallback(keras.callbacks.Callback):
    def __init__(self, val_in, val_out, train_in, train_out, test_in, test_out):
        self.val_in = val_in
        self.val_out = val_out
        self.train_in = train_in
        self.train_out = train_out
        self.test_in = test_in
        self.test_out = test_out
        self.mse_train = []
        self.mse_val = []
        self.mse_test = []

    def on_epoch_end(self, epoch, logs={}):
        self.mse_val.append(self.model.evaluate(self.val_in, self.val_out, verbose=0))
        self.mse_train.append(self.model.evaluate(self.train_in, self.train_out, verbose=0))
        self.mse_test.append(self.model.evaluate(self.test_in, self.test_out, verbose=0))


def mackey_glass_time_series(length, noise=0):
    beta = 0.2
    gamma = 0.1
    n = 10
    tau = 25

    x = np.zeros(length)
    x[0] = 1.5

    for i in range(0, length - 1):
        x[i + 1] = x[i] + (beta * x[i - tau]) / (1 + x[i - tau] ** n) - gamma * x[i]
        # if noise > 0:
        #     x[i+1] += np.random.normal(0, noise, 1)

    return x


def add_noise_to_dataset(dataset, noise):
    dataset += np.random.normal(0, noise, np.shape(dataset))
    return dataset




def write_to_Csv(dictionary):
    with open('parameters_1layer.csv', 'a', newline='') as csvfile:
        fieldnames = dictionary.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dictionary)


def get_data(input, output):
    X_train = input[0:900, :]
    X_val = input[900:1000, :]
    X_test = input[1000:1200, :]

    Y_train = output[0:900]
    Y_val = output[900:1000]
    Y_test = output[1000:1200]

    return X_val, Y_val, X_train, Y_train, X_test, Y_test


def create_mackey_glass_dataset(times, noise=0):
    start = 301
    end = 1501

    rows = end - start
    columns = len(times)
    inputs = np.zeros((rows, columns))

    sequence = mackey_glass_time_series(end + 5, noise)

    for i, time in enumerate(times):
        inputs[:, i] = sequence[0:end][(start - time): (end - time)]

    output = np.array(sequence[start + 5: end + 5])
    return np.array(inputs), output.reshape(output.shape[0], 1), sequence


# 5,0.1,1,5,L2 (lr=0.01),SGD,mse,0.1182
# 5,0.1,1,5,L2 (lr=0.0001),SGD,mse,0.0727
# 5,0.1,1,10,L2 (lr=0.0001),SGD,mse,0.0427
# 20,0.1,1,10,L2 (lr=0.0001),SGD,mse,0.0276
# 20,0.1,1,10,L2 (lr=0.0001),SGD,mse,0.0399
# 50,0.1,1,10,L2 (lr=0.0001),SGD,mse,0.0291


def run_noise_nodes_experiment():

    filepath = "weights.best.hdf5"

    batch_size = 32
    learning_rate = 0.01
    opt = "SGD"
    loss = 'mse'
    activation = 'linear'
    epochs = 200
    n_layers = 2


    nodes = np.arange(1,8,1)
    noises = [0.03, 0.09, 0.18]
    mse = []
    mse_train= []
    for node in nodes:
        print("node : {0}".format(node))
        val_mse = []
        train_mse = []
        for noise in noises:

            input, output, time_series = create_mackey_glass_dataset([20, 15, 10, 5, 0],noise)

            # Utils.plot_glass_data(time_series)
            X_val, Y_val, X_train, Y_train, X_test, Y_test = get_data(input, output)

            dim_2 = X_train.shape[1]

            X_train = add_noise_to_dataset(X_train, noise)

            if opt == 'Adam':
                optimizer = optimizers.Adam(lr=0.04)
            if opt == 'SGD':
                optimizer = optimizers.SGD(lr=0.01, clipvalue=0.5)
            else:
                optimizer = optimizers.Adam(lr=0.04)

            earlystop = EarlyStopping(monitor="val_loss", patience=15, verbose=0, mode='min')
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
            error = ErrorsCallback(X_val, Y_val, X_train, Y_train, X_test, Y_test)

            callbacks = [error, earlystop, checkpoint]

            regularizer = regularizers.l2(learning_rate)
            regularizer_first_layer = regularizers.l2(0.0001)

            model = Sequential()
            # model.add(GaussianNoise(noise, input_shape=(dim_2,)))
            model.add(Dense(4, input_dim=dim_2, activation=activation, kernel_regularizer=regularizer_first_layer))

            # model.add(Dropout(0.10))
            if n_layers == 2:
                model.add(Dense(node, activation=activation, kernel_regularizer=regularizer))
            # model.add(Dropout(0.10))
            model.add(Dense(Y_test.shape[1]))
            model.add(Activation('linear'))

            model.compile(loss=loss, optimizer=optimizer)

            print("Training...")
            model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), verbose=False,
                      callbacks=callbacks,
                      batch_size=batch_size, shuffle=True)

            val_mse.append(error.mse_val[-1])
            train_mse.append(error.mse_train[-1])

            print("Generating test predictions...")
            preds = model.predict(X_test)
            eval = model.evaluate(X_test, Y_test)

            print(eval)

        mse.append(val_mse)
        mse_train.append(train_mse)

    print(mse)
    print(mse_train)

    final_mse = np.hstack([mse, mse_train])

    legend_names = ['val mse sigma 0.03', 'val mse sigma 0.09', 'val mse sigma 0.18',
                    'train mse sigma 0.03', 'train mse sigma 0.09', 'train mse sigma 0.18']
    Utils.plot_nn_with_nodes(np.array(final_mse).T, legend_names, nodes,
                                 'Three layers network with lr = {0}, batch = 32'.format(learning_rate))



def run_exp():
    input, output, time_series = create_mackey_glass_dataset([20, 15, 10, 5, 0])

    # Utils.plot_glass_data(time_series)
    X_val, Y_val, X_train, Y_train, X_test, Y_test = get_data(input, output)

    # X_train = add_noise_to_dataset(X_train, noise=0.09)

    dim_2 = X_train.shape[1]

    filepath = "weights.best.hdf5"

    batch_size = 32
    learning_rate = 0.0001
    opt = "SGD"
    loss = 'mse'
    activation = 'linear'
    epochs = 500
    number_of_nodes_layer_1 = 4
    number_of_nodes_layer_2 = 2
    n_layers = 1
    validation_split = 0.35

    if opt == 'Adam':
        optimizer = optimizers.Adam(lr=0.04)
    if opt == 'SGD':
        optimizer = optimizers.SGD(lr=0.01, clipvalue=0.5)
    else:
        optimizer = optimizers.Adam(lr=0.04)

    earlystop = EarlyStopping(monitor="val_loss", patience=15, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    error = ErrorsCallback(X_val, Y_val, X_train, Y_train, X_test, Y_test)

    callbacks = [error, earlystop, checkpoint]

    regul = "L2 (lr={0})".format(learning_rate)
    regularizer = regularizers.l2(learning_rate)

    model = Sequential()
    model.add(Dense(number_of_nodes_layer_1, input_dim=dim_2, activation=activation, kernel_regularizer=regularizer))
    # model.add(Dropout(0.10))
    if n_layers == 2:
        model.add(Dense(number_of_nodes_layer_2, activation=activation, kernel_regularizer=regularizer))
    # model.add(Dropout(0.10))
    model.add(Dense(Y_test.shape[1]))
    model.add(Activation('linear'))

    model.compile(loss=loss, optimizer=optimizer)

    print("Training...")
    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), verbose=True, callbacks=callbacks,
              batch_size=batch_size, shuffle=True)

    print("Generating test predictions...")
    preds = model.predict(X_test)
    eval = model.evaluate(X_test, Y_test)

    print('test',eval)
    print('train')
    eval_trian = model.evaluate(X_train, Y_train)
    print('train',eval_trian)

    print('val')
    eval_val = model.evaluate(X_val, Y_val)
    print('val', eval_val)

    dictionary = {'Epochs': epochs, 'Val split': validation_split, 'n Layers': n_layers,
                  'n Nodes _layer 1': number_of_nodes_layer_1, 'n Nodes _layer 2': number_of_nodes_layer_2,
                  'Batch Size': batch_size, 'Regularizer': regul,
                  'Optimizer': opt, 'Metric': loss, 'Pred loss': round(eval, 4)}

    mse = [error.mse_train, error.mse_val, error.mse_test]

    legend_names = ['train', 'validation', 'test']
    Utils.plot_error_with_epochs(mse, legend_names, epochs, 'Two layers network with 8 nodes ,lr = 0.0001, batch = 32')

    # Utils.plot_glass_data_prediction(preds, Y_test, "Predictions")

    write_to_Csv(dictionary)

    # print(np.mean(error.mse_val))


def run_weights_distribution():

    rates = [0.00001,0.0001, 0.001, 0.01, 0.1]
    weights =[]
    for learning_rate in rates:
        input, output, time_series = create_mackey_glass_dataset([20, 15, 10, 5, 0])

        X_val, Y_val, X_train, Y_train, X_test, Y_test = get_data(input, output)

        dim_2 = X_train.shape[1]

        filepath = "weights.best.hdf5"

        batch_size = 32
        opt = "SGD"
        loss = 'mse'
        activation = 'linear'
        epochs = 40
        number_of_nodes = 8
        n_layers = 1

        if opt == 'Adam':
            optimizer = optimizers.Adam(lr=0.04)
        if opt == 'SGD':
            optimizer = optimizers.SGD(lr=0.01, clipvalue=0.5)
        else:
            optimizer = optimizers.Adam(lr=0.04)

        earlystop = EarlyStopping(monitor="val_loss", patience=5, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
        error = ErrorsCallback(X_val, Y_val, X_train, Y_train, X_test, Y_test)

        callbacks = [error, earlystop, checkpoint]

        regularizer = regularizers.l2(learning_rate)

        model = Sequential()
        model.add(Dense(number_of_nodes, input_dim=dim_2, activation=activation, kernel_regularizer=regularizer))

        if n_layers == 2:
            model.add(Dense(number_of_nodes, activation=activation, kernel_regularizer=regularizer))

        model.add(Dense(Y_test.shape[1]))
        model.add(Activation('linear'))

        model.compile(loss=loss, optimizer=optimizer)

        print("Training...")
        model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), verbose=False, callbacks=callbacks,
                  batch_size=batch_size, shuffle=True)

        first_layer_weights = model.layers[0].get_weights()[0]
        weights.append(first_layer_weights)

    Utils.plot_weights_distribution(rates,weights)




if __name__ == "__main__":

    run_noise_nodes_experiment()

    # run_exp()

    # run_weights_distribution()