from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras import regularizers
from keras import backend as K
import keras.optimizers as optimizers
from keras.regularizers import *
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import csv
import Utils

np.random.seed(0)


def mackey_glass_time_series(length, noise=0):
    beta = 0.2
    gamma = 0.1
    n = 10
    tau = 25

    x = np.zeros(length)
    x[0] = 1.5

    for i in range(0, length - 1):
        x[i + 1] = x[i] + (beta * x[i - tau]) / (1 + x[i - tau] ** n) - gamma * x[i]
        if noise > 0:
            x[i + 1] += np.random.normal(0, noise, 1)

    return x


def write_to_Csv(dictionary):
    with open('parameters.csv', 'a', newline='') as csvfile:
        fieldnames = dictionary.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow(dictionary)


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


if __name__ == "__main__":
    input, output, time_series = create_mackey_glass_dataset([20, 15, 10, 5, 0])

    # Utils.plot_glass_data(time_series)

    X_train = input[0:1000, :]
    X_test = input[1000:1200, :]
    Y_train = output[0:1000]
    Y_test = output[1000:1200]

    dim_1 = X_train.shape[0]
    dim_2 = X_train.shape[1]

    learning_rate = 0.00001
    opt = "SGD"
    loss = 'mse'
    activation = 'linear'
    epochs = 500
    number_of_nodes = 10
    n_layers = 2
    validation_split = 0.1

    if opt == 'Adam':
        optimizer = optimizers.Adam(lr=0.04)
    if opt == 'SGD':
        optimizer = optimizers.SGD(lr=0.01, clipvalue=0.5)
    else:
        optimizer = optimizers.Adam(lr=0.04)

    filepath = "weights.best.hdf5"

    earlystop = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode='min')
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    callbacks = [earlystop]
    batch_size = 32

    regul = "L2 (lr={0})".format(learning_rate)
    regularizer = regularizers.l2(learning_rate)

    model = Sequential()
    model.add(Dense(number_of_nodes, input_dim=dim_2, activation=activation, kernel_regularizer=regularizer))
    # model.add(Dropout(0.10))
    if n_layers==2:
        model.add(Dense(number_of_nodes, activation=activation,kernel_regularizer=regularizer))
    # model.add(Dropout(0.10))
    model.add(Dense(Y_test.shape[1]))
    model.add(Activation('linear'))

    model.compile(loss=loss, optimizer=optimizer)

    print("Training...")
    model.fit(X_train, Y_train, epochs=epochs, validation_split=validation_split, verbose=True, callbacks=callbacks,
              batch_size=batch_size, shuffle=True)

    print("Generating test predictions...")
    preds = model.predict(X_test)
    eval = model.evaluate(X_test, Y_test)

    print(eval)
    dictionary = {'Epochs': epochs, 'Validation split': validation_split, 'n_Layers': n_layers,
                  'n_Nodes': number_of_nodes,
                  'Regularizer': regul, 'Optimizer': opt, 'Metric': loss, 'Output loss': round(eval,4)}

    Utils.plot_glass_data_prediction(preds, Y_test, "Predictions")
    write_to_Csv(dictionary)