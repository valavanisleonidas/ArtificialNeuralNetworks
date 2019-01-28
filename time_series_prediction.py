from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras import regularizers
from keras import backend as K
from keras.optimizers import *
from keras.regularizers import *
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

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

    Utils.plot_glass_data(time_series)

    X_train = input[0:1000, :]
    X_test = input[1000:1200, :]
    Y_train = output[0:1000]
    Y_test = output[1000:1200]
    dim_1 = X_train.shape[0]
    dim_2 = X_train.shape[1]
    # X_train = np.reshape()


    print(X_train.shape, Y_test.shape)

    optimizer = Adam(lr=0.004)
    monitor = 'accuracy'
    earlystop = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode='auto')
    callbacks = [earlystop]
    batch_size = 10
    validation_data = [X_test, Y_test]
    epochs = 100

    model = Sequential()
    model.add(Dense(100, input_shape=(dim_2, )))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.10))
    # model.add(Dense(100))
    # model.add(Activation('sigmoid'))
    # model.add(Dropout(0.10))
    model.add(Dense(Y_test.shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[monitor])

    print("Training...")
    model.fit(X_train, Y_train, epochs=epochs, validation_split=0.1, verbose=True, callbacks=callbacks,
              batch_size=batch_size)

    print("Generating test predictions...")
    preds = model.predict(X_test)
    print(preds)
