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

# batch_size = 128
# num_classes = 10
# epochs = 1
#
# # input image dimensions
# img_rows, img_cols = 28, 28
#
# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
#
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# keras.callbacks.EarlyStopping(monitor='val_loss',
#                               min_delta=0,
#                               patience=2,
#                               verbose=0, mode='auto')
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
