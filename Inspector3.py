from datasethdr import get_dataset3, m, n

NO_VOICE_LABEL = 0
VOICE_LABEL = 1

batch_size = 128
num_classes = 2
epochs = 10
input_shape = (n, 2 * m)


def train_model(x, y):
    test_num = len(x) / 10
    import random
    index = [i for i in range(len(x))]
    random.shuffle(index)
    data = []
    label = []

    for i in index:
        data.append(x[i])
        label.append(y[i])
    x_train = data[:-test_num]
    y_train = label[:-test_num]
    x_test = data[-test_num:]
    y_test = label[-test_num:]

    import numpy
    import keras
    x_train = numpy.array(x_train)
    print x_train.shape
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2] * x_train.shape[3]))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    print y_train.shape
    x_test = numpy.array(x_test)
    print x_test.shape
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2] * x_test.shape[3]))
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print y_test.shape

    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Conv1D
    from keras.layers import Dense
    from keras.layers import Dropout

    model = Sequential()
    model.add(Conv1D(32, 3, input_shape=input_shape))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    tb_cb = keras.callbacks.TensorBoard(log_dir='keras_log', write_images=1, histogram_freq=1)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[tb_cb],
              validation_data=(x_test, y_test))

    x_ev = x_test
    y_ev = y_test
    score = model.evaluate(x_ev, y_ev, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('inspect.h5')


if __name__ == '__main__':
    pure_voice = get_dataset3('DataSet/pure_voice.csv', '192.168.31.219',draw=True)
    print('pure voice ', len(pure_voice))
    # draw_dataset3(pure_voice[240])
    mixed_voice = get_dataset3('DataSet/mixed_voice.csv', '192.168.31.219',draw=True)
    print('mix voice ', len(mixed_voice))

    voice_data = pure_voice + mixed_voice
    voice_label = [VOICE_LABEL] * len(voice_data)
    print('voice ', len(voice_data))

    no_voice_data = get_dataset3('DataSet/no_voice.csv', '192.168.31.219',draw=True)
    no_voice_label = [NO_VOICE_LABEL] * len(no_voice_data)
    print('no voice ', len(no_voice_data))

    train_model(voice_data + no_voice_data, voice_label + no_voice_label)
