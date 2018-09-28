
import keras
import numpy

from datasethdr import get_dataset3
from datasethdr import n, m

voice_train_files = ['DataSet/Voice.csv']
no_voice_train_files = ['DataSet/noVoice.csv']
webex_normalshare_train_files = ['DataSet/normalshare']
highFPS_train_files = ['DataSet/HighFPS.csv']

webex_ppt_train_files = ['DataSet/highfps_ppt.csv', 'DataSet/normal_ppt.csv']
webex_video_train_files = ['DataSet/highfps_video.csv', 'DataSet/normal_video.csv']


WEBEX_PPT_LABEL = 0
WEBEX_VIDEO_LABEL = 1
VOICE_LABEL = 2
OTHER_LABEL = 3

src = '10.224.168.94'

batch_size = 128
num_classes = 4
epochs = 10
input_shape = (n, 2 * m)

if __name__ == '__main__':
    webex_ppt_train_dataset = None
    for fi in webex_ppt_train_files:
        dataset = get_dataset3(fi, '10.224.168.94')
        # draw_dataset(dataset[0])
        if webex_ppt_train_dataset is None:
            webex_ppt_train_dataset = dataset
        else:
            webex_ppt_train_dataset += dataset
    ppt_train_label = [WEBEX_PPT_LABEL] * len(webex_ppt_train_dataset)
    print("get %d ppt train dataset" % len(webex_ppt_train_dataset))

    webex_video_train_dataset = None
    for fi in webex_video_train_files:
        dataset = get_dataset3(fi, '10.224.168.94')
        # draw_dataset(dataset[0])
        if webex_video_train_dataset is None:
            webex_video_train_dataset = dataset
        else:
            webex_video_train_dataset += dataset
    video_train_label = [WEBEX_VIDEO_LABEL] * len(webex_video_train_dataset)
    print("get %d video train dataset" % len(webex_video_train_dataset))

    for fi in voice_train_files:
        voice_train_dataset = get_dataset3(fi, '192.168.31.219')
        voice_train_label = [VOICE_LABEL] * len(voice_train_dataset)
        # draw_dataset(voice_train_dataset[0])

    print("get %d voice train dataset"%len(voice_train_dataset))

    for fi in no_voice_train_files:
        other_train_dataset = get_dataset3(fi, '192.168.31.219')
        other_train_label = [OTHER_LABEL] * len(other_train_dataset)
        #draw_dataset(other_train_dataset[0])

    print("get %d other train dataset"%len(other_train_dataset))

    x_train = webex_ppt_train_dataset + webex_video_train_dataset + voice_train_dataset + other_train_dataset
    y_train = ppt_train_label + video_train_label + voice_train_label + other_train_label
    test_data_num = len(x_train) / 10

    import random

    index = [i for i in range(len(x_train))]
    random.shuffle(index)
    data = []
    label = []

    for i in index:
        data.append(x_train[i])
        label.append(y_train[i])
    x_train = data[:-test_data_num]
    y_train = label[:-test_data_num]

    x_train = numpy.array(x_train)
    print x_train.shape
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2] * x_train.shape[3]))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    print y_train.shape

    x_test = data[-test_data_num:]
    x_test = numpy.array(x_test)
    print x_test.shape
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2] * x_test.shape[3]))
    y_test = label[-test_data_num:]
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print y_test.shape

    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Conv1D
    from keras.layers import Dense
    from keras.layers import Dropout

    model = Sequential()
    model.add(Conv1D(32, n / 10 + 1, input_shape=input_shape))
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
