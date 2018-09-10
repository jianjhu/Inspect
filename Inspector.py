import csv
import time

import keras
import numpy

voice_train_files = ['DataSet/Voice.csv']
no_voice_train_files = ['DataSet/noVoice.csv']
webex_normalshare_train_files = ['DataSet/normalshare']
highFPS_train_files = ['DataSet/HighFPS.csv']

webex_ppt_train_files = ['DataSet/highfps_ppt.csv', 'DataSet/normal_ppt.csv']
webex_video_train_files = ['DataSet/highfps_video.csv', 'DataSet/normal_video.csv']

'''
VOICE_LABEL = 0
OTHER_LABEL = 1
'''
WEBEX_PPT_LABEL = 0
WEBEX_VIDEO_LABEL = 1

src = '10.224.168.94'

batch_size = 128
num_classes = 2
epochs = 10

m = 1600
sample_rate = 10
n = 1 * sample_rate
input_shape = (n, 2 * m)


def get_dataset(file, source, interval=1):
    dataset = [0] * 2 * m
    dataset = [dataset] * n
    with open(file) as f:
        reader = csv.DictReader(f)
        time0 = None
        for row in reader:
            timex = row['Time']
            datetime = time.strptime(timex, "%H:%M:%S.%f")
            datetime = list(datetime)
            datetime[0] = 2018
            timex = time.mktime(datetime)
            if time0 is None:
                time0 = timex
            timex = int(sample_rate * (timex - time0))

            while timex >= len(dataset):
                dataset.append([0] * 2 * m)

            src = row['Source']
            direction = -1 if source == src else 1

            length = row['Length']
            length = int(length) * direction
            length = m + length
            dataset[timex][length] += 1

    return [dataset[i:i + n] for i in range(0, len(dataset) - n, interval)]


def draw_dataset(dataset):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for time, lengths in enumerate(dataset):
        cs = list(((x % 10) / 10.0, 1 - (x % 10) / 10.0, 0.5) for x in range(-1 * m, m))
        ax.bar(range(-1 * m, m), lengths, zs=time, zdir='y', edgecolor=cs, alpha=0.5)
        print('time %d draw, total %d' % (time, len(dataset)))
        if time > 10:
            break

    ax.set_xlabel('Length')
    ax.set_ylabel('Time')
    ax.set_zlabel('Count')
    plt.show()


if __name__ == '__main__':

    webex_ppt_train_dataset = None
    for fi in webex_ppt_train_files:
        dataset = get_dataset(fi, '10.224.168.94')
        # draw_dataset(dataset[0])
        if webex_ppt_train_dataset is None:
            webex_ppt_train_dataset = dataset
        else:
            webex_ppt_train_dataset += dataset
    ppt_train_label = [WEBEX_PPT_LABEL] * len(webex_ppt_train_dataset)
    print("get %d ppt train dataset" % len(webex_ppt_train_dataset))

    webex_video_train_dataset = None
    for fi in webex_video_train_files:
        dataset = get_dataset(fi, '10.224.168.94')
        # draw_dataset(dataset[0])
        if webex_video_train_dataset is None:
            webex_video_train_dataset = dataset
        else:
            webex_video_train_dataset += dataset
    video_train_label = [WEBEX_VIDEO_LABEL] * len(webex_video_train_dataset)
    print("get %d video train dataset" % len(webex_video_train_dataset))

    '''
    for fi in voice_train_files:
        voice_train_dataset = get_dataset(fi, src)
        voice_train_label = [VOICE_LABEL] * len(voice_train_dataset)
        draw_dataset(voice_train_dataset[0])

    print("get %d voice train dataset"%len(voice_train_dataset))

    for fi in no_voice_train_files:
        other_train_dataset = get_dataset(fi, src)
        other_train_label = [OTHER_LABEL] * len(other_train_dataset)
        draw_dataset(other_train_dataset[0])

    print("get %d other train dataset"%len(other_train_dataset))
    '''

    x_train = webex_ppt_train_dataset + webex_video_train_dataset
    y_train = ppt_train_label + video_train_label
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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    print y_train.shape

    x_test = data[-test_data_num:]
    x_test = numpy.array(x_test)
    print x_test.shape
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

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    x_ev = x_test
    y_ev = y_test
    score = model.evaluate(x_ev, y_ev, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('inspect.h5')
