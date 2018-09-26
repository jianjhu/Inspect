import csv
import time
from datetime import datetime

m = 1600
sample_rate = 10
n = 1 * sample_rate


def get_dataset(file, source, interval=1):
    dataset = [0] * 2 * m
    dataset = [dataset] * n
    with open(file) as f:
        reader = csv.DictReader(f)
        time0 = None
        for row in reader:
            timex = row['Time']
            datetime_obj = datetime.strptime(timex, "%H:%M:%S.%f")
            datetime_list = list(datetime_obj.timetuple())
            datetime_list[0] = 2018
            timex = long(
                time.mktime(datetime_list) * sample_rate + datetime_obj.microsecond / (1000 * 1000 / sample_rate))
            if time0 is None:
                time0 = timex
            timex = int(timex - time0)

            while timex >= len(dataset):
                dataset.append([0] * 2 * m)

            src = row['Source']
            direction = -1 if source == src else 1

            length = row['Length']
            length = int(length) * direction
            length = m + length
            dataset[timex][length] += 1

    return [dataset[i:i + n] for i in range(0, len(dataset) - n, interval)]


def draw_dataset(dataset, num=10):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for time, lengths in enumerate(dataset):
        for length, count in enumerate(lengths):
            if count == 0:
                continue
            x = length
            cs = ((x % 10) / 10.0, 1 - (x % 10) / 10.0, 0.5)
            ax.bar([length - m], [count], zs=time, zdir='y', edgecolor=cs, alpha=0.5)
        print('time %d draw, total %d' % (time, len(dataset)))
        if time > num:
            break

    ax.set_xlabel('Length')
    ax.set_ylabel('Time')
    ax.set_zlabel('Count')
    plt.xlim(-m, m)
    plt.show()


DIR_OUT = 0
DIR_IN = 1


def get_dataset3(file, source, interval=1, draw=False):
    dataset = []

    with open(file) as f:
        reader = csv.DictReader(f)
        time0 = None
        for row in reader:
            timex = row['Time']
            datetime_obj = datetime.strptime(timex, "%H:%M:%S.%f")
            datetime_list = list(datetime_obj.timetuple())
            datetime_list[0] = 2018
            timex = long(
                time.mktime(datetime_list) * sample_rate + datetime_obj.microsecond / (1000 * 1000 / sample_rate))
            if time0 is None:
                time0 = timex
            timex = int(timex - time0)

            while timex >= len(dataset):
                dataset.append([[0] * m, [0] * m])

            src = row['Source']
            direction = DIR_OUT if source == src else DIR_IN

            length = row['Length']
            length = int(length)

            dataset[timex][direction][length] += 1

    if draw:
        draw_dataset3(dataset)
    return [dataset[i:i + n] for i in range(0, len(dataset) - n, interval)]


def draw_dataset3(dataset, num=10):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ay = fig.add_subplot(122, projection='3d')
    for time, vector in enumerate(dataset):
        lengths = vector[DIR_OUT]
        for length, count in enumerate(lengths):
            if count == 0:
                continue
            x = length
            cs = ((x % 10) / 10.0, 1 - (x % 10) / 10.0, 0.5)
            ax.bar([length], [count], zs=time / sample_rate, zdir='y', edgecolor=cs, alpha=0.5)
        print('time %d draw, total %d' % (time, len(dataset)))

        lengths = vector[DIR_IN]
        for length, count in enumerate(lengths):
            if count == 0:
                continue
            x = length
            cs = ((x % 10) / 10.0, 1 - (x % 10) / 10.0, 0.5)
            ay.bar([length], [count], zs=time / sample_rate, zdir='y', edgecolor=cs, alpha=0.5)
        print('time %d draw, total %d' % (time, len(dataset)))

    ax.set_xlabel('Length')
    ax.set_ylabel('Time')
    ax.set_zlabel('Count')
    ax.set_xlim(0, m)
    ax.set_title('TX')
    ay.set_xlabel('Length')
    ay.set_ylabel('Time')
    ay.set_zlabel('Count')
    ay.set_xlim(0, m)
    ay.set_title('RX')
    plt.show()


def draw_result(result, title):
    yl = result.tolist()
    num_classes = len(yl[0])
    y = [l.index(max(l)) for l in yl]
    import matplotlib.pyplot as plt
    # plt.subplot(211)
    plt.ylim(-1 * num_classes, num_classes)
    plt.plot(y, 'o')
    plt.plot(result)
    plt.annotate('voice possibility', xy=(30, 1), xytext=(40, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.title(title)
    '''
    plt.subplot(212)
    plt.ylim(-1*num_classes, num_classes)
    plt.plot(filter, 'o')
    plt.plot(result)
    '''
    plt.show()
