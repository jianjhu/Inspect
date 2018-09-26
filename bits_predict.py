import matplotlib.pyplot as plt
import numpy
from keras.models import load_model

from Inspector import num_classes
from datasethdr import get_dataset3, sample_rate

model = load_model('inspect.h5')


def draw_result(result, title, filter):
    yl = result.tolist()
    y = [l.index(max(l)) for l in yl]
    plt.subplot(211)
    plt.ylim(-1 * num_classes, num_classes)
    plt.plot(y, 'o')
    plt.plot(result)
    # plt.annotate('voice possibility', xy=(30,1), xytext=(40,1.5),arrowprops=dict(facecolor='black',shrink=0.05))
    plt.title(title)
    plt.subplot(212)
    plt.ylim(-1 * num_classes, num_classes)
    plt.plot(filter, 'o')
    plt.plot(result)
    plt.show()


def smooth(y, length):
    yl = y.tolist()
    y = [l.index(max(l)) for l in yl]
    ret = [y[0]] * length
    for s in range(length, len(y)):
        v = sum(y[s - length:s])
        if v == length:
            ret.append(1)
        elif v == 0:
            ret.append(0)
        else:
            ret.append(ret[-1])
    return ret


x = get_dataset3('DataSet/normalshare', '10.224.168.94', 1 * sample_rate, True)
x = numpy.array(x)
print x.shape
x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
y = model.predict(x)
# print y

yl = y.tolist()

for idx, l in enumerate(yl):
    print('time:', idx, l.index(max(l)), max(l))

filtered = smooth(y, 5)
draw_result(y, 'normal share', filtered)

x = get_dataset3('DataSet/HighFPS.csv', '10.224.168.94', 1 * sample_rate, True)
x = numpy.array(x)
x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
y = model.predict(x)
yl = y.tolist()

for idx, l in enumerate(yl):
    print(idx, l.index(max(l)), max(l))

filtered = smooth(y, 5)
draw_result(y, 'HighFPS', filtered)
