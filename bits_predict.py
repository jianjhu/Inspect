import matplotlib.pyplot as plt
import numpy
from keras.models import load_model

from Inspector import get_dataset, sample_rate

model = load_model('inspect.h5')


def draw_result(result, title, filter):
    yl = result.tolist()
    y = [l.index(max(l)) for l in yl]
    plt.subplot(211)
    plt.ylim(-1, 2)
    plt.plot(y, 'o')
    plt.plot(result)
    plt.title(title)
    plt.subplot(212)
    plt.ylim(-1, 2)
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


x = get_dataset('DataSet/normalshare', '10.224.168.94', 1 * sample_rate)
x = numpy.array(x)
print x.shape

y = model.predict(x)
# print y

yl = y.tolist()

for idx, l in enumerate(yl):
    print('time:', idx, l.index(max(l)), max(l))

filtered = smooth(y, 5)
draw_result(y, 'normal share', filtered)

x = get_dataset('DataSet/HighFPS.csv', '10.224.168.94', 1 * sample_rate)
x = numpy.array(x)
y = model.predict(x)
yl = y.tolist()

for idx, l in enumerate(yl):
    print(idx, l.index(max(l)), max(l))

filtered = smooth(y, 5)
draw_result(y, 'HighFPS', filtered)
