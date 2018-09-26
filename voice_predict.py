import numpy
from keras.models import load_model

from datasethdr import draw_result
from datasethdr import get_dataset3, sample_rate

x = get_dataset3('DataSet/30svoice.csv', '192.168.31.219', 1 * sample_rate, True)

x = numpy.array(x)
x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

model = load_model('inspect.h5')
y = model.predict(x)

draw_result(y, '30s voice')
