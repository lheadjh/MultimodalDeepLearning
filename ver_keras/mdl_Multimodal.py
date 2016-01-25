__author__ = 'jhlee'

from keras.models import Graph, Sequential
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.utils.np_utils import accuracy

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
#from keras.datasets import imdb
#from keras.optimizers import RMSprop

import mdl_data
import numpy as np

np.random.seed(1337)  # for reproducibility

data = mdl_data.YLIMED('YLIMED_info.csv', '../YLIMED150924/audio/mfcc20', '../YLIMED150924/keyframe/fc7')

X_img_train = data.get_img_X_train()
X_aud_train = data.get_aud_X_train()
y_train = data.get_y_train()
Y_train = np_utils.to_categorical(y_train, 10)

model = Graph()
maxlen = len(X_img_train[0])
model.add_input(name='img_input', input_shape=(maxlen,))
model.add_node(Dense(1000, activation='relu'), name='img_dense1', input='img_input')
model.add_node(Dropout(0.5), name='img_dropout1', input='img_dense1')
model.add_node(Dense(600, activation='relu'), name='img_dense2', input='img_dropout1')
model.add_node(Dropout(0.5), name='img_dropout2', input='img_dense2')
model.add_node(Dense(10), name='img_dense3', input='img_dropout2')

max_feature = len(X_aud_train)
maxlen = len(X_aud_train[0])
model.add_input(name='aud_input', input_shape=(maxlen,), dtype=float)
model.add_node(Embedding(max_feature, 128, input_length=maxlen), name='embedding', input='aud_input')
model.add_node(LSTM(64), name='forward', input='embedding')
model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='aud_dropout', inputs=['forward', 'backward'])
model.add_node(Dense(10), name='aud_dense', input='aud_dropout')

model.add_node(Dropout(0.5), name='dropout', inputs=['img_dense3', 'aud_dense'], merge_mode='sum')
model.add_node(Dense(10, activation='softmax'), name = 'soft_max', input='dropout')
model.add_output(name='output', input='soft_max')
model.compile('rmsprop', {'output':'categorical_crossentropy'})

history = model.fit({'img_input':X_img_train, 'aud_input':X_aud_train, 'output':Y_train}, nb_epoch=10)
model.save_weights('MmDL.model', overwrite=False)

X_img_test = data.get_img_X_test()
X_aud_test = data.get_aud_X_test()
y_test = data.get_y_test()
Y_test = np_utils.to_categorical(y_test, 10)

pred = np.array(model.predict({'img_input':X_img_test, 'aud_input':X_aud_test})['output'])
ac = 0
for i in range(0, len(X_test)):
    if np.argmax(Y_test[i]) == np.argmax(pred[i]):
        ac += 1
print 'Test accuracy:	', float(ac) / float(len(X_test))

