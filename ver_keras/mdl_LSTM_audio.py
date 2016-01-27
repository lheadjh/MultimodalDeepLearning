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

data = mdl_data.YLIMED('YLIMED_info.csv', '/DATA/YLIMED150924/audio/mfcc20', '/DATA/YLIMED150924/keyframe/fc7')

X_train = data.get_aud_X_train()
X_test = data.get_aud_X_test()
y_train = data.get_y_train()
y_test = data.get_y_test()

max_features = len(X_train)	#43599
maxlen = len(X_train[0])	#2000
batch_size = 32
nb_epoch = 10

print len(X_train), 'train sequences'
print len(X_test), 'test sequences'

#Pad sequences (samples x time)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print 'X_train	shape:', X_train.shape
print 'X_test	shape:', X_test.shape

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=float)
model.add_node(Embedding(max_features, 128, input_length=maxlen), name='embedding', input='input')
model.add_node(LSTM(64), name='forward', input='embedding')
model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
model.add_node(Dense(10), name='dense', input='dropout')
model.add_node(Dense(10, activation='softmax'), name='soft_max', input='dense')
model.add_output(name='output', input='soft_max')
model.compile('rmsprop', {'output':'categorical_crossentropy'})

history = model.fit({'input':X_train, 'output':Y_train}, nb_epoch=10)
model.save_weights('LSTM0120.model', overwrite=False)

score = model.evaluate({'input':X_test, 'output':Y_test})
acc = accuracy(Y_test, np.round(np.array(model.predict({'input':X_test})['output'])))
print 'Test score:	', score
print 'Test accuracy:	', acc

pred = np.array(model.predict({'input':X_test})['output'])
ac = 0
for i in range(0, len(X_test)):
    if np.argmax(Y_test[i]) == np.argmax(pred[i]):
        ac += 1
print 'Test accuracy:	', float(ac) / float(len(X_test))

#https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py
#https://github.com/fchollet/keras/issues/1063
#http://keras.io/models/
