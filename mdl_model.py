__author__ = 'jhlee'

#from __future__ import absolute_import
#from __future__ import print_function

from keras.models import Graph, Sequential
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.utils.np_utils import accuracy

from keras.layers.core import TimeDistributedDense, Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
#from keras.datasets import imdb
#from keras.optimizers import RMSprop

import os.path
import glob
import numpy as np
import csv
from six.moves import cPickle 

#-------------------------------------------
#np.random.seed(1337)  # for reproducibility
np.random.seed(1237)  # for reproducibility

event = {'Ev101': 1, 'Ev102': 2, 'Ev103': 3, 'Ev104': 4, 'Ev105': 5, 'Ev106': 6, 'Ev107': 7, 'Ev108': 8, 'Ev109': 9, 'Ev110': 0}
settt = {'Training': 0, 'Test': 1}

infofile = open('YLIMED_info.csv', 'rb')
inforeader = csv.reader(infofile, )
next(inforeader)

labelset = np.zeros(10)
ttset = np.zeros(2)

trainingset = np.zeros(10)
testset = np.zeros(10)

VID = []
LABEL = []
SET = []
for info in inforeader:
    VID.append(info[0])
    LABEL.append(info[7])
    SET.append(info[13])
    
    labelset[event[info[7]]] += 1
    ttset[settt[info[13]]] += 1
    
    if info[13] == 'Training':
        trainingset[event[info[7]]] += 1
    else:
        testset[event[info[7]]] += 1
        
    
infofile.close()

print ('Data count =', int(sum(labelset)))
print ('Label set =', labelset)
print ('Training / Test ratio =', ttset)
print ('Training set  =', trainingset)
print ('Test set =', testset)

print('##########Loading data...')
X_test = []
y_test = []
X_train = []
y_train = []
count = 0

audioset = glob.glob('YLIMED150924/audio/mfcc20/*.mfcc20.ascii')
for temp in audioset:
    audiopath = temp
    audioid = temp.split('/')
    audioid = audioid[len(audioid)-1]
    audioid = audioid.split('.')[0]
    
    try:
        aud_label = event[LABEL[VID.index(audioid)]]
        aud_set = SET[VID.index(audioid)]
    except ValueError:
        continue
            
    count += 1
#    if count % 10 == 0:
#        print (count)
    audiofile = open(audiopath, 'r')
    audiodata = audiofile.readlines()
    feat_aud = []

    range_len=len(audiodata)/100

    if aud_set == 'Test':
        for i in range(range_len):
            aud_feat = []
            for j in range(100):
                aud_feat += [float(x) for x in audiodata[i*100+j].split()]
            #print len(feat_aud)
            X_test.append(aud_feat)
            y_test.append(aud_label)
    else:
        for i in range(range_len):
            aud_feat = []
            for j in range(100):
                aud_feat += [float(x) for x in audiodata[i*100+j].split()]
            #print len(feat_aud)
            X_train.append(aud_feat)
            y_train.append(aud_label)
            
    audiofile.close()

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train, )
y_test = np.asarray(y_test)

#max_features = 20000
#maxlen = 100  # cut texts after this number of words (among top max_features most common words)
max_features = len(X_train)#43599
maxlen = len(X_train[0])#2000
batch_size = 32
nb_epoch = 10

#print("Loading data...")
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#-----------------------------------------------------------------------
#https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py
#https://github.com/fchollet/keras/issues/1063
#http://keras.io/models/
print("###########Create Model...")
max_features = 43599
maxlen = 2000

model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=float)

model.add_node(Embedding(max_features, 256, input_length=maxlen), name='embedding', input='input')
model.add_node(LSTM(128), name='forward', input='embedding')
model.add_node(LSTM(128, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])

model.add_node(Dense(10), name='dense', input='dropout')
model.add_node(Dense(10, activation='softmax'), name='soft_max', input='dense')
model.add_output(name='output', input='soft_max')
model.compile('rmsprop', {'output':'categorical_crossentropy'})

print("##########Fit Model...")
history = model.fit({'input':X_train, 'output':Y_train}, nb_epoch=10)
#-----save model
model.save_weights('LSTM0120.model', overwrite=False)

#-----
print("##########Evaluate Model...")
score = model.evaluate({'input':X_test, 'output':Y_test})
acc = accuracy(Y_test, np.round(np.array(model.predict({'input':X_test})['output'])))
print('Test score:', score)
print('Test accuracy:', acc)

#-----
pred = np.array(model.predict({'input':X_test})['output'])
ac = 0
for i in range(0, len(X_test)):
    if np.argmax(Y_test[i]) == np.argmax(pred[i]):
        ac += 1
print(ac)
print(float(ac) / float(len(X_test)))
