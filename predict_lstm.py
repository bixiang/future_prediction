# ##定义网络结构
# def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
#     print 'Defining a Simple Keras Model...'
#     model = Sequential()  # or Graph or whatever
#     model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#
#     print('Compiling the Model...')
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',metrics=['accuracy'])
#
#     print("Train...")
#     model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=10,verbose=1, validation_data=(x_test, y_test),show_accuracy=True)
#
#     print("Evaluate...")
#     score = model.evaluate(x_test, y_test, batch_size=batch_size)
#
#     yaml_string = model.to_yaml()
#     # with open('lstm_data/lstm.yml', 'w') as outfile:
#     #     outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
#     # model.save_weights('lstm_data/lstm.h5')
#     print('Test score:', score)


import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
#
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from handle_for_lstm import handle_for_lstm
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD
from sklearn import cross_validation,metrics
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# tf.set_random_seed(2)

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

from keras import backend as K
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import os

# rn.seed(12345)

# fix random seed for reproducibility
np.random.seed(7)

x_train, y_train, x_test, y_test = handle_for_lstm(20)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# x_train, x_test = x_train.reshape(x_train.shape[0], 1, 960), x_test.reshape(x_test.shape[0], 1, 960)

# print(x_train)
# print(x_test.shape)

batch_size = 128

print('Build model...')
model = Sequential()
model.add(LSTM(units=64, input_shape=(x_train.shape[1], x_train.shape[2]),return_sequences=True))
model.add(LSTM(units=50, input_shape=(x_train.shape[1], x_train.shape[2])))
# model.add(Dropout(0.5))
model.add(Dense(units=1))
# model.add(Activation('sigmoid'))
# sgd = SGD(lr=0.1, momentum=0.0, decay=0.1, nesterov=False)

model.compile(loss='mae', optimizer='adam')

print("Train...")
history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=30,verbose=1, validation_data=(x_test, y_test), shuffle=False)

result = model.predict(x_test)
# print(result)

fenmu = 0
fenzi = 0
for i in range(result.shape[0]):
    print(result[i][0],y_test[i])
    if abs(result[i][0]) > 0.02:
        fenmu = fenmu + 1
        if np.sign(y_test[i]) == np.sign(result[i][0]):
            fenzi = fenzi + 1

print(fenzi,fenmu,fenzi/fenmu)


# classes = model.predict_classes(x_test)
# acc = metrics.accuracy_score(classes, y_test)
# print('Test accuracy:', acc)

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()