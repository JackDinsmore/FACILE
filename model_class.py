#!/usr/bin/env python
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
#from subtlenet.backend.keras_objects import *
#from subtlenet.backend.losses import *
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, SGD
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle 
import ROOT 
import time

VALSPLIT = 0.2 #0.7
np.random.seed(5)
Nrhs = 2100000

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

### Sample class
class Sample(object):
    def __init__(self, name, base):
        self.name = name 

        self.X = np.load('%s/%s.pkl'%(base,'X'),allow_pickle=True)[:Nrhs]
        self.X.drop(['PU','pt'],1,inplace=True)
        self.Y = np.load('%s/%s.pkl'%(base,'Y'),allow_pickle=True)[:Nrhs]

        print self.X.shape, self.Y.shape
        self.kin = np.load('%s/%s.pkl'%(base,'X'),allow_pickle=True)[:Nrhs][['PU','ieta','iphi','pt']]

        self.idx = np.random.permutation(self.X.shape[0])

    @property
    def tidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[int(VALSPLIT*len(self.idx)):]

    @property
    def vidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[:int(VALSPLIT*len(self.idx))]

    def infer(self, model, batch_size):
        self.Yhat, self.time = model.predict(self.X, batch_size)

    def standardize(self, mu, std):
        self.X = (self.X - mu) / std

### Model class
class ClassModel(object):
    def __init__(self, n_inputs):
        self._hidden = 0
        self.n_inputs = n_inputs
        self.n_targets = 1

        self.inputs = Input(shape=(n_inputs,), name='input')

        self.outputs = self.get_outputs()

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(), loss='mse')

        self.model.summary()

    def get_outputs(self):
        # To be overridden 
        self.name = None
        return None

    def train(self, sample, num_epochs):

        tX = sample.X.values[sample.tidx]
        vX = sample.X.values[sample.vidx]
        tY = sample.Y[['genE']].values[sample.tidx]
        vY = sample.Y[['genE']].values[sample.vidx] 

        history = self.model.fit(tX, tY, 
                                 batch_size=2024, epochs=num_epochs, shuffle=True,
                                 validation_data=(vX, vY))

        with open('history.log','w') as flog:
            history = history.history
            flog.write(','.join(history.keys())+'\n')
            for l in zip(*history.values()):
                flog.write(','.join([str(x) for x in l])+'\n')

    def save_as_keras(self, path):
        _make_parent(path)
        self.model.save(path)
        print 'Saved to',path

    def save_as_tf(self,path):
        _make_parent(path)
        sess = K.get_session()
        print [l.op.name for l in self.model.inputs],'->',[l.op.name for l in self.model.outputs]
        graph = graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          [n.op.name for n in self.model.outputs])
        p0 = '/'.join(path.split('/')[:-1])
        p1 = path.split('/')[-1]
        graph_io.write_graph(graph, p0, p1, as_text=False)
        print 'Saved to',path

    def predict(self, *args, **kwargs):
        start_time = time.time()
        predictions = self.model.predict(*args, **kwargs)
        total_time = time.time() - start_time
        return predictions, total_time

    def load_model(self, path):
        self.model = load_model(path)


