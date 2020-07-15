#!/usr/bin/env python
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from subtlenet.backend.keras_objects import *
#from subtlenet.backend.losses import *
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU
from keras.utils import to_categorical, np_utils
from keras.optimizers import Adam, Nadam, SGD
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle 
import ROOT 
from keras.utils.vis_utils import plot_model

import tensorflow as tf
VALSPLIT = 0.3 #0.7	
np.random.seed(4)
Nrhs = 4000000

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

def get_mu_std(sample):
    mu = np.mean(sample.X, axis=0)
    std = np.std(sample.X, axis=0)
    return mu, std


def savepickle(methods,binning,modeldir):

    a = np.concatenate([arr for _, arr in methods.iteritems()] + [arr for _,arr in binning.iteritems()],axis=1)

    df = pd.DataFrame(data=a,columns=[name for name, _ in methods.iteritems()] + [name for name, _ in binning.iteritems()])
    df.to_pickle(modeldir+"results_%s%s.pkl"%(args.region,args.inferencefile))

### Sample class
class Sample(object):
    def __init__(self, name, base):
        self.name = name 

        self.X = np.load('%s/%s_%s%s.pkl'%(base,'X',args.region,args.inferencefile),allow_pickle=True)[:Nrhs]
        self.X.drop(['PU','pt',],1,inplace=True)
        depth = np.load('%s/%s_%s%s.pkl'%(base,'X',args.region,args.inferencefile),allow_pickle=True)[:Nrhs][['depth']]
        depth = np_utils.to_categorical(depth,num_classes=8)
        depth = pd.DataFrame(depth, columns=['depth%i'%i for i in range(0,8)])
      
        ieta = abs(np.load('%s/%s_%s%s.pkl'%(base,'X',args.region,args.inferencefile),allow_pickle=True)[:Nrhs][['ieta']])
        
        ieta = np_utils.to_categorical(ieta,num_classes=30)
        ieta = pd.DataFrame(ieta, columns=['ieta%i'%i for i in range(0,30)])

        print depth, ieta
        self.X = self.X.join([depth,ieta])

       

        self.Y = np.load('%s/%s_%s%s.pkl'%(base,'Y',args.region,args.inferencefile),allow_pickle=True)[:Nrhs]
        
        #self.Y['genE'][self.X['depth'] == 1.] *= 0.5 # self.Y['genE'][self.X['depth'] == 1.]*4./12.

       

        if args.region == 'HB': self.X.drop(['depth','depth0','depth5','depth6','depth7'],1,inplace=True)
        else: self.X.drop(['depth','depth0','ieta','ieta0'],1,inplace=True)
        self.kin = np.load('%s/%s_%s%s.pkl'%(base,'X',args.region,args.inferencefile),allow_pickle=True)[:Nrhs][['PU','ieta','iphi','pt','depth']]
        print self.X.columns
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

    def infer(self, model):
        self.Yhat = model.predict(self.X)


    def standardize(self, mu, std):
        self.X = (self.X - mu) / std

### Model class
class ClassModel(object):
    def __init__(self, n_inputs):
        self._hidden = 0
        self.n_inputs = n_inputs
        self.n_targets = 1

        self.inputs = Input(shape=(n_inputs,), name='input')
        h = self.inputs

        if args.region == 'HB': 
         h = BatchNormalization(momentum=0.6)(h)
         h = Dense(12,activation='relu')(h)
         drop = Dropout(0.6)(h)
         h = BatchNormalization(momentum=0.2)(h)
         #h = Dense(12,activation='relu')(h)
         #drop = Dropout(0.6)(h)
         #h = BatchNormalization(momentum=0.2)(h)
         #h = Dense(7,activation='relu')(h)
         #h = BatchNormalization(momentum=0.6)(h)
         h = Dense(5,activation='relu')(h)
         h = BatchNormalization(momentum=0.6)(h)
         h = Dense(3,activation='relu')(h)
         h = BatchNormalization(momentum=0.6)(h)
         self.outputs = Dense(1,activation='relu',name='output')(h)


	if args.region == 'HE' or args.region == 'all':
         #h = BatchNormalization(momentum=0.6)(h)
         #h = Dense(40, activation='relu')(h)
         #drop = Dropout(0.1)(h)
         #norm = BatchNormalization(momentum=0.6)(h)
         #h = Dense(20, activation = 'relu')(norm)
         #norm = BatchNormalization(momentum=0.6)(h)
         #h = Dense(31, activation = 'relu')(h)
         #norm = BatchNormalization(momentum=0.6)(h)
         #h = Dense(11, activation = 'relu')(h)
         #norm = BatchNormalization(momentum=0.6)(h)
         #h = Dense(3, activation = 'relu')(h)
         #norm = BatchNormalization(momentum=0.6)(h)


         norm = BatchNormalization(momentum=0.6)(h)
         #h = Dense(200, activation = 'relu')(norm)
         #h = Dense(100, activation = 'relu')(norm)
         h = Dense(30, activation = 'relu')(norm)
         h = Dense(20, activation = 'relu')(h)
         h = Dense(10, activation = 'relu')(h)
         h = Dense(5, activation = 'relu')(h)


         self.outputs = Dense(1, activation='linear', name='output')(h)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')
        self.es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
        #mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        plot_model(self.model,to_file='models/v%i/model_%s.eps'%(args.version,args.region),show_shapes=True)
        plot_model(self.model,to_file='models/v%i/model_%s.svg'%(args.version,args.region),show_shapes=True)
        self.model.summary()

    def train(self, sample):

        print sample.X.values
        tX = sample.X.values[sample.tidx]
        vX = sample.X.values[sample.vidx]
        tY = sample.Y[['genE']].values[sample.tidx]
        vY = sample.Y[['genE']].values[sample.vidx] 
        history = self.model.fit(tX, tY, 
                                 batch_size=500, epochs=100, shuffle=True,
                                 validation_data=(vX, vY,),
				 callbacks=[self.es])
        model_json = self.model.to_json()	
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

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

        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in self.model.outputs])

        tf.train.write_graph(frozen_graph, '/'.join(path.split('/')[:-1]), path.split('/')[-1], as_text=False)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def load_model(self, path):
        self.model = load_model(path)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--region', type=str, default='HE')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=4)
    parser.add_argument('--inferencefile', type=str, default='')
    args = parser.parse_args()

    modeldir = 'models/v%i/'%(args.version)
    basedir = modeldir
    figsdir =  basedir+'plots/'
    sample = Sample("RecHits", basedir)
    n_inputs = sample.X.shape[1]


    model = ClassModel(n_inputs)

    if args.train:
        print 'Training...'
        model.train(sample)
        model.save_as_keras(modeldir+'/weights_%s.h5'%args.region)
        model.save_as_tf(modeldir+'/graph_%s.pb'%args.region)
    else:
        print 'Loading...'
        model.load_model(modeldir+'weights_%s.h5'%args.region)

    if args.plot:
        print 'Inferring...'

        sample.infer(model)
        shape = (sample.Y['genE'].values.shape[0],1)
        methods = {
                   "genE": sample.Y['genE'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "DNN" : sample.Yhat,
 		   "Mahi": sample.Y['energy'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "M0"  : sample.Y['eraw'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "M3"  : sample.Y['em3'].values.reshape(sample.kin['ieta'].shape[0],1)
		   }

        binning = {
		   "ieta" : sample.kin['ieta'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "iphi" : sample.kin['iphi'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "PU"   : sample.kin['PU'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "pt"   : sample.kin['pt'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "depth": sample.kin['depth'].values.reshape(sample.kin['ieta'].shape[0],1)
	 	  }

        savepickle(methods,binning,modeldir)


