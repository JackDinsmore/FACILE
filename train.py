#!/usr/bin/env python
# 64657374726f79206d616869
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

VALSPLIT = 0.2 #0.7
np.random.seed(5)
Nrhs = 180000

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

def get_mu_std(sample):

    #sample.X = np.reshape(sample.X, (sample.X.shape[0],sample.X.shape[1]))
    mu = np.mean(sample.X, axis=0)
    std = np.std(sample.X, axis=0)
    return mu, std

def saveroot(methods,binning,modeldir):

    tf = ROOT.TFile.Open(modeldir+"results.root","RECREATE")

    for name, arr in methods.iteritems():
       th = ROOT.TH3F(name,name,2,0.,72.,2,17.,30.,2,20.,80.)
       
       for it, rh in enumerate(np.nditer(arr)):
           ieta = abs(binning["ieta"][it])
           iphi = binning["iphi"][it]
           PU   = binning["PU"][it]
          
           th.Fill(ieta,PU,arr[it])

       tf.cd()
       th.Write(name)

    tf.Write() 

def savepickle(methods,binning,modeldir):

    print [arr for _,arr in methods.iteritems()]
    print [arr.shape for _,arr in methods.iteritems()]
    a = np.concatenate([arr for _, arr in methods.iteritems()] + [arr for _,arr in binning.iteritems()],axis=1)
    print a  

    df = pd.DataFrame(data=a,columns=[name for name, _ in methods.iteritems()] + [name for name, _ in binning.iteritems()])
    print df
    df.to_pickle(modeldir+"results_%s.pkl"%args.region)

### Sample class
class Sample(object):
    def __init__(self, name, base):
        self.name = name 

        self.X = np.load('%s/%s_%s.pkl'%(base,'X',args.region),allow_pickle=True)[:Nrhs]
        self.X.drop(['PU','pt'],1,inplace=True)
        self.Y = np.load('%s/%s_%s.pkl'%(base,'Y',args.region),allow_pickle=True)[:Nrhs]

        print self.X.shape, self.Y.shape
        self.kin = np.load('%s/%s_%s.pkl'%(base,'X',args.region),allow_pickle=True)[:Nrhs][['PU','ieta','iphi','pt']]

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
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(36, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(11, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(3, activation = 'relu')(norm)
        self.outputs = Dense(1, activation='linear', name='output')(h)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(), loss='mse')

        self.model.summary()

    def train(self, sample):

        tX = sample.X.values[sample.tidx]
        vX = sample.X.values[sample.vidx]
        tY = sample.Y[['genE']].values[sample.tidx]
        vY = sample.Y[['genE']].values[sample.vidx] 

        history = self.model.fit(tX, tY, 
                                 batch_size=1024, epochs=40, shuffle=True,
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
        return self.model.predict(*args, **kwargs)

    def load_model(self, path):
        self.model = load_model(path)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--region', type=str, default='HE')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=4)
    args = parser.parse_args()

    modeldir = 'models/v%i/'%(args.version)
    basedir = modeldir
    figsdir =  basedir+'plots/'
    sample = Sample("RecHits", basedir)
    n_inputs = sample.X.shape[1]

    print 'Standardizing...'
    mu, std = get_mu_std(sample)
    #sample.standardize(mu, std)

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
        print sample.infer(model)
        print sample.X.shape
        print sample.Yhat.shape
        print sample.Y['energy'].shape
        print sample.Y['eraw'].shape
        print sample.Y['em3'].shape

        shape = (sample.Y['genE'].values.shape[0],1)
        print shape
        methods = {
                   "genE": sample.Y['genE'].values.reshape(Nrhs,1),
		   "DNN" : sample.Yhat,
 		   "Mahi": sample.Y['energy'].values.reshape(Nrhs,1),
		   "M0"  : sample.Y['eraw'].values.reshape(Nrhs,1),
		   "M3"  : sample.Y['em3'].values.reshape(Nrhs,1)
		   }

        print sample.Y['energy'].values.reshape(Nrhs,1)[0:10]
        print sample.Y['genE'].values.reshape(Nrhs,1)[0:10]

        binning = {
		   "ieta" : sample.kin['ieta'].values.reshape(Nrhs,1),
		   "iphi" : sample.kin['iphi'].values.reshape(Nrhs,1),
		   "PU"   : sample.kin['PU'].values.reshape(Nrhs,1),
		   "pt"   : sample.kin['pt'].values.reshape(Nrhs,1)
	 	  }

        savepickle(methods,binning,modeldir)


