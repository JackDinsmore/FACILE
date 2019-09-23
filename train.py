#!/usr/bin/env python
# 64657374726f79206d616869
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

VALSPLIT = 0.3 #0.7	
np.random.seed(4)
Nrhs = 600000

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

def get_mu_std(sample):

    #sample.X = np.reshape(sample.X, (sample.X.shape[0],sample.X.shape[1]))
    mu = np.mean(sample.X, axis=0)
    std = np.std(sample.X, axis=0)
    return mu, std


def savepickle(methods,binning,modeldir):

    print [arr for _,arr in methods.iteritems()]
    print [arr.shape for _,arr in methods.iteritems()]
    a = np.concatenate([arr for _, arr in methods.iteritems()] + [arr for _,arr in binning.iteritems()],axis=1)
    print a  

    df = pd.DataFrame(data=a,columns=[name for name, _ in methods.iteritems()] + [name for name, _ in binning.iteritems()])
    print df
    df.to_pickle(modeldir+"results_%s%s.pkl"%(args.region,args.inferencefile))

### Sample class
class Sample(object):
    def __init__(self, name, base):
        self.name = name 

        self.X = np.load('%s/%s_%s%s.pkl'%(base,'X',args.region,args.inferencefile),allow_pickle=True)[:Nrhs]
        self.X.drop(['PU','pt',],1,inplace=True)
        print np.unique(self.X['ieta'].values)
        depth = np.load('%s/%s_%s%s.pkl'%(base,'X',args.region,args.inferencefile),allow_pickle=True)[:Nrhs][['depth']]
        #ieta = np.load('%s/%s_%s.pkl'%(base,'X',args.region),allow_pickle=True)[:Nrhs][['ieta']]
        #iphi = np.load('%s/%s_%s.pkl'%(base,'X',args.region),allow_pickle=True)[:Nrhs][['iphi']]
        #enc = OneHotEncoder(handle_unknown='ignore').fit(depth)

        #enc.transform(depth).toarray()
        #ieta = np_utils.to_categorical(ieta)
        #print ieta.shape
        #iphi = np_utils.to_categorical(iphi)
        depth = np_utils.to_categorical(depth,num_classes=8)
        depth = pd.DataFrame(depth, columns=['depth%i'%i for i in range(0,8)])
        #ieta = pd.DataFrame(ieta, columns=[['ieta%i'%i for i in range(-30,30)]])
        #iphi = pd.DataFrame(iphi, columns=[['iphi%i'%i for i in range(0,73)]])#,dtype='int')
        #self.X = np.concatenate((self.X,depth),axis=1)
        self.X = self.X.join([depth])

       

        self.Y = np.load('%s/%s_%s%s.pkl'%(base,'Y',args.region,args.inferencefile),allow_pickle=True)[:Nrhs]


        #self.Y['genE'][self.X['depth'] == 1.] *= 0.5 # self.Y['genE'][self.X['depth'] == 1.]*4./12.

       

        self.X.drop(['depth','depth0'],1,inplace=True)
        print self.X.columns
        self.kin = np.load('%s/%s_%s%s.pkl'%(base,'X',args.region,args.inferencefile),allow_pickle=True)[:Nrhs][['PU','ieta','iphi','pt','depth']]

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
         #h = BatchNormalization(momentum=0.2)(h)
         #h = Dense(7,activation='relu')(h)
         drop = Dropout(0.4)(h)
         h = BatchNormalization(momentum=0.6)(h)
         h = Dense(5,activation='relu')(h)
         h = BatchNormalization(momentum=0.6)(h)
         h = Dense(3,activation='relu')(h)
         h = BatchNormalization(momentum=0.6)(h)
         self.outputs = Dense(1,activation='linear',name='output')(h)


	if args.region == 'HE':
         #h = BatchNormalization(momentum=0.6)(h)
         #h = Dense(40, activation='relu')(h)
         #drop = Dropout(0.1)(h)
         norm = BatchNormalization(momentum=0.6)(h)
         #h = Dense(20, activation = 'relu')(norm)
         #norm = BatchNormalization(momentum=0.6)(h)
         h = Dense(31, activation = 'relu')(norm)
         norm = BatchNormalization(momentum=0.6)(h)
         h = Dense(11, activation = 'relu')(norm)
         norm = BatchNormalization(momentum=0.6)(h)
         h = Dense(3, activation = 'relu')(norm)
         norm = BatchNormalization(momentum=0.6)(h)
         self.outputs = Dense(1, activation='linear', name='output')(h)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')
        self.es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        #mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        plot_model(self.model,to_file='models/v%i/model_%s.eps'%(args.version,args.region),show_shapes=True)
        plot_model(self.model,to_file='models/v%i/model_%s.svg'%(args.version,args.region),show_shapes=True)
        self.model.summary()

    def train(self, sample):

        tX = sample.X.values[sample.tidx]
        vX = sample.X.values[sample.vidx]
        tY = sample.Y[['genE']].values[sample.tidx]
        vY = sample.Y[['genE']].values[sample.vidx] 

        history = self.model.fit(tX, tY, 
                                 batch_size=5000, epochs=200, shuffle=True,
                                 validation_data=(vX, vY,),
				 callbacks=[self.es])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.savefig('loss.pdf')
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

        #print sample.Y['energy'].values.reshape(Nrhs,1)[0:10]
        #print sample.Y['genE'].values.reshape(Nrhs,1)[0:10]

        binning = {
		   "ieta" : sample.kin['ieta'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "iphi" : sample.kin['iphi'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "PU"   : sample.kin['PU'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "pt"   : sample.kin['pt'].values.reshape(sample.kin['ieta'].shape[0],1),
		   "depth": sample.kin['depth'].values.reshape(sample.kin['ieta'].shape[0],1)
	 	  }

        savepickle(methods,binning,modeldir)


