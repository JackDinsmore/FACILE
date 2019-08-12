from model_class import ClassModel, Sample

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU

import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle 
import ROOT 

class ModelDefault(ClassModel):
    def get_outputs(self):
        self.name = 'default'
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(100, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(20, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model1(ClassModel):
    def get_outputs(self):
        self.name = '4layers'
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(100, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(20, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

MODELS = [ModelDefault, Model1]



VALSPLIT = 0.2 #0.7
np.random.seed(5)
Nrhs = 2100000

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
    df.to_pickle(modeldir+"results.pkl")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=4)
    args = parser.parse_args()

    basedir = 'output/'
    figsdir =  basedir+'plots/'
    modeldir = 'models/evt/v%i/'%(args.version)

    sample = Sample("RecHits", basedir)
    n_inputs = sample.X.shape[1]

    print 'Standardizing...'
    mu, std = get_mu_std(sample)
    sample.standardize(mu, std)

    for Model in MODELS:
        model = Model(n_inputs)

        if args.train:
            print 'Training...'
            model.train(sample)
            model.save_as_keras(modeldir+model.name+'/weights.h5')
            model.save_as_tf(modeldir+model.name+'/graph.pb')
        else:
            print 'Loading...'
            model.load_model(modeldir+model.name+'weights.h5')

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

            savepickle(methods,binning,modeldir+model.name)


