import model_class as mc

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

class ModelDefault(mc.ClassModel):
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

class Model1(mc.ClassModel):
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



VALSPLIT = 0.2
np.random.seed(5)
Nrhs = 2100000

top_power = 4 # Max batch size: 10,000
BATCH_SIZES = [a for i in range(top_power) for a in range(10**i, 10**(i+1), 10**i)] + [10**top_power]

def get_mu_std(sample):
    mu = np.mean(sample.X, axis=0)
    std = np.std(sample.X, axis=0)
    return mu, std

'''def saveroot(methods,binning,modeldir):
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

    tf.Write()'''

def savepickle(methods, binning, times, modeldir):
    print times['times']
    print [arr for _, arr in methods.iteritems()]
    print [arr.shape for _, arr in methods.iteritems()]
    a = np.concatenate([arr for _, arr in methods.iteritems()] + 
                       [arr for _, arr in binning.iteritems()], axis=1)
    print a
    time_data = np.concatenate([arr for _, arr in times.iteritems()], axis=1)

    df = pd.DataFrame(data=a,columns=[name for name, _ in methods.iteritems()] + 
                                     [name for name, _ in binning.iteritems()])
    df_time = pd.DataFrame(data=time_data,columns=[name for name, _ in times.iteritems()])
    print df
    df.to_pickle(modeldir+"results.pkl")
    df_time.to_pickle(modeldir+"times.pkl")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=4)
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()

    basedir = 'output/'
    figsdir =  basedir+'plots/'
    modeldir = 'models/evt/v%i/'%(args.version)

    sample = mc.Sample("RecHits", basedir)
    n_inputs = sample.X.shape[1]

    print 'Standardizing...'
    mu, std = get_mu_std(sample)
    sample.standardize(mu, std)

    for Model in MODELS:
        model = Model(n_inputs)

        if args.train:
            print 'Training...'
            model.train(sample, args.epochs)
            model.save_as_keras(modeldir+model.name+'/weights.h5')
            model.save_as_tf(modeldir+model.name+'/graph.pb')
        else:
            print 'Loading...'
            model.load_model(modeldir+model.name+'weights.h5')

        if args.plot:
            print 'Inferring...'
            times = [0]*len(BATCH_SIZES)
            for trial_num in range(args.trials):
                print "Trial", trial_num
                for i in range(len(BATCH_SIZES)):
                    print BATCH_SIZES[i]
                    sample.infer(model, BATCH_SIZES[i])
                    times[i] += sample.time / args.trials / Nrhs
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

            times = {
                "batches": BATCH_SIZES,
                "times"  : times,
            }

            savepickle(methods, binning, times, modeldir+model.name+'/')


