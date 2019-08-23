import model_class as mc

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU, Dropout

import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle 
import ROOT 


class ModelDefaultDropoutLow(mc.ClassModel):
    def get_outputs(self):
        self.name = 'defaultDropLow'
        self.epochs = 11
        RATE=0.2
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(100, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(20, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class ModelDefaultDropoutLLow(mc.ClassModel):
    def get_outputs(self):
        self.name = 'defaultDropLLow'
        self.epochs = 11
        RATE=0.05
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(100, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(20, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class ModelDefaultDropoutLLLow(mc.ClassModel):
    def get_outputs(self):
        self.name = 'defaultDropHighLLLow'
        self.epochs = 11
        RATE=0.01
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(100, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(20, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class ModelDefaultDropoutLLLLow(mc.ClassModel):
    def get_outputs(self):
        self.name = 'defaultDropLLLLow'
        self.epochs = 11
        RATE=0.001
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(100, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(20, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class ModelDefault(mc.ClassModel):
    def get_outputs(self):
        self.name = 'default'
        self.epochs = 11
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

class ModelOpen6(mc.ClassModel):
    def get_outputs(self):
        self.name = 'open6'
        self.epochs = 10
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(100, activation = 'relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(20, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)


class ModelHigherMomentumDefault(mc.ClassModel):
    def get_outputs(self):
        self.name = 'highMomentumDefault'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization()(h)
        h = Dense(n_inputs, activation='relu')(h)
        h = BatchNormalization()(h)
        h = Dense(100, activation = 'relu')(h)
        h = BatchNormalization()(h)
        h = Dense(50, activation = 'relu')(h)
        h = BatchNormalization()(h)
        h = Dense(20, activation = 'relu')(h)
        h = BatchNormalization()(h)
        h = Dense(10, activation = 'relu')(h)
        h = BatchNormalization()(h)
        h = Dense(5, activation = 'relu')(h)
        return Dense(1, activation='linear', name='output')(h)

class ModelShrinkDefault(mc.ClassModel):
    def get_outputs(self):
        self.name = 'shrinkDefault'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(25, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(5, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(3, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class ModelShrink7(mc.ClassModel):
    def get_outputs(self):
        self.name = 'shrink7'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(25, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(10, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model7(mc.ClassModel):
    def get_outputs(self):
        self.name = '7layers'
        self.epochs=11
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(100, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(15, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model6(mc.ClassModel):
    def get_outputs(self):
        self.name = '6layers'
        self.epochs = 11
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

class Model5(mc.ClassModel):
    def get_outputs(self):
        self.name = '5layers'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(75, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(25, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(5, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4Exp(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersExp'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(36, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(11, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(3, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4ExpLow(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersExpLow'
        RATE=0.05
        self.epochs = 11
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(36, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(11, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(3, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4ExpLLow(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersExpLLow'
        RATE=0.01
        self.epochs = 11
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(36, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(11, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(3, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4ExpNaught(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersExpNaught'
        RATE=0
        self.epochs = 11
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(36, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(11, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(3, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4ExpLLLow(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersExpLLLow'
        RATE=0.001
        self.epochs = 11
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(36, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(11, activation = 'relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(3, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4Exp(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersExp'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(36, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(11, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(3, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layers'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(7, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model4LowMomentum(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersLowMomentum'
        self.epochs = 11
        h = self.inputs
        h = BatchNormalization(momentum=0.1)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.1)(h)
        h = Dense(50, activation = 'relu')(norm)
        norm = BatchNormalization(momentum=0.1)(h)
        h = Dense(7, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model3(mc.ClassModel):
    def get_outputs(self):
        self.name = '3layers'
        self.epochs = 10
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model3Narrow(mc.ClassModel):
    def get_outputs(self):
        self.name = '3layersNarrow'
        self.epochs = 10
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(18, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model3Wide(mc.ClassModel):
    def get_outputs(self):
        self.name = '3layersWide'
        self.epochs = 10
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(100, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model3HighMomentum(mc.ClassModel):
    def get_outputs(self):
        self.name = '3layersHighMomentum'
        self.epochs = 10
        h = self.inputs
        h = BatchNormalization()(h)
        h = Dense(n_inputs, activation='relu')(h)
        norm = BatchNormalization()(h)
        h = Dense(50, activation = 'relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model3Tanh(mc.ClassModel):
    def get_outputs(self):
        self.name = '3layersTanh'
        self.epochs = 10
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='tanh')(h)
        norm = BatchNormalization(momentum=0.6)(h)
        h = Dense(50, activation = 'tanh')(norm)
        return Dense(1, activation='linear', name='output')(h)

class Model2(mc.ClassModel):
    def get_outputs(self):
        self.name = '2layers'
        self.epochs = 13
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(n_inputs, activation='relu')(h)
        return Dense(1, activation='linear', name='output')(h)

class ModelDumb(mc.ClassModel):
    def get_outputs(self):
        self.name = 'dumb'
        self.epochs = 20
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = Dense(2, activation='relu')(h)
        return Dense(1, activation='linear', name='output')(h)

MODELS = [Model4ExpLLLow, Model4ExpNaught]



VALSPLIT = 0.2
np.random.seed(5)
Nrhs = 2100000

BATCH_SIZES = [ 2 ** i for i in range(8, 17) ]

def get_mu_std(sample):
    mu = np.mean(sample.X, axis=0)
    std = np.std(sample.X, axis=0)
    return mu, std

def savepickle(methods, binning, times, modeldir):
    a = np.concatenate([arr for _, arr in methods.iteritems()] + 
                       [arr for _, arr in binning.iteritems()], axis=1)


    df = pd.DataFrame(data=a,columns=[name for name, _ in methods.iteritems()] + 
                                     [name for name, _ in binning.iteritems()])

    df.to_pickle(modeldir+"results.pkl")
    with open(modeldir+'times.pkl', 'wb') as handle:
        pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--mahi', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=4)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--datadir', type=str, default='output/')
    parser.add_argument('--region', type=str, default='')
    args = parser.parse_args()

    basedir = args.datadir
    figsdir =  basedir+'plots/'
    modeldir = 'models/evt/v%i/'%(args.version)

    sample = mc.Sample("RecHits", basedir, region=args.region)
    n_inputs = sample.X.shape[1]
    
    print 'N_INPUTS:', n_inputs

    print 'Standardizing...'
    mu, std = get_mu_std(sample)
    sample.standardize(mu, std)

    for Model in MODELS:
        model = Model(n_inputs, args.gpu)
        print '='*50, model.name, '='*50

        if args.train:
            print 'Training...'
            if args.epochs == 0:
                model.train(sample, mahi=args.mahi)
            else:
                model.train(sample, num_epochs=args.epochs, mahi=args.mahi)
            model.save_as_keras(modeldir+model.name+'/weights.h5')
            model.save_as_tf(modeldir+model.name+'/graph.pb')
        else:
            print 'Loading...'
            model.load_model(modeldir+model.name+'/weights.h5')

        if args.plot:
            print 'Inferring...'
            times = [0]*len(BATCH_SIZES)
            for trial_num in range(args.trials):
                print "Trial", trial_num
                for i in range(len(BATCH_SIZES)):
                    print BATCH_SIZES[i],
                    sample.infer(model, BATCH_SIZES[i])
                    times[i] += sample.time / args.trials / min(Nrhs, sample.X.shape[0])
            shape = (sample.Y['genE'].values.shape[0],1)
            print shape
            methods = {
                "genE": sample.Y['genE'].values.reshape(-1,1),
                "DNN" : sample.Yhat,
                "Mahi": sample.Y['energy'].values.reshape(-1,1),
                "M0"  : sample.Y['eraw'].values.reshape(-1,1),
                "M3"  : sample.Y['em3'].values.reshape(-1,1)
            }

            binning = {
                "ieta" : sample.kin['ieta'].values.reshape(-1,1),
                "iphi" : sample.kin['iphi'].values.reshape(-1,1),
                "PU"   : sample.kin['PU'].values.reshape(-1,1),
                "pt"   : sample.kin['pt'].values.reshape(-1,1)
            }

            times = {
                "batches": BATCH_SIZES,
                "times"  : times,
            }

            savepickle(methods, binning, times, modeldir+model.name+'/')
            K.clear_session()


