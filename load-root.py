import argparse, root_numpy
import numpy as np
import os 
import ROOT
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument('--HE', dest='HE',action='store_true', default=False, help='Provide a filename to read')
parser.add_argument('--HB', dest='HB',action='store_true', default=False, help='Provide a filename to read')
parser.add_argument('--filename', help='Provide a filename to read')
parser.add_argument('--outdir',   help='Provide an output directory')
#parser.add_argument('--branches', help='If you want, provide some branches to filter.', default='')

args = parser.parse_args()

#writing branches as list of tuples seems like an easy way to format the output
inputs = ['PU','depth','pt','ieta','iphi','gain','lambda','darkCurrent','inPedAvg',
          'raw[0]','raw[1]','raw[2]','raw[3]','raw[4]','raw[5]','raw[6]','raw[7]',]
          #'ped[0]','ped[1]','ped[2]','ped[3]','ped[4]','ped[5]','ped[6]','ped[7]',
          #'inNoiseADC[0]','inNoiseADC[1]','inNoiseADC[2]','inNoiseADC[3]','inNoiseADC[4]',
          #'inNoiseADC[5]','inNoiseADC[6]','inNoiseADC[7]',]
          #'inNoisePhoto[0]','inNoisePhoto[1]','inNoisePhoto[2]','inNoisePhoto[3]','inNoisePhoto[4]',
          #'inNoisePhoto[5]','inNoisePhoto[6]','inNoisePhoto[7]',]

x_branches = [(inp) for inp in inputs]
#x_branches = [('PU'),('depth'),('pt'),('ieta'),('gain'),('iphi'),('raw[0]'),('raw[1]'),('raw[2]'),('raw[3]'),('raw[4]'),('raw[5]'),('raw[6]'),('raw[7]')('ped[0]'),('ped[1]'),('ped[2]'),('ped[3]'),('ped[4]'),('ped[5]'),('ped[6]'),('ped[7]')]
y_branches = ['genE','energy','em3','eraw']

def save(name, arr):
    os.system('mkdir -p %s'%('/'.join(args.outdir.split('/')[:-1])))
    arr.to_pickle(args.outdir+name+".pkl")

def load_root(filename="Out.root_skimmedRH.root", branches=None):

    rfile = ROOT.TFile.Open(filename)
    rtree = rfile.Get("Events")

    if args.HE:    return root_numpy.tree2array(rtree, selection='abs(ieta) > 17', branches=branches)
    elif args.HB:  return root_numpy.tree2array(rtree, selection='abs(ieta) < 17', branches=branches)

if __name__ == '__main__':

    Xtmp = load_root(args.filename, x_branches)
    Ytmp = load_root(args.filename, y_branches)
    Xarr = np.zeros(shape=(Xtmp.shape[0],len(x_branches)))
    Yarr = np.zeros(shape=(Ytmp.shape[0],len(y_branches)))
 
    #it comes out of load_root in an annoying way
    for it in range(Xtmp.shape[0]):
        Xarr[it] = np.array(list(Xtmp[it]))
        Yarr[it] = np.array(list(Ytmp[it]))

    Xarr = pd.DataFrame(Xarr[:,:],columns=x_branches)
    Yarr = pd.DataFrame(Yarr[:,:],columns=y_branches)

    if args.HE:
      save('X_HE',Xarr)
      save('Y_HE',Yarr)
    elif args.HB:
      save('X_HB',Xarr)
      save('Y_HB',Yarr)
