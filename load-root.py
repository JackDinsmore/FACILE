# 64657374726f79206d616869
import argparse, root_numpy
import numpy as np
import os 
import ROOT
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument('--filename', help='Provide a filename to read', default='/data/t3home000/jkrupa/rh_studies/Out.root_skimmedRH2.root')
parser.add_argument('--outdir',   help='Provide an output directory')
parser.add_argument('--mahi', action='store_true', help='Set if you wish to include train to match mahi')

args = parser.parse_args()

#writing branches as list of tuples seems like an easy way to format the output
inputs = ['PU','pt','ieta','iphi','gain', 'inPedAvg', 'depth',
          'raw[0]','raw[1]','raw[2]','raw[3]','raw[4]','raw[5]','raw[6]','raw[7]']
          #'ped[0]','ped[1]','ped[2]','ped[3]','ped[4]','ped[5]','ped[6]','ped[7]',]
          #'inNoiseADC[0]','inNoiseADC[1]','inNoiseADC[2]','inNoiseADC[3]','inNoiseADC[4]',
          #'inNoiseADC[5]','inNoiseADC[6]','inNoiseADC[7]',
          #'inNoisePhoto[0]','inNoisePhoto[1]','inNoisePhoto[2]','inNoisePhoto[3]','inNoisePhoto[4]',
          #'inNoisePhoto[5]','inNoisePhoto[6]','inNoisePhoto[7]',]

x_branches = [(inp) for inp in inputs]
y_branches = ['genE','energy','em3','eraw']
if args.mahi:
    y_branches.append('mahiE')

def save(name, arr):
    os.system('mkdir -p %s'%('/'.join(args.outdir.split('/')[:-1])))
    arr.to_pickle(args.outdir+name+".pkl")

def load_root(filename, branches=None):

    rfile = ROOT.TFile.Open(filename)
    rtree = rfile.Get("Events")
    return root_numpy.tree2array(rtree, branches=branches)

if __name__ == '__main__':

    Xtmp = load_root(args.filename, x_branches)
    Ytmp = load_root(args.filename, y_branches[:4])
    Xarr = np.zeros(shape=(Xtmp.shape[0],len(x_branches)))
    Yarr = np.zeros(shape=(Ytmp.shape[0],len(y_branches)))
 
    #it comes out of load_root in an annoying way
    for it in range(Xtmp.shape[0]):
        Xarr[it] = np.array(list(Xtmp[it]))
        if args.mahi:
            for index, item in enumerate(Ytmp[it]):
                Yarr[it][index] = item
            Yarr[it][-1] = 0 # Make model match mahi
        else:
            Yarr[it] = np.array(list(Ytmp[it]))

    Xarr = pd.DataFrame(Xarr[:,:],columns=x_branches)
    Yarr = pd.DataFrame(Yarr[:,:],columns=y_branches)
    save('X',Xarr)
    save('Y',Yarr)
