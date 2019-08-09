import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle
import ROOT
import array
from ROOT import gROOT

gROOT.SetBatch(0)
def getQuantiles(th,quantiles):
    probSum = array.array('d',[quantiles[0],quantiles[1]])
    q = array.array('d',[0.0]*len(probSum))
    th.GetQuantiles(len(probSum),q,probSum)
    return q[0],q[1]

def rootfit(methodarr, genarr, methodname, purange, ptrange):

    name = "%s_%.1f_pu_%.1f_%.1f_pt_%.1f"%(methodname,purange[0],purange[1],ptrange[0],ptrange[1])
    tmp = ROOT.TH1F(name+"tmp",name+"tmp",500,-300.,300.)

    for it,val in enumerate(methodarr):
        tmp.Fill(val - genarr[it])

    q1,q2 = getQuantiles(tmp,[0.02,0.98])
    bq1 = tmp.FindBin(q1)
    bq2 = tmp.FindBin(q2)

    th = ROOT.TH1F(name,name,bq2 - bq1, q1, q2)
    for it0,it1 in enumerate(range(bq1, bq2)):
       th.SetBinContent(it0+1,tmp.GetBinContent(it1))
    print th.Print("all")
    th.Scale(1./th.Integral())

    func = ROOT.TF1(name,"gaus")
    #func = ROOT.TF1("gaus1","crystalball",q1[0],q2[0])
    th.Fit(name)
 
    canv = ROOT.TCanvas(name,name,800,600)
    th.SetStats(0)
    th.SetTitle("%s %.1f < PU < %.1f, %.0f < p_T < %.0f"%(methodname,purange[0],purange[1],ptrange[0],ptrange[1]))
    th.GetXaxis().SetTitle("E_{%s} - E_{gen}"%(methodname))
    th.Draw("hist")
    func.Draw("same")
    canv.SaveAs(name+".pdf")
    canv.SaveAs(name+".png")
    mu = func.GetParameter(1)
    std = func.GetParameter(2)

    print mu, std
    del tmp; del func; del th

def performance(df):

    methods     = ["Mahi","DNN"]
    target      = "genE"
    variables   = {
		   "PU" : [20,40,60],
                   "pt" : [10.,30.,50.,70.],
	          }

    if len(variables) == 2:

       for method in methods:
          for it0 in range(len(variables["PU"])):
             for it1 in range(len(variables["pt"])):

                if it0 == len(variables["PU"]) - 1: continue
                if it1 == len(variables["pt"]) - 1: continue
                tmp = df[(df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["pt"] > variables["pt"][it1]) & (df["pt"] < variables["pt"][it1+1]) ][[method]] 
                tmparr = tmp.values
                genarr = df[(df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["pt"] > variables["pt"][it1]) & (df["pt"] < variables["pt"][it1+1]) ][['genE']] 
                genarr = genarr.values
                rootfit(tmparr,genarr, method, [variables["PU"][it0], variables["PU"][it0+1]], [ variables["pt"][it1],variables["pt"][it1+1]])
  

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--pickle', dest="pickle", action='store')
    parser.add_argument('--figdir', action='store_true')
    args = parser.parse_args()
    print args
    performance(pickle.load(open(args.pickle,'r')))
