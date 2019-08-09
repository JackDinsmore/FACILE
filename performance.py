import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle
import ROOT
import array
from ROOT import gROOT,gPad

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
    th.Scale(1./th.Integral())

    func = ROOT.TF1(name,"gaus",q1,q2)
    th.Fit(func)
    canv = ROOT.TCanvas(name,name,800,600)
    th.SetStats(0)
    th.SetTitle("%s %.1f < PU < %.1f, %.0f < p_T < %.0f"%(methodname,purange[0],purange[1],ptrange[0],ptrange[1]))
    th.GetXaxis().SetTitle("E_{%s} - E_{gen}"%(methodname))
    th.Draw("hist")
    func.Draw("l same")
    canv.SaveAs(name+".pdf")
    canv.SaveAs(name+".png")
    mu = th.GetMean()#func.GetParameter(1)
    std = th.GetStdDev()#func.GetParameter(2)

    print mu, std
    del tmp; del func; del th
    return mu, std

def performance(df):

    methods     = ["Mahi","DNN","M3","M0"]
    target      = "genE"
    variables   = {
		   "PU" : [20,40,60],
                   "pt" : [5.,10.,20.,30.,40.,50.,60.,70.],
	          }

    if len(variables) == 2:


          for it0 in range(len(variables["PU"])-1):

             mg_resp  = ROOT.TMultiGraph()
             mg_reso  = ROOT.TMultiGraph()
             l0 = ROOT.TLegend(0.65,0.65,0.9,0.9)
 
             col = 0
             for method in methods:
              hresolution = ROOT.TGraph(len(variables["pt"]) - 1)
              hresponse   = ROOT.TGraph(len(variables["pt"]) - 1)
              col+=1

              for it1 in range(len(variables["pt"])-1):

                tmp = df[(df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["pt"] > variables["pt"][it1]) & (df["pt"] < variables["pt"][it1+1]) ][[method]] 
                tmparr = tmp.values
                genarr = df[(df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["pt"] > variables["pt"][it1]) & (df["pt"] < variables["pt"][it1+1]) ][['genE']] 
                genarr = genarr.values
                mu, std = rootfit(tmparr,genarr, method, [variables["PU"][it0], variables["PU"][it0+1]], [ variables["pt"][it1],variables["pt"][it1+1]])
                print mu,std 
                ptmean = variables["pt"][it1] + variables["pt"][it1+1] 
                hresolution.SetPoint(it1, ptmean/2., std/(ptmean/2.))
                hresponse.SetPoint(it1,   ptmean/2., 1 - mu/(ptmean/2.))

              l0.AddEntry(hresolution,method)
              hresolution.SetMarkerSize(4)
              hresolution.SetMarkerStyle(col+1)
              hresolution.SetMarkerColor(col+1)
              hresponse.SetMarkerSize(5)
              hresponse.SetMarkerStyle(col+1)
              hresponse.SetMarkerColor(col+1)

              mg_resp.Add(hresponse)
              mg_reso.Add(hresolution)
              del hresolution; del hresponse

            

             c0 = ROOT.TCanvas("c0","c0",800,600)
             #gPad.DrawFrame(10.,0.01,90.,0.5)
             mg_reso.Draw("ape")
             mg_reso.SetTitle("%.0f < PU < %.0f"%(variables["PU"][it0],variables["PU"][it0+1]))
             mg_reso.GetYaxis().SetTitle("#sigma E/E")
             mg_reso.GetXaxis().SetTitle("p_{T} (GeV)")
             l0.Draw()
             c0.SaveAs("resolution_%ipu.pdf"%it0)
             c0.SaveAs("resolution_%ipu.png"%it0)
              
             c1 = ROOT.TCanvas("c1","c1",800,600)
             mg_resp.Draw("ape")
             mg_resp.Draw("ape")
             mg_resp.SetTitle("%.0f < PU < %.0f"%(variables["PU"][it0],variables["PU"][it0+1]))
             mg_resp.GetYaxis().SetTitle("1 - #mu/E")
             mg_resp.GetXaxis().SetTitle("p_{T} (GeV)")
             l0.Draw()
             c1.SaveAs("response_%ipu.pdf"%it0)
             c1.SaveAs("response_%ipu.png"%it0)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--pickle', dest="pickle", action='store')
    parser.add_argument('--figdir', action='store_true')
    args = parser.parse_args()
    print args
    gROOT.SetBatch(ROOT.kTRUE)
    performance(pickle.load(open(args.pickle,'r')))
