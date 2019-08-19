import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle
import ROOT
import array
from ROOT import gROOT,gPad

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

def getQuantiles(th, quantiles):
    probSum = array.array('d',[quantiles[0],quantiles[1]])
    q = array.array('d',[0.0]*len(probSum))
    th.GetQuantiles(len(probSum),q,probSum)
    return q[0],q[1]

def rootfit(methodarr, genarr, methodname, purange, ptrange, figdir):
    name = "%s_%.0f_pu_%.0f_%.0f_pt_%.0f"%(methodname,purange[0],purange[1],ptrange[0],ptrange[1])
    tmp = ROOT.TH1F(name+"tmp",name+"tmp",500,-300.,300.)

    for it,val in enumerate(methodarr):
        tmp.Fill(val - genarr[it])

    q1,q2 = getQuantiles(tmp,[0.02,0.98])
    bq1 = tmp.FindBin(q1)
    bq2 = tmp.FindBin(q2)
 
    # only fit 2nd to 98th quantile
    th = ROOT.TH1F(name,name,bq2 - bq1, q1, q2)
    for it0,it1 in enumerate(range(bq1, bq2)):
       th.SetBinContent(it0+1,tmp.GetBinContent(it1))
    th.Scale(1./th.Integral())

    func = ROOT.TF1(name,"gaus",q1,q2)
    result = th.Fit(func,"q s")
    if result.Status(): print "fit problem: in %.0f < PU < %.0f, %.0f < pt < %.0f"%(purange[0],purange[1],ptrange[0],ptrange[1])
    mu = th.GetMean()#func.GetParameter(1)
    std = th.GetStdDev()#func.GetParameter(2)
    axis = {"x" : "E_{%s} - E_{gen}"%(methodname), "y" : "Density"}
    title = "%s %.0f < PU < %.0f, %.0f < p_{T} < %.0f"%(methodname,purange[0],purange[1],ptrange[0],ptrange[1])
    drawTH1([th,func], title, axis, figdir, name)

    del tmp; del func; del th
    return mu, std

def drawTH1(figs, title, axes, figdir, filename, l0 = None):
    c0 = ROOT.TCanvas("c0","c0",800,600)
    for i, fig in enumerate(figs):
        try:
            fig.SetStats(0)
        except: pass
        if i == 0: fig.Draw("ape")
        else:      fig.Draw("same")

        if fig.GetName() == "Time":
            print '='*200+"Time graph!"
            c0.SetLogx()
        
        fig.SetTitle(title)
        fig.GetXaxis().SetTitle(axes["x"])
        fig.GetYaxis().SetTitle(axes["y"])

    if l0 is not None: l0.Draw() 

    c0.SaveAs(figdir+filename+".pdf")
    c0.SaveAs(figdir+filename+".png")
    

def performance(df, times, figdir):
    #define methods to plot, target variable, and binning variables
    methods     = ["Mahi","DNN","M3","M0"]
    target      = "genE"
    variables   = {
        "PU" : [10,35,45,70],
        "pt" : [1.,5.,10.,15.,20.,25.,30.,40.,60.,80.,100.],#1.,5.,7.,10.,13.,15.,17.,20.,30.,40.,50.,60.,70.,80.,90.,100.],
    }

    if len(variables) == 2:
        for it0 in range(len(variables["PU"])-1):
            mg_resp = ROOT.TMultiGraph()
            mg_reso = ROOT.TMultiGraph()
            g_time  = ROOT.TGraph(len(times["batches"]))
            l0 = ROOT.TLegend(0.65,0.65,0.9,0.9)

            col = 0
            for method in methods:
                hresolution = ROOT.TGraph(len(variables["pt"]) - 1)
                hresponse   = ROOT.TGraph(len(variables["pt"]) - 1)
                col+=1

                for it1 in range(len(variables["pt"])-1):
                    tmp = df[(df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["pt"] > variables["pt"][it1]) & (df["pt"] < variables["pt"][it1+1]) ][[method]] 
                    tmparr = tmp.values
                    genarr = df[(df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["pt"] > variables["pt"][it1]) & (df["pt"] < variables["pt"][it1+1]) ][['genE']].values 
                    mu, std = rootfit(tmparr,genarr, method, [variables["PU"][it0], variables["PU"][it0+1]], [ variables["pt"][it1],variables["pt"][it1+1]], figdir)
                    ptmean = variables["pt"][it1] + variables["pt"][it1+1] 
                    response_correction = 1 - mu/(ptmean/2.)
                    if response_correction == 0:
                        raise RuntimeError("Response was zero: cannot response correct")
                    elif response_correction < 1:
                        response_correction = 1 / response_correction
                    hresolution.SetPoint(it1, ptmean/2., std/(ptmean/2.)*response_correction)# Plot response-corrected curve
                    hresponse.SetPoint(it1,   ptmean/2., 1 - mu/(ptmean/2.))

                l0.AddEntry(hresolution,method)
                hresolution.SetMarkerSize(4)
                hresolution.SetMarkerStyle(col+1)
                hresolution.SetMarkerColor(col+1)
                hresolution.SetName(method)
                hresponse.SetMarkerSize(5)
                hresponse.SetMarkerStyle(col+1)
                hresponse.SetMarkerColor(col+1)
                hresponse.SetName(method)
                print "NAME2", hresolution.GetName()
                mg_resp.Add(hresponse)
                mg_reso.Add(hresolution)
                del hresolution; del hresponse

            for i in range(len(times['batches'])):
                g_time.SetPoint(i, times['batches'][i], times['times'][i])

            g_time.SetMarkerSize(3)
            g_time.SetMarkerStyle(2)
            g_time.SetMarkerColor(ROOT.kRed)
            
            c_time = ROOT.TCanvas("c0", "c0", 800, 600)
            try:
                g_time.SetStats(0)
            except:
                pass
            c_time.SetLogx()
            g_time.Draw("ape")
            g_time.SetTitle('Inference Time')
            g_time.GetXaxis().SetTitle('Batch size')
            g_time.GetYaxis().SetTitle('Inference time per rec hit (s)') 
            c_time.SaveAs(figdir+'timing.pdf')
            c_time.SaveAs(figdir+'timing.png')

            name = "%.0f < PU < %.0f"%(variables["PU"][it0],variables["PU"][it0+1])
            axis = {"y":"#sigma_{E}/E", "x": "p_{T} (GeV)"}
            drawTH1([mg_reso], name, axis, figdir, "resolution_%i"%it0,l0) 
            
            axis = {"y":"1 - #mu/E", "x": "p_{T} (GeV)"} 
            drawTH1([mg_resp], name, axis, figdir, "response_%i"%it0,l0)
            

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--pickle', dest="pickle", action='store')
    parser.add_argument('--figdir', dest="figdir", action='store')
    global args 
    args = parser.parse_args()
    gROOT.SetBatch(ROOT.kTRUE)
    _make_parent(args.figdir)
    for directory in os.listdir(args.pickle):
        if not os.path.isdir(args.pickle+directory): continue
        os.system('mkdir '+args.figdir+directory+'/')
        performance(pickle.load(open(args.pickle+directory+'/results.pkl','r')),
            pickle.load(open(args.pickle+directory+'/times.pkl','rb')),
            args.figdir+directory+'/',
        )
