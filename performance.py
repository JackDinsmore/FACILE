import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle
import ROOT
import array
from ROOT import gROOT,gPad
import tdrstyle

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

def getQuantiles(th,quantiles):
    probSum = array.array('d',[quantiles[0],quantiles[1]])
    q = array.array('d',[0.0]*len(probSum))
    th.GetQuantiles(len(probSum),q,probSum)
    return q[0],q[1]

def rootfit(methodarr, genarr, methodname, purange, ptrange):

    name = "%s_%.0f_pu_%.0f_%.0f_pt_%.0f"%(methodname,purange[0],purange[1],ptrange[0],ptrange[1])
    tmp = ROOT.TH1F(name+"tmp",name+"tmp",2500,-50.,50.)

    for it,val in enumerate(methodarr):
        tmp.Fill(val/genarr[it])

    tmparr = methodarr - genarr
    q1,q2 = getQuantiles(tmp,[0.02,0.98])
    bq1 = tmp.FindBin(q1)
    bq2 = tmp.FindBin(q2)
 
    # only fit 2nd to 98th quantile
    th = ROOT.TH1F(name,name,bq2 - bq1, bq1, bq2)
    for it0,it1 in enumerate(range(bq1, bq2)):
       th.SetBinContent(it0+1,tmp.GetBinContent(it1))
    try: th.Scale(1./th.Integral())
    except: pass
    #func = ROOT.TF1(name,"gaus",q1,q2)
    #result = th.Fit(func,"q s")
    #if result.Status(): print "fit problem: in %.0f < PU < %.0f, %.0f < pt < %.0f"%(purange[0],purange[1],ptrange[0],ptrange[1])
    mu = tmp.GetMean()#func.GetParameter(1)
    std = tmp.GetStdDev()#func.GetParameter(2)
    axis = {"x" : "E_{%s}/E_{gen}"%(methodname), "y" : "Density"}
    title = "%s %.0f < PU < %.0f, %.0f < p_{T} < %.0f"%(methodname,purange[0],purange[1],ptrange[0],ptrange[1])

    #c0 = ROOT.TCanvas("c0","c0",800,600)
    #th.GetXaxis().SetTitle(axis["x"])
    #th.GetYaxis().SetTitle(axis["y"])
    #th.Draw("ape")
    #c0.SaveAs(args.figdir+name+".pdf")
    #c0.SaveAs(args.figdir+name+".png")
    #del tmp; del th; #del func; del th
    return mu, std

def drawTH1(figs, title, axes, filename, l0 = None, info = None, lims=None):

    c0 = ROOT.TCanvas("c0","c0",800,600)
    addInfo = ROOT.TPaveText(0.5,0.72,0.7,0.9,"NDC")

    for i, fig in enumerate(figs):
       try:
         fig.SetStats(0)
       except: pass
       if i == 0: fig.Draw("ape")
       else:      fig.Draw("same")
       
       #fig.SetTitle(title)
       print type(fig)
       #if 'TF1' in str(type(fig)):
       fig.GetXaxis().SetTitleSize(0.05)
       fig.GetYaxis().SetTitleSize(0.05)
       fig.GetXaxis().SetLabelOffset(0.01)
       fig.GetYaxis().SetLabelOffset(0.01)
       fig.GetXaxis().SetTitle(axes["x"])
       fig.GetYaxis().SetTitle(axes["y"])
       if lims is not None: 
         fig.GetYaxis().SetRangeUser(lims["ylower"],lims["yupper"])
         fig.GetXaxis().SetRangeUser(lims["xlower"],lims["xupper"])

    if info is not None:
       for key,var in info.iteritems():
         addInfo.AddText(key+" "+var)
       if 'resolution' in filename: addInfo.AddText("Response corrected")
       addInfo.SetBorderSize(0)
       addInfo.SetFillStyle(0)
       addInfo.SetTextSize(0.03)
       addInfo.SetTextFont(42)
       addInfo.Draw()

    if l0 is not None: 
       l0.SetTextAlign(22)
       l0.SetBorderSize(2)
       l0.SetFillStyle(0)
       l0.SetTextSize(0.05)
       l0.SetTextFont(42)
       l0.Draw("p") 

    #c0.SetMargin(0.1,0.1,0.1,0.1)
    #c0.SetLeftMargin(0.15)
    #c0.SetBottomMargin(0.15)
    c0.SaveAs(args.figdir+filename+".pdf")
    c0.SaveAs(args.figdir+filename+".png")
      

def performance(df):


    #define methods to plot, target variable, and binning variables
    methods     = ["Mahi","DNN",]
    target      = "genE"
    variables   = {
		   "PU" : [1,200],
                   "genE" : [ 0.2,2.,4.,6.,8.,11.,14.,17.,20.,25.,40.,60.,80.,100.] # [0.1,1.,1.5,2.,3.,4.,5.,8.,11.,14.,17.,20.,30.,40.,50.,70.,100.,150.] # [0.1,1.,1.5,2.,3.,4.,5.,8.,11.,14.,17.,20.,25.] 60.,70.,80.,90.,110.],
	          }

    if len(variables) == 2:

 
       if 'HE' in args.pickle: ietal = [16,30]
       elif 'HB' in args.pickle: ietal = [0,15]
       elif 'all' in args.pickle: ietal = [0,30]
       ietal = [16,30] 
       for ietait, ieta in enumerate(ietal):

          if ietait == len(ietal) - 1: break
          for it0 in range(len(variables["PU"])-1):

             mg_resp  = ROOT.TMultiGraph()
             mg_reso  = ROOT.TMultiGraph()
             l0 = ROOT.TLegend(0.75,0.75,0.9,0.9)
 
             color = {"DNN" : 797,
  		      "Mahi": 418}
             col = 0
             for method in methods:
              hresolution = ROOT.TGraph(len(variables["genE"]) - 1)
              hresponse   = ROOT.TGraph(len(variables["genE"]) - 1)
              col+=1

              for it1 in range(len(variables["genE"])-1):


               if args.depth == 0.:
                genarr = df[(abs(df["ieta"]) >= ietal[ietait]) & (abs(df["ieta"]) <= ietal[ietait+1]) & (df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["genE"] > variables["genE"][it1]) & (df["genE"] < variables["genE"][it1+1]) ][['genE']].values 
                tmparr = df[(abs(df["ieta"]) >= ietal[ietait]) & (abs(df["ieta"]) <= ietal[ietait+1]) & (df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["genE"] > variables["genE"][it1]) & (df["genE"] < variables["genE"][it1+1]) ][[method]].values

               else:
                tmparr = df[(abs(df["ieta"]) >= ietal[ietait]) & (abs(df["ieta"]) <= ietal[ietait+1]) & (df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["genE"] > variables["genE"][it1]) & (df["genE"] < variables["genE"][it1+1]) & (df["depth"] == args.depth )][[method]].values 
                genarr = df[(abs(df["ieta"]) >= ietal[ietait]) & (abs(df["ieta"]) <= ietal[ietait+1]) & (df["PU"] > variables["PU"][it0]) & (df["PU"] < variables["PU"][it0+1]) & (df["genE"] > variables["genE"][it1]) & (df["genE"] < variables["genE"][it1+1]) & ( df["depth"] == args.depth )][['genE']].values 

               meangenE = np.mean(genarr)
               mu, std = rootfit(tmparr,genarr, method, [variables["PU"][it0], variables["PU"][it0+1]], [ variables["genE"][it1],variables["genE"][it1+1]])
               #response_correction = 1 - mu/meangenE
               response_correction = mu
               if response_correction < 1: response_correction = 1./response_correction
               hresolution.SetPoint(it1, meangenE, std*response_correction)
               hresponse.SetPoint(it1,   meangenE, mu)
               print mu, std*response_correction
              l0.AddEntry(hresolution,method)
              hresolution.SetMarkerSize(1.2)
              hresolution.SetMarkerStyle(20)
              hresolution.SetMarkerColor(color[method])
              hresolution.SetName(method)
              hresponse.SetMarkerSize(1.2)
              hresponse.SetMarkerStyle(20)
              hresponse.SetMarkerColor(color[method])
              hresponse.SetName(method)
              mg_resp.Add(hresponse)
              mg_reso.Add(hresolution)
              del hresolution; del hresponse

             name = "%.0f < PU < %.0f, %i < ieta < %i"%(variables["PU"][it0],variables["PU"][it0+1], ietal[ietait], ietal[ietait+1])
             info = {"Depth" : "all" if not args.depth else str(int(args.depth)),
		     "PU"    : "%i, %i"%(variables["PU"][it0],variables["PU"][it0+1]),
		     "i#eta" : "%i, %i"%(ietal[ietait], ietal[ietait+1])
		    }
             lims = {"ylower" : 0.0, "yupper" : 1.0, "xlower" : variables["genE"][0], "xupper" : variables["genE"][-1]}
             axis = {"y":"#sigma_{E}/E", "x": "E_{gen} (GeV)"}

             drawTH1([mg_reso], name, axis, "resolution_%iPU%i_%iIETA%i_%sDEPTH"%(variables["PU"][it0],variables["PU"][it0+1],ietal[ietait],ietal[ietait+1], "all" if args.depth == 0. else str(int(args.depth))),l0, info,lims) 
             
             lims = {"ylower" : 0.4, "yupper" : 1.5, "xlower" : variables["genE"][0], "xupper" : variables["genE"][-1]}
             axis = {"y":"Response", "x": "E_{gen} (GeV)"} 
             drawTH1([mg_resp], name, axis, "response_%iPU%i_%iIETA%i_%sDEPTH"%(variables["PU"][it0],variables["PU"][it0+1],ietal[ietait],ietal[ietait+1], "all" if args.depth == 0. else str(int(args.depth))),l0, info,lims)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--pickle', dest="pickle", action='store')
    parser.add_argument('--figdir', dest="figdir", action='store')
    parser.add_argument('--depth', dest="depth", action='store',type=float)
    parser.add_argument('--inferencefile', dest="inferencefile", action='store')
    global args 
    args = parser.parse_args()
    gROOT.SetBatch(ROOT.kTRUE)
    _make_parent(args.figdir)
    performance(pickle.load(open(args.pickle,'r')))
