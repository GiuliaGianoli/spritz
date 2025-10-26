# ruff: noqa: E501

import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path
from CollinsSoper import build_4vector, cos_theta_collins_soper
from spritz.modules.lepton_sel import createLepton, leptonSel
year = "Full2022EEv12"

fw_path = get_fw_path()

with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

with open(f"{fw_path}/data/{year}/cfg.json") as file:
    cfg = json.load(file)

lumi = lumis[year]["tot"]
plot_label = "Fake"
year_label = "2022EE"
njobs = 1500

datasets = {}

datasets["DY"] = {
    "files": "DYto2L-2Jets_MLL-50",
    "task_weight": 8,
    "max_chunks": 100,
}

datasets["WJets"] = {
    "files": "WToLNu-2Jets",
    "task_weight": 8,
    "max_chunks": 100,
}

lumi_ele_low_pt   = "(ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt<=25, False)&(events.HLT.Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30>0.5) )"
lumi_ele_high_pt  = "((events.HLT.Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30>0.5) & ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt> 25, False))"
lumi_muon_low_pt  = "((events.HLT.Mu8_TrkIsoVVL>0.5)  & ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt<=20, False))"
lumi_muon_high_pt = "((events.HLT.Mu17_TrkIsoVVL>0.5) & ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt> 20, False))"
lumi_full_2022    = "(ak.ones_like(events.run, dtype=bool))"

datasets["DY"]["subsamples"] = {
    "ele_low_pt": (lumi_ele_low_pt, "(20.228/1000.)/27.007197591") ,
    "ele_high_pt": (lumi_ele_high_pt, "(20.228/1000.)/27.007197591"),
    "muon_low_pt": (lumi_muon_low_pt, "(4.987/1000.)/27.007197591"),
    "muon_high_pt": (lumi_muon_high_pt, "(20.517/1000.)/27.007197591"),
    "unprescaled": (lumi_full_2022, "(26671.7/1000.)/27.007197591"),
}

datasets["WJets"]["subsamples"] = {
    "ele_low_pt": (lumi_ele_low_pt, "(20.228/1000.)/27.007197591") ,
    "ele_high_pt": (lumi_ele_high_pt, "(20.228/1000.)/27.007197591"),
    "muon_low_pt": (lumi_muon_low_pt, "(4.987/1000.)/27.007197591"),
    "muon_high_pt": (lumi_muon_high_pt, "(20.517/1000.)/27.007197591"),
}

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

# qua metto i dati
DataRun = [
    ["E", "Run2022E-Prompt-v1"],
    ["F", "Run2022F-Prompt-v1"],
    ["G", "Run2022G-Prompt-v1"],
]

DataSets = ["Muon", "EGamma"]

DataTrig = {
    "Muon": "((ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt<=20, False) & (events.HLT.Mu8_TrkIsoVVL>0.5)) | (ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt>20, False) & (events.HLT.Mu17_TrkIsoVVL>0.5)))",
    "EGamma": "((events.HLT.Mu8_TrkIsoVVL<0.5) & (events.HLT.Mu17_TrkIsoVVL<0.5) & ((ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt<=25, False) & (events.HLT.Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30>0.5)) | (ak.fill_none(ak.pad_none(events.Lepton,1)[:,0].pt>25, False) & (events.HLT.Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30>0.5))))",
}

DataTrigUnprescaled = {
     "Muon" : "((events.HLT.IsoMu24 > 0.5) & (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 < 0.5))",
     "EGamma"  : "((events.HLT.IsoMu24 < 0.5) & (events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 < 0.5) & (events.HLT.Ele30_WPTight_Gsf > 0.5))",
}


samples_data = []
samples_data_unprescaled = []

for era, sd in DataRun:
    for pd in DataSets:
        tag = pd + "_" + sd

        # Dataset normale
        datasets[f"{pd}_{era}"] = {
            "files": tag,
            "trigger_sel": DataTrig[pd],
            "read_form": "data",
            "is_data": True,
            "era": f"2022EE_{era}",
        }
        samples_data.append(f"{pd}_{era}")

        # Dataset unprescaled
        datasets[f"{pd}_{era}_unprescaled"] = {
            "files": tag,
            "trigger_sel": DataTrigUnprescaled[pd],
            "read_form": "data",
            "is_data": True,
            "era": f"2022EE_{era}",
        }
        samples_data_unprescaled.append(f"{pd}_{era}_unprescaled")





samples = {}
colors = {}

samples["Data"] = {
    "samples": samples_data,
    "is_data": True,
}

samples["Data_unprescaled"] = {
    "samples": samples_data_unprescaled,
    "is_data": True,
}


#mc per ewk sub
#DY
samples["DY_ele_low_pt"] = {
    "samples": ["DY_ele_low_pt"],
    "is_signal": False,
}

samples["DY_ele_high_pt"] = {
    "samples": ["DY_ele_high_pt"],
    "is_signal": False,
}

samples["DY_muon_low_pt"] = {
    "samples": ["DY_muon_low_pt"],
    "is_signal": False,
}

samples["DY_muon_high_pt"] = {
    "samples": ["DY_muon_high_pt"],
    "is_signal": False,
}

# samples["DY_unprescaled"] = {
#     "samples": ["DY_unprescaled"],
#     "is_signal": False,
# }

#WJets
samples["WJets_ele_low_pt"] = {
    "samples": ["WJets_ele_low_pt"],
    "is_signal": False,
}

samples["WJets_ele_high_pt"] = {
    "samples": ["WJets_ele_high_pt"],
    "is_signal": False,
}

samples["WJets_muon_low_pt"] = {
    "samples": ["WJets_muon_low_pt"],
    "is_signal": False,
}

samples["WJets_muon_high_pt"] = {
    "samples": ["WJets_muon_high_pt"],
    "is_signal": False,
}



# # --- DY ---
colors["DY_ele_low_pt"] = '#336699'  # blu medio
colors["DY_ele_high_pt"] = '#99BBCC'  # blu chiaro
colors["DY_muon_low_pt"] = '#228855'  # verde profondo
colors["DY_muon_high_pt"] = '#88CCAA'  # verde chiaro
colors["DY_unprescaled"] = '#9999DD'  # viola blu chiaro

# --- WJets ---
colors["WJets_ele_low_pt"] = '#AA3377'  # rosa-viola intenso
colors["WJets_ele_high_pt"] = '#DDAADD'  # rosa-lilla
colors["WJets_muon_low_pt"] = '#EE7733'  # arancio acceso
colors["WJets_muon_high_pt"] = '#FFB377'  # arancio pastello

# regions
def _delta_phi(a, b):
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

def drlj(events, thr):
    lep_eta0 = ak.fill_none(ak.pad_none(events.Lepton.eta, 1)[:, 0], 0.0)
    lep_phi0 = ak.fill_none(ak.pad_none(events.Lepton.phi, 1)[:, 0], 0.0)

    j_pt  = events.jets.pt
    j_eta = events.jets.eta
    j_phi = events.jets.phi

    # candidati: pt > thr e |eta| <= 2.5
    cand = (j_pt > thr) & (abs(j_eta) <= 2.5)

    lep_eta0_b = ak.broadcast_arrays(lep_eta0, j_eta)[0]
    lep_phi0_b = ak.broadcast_arrays(lep_phi0, j_phi)[0]

    dphi = _delta_phi(lep_phi0_b, j_phi)
    deta = lep_eta0_b - j_eta
    dr   = np.sqrt(dphi * dphi + deta * deta)

    valid = cand & (dr > 1.0)

    dr_first = ak.firsts(dr[valid])
    dr_out   = ak.fill_none(dr_first, 0.0)

    has_lep0   = ak.num(events.Lepton) > 0
    has_anyjet = ak.num(j_pt) > 0
    dr_out = ak.where(has_lep0 & has_anyjet, dr_out, 0.0)

    return ak.to_numpy(dr_out, allow_missing=False)

def preselection_mask(events):
    # nLepton > 0
    has_lep = ak.num(events.Lepton) > 0

    # lepton[0] safe
    lep0_pt  = ak.fill_none(ak.pad_none(events.Lepton,   1)[:, 0].pt, 0.0)
    lep0_eta = ak.fill_none(ak.pad_none(events.Lepton,  1)[:, 0].eta, 0.0)
    lep0_id  = ak.fill_none(ak.pad_none(events.Lepton,1)[:, 0].pdgId, 0)

    is_ele = (abs(lep0_id) == 11) & (lep0_pt > 10.0) & (abs(lep0_eta) < 2.5) #le ho inverite
    is_mu  = (abs(lep0_id) == 13) & (lep0_pt > 13.0) & (abs(lep0_eta) < 2.4)
    lep_sel = is_ele | is_mu

    # MET < 20 
    met_ok = (events.PuppiMET.pt < 20)

    return (has_lep) & (lep_sel) & (met_ok)

def _lep_pair_pdg_prod(events):
    pdg0 = ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 0].pdgId, 0)
    pdg1 = ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 1].pdgId, 0)
    return pdg0 * pdg1

def zpeak_pr_common_mask(events):
    nlep = ak.num(events.Lepton)
    has2 = nlep > 1

    pt0 = ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 0].pt,-9999)
    pt1 = ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 1].pt,-9999)

    mll = events.mll 
    in_z = (mll > 76.0) & (mll < 106.0)

    return (has2) & (pt0 > 25.0) & (pt1 > 10.0) & (in_z)

regions = {}
#PROMPT RATE REGIONS
#LepWPCut1l e LepWPCut2l sono nel runner
regions["Zpeak_PR_loose_ele"] = {
    "func": lambda events: 
            (preselection_mask(events))
            &(zpeak_pr_common_mask(events))
            &(_lep_pair_pdg_prod(events) == -121),
     "mask": 0,
}

regions["Zpeak_PR_loose_mu"] = {
    "func": lambda events: 
            (preselection_mask(events))
            &(zpeak_pr_common_mask(events))
            &(_lep_pair_pdg_prod(events) == -169),
     "mask": 0,
}

regions["Zpeak_PR_tight_ele"] = {
    "func": lambda events: 
            (preselection_mask(events))
            &(zpeak_pr_common_mask(events))
            &(_lep_pair_pdg_prod(events) == -121),
     "mask": 0,
}

regions["Zpeak_PR_tight_mu"] = {
    "func": lambda events: 
            (preselection_mask(events))
            &(zpeak_pr_common_mask(events))
            &(_lep_pair_pdg_prod(events) == -169),
     "mask": 0,
}

#FAKE RATE REGIONS
jet_pt_thrs = [30, 35, 40]
for thr in jet_pt_thrs:
    regions[f"QCD_loose_jet_pt_{thr}_ele"] = {
        "func": lambda events: 
                (preselection_mask(events))
                &(ak.num(events.Lepton)==1)
                & (drlj(events, thr)>1)
                & (events.mtW1  < 20.0)
                & (abs(events.Lepton[:,0].pdgId) == 11),
        "mask": 0,
    }

    regions[f"QCD_loose_jet_pt_{thr}_mu"] = {
        "func": lambda events: 
                (preselection_mask(events))
                 &(ak.num(events.Lepton)==1)
                & (drlj(events, thr)>1)
                & (events.mtW1  < 20.0)
                & (abs(events.Lepton[:,0].pdgId) == 13),
        "mask": 0,
    }
    #LepWPCut1l è nel runner
    regions[f"QCD_tight_jet_pt_{thr}_ele"] = {
        "func": lambda events: 
                (preselection_mask(events))
                 &(ak.num(events.Lepton)==1)
                & (drlj(events, thr)>1)
                & (events.mtW1  < 20.0)
                & (abs(events.Lepton[:,0].pdgId) == 11),
        "mask": 0,
    }

    regions[f"QCD_tight_jet_pt_{thr}_mu"] = {
        "func": lambda events: 
                (preselection_mask(events))
                &(ak.num(events.Lepton)==1)
                & (drlj(events, thr)>1)
                & (events.mtW1  < 20.0)
                & (abs(events.Lepton[:,0].pdgId) == 13),
        "mask": 0,
    }



#variables
    
variables = {}

#for the selection
# for thr in jet_pt_thrs:
#     variables[f"drlj_{thr}"] = {
#         "func": (lambda events, _thr=thr: drlj(events, _thr)),
#         "axis": hist.axis.Regular(50, 0.0, 5.0, name=f"drlj_{thr}"),
#     }

# variables["nleptons"] = {
#     "func": lambda events: ak.num(events.Lepton),
#     "axis": hist.axis.Regular(5, 0, 3, name="nleptons"),
# }


#to plot
variables["ptl1"] = {
    "func": lambda events:ak.fill_none(ak.pad_none(events.Lepton, 1)[:, 0].pt, -9999),
    "axis": hist.axis.Regular(8, 10, 50, name="ptl1"),
 }

variables["ptl2"] = {
    "func": lambda events: ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 1].pt, -9999),
    "axis": hist.axis.Regular(8, 10, 50, name="ptl2"),
}

variables["etal1"] = {
    "func": lambda events: 
        ak.fill_none(abs(ak.pad_none(events.Lepton, 1)[:, 0].eta), -9999),
    "axis": hist.axis.Regular(5, 0, 2.5, name="etal1"),
}

variables["etal2"] = {
    "func": lambda events: 
        ak.fill_none(abs(ak.pad_none(events.Lepton, 2)[:, 1].eta), -9999),
    "axis": hist.axis.Regular(5, 0, 2.5, name="etal2"),
}

variables["ptl1_vs_etal1"] = {
    "axis": (
        hist.axis.Regular(8, 10, 50, name="ptl1"),
        hist.axis.Regular(5, 0, 2.5, name="etal1"),
    ),
}

variables["ptl2_vs_etal2"] = {
    "axis": (
        hist.axis.Regular(8, 10, 50, name="ptl2"),
        hist.axis.Regular(5, 0, 2.5, name="etal2"),
    ),
}

variables["mll"] = {
    "func": lambda events: ak.fill_none((ak.pad_none(events.Lepton, 2)[:, 0] + ak.pad_none(events.Lepton, 2)[:, 1]).mass,-9999),
    "axis": hist.axis.Regular(20, 60, 120, name="mll"),
}

variables["mtW1"] = {
    "func": lambda events: ak.fill_none(np.sqrt(
        2
        * ak.pad_none(events.Lepton, 1)[:, 0].pt
        * events.PuppiMET.pt
        * (1 - np.cos(ak.pad_none(events.Lepton, 1)[:, 0].phi - events.PuppiMET.phi))
    ),-9999),
    "axis": hist.axis.Regular(20, 0, 100, name="mtW1"),
}


nuisances = {}

mcs = [sample for sample in samples if not samples[sample].get("is_data", False)]

nuisances["lumi"] = {
    "name": "lumi",
    "type": "lnN",
    "samples": dict((skey, "1.014") for skey in mcs),
}





