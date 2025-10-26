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
plot_label = "VBS-SSWW"
year_label = "2022EE"
njobs = 1300

datasets = {}

# datasets["SSWW_EWK"] = {
#     "files": "SSWW_EWK",
#     "task_weight": 8,
# }

# datasets["SSWW_QCD"] = {
#     "files": "SSWW_QCD",
#     "task_weight": 8,
# }

datasets["WZ_EWK"] = {
    "files": "WZ_EWK",
    "do_theory_variations": True,
    "task_weight": 8,
}

datasets["WZ_QCD"] = {
    "files": "WZ_QCD",
    "do_theory_variations": True,
    "task_weight": 8,
}

# datasets["W_1JET"] = {
#     "files": "W_1JET",
#     "task_weight": 8,
# }

# datasets["W_2JET"] = {
#     "files": "W_2JET",
#     "task_weight": 8,
# }

# datasets["W_3JET"] = {
#     "files": "W_3JET",
#     "task_weight": 8,
# }

# datasets["W_4JET"] = {
#     "files": "W_4JET",
#     "task_weight": 8,
# }

# datasets["TTBAR"] = {
#     "files": "TTBAR",
#     "task_weight": 8,
# }

# datasets["SSWW_TT"] = {
#     "files": "SSWW_TT",
#     "task_weight": 8,
# }

# datasets["SSWW_TL"] = {
#     "files": "SSWW_TL",
#     "task_weight": 8,
# }

# datasets["SSWW_LL"] = {
#     "files": "SSWW_LL",
#     "task_weight": 8,
# }

datasets["TTWW"] = {
    "files": "TTWW",
    "task_weight": 8,
}

datasets["TTZZ"] = {
    "files": "TTZZ",
    "task_weight": 8,
}

datasets["TTLL_MLL-4to50"] = {
    "files": "TTLL_MLL-4to50",
    "task_weight": 8,
}

datasets["TTLL_MLL-50"] = {
    "files": "TTLL_MLL-50",
    "task_weight": 8,
}

datasets["TTLNu"] = {
    "files": "TTLNu",
    "task_weight": 8,
}

datasets["TZQB"] = {
    "files": "TZQB",
    "task_weight": 8,
}

# qua se devo aggiungere gli altri mc

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

# qua metto i dati
DataRun = [
    ["E", "Run2022E-Prompt-v1"],
    ["F", "Run2022F-Prompt-v1"],
    ["G", "Run2022G-Prompt-v1"],
]

DataSets = ["Muon", "EGamma", "MuonEG"]

DataTrig = {
    "Muon": "(events.DoubleMu | events.SingleMu) ",
    "EGamma": "(~(events.DoubleMu | events.SingleMu)) & (events.SingleEle | events.DoubleEle)",
    "MuonEG": "(~events.DoubleMu) & (~events.SingleMu) & (~events.SingleEle) & (~events.DoubleEle) & (events.EleMu)",
}


samples_data = []
samples_fake = []

for era, sd in DataRun:
    for pd in DataSets:
        tag = pd + "_" + sd

        # Dataset normale
        datasets[f"{pd}_{era}_prompt"] = {
            "files": tag,
            "trigger_sel": DataTrig[pd],
            "read_form": "data",
            "is_data": True,
            "era": f"2022EE_{era}",
        }
        samples_data.append(f"{pd}_{era}_prompt")

        # Dataset fake
        datasets[f"{pd}_{era}_fake"] = {
            "files": tag,
            "trigger_sel": DataTrig[pd],
            "read_form": "data",
            "is_data": True,
            "is_fake": True,
            "era": f"2022EE_{era}",
        }
        samples_fake.append(f"{pd}_{era}_fake")





samples = {}
colors = {}

samples["Data"] = {
    "samples": samples_data,
    "is_data": True,
}

samples["Fake"] = {
    "samples": samples_fake,
    "is_data": True,
    "is_fake": True,
}
colors["Fake"] = "#999999"


# samples["SSWW_EWK"] = {
#     "samples": ["SSWW_EWK"],
#     "is_signal": True,
# }

# samples["SSWW_QCD"] = {
#     "samples": ["SSWW_QCD"],
#     "is_signal": False,
# }
# colors["SSWW_QCD"] = '#DDAA33'

# samples["SSWW_QCD_pos"] = {
#     "samples": ["SSWW_QCD_pos"],
#     "is_signal": True,
# }
# colors["SSWW_QCD_pos"] = '#004488'

# samples["SSWW_QCD_neg"] = {
#     "samples": ["SSWW_QCD_neg"],
#     "is_signal": True,
# }
# colors["SSWW_QCD_neg"] = '#88CCEE'

# samples["SSWW_TT"] = {
#     "samples": ["SSWW_TT"],
#     "is_signal": True,
# }
# colors["SSWW_TT"] = '#BB5566'

# samples["SSWW_TL"] = {
#     "samples": ["SSWW_TL"],
#     "is_signal": True,
# }
# colors["SSWW_TL"] = '#117733'

# samples["SSWW_LL"] = {
#     "samples": ["SSWW_LL"],
#     "is_signal": True,
# }
# colors["SSWW_LL"] = '#332288'

# samples["WZ_EWK"] = {
#     "samples": ["WZ_EWK"],
#     "is_signal": False,
# }
# colors["WZ_EWK"] = '#88CCEE'

# samples["WZ_QCD"] = {
#     "samples": ["WZ_QCD"],
#     "is_signal": False,
# }
# colors["WZ_QCD"] = '#44AA99'

datasets["WZ_EWK"]["subsamples"] = {
    "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
    "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
}

datasets["WZ_QCD"]["subsamples"] = {
    "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
    "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
}

samples["WZ_EWK_pos"] = {
    "samples": ["WZ_EWK_pos"],
    "is_signal": False,
}
#colors["WZ_EWK_pos"] = '#004488'  # blu scuro

samples["WZ_EWK_neg"] = {
    "samples": ["WZ_EWK_neg"],
    "is_signal": False,
}
#colors["WZ_EWK_neg"] = '#88CCEE'  # blu chiaro

samples["WZ_QCD_pos"] = {
    "samples": ["WZ_QCD_pos"],
    "is_signal": False,
}
#colors["WZ_QCD_pos"] = '#117733'  # verde scuro

samples["WZ_QCD_neg"] = {
    "samples": ["WZ_QCD_neg"],
    "is_signal": False,
}
#colors["WZ_QCD_neg"] = '#44AA99'  # verde chiaro


# samples["TTBAR"] = {
#     "samples": ["TTBAR"],
#     "is_signal": False,
# }
# colors["TTBAR"] = '#556270'

# samples["W_JETS"] = {
#     "samples": [f"W_{j}JET" for j in range(1, 5)],
# }
# colors["W_JETS"] = '#CC99CC'

# datasets["SSWW_QCD"]["subsamples"] = {
#     "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
#     "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
# }

# datasets["SSWW_TT"]["subsamples"] = {
#     "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
#     "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
# }

# datasets["SSWW_TL"]["subsamples"] = {
#     "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
#     "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
# }

# datasets["SSWW_LL"]["subsamples"] = {
#     "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
#     "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
# }

# samples["SSWW_QCD_pos"] = {
#     "samples": ["SSWW_QCD_pos"],
#     "is_signal": False,
# }

# samples["SSWW_QCD_neg"] = {
#     "samples": ["SSWW_QCD_neg"],
#     "is_signal": False,
# }

# samples["SSWW_TT_pos"] = {
#     "samples": ["SSWW_TT_pos"],
#     "is_signal": True,
# }


# samples["SSWW_TT_neg"] = {
#     "samples": ["SSWW_TT_neg"],
#     "is_signal": True,
# }


# samples["SSWW_TL_pos"] = {
#     "samples": ["SSWW_TL_pos"],
#     "is_signal": True,
# }


# samples["SSWW_TL_neg"] = {
#     "samples": ["SSWW_TL_neg"],
#     "is_signal": True,
# }


# samples["SSWW_LL_pos"] = {
#     "samples": ["SSWW_LL_pos"],
#     "is_signal": True,
# }


# samples["SSWW_LL_neg"] = {
#     "samples": ["SSWW_LL_neg"],
#     "is_signal": True,
# }

# --- SSWW QCD ---
# colors["SSWW_QCD_pos"] = '#994C00'  # marrone bruciato
# colors["SSWW_QCD_neg"] = '#CC9966'  # beige caldo

# --- WZ EWK ---
colors["WZ_EWK_pos"] = '#336699'  # blu medio
colors["WZ_EWK_neg"] = '#99BBCC'  # blu chiaro

# --- WZ QCD ---
colors["WZ_QCD_pos"] = '#228855'  # verde profondo
colors["WZ_QCD_neg"] = '#88CCAA'  # verde chiaro

# --- SSWW TT ---
# colors["SSWW_TT_pos"] = '#AA3377'  # rosa-viola intenso
# colors["SSWW_TT_neg"] = '#DDAADD'  # rosa-lilla

#--- SSWW TL ---
# colors["SSWW_TL_pos"] = '#EE7733'  # arancio acceso
# colors["SSWW_TL_neg"] = '#FFB377'  # arancio pastello

# --- SSWW LL ---
# colors["SSWW_LL_pos"] = '#5555AA'  # viola blu intenso
# colors["SSWW_LL_neg"] = '#9999DD'  # viola blu chiaro


# samples["LL+TT+TL"] = {
#      "samples": ["SSWW_LL", "SSWW_TT", "SSWW_TL" ],
#  }
# colors["LL+TT+TL"] = '#556270'

samples["tvX"] = {
      "samples": ["TTWW", "TTZZ", "TTLL_MLL-4to50", "TTLL_MLL-50", "TTLNu", "TZQB"],
  }
colors["tvX"] = '#660066'



# regions

regions = {}
def tthmva(events):
    L = ak.pad_none(events.Lepton, 3)  

    l0 = L[:, 0]
    l1 = L[:, 1]
    l2 = L[:, 2]

    m0 = ak.where(ak.is_none(l0), False,
              ak.where(abs(l0.pdgId) == 11, l0.mvaTTH > 0.4,
              ak.where(abs(l0.pdgId) == 13, l0.mvaTTH > 0.5, False)))

    m1 = ak.where(ak.is_none(l1), False,
              ak.where(abs(l1.pdgId) == 11, l1.mvaTTH > 0.4,
              ak.where(abs(l1.pdgId) == 13, l1.mvaTTH > 0.5, False)))

    m2 = ak.where(ak.is_none(l2), False,
              ak.where(abs(l2.pdgId) == 11, l2.mvaTTH > 0.4,
              ak.where(abs(l2.pdgId) == 13, l2.mvaTTH > 0.5, False)))

    tthmva_mask_3l = m0 & m1 & m2
    return tthmva_mask_3l




def from_z_mass(lep1, lep2):
    return (lep1.pdgId + lep2.pdgId == 0) & (abs(lep1.pdgId) == abs(lep2.pdgId))


def select_wz_region(events):
    lep = ak.pad_none(events.Lepton, 3)

    # Tutte le 3 possibili combinazioni
    pairs = [
        (lep[:, 0], lep[:, 1]),
        (lep[:, 0], lep[:, 2]),
        (lep[:, 1], lep[:, 2]),
    ]
    third_lepton = [lep[:, 2], lep[:, 1], lep[:, 0]]

    # Calcola tutte le masse
    masses = [ak.fill_none((l1 + l2).mass, 9999) for l1, l2 in pairs]
    mass_diffs = [abs(m - 91.2) for m in masses]

    # Trova indice della combinazione più compatibile con Z
    best_idx = ak.argmin(mass_diffs, axis=0)

    lep_z1 = ak.where(best_idx == 0, pairs[0][0],
              ak.where(best_idx == 1, pairs[1][0], pairs[2][0]))
    lep_z2 = ak.where(best_idx == 0, pairs[0][1],
              ak.where(best_idx == 1, pairs[1][1], pairs[2][1]))
    lep_w  = ak.where(best_idx == 0, third_lepton[0],
              ak.where(best_idx == 1, third_lepton[1], third_lepton[2]))
    
    lep_z1_pt = lep_z1.pt
    lep_z2_pt = lep_z2.pt
    lep_z_ptll = (lep_z1+lep_z2).pt
    lep_w_pt = lep_w.pt
    lep_z1_eta = lep_z1.eta
    lep_z2_eta = lep_z2.eta
    lep_w_eta = lep_w.eta
    lep_w_phi = lep_w.phi
    lep_w_pdg = lep_w.pdgId



    mll = (lep_z1 + lep_z2).mass
    z_mask = from_z_mass(lep_z1, lep_z2) & (abs(mll - 91.2) < 15)

    mlll = (lep[:, 0] + lep[:, 1] + lep[:, 2]).mass
    mlll_mask = ak.fill_none(mlll > 100, False)

    final_mask = ak.fill_none(z_mask & mlll_mask, False)

    return final_mask, z_mask, mlll, mll, lep_z1_pt, lep_z2_pt, lep_w_pt, lep_z1_eta, lep_z2_eta, lep_w_eta, lep_w_phi, lep_z_ptll, lep_w_pdg 

regions["WZ"] = {
    "func": lambda events: 
            select_wz_region(events)[0]
            & (~ak.any(ak.pad_none(events.Lepton, 4)[:, 3:].pt > 10, axis=1))
            # & (mlll >100)
            # & (best_z_mass >= 91.2 - 15)
            # & (best_z_mass <= 91.2 + 15)
            &(events.mjj > 500)
            &(events.PuppiMET.pt > 30)
            & (abs(events.detajj) > 2.5)
            & (events.Zeppenfeld_Z <= 1.0)
            & (ak.num(events.Lepton) >= 3)
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False))
            & (events.bVeto)
            & (tthmva(events)),
     "mask": 0,
}

regions["WZ-(l3=e)"] = {
    "func": lambda events: 
            select_wz_region(events)[0]
             & (abs(events.Lepton[:, 2].pdgId) == 11)
            & (~ak.any(ak.pad_none(events.Lepton, 4)[:, 3:].pt > 10, axis=1))
            # & (mlll >100)
            # & (best_z_mass >= 91.2 - 15)
            # & (best_z_mass <= 91.2 + 15)
            &(events.mjj > 500)
            &(events.PuppiMET.pt > 30)
            & (abs(events.detajj) > 2.5)
            & (events.Zeppenfeld_Z <= 1.0)
            & (ak.num(events.Lepton) >= 3)
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False))
            & (events.bVeto)
            & (tthmva(events)),
     "mask": 0,
}

regions["WZ-(l3=mu)"] = {
    "func": lambda events: 
            select_wz_region(events)[0]
            & (abs(events.Lepton[:, 2].pdgId) == 13)
            & (~ak.any(ak.pad_none(events.Lepton, 4)[:, 3:].pt > 10, axis=1))
            # &  valid_z
            # & (mlll >100)
            # & (best_z_mass >= 91.2 - 15)
            # & (best_z_mass <= 91.2 + 15)
            &(events.mjj > 500)
            &(events.PuppiMET.pt > 30)
            & (abs(events.detajj) > 2.5)
            & (events.Zeppenfeld_Z <= 1.0)
            & (ak.num(events.Lepton) >= 3)
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False))
            # & (ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False)
            # & (ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False)
            # & (ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False)
            & (events.bVeto)
            & (tthmva(events)),
            #& ak.num(events.Lepton) >= 3,
     "mask": 0,
}

regions["WZb"] = {
    "func": lambda events: 
            select_wz_region(events)[0]
            & (~ak.any(ak.pad_none(events.Lepton, 4)[:, 3:].pt > 10, axis=1))
            # &  valid_z
            # & (mlll >100)
            # & (best_z_mass >= 91.2 - 15)
            # & (best_z_mass <= 91.2 + 15)
            &(events.mjj > 500)
            &(events.PuppiMET.pt > 30)
            & (abs(events.detajj) > 2.5)
            & (events.Zeppenfeld_Z <= 1.0)
            & (ak.num(events.Lepton) >= 3)
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False))
            # & (ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False)
            # & (ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False)
            # & (ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False)
            & ~(events.bVeto)
            & (tthmva(events)),
            #& ak.num(events.Lepton) >= 3,
     "mask": 0,
}

regions["WZ-loose"] = {
    "func": lambda events: 
            select_wz_region(events)[0]
            & (~ak.any(ak.pad_none(events.Lepton, 4)[:, 3:].pt > 10, axis=1))
            # & (mlll >100)
            # & (best_z_mass >= 91.2 - 15)
            # & (best_z_mass <= 91.2 + 15)
            &(events.mjj > 300)
            &(events.PuppiMET.pt > 30)
            & (abs(events.detajj) > 2.0)
            & (events.Zeppenfeld_Z <= 1.0)
            & (ak.num(events.Lepton) >= 3)
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False))
            & (ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False))
            # & (ak.pad_none(events.Lepton, 3)[:, 0].pt > 25, False)
            # & (ak.pad_none(events.Lepton, 3)[:, 1].pt > 20, False)
            # & (ak.pad_none(events.Lepton, 3)[:, 2].pt > 10, False)
            & (events.bVeto)
            & (tthmva(events)),
            #& ak.num(events.Lepton) >= 3,
     "mask": 0,
}



#variables
    
variables = {}

# variables["btagPNetB1"] = {
#     "func": lambda events: events.jets[:, 0].btagPNetB , 
#     "axis": hist.axis.Regular(6, 0, 0.1, name="btagPNetB1"),
# }

# variables["btagPNetB2"] = {
#     "func": lambda events: events.jets[:, 0].btagPNetB , 
#     "axis": hist.axis.Regular(6, 0, 0.1, name="btagPNetB2"),
# }

variables["tthMVA_l1"] = {
    "func": lambda events: events.Lepton[:, 0].mvaTTH , 
    "axis": hist.axis.Regular(4, 0.4, 1, name="tthMVA_l1"),
}

variables["tthMVA_l2"] = {
    "func": lambda events: events.Lepton[:, 1].mvaTTH , 
    "axis": hist.axis.Regular(4, 0.4, 1, name="tthMVA_l2"),
}

variables["tthMVA_l3"] = {
    "func": lambda events: events.Lepton[:, 2].mvaTTH , 
    "axis": hist.axis.Regular(4, 0.4, 1, name="tthMVA_l3"),
}

variables["tthMVA_l1_bin"] = {
    "func": lambda events: events.Lepton[:, 0].mvaTTH , 
    "axis": hist.axis.Regular(3, 0.4, 1, name="tthMVA_l1_bin"),
}

variables["tthMVA_l2_bin"] = {
    "func": lambda events: events.Lepton[:, 1].mvaTTH , 
    "axis": hist.axis.Regular(3, 0.4, 1, name="tthMVA_l2_bin"),
}

variables["tthMVA_l3_bin"] = {
    "func": lambda events: events.Lepton[:, 2].mvaTTH , 
    "axis": hist.axis.Regular(3, 0.4, 1, name="tthMVA_l3_bin"),
}

variables["pv"] = {
    "func": lambda events: events.PV.npvsGood ,
    "axis": hist.axis.Regular(65, 0, 65, name="pv"),
}

# variables["njet"] = {
#     "func": lambda events: ak.num(events.jets),
#     "axis": hist.axis.Regular(6, 2, 10, name="njet"),
# }

# Dijet
variables["mjj"] = {
    "func": lambda events: ak.fill_none(
        (events.jets[:, 0] + events.jets[:, 1]).mass, -9999
    ),
    "axis": hist.axis.Variable([500, 700, 1000, 1500, 2000], name="mjj"),
}

variables["detajj"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaeta(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(4, 2.5, 8, name="detajj"),
}

variables["dphijj"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaphi(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(4, 0, np.pi, name="dphijj"),
}

# Single jet
variables["ptj1"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 0].pt, -9999),
    "axis": hist.axis.Regular(4, 50, 500, name="ptj1"),
}
variables["ptj2"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 1].pt, -9999),
    "axis": hist.axis.Regular(4, 30, 250, name="ptj2"),
}


# Dilepton

 #per WZ
variables["mll"] = {
    "func": lambda events: select_wz_region(events)[3],
    "axis": hist.axis.Regular(4, 75, 110, name="mll"),
}

# variables["mlll"] = {
#     "func": lambda events: (ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).mass,
#     "axis": hist.axis.Regular(10, 100, 500, name="mlll"),
# }

#Single lepton
variables["ptl1"] = {
    "func": lambda events: events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(4, 25, 300, name="ptl1"),
 }

variables["ptl2"] = {
    "func": lambda events: events.Lepton[:, 1].pt,
    "axis": hist.axis.Regular(4, 20, 180, name="ptl2"),
}

variables["ptl3"] = {
    "func": lambda events: ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt, -9999),
    "axis": hist.axis.Regular(4, 10, 100, name="ptl3"),
}

# variables["ptl3zommed"] = {
#     "func": lambda events: ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt, -9999),
#     "axis": hist.axis.Variable([10, 20, 30, 40, 50, 70, 90, 125], name="ptl3zommed"),
# }


variables["ptl1Z"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[4],-9999),
    "axis": hist.axis.Regular(4, 10, 300, name="ptl1Z"),
}
variables["ptl2Z"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[5],-9999),
    "axis": hist.axis.Regular(4, 10, 140, name="ptl2Z"),
}
variables["ptl3W"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[6],-9999),
    "axis": hist.axis.Regular(4, 10, 230, name="ptl3W"),
}

variables["ptll"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[11],-9999),
     "axis": hist.axis.Regular(4, 25, 400, name="ptll"),
}


variables["Zeppenfeld_Z"] = {
    "func": lambda events: ak.max(abs(events.Lepton.eta - 0.5 * (events.jets[:, 0].eta + events.jets[:, 1].eta))
        / events.detajj, axis=1),
    "axis": hist.axis.Regular(4, 0, 1, name="Zeppenfeld_Z"),
}

variables["MET"] = {
    "func": lambda events: events.PuppiMET.pt,
    "axis": hist.axis.Regular(4, 30, 300, name="MET"),
}

variables["mtW"] = {
    "func": lambda events: ak.fill_none(np.sqrt(
        2
        * select_wz_region(events)[6]
        * events.PuppiMET.pt
        * (1 - np.cos(select_wz_region(events)[10] - events.PuppiMET.phi))
    ),-9999),
    "axis": hist.axis.Regular(4, 0, 200, name="mtW"),
}

variables["mtWZ"] = {
    "func": lambda events: ak.fill_none(np.sqrt(
        ((ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).pt
        + events.PuppiMET.pt)**2
        - (
            ak.zip(
                {
                    "pt": (ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).pt,
                    "phi": (ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).phi,
                },
                with_name="Momentum2D",
            )
            + ak.zip(
                {
                    "pt": events.PuppiMET.pt,
                    "phi": events.PuppiMET.phi,
                },
                with_name="Momentum2D",
            )
        ).pt**2
    ),-9999),
    "axis": hist.axis.Regular(4, 0, 400, name="mtWZ"),
 }

#Rpt, Hrt and thetaCS
# variables["Rpt"] = {
#     "func": lambda events: (events.Lepton[:, 0].pt*events.Lepton[:, 1].pt)/(events.jets[:, 0].pt*events.jets[:, 1].pt),
#     "axis": hist.axis.Regular(10, 0, 1.5, name="Rpt"),
# }

variables["Ht"] = {
    "func": lambda events: ak.sum(events.jets.pt, axis=1)/events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(4, 0, 10, name="Ht"),
}

# variables["cos_theta_CS"] = {
#     "func": lambda events: cos_theta_collins_soper(build_4vector(events.Lepton[:, 0]),build_4vector(events.Lepton[:, 1])),
#     "axis": hist.axis.Regular(10, -1, 1, name="cos_theta_CS"),
# }

#single eta
variables["etaj1"] = {
    "func": lambda events: 
        (ak.fill_none(abs(events.jets[:, 0].eta), -9999)),
    "axis": hist.axis.Regular(4, 0, 4.7, name="etaj1"),
}

variables["etaj2"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.jets[:, 1].eta), -9999),
    "axis": hist.axis.Regular(4, 0, 4.7, name="etaj2"),
}

variables["etal1"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 0].eta), -9999),
    "axis": hist.axis.Regular(4, 0, 2.5, name="etal1"),
}

variables["etal2"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 1].eta), -9999),
    "axis": hist.axis.Regular(4, 0, 2.5, name="etal2"),
}

variables["etal3"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 2].eta), -9999),
    "axis": hist.axis.Regular(4, 0, 2.5, name="etal3"),
}

variables["etal1Z"] = {
    "func": lambda events: 
        ak.fill_none(abs(select_wz_region(events)[7]), -9999),
    "axis": hist.axis.Regular(4, 0, 2.5, name="etal1Z"),
}

variables["etal2Z"] = {
    "func": lambda events: 
        ak.fill_none(abs(select_wz_region(events)[8]), -9999),
    "axis": hist.axis.Regular(4, 0, 2.5, name="etal2Z"),
}

variables["etal3W"] = {
    "func": lambda events: 
        ak.fill_none(abs(select_wz_region(events)[9]), -9999),
    "axis": hist.axis.Regular(4, 0, 2.5, name="etal3W"),
}



# variables["run_period"] = {
#     "func": lambda events: events.run_period,
#     "axis": hist.axis.Regular(30, -1, 10, name="run_period"),
# }

#same but with 3 bin

# Dijet
variables["mjj_bin"] = {
    "func": lambda events: ak.fill_none(
        (events.jets[:, 0] + events.jets[:, 1]).mass, -9999
    ),
    "axis": hist.axis.Variable([500, 800, 1200, 2000], name="mjj_bin"),
}




variables["detajj_bin"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaeta(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(3, 2.5, 8, name="detajj_bin"),
}

variables["dphijj_bin"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaphi(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(3, 0, np.pi, name="dphijj_bin"),
}

# Single jet
variables["ptj1_bin"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 0].pt, -9999),
    "axis": hist.axis.Regular(3, 50, 500, name="ptj1_bin"),
}
variables["ptj2_bin"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 1].pt, -9999),
    "axis": hist.axis.Regular(3, 30, 250, name="ptj2_bin"),
}


# Dilepton

 #per WZ
variables["mll_bin"] = {
    "func": lambda events: select_wz_region(events)[3],
    "axis": hist.axis.Regular(3, 75, 110, name="mll_bin"),
}

# variables["mlll"] = {
#     "func": lambda events: (ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).mass,
#     "axis": hist.axis.Regular(10, 100, 500, name="mlll"),
# }

#Single lepton
variables["ptl1_bin"] = {
    "func": lambda events: events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(3, 25, 300, name="ptl1_bin"),
 }

variables["ptl2_bin"] = {
    "func": lambda events: events.Lepton[:, 1].pt,
    "axis": hist.axis.Regular(3, 20, 180, name="ptl2_bin"),
}

variables["ptl3_bin"] = {
    "func": lambda events: ak.fill_none(ak.pad_none(events.Lepton, 3)[:, 2].pt, -9999),
    "axis": hist.axis.Regular(3, 10, 100, name="ptl3_bin"),
}

variables["ptl1Z_bin"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[4],-9999),
    "axis": hist.axis.Regular(3, 10, 300, name="ptl1Z_bin"),
}
variables["ptl2Z_bin"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[5],-9999),
    "axis": hist.axis.Regular(3, 10, 140, name="ptl2Z_bin"),
}
variables["ptl3W_bin"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[6],-9999),
    "axis": hist.axis.Regular(3, 10, 230, name="ptl3W_bin"),
}

variables["ptll_bin"] = {
    "func": lambda events: ak.fill_none(select_wz_region(events)[11],-9999),
     "axis": hist.axis.Regular(3, 25, 400, name="ptll_bin"),
}


variables["Zeppenfeld_Z_bin"] = {
    "func": lambda events: ak.max(abs(events.Lepton.eta - 0.5 * (events.jets[:, 0].eta + events.jets[:, 1].eta))
        / events.detajj, axis=1),
    "axis": hist.axis.Regular(3, 0, 1, name="Zeppenfeld_Z_bin"),
}

variables["MET_bin"] = {
    "func": lambda events: events.PuppiMET.pt,
    "axis": hist.axis.Regular(3, 30, 300, name="MET_bin"),
}

variables["mtW_bin"] = {
    "func": lambda events: ak.fill_none(np.sqrt(
        2
        * select_wz_region(events)[6]
        * events.PuppiMET.pt
        * (1 - np.cos(select_wz_region(events)[10] - events.PuppiMET.phi))
    ),-9999),
    "axis": hist.axis.Regular(3, 0, 200, name="mtW_bin"),
}

variables["mtWZ_bin"] = {
    "func": lambda events: ak.fill_none(np.sqrt(
        ((ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).pt
        + events.PuppiMET.pt)**2
        - (
            ak.zip(
                {
                    "pt": (ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).pt,
                    "phi": (ak.pad_none(events.Lepton, 3)[:, 0] + ak.pad_none(events.Lepton, 3)[:, 1] + ak.pad_none(events.Lepton, 3)[:, 2]).phi,
                },
                with_name="Momentum2D",
            )
            + ak.zip(
                {
                    "pt": events.PuppiMET.pt,
                    "phi": events.PuppiMET.phi,
                },
                with_name="Momentum2D",
            )
        ).pt**2
    ),-9999),
    "axis": hist.axis.Regular(3, 0, 400, name="mtWZ_bin"),
 }

#Rpt, Hrt and thetaCS
# variables["Rpt"] = {
#     "func": lambda events: (events.Lepton[:, 0].pt*events.Lepton[:, 1].pt)/(events.jets[:, 0].pt*events.jets[:, 1].pt),
#     "axis": hist.axis.Regular(10, 0, 1.5, name="Rpt"),
# }

variables["Ht_bin"] = {
    "func": lambda events: ak.sum(events.jets.pt, axis=1)/events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(3, 0, 10, name="Ht_bin"),
}

# variables["cos_theta_CS"] = {
#     "func": lambda events: cos_theta_collins_soper(build_4vector(events.Lepton[:, 0]),build_4vector(events.Lepton[:, 1])),
#     "axis": hist.axis.Regular(10, -1, 1, name="cos_theta_CS"),
# }

#single eta
variables["etaj1_bin"] = {
    "func": lambda events: 
        (ak.fill_none(abs(events.jets[:, 0].eta), -9999)),
    "axis": hist.axis.Regular(3, 0, 4.7, name="etaj1_bin"),
}

variables["etaj2_bin"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.jets[:, 1].eta), -9999),
    "axis": hist.axis.Regular(3, 0, 4.7, name="etaj2_bin"),
}

variables["etal1_bin"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 0].eta), -9999),
    "axis": hist.axis.Regular(3, 0, 2.5, name="etal1_bin"),
}

variables["etal2_bin"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 1].eta), -9999),
    "axis": hist.axis.Regular(3, 0, 2.5, name="etal2_bin"),
}

variables["etal3_bin"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 2].eta), -9999),
    "axis": hist.axis.Regular(3, 0, 2.5, name="etal3_bin"),
}

variables["etal1Z_bin"] = {
    "func": lambda events: 
        ak.fill_none(abs(select_wz_region(events)[7]), -9999),
    "axis": hist.axis.Regular(3, 0, 2.5, name="etal1Z_bin"),
}

variables["etal2Z_bin"] = {
    "func": lambda events: 
        ak.fill_none(abs(select_wz_region(events)[8]), -9999),
    "axis": hist.axis.Regular(3, 0, 2.5, name="etal2Z_bin"),
}

variables["etal3W_bin"] = {
    "func": lambda events: 
        ak.fill_none(abs(select_wz_region(events)[9]), -9999),
    "axis": hist.axis.Regular(3, 0, 2.5, name="etal3W_bin"),
}
nuisances = {}

mcs = [sample for sample in samples if not samples[sample].get("is_data", False)]
fake = [sample for sample in samples if samples[sample].get("is_fake", False)]

nuisances["lumi"] = {
    "name": "lumi",
    "type": "lnN",
    "samples": dict((skey, "1.014") for skey in mcs),
}


for shift in [
    "lf",
    "hf",
    "hfstats1",
    "hfstats2",
    "lfstats1",
    "lfstats2",
    "cferr1",
    "cferr2",
    "jes",
]:
    nuisances[f"btag_{shift}"] = {
        "name": "CMS_btag_SF_" + shift,
        "kind": "suffix",
        "type": "shape",
        "samples": dict((skey, ["1", "1"]) for skey in mcs),
    }


for js in cfg["jme"]["jes"]:
    nuisances[f"JES_{js}"] = {
        "name": "CMS_jet_scale_" + js,
        "kind": "suffix",
        "type": "shape",
        "samples": dict((skey, ["1", "1"]) for skey in mcs),
    }

nuisances["JER"] = {
    "name": "CMS_jet_res",
    "kind": "suffix",
    "type": "shape",
    "samples": dict((skey, ["1", "1"]) for skey in mcs),
}


nuisances["PU"] = {
    "name": "CMS_PU",
    "kind": "suffix",
    "type": "shape",
    "samples": dict((skey, ["1", "1"]) for skey in mcs),
}

nuisances["ele_reco"] = {
    "name": "CMS_ele_reco",
    "kind": "suffix",
    "type": "shape",
    "samples": dict((skey, ["1", "1"]) for skey in mcs),
}

nuisances["ele_idiso"] = {
    "name": "CMS_ele_idiso",
    "kind": "suffix",
    "type": "shape",
    "samples": dict((skey, ["1", "1"]) for skey in mcs),
}

nuisances["mu_idiso"] = {
    "name": "CMS_mu_idiso",
    "kind": "suffix",
    "type": "shape",
    "samples": dict((skey, ["1", "1"]) for skey in mcs),
}

#30%
nuisances["fake_syst_30"] = {
    "name": "CMS_fake_syst",
    "type": "lnN",
    "samples": {
        "Fake": "1.3"
    },
    "exclude_cuts": ["WZ"],
}

#quella calcolata in in evaluate
nuisances["FW"] = {
        "name": "CMS_fake_stat",
        "kind": "suffix",
        "type": "shape",
        "samples": dict((skey, ["1", "1"]) for skey in fake),
        "exclude_cuts": ["WZ"],
}

for sname in ["WZ_EWK", "WZ_QCD"]:
    mcs_theory_unc = [
        sample
        for sample in samples
        if not samples[sample].get("is_data", False) and (sname in sample)
    ]

    nuisances[f"PDFWeight_{sname}"] = {
        "name": f"PDF_{sname}",
        "kind": "weight_square",
        "type": "shape",
        "samples": {
            skey: [f"PDFWeight_{i}" for i in range(100)] for skey in mcs_theory_unc
        },
    }

    nuisances[f"QCDScale_{sname}"] = {
        "name": f"QCDScale_{sname}",
        "kind": "weight_envelope",
        "type": "shape",
        "samples": {
            skey: [f"QCDScale_{i}" for i in range(6)] for skey in mcs_theory_unc
        },
    }

## Use the following if you want to apply the automatic combine MC stat nuisances.
nuisances["stat"] = {
    "type": "auto",
    "maxPoiss": "10",
    "includeSignal": "0",
    #  nuisance ['maxPoiss'] =  Number of threshold events for Poisson modelling
    #  nuisance ['includeSignal'] =  Include MC stat nuisances on signal processes (1=True, 0=False)
    "samples": {},
}


