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
njobs = 1500

datasets = {}

# datasets["SSWW_EWK"] = {
#     "files": "SSWW_EWK",
#     "task_weight": 8,
# }

datasets["SSWW_QCD"] = {
    "files": "SSWW_QCD",
    "do_theory_variations": True,
    "task_weight": 8,
}

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

datasets["SSWW_TT"] = {
    "files": "SSWW_TT",
    "do_theory_variations": True,
    "task_weight": 8,
}

datasets["SSWW_TL"] = {
    "files": "SSWW_TL",
    "do_theory_variations": True,
    "task_weight": 8,
}

datasets["SSWW_LL"] = {
    "files": "SSWW_LL",
    "do_theory_variations": True,
    "task_weight": 8,
}

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
colors["WZ_EWK_pos"] = '#004488'  # blu scuro

samples["WZ_EWK_neg"] = {
    "samples": ["WZ_EWK_neg"],
    "is_signal": False,
}
colors["WZ_EWK_neg"] = '#88CCEE'  # blu chiaro

samples["WZ_QCD_pos"] = {
    "samples": ["WZ_QCD_pos"],
    "is_signal": False,
}
colors["WZ_QCD_pos"] = '#117733'  # verde scuro

samples["WZ_QCD_neg"] = {
    "samples": ["WZ_QCD_neg"],
    "is_signal": False,
}
colors["WZ_QCD_neg"] = '#44AA99'  # verde chiaro


# samples["TTBAR"] = {
#     "samples": ["TTBAR"],
#     "is_signal": False,
# }
# colors["TTBAR"] = '#556270'

# samples["W_JETS"] = {
#     "samples": [f"W_{j}JET" for j in range(1, 5)],
# }
# colors["W_JETS"] = '#CC99CC'

datasets["SSWW_QCD"]["subsamples"] = {
    "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
    "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
}

datasets["SSWW_TT"]["subsamples"] = {
    "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
    "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
}

datasets["SSWW_TL"]["subsamples"] = {
    "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
    "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
}

datasets["SSWW_LL"]["subsamples"] = {
    "neg": "ak.sum(events.Lepton.pdgId > 0, axis=1) == 2",
    "pos": "ak.sum(events.Lepton.pdgId < 0, axis=1) == 2",
}

samples["SSWW_QCD_pos"] = {
    "samples": ["SSWW_QCD_pos"],
    "is_signal": False,
}

samples["SSWW_QCD_neg"] = {
    "samples": ["SSWW_QCD_neg"],
    "is_signal": False,
}

samples["SSWW_TT_pos"] = {
    "samples": ["SSWW_TT_pos"],
    "is_signal": False,
}


samples["SSWW_TT_neg"] = {
    "samples": ["SSWW_TT_neg"],
    "is_signal": False,
}


samples["SSWW_TL_pos"] = {
    "samples": ["SSWW_TL_pos"],
    "is_signal": False,
}


samples["SSWW_TL_neg"] = {
    "samples": ["SSWW_TL_neg"],
    "is_signal": False,
}


samples["SSWW_LL_pos"] = {
    "samples": ["SSWW_LL_pos"],
    "is_signal": True,
}


samples["SSWW_LL_neg"] = {
    "samples": ["SSWW_LL_neg"],
    "is_signal": True,
}

# --- SSWW QCD ---
colors["SSWW_QCD_pos"] = '#994C00'  # marrone bruciato
colors["SSWW_QCD_neg"] = '#CC9966'  # beige caldo

# --- WZ EWK ---
colors["WZ_EWK_pos"] = '#336699'  # blu medio
colors["WZ_EWK_neg"] = '#99BBCC'  # blu chiaro

# --- WZ QCD ---
colors["WZ_QCD_pos"] = '#228855'  # verde profondo
colors["WZ_QCD_neg"] = '#88CCAA'  # verde chiaro

#--- SSWW TT ---
colors["SSWW_TT_pos"] = '#AA3377'  # rosa-viola intenso
colors["SSWW_TT_neg"] = '#DDAADD'  # rosa-lilla

#--- SSWW TL ---
colors["SSWW_TL_pos"] = '#EE7733'  # arancio acceso
colors["SSWW_TL_neg"] = '#FFB377'  # arancio pastello

# --- SSWW LL ---
colors["SSWW_LL_pos"] = '#5555AA'  # viola blu intenso
colors["SSWW_LL_neg"] = '#9999DD'  # viola blu chiaro


# samples["LL+TT+TL"] = {
#      "samples": ["SSWW_LL", "SSWW_TT", "SSWW_TL" ],
#  }
# colors["LL+TT+TL"] = '#556270'

samples["tvX"] = {
      "samples": ["TTWW", "TTZZ", "TTLL_MLL-4to50", "TTLL_MLL-50", "TTLNu", "TZQB"],
  }
colors["tvX"] = '#660066'



# regions

def tthmva(events):
    L = ak.pad_none(events.Lepton, 2) 

    l0 = L[:, 0]
    l1 = L[:, 1]

    m0 = ak.where(ak.is_none(l0), False,
              ak.where(abs(l0.pdgId) == 11, l0.mvaTTH > 0.4,
              ak.where(abs(l0.pdgId) == 13, l0.mvaTTH > 0.5, False)))

    m1 = ak.where(ak.is_none(l1), False,
              ak.where(abs(l1.pdgId) == 11, l1.mvaTTH > 0.4,
              ak.where(abs(l1.pdgId) == 13, l1.mvaTTH > 0.5, False)))

    tthmva_mask_2l = m0 & m1
    return tthmva_mask_2l


regions = {}

# regions["preselections_e"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (abs(events.Lepton[:, 0].pdgId) == 11)
#     & (events.Lepton[:, 0].pt > 10) & (events.Lepton[:, 1].pt > 10)
#     & (events.PuppiMET.pt > 30)
#     & (tthmva(events)),
#     "mask": 0, 
# }

# regions["preselections_mu"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (abs(events.Lepton[:, 0].pdgId) == 13)
#     & (events.Lepton[:, 0].pt > 10) & (events.Lepton[:, 1].pt > 10)
#     & (events.PuppiMET.pt > 30)
#     & (tthmva(events)),
#     "mask": 0, 
# }

regions["VBS_SSWW"] = {
    "func": lambda events:  (ak.num(events.Lepton) >= 2)
    & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
    & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
    & (events.mll > 20)
    & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
         abs(events.Lepton[:, 1].pdgId) == 11
     )) & (abs(events.mll - 91.2) <= 15))
    & (events.mjj > 500)
    & (events.PuppiMET.pt > 30)
    & (abs(events.detajj) > 2.5)
    & (events.Zeppenfeld_Z <= 0.75)
    & (tthmva(events))
    & (events.bVeto),
    "mask": 0, 
}

# regions["Fake_enriched_ee"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId == 121),
#     "mask": 0, 
# }

# regions["Fake_enriched_mumu"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId == 169),
#     "mask": 0, 
# }

# regions["Fake_enriched_emu"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId == 143),
#     "mask": 0, 
# }

# regions["VBS_SSWW_l2e"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (abs(events.Lepton[:, 1].pdgId)== 11)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 500)
#     & (events.PuppiMET.pt > 30)
#     & (abs(events.detajj) > 2.5)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (events.bVeto),
#     "mask": 0, 
# }

# regions["VBS_SSWW_l2mu"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (abs(events.Lepton[:, 1].pdgId)== 13)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 500)
#     & (events.PuppiMET.pt > 30)
#     & (abs(events.detajj) > 2.5)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (events.bVeto),
#     "mask": 0, 
# }

regions["InverseMET"] = {
    "func": lambda events:  (ak.num(events.Lepton) >= 2)
    & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
    & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
    & (events.mll > 20)
    & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
         abs(events.Lepton[:, 1].pdgId) == 11
     )) & (abs(events.mll - 91.2) <= 15))
    & (events.mjj > 500)
    & (events.PuppiMET.pt < 30)
    & (abs(events.detajj) > 2.5)
    & (events.Zeppenfeld_Z <= 0.75)
    & (tthmva(events))
    & (events.bVeto),
    "mask": 0, 
}

# regions["VBS_SSWW_loose"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 300)
#     & (events.PuppiMET.pt > 30)
#     & (abs(events.detajj) > 2.0)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

# regions["VBS_SSWW_(ee)"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (abs(events.Lepton[:, 0].pdgId)==11) & (abs(events.Lepton[:, 1].pdgId)==11)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 500)
#     & (events.PuppiMET.pt > 30)
#     & (abs(events.detajj) > 2.5)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

# regions["VBS_SSWW_(mumu)"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (abs(events.Lepton[:, 0].pdgId)==13) & (abs(events.Lepton[:, 1].pdgId)==13)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 500)
#     & (events.PuppiMET.pt > 30)
#     & (abs(events.detajj) > 2.5)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

# regions["VBS_SSWW_(emu)"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (abs(events.Lepton[:, 0].pdgId)==11) & (abs(events.Lepton[:, 1].pdgId)==13)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 500)
#     & (events.PuppiMET.pt > 30)
#     & (abs(events.detajj) > 2.5)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

# regions["VBS_SSWW_(mue)"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (abs(events.Lepton[:, 0].pdgId)==13) & (abs(events.Lepton[:, 1].pdgId)==11)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 500)
#     & (events.PuppiMET.pt > 30)
#     & (abs(events.detajj) > 2.5)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

regions["SSWWb"] = {
    "func": lambda events:  (ak.num(events.Lepton) >= 2)
    & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
    & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
    & (events.mll > 20)
    & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
         abs(events.Lepton[:, 1].pdgId) == 11
     )) & (abs(events.mll - 91.2) <= 15))
    & (events.mjj > 500)
    & (events.PuppiMET.pt > 30)
    & (abs(events.detajj) > 2.5)
    & (events.Zeppenfeld_Z <= 0.75)
    & (tthmva(events))
    & ~(events.bVeto),
    "mask": 0, 
}

# regions["Low_mjj"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 250)
#     & (events.mjj < 500)
#     & (events.PuppiMET.pt > 30)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

# regions["Lowlow_mjj"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 250)
#     & (events.mjj < 350)
#     & (events.PuppiMET.pt > 30)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

# regions["LowHigh_mjj"] = {
#     "func": lambda events:  (ak.num(events.Lepton) >= 2)
#     & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
#     & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
#     # & (ak.num(events.Jet) >= 2)
#     # & (events.Jet[:, 0].pt > 50)
#     # & ((events.Jet[:, 1].pt > 30)
#     #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
#     # ) | (
#     #     (events.Jet[:, 1].pt > 50)
#     #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
#     & (events.mll > 20)
#     & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
#          abs(events.Lepton[:, 1].pdgId) == 11
#      )) & (abs(events.mll - 91.2) <= 15))
#     & (events.mjj > 350)
#     & (events.mjj < 500)
#     & (events.PuppiMET.pt > 30)
#     & (events.Zeppenfeld_Z <= 0.75)
#     & (tthmva(events))
#     & (events.bVeto),
#     "mask": 0, 
# }

#variables
    
variables = {}

# variables["events"] = {
#     "func": lambda events: events.Lepton[:, 0].mass, 
#     "axis": hist.axis.Regular(1, 0, 1, name="events"),
# }

# variables["tthMVA_l1"] = {
#     "func": lambda events: events.Lepton[:, 0].mvaTTH , 
#     "axis": hist.axis.Regular(6, 0.4, 1, name="tthMVA_l1"),
# }

# variables["tthMVA_l2"] = {
#     "func": lambda events: events.Lepton[:, 1].mvaTTH , 
#     "axis": hist.axis.Regular(6, 0.4, 1, name="tthMVA_l2"),
# }

# variables["tthMVA_l1_b"] = {
#     "func": lambda events: events.Lepton[:, 0].mvaTTH , 
#     "axis": hist.axis.Regular(3, 0.4, 1, name="tthMVA_l1_b"),
# }

# variables["tthMVA_l2_b"] = {
#     "func": lambda events: events.Lepton[:, 1].mvaTTH , 
#     "axis": hist.axis.Regular(3, 0.4, 1, name="tthMVA_l2_b"),
# }

# variables["tthMVA_l1_low"] = {
#     "func": lambda events: events.Lepton[:, 0].mvaTTH , 
#     "axis": hist.axis.Regular(4, 0.4, 1, name="tthMVA_l1_low"),
# }

# variables["tthMVA_l2_low"] = {
#     "func": lambda events: events.Lepton[:, 1].mvaTTH , 
#     "axis": hist.axis.Regular(4, 0.4, 1, name="tthMVA_l2_low"),
# }

# variables["tthMVA_l1_inverse"] = {
#     "func": lambda events: events.Lepton[:, 0].mvaTTH , 
#     "axis": hist.axis.Regular(2, 0.4, 1, name="tthMVA_l1_inverse"),
# }

# variables["tthMVA_l2_inverse"] = {
#     "func": lambda events: events.Lepton[:, 1].mvaTTH , 
#     "axis": hist.axis.Regular(2, 0.4, 1, name="tthMVA_l2_inverse"),
# }

# variables["btagPNetB1"] = {
#     "func": lambda events: events.jets[:, 0].btagPNetB , 
#     "axis": hist.axis.Regular(6, 0, 0.1, name="btagPNetB1"),
# }

# variables["btagPNetB2"] = {
#     "func": lambda events: events.jets[:, 0].btagPNetB , 
#     "axis": hist.axis.Regular(6, 0, 0.1, name="btagPNetB2"),
# }

# variables["pv"] = {
#     "func": lambda events: events.PV.npvsGood ,
#     "axis": hist.axis.Regular(60, 0, 60, name="pv"),
# }

# variables["pv_bin"] = {
#     "func": lambda events: events.PV.npvsGood ,
#     "axis": hist.axis.Regular(10, 0, 60, name="pv"),
# }

# variables["njet"] = {
#     "func": lambda events: ak.num(events.jets),
#     "axis": hist.axis.Regular(6, 2, 10, name="njet"),
# }

# Dijet
variables["mjj"] = {
    "func": lambda events: ak.fill_none(
        (events.jets[:, 0] + events.jets[:, 1]).mass, -9999
    ),
    "axis": hist.axis.Variable([500, 700, 1000, 1400, 1800, 2300, 3000], name="mjj"),
}

# variables["mjj_b"] = {
#     "func": lambda events: ak.fill_none(
#         (events.jets[:, 0] + events.jets[:, 1]).mass, -9999
#     ),
#     "axis": hist.axis.Variable([500, 800, 1300, 3000], name="mjj_b"),
# }

# variables["mjj_low"] = {
#     "func": lambda events: ak.fill_none(
#         (events.jets[:, 0] + events.jets[:, 1]).mass, -9999
#     ),
#     "axis": hist.axis.Regular(4, 250, 500, name="mjj_low"),
# }

# variables["mjj_inverse"] = {
#     "func": lambda events: ak.fill_none(
#         (events.jets[:, 0] + events.jets[:, 1]).mass, -9999
#     ),
#     "axis": hist.axis.Variable([500, 1000, 3000], name="mjj_inverse"),
# }

variables["detajj"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaeta(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(6, 2.5, 8, name="detajj"),
}

# variables["detajj_low"] = {
#     "func": lambda events: abs(
#         ak.fill_none(events.jets[:, 0].deltaeta(events.jets[:, 1]), -9999)
#     ),
#     "axis": hist.axis.Regular(4, 0, 5, name="detajj_low"),
# }

# variables["detajj_b"] = {
#     "func": lambda events: abs(
#         ak.fill_none(events.jets[:, 0].deltaeta(events.jets[:, 1]), -9999)
#     ),
#     "axis": hist.axis.Regular(3, 0, 6, name="detajj_b"),
# }

# variables["detajj_inverse"] = {
#     "func": lambda events: abs(
#         ak.fill_none(events.jets[:, 0].deltaeta(events.jets[:, 1]), -9999)
#     ),
#     "axis": hist.axis.Regular(2, 0, 8, name="detajj_inverse"),
# }


variables["dphijj"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaphi(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(6, 0, np.pi, name="dphijj"),
}

# variables["dphijj_b"] = {
#     "func": lambda events: abs(
#         ak.fill_none(events.jets[:, 0].deltaphi(events.jets[:, 1]), -9999)
#     ),
#     "axis": hist.axis.Regular(3, 0, np.pi, name="dphijj_b"),
# }

# variables["dphijj_low"] = {
#     "func": lambda events: abs(
#         ak.fill_none(events.jets[:, 0].deltaphi(events.jets[:, 1]), -9999)
#     ),
#     "axis": hist.axis.Regular(4, 0, np.pi, name="dphijj_low"),
# }

# variables["dphijj_inverse"] = {
#     "func": lambda events: abs(
#         ak.fill_none(events.jets[:, 0].deltaphi(events.jets[:, 1]), -9999)
#     ),
#     "axis": hist.axis.Regular(2, 0, np.pi, name="dphijj_inverse"),
# }

# # Single jet
variables["ptj1"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 0].pt, -9999),
    "axis": hist.axis.Regular(6, 50, 500, name="ptj1"),
}
variables["ptj2"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 1].pt, -9999),
    "axis": hist.axis.Regular(6, 30, 250, name="ptj2"),
}

# variables["ptj1_b"] = {
#     "func": lambda events: ak.fill_none(events.jets[:, 0].pt, -9999),
#     "axis": hist.axis.Regular(3, 50, 300, name="ptj1_b"),
# }

# variables["ptj1_low"] = {
#     "func": lambda events: ak.fill_none(events.jets[:, 0].pt, -9999),
#     "axis": hist.axis.Regular(4, 50, 300, name="ptj1_low"),
# }

# variables["ptj1_inverse"] = {
#     "func": lambda events: ak.fill_none(events.jets[:, 0].pt, -9999),
#     "axis": hist.axis.Regular(2, 50, 250, name="ptj1_inverse"),
# }

# variables["ptj2_b"] = {
#     "func": lambda events: ak.fill_none(events.jets[:, 1].pt, -9999),
#     "axis": hist.axis.Regular(3, 30, 200, name="ptj2_b"),
# }
# variables["ptj2_low"] = {
#     "func": lambda events: ak.fill_none(events.jets[:, 1].pt, -9999),
#     "axis": hist.axis.Regular(4, 30, 200, name="ptj2_low"),
# }

# variables["ptj2_inverse"] = {
#     "func": lambda events: ak.fill_none(events.jets[:, 1].pt, -9999),
#     "axis": hist.axis.Regular(2, 30, 150, name="ptj2_inverse"),
# }


# Dilepton
variables["mll"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    "axis": hist.axis.Variable([20, 60, 140, 250, 500], name="mll"),
    #"axis": hist.axis.Variable([0, 40, 75, 105, 230], name="mll"),
}

# variables["mll_b"] = {
#     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
#     "axis": hist.axis.Variable([20, 80, 200, 500], name="mll_b"),
# }

# variables["mll_low"] = {
#     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
#     "axis": hist.axis.Variable([20, 80, 250, 500], name="mll_low"),
# }

# variables["mll_inverse"] = {
#     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
#     "axis": hist.axis.Variable([20, 140, 500], name="mll_inverse"),
# }

variables["ptll"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
    "axis": hist.axis.Regular(6, 25, 300, name="ptll"),
}

# variables["ptll_low"] = {
#     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
#     "axis": hist.axis.Regular(4, 25, 170, name="ptll_low"),
# }

# variables["ptll_b"] = {
#     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
#     "axis": hist.axis.Regular(3, 25, 170, name="ptll_b"),
# }

# variables["ptll_inverse"] = {
#     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
#     "axis": hist.axis.Regular(2, 25, 250, name="ptll_inverse"),
# }

variables["dphill"] = {
    "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
    "axis": hist.axis.Regular(6, 0, np.pi, name="dphill"),
}

# variables["dphill_b"] = {
#     "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
#     "axis": hist.axis.Regular(3, 0, np.pi, name="dphill_b"),
# }
# variables["dphill_low"] = {
#     "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
#     "axis": hist.axis.Regular(4, 0, np.pi, name="dphill_low"),
# }
# variables["dphill_inverse"] = {
#     "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
#     "axis": hist.axis.Regular(2, 0, np.pi, name="dphill_inverse"),
# }

#Single lepton

variables["ptl1"] = {
    "func": lambda events: events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(6, 25, 300, name="ptl1"),
    #"axis": hist.axis.Regular(6, 0, 150, name="ptl1"),
 }

variables["ptl2"] = {
    "func": lambda events: events.Lepton[:, 1].pt,
    "axis": hist.axis.Regular(6, 20, 120, name="ptl2"),
    #"axis": hist.axis.Regular(6, 0, 60, name="ptl2"),
}

# variables["ptl1_emu"] = {
#     "func": lambda events: events.Lepton[:, 0].pt,
#     #"axis": hist.axis.Regular(6, 25, 300, name="ptl1"),
#     "axis": hist.axis.Regular(6, 0, 170, name="ptl1_emu"),
#  }

# variables["ptl2_emu"] = {
#     "func": lambda events: events.Lepton[:, 1].pt,
#     #"axis": hist.axis.Regular(6, 20, 120, name="ptl2"),
#     "axis": hist.axis.Regular(6, 0, 80, name="ptl2_emu"),
# }

# variables["ptl1_low"] = {
#     "func": lambda events: events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(4, 25, 200, name="ptl1_low"),
# }

# variables["ptl1_b"] = {
#     "func": lambda events: events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(3, 25, 200, name="ptl1_b"),
# }

# variables["ptl1_inverse"] = {
#     "func": lambda events: events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(2, 25, 200, name="ptl1_inverse"),
# }

# variables["ptl2_low"] = {
#     "func": lambda events: events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(4, 20, 120, name="ptl2_low"),
# }

# variables["ptl2_b"] = {
#     "func": lambda events: events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(3, 20, 120, name="ptl2_b"),
# }

# variables["ptl2_inverse"] = {
#     "func": lambda events: events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(2, 20, 120, name="ptl2_inverse"),
# }

variables["Zeppenfeld_Z"] = {
    "func": lambda events: ak.max(abs(events.Lepton.eta - 0.5 * (events.jets[:, 0].eta + events.jets[:, 1].eta))
        / events.detajj, axis=1),
    "axis": hist.axis.Regular(10, 0, 1, name="Zeppenfeld_Z"),
}

variables["MET"] = {
    "func": lambda events: events.PuppiMET.pt,
    "axis": hist.axis.Regular(6, 30, 350, name="MET"),
    #"axis": hist.axis.Regular(6, 0, 120, name="MET"), #fake
}

# variables["MET_emu"] = {
#     "func": lambda events: events.PuppiMET.pt,
#     #"axis": hist.axis.Regular(6, 30, 350, name="MET"),
#     "axis": hist.axis.Regular(6, 0, 200, name="MET_emu"), #fake
# }

# variables["MET_inverse"] = {
#     "func": lambda events: events.PuppiMET.pt,
#     "axis": hist.axis.Regular(2, 0, 30, name="MET_inverse"),
# }

# #mtWW
variables["mtWW"] = {
    "func": lambda events: np.sqrt(
        ((events.Lepton[:, 0] + events.Lepton[:, 1]).pt + events.PuppiMET.pt) ** 2
        - (
            (
                ak.zip(
                    {
                        "pt": (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
                        "phi": (events.Lepton[:, 0] + events.Lepton[:, 1]).phi,
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
            ).pt ** 2
        )
    ),
    "axis": hist.axis.Regular(6, 0, 300, name="mtWW"),
    #"axis": hist.axis.Regular(6, 0, 150, name="mtWW"), #fake
}

# variables["mtWW_low"] = {
#     "func": lambda events: np.sqrt(
#         ((events.Lepton[:, 0] + events.Lepton[:, 1]).pt + events.PuppiMET.pt) ** 2
#         - (
#             (
#                 ak.zip(
#                     {
#                         "pt": (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
#                         "phi": (events.Lepton[:, 0] + events.Lepton[:, 1]).phi,
#                     },
#                     with_name="Momentum2D",
#                 )
#                 + ak.zip(
#                     {
#                         "pt": events.PuppiMET.pt,
#                         "phi": events.PuppiMET.phi,
#                     },
#                     with_name="Momentum2D",
#                 )
#             ).pt ** 2
#         )
#     ),
#     "axis": hist.axis.Regular(4, 0, 300, name="mtWW_low"),
# }

# variables["mtWW_b"] = {
#     "func": lambda events: np.sqrt(
#         ((events.Lepton[:, 0] + events.Lepton[:, 1]).pt + events.PuppiMET.pt) ** 2
#         - (
#             (
#                 ak.zip(
#                     {
#                         "pt": (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
#                         "phi": (events.Lepton[:, 0] + events.Lepton[:, 1]).phi,
#                     },
#                     with_name="Momentum2D",
#                 )
#                 + ak.zip(
#                     {
#                         "pt": events.PuppiMET.pt,
#                         "phi": events.PuppiMET.phi,
#                     },
#                     with_name="Momentum2D",
#                 )
#             ).pt ** 2
#         )
#     ),
#     "axis": hist.axis.Regular(3, 0, 300, name="mtWW_b"),
# }

# variables["mtWW_inverse"] = {
#     "func": lambda events: np.sqrt(
#         ((events.Lepton[:, 0] + events.Lepton[:, 1]).pt + events.PuppiMET.pt) ** 2
#         - (
#             (
#                 ak.zip(
#                     {
#                         "pt": (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
#                         "phi": (events.Lepton[:, 0] + events.Lepton[:, 1]).phi,
#                     },
#                     with_name="Momentum2D",
#                 )
#                 + ak.zip(
#                     {
#                         "pt": events.PuppiMET.pt,
#                         "phi": events.PuppiMET.phi,
#                     },
#                     with_name="Momentum2D",
#                 )
#             ).pt ** 2
#         )
#     ),
#     "axis": hist.axis.Regular(2, 0, 300, name="mtWW_inverse"),
# }



# # #Rpt, Hrt and thetaCS
variables["Rpt"] = {
    "func": lambda events: (events.Lepton[:, 0].pt*events.Lepton[:, 1].pt)/(events.jets[:, 0].pt*events.jets[:, 1].pt),
    "axis": hist.axis.Regular(6, 0, 1.5, name="Rpt"),
}

variables["Ht"] = {
    "func": lambda events: ak.sum(events.jets.pt, axis=1)/events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(6, 0, 13, name="Ht"),
}

# variables["Rpt_low"] = {
#     "func": lambda events: (events.Lepton[:, 0].pt*events.Lepton[:, 1].pt)/(events.jets[:, 0].pt*events.jets[:, 1].pt),
#     "axis": hist.axis.Regular(4, 0, 1.5, name="Rpt_low"),
# }

# variables["Ht_low"] = {
#     "func": lambda events: ak.sum(events.jets.pt, axis=1)/events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(4, 0, 13, name="Ht_low"),
# }

# variables["Rpt_b"] = {
#     "func": lambda events: (events.Lepton[:, 0].pt*events.Lepton[:, 1].pt)/(events.jets[:, 0].pt*events.jets[:, 1].pt),
#     "axis": hist.axis.Regular(3, 0, 1.5, name="Rpt_b"),
# }

# variables["Ht_b"] = {
#     "func": lambda events: ak.sum(events.jets.pt, axis=1)/events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(3, 0, 13, name="Ht_b"),
# }

# variables["Rpt_inverse"] = {
#     "func": lambda events: (events.Lepton[:, 0].pt*events.Lepton[:, 1].pt)/(events.jets[:, 0].pt*events.jets[:, 1].pt),
#     "axis": hist.axis.Regular(2, 0, 1.5, name="Rpt_inverse"),
# }

# variables["Ht_inverse"] = {
#     "func": lambda events: ak.sum(events.jets.pt, axis=1)/events.Lepton[:, 0].pt,
#     "axis": hist.axis.Regular(2, 0, 13, name="Ht_inverse"),
# }

# variables["cos_theta_CS"] = {
#     "func": lambda events: cos_theta_collins_soper(build_4vector(events.Lepton[:, 0]),build_4vector(events.Lepton[:, 1])),
#     "axis": hist.axis.Regular(10, -1, 1, name="cos_theta_CS"),
# }

#single eta
variables["etaj1"] = {
    "func": lambda events: 
        (ak.fill_none(abs(events.jets[:, 0].eta), -9999)),
    "axis": hist.axis.Regular(6, 0, 4.7, name="etaj1"),
}

variables["etaj2"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.jets[:, 1].eta), -9999),
    "axis": hist.axis.Regular(6, 0, 4.7, name="etaj2"),
}

variables["etal1"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 0].eta), -9999),
    "axis": hist.axis.Regular(6, 0, 2.5, name="etal1"),
}

variables["etal2"] = {
    "func": lambda events: 
        ak.fill_none(abs(events.Lepton[:, 1].eta), -9999),
    "axis": hist.axis.Regular(6, 0, 2.5, name="etal2"),
}

# variables["etaj1_low"] = {
#     "func": lambda events: 
#         (ak.fill_none(abs(events.jets[:, 0].eta), -9999)),
#     "axis": hist.axis.Regular(4, 0, 4.7, name="etaj1_low"),
# }

# variables["etaj2_low"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.jets[:, 1].eta), -9999),
#     "axis": hist.axis.Regular(4, 0, 4.7, name="etaj2_low"),
# }

# variables["etal1_low"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.Lepton[:, 0].eta), -9999),
#     "axis": hist.axis.Regular(4, 0, 2.5, name="etal1_low"),
# }

# variables["etal2_low"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.Lepton[:, 1].eta), -9999),
#     "axis": hist.axis.Regular(4, 0, 2.5, name="etal2_low"),
# }

# variables["etaj1_b"] = {
#     "func": lambda events: 
#         (ak.fill_none(abs(events.jets[:, 0].eta), -9999)),
#     "axis": hist.axis.Regular(3, 0, 4.7, name="etaj1_b"),
# }

# variables["etaj2_b"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.jets[:, 1].eta), -9999),
#     "axis": hist.axis.Regular(3, 0, 4.7, name="etaj2_b"),
# }

# variables["etal1_b"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.Lepton[:, 0].eta), -9999),
#     "axis": hist.axis.Regular(3, 0, 2.5, name="etal1_b"),
# }

# variables["etal2_b"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.Lepton[:, 1].eta), -9999),
#     "axis": hist.axis.Regular(3, 0, 2.5, name="etal2_b"),
# }

# variables["etaj1_inverse"] = {
#     "func": lambda events: 
#         (ak.fill_none(abs(events.jets[:, 0].eta), -9999)),
#     "axis": hist.axis.Regular(2, 0, 4.7, name="etaj1_inverse"),
# }

# variables["etaj2_inverse"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.jets[:, 1].eta), -9999),
#     "axis": hist.axis.Regular(2, 0, 4.7, name="etaj2_inverse"),
# }

# variables["etal1_inverse"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.Lepton[:, 0].eta), -9999),
#     "axis": hist.axis.Regular(2, 0, 2.5, name="etal1_inverse"),
# }

# variables["etal2_inverse"] = {
#     "func": lambda events: 
#         ak.fill_none(abs(events.Lepton[:, 1].eta), -9999),
#     "axis": hist.axis.Regular(2, 0, 2.5, name="etal2_inverse"),
# }

# variables["run_period"] = {
#     "func": lambda events: events.run_period,
#     "axis": hist.axis.Regular(30, -1, 10, name="run_period"),
# }

 #variabili 2d
variables["mjj_vs_mll"] = {
    "axis": (
        hist.axis.Variable([500, 1000, 3000], name="mjj"),
        hist.axis.Variable([20, 80, 200, 500], name="mll"),
    ),
}

variables["mjj_vs_dphill"] = {
    "axis": (
        hist.axis.Variable([500, 1000, 3000], name="mjj"),
        hist.axis.Regular(3, 0, np.pi, name="dphill"),
    ),
}

variables["mjj_vs_ptl1"] = {
    "axis": (
        hist.axis.Variable([500, 1000, 3000], name="mjj"),
        hist.axis.Regular(3, 25, 300, name="ptl1"),
    ),
}

variables["detajj_vs_ptl1"] = {
    "axis": (
        hist.axis.Regular(2, 2.5, 8, name="detajj"),
        hist.axis.Regular(3, 25, 300, name="ptl1"),
    ),
}

variables["detajj_vs_dphill"] = {
    "axis": (
        hist.axis.Regular(2, 2.5, 8, name="detajj"),
        hist.axis.Regular(3, 0, np.pi, name="dphill"),
    ),
}

variables["mjj_vs_Rpt"] = {
    "axis": (
        hist.axis.Variable([500, 1000, 3000], name="mjj"),
        hist.axis.Regular(3, 0, 1.5, name="Rpt"),
    ),
}



#nuisances

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
}

#quella calcolata in in evaluate
nuisances["FW"] = {
        "name": "CMS_fake_stat",
        "kind": "suffix",
        "type": "shape",
        "samples": dict((skey, ["1", "1"]) for skey in fake),
}

for sname in ["WZ_EWK", "WZ_QCD", "SSWW_QCD", "SSWW_LL", "SSWW_TL", "SSWW_TL"]:
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



