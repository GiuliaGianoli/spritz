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

# lumi = lumis[year]["B"] / 1000  # ERA C of 2017
#lumi = lumis[year]["tot"] / 1000  # All of 2017 # da cambiare
lumi = lumis[year]["tot"]
plot_label = "VBS-SSWW"
year_label = "2022EE"
njobs = 200

datasets = {}

datasets["SSWW_EWK"] = {
    "files": "SSWW_EWK",
    "task_weight": 8,
}

datasets["SSWW_QCD"] = {
    "files": "SSWW_QCD",
    "task_weight": 8,
}

datasets["WZ_EWK"] = {
    "files": "WZ_EWK",
    "task_weight": 8,
}

datasets["WZ_QCD"] = {
    "files": "WZ_QCD",
    "task_weight": 8,
}

datasets["W_1JET"] = {
    "files": "W_1JET",
    "task_weight": 8,
}

datasets["W_2JET"] = {
    "files": "W_2JET",
    "task_weight": 8,
}

datasets["W_3JET"] = {
    "files": "W_3JET",
    "task_weight": 8,
}

datasets["W_4JET"] = {
    "files": "W_4JET",
    "task_weight": 8,
}

datasets["TTBAR"] = {
    "files": "TTBAR",
    "task_weight": 8,
}

datasets["SSWW_TT"] = {
    "files": "SSWW_TT",
    "task_weight": 8,
}

datasets["SSWW_TL"] = {
    "files": "SSWW_TL",
    "task_weight": 8,
}

datasets["SSWW_LL"] = {
    "files": "SSWW_LL",
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
for era, sd in DataRun:
    for pd in DataSets:
        tag = pd + "_" + sd

        datasets[f"{pd}_{era}"] = {
            "files": tag,
            "trigger_sel": DataTrig[pd],
            "read_form": "data",
            "is_data": True,
            "era": f"2022EE_{era}",
        }
        samples_data.append(f"{pd}_{era}")


samples = {}
colors = {}

samples["Data"] = {
    "samples": samples_data,
    "is_data": True,
}

samples["SSWW_EWK"] = {
    "samples": ["SSWW_EWK"],
    "is_signal": True,
}

colors["SSWW_EWK"] = '#004488'

samples["SSWW_QCD"] = {
    "samples": ["SSWW_QCD"],
    "is_signal": False,
}
colors["SSWW_QCD"] = '#DDAA33'

samples["SSWW_TT"] = {
    "samples": ["SSWW_TT"],
    "is_signal": True,
}
colors["SSWW_TT"] = '#BB5566'

samples["SSWW_TL"] = {
    "samples": ["SSWW_TL"],
    "is_signal": True,
}
colors["SSWW_TL"] = '#117733'

samples["SSWW_LL"] = {
    "samples": ["SSWW_LL"],
    "is_signal": True,
}
colors["SSWW_LL"] = '#332288'

samples["WZ_EWK"] = {
    "samples": ["WZ_EWK"],
    "is_signal": False,
}
colors["WZ_EWK"] = '#88CCEE'

samples["WZ_QCD"] = {
    "samples": ["WZ_QCD"],
    "is_signal": False,
}
colors["WZ_QCD"] = '#44AA99'

samples["TTBAR"] = {
    "samples": ["TTBAR"],
    "is_signal": False,
}
colors["TTBAR"] = '#556270'

samples["W_JETS"] = {
    "samples": [f"W_{j}JET" for j in range(1, 5)],
}
colors["W_JETS"] = '#CC99CC'

# samples["LL+TT+TL"] = {
#      "samples": ["SSWW_LL", "SSWW_TT", "SSWW_TL" ],
#  }
# colors["LL+TT+TL"] = '#556270'



# regions

regions = {}

regions["VBS-SSWW"] = {
    "func": lambda events:  (ak.num(events.Lepton) >= 2)
    & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
    & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
    # & (ak.num(events.Jet) >= 2)
    # & (events.Jet[:, 0].pt > 50)
    # & ((events.Jet[:, 1].pt > 30)
    #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
    # ) | (
    #     (events.Jet[:, 1].pt > 50)
    #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
    & (events.mll > 20)
    & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
         abs(events.Lepton[:, 1].pdgId) == 11
     )) & (abs(events.mll - 91.2) <= 15))
    & (events.mjj > 500)
    & (events.PuppiMET.pt > 30)
    & (abs(events.detajj) > 2.5)
    & (events.Zeppenfeld_Z <= 0.75)
    & (events.bVeto),
    "mask": 0, 
}

regions["SSWWb"] = {
    "func": lambda events:  (ak.num(events.Lepton) >= 2)
    & (events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId > 0)
    & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20)
    # & (ak.num(events.Jet) >= 2)
    # & (events.Jet[:, 0].pt > 50)
    # & ((events.Jet[:, 1].pt > 30)
    #     & ((abs(events.Jet[:, 1].eta) < 2.5) | (abs(events.Jet[:, 1].eta) > 3.0))
    # ) | (
    #     (events.Jet[:, 1].pt > 50)
    #     & ((abs(events.Jet[:, 1].eta) > 2.5) & (abs(events.Jet[:, 1].eta) < 3.0)))
    & (events.mll > 20)
    & ~(((abs(events.Lepton[:, 0].pdgId) == 11) & (
         abs(events.Lepton[:, 1].pdgId) == 11
     )) & (abs(events.mll - 91.2) <= 15))
    & (events.mjj > 500)
    & (events.PuppiMET.pt > 30)
    & (abs(events.detajj) > 2.5)
    & (events.Zeppenfeld_Z <= 0.75)
    & ~(events.bVeto),
    "mask": 0, 
}


# def select_wz_region(events):
#     # 1. Picking first 3 leptons 
#     leptons3 = events.Lepton[:, :3]

#     # 2. Generate all pairs among the 3 leptons
#     lep_pairs = ak.combinations(leptons3, 2, fields=["l1", "l2"])

#     # 3. OS-SF filter: pairs with opposite charge and same flavor
#     os_sf_mask = (
#         (lep_pairs.l1.pdgId + lep_pairs.l2.pdgId == 0)
#         & (abs(lep_pairs.l1.pdgId) == abs(lep_pairs.l2.pdgId))
#     )
#     os_sf_pairs = lep_pairs[os_sf_mask]

#     # 4. Compute invariant mass of each OS-SF pair
#     z_cand_mass = (os_sf_pairs.l1 + os_sf_pairs.l2).mass

#     # 5. Choose the pair closest to nominal Z-mass (91.2 GeV)
#     z_mass_diff = abs(z_cand_mass - 91.2)
#     best_z_idx = ak.argmin(z_mass_diff, axis=1)

#     # 6. Extract the best Z candidate pair per event
#     best_z_pair = os_sf_pairs[best_z_idx]

#     # 7. Their invariant mass
#     best_z_mass = (best_z_pair.l1 + best_z_pair.l2).mass

#     # 8. Ensure at least one valid OS-SF pair exists
#     valid_z = ak.num(os_sf_pairs) > 0

#     #mlll
#     mlll = (events.Lepton[:, 0] + events.Lepton[:, 1] + events[ak.num(events.Lepton) >= 3].Lepton[:, 2]).mass

#     # 9. Build final mask including mass window
#     mask = ( (ak.num(events.Lepton) >= 3)      
#         &  valid_z
#         & (mlll >100)
#         & (events.mjj>500)
#         & (best_z_mass >= 91.2 - 15)
#         & (best_z_mass <= 91.2 + 15)
#         & (events.mjj > 500)
#         & (events.PuppiMET.pt > 30)
#         & (abs(events.detajj) > 2.5)
#         & (events.Zeppenfeld_Z <= 1.0)
#     )

#     return mask, best_z_pair, best_z_mass

# regions["WZ"] = {
#     "func": lambda events: (ak.num(events.Lepton) >= 3) &  (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20) & (events.Lepton[:, 2].pt > 10) & select_wz_region(events)[0],
#     "mask": 0,
# }

# regions["WZb"] = {
#     "func": lambda events: (ak.num(events.Lepton) >= 3) & (events.Lepton[:, 0].pt > 25) & (events.Lepton[:, 1].pt > 20) & (events.Lepton[:, 2].pt > 10) & select_wz_region(events)[0], 
#     "mask": 0,
# }

regions["WZ"] = {
    "func": lambda events: 
            #& select_wz_region(events)[0]
            (events.bVeto),
            #& ak.num(events.Lepton) >= 3,
     "mask": 0,
}

# regions["WZb"] = {
#     "func": lambda events: ak.fill_none(
#         ak.mask(
#             (events.Lepton[:, 0].pt > 25)
#             & (events.Lepton[:, 1].pt > 20)
#             & ak.where(
#                 ak.num(events.Lepton) >= 3,
#                 events.Lepton[:, 2].pt > 10,
#                 False,
#             )
#             & select_wz_region(events)[0]
#             & ~(events.bVeto),
#             ak.num(events.Lepton) >= 3
#         ),
#         False
#     )
# }
#variables

variables = {}

# variables["njet"] = {
#     "func": lambda events: events.njet,
#     "axis": hist.axis.Regular(6, 0, 6, name="njet"),
# }

# Dijet
variables["mjj"] = {
    "func": lambda events: ak.fill_none(
        (events.jets[:, 0] + events.jets[:, 1]).mass, -9999
    ),
    "axis": hist.axis.Variable([500, 650, 800, 1000, 1200, 1500, 1800, 2300, 3000], name="mjj"),
}

variables["detajj"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaeta(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(10, 2.5, 8, name="detajj"),
}


variables["dphijj"] = {
    "func": lambda events: abs(
        ak.fill_none(events.jets[:, 0].deltaphi(events.jets[:, 1]), -9999)
    ),
    "axis": hist.axis.Regular(10, 0, np.pi, name="dphijj"),
}

# Single jet
variables["ptj1"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 0].pt, -9999),
    "axis": hist.axis.Regular(10, 50, 500, name="ptj1"),
}
variables["ptj2"] = {
    "func": lambda events: ak.fill_none(events.jets[:, 1].pt, -9999),
    "axis": hist.axis.Regular(10, 30, 250, name="ptj2"),
}


# Dilepton
variables["mll"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    "axis": hist.axis.Variable([20, 60, 140, 250, 500], name="mll"),
}

# variables["mlll"] = {
#     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1] + events.Lepton[:, 2]).mass,
# }

variables["ptll"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
    "axis": hist.axis.Regular(10, 25, 250, name="ptll"),
}

variables["dphill"] = {
    "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
    "axis": hist.axis.Regular(10, 0, np.pi, name="dphill"),
}

# Single lepton
variables["ptl1"] = {
    "func": lambda events: events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(10, 25, 300, name="ptl1"),
}
variables["ptl2"] = {
    "func": lambda events: events.Lepton[:, 1].pt,
    "axis": hist.axis.Regular(10, 20, 200, name="ptl2"),
}


variables["Zeppenfeld_Z"] = {
    "func": lambda events: ak.max(abs(events.Lepton.eta - 0.5 * (events.jets[:, 0].eta + events.jets[:, 1].eta))
        / events.detajj, axis=1),
}

variables["MET"] = {
    "func": lambda events: events.PuppiMET.pt,
}

#mtWW
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
    "axis": hist.axis.Regular(10, 0, 300, name="mtWW"),
}

#Rpt, Hrt and thetaCS
variables["Rpt"] = {
    "func": lambda events: (events.Lepton[:, 0].pt*events.Lepton[:, 1].pt)/(events.jets[:, 0].pt*events.jets[:, 1].pt),
    "axis": hist.axis.Regular(10, 0, 1.5, name="Rpt"),
}

variables["Ht"] = {
    "func": lambda events: ak.sum(events.jets.pt, axis=1)/events.Lepton[:, 0].pt,
    "axis": hist.axis.Regular(10, 0, 13, name="Ht"),
}

variables["cos_theta_CS"] = {
    "func": lambda events: cos_theta_collins_soper(build_4vector(events.Lepton[:, 0]),build_4vector(events.Lepton[:, 1])),
    "axis": hist.axis.Regular(10, -1, 1, name="cos_theta_CS"),
}

#single eta
variables["etaj1"] = {
    "func": lambda events: 
        ak.fill_none(events.jets[:, 0].eta, -9999),
    "axis": hist.axis.Regular(10, -4.7, 4.7, name="etaj1"),
}

variables["etaj2"] = {
    "func": lambda events: 
        ak.fill_none(events.jets[:, 1].eta, -9999),
    "axis": hist.axis.Regular(10, -4.7, 4.7, name="etaj2"),
}

variables["etal1"] = {
    "func": lambda events: 
        ak.fill_none(events.Lepton[:, 0].eta, -9999),
    "axis": hist.axis.Regular(10, -2.5, 2.5, name="etal1"),
}

variables["etal2"] = {
    "func": lambda events: 
        ak.fill_none(events.Lepton[:, 1].eta, -9999),
    "axis": hist.axis.Regular(10, -2.5, 2.5, name="etal2"),
}



# variables["run_period"] = {
#     "func": lambda events: events.run_period,
#     "axis": hist.axis.Regular(30, -1, 10, name="run_period"),
# }

nuisances = {}


