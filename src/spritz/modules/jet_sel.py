import awkward as ak
import numba
import numpy as np


def jetSel(events, cfg):
    # jetId = 2, puId = "loose", minpt = 15.0, maxeta = 4.7,"CleanJet",False
    jetId = 2
    minpt = 15.0
    maxeta = 4.7

    jetId = cfg["jet_sel"]["jetId"]
    minpt = cfg["jet_sel"]["minpt"]
    maxeta = cfg["jet_sel"]["maxeta"]
    jet = events.Jet

    # pu loose
    if "2016" not in cfg["era"]:
        puId_shift = 1 << 2
    else:
        puId_shift = 1 << 0

    #pass_puId = ak.values_astype(jet.puId & puId_shift, bool)
    select = jet.pt >= minpt
    select = select & (abs(jet.eta) <= maxeta)
    select = select & (jet.jetId >= jetId)
    select = select & ((jet.pt > minpt))
    events["Jet"] = events.Jet[select]
    return events


@numba.njit
def goodJet_kernel(jet, lepton, builder):
    for ievent in range(len(jet)):
        builder.begin_list()
        for ijet in range(len(jet[ievent])):
            dRs = np.ones(len(lepton[ievent])) * 10
            for ipart in range(len(lepton[ievent])):
                single_jet = jet[ievent][ijet]
                single_lepton = lepton[ievent][ipart]
                dRs[ipart] = single_jet.deltaR(single_lepton)
            builder.boolean(~np.any(dRs < 0.3))
        builder.end_list()
    return builder


def goodJet_func(jets, leptons):
    if ak.backend(jets) == "typetracer":
        # here we fake the output of find_4lep_kernel since
        # operating on length-zero data returns the wrong layout!
        ak.typetracer.length_zero_if_typetracer(
            jets.pt
        )  # force touching of the necessary data
        return ak.Array(ak.Array([[True]]).layout.to_typetracer(forget_length=True))

    return goodJet_kernel(jets, leptons, ak.ArrayBuilder()).snapshot()


def cleanJet(events):
    mask = goodJet_func(events.Jet, events.Lepton[events.Lepton.pt >= 10])
    mask = ak.values_astype(mask, bool, including_unknown=True)

    events["Jet"] = events.Jet[mask]
    return events
