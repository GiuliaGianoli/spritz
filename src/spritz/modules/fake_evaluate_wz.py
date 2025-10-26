import awkward as ak
from spritz.framework.framework import correctionlib_wrapper

def fake_evaluate_wz(events, ceval_fake, cfg):
    wrap_ElePR = correctionlib_wrapper(ceval_fake["ElePR"])
    wrap_MuPR = correctionlib_wrapper(ceval_fake["MuPR"])

    lep = ak.copy(events.Lepton[(abs(events.Lepton.pdgId) == 11) | (abs(events.Lepton.pdgId) == 13)])
    lep = ak.pad_none(lep, 3, axis=1, clip=True)

    lep["pt"] = ak.where(lep.pt >= 50.0, 49.999, lep.pt)
    lep["pt"] = ak.where(lep.pt <= 10.0, 10.001, lep.pt)
    lep["eta"] = abs(lep.eta)

    ele_mask = abs(lep.pdgId) == 11
    mu_mask = abs(lep.pdgId) == 13

    eleWP = cfg["leptonsWP"]["eleWP"]
    muWP = cfg["leptonsWP"]["muWP"]

    lep["isTight"] = ak.zeros_like(lep.pt, dtype=bool)
    lep["isTight"] = ak.where(ele_mask, lep["isTightElectron_" + eleWP], lep["isTight"])
    lep["isTight"] = ak.where(mu_mask, lep["isTightMuon_" + muWP], lep["isTight"])

    # tags = [
    #     ("3l0j", "EleFR_jet35", "MuFR_jet20"),
    #     ("3l1j", "EleFR_jet35", "MuFR_jet25"),
    #     ("3l2j", "EleFR_jet35", "MuFR_jet35"),
    # ]


    tags = [
        ("3l0j", "EleFR_jet35", "MuFR_jet20"),
        ("3l1j", "EleFR_jet35", "MuFR_jet25"),
        ("3l2j", "EleFR_jet40", "MuFR_jet40"),
    ]

    if len(events) == 0:
        return events

    njets = ak.num(events.Jet[events.Jet.pt >= 30], axis=1)

    for fakeTag, eleName, muName in tags:
        wrap_EleFR = correctionlib_wrapper(ceval_fake[eleName])
        wrap_MuFR = correctionlib_wrapper(ceval_fake[muName])

        lep["PR"] = ak.ones_like(lep.pt)
        lep["FR"] = ak.zeros_like(lep.pt)

        _lep = ak.mask(lep, ele_mask)
        lep["PR"] = ak.where(ele_mask, wrap_ElePR(_lep.pt, _lep.eta), lep["PR"])
        lep["FR"] = ak.where(ele_mask, wrap_EleFR(_lep.pt, _lep.eta), lep["FR"])

        _lep = ak.mask(lep, mu_mask)
        lep["PR"] = ak.where(mu_mask, wrap_MuPR(_lep.pt, _lep.eta), lep["PR"])
        lep["FR"] = ak.where(mu_mask, wrap_MuFR(_lep.pt, _lep.eta), lep["FR"])

        lep["prompt_prob"] = ak.where(
            lep.isTight,
            lep.PR * (1 - lep.FR) / (lep.PR - lep.FR),
            lep.PR * lep.FR / (lep.PR - lep.FR),
        )

        lep["fake_prob"] = ak.where(
            lep.isTight,
            lep.FR * (1 - lep.PR) / (lep.PR - lep.FR),
            lep.FR * lep.PR / (lep.PR - lep.FR),
        )

        P = lep.prompt_prob
        F = lep.fake_prob

        events["PPP"] = P[:, 0] * P[:, 1] * P[:, 2]
        events["PPF"] = P[:, 0] * P[:, 1] * F[:, 2]
        events["PFP"] = P[:, 0] * F[:, 1] * P[:, 2]
        events["FPP"] = F[:, 0] * P[:, 1] * P[:, 2]
        events["PFF"] = P[:, 0] * F[:, 1] * F[:, 2]
        events["FPF"] = F[:, 0] * P[:, 1] * F[:, 2]
        events["FFP"] = F[:, 0] * F[:, 1] * P[:, 2]
        events["FFF"] = F[:, 0] * F[:, 1] * F[:, 2]

        n_tight = ak.num(lep[lep.isTight], axis=1)


        # 1 o 3 tight leptoni
        for key in ["PPF", "PFP", "FPP", "FFF"]:
            events[key] = ak.where((n_tight == 1) | (n_tight == 3), -1 * events[key], events[key])

        # 0 o 2 tight
        for key in ["PFF", "FPF", "FFP"]:
            events[key] = ak.where((n_tight != 1) & (n_tight != 3), -1 * events[key], events[key])

        # fake weight
        fake_weight = (
            events["PPF"] + events["PFP"] + events["FPP"] +
            events["PFF"] + events["FPF"] + events["FFP"] +
            events["FFF"]
        )

        events[f"fake_weight_{fakeTag}"] = fake_weight

    masks = {
        "0j": (njets == 0),
        "1j": (njets == 1),
        "2j": (njets >= 2),
    }

    events["fakeWeight"] = ak.where(
        masks["0j"],
        events.fake_weight_3l0j,
        ak.where(masks["1j"], events.fake_weight_3l1j, events.fake_weight_3l2j),
    )

    return events
