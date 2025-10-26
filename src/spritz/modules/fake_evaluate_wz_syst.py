import awkward as ak
from spritz.framework.framework import correctionlib_wrapper
from spritz.framework.variation import Variation

def fake_evaluate_wz_syst(events, variations: Variation, ceval_fake, cfg):
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

    # Mapping: tag → (ele_nom, mu_nom, ele_up, mu_up, ele_down, mu_down)
    # tags = {
    #     "3l0j": ("EleFR_jet35", "MuFR_jet20", "EleFR_jet40", "MuFR_jet25", "EleFR_jet30", "MuFR_jet15"),
    #     "3l1j": ("EleFR_jet35", "MuFR_jet25", "EleFR_jet40", "MuFR_jet30", "EleFR_jet30", "MuFR_jet20"),
    #     "3l2j": ("EleFR_jet40", "MuFR_jet40", "EleFR_jet45", "MuFR_jet45", "EleFR_jet35", "MuFR_jet35"),
    # }


    #Mapping: tag → (ele_nom, mu_nom, ele_up, mu_up, ele_down, mu_down)
    tags = {
        "3l0j": ("EleFR_jet35", "MuFR_jet35", "EleFR_jet40", "MuFR_jet40", "EleFR_jet30", "MuFR_jet30"),
        "3l1j": ("EleFR_jet35", "MuFR_jet35", "EleFR_jet40", "MuFR_jet40", "EleFR_jet30", "MuFR_jet30"),
        "3l2j": ("EleFR_jet35", "MuFR_jet35", "EleFR_jet40", "MuFR_jet40", "EleFR_jet30", "MuFR_jet30"),
    }

    

    if len(events) == 0:
        return events, variations

    njets = ak.num(events.Jet[events.Jet.pt >= 30], axis=1)

    fake_weights = {"": {}, "_up": {}, "_down": {}}
    for var_suffix, var_tag in zip(["", "_up", "_down"], ["nom", "FakeUp", "FakeDown"]):
        for fakeTag, (ele_nom, mu_nom, ele_up, mu_up, ele_down, mu_down) in tags.items():
            if var_suffix == "":
                eleFR = ele_nom
                muFR = mu_nom
            elif var_suffix == "_up":
                eleFR = ele_up
                muFR = mu_up
            elif var_suffix == "_down":
                eleFR = ele_down
                muFR = mu_down

            lep_tag = ak.copy(lep)

            wrap_EleFR = correctionlib_wrapper(ceval_fake[eleFR])
            wrap_MuFR = correctionlib_wrapper(ceval_fake[muFR])

            lep_tag["PR"] = ak.ones_like(lep_tag.pt)
            lep_tag["FR"] = ak.zeros_like(lep_tag.pt)

            _lep_ele = ak.mask(lep_tag, ele_mask)
            lep_tag["PR"] = ak.where(ele_mask, wrap_ElePR(_lep_ele.pt, _lep_ele.eta), lep_tag["PR"])
            lep_tag["FR"] = ak.where(ele_mask, wrap_EleFR(_lep_ele.pt, _lep_ele.eta), lep_tag["FR"])

            _lep_mu = ak.mask(lep_tag, mu_mask)
            lep_tag["PR"] = ak.where(mu_mask, wrap_MuPR(_lep_mu.pt, _lep_mu.eta), lep_tag["PR"])
            lep_tag["FR"] = ak.where(mu_mask, wrap_MuFR(_lep_mu.pt, _lep_mu.eta), lep_tag["FR"])

            lep_tag["prompt_prob"] = ak.where(
                lep_tag.isTight,
                lep_tag.PR * (1 - lep_tag.FR) / (lep_tag.PR - lep_tag.FR),
                lep_tag.PR * lep_tag.FR / (lep_tag.PR - lep_tag.FR),
            )

            lep_tag["fake_prob"] = ak.where(
                lep_tag.isTight,
                lep_tag.FR * (1 - lep_tag.PR) / (lep_tag.PR - lep_tag.FR),
                lep_tag.FR * lep_tag.PR / (lep_tag.PR - lep_tag.FR),
            )

            P = lep_tag.prompt_prob
            F = lep_tag.fake_prob

            terms = {
                "PPP": P[:, 0] * P[:, 1] * P[:, 2],
                "PPF": P[:, 0] * P[:, 1] * F[:, 2],
                "PFP": P[:, 0] * F[:, 1] * P[:, 2],
                "FPP": F[:, 0] * P[:, 1] * P[:, 2],
                "PFF": P[:, 0] * F[:, 1] * F[:, 2],
                "FPF": F[:, 0] * P[:, 1] * F[:, 2],
                "FFP": F[:, 0] * F[:, 1] * P[:, 2],
                "FFF": F[:, 0] * F[:, 1] * F[:, 2],
            }

            n_tight = ak.num(lep_tag[lep_tag.isTight], axis=1)

            for key in ["PPF", "PFP", "FPP", "FFF"]:
                terms[key] = ak.where((n_tight == 1) | (n_tight == 3), -1 * terms[key], terms[key])
            for key in ["PFF", "FPF", "FFP"]:
                terms[key] = ak.where((n_tight != 1) & (n_tight != 3), -1 * terms[key], terms[key])

            fake_weight = (
                terms["PPF"] + terms["PFP"] + terms["FPP"]
                + terms["PFF"] + terms["FPF"] + terms["FFP"]
                + terms["FFF"]
            )
            fake_weights[var_suffix][fakeTag] = fake_weight


    # Applica selezione per njets
    masks = {
        "0j": (njets == 0),
        "1j": (njets == 1),
        "2j": (njets >= 2),
    }

    events["fakeWeight"] = ak.where(
    masks["0j"],
    fake_weights[""]["3l0j"],
    ak.where(masks["1j"], fake_weights[""]["3l1j"], fake_weights[""]["3l2j"]),
    )

    events["fakeWeight_FW_up"] = ak.where(
        masks["0j"],
        fake_weights["_up"]["3l0j"],
        ak.where(masks["1j"], fake_weights["_up"]["3l1j"], fake_weights["_up"]["3l2j"]),
    )

    events["fakeWeight_FW_down"] = ak.where(
        masks["0j"],
        fake_weights["_down"]["3l0j"],
        ak.where(masks["1j"], fake_weights["_down"]["3l1j"], fake_weights["_down"]["3l2j"]),
    )

    # for suffix, _ in fake_weights.items():
    #     key = "fakeWeight_FW" + ("" if suffix == "" else suffix)
    #     events[key] = ak.where(
    #         masks["0j"],
    #         fake_weights[suffix]["3l0j"],
    #         ak.where(masks["1j"], fake_weights[suffix]["3l1j"], fake_weights[suffix]["3l2j"]),
    #     )

    variations.register_variation(['fakeWeight'],'FW_up')
    variations.register_variation(['fakeWeight'], 'FW_down')

    # print("fakeWeight:", ak.type(events["fakeWeight"]))
    # print("fakeWeight_Up:", ak.type(events["fakeWeight_Up"]))
    # print("fakeWeight_Down:", ak.type(events["fakeWeight_Down"]))
    # print("event length:", len(events))
    # print("Up length:", len(events["fakeWeight_Up"]))

    return events, variations
