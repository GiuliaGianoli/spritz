import awkward as ak
from spritz.framework.framework import correctionlib_wrapper
from spritz.framework.variation import Variation

def fake_evaluate_ssww_syst(events, variations: Variation, ceval_fake, cfg):
    wrap_ElePR = correctionlib_wrapper(ceval_fake["ElePR"])
    wrap_MuPR = correctionlib_wrapper(ceval_fake["MuPR"])

    # per SSWW  2 leptoni
    lep = ak.copy(events.Lepton[(abs(events.Lepton.pdgId) == 11) | (abs(events.Lepton.pdgId) == 13)])
    lep = ak.pad_none(lep, 2, axis=1, clip=True)

    lep["pt"]  = ak.where(lep.pt >= 50.0, 49.999, lep.pt)
    lep["pt"]  = ak.where(lep.pt <= 10.0, 10.001, lep.pt)
    lep["eta"] = abs(lep.eta)

    eleWP = cfg["leptonsWP"]["eleWP"]
    muWP  = cfg["leptonsWP"]["muWP"]

    lep["isTight"] = ak.zeros_like(lep.pt, dtype=bool)
    lep["isTight"] = ak.where(abs(lep.pdgId) == 11, lep["isTightElectron_" + eleWP], lep["isTight"])
    lep["isTight"] = ak.where(abs(lep.pdgId) == 13, lep["isTightMuon_" + muWP],     lep["isTight"])

    # lep["isTight"] = ak.where(abs(lep.pdgId) == 11, lep[eleWP], lep["isTight"])
    # lep["isTight"] = ak.where(abs(lep.pdgId) == 13, lep[muWP],     lep["isTight"])

    # Mapping: tag  (ele_nom, mu_nom, ele_up, mu_up, ele_down, mu_down)
    tags = {
        "3l0j": ("EleFR_jet35","MuFR_jet35","EleFR_jet40","MuFR_jet40","EleFR_jet30","MuFR_jet35"),
        "3l1j": ("EleFR_jet35","MuFR_jet35","EleFR_jet40","MuFR_jet40","EleFR_jet30","MuFR_jet30"),
        "3l2j": ("EleFR_jet35","MuFR_jet35","EleFR_jet40","MuFR_jet40","EleFR_jet30","MuFR_jet30"),
    }

    if len(events) == 0:
        return events, variations

    njets = ak.num(events.Jet[events.Jet.pt >= 30], axis=1)

    fake_weights = {"": {}, "_up": {}, "_down": {}}
    for var_suffix in ["", "_up", "_down"]:
        for fakeTag, (ele_nom, mu_nom, ele_up, mu_up, ele_down, mu_down) in tags.items():
            if var_suffix == "":
                eleFR, muFR = ele_nom, mu_nom
            elif var_suffix == "_up":
                eleFR, muFR = ele_up, mu_up
            else:
                eleFR, muFR = ele_down, mu_down

            
            lep_tag = ak.copy(lep)

           
            ele_mask_tag = abs(lep_tag.pdgId) == 11
            mu_mask_tag  = abs(lep_tag.pdgId) == 13

            wrap_EleFR = correctionlib_wrapper(ceval_fake[eleFR])
            wrap_MuFR  = correctionlib_wrapper(ceval_fake[muFR])

            lep_tag["PR"] = ak.ones_like(lep_tag.pt)
            lep_tag["FR"] = ak.zeros_like(lep_tag.pt)

            _lep_ele = ak.mask(lep_tag, ele_mask_tag)
            lep_tag["PR"] = ak.where(ele_mask_tag, wrap_ElePR(_lep_ele.pt, _lep_ele.eta), lep_tag["PR"])
            lep_tag["FR"] = ak.where(ele_mask_tag, wrap_EleFR(_lep_ele.pt, _lep_ele.eta), lep_tag["FR"])

            _lep_mu = ak.mask(lep_tag, mu_mask_tag)
            lep_tag["PR"] = ak.where(mu_mask_tag, wrap_MuPR(_lep_mu.pt, _lep_mu.eta), lep_tag["PR"])
            lep_tag["FR"] = ak.where(mu_mask_tag, wrap_MuFR(_lep_mu.pt, _lep_mu.eta), lep_tag["FR"])

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

            # 2 leptoni
            terms = {
                "PF": P[:, 0] * F[:, 1],
                "FP": F[:, 0] * P[:, 1],
                "FF": F[:, 0] * F[:, 1],
            }

            n_tight = ak.num(lep_tag[lep_tag.isTight], axis=1)

            # segni: PF/FP negativi se n_tight != 1; FF negativo se n_tight == 1
            for key in ["PF", "FP"]:
                terms[key] = ak.where(n_tight != 1, -terms[key], terms[key])
            terms["FF"] = ak.where(n_tight == 1, -terms["FF"], terms["FF"])

            fake_weight = terms["PF"] + terms["FP"] + terms["FF"]
            fake_weights[var_suffix][fakeTag] = fake_weight

    # selezione per njets
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

    variations.register_variation(['fakeWeight'], 'FW_up')
    variations.register_variation(['fakeWeight'], 'FW_down')

    return events, variations
