
import sys

import awkward as ak
from spritz.framework.framework import correctionlib_wrapper


def fake_evaluate(events, ceval_fake, cfg):
    wrap_ElePR = correctionlib_wrapper(ceval_fake["ElePR"])
    wrap_MuPR = correctionlib_wrapper(ceval_fake["MuPR"])

    lep = ak.copy(
        events.Lepton[
            (abs(events.Lepton.pdgId) == 11) | (abs(events.Lepton.pdgId) == 13)
        ]
    )
    lep = ak.pad_none(lep, 2, axis=1, clip=True)
    lep["pt"] = ak.where(lep.pt >= 50.0, 50.0 - 1e-3, lep.pt)
    lep["pt"] = ak.where(lep.pt <= 10.0, 10.0 + 1e-3, lep.pt)
    lep["eta"] = abs(lep.eta)

    ele_mask = abs(lep.pdgId) == 11
    mu_mask = abs(lep.pdgId) == 13

    eleWP = cfg["leptonsWP"]["eleWP"]
    muWP = cfg["leptonsWP"]["muWP"]
    # FIXME
    lep["isTight"] = ak.ones_like(lep.pt) == 0

    lep["isTight"] = ak.where(ele_mask, lep["isTightElectron_" + eleWP], lep["isTight"])
    lep["isTight"] = ak.where(mu_mask, lep["isTightMuon_" + muWP], lep["isTight"])

    tags = [
        ("2l0j", "EleFR_jet35", "MuFR_jet20"),
        ("2l1j", "EleFR_jet35", "MuFR_jet25"),
        ("2l2j", "EleFR_jet35", "MuFR_jet35"),
    ]

    if len(events) == 0:
        return events

    # print(events, len(events), file=sys.stderr)
    # print(events.Jet.pt.show(), file=sys.stderr)
    # print(events.Jet.pt[events.Jet.pt >= 30].show(), file=sys.stderr)
    njets = ak.num(events.Jet[events.Jet.pt >= 30], axis=1)

    # loop over FRs
    for fakeTag, eleName, muName in tags:
        wrap_EleFR = correctionlib_wrapper(ceval_fake[eleName])
        wrap_MuFR = correctionlib_wrapper(ceval_fake[muName])

        lep["PR"] = ak.ones_like(lep.pt)
        lep["FR"] = ak.zeros_like(lep.pt)

        _lep = ak.mask(lep, ele_mask)
        lep["PR"] = ak.where(
            ele_mask, wrap_ElePR(_lep.pt, _lep.eta), lep["PR"]
        )
        lep["FR"] = ak.where(
            ele_mask, wrap_EleFR( _lep.pt, _lep.eta), lep["FR"]
        )

        _lep = ak.mask(lep, mu_mask)
        lep["PR"] = ak.where(
            mu_mask, wrap_MuPR( _lep.pt, _lep.eta), lep["PR"]
        )
        lep["FR"] = ak.where(
            mu_mask, wrap_MuFR( _lep.pt, _lep.eta), lep["FR"]
        )

        # print("PR", lep.PR, file=sys.stderr)
        # print("FR", lep.FR, file=sys.stderr)

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

        events["PF"] = lep.prompt_prob[:, 0] * lep.fake_prob[:, 1]
        events["FP"] = lep.fake_prob[:, 0] * lep.prompt_prob[:, 1]
        events["FF"] = lep.fake_prob[:, 0] * lep.fake_prob[:, 1]

        # only one tight, flip FF sign
        events["FF"] = ak.where(
            ak.num(lep[lep.isTight]) == 1, events.FF * -1, events.FF
        )

        # not only one tight, flip PF and FP sign
        events["PF"] = ak.where(
            ak.num(lep[lep.isTight]) != 1, events.PF * -1, events.PF
        )
        events["FP"] = ak.where(
            ak.num(lep[lep.isTight]) != 1, events.FP * -1, events.FP
        )

        events[f"fake_weight_{fakeTag}"] = events.PF + events.FP + events.FF
        # print(f"fake_weight_{fakeTag}", file=sys.stderr)

    # print([k for k in ak.fields(events) if "fake_weight" in k], file=sys.stderr)

    masks = {
        "0j": (njets == 0),
        "1j": (njets == 1),
        "2j": (njets >= 2),
    }

    events["fakeWeight"] = ak.where(
        masks["0j"],
        events.fake_weight_2l0j,
        ak.where(masks["1j"], events.fake_weight_2l1j, events.fake_weight_2l2j),
    )

    return events
