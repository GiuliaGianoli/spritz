import gc
import json
import sys 
import traceback as tb
from copy import deepcopy

import awkward as ak
import correctionlib
import hist
import numpy as np
import onnxruntime as ort
import spritz.framework.variation as variation_module
import uproot
import vector
from spritz.framework.framework import (
    big_process,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
    write_chunks,
)
from spritz.modules.basic_selections import (
    LumiMask,
    lumi_mask,
    pass_flags,
    pass_trigger,
)
from spritz.modules.btag_sf import btag_sf
from spritz.modules.dnn_evaluator import dnn_evaluator, dnn_transform
from spritz.modules.gen_analysis import gen_analysis
from spritz.modules.jet_sel import cleanJet, jetSel
from spritz.modules.jme import (
    correct_jets_data,
    correct_jets_mc,
    jet_veto,
    remove_jets_HEM_issue,
)
from spritz.modules.lepton_sel import createLepton, leptonSel
from spritz.modules.lepton_sf import lepton_sf
from spritz.modules.prompt_gen import prompt_gen_match_leptons
from spritz.modules.puweight import puweight_sf
from spritz.modules.rochester import correctRochester, getRochester
from spritz.modules.run_assign import assign_run_period
from spritz.modules.theory_unc import theory_unc
from spritz.modules.trigger_sf import trigger_sf
from spritz.modules.fake_evaluate import fake_evaluate
from spritz.modules.fake_evaluate_ssww_syst import fake_evaluate_ssww_syst
vector.register_awkward()

print("uproot version", uproot.__version__)
print("awkward version", ak.__version__)

path_fw = get_fw_path()
with open("/gwpool/users/ggianoli/spritz/data/Full2022EEv12/cfg.json") as file:
    txt = file.read()
    txt = txt.replace("RPLME_PATH_FW", path_fw)
    cfg = json.loads(txt)

ceval_btag = correctionlib.CorrectionSet.from_file(cfg["btagSF"])
ceval_puWeight = correctionlib.CorrectionSet.from_file(cfg["puWeights"])
ceval_lepton_sf = correctionlib.CorrectionSet.from_file(cfg["leptonSF"])
#ceval_assign_run = correctionlib.CorrectionSet.from_file(cfg["run_to_era"])

#Fake rate
import os

analysis_path = sys.argv[1]
analysis_cfg = get_analysis_dict(analysis_path)
#special_analysis_cfg = analysis_cfg["special_analysis_cfg"]
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 1
sess_opt.inter_op_num_threads = 1
#dnn_cfg = special_analysis_cfg["dnn"]
#onnx_session = ort.InferenceSession(dnn_cfg["model"], sess_opt)
#dnn_t = dnn_transform(dnn_cfg["cumulative_signal"])


def ensure_not_none(arr):
    if ak.any(ak.is_none(arr)):
        raise Exception("There are some None in branch", arr[ak.is_none(arr)])
    return ak.fill_none(arr, -9999.9)


def process(events, **kwargs):
    dataset = kwargs["dataset"]
    trigger_sel = kwargs.get("trigger_sel", "")
    isData = kwargs.get("is_data", False)
    era = kwargs.get("era", None)
    isData = kwargs.get("is_data", False)
    do_theory_variations = kwargs.get("do_theory_variations", False)
    subsamples = kwargs.get("subsamples", {})
    special_weight = eval(kwargs.get("weight", "1.0"))

    # variations = {}
    # variations["nom"] = [()]
    variations = variation_module.Variation()
    variations.register_variation([], "nom")

    if isData:
        events["weight"] = ak.ones_like(events.run)
    else:
        events["weight"] = events.genWeight

    if isData:
        lumimask = LumiMask(cfg["lumiMask"])
        events = lumi_mask(events, lumimask)

    sumw = ak.sum(events.weight)
    nevents = ak.num(events.weight, axis=0)

    # # Add special weight for each dataset (not subsamples)
    if special_weight != 1.0:
        print(f"Using special weight for {dataset}: {special_weight}")

    events["weight"] = events.weight * special_weight

    # pass trigger and flags
    #events = assign_run_period(events, isData, cfg, ceval_assign_run)
    events = pass_flags(events, cfg["flags"])
    events = events[events.pass_flags]

    print("[DBG] after flags&trigger:", len(events))

    events = jetSel(events, cfg)

    events = createLepton(events)

    events = leptonSel(events, cfg)
    # Latinos definitions, only consider loose leptons
    # remove events where ptl1 < 8
    events["Lepton"] = events.Lepton[events.Lepton.isLoose]
    # Apply a skim!
    events = events[ak.num(events.Lepton) > 0]
     
    #ci sono i leptoni qui 
    if isData:
        # each data DataSet has its own trigger_sel
        events = events[eval(trigger_sel)]
    
    if not isData:
        events = prompt_gen_match_leptons(events)


    # FIXME should clean from only tight / loose?
    events = cleanJet(events)

    # Require at least one good PV (tolgo)
    events = events[events.PV.npvsGood > 0]

    # # Jet veto maps
    events = jet_veto(events, cfg)

    if not isData:
        # puWeight
        events, variations = puweight_sf(events, variations, ceval_puWeight, cfg)

        # add LeptonSF
        events, variations = lepton_sf(events, variations, ceval_lepton_sf, cfg)

        # FIXME add Electron Scale
        # FIXME add MET corrections?

        # Jets corrections
        #JEC + JER + JES
        events, variations = correct_jets_mc(
            events, variations, cfg, run_variations=False
        )

    else:
         events = correct_jets_data(events, cfg, era)

    originalEvents = ak.copy(events)
    jet_pt_backup = ak.copy(events.Jet.pt)

    regions = deepcopy(analysis_cfg["regions"])
    variables = deepcopy(analysis_cfg["variables"])

    # # FIXME removing all variations
    # variations.variations_dict = {
    #     k: v for k, v in variations.variations_dict.items() if k == "nom"
    # }

    default_axis = [
        hist.axis.StrCategory(
            [region for region in regions],
            name="category",
        ),
        hist.axis.StrCategory(
            sorted(list(variations.get_variations_all())), name="syst"
        ),
    ]

    results = {}
    results = {dataset: {"sumw": sumw, "nevents": nevents, "events": 0, "histos": 0}}
    if subsamples != {}:
        results = {}
        for subsample in subsamples:
            results[f"{dataset}_{subsample}"] = {
                "sumw": sumw,
                "nevents": nevents,
                "events": 0,
                "histos": 0,
            }

    for dataset_name in results:
        _events = {}
        histos = {}
        for variable in variables:
            _events[variable] = ak.Array([])

            # if "axis" in variables[variable]:
            #     if isinstance(variables[variable]["axis"], list):
            #         histos[variable] = hist.Hist(
            #             *variables[variable]["axis"],
            #             *default_axis,
            #             hist.storage.Weight(),
            #         )
            #     else:
            #         histos[variable] = hist.Hist(
            #             variables[variable]["axis"],
            #             *default_axis,
            #             hist.storage.Weight(),
            #         )
            #per variabili 2d
            if "axis" in variables[variable]:
                axes_spec = variables[variable]["axis"]
                if isinstance(axes_spec, (list, tuple)):
                    histos[variable] = hist.Hist(
                        *axes_spec,
                        *default_axis,
                        storage=hist.storage.Weight(),
                    )
                else:
                    histos[variable] = hist.Hist(
                        axes_spec,
                        *default_axis,
                        storage=hist.storage.Weight(),
                    )


        results[dataset_name]["histos"] = histos
        results[dataset_name]["events"] = _events



    # FIXME add FakeW

 
    print("Doing variations")
    # for variation in sorted(list(variations.keys())):
    # for variation in ["nom"]:
    for variation in sorted(variations.get_variations_all()):
        events = ak.copy(originalEvents)
        assert ak.all(events.Jet.pt == jet_pt_backup)

        print(variation)
        for switch in variations.get_variation_subs(variation):
            if len(switch) == 2:
                # print(switch)
                variation_dest, variation_source = switch
                events[variation_dest] = events[variation_source]

        # resort Leptons
        lepton_sort = ak.argsort(events[("Lepton", "pt")], ascending=False, axis=1)
        events["Lepton"] = events.Lepton[lepton_sort]

 
        if len(events) == 0:
            continue

        # Jet real selections

        # resort Jets
        jet_sort = ak.argsort(events[("Jet", "pt")], ascending=False, axis=1)
        events["Jet"] = events.Jet[jet_sort]

        #per i fake 
        if not isData:
            events["prompt_gen_match_1l"] = (
                events.Lepton[:, 0].promptgenmatched
            )
            events = events[events.prompt_gen_match_1l]
            print("prompt gen:")
            print(events.Lepton.promptgenmatched)

        #per i prompt
        # if not isData:
        #     events["prompt_gen_match_2l"] = (
        #         events.Lepton[:, 0].promptgenmatched
        #         & events.Lepton[:, 1].promptgenmatched
        #     )
        #     events = events[events.prompt_gen_match_2l]
        #     print("prompt gen:")
        #     print(events.Lepton.promptgenmatched)

        if len(events) == 0:
            continue

        if not isData:
            events["RecoSF"] = ak.fill_none(ak.pad_none(events.Lepton, 1)[:, 0].RecoSF, 1) * ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 1].RecoSF, 1) 
            events["TightSF"] = (
            ak.fill_none(ak.pad_none(events.Lepton, 1)[:, 0].TightSF, 1) * ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 1].TightSF, 1) 
            )

            events["weight"] = (
                events.weight
                * events.puWeight
                * events.RecoSF
                * events.TightSF
            )
        
        #Variable definitions
        events["jets"] = ak.pad_none(events.Jet, 0)
        for variable in variables:
            if "func" in variables[variable]:
                events[variable] = variables[variable]["func"](events)


        events[f"mask_{dataset}"] = ak.ones_like(events.run) == 1.0
        events[f"weight_{dataset}"] = events.weight

        if subsamples != {}:
            for subsample in subsamples:
                subsample_val = subsamples[subsample]

                if isinstance(subsample_val, str):
                    subsample_mask = eval(subsample_val)
                    subsample_weight = 1.0
                elif (
                    isinstance(subsample_val, tuple) or isinstance(subsample_val, list)
                ) and len(subsample_val) == 2:
                    subsample_mask = eval(subsample_val[0])
                    subsample_weight = eval(subsample_val[1])
                else:
                    raise Exception(
                        "subsample value can either be a str (mask) or tuple/list of "
                        "len 2 (mask, weight)"
                    )

                events[f"mask_{dataset}_{subsample}"] = subsample_mask
                events[f"weight_{dataset}_{subsample}"] = (
                    events.weight * subsample_weight
                )
        #per definire tight e loose
        eleWP = cfg["leptonsWP"]["eleWP"]
        muWP  = cfg["leptonsWP"]["muWP"]

        def LepWPCut1l_mask(events):
            #l0   = _lep(events, 0)
            is_e = ak.fill_none(abs(ak.fill_none(ak.pad_none(events.Lepton, 1)[:, 0].pdgId, -9999)) == 11,False)
            is_m = ak.fill_none(abs(ak.fill_none(ak.pad_none(events.Lepton, 1)[:, 0].pdgId, -9999)) == 13,False)
            istight_e = ak.fill_none((ak.pad_none(events.Lepton, 1)[:, 0]["isTightElectron_" + eleWP]), False)
            istight_m = ak.fill_none((ak.pad_none(events.Lepton, 1)[:, 0]["isTightMuon_" + muWP]), False)
            #is_e = ak.fill_none(abs(l0.pdgId) == 11, False)
            #is_m = ak.fill_none(abs(l0.pdgId) == 13, False)
            return (is_e & istight_e) | (is_m & istight_m)

        def LepWPCut2l_mask(events):
            #l0   = _lep(events, 0)
            #l1   = _lep(events, 1)
            # is_e0 = ak.fill_none(abs(l0.pdgId) == 11, False)
            # is_m0 = ak.fill_none(abs(l0.pdgId) == 13, False)
            # is_e1 = ak.fill_none(abs(l1.pdgId) == 11, False)
            # is_m1 = ak.fill_none(abs(l1.pdgId) == 13, False)
            is_e0 = ak.fill_none(abs(ak.fill_none(ak.pad_none(events.Lepton, 1)[:, 0].pdgId, -9999)) == 11,False)
            is_m0 = ak.fill_none(abs(ak.fill_none(ak.pad_none(events.Lepton, 1)[:, 0].pdgId, -9999)) == 13,False)
            is_e1 = ak.fill_none(abs(ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 1].pdgId, -9999)) == 11,False)
            is_m1 = ak.fill_none(abs(ak.fill_none(ak.pad_none(events.Lepton, 2)[:, 1].pdgId, -9999)) == 13,False)
            istight_e0 = ak.fill_none((ak.pad_none(events.Lepton, 1)[:, 0]["isTightElectron_" + eleWP]), False)
            istight_m0 = ak.fill_none((ak.pad_none(events.Lepton, 1)[:, 0]["isTightMuon_" + muWP]), False)
            istight_e1 = ak.fill_none((ak.pad_none(events.Lepton, 2)[:, 1]["isTightElectron_" + eleWP]), False)
            istight_m1 = ak.fill_none((ak.pad_none(events.Lepton, 2)[:, 1]["isTightMuon_" + muWP]), False)
            pass0 = (is_e0 & istight_e0) | (is_m0 & istight_m0)
            pass1 = (is_e1 & istight_e1) | (is_m1 & istight_m1)
            return ak.fill_none(pass0, False) & ak.fill_none(pass1, False)
        
        print("[DBG] before loops region:", len(events))
        for region in regions:
                base_mask = regions[region]["func"](events)

                # WP richiesti dalle regioni 
                # - QCD_loose_*           : nessun WP
                # - QCD_tight_*           : LepWPCut1l (1 lepton tight)
                # - Zpeak_PR_loose_*      : LepWPCut1l (leading tight)
                # - Zpeak_PR_tight_*      : LepWPCut2l (entrambi tight)

                if ("QCD" in region) and ("tight" in region):
                    wp_mask = (ak.num(events.Lepton, axis=1) == 1) & LepWPCut1l_mask(events)

                elif ("Zpeak" in region) and ("PR" in region) and ("tight" in region):
                    wp_mask = LepWPCut2l_mask(events)

                elif ("Zpeak" in region) and ("PR" in region) and ("loose" in region):
                    wp_mask = LepWPCut1l_mask(events)

                else:
                    wp_mask = ak.ones_like(events.run, dtype=bool)

                regions[region]["mask"] = base_mask & wp_mask

        print("[DBG] after loops region:", len(events))

        # Fill histograms
        for dataset_name in results:
            # if dataset_name == "Fake":
            #     continue
            for region in regions:
                # for category in categories:
                # Apply mask for specific region, category and dataset_name
                mask = regions[region]["mask"] & events[f"mask_{dataset_name}"]
                # region_events = region_events_map[region]
                # mask = regions[region]["mask"] & region_events[f"mask_{dataset_name}"]
                #print(ak.num(select_wz_region(events)[0]))
                if len(events[mask]) == 0:
                    continue

                # for variable in results[dataset_name]["histos"]:
                #     if isinstance(variables[variable]["axis"], list):
                #         var_names = [k.name for k in variables[variable]["axis"]]
                #         vals = {
                #             var_name: events[var_name][mask] for var_name in var_names
                #         }

                #         try:
                #             results[dataset_name]["histos"][variable].fill(
                #             **vals,
                #             category=region,
                #             syst=variation,
                #             weight=events[f"weight_{dataset_name}"][mask],
                #             )
                #         except Exception as e:
                #             print(f"[ERROR] Multi-variable '{variable}' caused an error in dataset '{dataset_name}', region '{region}'")
                #             print(f"  Variables used: {list(vals.keys())}")
                #             for k in vals:
                #                 print(f"  {k} → {vals[k]}")
                #             print(f"  Exception: {e}")
                #             raise



            
                #     else:
                #         var_name = variables[variable]["axis"].name
                #         try:
                #             results[dataset_name]["histos"][variable].fill(
                #             events[var_name][mask],
                #             category=region,
                #             syst=variation,
                #             weight=events[f"weight_{dataset_name}"][mask],
                #             )
                #         except Exception as e:
                #             print(f"[ERROR] Variable '{variable}' caused an error in dataset '{dataset_name}', region '{region}'")
                #             print(f"  Variable axis name: {var_name}")
                #             print(f"  Events[var_name][mask]: {events[var_name][mask]}")
                #             print(f"  Exception: {e}")
                #             raise
                #per variabili 2d
                for variable in results[dataset_name]["histos"]:
                        axes_spec = variables[variable]["axis"]
                        if isinstance(axes_spec, (list, tuple)):
                            var_names = [ax.name for ax in axes_spec]
                            vals = {vn: events[vn][mask] for vn in var_names}
                            try:
                                results[dataset_name]["histos"][variable].fill(
                                    **vals,
                                    category=region,
                                    syst=variation,
                                    weight=events[f"weight_{dataset_name}"][mask],
                                )
                            except Exception as e:
                                print(f"[ERROR] Multi-variable '{variable}' caused an error in dataset '{dataset_name}', region '{region}'")
                                print(f"  Variables used: {list(vals.keys())}")
                                for k in vals:
                                    print(f"  {k} → {vals[k]}")
                                print(f"  Exception: {e}")
                                raise
                        else:
                            var_name = axes_spec.name
                            try:
                                results[dataset_name]["histos"][variable].fill(
                                    events[var_name][mask],
                                    category=region,
                                    syst=variation,
                                    weight=events[f"weight_{dataset_name}"][mask],
                                )
                            except Exception as e:
                                print(f"[ERROR] Variable '{variable}' caused an error in dataset '{dataset_name}', region '{region}'")
                                print(f"  Variable axis name: {var_name}")
                                print(f"  Events[var_name][mask]: {events[var_name][mask]}")
                                print(f"  Exception: {e}")
                                raise


                # # Snapshot
                # # print("Saving", len(events[mask]), "events for dataset", dataset_name)
                # for variable in results[dataset_name]["events"]:
                #     branch = ensure_not_none(events[variable][mask])
                #     # print(variable)
                #     # assert ak.any(ak.is_none(branch))
                #     # assert ak.all(ak.num(branch) == 1)
                #     results[dataset_name]["events"][variable] = ak.concatenate(
                #         [
                #             results[dataset_name]["events"][variable],
                #             ak.copy(branch),
                #         ]
                #     )
                #     # print("is ak", isinstance(branch, ak.highlevel.Array))
                #     # print("is np", isinstance(branch, np.ndarray))

    gc.collect()
    return results


if __name__ == "__main__":
    print("Fake")
    chunks_readable = False
    new_chunks = read_chunks("chunks_job.pkl", readable=chunks_readable)
    print("N chunks to process", len(new_chunks))

    results = {}
    errors = []
    processed = []

    for i in range(len(new_chunks)):
        new_chunk = new_chunks[i]

        if new_chunk["result"] != {}:
            print(
                "Skip chunk",
                {k: v for k, v in new_chunk["data"].items() if k != "read_form"},
                "was already processed",
            )
            continue

        print(new_chunk["data"]["dataset"])

        # # # FIXME run only on Zjj and DY
        # if new_chunk["data"]["dataset"] not in ["Zjj"]:
        #     continue

        # # FIXME run only on data
        # if not new_chunk["data"].get("is_data", False):
        #     continue

        # # FIXME process only one chunk per dataset
        # if new_chunk["data"]["dataset"] in processed:
        #     continue
        # processed.append(new_chunk["data"]["dataset"])

        try:
            new_chunks[i]["result"] = big_process(process=process, **new_chunk["data"])
            new_chunks[i]["error"] = ""
        except Exception as e:
            print("\n\nError for chunk", new_chunk, file=sys.stderr)
            nice_exception = "".join(tb.format_exception(None, e, e.__traceback__))
            print(nice_exception, file=sys.stderr)
            new_chunks[i]["result"] = {}
            new_chunks[i]["error"] = nice_exception

        print(f"Done {i+1}/{len(new_chunks)}")

        # # FIXME run only on first chunk
        # if i >= 1:
        #     break

    # file = uproot.recreate("results.root")
    datasets = list(filter(lambda k: "root:/" not in k, results.keys()))
    # for dataset in datasets:
    #     print("Done", results[dataset]["nevents"], "events for dataset", dataset)
    #     file[dataset] = results[dataset]["events"]
    # file.close()

    # clean the events dictionary (too heavy and already saved in the root file)
    # for dataset in datasets:
    #     results[dataset]["events"] = {}

    write_chunks(new_chunks, "results.pkl", readable=chunks_readable)
