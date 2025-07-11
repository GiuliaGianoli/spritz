#!/usr/bin/env python
# coding: utf-8

import gzip
import json

import correctionlib
import correctionlib.convert
import hist
import numpy as np
import pandas as pd
import requests
import rich
import uproot
from data.common.LeptonSel_cfg import ElectronWP, MuonWP

#path_jsonpog = "/Users/giorgiopizzati/Downloads/jsonpog-integration-master/POG"
path_jsonpog = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG"
url_latinos = "https://raw.githubusercontent.com/latinos/LatinoAnalysis/UL_production/NanoGardener/python/data/scale_factor/"
for ERA, year in [
    ("Full2022EEv12", "2022Re-recoE+PromptFG"),
]:
    print(ElectronWP[ERA]["TightObjWP"]["mvaWinter22V2Iso_WP90"]["tkSF"])
    print(list(ElectronWP[ERA]["TightObjWP"]["mvaWinter22V2Iso_WP90"]["tkSF"].values()))
    file_list = list(ElectronWP[ERA]["TightObjWP"]["mvaWinter22V2Iso_WP90"]["tkSF"].values())[0]

    print("Lista:", file_list)

    # Percorso del file JSON (il terzo elemento della lista)
    json_path = file_list[2]
    print("Path completo nel JSON:", json_path)

    relative_path = json_path.split("POG")[-1]
    print("Path relativo:", relative_path)

    fname = path_jsonpog + relative_path
    print("Path finale del file:", fname)
    # fname = (
    #     path_jsonpog
    #     + list(ElectronWP[ERA]["TightObjWP"]["mvaWinter22V2Iso_WP90"]["tkSF"].values())[
    #         0
    #     ].split("POG")[-1]
    # )
    # print(fname)

    ceval = correctionlib.CorrectionSet.from_file(fname)

    # ## Explore correctionlib json file

    with gzip.open(fname) as file:
        corr = json.load(file)

    def find_key(key: str, l: list):
        for i in range(len(l)):
            if l[i]["key"] == key:
                return i

    def print_keys(l: list):
        for i in range(len(l)):
            print(l[i]["key"])

    corr["corrections"][0]["data"]["content"][0]["value"]["input"]

    real_content = corr["corrections"][0]["data"]["content"][0]["value"]["content"]

    print_keys(real_content)

    find_key("sf", real_content)

    sf = real_content[0]["value"]
    sf["input"], sf.keys()

    print_keys(sf["content"])

    find_key("RecoBelow20", sf["content"])

    def get_cset_electron(era, wp, return_histo=False):
        real_content = corr["corrections"][0]["data"]["content"][0]["value"]["content"]
        content_syst = []
        for valType in ["sf", "sfdown", "sfup"]:
            sf_ind = find_key(valType, real_content)
            sf = real_content[sf_ind]["value"]["content"]
            wp_ind = find_key(wp, sf)
            obj = sf[wp_ind]["value"]
            content = np.array(obj["content"])
            content_syst.append(content)
            axis = [
                hist.axis.Variable(edges, name=name)
                for edges, name in zip(obj["edges"], obj["inputs"])
            ]
        content_syst = np.array(content_syst)
        shape = [ax.edges.shape[0] - 1 for ax in axis]
        content_syst = content_syst.reshape(3, *shape)
        syst = ["nominal", "syst_down", "syst_up"]
        # syst = ['sf', 'sfdown', 'sfup']
        h = hist.Hist(
            hist.axis.StrCategory(syst, name="syst"),
            *axis,
            hist.storage.Double(),
            data=content_syst,
        )
        h.name = "Electron_RecoSF_" + wp
        h.label = "out"
        cset = correctionlib.convert.from_histogram(h)

        if return_histo:
            return h, cset
        return cset

    def rand_in_axis(axis):
        min = axis[0] if axis[0] > -np.inf else axis[1] - (axis[2] - axis[1]) / 2
        max = axis[-1] if axis[-1] < np.inf else axis[-2] + (axis[-2] - axis[-3]) / 2
        return np.random.uniform(min, max)

    def rand_in_histos(h, nrands=10):
        result = []
        for i in range(nrands):
            r = []
            for j in range(len(h.axes)):
                r.append(rand_in_axis(h.axes[j].edges))
            result.append(r)
        return result

    def different_val(a, b):
        precision = 1e-10
        if a == b:
            return False
        if abs(a) < precision and abs(b) < precision:
            return False
        d = abs(a - b)
        if d < precision:
            return False
        return True

    for wp in ["RecoBelow20", "Reco20to75", "RecoAbove75"]:
        valTypes = ["sf", "sfdown", "sfup"]
        systs = ["nominal", "syst_down", "syst_up"]
        h, cset = get_cset_electron(year, wp, return_histo=True)
        ceval_new = cset.to_evaluator()
        l = [h.axes[i].centers for i in range(1, len(h.axes))]
        rand_inputs = np.transpose(
            [np.tile(l[0], len(l[1])), np.repeat(l[1], len(l[0]))]
        )
        rand_inputs[rand_inputs == np.inf] = 100
        rand_inputs[rand_inputs == -np.inf] = -100
        # rand_inputs = rand_in_histos(h, nrands=100)
        for syst, valType in zip(systs, valTypes):
            for eta, pt in rand_inputs:
                # new_val = h[hist.loc(syst), hist.loc(eta), hist.loc(pt)]
                new_val = ceval_new.evaluate(syst, eta, pt)
                print("Chiavi disponibili nel CorrectionSet:")
                print(list(ceval.keys()))
                for var in ceval["Electron-ID-SF"].inputs:
                    if var.name == "year":
                        print(var.type)
                old_val = ceval["Electron-ID-SF"].evaluate(
                    year, valType, wp, eta, pt
                )
                if different_val(new_val, old_val):
                    raise Exception(
                        "Different values for eta", eta, "pt", pt, old_val, new_val
                    )
        print("Everything ok for", wp)

    csets = []
    for wp in ["RecoBelow20", "Reco20to75", "RecoAbove75"]:
        csets.append(get_cset_electron(year, wp))
        rich.print(csets[-1])

    #Qui parte tutta diversa
    electron_cset = correctionlib.CorrectionSet.from_file("electron.json")
    csets.append(electron_cset["Electron-ID-SF"])

    # Muons SF

    # for corrName, histoName in corrections.items():
    #      csets.append(get_muon_cset(corrName, histoName))

    muon_cset1 = correctionlib.CorrectionSet.from_file("muon_scale_Run2022E.json")
    muon_cset2 = correctionlib.CorrectionSet.from_file("muonSF_latinos_HWW.json")

    print("muon_scale_Run2022E keys:", list(muon_cset1.keys()))
    print("muonSF_latinos_HWW keys:", list(muon_cset2.keys()))

    # csets.append(muon_cset1["NUM_TightIDIso_DEN_TightID"]) 
    # csets.append(muon_cset1["NUM_TightIDMiniIso_DEN_TightID"]) 
    # csets.append(muon_cset1["NUM_TightID_DEN_TrackerMuons"])

    # csets.append(muon_cset2["NUM_LoosePFIso_DEN_TightID_HWW"])
    # csets.append(muon_cset2["NUM_TightID_HWW_DEN_TrackerMuons"]) 
    # csets.append(muon_cset2["NUM_TightID_HWW_LooseIso_tthMVA_DEN_LoosePFIso"])
    # csets.append(muon_cset2["NUM_TightID_HWW_TightIso_tthMVA_DEN_TightPFIso"])
    # csets.append(muon_cset2["NUM_TightPFIso_DEN_TightID_HWW"])

    print(type(muon_cset2["NUM_LoosePFIso_DEN_TightID_HWW"]))             


    # Save everything

    # cset = correctionlib.schemav2.CorrectionSet(
    #       schema_version=2, description="", corrections=csets
    #  )

    # rich.print(cset)

    # import os

    # os.makedirs(f"../data/{ERA}/clib", exist_ok=True)
    # with gzip.open(f"../data/{ERA}/clib/lepton_sf.json.gz", "wt") as fout:
    #     fout.write(cset.json(exclude_unset=True))
    # final_cset = correctionlib.schemav2.CorrectionSet(schema_version=2, corrections=csets)

    # with open("lepton_sf.json", "w") as fout:
    #     fout.write(final_cset.model_dump_json(indent=2))
