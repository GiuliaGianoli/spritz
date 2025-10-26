import uproot
import correctionlib.schemav2 as cs
import json
import numpy as np
import os

def convert_th2_to_correctionlib(root_path, hist_name, output_name, description="Fake/Prompt Rate"):
    with uproot.open(root_path) as f:
        if hist_name not in f:
            print(f"⚠️  {hist_name} not found in {root_path}")
            return

        h = f[hist_name]
        values = h.values()
        pt_edges = h.axis(0).edges()
        eta_edges = h.axis(1).edges()

    content = cs.MultiBinning(
        nodetype="multibinning",
        inputs=["pt", "eta"],
        edges=[pt_edges.tolist(), eta_edges.tolist()],
        content=values.T.flatten().tolist(),
        flow="clamp",
    )

    corr = cs.Correction(
        version=1,
        name=output_name,
        description=description,
        inputs=[
            {"name": "pt", "type": "real"},
            {"name": "eta", "type": "real"},
        ],
        output={"name": "weight", "type": "real"},
        data=content,
    )

    cset = cs.CorrectionSet(
        schema_version=2,
        corrections=[corr]
    )

    out_json = output_name + ".json"
    with open(out_json, "w") as fout:
        json.dump(cset.model_dump(), fout, indent=2)
    print(f"✅ Saved: {out_json}")


def batch_convert_from_dir(dir_path):
    for filename in os.listdir(dir_path):
        if not filename.endswith(".root"):
            continue
        root_path = os.path.join(dir_path, filename)

        # Determine name prefix
        if "Muon" in filename or "muon" in filename.lower():
            flavor = "Mu"
        elif "Ele" in filename or "ele" in filename.lower():
            flavor = "Ele"
        else:
            continue

        if "PR" in filename:
            name = f"{flavor}PR"
            if flavor == "Mu":
                hist_name = "h_Muon_signal_pt_eta_bin"
            elif flavor == "Ele":
                hist_name = "h_Ele_signal_pt_eta_bin"
            else:
                hist_name = "PR_pT_eta"
        elif "FR" in filename:
            jet = "".join(filter(str.isdigit, filename))
            name = f"{flavor}FR_jet{jet}"
            hist_name = "FR_pT_eta_EWKcorr"
        else:
            continue

        convert_th2_to_correctionlib(
            root_path=root_path,
            hist_name=hist_name,
            output_name=name,
            description=f"{flavor} Fake/Prompt Rate"
        )
#main

muon_dir = "cut_Tight_HWW"
electron_dir = "wp90iso"

# Chiama il convertitore su entrambe le directory
batch_convert_from_dir(muon_dir)
batch_convert_from_dir(electron_dir)
