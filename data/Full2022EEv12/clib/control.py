import correctionlib

ceval = correctionlib.CorrectionSet.from_file("lepton_sf.json")
list(ceval.keys())

for corr in ceval.values():
    print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
    for ix in corr.inputs:
        print(f"   Input {ix.name} ({ix.type}): {ix.description}")