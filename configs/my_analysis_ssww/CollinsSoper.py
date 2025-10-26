import numpy as np
import awkward as ak

def build_4vector(obj):
    e_mass = 0.000510998950
    mu_mass = 0.10565837
    mass = ak.where(abs(obj.pdgId) == 11, e_mass, mu_mass)
    px = obj.pt * np.cos(obj.phi)
    py = obj.pt * np.sin(obj.phi)
    pz = obj.pt * np.sinh(obj.eta)
    E = np.sqrt((mass)**2 + px**2 + py**2 + pz**2)
    return ak.zip(
        {
            "E": E,
            "px": px,
            "py": py,
            "pz": pz,
        },
        with_name="Momentum4D",
    )

def cos_theta_collins_soper(lep1_4p, lep2_4p):
    # Somma dei due quadrivettori
    E12  = lep1_4p.E + lep2_4p.E
    px12 = lep1_4p.px + lep2_4p.px
    py12 = lep1_4p.py + lep2_4p.py
    pz12 = lep1_4p.pz + lep2_4p.pz

    # Massa invariante del sistema dileptonico
    M2 = E12**2 - px12**2 - py12**2 - pz12**2
    M = np.sqrt(M2)

    # pT del sistema dileptonico
    pt12 = np.sqrt(px12**2 + py12**2)

    # Combinazioni light-cone dei due leptoni
    p1_plus  = (lep1_4p.E + lep1_4p.pz) / np.sqrt(2)
    p1_minus = (lep1_4p.E - lep1_4p.pz) / np.sqrt(2)

    p2_plus  = (lep2_4p.E + lep2_4p.pz) / np.sqrt(2)
    p2_minus = (lep2_4p.E - lep2_4p.pz) / np.sqrt(2)

    # Segno del boost lungo z
    sign_pz = pz12 / np.abs(pz12)

    # Numeratore e denominatore della formula
    num = 2 * (p1_plus * p2_minus - p1_minus * p2_plus)
    denom = M * np.sqrt(M2 + pt12**2)

    cos_theta = sign_pz * (num / denom)

    return cos_theta