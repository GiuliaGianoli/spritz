import concurrent.futures
import json
import os
import subprocess
import sys
from copy import deepcopy
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import matplotlib as mpl
import mplhep as hep
import numpy as np
import uproot
from spritz.framework.framework import get_analysis_dict, get_fw_path

mpl.use("Agg")
from matplotlib import pyplot as plt

d = deepcopy(hep.style.CMS)

d["font.size"] = 12
d["figure.figsize"] = (5, 5)

plt.style.use(d)

# --- plotting filters ---
PREFER_DATA = "Data"
EXCLUDE_SAMPLES = {"Data_unprescaled", "DY_unprescaled"}


def darker_color(color):
    rgb = list(mpl.colors.to_rgba(color)[:-1])
    darker_factor = 4 / 5
    rgb[0] = rgb[0] * darker_factor
    rgb[1] = rgb[1] * darker_factor
    rgb[2] = rgb[2] * darker_factor
    return tuple(rgb)

xlabels = {
    'mjj': r"$m_{jj}$ [GeV]",
    'mll': r"$m_{\ell\ell}$ [GeV]",
    'mtWW': r"$m_T^{WW}$ [GeV]",
    'ptl1Z': r"$p_T(\ell_{1Z})$ [GeV]",
    'ptl2Z': r"$p_T(\ell_{2Z})$ [GeV]",
    #'ptll': r"$p_T(\ell\ell Z)$ [GeV]",
    'ptll': r"$p_T(\ell\ell)$ [GeV]",
    'ptj1': r"$p_T(j_1)$ [GeV]",
    'ptj2': r"$p_T(j_2)$ [GeV]",
    'detajj': r"$|\Delta\eta_{jj}|$",
    'dphill': r"$\Delta\phi_{\ell\ell}$ [rad]",
    'dphijj': r"$\Delta\phi_{jj}$ [rad]",
    'pt_balance': r"$R_{p_{T}}$",
    'MET': r"$MET$ [GeV]",
    'cos_theta_CS': r"$\cos(\theta_{CS})$",
    'Ht': r"$H_T$",
    'etaj1': r"$\eta(j_{1})$",
    'etaj2': r"$\eta(j_{2})$",
    'etal1Z': r"$\eta(\ell_{1Z})$",
    'etal2Z': r"$\eta(\ell_{2Z})$",
    'etal3W': r"$\eta(\ell_{W})$",
    'ptl3W': r"$p_T(\ell_{W})$ [GeV]",
    'mtW': r"$m_T^{W}$ [GeV]",
    'mtWZ': r"$m_T^{WZ}$ [GeV]",
    'mjj2': r"$m_{jj}$ [GeV]",
    'etal1': r"$\eta(\ell_{1})$",
    'etal2': r"$\eta(\ell_{2})$",
    'etal3': r"$\eta(\ell_{3})$",
    'ptl1': r"$p_T(\ell_{1})$ [GeV]",
    'ptl2': r"$p_T(\ell_{2})$ [GeV]",
    'ptl3': r"$p_T(\ell_{3})$ [GeV]",
    'tthMVA_l1': r"$tthMVA(\ell_{1})$",
    'tthMVA_l2': r"$tthMVA(\ell_{2})$",
    'tthMVA_l3': r"$tthMVA(\ell_{3})$",
    'mjj_vs_mll': r"$m_{jj}$ [GeV] x $m_{\ell\ell}$ [GeV]",
    'mjj_vs_dphill': r"$m_{jj}$ [GeV] x $\Delta\phi_{\ell\ell}$ [rad]",
    'mjj_vs_ptl1': r"$m_{jj}$ [GeV] x $p_T(\ell_{1})$ [GeV]",
    'detajj_vs_ptl1': r"$|\Delta\eta_{jj}|$ x $p_T(\ell_{1})$ [GeV]",
    'detajj_vs_dphill': r"$|\Delta\eta_{jj}|$ x $\Delta\phi_{\ell\ell}$ [rad]",
    'mjj_vs_Rpt': r"$m_{jj}$ [GeV] x $R_{p_{T}}$",
    }

#xlabels.update({f"{k}_bin": v for k, v in xlabels.items()})
xlabels.update({f"{k}_inverse": v for k, v in xlabels.items()})
xlabels.update({f"{k}_low": v for k, v in xlabels.items()})
xlabels.update({f"{k}_b": v for k, v in xlabels.items()})

legend_labels = {
        'SSWW_LL': r'SSWW LL (Lab Frame)',
        'SSWW_TL': r'SSWW TL (Lab Frame)',
        'SSWW_TT': r'SSWW TT (Lab Frame)',
        'TTBAR': r'$t\bar{t}$',
        'WZ_QCD': r'WZ QCD',
        'WZ_EWK': r'WZ EWK',
        'SSWW_QCD': 'SSWW QCD',
        'SSWW_EWK': 'SSWW EWK',
        'W_JETS': 'W + jets',
        'SSWW_LL_pos': r'SSW+W+ LL (Lab Frame)',
        'SSWW_LL_neg': r'SSW-W- LL (Lab Frame)',
        'SSWW_TL_pos': r'SSW+W+ TL (Lab Frame)',
        'SSWW_TL_neg': r'SSW-W- TL (Lab Frame)',
        'SSWW_TT_pos': r'SSW+W+ TT (Lab Frame)',
        'SSWW_TT_neg': r'SSW-W- TT (Lab Frame)',
        'WZ_QCD_pos': r'W+Z QCD',
        'WZ_QCD_neg': r'W-Z QCD',
        'WZ_EWK_pos': r'W+Z EWK',
        'WZ_EWK_neg': r'W-Z EWK',
        'SSWW_QCD_pos': r'W+W+ QCD',
        'SSWW_QCD_neg': r'W-W- QCD',
        'Fake': r'Fake'
        }

regions_label = {
        'VBS_SSWW': r'SSWW',
        'InverseMET': r'Inverse MET',
        'VBS_SSWW_loose': r'SSWW loose',
        'VBS_SSWW_(ee)': r'SSWW (ee)',
        'VBS_SSWW_(mumu)': r'SSWW ($\mu\mu$)',
        'VBS_SSWW_(emu)': r'SSWW ($e\mu$)',
        'VBS_SSWW_(mue)': 'SSWW ($\mu e$) ',
        'SSWWb': 'SSWWb',
        'Low_mjj': 'Low mjj',
        'Lowlow_mjj': r'Very low mjj',
        'LowHigh_mjj': r'350 GeV < mjj <500 GeV',
        'preselections_e': 'preselections ($\ell_{1} =e$)',
        'preselections_mu': 'preselections ($\ell_{1} =\mu$)',
        }

UNROLL_LABELLED = {"mjj_vs_mll", "mjj_vs_dphill", "mjj_vs_ptl1", "detajj_vs_ptl1", "detajj_vs_dphill", "mjj_vs_Rpt"}
VARIABLES_CFG = {}  

def per_bin_labels(xedges, yedges):
    nx = len(xedges) - 1
    ny = len(yedges) - 1
    labels = []
    for j in range(ny):         
        for i in range(nx):     
            labels.append(f"[{xedges[i]:g},{xedges[i+1]:g})×[{yedges[j]:g},{yedges[j+1]:g})")
    return labels


def plot(
    input_file,
    region,
    variable,
    samples,
    nuisances,
    lumi,
    colors,
    year_label,
    blind,
    variable_label=None,
    fit_scales={},
    is_log=True,
):
    print("Doing ", region, variable)

    histos = {}
    directory = input_file[f"{region}/{variable}"]
    dummy_histo = 0
    axis = 0
    hmin = 1e7
    plot_nuisances = ["stat"]
    for sample in samples:
        if sample in EXCLUDE_SAMPLES:
            continue
        h = directory[f"histo_{sample}"].to_hist()
        if isinstance(axis, int):
            axis = h.axes[0]
        if isinstance(dummy_histo, int):
            dummy_histo = np.zeros_like(h.values())
        histos[sample] = {}
        histo = histos[sample]
        fit_scale = fit_scales.get(sample, 1.0)
        histo["nom"] = h.values() * fit_scale
        hmin = min(hmin, np.min(histo["nom"]))
        stat = np.sqrt(h.variances()) * fit_scale
        histo["stat_up"] = h.values() + stat
        histo["stat_down"] = h.values() - stat
        for nuisance in nuisances:
            if nuisances[nuisance]["type"] in ["removeStat", "rateParam", "auto", "stat"]:
                continue
            incuts = nuisances[nuisance].get("cuts", None)
            excuts = nuisances[nuisance].get("exclude_cuts", None)
            if incuts is not None and (region not in set(incuts)):
                continue
            if excuts is not None and (region in set(excuts)):
                continue
            name = nuisances[nuisance]["name"]
            plot_nuisances.append(nuisance)

            if sample not in nuisances[nuisance]["samples"]:
                histo[f"{nuisance}_up"] = h.values().copy()
                histo[f"{nuisance}_down"] = h.values().copy()
                continue

            if nuisances[nuisance]["type"] == "lnN":
                scaling = float(nuisances[nuisance]["samples"][sample])
                histo[f"{nuisance}_up"] = scaling * h.values()
                histo[f"{nuisance}_down"] = 1.0 / scaling * h.values()
            else:
                histo[f"{nuisance}_up"] = (
                    directory[f"histo_{sample}_{name}Up"].values().copy() * fit_scale
                )
                histo[f"{nuisance}_down"] = (
                    directory[f"histo_{sample}_{name}Down"].values().copy() * fit_scale
                )
    plot_nuisances = list(set(plot_nuisances))

    hlast = dummy_histo.copy()
    v_syst_tot = {
        syst: {
            "up": dummy_histo.copy(),
            "down": dummy_histo.copy(),
        }
        for syst in plot_nuisances
    }

    for histoName in histos:
        if not samples[histoName].get("plot_stacked", True):
            continue
        if samples[histoName].get("is_data", False) and not samples[histoName].get("is_fake", False):
            continue

        hlast += histos[histoName]["nom"].copy()

        for vname in plot_nuisances:
            if histoName not in nuisances[vname]["samples"]:
                v_syst_tot[vname]["up"] += histos[histoName]["nom"].copy()
                v_syst_tot[vname]["down"] += histos[histoName]["nom"].copy()
            else:
                v_syst_tot[vname]["up"] += histos[histoName][f"{vname}_up"].copy()
                v_syst_tot[vname]["down"] += histos[histoName][f"{vname}_down"].copy()

    vvar_up = dummy_histo.copy()
    vvar_do = dummy_histo.copy()
    for syst in v_syst_tot:
        vvar_up += np.square(v_syst_tot[syst]["up"] - hlast)
        vvar_do += np.square(v_syst_tot[syst]["down"] - hlast)
    vvar_up = np.sqrt(vvar_up)
    vvar_do = np.sqrt(vvar_do)

    hlast = np.where(hlast >= 1e-6, hlast, 1e-6)

    signal_tot = 0
    bkg_tot = 0
    for name in histos:
        if samples[name].get("is_signal", False):
            signal_tot += np.sum(histos[name]["nom"])
        elif samples[name].get("plot_stacked", True):
            bkg_tot += np.sum(histos[name]["nom"])
    #significance = (hlast - bkg_tot) / bkg_tot
    #blind_mask = significance > 0.0000001
    #blind_mask = significance > 1000000000
    blind_mask = 1 < 1000000000

    data_key = ""
    for name in histos:
        if samples[name].get("is_data", False) and not samples[name].get("is_fake", False):
            data_key = name
            break

    if data_key == "":
        ydata = hlast.copy()
        ydata_up = ydata + np.sqrt(ydata)
        ydata_down = ydata - np.sqrt(ydata)
    else:
        ydata = histos[data_key]["nom"].copy()
        ydata_up = histos[data_key]["stat_up"].copy()
        ydata_down = histos[data_key]["stat_down"].copy()

    if blind:
        ydata = np.where(blind_mask, 0, ydata)
        ydata_up = np.where(blind_mask, 0, ydata_up)
        ydata_down = np.where(blind_mask, 0, ydata_down)

    x = axis.centers
    edges = axis.edges

    tmp_sum = dummy_histo.copy()
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=200)
    fig.tight_layout(pad=-0.5)
    region_text = regions_label.get(region, region)  
    hep.cms.label(region_text, data=True, lumi=round(lumi, 2), com=13.6, ax=ax[0], year=year_label)


    priority = ["SSWW_LL_pos", "SSWW_LL_neg", "SSWW_TL_pos", "SSWW_TL_neg", "SSWW_TT_pos", "SSWW_TT_neg"]
    ordered_names = [n for n in reversed(priority) if n in histos]
    for name in histos:
        if name not in ordered_names and samples[name].get("plot_stacked", True):
            ordered_names.insert(0, name)

    for i, histoName in enumerate(ordered_names):
        is_signal = samples[histoName].get("is_signal", False)
        plot_stacked = samples[histoName].get("plot_stacked", True)
        y = histos[histoName]["nom"].copy()
        integral = round(np.sum(y), 2)
        label = legend_labels.get(histoName, histoName) + f" [{integral}]"

        if samples[histoName].get("is_data", False) and not samples[histoName].get("is_fake", False):
            yup = histos[histoName]["stat_up"] - y
            ydown = y - histos[histoName]["stat_down"]
            if blind:
                y[:] = 0
                yup[:] = 0
                ydown[:] = 0
            ax[0].errorbar(
                x,
                #ydata,
                #yerr=(np.sqrt(ydata), np.sqrt(ydata)),
                y,
                yerr=(ydown, yup),
                #yerr=(ydown, yup),
                fmt="ko",
                markersize=4,
                #label="Data" + f" [{integral}]",
                zorder=len(histos),
            )
            continue

        color = colors[histoName]
        if histoName in ["SSWW_LL_pos", "SSWW_LL_neg"]:
            _label = label + " ×10"
            if histoName == "SSWW_LL_pos":
                line_color = "black"       # nero
            elif histoName == "SSWW_LL_neg":
                line_color = "navy"        # blu scuro
            ax[0].stairs(y * 10, edges, zorder=10, linewidth=2, color=line_color, label=_label)
            if not plot_stacked:
                continue

        if isinstance(tmp_sum, int):
            tmp_sum = y.copy()
        else:
            tmp_sum += y
        #if not samples[histoName].get("is_fake", False):
        ax[0].stairs(
            tmp_sum,
            edges,
            label=label,
            fill=True,
            zorder=-i,
            color=color,
            edgecolor=darker_color(color),
            linewidth=1.0,
        )

    unc_up = round(np.sum(vvar_up) / np.sum(hlast) * 100, 2)
    unc_down = round(np.sum(vvar_do) / np.sum(hlast) * 100, 2)
    unc_dict = dict(fill=True, hatch="///", color="darkgrey", facecolor="none", zorder=9)
    ax[0].stairs(
        hlast + vvar_up,
        edges,
        baseline=hlast - vvar_do,
        label=f"Syst [-{unc_down}, +{unc_up}]%",
        **unc_dict,
    )

    integral = round(np.sum(hlast), 2)
    ax[0].stairs(
        hlast, edges, label=f"Tot MC [{integral}]", color="darkgrey", linewidth=1
    )
    ax[0].legend(loc="upper center", frameon=True, ncols=3, framealpha=0.8, fontsize=6.8)
    #ax[0].legend(loc="upper center", frameon=True, ncols=3, framealpha=0.8, fontsize=9.3)

    if is_log:
        ax[0].set_yscale("log")
        ax[0].set_ylim(max(0.5, hmin), np.max(hlast) * 5e3)
    else:
        ax[0].set_ylim(None, np.max(hlast) + (np.max(hlast) - hmin))

    ratio_err_up = vvar_up / hlast
    ratio_err_down = vvar_do / hlast
    ax[1].stairs(
        1 + ratio_err_up,
        edges,
        baseline=1 - ratio_err_down,
        fill=True,
        color="lightgray",
    )

    # ratio = ydata / hlast
    # ratio_data_up = abs(ydata_up / hlast - ratio)
    # ratio_data_down = abs(ydata_down / hlast - ratio)

    ratio = ydata/ hlast
    # ratio_data_up = abs(yup / hlast - ratio)
    # ratio_data_down = abs(ydown / hlast - ratio)
    # ratio_data_up = ydata_up / hlast
    # ratio_data_down = ydata_down / hlast

    ratio_up_abs   = np.divide(ydata_up,   hlast, out=np.zeros_like(hlast), where=hlast > 0)
    ratio_down_abs = np.divide(ydata_down, hlast, out=np.zeros_like(hlast), where=hlast > 0)
    if blind: 
        ratio[:] = 0
        ratio_up_abs[:] = 0
        ratio_down_abs[:] = 0

    ratio_data_up = np.clip(ratio_up_abs   - ratio, 0, None)
    ratio_data_down = np.clip(ratio - ratio_down_abs, 0, None)


    ax[1].errorbar(x, ratio, (ratio_data_down, ratio_data_up), fmt="ko", markersize=4)
    ax[1].plot(edges, np.ones_like(edges), color="black", linestyle="dashed")

    valid_ratios = ratio[~np.isnan(ratio) & ~np.isinf(ratio)]
    # if valid_ratios.size > 0:
    #     ymin = np.min(valid_ratios)
    #     ymax = np.max(valid_ratios)
    #     margin = 0.1
    #     yrange = ymax - ymin if ymax - ymin > 0 else 1e-3
    #     ymin -= yrange * margin
    #     ymax += yrange * margin
    #     ax[1].set_ylim(ymin, ymax)
    # else:
    #     ax[1].set_ylim(0.0, 2.0)

    ax[1].set_ylim(0.0, 2.0)

    ax[1].set_xlim(np.min(edges), np.max(edges))
    # ax[0].set_ylabel("Events")
    # ax[1].set_ylabel("DATA / MC")
    # ax[1].set_xlabel(variable_label if variable_label else variable)

    ax[0].set_ylabel("Events", fontsize=14)
    ax[1].set_ylabel("DATA / MC", fontsize=14)
    ax[1].set_xlabel(variable_label if variable_label else variable, fontsize=14)


    _name = "log" if is_log else "lin"
    #per variabili unrolled
    if variable in UNROLL_LABELLED:
        axspec = VARIABLES_CFG.get(variable, {}).get("axis")
        if isinstance(axspec, (tuple, list)) and len(axspec) == 2:
            xedges2 = np.asarray(axspec[0].edges)
            yedges2 = np.asarray(axspec[1].edges)

            nx = len(xedges2) - 1
            ny = len(yedges2) - 1


            labels  = per_bin_labels(xedges2, yedges2)
            centers = np.arange(len(labels)) + 0.5  # centri dei bin unroll


            ax[1].set_xticks(centers)
            ax[1].set_xticklabels(labels, rotation=90, fontsize=8)


            for j in range(1, ny):
                ax[1].axvline(j*nx, ls=":", lw=0.8, color="grey", alpha=0.7)


            for lab in ax[0].get_xticklabels():
                lab.set_visible(False)


            fig.subplots_adjust(bottom=0.28)

    fig.savefig(
        f"plots/{_name}_{region}_{variable}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )
    plt.close()

def main():
    analysis_dict = get_analysis_dict()
    samples = analysis_dict["samples"]

    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]
    nuisances = analysis_dict["nuisances"]
    blind = analysis_dict.get("blind", True)

    global VARIABLES_CFG
    VARIABLES_CFG = variables


    colors = analysis_dict["colors"]
    plot_label = analysis_dict["plot_label"]
    year_label = analysis_dict.get("year_label", "Run-III")
    lumi = analysis_dict["lumi"]
    print("Doing plots")

    proc = subprocess.Popen(
        "mkdir -p plots && " + f"cp {get_fw_path()}/data/common/index.php plots/",
        shell=True,
    )
    proc.wait()

    # FIXME add nuisance for stat
    nuisances["stat"] = {
        "name": "stat",
        "type": "stat",
        "samples": dict((skey, "1.00") for skey in samples),
    }

    fit_scales = {}

    cpus = 10

    # good_variables = [
    #     "dnn",
    #     # "ptll_unfold",
    #     # "mjj_unfold",
    #     # "dphijj_unfold",
    #     # "dphill_unfold",
    #     # "ptjj_unfold",
    #     # "detajj_unfold",
    #     # "ptj3_unfold",
    #     # "add_HT_unfold",
    # ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = []

        input_file = uproot.open("histos.root")
        for region in regions:
            # if region != "sr_inc_ee":
            #     continue
            for variable in variables:
                if "axis" not in variables[variable]:
                    continue
                # if variable not in good_variables:
                #     continue
                # if variable != "dphijj_reverse_flat_dnn" or "sr" not in region:
                # if variable != "dnn" or "sr" not in region:
                #     continue

                print("submitting", region, variable)
                tasks.append(
                    executor.submit(
                        plot,
                        input_file,
                        region,
                        variable,
                        samples,
                        nuisances,
                        lumi,
                        colors,
                        year_label,
                        blind,
                        variables[variable].get("label"),
                        fit_scales,
                        True,
                    )
                )

                # lin
                tasks.append(
                    executor.submit(
                        plot,
                        input_file,
                        region,
                        variable,
                        samples,
                        nuisances,
                        lumi,
                        colors,
                        year_label,
                        blind,
                        xlabels.get(variable, variable),
                        fit_scales,
                        False,
                    )
                )

        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()
    



if __name__ == "__main__":
    main()