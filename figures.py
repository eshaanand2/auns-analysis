from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import auns_analysis as aa

FIG_DIR = Path("poster_figs")
FIG_DIR.mkdir(exist_ok=True)
DPI = 300
SAT_LIMIT = 2.0 

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
LINESTYLES = ["-", "--", ":", "-."]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def color_map(spectra):
    concs = sorted({c for c, _ in spectra})
    return {c: COLORS[i % len(COLORS)] for i, c in enumerate(concs)}


def fig_spectra(spectra):
    colors = color_map(spectra)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    any_sat = False
    for (c, r), df in sorted(spectra.items()):
        sat = df["abs"].max() > SAT_LIMIT
        any_sat |= sat
        label = f"{c:g} mM (round {r})" + (" *" if sat else "")
        ax.plot(df.wl, df["abs_corr"], color=colors[c],
                ls=LINESTYLES[(r - 1) % len(LINESTYLES)], label=label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (dilution-corrected)")
    ax.set_xlim(aa.WL_MIN, aa.WL_MAX)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    if any_sat:
        fig.text(0.01, -0.02, "* measured absorbance exceeds 2 (above reliable range)",
                 fontsize=13, ha="left")
    fig.tight_layout()
    save(fig, "fig1_spectra")


def fig_peak_trends(spectra, pk):
    sat = {k for k, df in spectra.items() if df["abs"].max() > SAT_LIMIT}
    concs = sorted(pk.conc_mM.unique())
    colors = color_map(spectra)
    panels = [("peak_wl_nm", "Peak wavelength (nm)"),
              ("max_abs", "Max absorbance (dilution-corrected)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    any_n1 = False
    for ax, (col, ylabel) in zip(axes, panels):
        for c in concs:
            sub = pk[pk.conc_mM == c]
            vals = sub[col].values
            for (_, row) in sub.iterrows():
                is_sat = (c, row["round"]) in sat
                ax.plot(c, row[col], "o", ms=10, color=colors[c], alpha=0.6,
                        mfc="white" if is_sat else colors[c], mew=2)
            if len(vals) >= 2:
                ax.errorbar(c, vals.mean(), yerr=vals.std(ddof=1), fmt="_", ms=28,
                            color=colors[c], capsize=8, elinewidth=2.5, mew=3)
            else:
                any_n1 = True
        ax.margins(y=0.15)
        ax.set_xlabel("Concentration (mM)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(concs)
        ax.set_xticklabels([f"{c:g}" for c in concs])
        pad = 0.5 * (concs[-1] - concs[0]) if len(concs) > 1 else 1
        ax.set_xlim(concs[0] - 0.25 * pad, concs[-1] + 0.25 * pad)
    notes = []
    if any_n1:
        notes.append("no error bars where n = 1")
    if sat:
        notes.append("open markers: measured absorbance > 2")
    if notes:
        fig.text(0.01, -0.02, "; ".join(notes), fontsize=13, ha="left")
    fig.tight_layout()
    save(fig, "fig2_peak_trends")


if __name__ == "__main__":
    sp = aa.load_all()
    pk = aa.peak_table(sp)
    fig_spectra(sp)
    fig_peak_trends(sp, pk)
    print(f"Saved figures to {FIG_DIR.resolve()}")