import re
from pathlib import Path
from itertools import combinations
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy import stats
 
DATA_DIR = Path("data")
OUT_DIR = Path("results")
WL_MIN, WL_MAX = 400, 1100  #analysis window
PEAK_WINDOW = (600, 1100)  #LSPR peak
SHOULDER_WINDOW = (450, 650)  #shoulder window --> remember to edit to match spectra
SMOOTH = True
EPSILON = None  #no standard value for AuNS
PATH_LENGTH = 1.0
DILUTION_TAGS = {"50-50": 2.0}  # filename tag -> dilution factor (no tag = undiluted)
GROUPS = {"undiluted": 1.0, "diluted": 2.0}
SAT_LIMIT = 2.0
 
EXCLUDE = []
 
OUT_DIR.mkdir(exist_ok=True)
 
def parse_name(stem):
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*mM", stem, re.I)
    if not m:
        raise ValueError(f"Can't find a concentration like '10.5mM' in '{stem}'")
    conc = float(m.group(1))
    
    r = re.search(r"round[_ ]?(\d+)|[_ ]r(\d+)", stem, re.I)
    rnd = int(r.group(1) or r.group(2)) if r else 1
    return conc, rnd

def read_uds_txt(f):
    """Hitachi UH4150 export: the real data sits under a 'Data Points' header as
    wavelength<TAB>%T (descending). Convert %T -> absorbance: A = 2 - log10(%T)."""
    lines = f.read_text(errors="ignore").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "Data Points")
    rows = []
    for l in lines[start + 1:]:
        parts = l.split("\t")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            pass
    df = pd.DataFrame(rows, columns=["wl", "pT"])
    df = df[df.pT > 0].copy()  # %t <= 0 --> noise, log will be undefined
    df["abs"] = 2 - np.log10(df.pT)
    return df[["wl", "abs"]]

def get_dilution(stem):
    for tag, d in DILUTION_TAGS.items():
        if tag in stem:
            return d
    return 1.0
 
def load_all(group):
    want = GROUPS[group]
    spectra, seen = {}, {}
    for f in sorted(DATA_DIR.glob("*.[Tt][Xx][Tt]")):
        if f.stem in EXCLUDE or get_dilution(f.stem) != want:
            continue
        df = read_uds_txt(f)
        df = df[(df.wl >= WL_MIN) & (df.wl <= WL_MAX)].sort_values("wl").reset_index(drop=True)
        df["abs_corr"] = df["abs"] * want
        key = parse_name(f.stem)
        if key in spectra:
            raise ValueError(f"'{seen[key]}' and '{f.name}' both map to {key}")
        seen[key] = f.name
        if df["abs"].max() > SAT_LIMIT:
            print(f"WARNING {f.name}: absorbance reaches {df['abs'].max():.2f} (>{SAT_LIMIT} is unreliable)")
        spectra[key] = df
    return spectra
 
def smooth(y):
    return savgol_filter(y, 11, 3) if SMOOTH and len(y) > 11 else y

def plot_spectra(spectra):
    concs = sorted({c for c, _ in spectra})
    fig, axes = plt.subplots(1, len(concs), figsize=(5 * len(concs), 4), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], concs):
        for (cc, r), df in spectra.items():
            if cc == c:
                ax.plot(df.wl, df["abs"], label=f"round {r}")
        ax.set_title(f"{c} mM")
        ax.set_xlabel("Wavelength (nm)")
        ax.legend()
    axes[0][0].set_ylabel("Absorbance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectra_by_conc.png", dpi=200)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (c, r), df in spectra.items():
        ax.plot(df.wl, df["abs"], label=f"{c} mM r{r}")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectra_all.png", dpi=200)

def beer_lambert(spectra):
    if EPSILON is None:
        print("EPSILON not set - skipping Beer-Lambert conversion")
        return
    frames = []
    for (c, r), df in spectra.items():
        df["conc_M"] = df["abs_corr"] / (EPSILON * PATH_LENGTH)
        out = df[["wl", "abs", "abs_corr", "conc_M"]].copy()
        out.insert(0, "round", r)
        out.insert(0, "conc_mM", c)
        frames.append(out)
    pd.concat(frames).to_csv(OUT_DIR / "beer_lambert.csv", index=False)

def peak_table(spectra):
    rows = []
    for (c, r), df in spectra.items():
        m = (df.wl >= PEAK_WINDOW[0]) & (df.wl <= PEAK_WINDOW[1])
        sub = df[m]
        y = smooth(sub["abs_corr"].values)
        i = int(np.argmax(y))
        rows.append(dict(conc_mM=c, round=r, peak_wl_nm=sub.wl.values[i], max_abs=y[i]))
    t = pd.DataFrame(rows).sort_values(["conc_mM", "round"])
    t.to_csv(OUT_DIR / "peak_table.csv", index=False)
    return t

def summary(t):
    s = t.groupby("conc_mM").agg(
        n=("round", "count"),
        peak_wl_mean=("peak_wl_nm", "mean"), peak_wl_std=("peak_wl_nm", "std"),
        max_abs_mean=("max_abs", "mean"), max_abs_std=("max_abs", "std"),
    )
    s.to_csv(OUT_DIR / "summary_stats.csv")
    return s

def ttests(t):
    rows = []
    for a, b in combinations(sorted(t.conc_mM.unique()), 2):
        for col in ["max_abs", "peak_wl_nm"]:
            x = t[t.conc_mM == a][col]; y = t[t.conc_mM == b][col]
            if len(x) < 2 or len(y) < 2:
                continue
            res = stats.ttest_ind(x, y, equal_var=False)  #two-tailed
            rows.append(dict(A=a, B=b, metric=col, t=res.statistic, p=res.pvalue))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bonferroni"] = (out.p * out.groupby("metric").p.transform("count")).clip(upper=1)
    out.to_csv(OUT_DIR / "t_tests.csv", index=False)
    return out

def shoulders(spectra):
    """Detect shoulder as a local peak (by prominence) in SHOULDER_WINDOW,
    or as a bump in the 2nd derivative if there is no true rel max.
    Ratio = shoulder abs / main peak abs. Higher ratio --> more of aggregates/less 
    branched particles, will track across conditions."""
    rows = []
    for (c, r), df in spectra.items():
        y = smooth(df["abs"].values); wl = df.wl.values
        pm = (wl >= PEAK_WINDOW[0]) & (wl <= PEAK_WINDOW[1])
        main_abs = y[pm].max(); main_wl = wl[pm][np.argmax(y[pm])]
        sm = (wl >= SHOULDER_WINDOW[0]) & (wl <= SHOULDER_WINDOW[1])
        idx, _ = find_peaks(y[sm], prominence=0.005)
        if len(idx):
            j = idx[np.argmax(y[sm][idx])]
            sh_wl, sh_abs, how = wl[sm][j], y[sm][j], "local max"
        else:
            d2 = np.gradient(np.gradient(y, wl), wl)
            k = np.argmin(d2[sm])
            sh_wl, sh_abs, how = wl[sm][k], y[sm][k], "2nd-derivative"
        rows.append(dict(conc_mM=c, round=r, main_wl=main_wl, main_abs=main_abs,
                         shoulder_wl=sh_wl, shoulder_abs=sh_abs,
                         ratio_shoulder_to_main=sh_abs / main_abs, method=how))
    t = pd.DataFrame(rows).sort_values(["conc_mM", "round"])
    t.to_csv(OUT_DIR / "shoulder_table.csv", index=False)
    return t


if __name__ == "__main__":
    for group in GROUPS:
        print(f"\n===== {group} =====")
        sp = load_all(group)
        if not sp:
            print("no files in this group")
            continue
        OUT_DIR = Path("results") / group
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        beer_lambert(sp)
        plot_spectra(sp)
        pk = peak_table(sp); print(pk, "\n")
        print(summary(pk), "\n")
        print(ttests(pk), "\n")
        print(shoulders(sp))
