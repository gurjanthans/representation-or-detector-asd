#!/usr/bin/env python3
# ============================================================================
#  make_figures_from_json.py
#  Regenerates fig_interaction.pdf and fig_dcase.pdf from the saved result
#  JSONs — no GPU, no pipeline rerun. Run locally:  python3 make_figures_from_json.py
#  Requires: numpy, matplotlib   (pip install numpy matplotlib)
# ============================================================================
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- EDIT THIS if your folder moves -----------------------------------------
DATA_DIR = "/Users/student/Desktop/First/1. Research Work/3. Papers/8. AD Apch 2 (Moitrayan)/TASLP_reproduce_all_final"
# -----------------------------------------------------------------------------

def _find(name):
    """Locate a JSON in DATA_DIR, or a results/ subfolder, or cwd."""
    for cand in (os.path.join(DATA_DIR, name),
                 os.path.join(DATA_DIR, "results", name),
                 name):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find {name}. Looked in:\n"
        f"  {os.path.join(DATA_DIR, name)}\n"
        f"  {os.path.join(DATA_DIR, 'results', name)}\n"
        f"Put {name} in DATA_DIR (or a results/ subfolder) and re-run.")

us8k_res  = json.load(open(_find("us8k_grid.json")))
dcase_res = json.load(open(_find("dcase_results.json")))

OUT_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def _scalar(v):
    return float(np.mean(v)) if isinstance(v, list) else float(v)

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# ============================================================================
#  Fig. (interaction): representation x detector on UrbanSound8K
# ============================================================================
ENC   = ["panns", "ast", "beats", "wav2vec2"]
ENC_L = ["PANNs", "AST", "BEATs", "wav2vec2"]
DETS = [("knn",        "$k$-NN",       "#0072B2", "-",  "o"),
        ("mahalanobis","Mahalanobis",  "#009E73", "-",  "s"),
        ("pca_recon",  "PCA recon.",   "#D55E00", "-",  "^"),
        ("gmm",        "GMM",          "#56B4E9", "--", "D"),
        ("ocsvm",      "OC-SVM",       "#CC79A7", "--", "v"),
        ("isoforest",  "Iso. Forest",  "#E69F00", "--", "P")]

fig, ax = plt.subplots(figsize=(3.5, 2.7))
x = np.arange(len(ENC))
for key, label, col, ls, mk in DETS:
    means = np.array([np.mean(us8k_res[e][key]["auroc"]) for e in ENC])
    sds   = np.array([np.std (us8k_res[e][key]["auroc"]) for e in ENC])
    ax.errorbar(x, means, yerr=sds, label=label, color=col, ls=ls, marker=mk,
                ms=4, lw=1.3, capsize=2, elinewidth=0.7, alpha=0.95)
ax.set_xticks(x); ax.set_xticklabels(ENC_L)
ax.set_ylabel("AUROC (mean $\\pm$ SD, 9 rotations)")
ax.set_xlim(-0.35, len(ENC) - 0.65)
ax.set_ylim(0.50, 0.92)
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
ax.legend(ncol=2, fontsize=6.5, frameon=False, loc="lower left",
          handlelength=1.8, columnspacing=1.0, labelspacing=0.3)
fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"fig_interaction.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved:", os.path.join(OUT_DIR, "fig_interaction.pdf"))

# ============================================================================
#  Fig. (DCASE): per-machine k-NN AUROC vs official baseline
# ============================================================================
MACHINES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
ENC_C = {"panns": "#0072B2", "ast": "#D55E00", "beats": "#009E73", "wav2vec2": "#CC79A7"}
OFFICIAL = {"fan": 0.658, "pump": 0.729, "slider": 0.848, "valve": 0.663,
            "ToyCar": 0.788, "ToyConveyor": 0.725}   # DCASE 2020 official baseline

fig, ax = plt.subplots(figsize=(3.5, 2.8))
x = np.arange(len(MACHINES)); w = 0.19
for i, e in enumerate(ENC):
    vals = [_scalar(dcase_res[m][e]["knn"]["auroc"]) for m in MACHINES]
    ax.bar(x + (i - 1.5) * w, vals, width=w, label=ENC_L[i],
           color=ENC_C[e], edgecolor="white", linewidth=0.3)
for j, m in enumerate(MACHINES):
    ax.plot([x[j] - 2 * w, x[j] + 2 * w], [OFFICIAL[m]] * 2, color="black",
            lw=1.2, ls="--", zorder=5, label="baseline" if j == 0 else None)
ax.set_xticks(x); ax.set_xticklabels(MACHINES, rotation=25, ha="right")
ax.set_ylabel("$k$-NN AUROC")
ax.set_ylim(0.50, 0.98)
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.4)
fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"fig_dcase.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved:", os.path.join(OUT_DIR, "fig_dcase.pdf"))
print("\nDone. Two PDFs are in:", OUT_DIR)
