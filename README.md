# Representation or Detector? — Reproducibility Repository

Code and results for the paper:

> **Representation or Detector? A Factorial Study of Embedding-Based Audio Anomaly Detection, and a Label-Supervision Bias in Held-Out-Class Protocols**
> Gurjant Singh, Moitrayan Chakravarty, Brejesh Lall.

This repository regenerates **every table and figure** in the paper from one notebook, plus a lightweight script to redraw the figures from cached results.

---

## Contents

| Path | Description |
|---|---|
| `reproduce-all.ipynb` | End-to-end notebook: embedding extraction, the factorial grid, ANOVA/Wilcoxon, DCASE, SNR robustness, embedding geometry, the within-class overlap test, and a self-check that prints PASS/FAIL against the paper values. |
| `make_figures_from_json.py` | Redraws `fig_interaction.pdf` and `fig_dcase.pdf` from `results/*.json` with no GPU. |
| `results/` | The result JSONs the paper's tables/figures are built from (`us8k_grid.json`, `dcase_results.json`, `snr_results.json`, `table4_bootstrap.json`, `geometry.json`). |
| `figures/` | Publication figures (PDF/PNG). |
| `audioset_ontology.json` | Frozen AudioSet-ontology snapshot used for the ESC-50 overlap mapping (makes the high/low-overlap class assignment deterministic). |
| `requirements.txt` | Pinned Python environment that reproduces the reported numbers. |

---

## Environment

The reported numbers were produced with **Python 3.11** and the versions in `requirements.txt`
(PyTorch 2.10.0, HuggingFace Transformers **5.13.1**, scikit-learn 1.6.1, SciPy 1.16.3, Pingouin 0.6.1)
on an NVIDIA T4 GPU.

> **Note.** The wav2vec 2.0 embeddings — and hence the small-sample low-overlap AUROC of the speech control in the within-class test — are sensitive to the Transformers version. The pinned environment reproduces the reported values; the qualitative conclusions (supervised encoders show the smallest per-class inflation; detector-dominant variance; matched-domain gap) are invariant to this choice.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

All datasets are public. Download them and point the path variables at the top of the notebook to your local copies:

- **UrbanSound8K** — Salamon et al., ACM MM 2014.
- **ESC-50** — Piczak, ACM MM 2015.
- **DCASE 2020 Task 2** (development data, 6 machines) — downloaded automatically from Zenodo by the notebook.
- **BEATs_iter3.pt** — the *self-supervised* (pre-fine-tuning) checkpoint from microsoft/unilm. **Do not** use the AudioSet-fine-tuned `BEATs_iter3_plus_AS2M.pt`; it would invalidate the label-free control.

---

## Reproduce

**Full pipeline (GPU):** run `reproduce-all.ipynb` top to bottom. The final cell prints a PASS/FAIL self-check against the paper values. On a 12-hour-limited platform, use per-encoder/per-level checkpoints and resume across sessions.

**Figures only (no GPU):** edit `DATA_DIR` in `make_figures_from_json.py` to point at the `results/` folder, then:

```bash
python make_figures_from_json.py
```

---

## Citation

```bibtex
@article{singh2026representation,
  author  = {Singh, Gurjant and Chakravarty, Moitrayan and Lall, Brejesh},
  title   = {Representation or Detector? A Factorial Study of Embedding-Based Audio Anomaly Detection, and a Label-Supervision Bias in Held-Out-Class Protocols},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the MIT License (see `LICENSE`).
