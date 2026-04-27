# 🌍 ORAC-NT v10 — Global Gravitational Wave Triangulation Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19553825.svg)](https://doi.org/10.5281/zenodo.19553825)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

A physically validated gravitational wave matched filter pipeline with
3-detector sky localization and live NASA GCN alert integration,
built on real LIGO/GWOSC public data.

**Live demo:** https://orac-nt-core.onrender.com

![ORAC-NT v10 Sky Localization](ORAC-NT%20v9_2.png)

---

## What's new in v10

| Version | Change |
|---------|--------|
| v9 | Validated pipeline: 2PN matched filter, 3-ring sky localization |
| **v10** | **Dual-tab UI: Historical (GWOSC) + Live GCN Alert mode** |

**Tab 1 — Historical:** select any of 4 LIGO events, run full triangulation on archived data.

**Tab 2 — Live GCN Alert:** connects to NASA GCN Kafka stream via `orac_live.py`. When a real alert arrives, auto-selects the template (BBH or BNS), fetches live strain data, and displays the sky localization in real time.

---

## Pipeline

```
GWOSC / Live GCN data (H1 + L1 + V1)
        ↓
Bandpass [20–f_ISCO Hz] + Whitening (PSD, nperseg=4×FS)
        ↓
2PN SPA Einstein chirp template
        ↓
Zero-padded linear FFT matched filter (MAD-normalized SNR)
        ↓
Causal peak search (±light-travel time per detector pair)
        ↓
3 sky rings via Rodrigues rotation (ECEF coordinates)
        ↓
Ring intersection finder → 2D candidate (RA, Dec)
```

---

## Validated results

| Event | Our Δt(H1-L1) | LIGO published | Status |
|-------|--------------|----------------|--------|
| GW150914 | −4.15 ms | −6.9 ms | ✓ correct sign & order of magnitude |
| GW170814 | 3-ring intersection | ~RA 50°, Dec −45° | ✓ correct quadrant |
| GW170817 | NS template dominates BH | ✓ | ✓ |

> Single-template SNR < 8σ is expected. Full LIGO pipelines use ~250,000 templates.
> Timing and sky geometry are physically validated.

---

## Installation

```bash
pip install streamlit gwpy scipy numpy matplotlib
streamlit run orac_nt_v10.py
```

Data is downloaded automatically from GWOSC on first run and cached locally.

---

## Supported events (Historical tab)

| Event | Detectors | Type |
|-------|-----------|------|
| GW150914 | H1 + L1 | First BBH merger (~35+30 M☉) |
| GW170104 | H1 + L1 | BBH ~20 M☉ |
| GW170814 | H1 + L1 + V1 | First 3-detector BBH merger |
| GW170817 | H1 + L1 + V1 | Binary NS merger (kilonova) |

---

## Key physics

### 2PN SPA Template
```
ψ(f) = (3/128η) · v⁻⁵ · [1 + (20/9)(743/336 + 11η/4)·v²
                           − 16π·v³
                           + 10·(3058673/1016064 + 5429η/1008 + 617η²/144)·v⁴]
```
where `v = (πGMf/c³)^(1/3)`. Ref: Blanchet et al. 1995; Cutler & Flanagan 1994.

### Sky ring geometry
Each detector pair defines a constant time-delay cone:
`cos θ = c·Δt / |baseline|`
Rotated via Rodrigues rotation around the true ECEF baseline vector.
Three rings (H1-L1, H1-V1, L1-V1) intersect at the source.

### MAD normalization
`σ_MAD = 1.4826 · median(|x − median(x)|)`
Robust to signal self-suppression — unlike std or rolling-std.

---

## Optional: Live GW Alert Listener

`orac_live.py` connects to the NASA GCN Kafka stream and listens for
real gravitational wave alerts. When a new event arrives, it saves
GPS time and classification to `latest_event.json`, which Tab 2
picks up automatically.

```powershell
# Windows PowerShell — Terminal 1
$env:GCN_CLIENT_ID="your_client_id"
$env:GCN_CLIENT_SECRET="your_client_secret"
python orac_live.py

# Terminal 2
streamlit run orac_nt_v10.py
```

```bash
# Linux / macOS — Terminal 1
export GCN_CLIENT_ID="your_client_id"
export GCN_CLIENT_SECRET="your_client_secret"
python orac_live.py
```

Register for free at https://gcn.nasa.gov/ to get credentials.
> Never commit credentials to Git. `.env` is in `.gitignore`.

---

## Pipeline evolution

| Version | Key fix |
|---------|---------|
| v5.1 | 2PN SPA template; ±10ms physical delay constraint |
| v6 | Virgo (V1) third detector; 3-ring sky map |
| v7 | Ring intersection finder |
| v8 | Merger-window peak search; bandpass 20–f_ISCO; MAD SNR |
| v8.3 | Zero-padded linear FFT (no circular wrap artifact) |
| v8.5 | Template-peak aligned trimming (correct delay sign) |
| v9 | Validated release — Δt(H1-L1) = −4.15 ms ✓ |
| **v10** | **Dual-tab UI + Live GCN Alert integration** |

---

## Boyajian's Star Application (W-Functional)

The same stability framework — **W(t) = η·α·D(r,t) − β(t)** — was applied
to NASA Kepler photometric data for KIC 8462852 (Boyajian's Star), one of
the most anomalous stellar light curves in the Kepler dataset.

The W-functional amplifies photometric deviations and provides a
stability-based diagnostic complementary to standard flux-ratio analysis.
KIC 8462852 exhibits extreme W excursions (W ≈ −0.41 during largest dip
events), distinguishing it from typical Kepler targets.

**Preprint:** https://zenodo.org/records/19536321

**To replicate:**
1. Download Kepler Q1-Q17 long-cadence data for KIC 8462852 from MAST:
   https://archive.stsci.edu/kepler/
2. Apply the W-functional as described in the preprint
3. Compare W(t) excursions against a control sample of stable Kepler stars

The MAD normalization used in ORAC-NT v10 originates from the same
robust statistics approach developed for the Boyajian's Star analysis.

---

## Citation

```bibtex
@software{kretski_orac_nt_2026,
  author    = {Kretski, Dimitar},
  title     = {{ORAC-NT v9: A 2PN Matched Filter Pipeline for Gravitational
                Wave Detection and 3-Detector Sky Localization}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19553825},
  url       = {https://doi.org/10.5281/zenodo.19553825}
}
```

---

## Related work

- **Boyajian's Star anomaly detection** (ORAC-NT W-functional on Kepler light curves):
  https://zenodo.org/records/19536321

- **SynergyFF** — molecular ensemble optimizer for drug discovery (patent pending BG):
  https://github.com/Kretski

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Dimitar Kretski, 2026

Data: LIGO Open Science Center (GWOSC), CC BY 4.0
