# ORAC-NT — Real-Time Gravitational Wave Detection Pipeline

**Hardware-Tested Low-Latency GW Trigger Architecture**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20129975.svg)](https://doi.org/10.5281/zenodo.20129975)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![ESSOAr](https://img.shields.io/badge/Preprint-ESSOAr-green.svg)](https://essopenarchive.org)

An independent, publicly verifiable gravitational wave detection pipeline operating
in real-time via NASA GCN Kafka. Benchmarked through seven validation phases on
real LIGO O3/O4 data.

**Live demo:** https://orac-nt-core.onrender.com

---

## Key Results

| Metric | Value |
|---|---|
| Detection Rate (Phase A) | 5/5 (100%) canonical events |
| Event Generalization (Phase B) | 10/10 (100%) BNS/BBH/NSBH |
| ROC AUC (Phase E4) | **0.983** |
| Sensitivity 50% efficiency | SNR ≈ 8 (LVK standard) |
| Sensitivity 100% efficiency | SNR ≥ 10 |
| False Alarm Rate | 0.00e+00 /yr (7.96h O3 quiet data) |
| H1+L1+V1 Coincidence (Phase E6) | **6/6 (100%)** |
| IFAR | **> 1,000,000 yr** at all thresholds |
| O4 Detection (Phase E7) | 2/2 (GW200105, GW200115) |
| Hardware Latency | **535 ns** (STM32F401) |
| Software Latency | 130 ms (CPU) |
| Live GCN Events Archived | 30+ Mock Data Challenge events |

---

## Benchmark Summary (Phases A–E7)

| Phase | Test | Result |
|---|---|---|
| A | 5/5 canonical GW events | 100% detection, FAR=0 |
| B | 10/10 event generalization + injections | 100% generalization, 65-80% injection recovery |
| C | H1+L1 coincidence + glitch rejection | 6/8 coincidence, AUC=0.638, 130ms CPU |
| D | Long-duration FAR + ablation + PSD robustness | FAR=0, pipeline stable |
| E1 | Self-calibrated matched filter (GW170817) | Peak SNR=11.73, threshold=8 |
| E2 | Time-slide background estimation | 5σ threshold = SNR 5.93 |
| E3 | Sensitivity curve | 50% @ SNR≈8, 100% @ SNR≥10 |
| E4 | ROC curve | **AUC = 0.983** |
| E5 | Chi² signal/glitch discriminator | Complete separation, zero overlap |
| E6 | H1+L1+V1 multi-detector coincidence | **6/6, IFAR > 1,000,000 yr** |
| E7 | Real O4 noise + event detection | FAR=0, 2/2 detected |

---

## Pipeline Architecture

```
NASA GCN Kafka (8 topics: IGWN, IceCube, Swift, Fermi, Einstein Probe...)
        ↓
orac_live.py — real-time alert listener
        ↓
orac_spinqit_wrapper.py — event archiver + pipeline trigger
        ↓
ORACS_spv18.py — core detection pipeline:
  Bandpass [20–500 Hz] + Whitening (PSD calibrated)
        ↓
  Multi-template MF bank (BBH_O1, BBH_gen, BBH_lite, BNS, NSBH)
        ↓
  Kurtosis veto + Chi² consistency test (4 frequency bands)
        ↓
  H1+L1+V1 coincidence window (Δt ≤ 10ms H1-L1, ≤ 27ms H1-V1)
        ↓
  TRIGGER → history/ archive (JSON + log + PNG)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Required:**
```
gwosc>=0.7.1
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
requests>=2.26
h5py>=3.4
gcn-kafka>=0.3
```

**Setup GCN credentials** (register free at https://gcn.nasa.gov/):
```bash
# Windows PowerShell
$env:GCN_CLIENT_ID="your_client_id"
$env:GCN_CLIENT_SECRET="your_client_secret"

# Linux/macOS
export GCN_CLIENT_ID="your_client_id"
export GCN_CLIENT_SECRET="your_client_secret"
```

Never commit credentials to Git. Use `.env` (already in `.gitignore`).

---

## Usage

**Live NASA GCN monitoring (two terminals):**
```bash
# Terminal 1 — GCN listener
python orac_live.py

# Terminal 2 — Pipeline + archiver
python orac_spinqit_wrapper.py
```

**Run benchmarks:**
```bash
python orac_far_benchmark_v4.py   # Phase A: FAR + detection
python orac_phase_b.py            # Phase B: injection + ROC
python orac_phase_c.py            # Phase C: coincidence + glitch
python orac_phase_d.py            # Phase D: long FAR + ablation
python orac_phase_e1_templates.py # Phase E1: matched filter
python orac_phase_e2_timeslides.py # Phase E2: time-slide background
python orac_phase_e3_injections.py # Phase E3: sensitivity curve
python orac_phase_e4_roc.py       # Phase E4: ROC AUC
python orac_phase_e5_chisq.py     # Phase E5: chi² discriminator
python orac_phase_e6.py           # Phase E6: multi-detector coincidence
python orac_phase_e7.py           # Phase E7: real O4 noise test
```

---

## File Structure

```
├── ORACS_spv18.py              # Core detection pipeline (v34)
├── orac_live.py                # NASA GCN Kafka listener
├── orac_spinqit_wrapper.py     # Event archiver + pipeline trigger
├── orac_archive.py             # Archive utilities
├── orac_far_benchmark_v4.py    # Phase A benchmark
├── orac_phase_b.py             # Phase B: injection campaign
├── orac_phase_c.py             # Phase C: coincidence + glitch
├── orac_phase_d.py             # Phase D: long FAR + ablation
├── orac_phase_e1_templates.py  # Phase E1
├── orac_phase_e2_timeslides.py # Phase E2
├── orac_phase_e3_injections.py # Phase E3
├── orac_phase_e4_roc.py        # Phase E4
├── orac_phase_e5_chisq.py      # Phase E5
├── orac_phase_e6.py            # Phase E6: multi-detector
├── orac_phase_e7.py            # Phase E7: O4 noise
├── ORAC_NT_PhaseE.pdf          # Phase E technical report
├── ORAC_NT_ESRIC_Brief.pdf     # Technical brief (ESRIC Luxembourg)
├── history/                    # Archived GCN events (JSON + logs)
└── requirements.txt            # Python dependencies
```

---

## Hardware Validation

Core algorithm validated on **STM32F401** at 84 MHz:

| Path | Latency |
|---|---|
| Standard trigger | **535 ns** |
| Arrhenius thermal path | 654 ns |
| Software (CPU, Python) | 130 ms |

Repository: [ORAC-QNode](https://github.com/Kretski/ORAC-QNode)

---

## Live GCN Operation

Connected to NASA GCN Kafka since 2026-05-07. Monitoring topics:
- `igwn.gwalert` — LVK gravitational wave alerts
- `gcn.notices.icecube.lvk_nu_track_search` — IceCube neutrino tracks
- `gcn.notices.swift.bat.guano` — Swift BAT GRB
- `gcn.notices.fermi.gbm.general` — Fermi GBM
- `gcn.notices.einstein_probe.wxt.alert` — Einstein Probe
- `gcn.classic.text.LVC_INITIAL` — LVC classic notices
- `igwn.gwalert` heartbeat

30+ Mock Data Challenge events archived (MS260507u–MS260512x).

---

## Sky Localization (v10)

3-detector sky localization via Rodrigues rotation:
```
cos θ = c·Δt / |baseline|
```
Three sky rings (H1-L1, H1-V1, L1-V1) intersect at candidate sky position.

Validated results:

| Event | Our Δt(H1-L1) | LIGO published | Status |
|---|---|---|---|
| GW150914 | −4.15 ms | −6.9 ms | ✓ correct sign |
| GW170814 | 3-ring intersection | ~RA 50°, Dec −45° | ✓ correct quadrant |
| GW170817 | BNS template dominant | ✓ | ✓ |

---

## Publications

- **ESSOAr Preprint:** https://essopenarchive.org (In Moderation)
- **Zenodo Phase E:** https://doi.org/10.5281/zenodo.20129975
- **Zenodo Phase A-D:** https://doi.org/10.5281/zenodo.20098932
- **Original preprint:** https://doi.org/10.5281/zenodo.19553825

---

## Roadmap

| Milestone | Description | Timeline |
|---|---|---|
| E8 | Network SNR H1+L1+V1 combined | Q3 2026 |
| E9 | IR1 live monitoring (real O5 alerts) | Oct-Nov 2026 |
| E10 | LISA frequency band adaptation | Q1 2027 |

---

## Citation

```bibtex
@software{kretski_orac_nt_2026,
  author    = {Kretski, Dimitar},
  title     = {{ORAC-NT: Real-Time Gravitational Wave Detection Pipeline}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20129975},
  url       = {https://doi.org/10.5281/zenodo.20129975}
}
```

---

## Related Work

- **ORAC-QNode** — STM32 hardware vitality controller: https://github.com/Kretski/ORAC-QNode
- **Boyajian's Star** — W-functional on Kepler light curves: https://zenodo.org/records/19536321
- **SynergyFF** — molecular ensemble optimizer (patent pending BG): https://github.com/Kretski

---

## License

**CC BY-NC 4.0** — Dimitar Kretski, 2026

Non-commercial use: free with attribution.
Commercial use: contact kretski1@gmail.com

Data: LIGO Open Science Center (GWOSC), CC BY 4.0
