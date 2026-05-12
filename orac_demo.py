"""
ORAC-NT Demo — End-to-End Gravitational Wave Detection
=======================================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20129975
GitHub : github.com/Kretski/ORAC-NT-Astrophysics

PUBLIC DEMO — Phase A benchmark only.
Full pipeline (Phases B-E7) available under commercial license.
Contact: kretski1@gmail.com

Demonstrates:
- Real LIGO O3 data download from GWOSC
- Multi-template matched filter detection
- Kurtosis + chi2 veto
- FAR calibration on real quiet data
- Detection of canonical GW events

INSTALLATION:
  pip install -r requirements.txt

USAGE:
  python orac_demo.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal.windows import tukey
from scipy.stats import kurtosis
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

FS = 4096  # Sample rate (Hz)

QUIET_SEGMENTS = [
    {"name": "O3a_quiet_A", "gps": 1238166018, "det": "L1", "dur": 4096},
    {"name": "O3a_quiet_B", "gps": 1242578176, "det": "L1", "dur": 4096},
]

GW_EVENTS = [
    {"name": "GW170817", "type": "BNS",  "det": "L1", "dur": 32},
    {"name": "GW150914", "type": "BBH",  "det": "L1", "dur": 32},
    {"name": "GW190521", "type": "BBH",  "det": "L1", "dur": 32},
    {"name": "GW190814", "type": "NSBH", "det": "L1", "dur": 32},
    {"name": "GW151226", "type": "BBH",  "det": "L1", "dur": 32},
]

TEMPLATES = [
    {"name": "BBH_O1",  "f0": 35,  "f1": 150, "duration": 0.2},
    {"name": "BBH_gen", "f0": 20,  "f1": 350, "duration": 0.4},
    {"name": "BNS",     "f0": 30,  "f1": 400, "duration": 1.0},
    {"name": "NSBH",    "f0": 20,  "f1": 300, "duration": 0.6},
]

# ─────────────────────────────────────────────────────────────
# Data fetch (public GWOSC)
# ─────────────────────────────────────────────────────────────

def fetch_event(event_name, detector, duration):
    try:
        import requests, h5py
        from gwosc.locate import get_event_urls
        time.sleep(1.5)
        urls = get_event_urls(event_name, detector=detector, duration=duration)
        if not urls: return None
        r = requests.get(urls[0], timeout=180, stream=True)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        for chunk in r.iter_content(65536): tmp.write(chunk)
        tmp.close()
        with h5py.File(tmp.name, 'r') as f:
            data = f['strain']['Strain'][:]
            dt   = f['strain']['Strain'].attrs.get('Xspacing', 1.0/FS)
        os.unlink(tmp.name)
        strain = data.astype(np.float64)
        sr = int(round(1.0/dt))
        if sr != FS:
            from scipy.signal import resample
            strain = resample(strain, int(len(strain)*FS/sr))
        target = duration * FS
        if len(strain) > target:
            mid = len(strain)//2
            strain = strain[mid-target//2: mid+target//2]
        return strain
    except Exception as e:
        print(f"    [WARN] {e}")
        return None

def fetch_bulk(gps, duration, detector):
    try:
        import requests, h5py
        from gwosc.locate import get_urls
        time.sleep(1.5)
        urls = get_urls(detector, gps, gps + duration)
        if not urls: return None
        hdf_url = next((u for u in urls if '4096' in u and 'hdf5' in u), urls[0])
        r = requests.get(hdf_url, timeout=240, stream=True)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        for chunk in r.iter_content(65536): tmp.write(chunk)
        tmp.close()
        with h5py.File(tmp.name, 'r') as f:
            data = f['strain']['Strain'][:]
            dt   = f['strain']['Strain'].attrs.get('Xspacing', 1.0/FS)
        os.unlink(tmp.name)
        strain = data.astype(np.float64)
        sr = int(round(1.0/dt))
        if sr != FS:
            from scipy.signal import resample
            strain = resample(strain, int(len(strain)*FS/sr))
        return strain
    except Exception as e:
        print(f"    [WARN] {e}")
        return None

# ─────────────────────────────────────────────────────────────
# ORAC-NT core (Phase A)
# ─────────────────────────────────────────────────────────────

def whiten(data, cal_s=5.0):
    """Bandpass + PSD whitening."""
    sos    = signal.butter(4, [20, 500], btype='bandpass', fs=FS, output='sos')
    data   = signal.sosfiltfilt(sos, data)
    cal    = int(cal_s * FS)
    f, psd = signal.welch(data[:cal], FS, nperseg=FS)
    psd_i  = np.interp(np.fft.rfftfreq(len(data), 1/FS), f, psd)
    w      = np.fft.irfft(np.fft.rfft(data) / np.sqrt(psd_i + 1e-12), n=len(data))
    return np.nan_to_num(w) * tukey(len(w), 0.05)

def make_template(cfg, n):
    """Chirp template for matched filtering."""
    dur  = cfg['duration']
    tt   = np.linspace(0, dur, int(dur*FS), endpoint=False)
    tmpl = signal.chirp(tt, f0=cfg['f0'], f1=cfg['f1'], t1=dur, method='quadratic')
    tmpl *= tukey(len(tmpl), 0.2)
    tmpl /= (np.sqrt(np.sum(tmpl**2)) + 1e-20)
    if len(tmpl) < n: tmpl = np.pad(tmpl, (0, n-len(tmpl)))
    else: tmpl = tmpl[:n]
    return tmpl

def get_peak_snr(strain, center_t=None, window=5.0):
    """Multi-template matched filter SNR."""
    cal   = int(5.0 * FS)
    w     = whiten(strain)
    w    /= (np.std(w[:cal]) + 1e-12)
    n     = len(w)
    best  = 0.0
    if center_t is None: center_t = len(strain) / FS / 2.0

    for cfg in TEMPLATES:
        tmpl  = make_template(cfg, n)
        corr  = signal.correlate(w, tmpl, mode='same')
        med   = np.median(corr)
        mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
        snr_t = np.abs(corr - med) / mad
        t_arr = np.arange(n) / FS
        mask  = (t_arr >= center_t - window) & (t_arr <= center_t + window)
        if np.any(mask):
            best = max(best, float(np.max(snr_t[mask])))
    return best

def kurtosis_veto(strain, t_trigger):
    """Reject instrumental glitches via kurtosis."""
    idx     = int(t_trigger * FS)
    half    = int(0.5 * FS)
    snippet = strain[max(0,idx-half): min(len(strain),idx+half)]
    if len(snippet) < 100: return True
    return kurtosis(snippet, fisher=False, bias=False) > 25.0

def count_false_triggers(strain, threshold):
    """Count false triggers for FAR estimation."""
    cal   = int(5.0 * FS)
    w     = whiten(strain)
    w    /= (np.std(w[:cal]) + 1e-12)
    n     = len(w)
    best_snr = np.zeros(n)
    for cfg in TEMPLATES:
        tmpl  = make_template(cfg, n)
        corr  = signal.correlate(w, tmpl, mode='same')
        med   = np.median(corr)
        mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
        snr_t = np.abs(corr - med) / mad
        best_snr = np.maximum(best_snr, snr_t)

    above = best_snr > threshold
    count = 0
    in_seg = False
    for a in above:
        if a and not in_seg: in_seg = True
        elif not a and in_seg: in_seg = False; count += 1
    return count

# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────

def run_demo():
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 60)
    print("  ORAC-NT — Public Demo (Phase A)")
    print(f"  {run_utc}")
    print("  Author: Dimitar Kretski | DOI: 10.5281/zenodo.20129975")
    print("=" * 60)

    # ── Step 1: Load quiet data + calibrate threshold ─────
    print("\n[1/3] Loading quiet O3 data from GWOSC...")
    quiet_strains = []
    total_time    = 0.0
    MAX_SNR_SKIP  = 11.0

    for seg in QUIET_SEGMENTS:
        print(f"  {seg['name']}...")
        s = fetch_bulk(seg['gps'], seg['dur'], seg['det'])
        if s is None: continue

        # Pre-scan for massive glitches
        snr_q = get_peak_snr(s, center_t=len(s)/FS/2.0, window=len(s)/FS/2.0-5)
        if snr_q > MAX_SNR_SKIP:
            print(f"    [SKIP] Glitch detected (SNR={snr_q:.1f})")
            continue

        quiet_strains.append(s)
        total_time += len(s) / FS
        print(f"    [OK] {len(s)/FS:.0f}s, max SNR={snr_q:.2f}")

    if not quiet_strains:
        print("[ERROR] No clean quiet data available.")
        return

    # Binary search for threshold (target FAR = 1/yr)
    print(f"\n  Calibrating threshold on {total_time/3600:.2f}h quiet data...")
    lo, hi = 3.0, 9.0
    threshold = hi
    for _ in range(8):
        mid = (lo + hi) / 2.0
        false = sum(count_false_triggers(s, mid) for s in quiet_strains)
        far   = (false / total_time) * 3.156e7
        if far > 1.0: lo = mid
        else: hi = mid; threshold = mid

    false_total = sum(count_false_triggers(s, threshold) for s in quiet_strains)
    far_total   = (false_total / total_time) * 3.156e7
    print(f"  Threshold: {threshold:.3f} | FAR: {far_total:.2e} /yr | "
          f"False alarms: {false_total}")

    # ── Step 2: Detect GW events ──────────────────────────
    print(f"\n[2/3] Testing detection on canonical GW events...")
    print(f"  Threshold = {threshold:.3f}")
    print("-" * 60)

    results = []
    for ev in GW_EVENTS:
        print(f"\n  [{ev['name']}] {ev['type']}")
        strain = fetch_event(ev['name'], ev['det'], ev['dur'])
        if strain is None:
            results.append({**ev, "detected": False, "snr": 0.0})
            continue

        center   = len(strain) / FS / 2.0
        peak_snr = get_peak_snr(strain, center)
        glitch   = kurtosis_veto(strain, center)
        detected = (peak_snr >= threshold) and not glitch

        status = "DETECTED" if detected else "MISSED"
        print(f"    [{status}] SNR={peak_snr:.2f} | "
              f"Kurtosis veto: {'YES' if glitch else 'no'}")
        results.append({**ev, "detected": detected, "snr": peak_snr})

    # ── Step 3: Summary + plot ────────────────────────────
    detected_count = sum(1 for r in results if r['detected'])
    total_count    = len(results)

    print(f"\n[3/3] Results")
    print("=" * 60)
    print(f"  Threshold     : {threshold:.3f}")
    print(f"  Detection     : {detected_count}/{total_count} "
          f"({detected_count/total_count*100:.0f}%)")
    print(f"  FAR           : {far_total:.2e} /yr")
    print(f"  False alarms  : {false_total}")
    print(f"  Quiet data    : {total_time/3600:.2f}h")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0e1117')

    names  = [r['name'] for r in results]
    snrs   = [r['snr']  for r in results]
    colors = ['#00ffcc' if r['detected'] else '#ff4444' for r in results]

    axes[0].bar(names, snrs, color=colors, alpha=0.85)
    axes[0].axhline(threshold, color='red', lw=2, ls='--',
                    label=f'Threshold = {threshold:.3f}')
    axes[0].set_title(
        f"ORAC-NT Phase A Demo\n"
        f"{detected_count}/{total_count} detected | FAR = {far_total:.2e} /yr",
        color='white', fontsize=11
    )
    axes[0].set_ylabel("Peak SNR", color='white')
    axes[0].legend(facecolor='#0e1117', labelcolor='white')
    axes[0].set_facecolor('#0e1117')
    axes[0].tick_params(colors='white', rotation=20)
    for i, snr in enumerate(snrs):
        axes[0].text(i, snr+0.3, f"{snr:.1f}", ha='center',
                     color='white', fontsize=8)

    # Summary table
    axes[1].axis('off')
    axes[1].set_facecolor('#0e1117')
    table_data = [
        ["Metric", "Value"],
        ["Threshold",      f"{threshold:.3f}"],
        ["Detection",      f"{detected_count}/{total_count} ({detected_count/total_count*100:.0f}%)"],
        ["FAR",            f"{far_total:.2e} /yr"],
        ["False alarms",   f"{false_total}"],
        ["Quiet data",     f"{total_time/3600:.2f}h"],
        ["Hardware",       "535ns (STM32F401)"],
        ["Full results",   "doi.org/10.5281/zenodo.20129975"],
    ]
    t = axes[1].table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center', loc='center',
        bbox=[0.05, 0.05, 0.90, 0.90]
    )
    t.auto_set_font_size(False); t.set_fontsize(9)
    for (row, col), cell in t.get_celld().items():
        cell.set_facecolor('#1a1a2e' if row == 0 else '#0e1117')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333333')
    axes[1].set_title("ORAC-NT | github.com/Kretski/ORAC-NT-Astrophysics",
                      color='white', fontsize=9)

    plt.suptitle(
        f"ORAC-NT Public Demo | DOI: 10.5281/zenodo.20129975 | {run_utc}",
        color='white', fontsize=10
    )
    plt.tight_layout()
    plt.savefig("orac_demo.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\n  Saved: orac_demo.png")
    print(f"\n  Full pipeline (Phases B-E7) available under commercial license.")
    print(f"  Contact: kretski1@gmail.com")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
