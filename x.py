"""
ORAC-NT Real-Time Trigger & Cascading Handoff (v31 — THE REALITY CHECK)
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.19553825

ORAC-NT acts as a first-line, template-free burst detector.
When H-factor exceeds threshold, it cuts a data window and
hands it off to a matched filter for parameter estimation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal.windows import tukey
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. ORAC-NT TRIGGER NODE
# =========================================================
class ORACTriggerNode:
    def __init__(self, fs=4096):
        self.fs          = fs
        self.h_threshold = 2.5    # Alert threshold
        self.window_size = 2.0    # seconds to hand off to MF
        self.leak_up     = 0.15   # integrator rise per sample above floor
        self.leak_down   = 0.03   # integrator decay per sample below floor
        self.calibration_s = 5.0  # seconds used for noise floor calibration

    def whiten(self, data):
        # Estimate PSD ONLY on the clean noise segment!
        cal_samples = int(self.calibration_s * self.fs)
        noise_segment = data[:cal_samples]
        
        f, psd = signal.welch(noise_segment, self.fs, nperseg=self.fs//2)
        psd_i  = np.interp(np.fft.rfftfreq(len(data), 1/self.fs), f, psd)
        w      = np.fft.irfft(np.fft.rfft(data) / np.sqrt(psd_i + 1e-12), n=len(data))
        return w * tukey(len(w), alpha=0.05)

    def process_stream(self, data_stream):
        print("📡 ORAC-NT listening to live stream...")

        w_stream = self.whiten(data_stream)
        
        # Normalize based on the clean calibration segment
        cal_samples = int(self.calibration_s * self.fs)
        w_stream = w_stream / (np.std(w_stream[:cal_samples]) + 1e-12)
        
        envelope = np.abs(signal.hilbert(w_stream))

        # Noise floor from pre-burst calibration window ONLY
        cal_segment = envelope[:cal_samples]
        med         = np.median(cal_segment)
        mad         = 1.4826 * np.median(np.abs(cal_segment - med))
        noise_floor = med + 3.0 * mad

        print(f"   Noise floor (from first {self.calibration_s}s): {noise_floor:.4f}")

        h             = 0.0
        h_history     = []
        trigger_times = []
        triggered     = False

        for i, val in enumerate(envelope):
            if val > noise_floor:
                h += self.leak_up
            else:
                h -= self.leak_down
            h = float(np.clip(h, 0.0, 5.0))
            h_history.append(h)

            # Fire trigger on first threshold crossing
            if h >= self.h_threshold and not triggered:
                triggered    = True
                trigger_time = i / self.fs
                trigger_times.append(trigger_time)

                print(f"\n🚨 ALERT: Unmodeled burst at T = {trigger_time:.3f}s")
                print(f"   H-factor = {h:.2f}  (threshold = {self.h_threshold})")
                print(f"   Cutting {self.window_size}s window → Matched Filter...")

                # Extract event window
                half      = int(self.window_size / 2 * self.fs)
                start_idx = max(0, i - half)
                end_idx   = min(len(data_stream), i + half)
                snippet   = data_stream[start_idx:end_idx]

                self.export_trigger(trigger_time, snippet, noise_floor)

        return np.array(h_history), trigger_times, noise_floor

    def export_trigger(self, t_trigger, snippet, noise_floor):
        alert = {
            "detector"         : "ORAC-NT Spherical (Mario Schenberg proxy)",
            "trigger_timestamp": datetime.now().isoformat(),
            "event_time_s"     : round(t_trigger, 4),
            "trigger_source"   : "ORAC-NT Phase-Agnostic H-factor",
            "noise_floor"      : round(float(noise_floor), 6),
            "h_threshold"      : self.h_threshold,
            "action"           : "HANDOFF_TO_MATCHED_FILTER",
            "window_samples"   : len(snippet),
            "window_seconds"   : self.window_size,
            "recommendation"   : "Run Bayesian parameter estimation on provided window.",
            "doi"              : "10.5281/zenodo.19553825"
        }
        with open("ORAC_TRIGGER_EVENT.json", "w") as f:
            json.dump(alert, f, indent=4)
        print("   ✅ Saved: ORAC_TRIGGER_EVENT.json")


# =========================================================
# 2. MATCHED FILTER CONFIRMATION (post-handoff) - FIXED
# =========================================================
def matched_filter_confirm(snippet, fs=4096, f0=100, f1=200, duration=0.5):
    """
    Realistic Matched Filter: Properly whitens snippet AND template
    to prevent artificial SNR inflation.
    """
    if len(snippet) < fs // 2:
        return 0.0 # Safety check

    # 1. PSD evaluation on the snippet
    f, psd = signal.welch(snippet, fs, nperseg=fs//2)
    psd_i  = np.interp(np.fft.rfftfreq(len(snippet), 1/fs), f, psd)
    
    # 2. Whiten snippet + Tukey window
    w_snippet = np.fft.irfft(np.fft.rfft(snippet) / np.sqrt(psd_i + 1e-12), n=len(snippet))
    w_snippet *= tukey(len(w_snippet), alpha=0.1)
    
    # 3. Generate wrong template (Chirp vs real Burst)
    t_tmpl = np.linspace(0, duration, int(duration * fs), endpoint=False)
    tmpl   = signal.chirp(t_tmpl, f0=f0, f1=f1, t1=duration, method='linear')
    
    # 4. Whiten the template dynamically
    w_tmpl = np.fft.irfft(np.fft.rfft(tmpl, n=len(snippet)) / np.sqrt(psd_i + 1e-12), n=len(snippet))
    w_tmpl /= np.sqrt(np.sum(w_tmpl**2) + 1e-20)

    # 5. Cross-correlation
    corr = signal.correlate(w_snippet, w_tmpl, mode='same')

    # 6. Realistic SNR calculation
    med = np.median(corr)
    mad = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
    snr = float(np.max(np.abs(corr)) / mad)
    
    return snr


# =========================================================
# 3. SIMULATION
# =========================================================
FS       = 4096
DURATION = 16        # seconds
BURST_T  = 11.5      # inject burst here

t      = np.arange(FS * DURATION) / FS
stream = np.random.randn(len(t))

pre_noise_std = np.std(stream[:int(5.0 * FS)])

burst = (np.sin(2 * np.pi * 150 * t)
         * np.exp(-((t - BURST_T)**2) / (2 * 0.08**2)))
burst_scaled = burst * 4.0 * pre_noise_std / (np.std(burst[burst != 0]) + 1e-20)
stream += burst_scaled

print("="*55)
print("ORAC-NT v31 — Real-Time Trigger Simulation")
print("="*55)
print(f"Stream: {DURATION}s  |  Burst at t={BURST_T}s  |  SNR=4")
print()

engine               = ORACTriggerNode(fs=FS)
h_curve, triggers, nf = engine.process_stream(stream)

# MF confirmation on extracted window
mf_snr = 0.0
if triggers:
    t_tr    = triggers[0]
    half    = int(engine.window_size / 2 * FS)
    idx_tr  = int(t_tr * FS)
    s_lo    = max(0, idx_tr - half)
    s_hi    = min(len(stream), idx_tr + half)
    snippet = stream[s_lo:s_hi]
    mf_snr  = matched_filter_confirm(snippet, fs=FS)
    print(f"\n🔬 Realistic MF SNR on extracted window (wrong template): {mf_snr:.2f}")

# =========================================================
# 4. VISUALIZATION
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                          facecolor='#0e1117', sharex=True)

# Panel 1: Raw stream
axes[0].plot(t, stream, color='#888888', lw=0.4, alpha=0.8,
             label='Raw stream (noise + burst)')
axes[0].axvline(BURST_T, color='yellow', lw=1.5, ls='--', alpha=0.7,
                label=f'Injected burst (t={BURST_T}s)')
if triggers:
    axes[0].axvline(triggers[0], color='red', lw=2, ls='-',
                    label=f'ORAC trigger (t={triggers[0]:.2f}s)')
    axes[0].axvspan(triggers[0] - engine.window_size/2,
                    triggers[0] + engine.window_size/2,
                    color='cyan', alpha=0.12, label='Window → MF')
axes[0].set_title("Live Detector Stream", color='white', fontsize=12)
axes[0].set_ylabel("Strain", color='white')
axes[0].legend(facecolor='#0e1117', labelcolor='white', loc='upper right')
axes[0].set_facecolor('#0e1117')
axes[0].tick_params(colors='white')

# Panel 2: H-factor
axes[1].fill_between(t, h_curve, color='#00ffcc', alpha=0.6,
                      label='H-factor (leaky integrator)')
axes[1].axhline(engine.h_threshold, color='red', lw=1.5, ls='--',
                label=f'Alert threshold = {engine.h_threshold}')
axes[1].axhline(nf, color='yellow', lw=1.0, ls=':', alpha=0.6,
                label=f'Noise floor = {nf:.3f}')
if triggers:
    axes[1].axvline(triggers[0], color='red', lw=2)
axes[1].set_title("ORAC-NT H-factor (Energy Accumulation)",
                   color='white', fontsize=12)
axes[1].set_ylabel("H-Factor Score", color='white')
axes[1].legend(facecolor='#0e1117', labelcolor='white', loc='upper right')
axes[1].set_facecolor('#0e1117')
axes[1].tick_params(colors='white')

# Panel 3: Zoom
if triggers:
    zoom_lo = max(0.0, triggers[0] - 2.0)
    zoom_hi = min(float(DURATION), triggers[0] + 2.0)
    zm = (t >= zoom_lo) & (t <= zoom_hi)
    axes[2].plot(t[zm], stream[zm], color='#00d2ff', lw=0.8,
                 label='Zoomed stream (±2s around trigger)')
    axes[2].axvline(triggers[0], color='red', lw=2, ls='-',
                    label='Trigger point')
    axes[2].axvline(BURST_T, color='yellow', lw=1.5, ls='--',
                    label='True burst center')
    axes[2].set_title(
        f"Zoom: Trigger at {triggers[0]:.3f}s  |  "
        f"True burst at {BURST_T}s  |  "
        f"Realistic MF SNR = {mf_snr:.2f}",
        color='white', fontsize=11)
    axes[2].set_ylabel("Strain", color='white')
    axes[2].legend(facecolor='#0e1117', labelcolor='white', loc='upper right')
    axes[2].set_facecolor('#0e1117')
    axes[2].tick_params(colors='white')

axes[2].set_xlabel("Time (s)", color='white')
plt.suptitle(
    "ORAC-NT v31 — Real-Time Burst Trigger + MF Handoff",
    color='white', fontsize=13)
plt.tight_layout()
plt.savefig("orac_trigger_demo.png", dpi=150, facecolor='#0e1117',
            bbox_inches='tight')
print("\n✅ Saved: orac_trigger_demo.png")
print("✅ Saved: ORAC_TRIGGER_EVENT.json")https://imgur.com/a/7NQptP6