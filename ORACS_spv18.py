"""
ORAC-NT Cascading Pipeline (v33 — THE FULL PRESENTATION)
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.19553825

Features:
1. Blind Calibrated Energy Trigger (ORAC-NT)
2. Smart Kurtosis Veto for Glitches
3. Matched Filter Handoff
4. Full 3-Panel Visual Export
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal.windows import tukey
from scipy.stats import kurtosis
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

FS = 4096
DURATION = 16
t = np.arange(FS * DURATION) / FS

# ==============================
# 1. SIGNAL INJECTION
# ==============================
def generate_noise():
    return np.random.randn(len(t))

def gw_signal(t0):
    return 6.0 * np.sin(2*np.pi*150*t) * np.exp(-((t-t0)**2)/(2*0.08**2))

def glitch(t0):
    return 15.0 * np.sin(2*np.pi*400*t) * np.exp(-((t-t0)**2)/(2*0.01**2))

# ==============================
# 2. ORAC-NT TRIGGER NODE
# ==============================
class ORAC_Trigger:
    def __init__(self):
        self.fs = FS
        self.h_threshold = 2.5
        self.calibration_s = 5.0 

    def whiten(self, data):
        cal_samples = int(self.calibration_s * self.fs)
        f, psd = signal.welch(data[:cal_samples], self.fs, nperseg=self.fs//2)
        psd_i = np.interp(np.fft.rfftfreq(len(data), 1/self.fs), f, psd)
        w = np.fft.irfft(np.fft.rfft(data) / np.sqrt(psd_i + 1e-12), n=len(data))
        return w * tukey(len(w), 0.05)

    def scan(self, stream):
        w = self.whiten(stream)
        cal_samples = int(self.calibration_s * self.fs)
        w /= (np.std(w[:cal_samples]) + 1e-12) 

        env = np.abs(signal.hilbert(w))
        
        cal_segment = env[:cal_samples]
        med = np.median(cal_segment)
        mad = 1.4826 * np.median(np.abs(cal_segment - med))
        noise_floor = med + 3.0 * mad

        h = 0.0
        h_history = []
        raw_triggers = []

        for i, val in enumerate(env):
            if i < cal_samples:
                h_history.append(0.0)
                continue

            if val > noise_floor:
                h += 0.15
            else:
                h -= 0.03

            h = np.clip(h, 0.0, 5.0)
            h_history.append(h)

            if h >= self.h_threshold:
                raw_triggers.append(i / self.fs)

        clustered = self.cluster(raw_triggers)
        return clustered, np.array(h_history), noise_floor

    def cluster(self, triggers, dt=0.5):
        if not triggers: return []
        clusters = [[triggers[0]]]
        for tr in triggers[1:]:
            if tr - clusters[-1][-1] < dt:
                clusters[-1].append(tr)
            else:
                clusters.append([tr])
        return [c[0] for c in clusters]

# ==============================
# 3. VETO & HANDOFF LOGIC
# ==============================
def get_snippet(data, t_event, window=2.0):
    idx = int(t_event * FS)
    half = int((window / 2) * FS)
    start = max(0, idx - half)
    end = min(len(data), idx + half)
    return data[start:end]

def is_glitch(snippet):
    if snippet is None or len(snippet) < 100: return True
    k = kurtosis(snippet, fisher=False, bias=False)
    return k > 25.0 

def matched_filter_confirm(snippet):
    if len(snippet) < FS // 2: return 0.0
    f, psd = signal.welch(snippet, FS, nperseg=FS//2)
    psd_i = np.interp(np.fft.rfftfreq(len(snippet), 1/FS), f, psd)
    
    w_snippet = np.fft.irfft(np.fft.rfft(snippet) / np.sqrt(psd_i + 1e-12), n=len(snippet))
    w_snippet *= tukey(len(w_snippet), 0.1)
    
    tt = np.linspace(0, 0.5, int(0.5 * FS), endpoint=False)
    tmpl = signal.chirp(tt, f0=80, f1=250, t1=0.5, method='linear')
    
    w_tmpl = np.fft.irfft(np.fft.rfft(tmpl, n=len(snippet)) / np.sqrt(psd_i + 1e-12), n=len(snippet))
    w_tmpl /= np.sqrt(np.sum(w_tmpl**2) + 1e-20)

    corr = signal.correlate(w_snippet, w_tmpl, mode='same')
    med = np.median(corr)
    mad = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
    return float(np.max(np.abs(corr)) / mad)

# ==============================
# 4. MAIN & VISUALIZATION
# ==============================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 RUNNING THE ULTIMATE TEST (Signal + Glitch)")
    print("="*50)

    # Prepare stream with BOTH artifacts
    stream = generate_noise()
    stream += glitch(6.0)
    stream += gw_signal(11.5)

    # Run ORAC-NT
    engine = ORAC_Trigger()
    triggers, h_curve, nf = engine.scan(stream)
    
    valid_trigger = None
    mf_snr = 0.0

    # Process Triggers
    for tr in triggers:
        snippet = get_snippet(stream, tr)
        print(f"\n   🚨 TRIGGER FIRED at t = {tr:.3f}s")
        
        if is_glitch(snippet):
            print(f"      🛡️ VETO: Instrumental artifact detected. Event rejected.")
        else:
            print(f"      ✂️ HANDOFF: Passing 2.0s window to Matched Filter...")
            mf_snr = matched_filter_confirm(snippet)
            print(f"      🏆 EVENT CONFIRMED! | MF SNR = {mf_snr:.2f}")
            valid_trigger = tr

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor='#0e1117', sharex=True)

    # Panel 1
    axes[0].plot(t, stream, color='#888888', lw=0.4, alpha=0.8, label='Raw stream')
    axes[0].axvline(6.0, color='orange', lw=1.5, ls='--', alpha=0.7, label='Injected Glitch (t=6.0s)')
    axes[0].axvline(11.5, color='yellow', lw=1.5, ls='--', alpha=0.7, label='Injected GW Burst (t=11.5s)')
    
    if valid_trigger:
        axes[0].axvline(valid_trigger, color='red', lw=2, ls='-', label=f'Confirmed Trigger (t={valid_trigger:.2f}s)')
        axes[0].axvspan(valid_trigger - 1.0, valid_trigger + 1.0, color='cyan', alpha=0.12, label='Window → MF')
    
    axes[0].set_title("Live Detector Stream (Contains Artifacts & Real Signals)", color='white', fontsize=12)
    axes[0].set_ylabel("Strain", color='white')
    axes[0].legend(facecolor='#0e1117', labelcolor='white', loc='upper right', fontsize=9)
    axes[0].set_facecolor('#0e1117'); axes[0].tick_params(colors='white')

    # Panel 2
    axes[1].fill_between(t, h_curve, color='#00ffcc', alpha=0.6, label='H-factor (Energy Accumulation)')
    axes[1].axhline(engine.h_threshold, color='red', lw=1.5, ls='--', label=f'Alert threshold = {engine.h_threshold}')
    axes[1].axhline(nf, color='yellow', lw=1.0, ls=':', alpha=0.6, label=f'Noise floor = {nf:.3f}')
    
    # Mark all triggers (Vetoed and Valid)
    for tr in triggers:
        if tr == valid_trigger:
            axes[1].axvline(tr, color='red', lw=2) # Valid
        else:
            axes[1].axvline(tr, color='orange', lw=2, ls='-.') # Vetoed

    axes[1].set_title("ORAC-NT H-factor (Triggering & Veto System)", color='white', fontsize=12)
    axes[1].set_ylabel("H-Factor Score", color='white')
    axes[1].legend(facecolor='#0e1117', labelcolor='white', loc='upper right', fontsize=9)
    axes[1].set_facecolor('#0e1117'); axes[1].tick_params(colors='white')

    # Panel 3
    if valid_trigger:
        zm = (t >= valid_trigger - 2.0) & (t <= valid_trigger + 2.0)
        axes[2].plot(t[zm], stream[zm], color='#00d2ff', lw=0.8, label='Zoomed stream (±2s around confirmed trigger)')
        axes[2].axvline(valid_trigger, color='red', lw=2, ls='-', label='Trigger point')
        axes[2].axvline(11.5, color='yellow', lw=1.5, ls='--', label='True burst center')
        axes[2].set_title(f"Zoom: Confirmed Event at {valid_trigger:.3f}s  |  Realistic MF SNR = {mf_snr:.2f}", color='white', fontsize=11)
    else:
        axes[2].set_title("No valid events confirmed.", color='white')

    axes[2].set_xlabel("Time (s)", color='white')
    axes[2].set_ylabel("Strain", color='white')
    axes[2].legend(facecolor='#0e1117', labelcolor='white', loc='upper right', fontsize=9)
    axes[2].set_facecolor('#0e1117'); axes[2].tick_params(colors='white')

    plt.suptitle("ORAC-NT Cascading Pipeline — Full System Test", color='white', fontsize=13)
    plt.tight_layout()
    plt.savefig("orac_final_presentation.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print("\n✅ Saved visual proof: orac_final_presentation.png")