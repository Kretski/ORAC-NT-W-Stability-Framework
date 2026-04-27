"""
boyajian_w.py — W-Functional Analysis of KIC 8462852 (Boyajian's Star)
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.19536321
License: CC BY 4.0

W(t) = η · α · D(r,t) − β(t)

Where:
  D(r,t) = normalized flux deviation from baseline
  β(t)   = MAD-based local noise floor
  η, α   = sensitivity scaling parameters

Usage:
  pip install lightkurve matplotlib numpy scipy
  python boyajian_w.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

try:
    import lightkurve as lk
    LK_AVAILABLE = True
except ImportError:
    LK_AVAILABLE = False
    print("Install lightkurve: pip install lightkurve")
    exit(1)

# =========================================================
# 1. PARAMETERS
# =========================================================
ETA   = 1.0    # overall sensitivity
ALPHA = 1.0    # deviation amplification
MAD_WINDOW = 200  # samples for local noise floor β(t)

# Control star — typical quiet Kepler target for comparison
CONTROL_KIC = "KIC 3733346"

# =========================================================
# 2. DOWNLOAD KEPLER DATA
# =========================================================
def download_kepler(target, quarter="all"):
    print(f"Downloading {target} from MAST...")
    search = lk.search_lightcurve(target, mission="Kepler", cadence="long")
    if len(search) == 0:
        raise ValueError(f"No Kepler data found for {target}")
    lc = search.download_all().stitch()
    lc = lc.remove_nans().normalize()
    print(f"  {len(lc.time)} data points loaded.")
    return lc

# =========================================================
# 3. W-FUNCTIONAL
# =========================================================
def compute_w(flux, eta=ETA, alpha=ALPHA, window=MAD_WINDOW):
    """
    W(t) = η · α · D(r,t) − β(t)

    D(r,t): normalized deviation from smoothed baseline
    β(t)  : local MAD noise floor (robust, unaffected by dips)
    """
    flux = np.array(flux, dtype=float)

    # Baseline: smoothed flux (uniform filter)
    baseline = uniform_filter1d(flux, size=window)

    # D(r,t): signed deviation from baseline, normalized
    D = (flux - baseline) / (np.median(np.abs(flux)) + 1e-10)

    # β(t): local MAD in sliding window
    beta = np.zeros(len(flux))
    half = window // 2
    for i in range(len(flux)):
        lo = max(0, i - half)
        hi = min(len(flux), i + half)
        segment = flux[lo:hi]
        med = np.median(segment)
        beta[i] = 1.4826 * np.median(np.abs(segment - med))

    W = eta * alpha * D - beta
    return W, D, baseline, beta

# =========================================================
# 4. DETECTION THRESHOLD
# =========================================================
def mad_threshold(W, sigma=3.0):
    med  = np.median(W)
    mad  = 1.4826 * np.median(np.abs(W - med))
    return med - sigma * mad   # negative excursions = dips

# =========================================================
# 5. MAIN
# =========================================================
def main():
    # --- Load Boyajian's Star ---
    try:
        lc_boy = download_kepler("KIC 8462852")
    except Exception as e:
        print(f"Error loading KIC 8462852: {e}")
        return

    time_boy  = lc_boy.time.value
    flux_boy  = lc_boy.flux.value

    W_boy, D_boy, baseline_boy, beta_boy = compute_w(flux_boy)
    thresh_boy = mad_threshold(W_boy, sigma=3.0)

    # Dip events: W below threshold
    dip_mask = W_boy < thresh_boy
    n_dips   = int(np.sum(dip_mask))
    W_min    = float(np.min(W_boy))

    print(f"\nKIC 8462852 results:")
    print(f"  W minimum       : {W_min:.4f}")
    print(f"  Detection thresh: {thresh_boy:.4f}")
    print(f"  Dip events (3σ) : {n_dips} samples")

    # --- Load control star ---
    try:
        lc_ctrl = download_kepler(CONTROL_KIC)
        time_ctrl = lc_ctrl.time.value
        flux_ctrl = lc_ctrl.flux.value
        W_ctrl, _, _, _ = compute_w(flux_ctrl)
        thresh_ctrl = mad_threshold(W_ctrl, sigma=3.0)
        ctrl_loaded = True
        print(f"\nControl star ({CONTROL_KIC}):")
        print(f"  W minimum       : {float(np.min(W_ctrl)):.4f}")
        print(f"  Detection thresh: {thresh_ctrl:.4f}")
        print(f"  Dip events (3σ) : {int(np.sum(W_ctrl < thresh_ctrl))} samples")
    except Exception as e:
        print(f"Control star not loaded: {e}")
        ctrl_loaded = False

    # =========================================================
    # 6. PLOT
    # =========================================================
    n_rows = 3 if ctrl_loaded else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows),
                              facecolor='#0e1117')
    fig.suptitle("W-Functional Analysis — KIC 8462852 (Boyajian's Star)",
                 color='white', fontsize=13)

    # Panel 1: Raw flux + baseline
    ax1 = axes[0]
    ax1.plot(time_boy, flux_boy,  color='#00d2ff', lw=0.4, alpha=0.8, label='Flux')
    ax1.plot(time_boy, baseline_boy, color='yellow', lw=1.0, alpha=0.7, label='Baseline')
    ax1.set_ylabel("Normalized Flux", color='white')
    ax1.set_title("Step 1: Kepler Photometry (KIC 8462852)", color='white')
    ax1.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
    ax1.set_facecolor('#0e1117')
    ax1.tick_params(colors='white')

    # Panel 2: W(t) for Boyajian's Star
    ax2 = axes[1]
    ax2.plot(time_boy, W_boy, color='#ff007f', lw=0.5, alpha=0.9, label='W(t)')
    ax2.axhline(thresh_boy, color='lime', lw=1.0, ls='--',
                label=f'3σ threshold = {thresh_boy:.3f}')
    ax2.scatter(time_boy[dip_mask], W_boy[dip_mask],
                color='red', s=4, zorder=5, label=f'Dip events ({n_dips})')
    ax2.set_ylabel("W(t)", color='white')
    ax2.set_title(f"Step 2: W-Functional  |  W_min = {W_min:.3f}", color='white')
    ax2.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
    ax2.set_facecolor('#0e1117')
    ax2.tick_params(colors='white')

    # Panel 3: Control star comparison
    if ctrl_loaded:
        ax3 = axes[2]
        ax3.plot(time_ctrl, W_ctrl, color='#39ff14', lw=0.5,
                 alpha=0.8, label=f'W(t) — {CONTROL_KIC}')
        ax3.axhline(thresh_ctrl, color='yellow', lw=1.0, ls='--',
                    label=f'3σ threshold = {thresh_ctrl:.3f}')
        ax3.set_ylabel("W(t)", color='white')
        ax3.set_xlabel("Time (BKJD)", color='white')
        ax3.set_title("Step 3: Control Star Comparison", color='white')
        ax3.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
        ax3.set_facecolor('#0e1117')
        ax3.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig("boyajian_w_result.png", dpi=150,
                bbox_inches='tight', facecolor='#0e1117')
    print("\nSaved: boyajian_w_result.png")
    plt.show()

if __name__ == "__main__":
    main()
