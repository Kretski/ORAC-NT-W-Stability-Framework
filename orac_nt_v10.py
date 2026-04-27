"""
ORAC-NT v10 — Global Triangulation Engine + Live GCN Integration
New in v10:
  • Tab 1: Historical analysis (GWOSC archive, v9 pipeline)
  • Tab 2: Live Mode — loads latest_event.json from orac_live.py
            and runs full triangulation automatically
  • Auto-detect: if latest_event.json exists on startup → badge shown
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, correlate
from scipy.signal.windows import tukey

try:
    from gwpy.timeseries import TimeSeries
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False

# =========================================================
# 1. CONSTANTS & DETECTOR POSITIONS (ECEF, meters)
# =========================================================
FS = 4096
C  = 299_792_458.0

H1_POS = np.array([-2161414.9, -3834695.2,  4600350.0])
L1_POS = np.array([-74276.0,   -5496284.0,  3224257.0])
V1_POS = np.array([ 4546374.0,   842989.6,  4378576.9])

EVENTS = {
    "GW150914 (First BH Merger — H1/L1 only)":   (1126259462.4, False),
    "GW170104 (BH Merger ~20M — H1/L1 only)":    (1167559936.6, False),
    "GW170814 (First 3-Detector BH Merger)":      (1186741861.5, True),
    "GW170817 (Neutron Star Merger — all 3)":     (1187008882.4, True),
}

# =========================================================
# 2. SIGNAL CONDITIONING
# =========================================================
def apply_tukey(x, alpha=0.25):   # wider taper → less edge leakage
    return x * tukey(len(x), alpha=alpha)

def bandpass(x, low=20.0, high=500.0):
    nyq = FS / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x)

def whiten(x):
    # nperseg=4*FS → 4× better frequency resolution for PSD
    freqs, psd = welch(x, FS, nperseg=min(4 * FS, len(x) // 2))
    rffreqs    = np.fft.rfftfreq(len(x), 1.0 / FS)
    psd_interp = np.interp(rffreqs, freqs, psd)
    X          = np.fft.rfft(x)
    X_w        = X / np.sqrt(psd_interp + 1e-20)
    return apply_tukey(np.fft.irfft(X_w, n=len(x)))

def condition(raw):
    return whiten(bandpass(apply_tukey(raw)))

# =========================================================
# 3. 2PN EINSTEIN CHIRP TEMPLATE
# =========================================================
def einstein_chirp(m1_msun, m2_msun, f_low=20.0, f_high=500.0, duration=4.0):
    """
    SPA chirp template (time domain, center-aligned).
    We keep merger at the natural peak location — scipy.correlate mode='same'
    with mode='full' trimming gives linear (not circular) correlation,
    so the peak index directly encodes the time delay.
    """
    G     = 6.674e-11
    M_sun = 1.989e30
    m1    = m1_msun * M_sun
    m2    = m2_msun * M_sun
    M     = m1 + m2
    eta   = (m1 * m2) / M**2

    def psi_spa(f):
        v = (np.pi * G * M / C**3 * f) ** (1.0/3.0)
        return (3.0 / (128.0 * eta)) * v**(-5) * (
            1.0
            + (20.0/9.0) * (743.0/336.0 + 11.0/4.0 * eta) * v**2
            - 16.0 * np.pi * v**3
            + 10.0 * (3058673.0/1016064.0
                      + 5429.0/1008.0 * eta
                      + 617.0/144.0 * eta**2) * v**4
        )

    N      = int(duration * FS)
    freqs  = np.fft.rfftfreq(N, 1.0 / FS)
    f_mask = (freqs >= f_low) & (freqs <= f_high)
    H      = np.zeros(len(freqs), dtype=complex)
    f_s    = freqs[f_mask]
    H[f_mask] = f_s**(-7.0/6.0) * np.exp(1j * (psi_spa(f_s) - np.pi/4.0))
    h      = np.fft.irfft(H, n=N)
    h      = apply_tukey(h, alpha=0.1)
    pk     = np.max(np.abs(h))
    return h / pk if pk > 0 else h

# =========================================================
# 4. MATCHED FILTER — linear FFT, template-peak aligned
# =========================================================
def matched_filter_snr(data_white, template):
    """
    Linear matched filter via zero-padded FFT.
    Trimming uses the template envelope peak offset so that
    SNR[i] = matched filter output at time_axis[i] (correct sign & delay).

    MAD normalization: robust to single loud chirp in 8s stream.
    """
    Nd = len(data_white)
    Nt = len(template)

    # Pad to next power of 2 for linear (non-circular) convolution
    Nfft = 1
    while Nfft < Nd + Nt - 1:
        Nfft <<= 1

    D   = np.fft.rfft(data_white, n=Nfft)
    T   = np.fft.rfft(template,   n=Nfft)
    raw = np.abs(np.fft.irfft(D * np.conj(T), n=Nfft))

    # Offset = location of template envelope peak
    # This aligns SNR[i] with the moment the chirp merger passes sample i
    tmpl_peak = int(np.argmax(np.abs(template)))
    snr = raw[tmpl_peak: tmpl_peak + Nd]
    if len(snr) < Nd:
        snr = np.pad(snr, (0, Nd - len(snr)))

    # MAD normalization
    med   = np.median(snr)
    mad   = np.median(np.abs(snr - med))
    sigma = 1.4826 * mad + 1e-20
    return snr / sigma

# =========================================================
# 5. PEAK DETECTION
# =========================================================
def merger_window_indices(time_axis, t0, half_width=2.0):  # ±2s — wider
    lo = int(np.searchsorted(time_axis, t0 - half_width))
    hi = int(np.searchsorted(time_axis, t0 + half_width))
    return max(0, lo), min(len(time_axis) - 1, hi)

def find_causal_peak(snr, anchor_idx, pos1, pos2):
    max_s = int(np.linalg.norm(pos2 - pos1) / C * FS) + 1
    lo    = max(0, anchor_idx - max_s)
    hi    = min(len(snr), anchor_idx + max_s + 1)
    return lo + int(np.argmax(snr[lo:hi]))

# =========================================================
# 6. SKY RING GEOMETRY
# =========================================================
def sky_ring(pos1, pos2, delay_s, n_pts=500):
    baseline = pos2 - pos1
    d        = np.linalg.norm(baseline)
    cos_th   = float(np.clip(C * delay_s / d, -1.0, 1.0))
    theta    = np.arccos(cos_th)
    k        = baseline / d

    perp = np.array([0.0, 0.0, 1.0])
    perp = perp - np.dot(perp, k) * k
    n    = np.linalg.norm(perp)
    if n < 1e-8:
        perp = np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, k) * k
        n    = np.linalg.norm(perp)
    perp  /= n
    perp2  = np.cross(k, perp)

    phi  = np.linspace(0, 2*np.pi, n_pts)
    pts  = (np.cos(theta) * k[:, None]
            + np.sin(theta) * (np.cos(phi) * perp[:, None]
                                + np.sin(phi) * perp2[:, None]))
    pts /= np.linalg.norm(pts, axis=0, keepdims=True)

    ra  = np.arctan2(pts[1], pts[0])
    dec = np.arcsin(np.clip(pts[2], -1.0, 1.0))
    return ra, dec, pts

# =========================================================
# 7. RING INTERSECTION FINDER
# =========================================================
def find_intersection(pts_a, pts_b, pts_c):
    def ds(pts): return pts[:, ::max(1, pts.shape[1]//180)]
    A, B, Cp = ds(pts_a), ds(pts_b), ds(pts_c)

    best_score = np.inf
    best_pt    = None

    for pa in A.T:
        pb = B[:, np.argmax(B.T @ pa)]
        pc = Cp[:, np.argmax(Cp.T @ pa)]
        score = (np.arccos(np.clip(pa @ pb, -1, 1))
               + np.arccos(np.clip(pa @ pc, -1, 1))
               + np.arccos(np.clip(pb @ pc, -1, 1)))
        if score < best_score:
            best_score = score
            avg = pa + pb + pc
            avg /= np.linalg.norm(avg)
            best_pt = avg

    if best_pt is None:
        return None, None, None
    ra  = np.arctan2(best_pt[1], best_pt[0])
    dec = np.arcsin(np.clip(best_pt[2], -1.0, 1.0))
    return ra, dec, float(np.degrees(best_score / 3.0))

# =========================================================
# 8. STREAMLIT UI
# =========================================================
st.set_page_config(layout="wide", page_title="ORAC-NT v10")
st.title("🌍 ORAC-NT v10 | Global Triangulation Engine")
st.markdown(
    "Real LIGO/GWOSC data · **2PN Einstein chirp** · "
    "MAD SNR · 3-ring sky localization · **Live GCN Mode**"
)

# ---- Live event badge ----
import json, os, pathlib
live_file = pathlib.Path("latest_event.json")
live_event = None
if live_file.exists():
    try:
        live_event = json.loads(live_file.read_text())
        st.success(
            f"📡 Live alert loaded: **{live_event['id']}**  "
            f"| BBH {live_event['bbh_prob']*100:.0f}%  "
            f"| BNS {live_event['bns_prob']*100:.0f}%  "
            f"| GPS {live_event['gps_time']}"
        )
    except Exception:
        live_event = None

tab1, tab2 = st.tabs(["📂 Historical (GWOSC)", "📡 Live GCN Alert"])

# =========================================================
# TAB 2 — LIVE MODE
# =========================================================
with tab2:
    st.markdown("### Live GCN Alert Triangulation")
    st.markdown(
        "Run `orac_live.py` in a separate terminal to populate `latest_event.json`. "
        "When a new LIGO/Virgo alert arrives, click **Run Live Triangulation** to analyze it instantly."
    )

    if live_event is None:
        st.warning("No `latest_event.json` found. Start `orac_live.py` and wait for an alert.")
        st.code(
            "# Terminal 1\n"
            "$env:GCN_CLIENT_ID=your_id\n"
            "$env:GCN_CLIENT_SECRET=your_secret\n"
            "python orac_live.py\n\n"
            "# Terminal 2\n"
            "streamlit run orac_nt_v10.py",
            language="powershell"
        )
    else:
        eid  = live_event['id']
        gps  = live_event['gps_time']
        bbhp = live_event['bbh_prob']*100
        bnsp = live_event['bns_prob']*100
        st.info(f"Event **{eid}** at GPS {gps} | BBH {bbhp:.0f}% | BNS {bnsp:.0f}%")
        # Auto-select template based on classification
        if live_event['bns_prob'] > live_event['bbh_prob']:
            live_m1, live_m2 = 1.4, 1.4
            live_v1 = True
            st.markdown("🔬 Auto-selected: **Neutron Star template** (1.4 + 1.4 M☉)")
        else:
            live_m1, live_m2 = 30.5, 25.3
            live_v1 = True
            st.markdown("🔬 Auto-selected: **Black Hole template** (30.5 + 25.3 M☉)")

        if st.button("🚀 Run Live Triangulation", type="primary", key="live_btn"):
            live_t0 = float(live_event['gps_time'])
            st.info(f"Fetching data for GPS {live_t0} ...")

            if not GWPY_AVAILABLE:
                st.error("gwpy not found. Run: pip install gwpy")
                st.stop()

            try:
                with st.spinner("Downloading live event data from GWOSC..."):
                    h1_ts  = TimeSeries.fetch_open_data('H1', live_t0-4, live_t0+4, cache=True).resample(FS)
                    l1_ts  = TimeSeries.fetch_open_data('L1', live_t0-4, live_t0+4, cache=True).resample(FS)
                    h1_raw, l1_raw = h1_ts.value, l1_ts.value
                    time_ax = h1_ts.times.value
                    try:
                        v1_ts  = TimeSeries.fetch_open_data('V1', live_t0-4, live_t0+4, cache=True).resample(FS)
                        v1_raw = v1_ts.value
                        live_v1 = True
                    except Exception:
                        live_v1 = False
                        st.warning("V1 data not available for this event — using H1+L1 only.")

                with st.spinner("Processing..."):
                    h1_w = condition(h1_raw)
                    l1_w = condition(l1_raw)
                    if live_v1:
                        v1_w = condition(v1_raw)

                    G_SI, M_sun_kg = 6.674e-11, 1.989e30
                    M_tot  = (live_m1 + live_m2) * M_sun_kg
                    f_isco = float(np.clip(C**3 / (6**1.5 * np.pi * G_SI * M_tot), 30.0, 1500.0))
                    tmpl   = einstein_chirp(live_m1, live_m2, f_high=f_isco)

                    snr_h = matched_filter_snr(h1_w, tmpl)
                    snr_l = matched_filter_snr(l1_w, tmpl)
                    if live_v1:
                        snr_v = matched_filter_snr(v1_w, tmpl)

                win_lo, win_hi = merger_window_indices(time_ax, live_t0, half_width=2.0)
                pk_h1 = win_lo + int(np.argmax(snr_h[win_lo:win_hi]))
                pk_l1 = find_causal_peak(snr_l, pk_h1, H1_POS, L1_POS)
                if live_v1:
                    pk_v1 = find_causal_peak(snr_v, pk_h1, H1_POS, V1_POS)

                delay_HL = (pk_l1 - pk_h1) / FS
                if live_v1:
                    delay_HV = (pk_v1 - pk_h1) / FS
                    delay_LV = (pk_v1 - pk_l1) / FS

                net_snr = float(np.sqrt(snr_h[pk_h1]**2 + snr_l[pk_l1]**2 +
                                        (snr_v[pk_v1]**2 if live_v1 else 0)))

                ra_HL, dec_HL, pts_HL = sky_ring(H1_POS, L1_POS, delay_HL)
                if live_v1:
                    ra_HV, dec_HV, pts_HV = sky_ring(H1_POS, V1_POS, delay_HV)
                    ra_LV, dec_LV, pts_LV = sky_ring(L1_POS, V1_POS, delay_LV)
                    ra_ix, dec_ix, ang_err = find_intersection(pts_HL, pts_HV, pts_LV)

                # Metrics
                lc1, lc2, lc3, lc4 = st.columns(4)
                lc1.metric("H1 SNR", f"{snr_h[pk_h1]:.1f} σ")
                lc2.metric("L1 SNR", f"{snr_l[pk_l1]:.1f} σ")
                lc3.metric("Network SNR", f"{net_snr:.1f} σ")
                lc4.metric("Δt H1-L1", f"{delay_HL*1000:.2f} ms")

                if live_v1 and ra_ix is not None:
                    st.success(
                        f"📍 Candidate: RA = {float(np.degrees(ra_ix)):.1f}°  "
                        f"Dec = {float(np.degrees(dec_ix)):.1f}°  "
                        f"(residual {ang_err:.2f}°)"
                    )

                # Sky map
                fig_l, ax_l = plt.subplots(1, 1, figsize=(12, 5),
                                            subplot_kw={'projection': 'mollweide'},
                                            facecolor='#0e1117')
                ax_l.scatter(ra_HL, dec_HL, color='lime', s=3, label='H1-L1')
                if live_v1:
                    ax_l.scatter(ra_HV, dec_HV, color='cyan', s=3, label='H1-V1')
                    ax_l.scatter(ra_LV, dec_LV, color='magenta', s=3, label='L1-V1')
                    if ra_ix is not None:
                        ax_l.scatter([ra_ix], [dec_ix], color='white', s=300,
                                     marker='*', zorder=10,
                                     label=f'★ RA={np.degrees(ra_ix):.0f}° Dec={np.degrees(dec_ix):.0f}°')
                ax_l.set_title(f"Live Sky Localization — {live_event['id']}", color='white')
                ax_l.set_facecolor('#050505')
                ax_l.grid(color='white', alpha=0.2)
                ax_l.tick_params(colors='white', labelbottom=False, labelleft=False)
                ax_l.legend(loc='lower right', facecolor='#0e1117',
                            labelcolor='white', markerscale=4, fontsize=8)
                st.pyplot(fig_l)

            except Exception as e:
                st.error(f"Error: {e}")

# =========================================================
# TAB 1 — HISTORICAL (existing pipeline)
# =========================================================
with tab1:
    col_ev, col_m1, col_m2 = st.columns(3)
    with col_ev:
        selected = st.selectbox("Target Event", list(EVENTS.keys()))

    t0, v1_available = EVENTS[selected]
    default_m1 = 1.4  if "170817" in selected else (30.5 if "170814" in selected else 35.0)
    default_m2 = 1.4  if "170817" in selected else (25.3 if "170814" in selected else 30.0)

    with col_m1:
        m1 = st.number_input("Mass 1 (M☉)", 1.0, 100.0, default_m1, 0.1)
    with col_m2:
        m2 = st.number_input("Mass 2 (M☉)", 1.0, 100.0, default_m2, 0.1)

    if not v1_available:
        st.info("ℹ️ Virgo was not operational for this event — running H1 + L1 only (single ring).")

    if st.button("🚀 Run Triangulation", type="primary"):

        if not GWPY_AVAILABLE:
            st.error("gwpy not found. Run: pip install gwpy")
            st.stop()

        # ---- DOWNLOAD ----
        with st.spinner("Downloading from GWOSC..."):
            try:
                h1_ts  = TimeSeries.fetch_open_data('H1', t0-4, t0+4, cache=True).resample(FS)
                l1_ts  = TimeSeries.fetch_open_data('L1', t0-4, t0+4, cache=True).resample(FS)
                h1_raw = h1_ts.value
                l1_raw = l1_ts.value
                time_ax = h1_ts.times.value
                if v1_available:
                    v1_ts  = TimeSeries.fetch_open_data('V1', t0-4, t0+4, cache=True).resample(FS)
                    v1_raw = v1_ts.value
            except Exception as e:
                st.error(f"Download error: {e}")
                st.stop()

        # ---- CONDITION ----
        with st.spinner("Bandpass + Whitening..."):
            h1_w = condition(h1_raw)
            l1_w = condition(l1_raw)
            if v1_available:
                v1_w = condition(v1_raw)

        # ---- TEMPLATE ----
        with st.spinner("2PN template + Matched Filter..."):
            G_SI, M_sun_kg = 6.674e-11, 1.989e30
            M_tot  = (m1 + m2) * M_sun_kg
            f_isco = float(np.clip(C**3 / (6**1.5 * np.pi * G_SI * M_tot), 30.0, 1500.0))
            tmpl   = einstein_chirp(m1, m2, f_high=f_isco)

            snr_h = matched_filter_snr(h1_w, tmpl)
            snr_l = matched_filter_snr(l1_w, tmpl)
            if v1_available:
                snr_v = matched_filter_snr(v1_w, tmpl)

        # ---- PEAK DETECTION (merger window) ----
        win_lo, win_hi = merger_window_indices(time_ax, t0, half_width=1.0)
        # H1 peak: best within merger window
        pk_h1 = win_lo + int(np.argmax(snr_h[win_lo:win_hi]))
        # L1/V1: causal search around H1 peak
        pk_l1 = find_causal_peak(snr_l, pk_h1, H1_POS, L1_POS)
        if v1_available:
            pk_v1 = find_causal_peak(snr_v, pk_h1, H1_POS, V1_POS)

        delay_HL = (pk_l1 - pk_h1) / FS
        if v1_available:
            delay_HV = (pk_v1 - pk_h1) / FS
            delay_LV = (pk_v1 - pk_l1) / FS

        net_snr = np.sqrt(
            snr_h[pk_h1]**2 + snr_l[pk_l1]**2
            + (snr_v[pk_v1]**2 if v1_available else 0)
        )

        # ---- SKY RINGS ----
        ra_HL, dec_HL, pts_HL = sky_ring(H1_POS, L1_POS, delay_HL)
        if v1_available:
            ra_HV, dec_HV, pts_HV = sky_ring(H1_POS, V1_POS, delay_HV)
            ra_LV, dec_LV, pts_LV = sky_ring(L1_POS, V1_POS, delay_LV)
            with st.spinner("Computing ring intersection..."):
                ra_ix, dec_ix, ang_err = find_intersection(pts_HL, pts_HV, pts_LV)

        # =========================================================
        # 9. METRICS
        # =========================================================
        st.markdown("---")
        if v1_available:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("H1 SNR",      f"{snr_h[pk_h1]:.1f} σ")
            c2.metric("L1 SNR",      f"{snr_l[pk_l1]:.1f} σ")
            c3.metric("V1 SNR",      f"{snr_v[pk_v1]:.1f} σ")
            c4.metric("Network SNR", f"{net_snr:.1f} σ")
            c5.metric("Δt H1-L1",   f"{delay_HL*1000:.2f} ms")
            c6.metric("Δt H1-V1",   f"{delay_HV*1000:.2f} ms")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("H1 SNR",    f"{snr_h[pk_h1]:.1f} σ")
            c2.metric("L1 SNR",    f"{snr_l[pk_l1]:.1f} σ")
            c3.metric("Δt H1-L1", f"{delay_HL*1000:.2f} ms")

        if v1_available and ra_ix is not None:
            st.success(
                f"📍 Candidate: RA = {float(np.degrees(ra_ix)):.1f}°  "
                f"Dec = {float(np.degrees(dec_ix)):.1f}°  "
                f"(ring residual {ang_err:.2f}°)"
            )

        # =========================================================
        # 10. PLOTS — clip time axis to [t0-3, t0+3] for clarity
        # =========================================================
        t_lo = t0 - 3.0
        t_hi = t0 + 3.0
        mask = (time_ax >= t_lo) & (time_ax <= t_hi)
        tx   = time_ax[mask] - t0   # center on merger

        fig = plt.figure(figsize=(15, 14), facecolor='#0e1117')

        # Plot 1: Whitened strain (zoomed)
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(tx, h1_w[mask], color='#00d2ff', alpha=0.7, lw=0.5, label='H1 Hanford')
        ax1.plot(tx, l1_w[mask], color='#ff007f', alpha=0.7, lw=0.5, label='L1 Livingston')
        if v1_available:
            ax1.plot(tx, v1_w[mask], color='#39ff14', alpha=0.4, lw=0.5, label='V1 Virgo')
        ax1.axvspan(-1.0, 1.0, color='yellow', alpha=0.07, label='Merger window ±1s')
        ax1.axvline(0, color='white', lw=0.8, ls=':', alpha=0.4, label='GPS t₀')
        ax1.set_xlabel("Time from merger (s)", color='white')
        ax1.set_title("Step 1: Whitened GWOSC Data (zoomed ±3s)", color='white', fontsize=12)
        ax1.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
        ax1.set_facecolor('#0e1117')
        ax1.tick_params(colors='white')

        # Plot 2: SNR (zoomed, merger-centered)
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(tx, snr_h[mask], color='#00d2ff', lw=1.2, label='H1 SNR')
        ax2.plot(tx, snr_l[mask], color='#ff007f', lw=1.2, label='L1 SNR')
        if v1_available:
            ax2.plot(tx, snr_v[mask], color='#39ff14', lw=1.2, label='V1 SNR')

        # Peak markers — convert absolute index to relative time
        t_pk_h1 = time_ax[pk_h1] - t0
        t_pk_l1 = time_ax[pk_l1] - t0
        ax2.axvline(t_pk_h1, color='cyan',    lw=1.5, ls='--',
                    label=f'H1 peak  SNR={snr_h[pk_h1]:.1f}σ  t={t_pk_h1:.2f}s')
        ax2.axvline(t_pk_l1, color='magenta', lw=1.5, ls='--',
                    label=f'L1 peak  Δt={delay_HL*1000:.1f}ms')
        if v1_available:
            t_pk_v1 = time_ax[pk_v1] - t0
            ax2.axvline(t_pk_v1, color='lime', lw=1.5, ls='--',
                        label=f'V1 peak  ΔtHV={delay_HV*1000:.1f}ms')
        ax2.axhline(8, color='yellow', lw=0.8, ls=':', alpha=0.5, label='SNR=8 threshold')
        ax2.axvspan(-1.0, 1.0, color='yellow', alpha=0.06)
        ax2.set_xlabel("Time from merger (s)", color='white')
        ax2.set_title("Step 2: Rolling-Window Normalized SNR (zoomed ±3s)", color='white', fontsize=12)
        ax2.legend(facecolor='#0e1117', labelcolor='white', fontsize=7)
        ax2.set_facecolor('#0e1117')
        ax2.tick_params(colors='white')

        # Plot 3: Sky Map
        ax3 = fig.add_subplot(3, 1, 3, projection='mollweide')
        ax3.scatter(ra_HL, dec_HL, color='lime',    s=3, alpha=0.8, label='H1-L1 Ring')
        if v1_available:
            ax3.scatter(ra_HV, dec_HV, color='cyan',    s=3, alpha=0.8, label='H1-V1 Ring')
            ax3.scatter(ra_LV, dec_LV, color='magenta', s=3, alpha=0.8, label='L1-V1 Ring')
            if ra_ix is not None:
                ax3.scatter([ra_ix], [dec_ix], color='white', s=250, marker='*', zorder=10,
                            label=f'★ RA={np.degrees(ra_ix):.0f}°  Dec={np.degrees(dec_ix):.0f}°')
        ax3.set_title("Step 3: 2D Source Localization (Ring Intersection)", color='white', fontsize=12)
        ax3.set_facecolor('#050505')
        ax3.grid(color='white', alpha=0.2)
        ax3.tick_params(colors='white', labelbottom=False, labelleft=False)
        ax3.legend(loc='lower right', facecolor='#0e1117',
                   labelcolor='white', markerscale=4, fontsize=7)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig)

        with st.expander("Physics notes: Pipeline validation log (v5→v9)"):
            st.markdown(f"""
    **Fix 1 — Template-peak aligned trimming:**  
    v8.4 trimmed with offset=(Nt−1)//2 which assumed flat template — but SPA chirp peaks at ~70% of template duration.  
    v8.5 trims at argmax(|template|) → SNR[i] correctly aligned with time_axis[i], correct delay sign.

    **Fix 2 — Zero-padded linear FFT retained:**  
    Pad to next power-of-2 ≥ Nd+Nt−1 prevents circular wrap-around artifact from v8.3.

    **Fix 3 — MAD normalization retained:**  
    1.4826 × median|x − median(x)| — robust to single chirp, flat SNR baseline.

    **Current results for {selected}:**  
    H1 SNR = {snr_h[pk_h1]:.1f}σ · L1 SNR = {snr_l[pk_l1]:.1f}σ · Network = {net_snr:.1f}σ  
    Δt(H1-L1) = {delay_HL*1000:.2f} ms  
    *(Single 2PN template; full LIGO pipeline uses ~250,000 templates)*
            """)
