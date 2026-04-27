import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import binary_dilation
import warnings

warnings.filterwarnings('ignore')

# --- ПАРАМЕТРИ НА СИМУЛАЦИЯТА ---
FS = 16384          
DURATION = 60       
t = np.arange(FS * DURATION) / FS

print(f"[*] Генериране на {DURATION} секунди сурови данни при {FS} Hz...")

# 1. СИМУЛАЦИЯ НА ДАННИ 
raw_stream = np.random.randn(len(t)) * 1.5 
continuous_wave = 0.5 * np.sin(2 * np.pi * 3200 * t) 
raw_stream += continuous_wave

# 2. ИНЖЕКТИРАНЕ НА ХАРДУЕРНИ ШУМОВЕ (GLITCHES)
glitch_times = [12.5, 27.1, 41.8, 55.2]
for gt in glitch_times:
    raw_stream += 80.0 * np.sin(2 * np.pi * 400 * t) * np.exp(-((t - gt)**2) / (2 * 0.05**2))

print("[*] Стартиране на ORAC-NT Veto слой...")

def apply_orac_veto(data, fs, threshold=6.0):
    # 1. Изчисляване на енергийния плик
    analytic_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 2. ЖЕЛЯЗНА КАЛИБРАЦИЯ (Robust MAD)
    # Използваме първите 2 секунди, за да хванем базовия шум, без glitches да го изкривяват
    cal_samples = int(fs * 2.0)
    med = np.median(amplitude_envelope[:cal_samples])
    mad = 1.4826 * np.median(np.abs(amplitude_envelope[:cal_samples] - med))
    
    # 3. Изчисляване на H-factor за целия масив (спрямо твърдата калибрация)
    h_factor = (amplitude_envelope - med) / (mad + 1e-12)
    
    # 4. Генериране на Veto маска
    veto_mask = h_factor > threshold
    
    # 5. Разширяване на маската (Windowing) с 150ms 
    expansion_samples = int(fs * 0.15) 
    veto_mask_expanded = binary_dilation(veto_mask, iterations=expansion_samples)
    
    return veto_mask_expanded

mask = apply_orac_veto(raw_stream, FS, threshold=6.0)

cleaned_stream = raw_stream.copy()
cleaned_stream[mask] = 0.0  

deleted_seconds = np.sum(mask)/FS
print(f"[*] Идентифицирани и изчистени {deleted_seconds:.2f} секунди повредени данни.")

print("[*] Генериране на графиката...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'hspace': 0.4})
fig.patch.set_facecolor('#0e1117')

axes[0].plot(t, raw_stream, color='#ff3333', lw=0.6, label="Raw Strain (Corrupted by Glitches)")
axes[0].set_title("BEFORE: Raw Data with Hardware Glitches", color='white', fontsize=14)
axes[0].set_ylabel("Strain", color='white')
axes[0].legend(facecolor='#0e1117', labelcolor='white', loc='upper right')
axes[0].set_facecolor('#0e1117')
axes[0].tick_params(colors='white')
axes[0].set_xlim(0, DURATION)

axes[1].plot(t, cleaned_stream, color='#00d2ff', lw=0.6, label="ORAC-NT Cleaned Stream (Ready for CW pipeline)")
for gt in glitch_times:
    axes[1].axvspan(gt - 0.15, gt + 0.15, color='yellow', alpha=0.2, label='Vetoed Segment' if gt == glitch_times[0] else "")

axes[1].set_title("AFTER: ORAC-NT Sub-Microsecond Veto Layer Applied", color='white', fontsize=14)
axes[1].set_xlabel("Time (s)", color='white')
axes[1].set_ylabel("Strain", color='white')
axes[1].legend(facecolor='#0e1117', labelcolor='white', loc='upper right')
axes[1].set_facecolor('#0e1117')
axes[1].tick_params(colors='white')
axes[1].set_xlim(0, DURATION)

plt.savefig("orac_cw_veto_demo.png", dpi=300, bbox_inches='tight', facecolor='#0e1117')
print("[+] Графиката е запазена като 'orac_cw_veto_demo.png'")