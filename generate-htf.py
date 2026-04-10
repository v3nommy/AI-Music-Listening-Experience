```python
import os, json, math, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal

# Optional (preferred) WAV reading:
try:
    import soundfile as sf
    HAS_SF = True
except Exception:
    HAS_SF = False
    from scipy.io import wavfile

# -------------------------
# CONFIG (HTF v2 standard)
# -------------------------
SR_TARGET = 22050
HOP = 512
N_FFT = 2048
WINDOW_S = 10
CHROMA_BIN_S = 2

ENERGY_THRESH = {"low": 0.05, "high": 0.10}
BRIGHT_THRESH = {"dark": 1500.0, "bright": 2200.0}

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)

MAJOR_PROFILE /= MAJOR_PROFILE.sum()
MINOR_PROFILE /= MINOR_PROFILE.sum()

def tier_energy(x):
    if x < ENERGY_THRESH["low"]:
        return "low"
    if x <= ENERGY_THRESH["high"]:
        return "medium"
    return "high"

def tier_brightness(hz):
    if hz < BRIGHT_THRESH["dark"]:
        return "dark"
    if hz <= BRIGHT_THRESH["bright"]:
        return "moderate"
    return "bright"

def load_audio_mono(path):
    \"\"\"Load audio (any SR), convert to mono float32 in [-1,1] if possible.\"\"\"
    if HAS_SF:
        y, sr = sf.read(path, always_2d=True)
        y = y.mean(axis=1).astype(np.float32)
        # If integer-like, soundfile already gives float; keep as-is
        return y, int(sr)
    else:
        sr, y = wavfile.read(path)
        y = y.astype(np.float32)
        # Normalize common integer PCM formats
        if y.dtype != np.float32:
            pass
        if y.ndim == 2:
            y = y.mean(axis=1)
        # Heuristic normalization if looks like int range
        max_abs = np.max(np.abs(y)) + 1e-9
        if max_abs > 1.5:
            y = y / max_abs
        return y.astype(np.float32), int(sr)

def resample_to_22050(y, sr):
    if sr == SR_TARGET:
        return y, sr
    # polyphase resample for quality/stability
    g = math.gcd(sr, SR_TARGET)
    up = SR_TARGET // g
    down = sr // g
    y_rs = signal.resample_poly(y, up, down).astype(np.float32)
    return y_rs, SR_TARGET

def stft_mag(y, sr):
    # SciPy STFT (consistent with N_FFT/HOP)
    f, t, Zxx = signal.stft(
        y, fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False
    )
    mag = np.abs(Zxx).astype(np.float32)  # shape: (freq_bins, frames)
    return f, t, mag

def frame_rms_from_mag(mag):
    # Approx RMS from magnitude spectrum energy (not exact window correction; adequate proxy)
    # Use mean magnitude squared across freq bins
    p = np.mean(mag**2, axis=0)
    return np.sqrt(p + 1e-12)

def spectral_centroid(f, mag):
    # centroid per frame
    num = np.sum((f[:, None] * mag), axis=0)
    den = np.sum(mag, axis=0) + 1e-12
    return num / den

def spectral_flux(mag):
    # positive differences between successive frames
    d = np.diff(mag, axis=1)
    d = np.maximum(d, 0.0)
    flux = np.sum(d, axis=0)
    # pad to same length as frames
    flux = np.concatenate([[0.0], flux])
    return flux

def smooth_1d(x, win=5):
    if win <= 1:
        return x
    w = np.ones(win, dtype=float) / win
    return np.convolve(x, w, mode="same")

def onset_envelope_from_flux(flux):
    # normalize and smooth
    x = flux.astype(float)
    x = x - np.min(x)
    x = x / (np.max(x) + 1e-9)
    x = smooth_1d(x, win=7)
    return x

def estimate_tempo_autocorr(onset_env, frame_rate, bpm_min=40, bpm_max=240):
    # autocorrelation
    x = onset_env - np.mean(onset_env)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac)//2:]  # non-negative lags
    # lag range
    lag_min = int(frame_rate * 60.0 / bpm_max)
    lag_max = int(frame_rate * 60.0 / bpm_min)
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, len(ac)-1)

    segment = ac[lag_min:lag_max]
    if len(segment) <= 0:
        return 120.0, [120.0, 240.0, 60.0], None

    best_rel = int(np.argmax(segment))
    best_lag = lag_min + best_rel
    tempo = 60.0 * frame_rate / best_lag

    cands = [tempo, tempo*2.0, tempo/2.0]
    # crude confidence: peak / mean of segment
    conf = float(segment[best_rel] / (np.mean(segment) + 1e-9))
    return float(tempo), [float(c) for c in cands], float(conf)

def first_strong_onset_time(onset_env, t_frames):
    # pick first onset peak above mean+std
    mu = float(np.mean(onset_env))
    sd = float(np.std(onset_env))
    thr = mu + sd
    for i in range(1, len(onset_env)-1):
        if onset_env[i] > thr and onset_env[i] > onset_env[i-1] and onset_env[i] >= onset_env[i+1]:
            return float(t_frames[i])
    # fallback
    return float(t_frames[0]) if len(t_frames) else 0.0

def build_beat_times(beat0, tempo_bpm, duration_s):
    if tempo_bpm <= 0:
        return []
    period = 60.0 / tempo_bpm
    times = []
    k = 0
    while True:
        t = beat0 + k*period
        if t > duration_s:
            break
        times.append(float(t))
        k += 1
    return times

def bar_times_every_4_beats(beat_times):
    return beat_times[::4] if len(beat_times) >= 4 else []

def pitch_class_from_freq(freq):
    if freq <= 0:
        return None
    midi = 69.0 + 12.0 * math.log2(freq / 440.0)
    pc = int(round(midi)) % 12
    return pc

def chroma_from_mag(f, mag):
    # mag: (freq_bins, frames)
    pcs = np.zeros((12, mag.shape[1]), dtype=np.float32)
    for bi, freq in enumerate(f):
        if freq < 50:  # ignore very low bins (rumble/DC)
            continue
        pc = pitch_class_from_freq(float(freq))
        if pc is None:
            continue
        pcs[pc, :] += mag[bi, :]
    return pcs

def normalize_chroma(v):
    s = float(np.sum(v))
    if s <= 0:
        return v
    return v / s

def chroma_bins(chroma, t_frames, duration_s, bin_s=2):
    n_bins = int(math.ceil(duration_s / bin_s))
    out = []
    for b in range(n_bins):
        start = b * bin_s
        end = min((b + 1) * bin_s, duration_s)
        mask = (t_frames >= start) & (t_frames < end)
        if np.any(mask):
            v = np.mean(chroma[:, mask], axis=1)
        else:
            v = out[-1]["chroma"] if out else np.zeros(12, dtype=float)
        v = normalize_chroma(np.array(v, dtype=float))
        out.append({"start": float(round(start, 3)), "end": float(round(end, 3)), "chroma": [float(x) for x in v]})
    return out

def estimate_key_from_chroma(chroma_mean):
    cm = normalize_chroma(np.array(chroma_mean, dtype=float))
    best = (-1.0, None, None)
    for i in range(12):
        score = float(np.dot(cm, np.roll(MAJOR_PROFILE, i)))
        if score > best[0]:
            best = (score, i, "major")
    for i in range(12):
        score = float(np.dot(cm, np.roll(MINOR_PROFILE, i)))
        if score > best[0]:
            best = (score, i, "minor")
    idx, mode = best[1], best[2]
    return f"{NOTE_NAMES[idx]} {mode}", "Krumhansl-Schmuckler on mean chroma (STFT pitch-class mapping)"

def agg_to_1hz(values, times, duration_s):
    n = int(math.ceil(duration_s))
    out = np.zeros(n, dtype=float)
    for i in range(n):
        start = i
        end = i + 1
        mask = (times >= start) & (times < end)
        if np.any(mask):
            out[i] = float(np.mean(values[mask]))
        else:
            out[i] = float(out[i-1] if i > 0 else float(values[0] if len(values) else 0.0))
    return out.tolist()

def peak_events(t_frames, values, kind, min_gap_s=12.0, top_k=12):
    vals = np.array(values, dtype=float)
    peaks = []
    for i in range(1, len(vals)-1):
        if vals[i] > vals[i-1] and vals[i] >= vals[i+1]:
            peaks.append(i)
    peaks = sorted(peaks, key=lambda i: vals[i], reverse=True)
    selected = []
    for i in peaks:
        t = float(t_frames[i])
        if all(abs(t - e["t_s"]) >= min_gap_s for e in selected):
            selected.append({"t_s": float(round(t, 3)), "kind": kind, "strength": float(round(float(vals[i]), 6))})
        if len(selected) >= top_k:
            break
    selected.sort(key=lambda e: e["t_s"])
    return selected

def build_interpretive_map(energy_1hz, bright_1hz, flux_1hz, onset_1hz, window_s=10):
    N = len(energy_1hz)
    windows = []
    for start in range(0, N, window_s):
        end = min(start + window_s, N)
        e = float(np.mean(energy_1hz[start:end]))
        b = float(np.mean(bright_1hz[start:end]))
        f = float(np.mean(flux_1hz[start:end]))
        o = float(np.mean(onset_1hz[start:end])) if onset_1hz is not None else None

        w = {
            "start": int(start),
            "end": int(end),
            "energy_avg": float(round(e, 6)),
            "energy_tier": tier_energy(e),
            "brightness_avg_hz": float(round(b, 3)),
            "brightness_tier": tier_brightness(b),
            "flux_avg": float(round(f, 6)),
        }
        if o is not None:
            w["onset_avg"] = float(round(o, 6))
        windows.append(w)

    intro = windows[0] if windows else None
    peak_e = max(windows, key=lambda x: x["energy_avg"]) if windows else None
    peak_b = max(windows, key=lambda x: x["brightness_avg_hz"]) if windows else None

    summary = []
    if intro:
        summary.append(f"Intro {intro['start']}-{intro['end']}s: energy={intro['energy_tier']} ({intro['energy_avg']}), brightness={intro['brightness_tier']} ({intro['brightness_avg_hz']} Hz).")
    if peak_e:
        summary.append(f"Peak energy {peak_e['start']}-{peak_e['end']}s: {peak_e['energy_avg']} ({peak_e['energy_tier']}).")
    if peak_b:
        summary.append(f"Peak brightness {peak_b['start']}-{peak_b['end']}s: {peak_b['brightness_avg_hz']} Hz ({peak_b['brightness_tier']}).")
    summary.append("Tier thresholds are stored in interpretive_map.thresholds for reproducibility.")

    return {
        "window_s": int(window_s),
        "thresholds": {
            "energy_rms": {"low_lt": ENERGY_THRESH["low"], "medium_le": ENERGY_THRESH["high"], "high_gt": ENERGY_THRESH["high"]},
            "brightness_hz": {"dark_lt": BRIGHT_THRESH["dark"], "moderate_le": BRIGHT_THRESH["bright"], "bright_gt": BRIGHT_THRESH["bright"]},
        },
        "windows": windows,
        "summary_text": " ".join(summary),
    }

# --- Mel filterbank (for mel spectrogram graph) ---
def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz/700.0)
def mel_to_hz(m): return 700.0 * (10.0**(m/2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=8000.0):
    n_freqs = n_fft//2 + 1
    freqs = np.linspace(0, sr/2, n_freqs)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels+2)
    hz = mel_to_hz(mels)

    fb = np.zeros((n_mels, n_freqs), dtype=float)
    for i in range(n_mels):
        f_left, f_center, f_right = hz[i], hz[i+1], hz[i+2]
        left = (freqs - f_left) / (f_center - f_left + 1e-9)
        right = (f_right - freqs) / (f_right - f_center + 1e-9)
        fb[i, :] = np.maximum(0, np.minimum(left, right))
    return fb

def power_to_db(S, ref=1.0, amin=1e-10):
    S = np.maximum(S, amin)
    return 10.0 * np.log10(S / ref)

def generate_htf_v2(audio_path, out_dir, title="", artist="", slug="song", phases=None):
    os.makedirs(out_dir, exist_ok=True)

    # Load + resample
    y, sr = load_audio_mono(audio_path)
    y, sr = resample_to_22050(y, sr)
    duration_s = float(len(y) / sr)
    frame_dt_s = float(HOP / sr)

    # STFT magnitude
    f, t_frames, mag = stft_mag(y, sr)

    # Frame features
    rms_f = frame_rms_from_mag(mag)
    cent_f = spectral_centroid(f, mag)
    flux_f = spectral_flux(mag)
    onset_env = onset_envelope_from_flux(flux_f)

    # Tempo via autocorr
    frame_rate = sr / HOP
    tempo_bpm, tempo_cands, tempo_conf = estimate_tempo_autocorr(onset_env, frame_rate)
    beat0 = first_strong_onset_time(onset_env, t_frames)

    beat_times = build_beat_times(beat0, tempo_bpm, duration_s)
    bar_times = bar_times_every_4_beats(beat_times)

    # Harmony: pitch-class mapping chroma
    chroma = chroma_from_mag(f, mag)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = normalize_chroma(chroma_mean)
    chroma_bins_2s = chroma_bins(chroma, t_frames, duration_s, bin_s=CHROMA_BIN_S)

    est_key, key_method = estimate_key_from_chroma(chroma_mean)

    # 1Hz aggregation
    energy_1hz = agg_to_1hz(rms_f, t_frames, duration_s)
    bright_1hz = agg_to_1hz(cent_f, t_frames, duration_s)
    flux_1hz = agg_to_1hz(flux_f, t_frames, duration_s)
    onset_1hz = agg_to_1hz(onset_env, t_frames, duration_s)

    t_s = list(range(len(energy_1hz)))

    # Events
    events = []
    events += peak_events(t_frames, onset_env, "onset_peak", min_gap_s=12.0, top_k=12)
    events += peak_events(t_frames, flux_f, "flux_peak", min_gap_s=12.0, top_k=12)

    # Phases
    if phases is None:
        phases = [
            {"label":"build", "start":0.0, "end":min(20.0, duration_s)},
            {"label":"drive", "start":min(20.0, duration_s), "end":min(100.0, duration_s)},
            {"label":"bridge_dip", "start":min(100.0, duration_s), "end":min(130.0, duration_s)},
            {"label":"rebuild_climax", "start":min(130.0, duration_s), "end":min(200.0, duration_s)},
            {"label":"taper", "start":min(200.0, duration_s), "end":duration_s},
        ]

    def mean_over(start_s, end_s, arr):
        a = int(max(0, math.floor(start_s)))
        b = int(min(len(arr), math.ceil(end_s)))
        if b <= a:
            return float(arr[a]) if a < len(arr) else 0.0
        return float(np.mean(arr[a:b]))

    phase_stats = []
    for ph in phases:
        phase_stats.append({
            "label": ph["label"],
            "start": float(round(ph["start"], 3)),
            "end": float(round(ph["end"], 3)),
            "energy_mean": float(round(mean_over(ph["start"], ph["end"], energy_1hz), 6)),
            "brightness_mean_hz": float(round(mean_over(ph["start"], ph["end"], bright_1hz), 3)),
            "flux_mean": float(round(mean_over(ph["start"], ph["end"], flux_1hz), 6)),
            "onset_mean": float(round(mean_over(ph["start"], ph["end"], onset_1hz), 6)),
        })

    interpretive = build_interpretive_map(energy_1hz, bright_1hz, flux_1hz, onset_1hz, window_s=WINDOW_S)

    # ---- Graphs ----
    # 1) Waveform
    plt.figure(figsize=(12,3))
    x = np.arange(len(y)) / sr
    plt.plot(x, y)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    wf_path = os.path.join(out_dir, f"{slug}_waveform.png")
    plt.savefig(wf_path, dpi=200); plt.close()

    # 2) Mel spectrogram (from STFT power + mel filterbank)
    power = (mag**2).astype(float)
    fb = mel_filterbank(sr, N_FFT, n_mels=128, fmin=0.0, fmax=8000.0)
    mel = fb @ power
    mel_db = power_to_db(mel, ref=np.max(mel) + 1e-9)

    plt.figure(figsize=(12,4))
    plt.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        extent=[float(t_frames[0]), float(t_frames[-1]) if len(t_frames) else 0.0, 0, 128]
    )
    plt.title("Mel Spectrogram (dB) — STFT + mel filterbank")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bands")
    plt.colorbar(label="dB")
    plt.tight_layout()
    ms_path = os.path.join(out_dir, f"{slug}_mel_spectrogram.png")
    plt.savefig(ms_path, dpi=200); plt.close()

    # 3) RMS 1Hz
    plt.figure(figsize=(12,3))
    plt.plot(t_s, energy_1hz)
    plt.title("Energy (RMS) — 1Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS")
    plt.tight_layout()
    rms_path = os.path.join(out_dir, f"{slug}_rms_energy.png")
    plt.savefig(rms_path, dpi=200); plt.close()

    # 4) Centroid 1Hz
    plt.figure(figsize=(12,3))
    plt.plot(t_s, bright_1hz)
    plt.title("Brightness (Spectral Centroid) — 1Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.tight_layout()
    sc_path = os.path.join(out_dir, f"{slug}_spectral_centroid.png")
    plt.savefig(sc_path, dpi=200); plt.close()

    # ---- JSON ----
    obj = {
        "meta": {
            "schema_version": "HTF_v2",
            "title": title,
            "artist": artist,
            "source_file": os.path.basename(audio_path),
            "duration_s": float(round(duration_s, 3)),
            "sr_hz": int(sr),
            "hop": int(HOP),
            "n_fft": int(N_FFT),
            "frame_dt_s": float(frame_dt_s),
            "created_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "analysis_notes": (
                "HTF v2 fallback pipeline: soundfile/scipy load+resample, scipy.signal.stft, "
                "RMS/centroid/flux/onset proxy, tempo via autocorr, chroma via pitch-class mapping, "
                "1Hz aggregation, phases+stats, 10s interpretive map."
            ),
            "tempo_bpm": float(round(tempo_bpm, 3)),
            "tempo_confidence": None if tempo_conf is None else float(round(tempo_conf, 6)),
            "tempo_candidates_bpm": [float(round(x, 3)) for x in tempo_cands],
            "beat0_s_est": float(round(beat0, 3)),
            "estimated_key": est_key,
            "key_method": key_method,
        },
        "time_series_1hz": {
            "t_s": t_s,
            "energy_rms": [float(x) for x in energy_1hz],
            "brightness_hz": [float(x) for x in bright_1hz],
            "spectral_flux": [float(x) for x in flux_1hz],
            "onset_strength": [float(x) for x in onset_1hz],
        },
        "rhythm": {
            "tempo_bpm": float(round(tempo_bpm, 3)),
            "beat_times_s": [float(round(x, 3)) for x in beat_times],
            "beats_count": int(len(beat_times)),
            "bar_times_s_every_4_beats": [float(round(x, 3)) for x in bar_times],
            "bars_count": int(len(bar_times)),
            "double_time_bpm": float(round(tempo_bpm*2, 3)),
            "half_time_bpm": float(round(tempo_bpm/2, 3)),
        },
        "harmony": {
            "chroma_mean_12_C_to_B": [float(x) for x in chroma_mean],
            "chroma_bins_2s_C_to_B": chroma_bins_2s,
            "chroma_method": "STFT pitch-class mapping",
        },
        "structure": {
            "phases": [{"label":p["label"], "start": float(round(p["start"],3)), "end": float(round(p["end"],3))} for p in phases],
            "phase_stats": phase_stats,
            "events": events,
        },
        "interpretive_map": interpretive,
    }

    json_path = os.path.join(out_dir, f"flux_song_sensory_object_{slug}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    return {
        "json": json_path,
        "waveform": wf_path,
        "mel": ms_path,
        "rms": rms_path,
        "centroid": sc_path,
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="")
    ap.add_argument("--slug", default="song")
    args = ap.parse_args()

    out = generate_htf_v2(args.audio, args.out_dir, args.title, args.artist, args.slug)
    print("Wrote:", out)
```

