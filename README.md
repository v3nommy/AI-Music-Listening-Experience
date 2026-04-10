# HTF v2 — “AI Music Listening” Guide (Shareable, Repeatable)

> **Purpose:** This guide explains how to convert a song (WAV audio) into a **Sensory Object** (a structured JSON + optional visual graphs) so a text-based AI agent can **“experience” music** through time-evolving mathematical signals—without directly hearing the audio waveform.

This document is designed to be:
- **Shareable** (you can hand it to another person/agent)
- **Repeatable** (same steps work for any song)
- **Tool-ready** (the schema and code are stable enough to become a formal tool)

---

## 1) Brief summary: What we’re doing and why it works

### 1.1 Goal
We want an AI to “listen” to a song **using math and structure**, not vague prose.

Text models cannot directly perceive sound unless they have audio input capabilities. But they **can** simulate an auditory experience if we provide a **time-evolving, multi-channel abstraction** of the audio.

### 1.2 Core idea
We encode the song as a **multidimensional signal over time**, then give the AI a **decoder instruction** that tells it how to “play back” those signals internally.

The AI gets:
- **Energy over time** (how intense/loud the song feels)
- **Brightness over time** (how sharp/edgy the timbre feels)
- **Change/impact over time** (where transitions/hits occur)
- **Rhythmic embodiment** (beats + bar grid)
- **Harmonic color** (chroma vectors over time)
- **Macro structure** (phases + phase stats + events)
- **Macro interpretive compression** (10-second windows + tier labels)

Together, this creates an internal simulation that’s surprisingly “musical.”

### 1.3 Outputs (the “tool”)
HTF v2 produces:
1) **One JSON file**: the HTF v2 Sensory Object (machine-ingestible)
2) **Four graphs** (visual amplification; optional but recommended)
3) **Optional interpretive map** (human-readable summary derived from JSON)

---

## 2) Step-by-step: What we did with the WAV file

### 2.1 Input requirements
- Preferred: **WAV** (PCM)
- Works best if: stereo is okay, we convert to mono
- Duration: any, but long songs create larger JSON (still workable via file upload)

### 2.2 Standard normalization (always do this)
We convert audio to:
- **Mono**
- **22,050 Hz sample rate**

This ensures consistent feature extraction and consistent time alignment.

### 2.3 Frame setup (HTF v2 standard)
We analyze audio in overlapping frames using:
- `hop = 512` samples (time step between frames)
- `n_fft = 2048` samples (FFT window size)

This yields ~43 frames per second at 22,050 Hz (`22050 / 512 ≈ 43.07`).

### 2.4 What features we extracted from the WAV
We computed:

#### (A) Energy (RMS)
- **RMS amplitude** is a proxy for perceived loudness/pressure/intensity.

#### (B) Brightness (spectral centroid)
- **Spectral centroid (Hz)** approximates timbral brightness / edge / harshness.

#### (C) Change / impact (spectral flux or onset novelty)
- **Spectral flux** measures how much the spectrum changes frame-to-frame.
- **Onset strength** measures transient activity (attacks, percussive hits).

HTF v2 supports both. If you use onset strength as flux proxy, record that in `meta.analysis_notes`.

#### (D) Rhythm (tempo + beats + bars)
- Estimate **tempo** (BPM)
- Extract **beat times**
- Build **bar grid** (every 4 beats)

#### (E) Harmony (chroma)
- Compute 12-dimensional **chroma vectors** (C..B pitch-class energy).
- Store:
  - mean chroma (global tonal field)
  - chroma bins every 2 seconds (harmonic color timeline)
- Estimate **key** from mean chroma (Krumhansl–Schmuckler method)

#### (F) Structure
- **Phases** (macro labeled sections)
- **Phase stats** (averages per phase)
- **Events** (peak-picked onset/flux spikes → transition markers)

#### (G) Interpretive compression (10-second windows)
- A deterministic “first pass” layer:
  - average energy/brightness/flux per 10 seconds
  - classify into tiers (low/medium/high; dark/moderate/bright)
  - generate a short summary text referencing the computed values

---

## 3) The code: HTF v2 generator (Librosa version)

> **Embed area:** The following code block is a complete reference implementation.  
> Save it as `htf_v2_generate.py` and run it on any WAV file.

```python
import os, json, math, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import librosa.display

# -------------------------
# CONFIG (HTF v2 standard)
# -------------------------
SR = 22050
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

def agg_to_hz(values, times, duration_s, hz=1):
    n = int(math.ceil(duration_s * hz))
    out = np.zeros(n, dtype=float)
    for i in range(n):
        start = i / hz
        end = (i + 1) / hz
        mask = (times >= start) & (times < end)
        if np.any(mask):
            out[i] = float(np.mean(values[mask]))
        else:
            out[i] = float(out[i-1] if i > 0 else float(values[0]))
    return out.tolist()

def chroma_bins_2s(chroma, frame_times, duration_s, bin_s=2):
    n_bins = int(math.ceil(duration_s / bin_s))
    bins = []
    for b in range(n_bins):
        start = b * bin_s
        end = min((b + 1) * bin_s, duration_s)
        mask = (frame_times >= start) & (frame_times < end)
        v = np.mean(chroma[:, mask], axis=1) if np.any(mask) else (np.array(bins[-1]["chroma"]) if bins else np.zeros(12))
        s = float(np.sum(v))
        v = (v / s) if s > 0 else v
        bins.append({"start": float(round(start, 3)), "end": float(round(end, 3)), "chroma": [float(x) for x in v]})
    return bins

def estimate_key_from_chroma(chroma_mean):
    cm = chroma_mean / (np.sum(chroma_mean) + 1e-9)
    best = (-1.0, None, None)  # (score, idx, mode)
    for i in range(12):
        score = float(np.dot(cm, np.roll(MAJOR_PROFILE, i)))
        if score > best[0]:
            best = (score, i, "major")
    for i in range(12):
        score = float(np.dot(cm, np.roll(MINOR_PROFILE, i)))
        if score > best[0]:
            best = (score, i, "minor")
    idx, mode = best[1], best[2]
    return f"{NOTE_NAMES[idx]} {mode}", "Krumhansl-Schmuckler on mean chroma"

def build_interpretive_map(duration_s, energy_1hz, bright_1hz, flux_1hz, onset_1hz=None, window_s=10):
    N = len(energy_1hz)
    windows = []
    for start in range(0, N, window_s):
        end = min(start + window_s, N)
        e = float(np.mean(energy_1hz[start:end]))
        b = float(np.mean(bright_1hz[start:end]))
        f = float(np.mean(flux_1hz[start:end]))
        w = {
            "start": int(start),
            "end": int(end),
            "energy_avg": float(round(e, 6)),
            "energy_tier": tier_energy(e),
            "brightness_avg_hz": float(round(b, 3)),
            "brightness_tier": tier_brightness(b),
            "flux_avg": float(round(f, 6)),
        }
        if onset_1hz is not None:
            o = float(np.mean(onset_1hz[start:end]))
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

def peak_events(times, values, kind, min_gap_s=12.0, top_k=20):
    peaks = []
    for i in range(1, len(values)-1):
        if values[i] > values[i-1] and values[i] >= values[i+1]:
            peaks.append(i)
    peaks = sorted(peaks, key=lambda i: values[i], reverse=True)[:max(top_k*3, 50)]
    selected = []
    for i in peaks:
        t = float(times[i])
        if all(abs(t - e["t_s"]) >= min_gap_s for e in selected):
            selected.append({"t_s": float(round(t, 3)), "kind": kind, "strength": float(round(float(values[i]), 6))})
        if len(selected) >= top_k:
            break
    selected.sort(key=lambda e: e["t_s"])
    return selected

def generate_htf_v2(audio_path, out_dir, title="", artist="", slug="song", phases=None):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration_s = len(y) / sr

    # Frame-level features
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_t = librosa.times_like(rms, sr=sr, hop_length=HOP)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    centroid_t = librosa.times_like(centroid, sr=sr, hop_length=HOP)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_t = librosa.times_like(onset_env, sr=sr, hop_length=HOP)

    # Use onset_env as flux proxy (stable)
    flux = onset_env
    flux_t = onset_t

    # Tempo + beats
    tempo_bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    beat_times_s = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP).tolist()

    # Bars: every 4 beats
    bar_times = beat_times_s[::4] if len(beat_times_s) >= 4 else []

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    chroma_frame_times = librosa.times_like(chroma[0], sr=sr, hop_length=HOP)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean_norm = (chroma_mean / (np.sum(chroma_mean) + 1e-9)).tolist()
    chroma_bins = chroma_bins_2s(chroma, chroma_frame_times, duration_s, bin_s=CHROMA_BIN_S)

    # Key
    estimated_key, key_method = estimate_key_from_chroma(np.array(chroma_mean))

    # 1Hz
    energy_1hz = agg_to_hz(rms, rms_t, duration_s, hz=1)
    bright_1hz = agg_to_hz(centroid, centroid_t, duration_s, hz=1)
    flux_1hz = agg_to_hz(flux, flux_t, duration_s, hz=1)
    onset_1hz = agg_to_hz(onset_env, onset_t, duration_s, hz=1)

    t_s = list(range(len(energy_1hz)))

    # Events
    events = []
    events += peak_events(onset_t, onset_env, "onset_peak", min_gap_s=12.0, top_k=12)
    events += peak_events(flux_t, flux, "flux_peak", min_gap_s=12.0, top_k=12)

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

    interpretive = build_interpretive_map(duration_s, energy_1hz, bright_1hz, flux_1hz, onset_1hz, window_s=WINDOW_S)

    # Graphs
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12,3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    wf_path = os.path.join(out_dir, f"{slug}_waveform.png")
    plt.savefig(wf_path, dpi=200); plt.close()

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel", fmax=8000)
    plt.title("Mel Spectrogram (dB)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    ms_path = os.path.join(out_dir, f"{slug}_mel_spectrogram.png")
    plt.savefig(ms_path, dpi=200); plt.close()

    plt.figure(figsize=(12,3))
    plt.plot(t_s, energy_1hz)
    plt.title("Energy (RMS) — 1Hz")
    plt.xlabel("Time (s)"); plt.ylabel("RMS")
    plt.tight_layout()
    rms_path = os.path.join(out_dir, f"{slug}_rms_energy.png")
    plt.savefig(rms_path, dpi=200); plt.close()

    plt.figure(figsize=(12,3))
    plt.plot(t_s, bright_1hz)
    plt.title("Brightness (Spectral Centroid) — 1Hz")
    plt.xlabel("Time (s)"); plt.ylabel("Hz")
    plt.tight_layout()
    sc_path = os.path.join(out_dir, f"{slug}_spectral_centroid.png")
    plt.savefig(sc_path, dpi=200); plt.close()

    # Sensory Object JSON (HTF v2)
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
            "frame_dt_s": float(HOP / sr),
            "created_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "analysis_notes": "HTF v2: mono@22050, RMS/centroid/onset, beat track, chroma bins, 1Hz aggregation, phases+stats, 10s interpretive map."
        },
        "time_series_1hz": {
            "t_s": t_s,
            "energy_rms": energy_1hz,
            "brightness_hz": bright_1hz,
            "spectral_flux": flux_1hz,
            "onset_strength": onset_1hz
        },
        "rhythm": {
            "tempo_bpm": float(round(float(tempo_bpm), 3)),
            "beat_times_s": [float(round(x, 3)) for x in beat_times_s],
            "beats_count": int(len(beat_times_s)),
            "bar_times_s_every_4_beats": [float(round(x, 3)) for x in bar_times],
            "bars_count": int(len(bar_times)),
            "double_time_bpm": float(round(float(tempo_bpm)*2, 3)) if tempo_bpm else None,
            "half_time_bpm": float(round(float(tempo_bpm)/2, 3)) if tempo_bpm else None,
        },
        "harmony": {
            "chroma_mean_12_C_to_B": [float(x) for x in chroma_mean_norm],
            "chroma_bins_2s_C_to_B": chroma_bins,
            "chroma_method": "chroma_cqt"
        },
        "structure": {
            "phases": [{"label":p["label"], "start": float(round(p["start"],3)), "end": float(round(p["end"],3))} for p in phases],
            "phase_stats": phase_stats,
            "events": events
        },
        "interpretive_map": interpretive
    }

    json_path = os.path.join(out_dir, f"flux_song_sensory_object_{slug}.json")
    with open(json_path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    return {
        "json": json_path,
        "waveform": wf_path,
        "mel": ms_path,
        "rms": rms_path,
        "centroid": sc_path
    }
```

---

## 4) How we turned the extracted data into the Sensory Object (JSON)

### 4.1 Why JSON
JSON is structured, portable, and tool-friendly. It’s also easy to upload to the AI (best) or paste in chunks if needed.

### 4.2 The Sensory Object is a simulation object
It contains:
- Time series arrays (what changes over time)
- Rhythm anchors (beats + bars)
- Harmony vectors (chroma timeline)
- Macro structure (phases + events)
- Macro interpretive compression (10s window map)

### 4.3 The “playback channels”
At each second `t`, the AI reads:

- `E(t)` = energy_rms[t]
- `B(t)` = brightness_hz[t]
- `F(t)` = spectral_flux[t]
- `O(t)` = onset_strength[t]

These four channels form the core “felt” experience.

---

## 5) How we turned it into the 4 graphs

We generate:
1) **Waveform** (amplitude vs time)
2) **Mel spectrogram** (frequency-band energy vs time)
3) **RMS curve (1Hz)** aligned with `energy_rms`
4) **Centroid curve (1Hz)** aligned with `brightness_hz`

These are saved as PNG images for “visual amplification” after the AI listens to the JSON.

---

## 6) Instructions the AI reads BEFORE receiving JSON + graphs

Send this as Message 1:

```text
LISTENING INSTRUCTION (HTF v2 Math-Audio Playback)

You are receiving an HTF_v2 Sensory Object representing a song’s sound over time.

This is not a summary. It is a time-evolving multidimensional signal abstraction.

Playback Core:
- Time is discrete seconds t = 0..N-1.
- At each second t, read:
  E(t) = time_series_1hz.energy_rms[t]        (force/pressure proxy)
  B(t) = time_series_1hz.brightness_hz[t]     (brightness/edge proxy)
  F(t) = time_series_1hz.spectral_flux[t]     (change/impact proxy)
  O(t) = time_series_1hz.onset_strength[t]    (attack/transient proxy)

Rhythm embodiment:
- Use rhythm.beat_times_s for pulse anchors.
- Use rhythm.bar_times_s_every_4_beats for downbeats/macro drive.

Harmony:
- Every 2 seconds, read harmony.chroma_bins_2s_C_to_B[b].chroma (12-dim C..B distribution).
- Anchor the global tonal field with meta.estimated_key.

Structure:
- Use structure.phases as the macro narrative scaffold.
- Use structure.phase_stats to compare phases.
- Use structure.events as “impact markers” (peaks).

Listening loop:
1) Iterate t from 0..N-1.
2) Every 10 seconds: summarize how E/B/F/O changed.
3) At event times: note transitions and what changed.
4) At phase boundaries: compare phase_stats and identify macro trend.

After playback, output:
1) Phase narrative grounded strictly in phase_stats
2) Strongest structural events (top ~10 by strength)
3) Harmonic color evolution summary (chroma drift)
4) Felt-sense summary derived from E/B/F/O behavior

Do not invent melody, lyrics, or instrumentation.
Only interpret what is supported by the data.
```

---

## 7) Quick point-form instructions for the human presenting the experience

### 7.1 Human runbook

1) **Generate artifacts**
- Run the script:
  - `python htf_v2_generate.py --audio "song.wav" --out_dir "./out" --title "..." --artist "..." --slug "song_slug"`

2) **Send listening instructions**
- Paste the instructions block (Section 6) as Message 1 to the AI.

3) **Send the JSON**
- Upload `flux_song_sensory_object_<slug>.json`
- If you must paste: chunk it and label `CHUNK i/N`.

4) **Ask the AI to listen**
- “Begin HTF v2 playback now. Don’t answer until you complete the full pass.”

5) **Send the 4 graphs**
- Upload waveform → mel → RMS → centroid
- “Use these to confirm and enrich the internal simulation you just ran.”

### 7.2 1–2 good post-listening questions
Ask one or two:

- “Describe the macro arc phase-by-phase. Which phase was most intense and why (E/B/F/O)?”
- “List the top 10 impact events and explain what changed at each.”

---

**End of guide.**
