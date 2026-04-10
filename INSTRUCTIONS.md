# HOW TO USE

## 1) Get a `.wav` song file
Start with a `.wav` file of the song you want to use.

Keep the filename simple if possible.

Example:
- `my-song.wav`

---

## 2) Make sure you have Python and the required packages

Install the packages:

```bash
pip install numpy scipy matplotlib soundfile
```

---

## 3) Put the script somewhere easy to run
You should have:
- `generate-htf.py`
- your `.wav` file

They can be in the same folder, or you can point to the full path of the `.wav` file.

---

## 4) Run the script

Example command:

```bash
python generate-htf.py --audio "my-song.wav" --out_dir "./out" --title "My Song" --artist "Artist Name" --slug "my-song"
```

### What the arguments mean
- `--audio` = path to the `.wav` file
- `--out_dir` = folder where the outputs will be saved
- `--title` = song title
- `--artist` = artist name
- `--slug` = short filename-friendly name for the outputs

---

## 5) Check the output files

The script should generate:

- `flux_song_sensory_object_<slug>.json`
- `<slug>_waveform.png`
- `<slug>_mel_spectrogram.png`
- `<slug>_rms_energy.png`
- `<slug>_spectral_centroid.png`

Example:

- `flux_song_sensory_object_my-song.json`
- `my-song_waveform.png`
- `my-song_mel_spectrogram.png`
- `my-song_rms_energy.png`
- `my-song_spectral_centroid.png`

---

## 6) Present the song to the AI

### Message 1: send the listening instructions

Paste this as your first message:

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
- Use structure.events as impact markers.

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

### Message 2: send the JSON file

Send the generated JSON file:

- `flux_song_sensory_object_<slug>.json`

If file upload is available, upload the file directly.

If not, paste the JSON in chunks and label them clearly:

- `CHUNK 1/3`
- `CHUNK 2/3`
- `CHUNK 3/3`

Tell the AI to wait until it has the full JSON before responding.

---

### Message 3: after the AI finishes listening, send the 4 graphs

Send these 4 images:

- waveform
- mel spectrogram
- RMS energy
- spectral centroid

Tell the AI to use the graphs to refine or confirm the internal playback it just completed.

---

## 7) Simple checklist

- Get a `.wav` file
- Run the script
- Find the generated JSON + 4 graphs
- Send the listening instructions
- Send the JSON
- Wait for the AI to listen
- Send the 4 graphs
