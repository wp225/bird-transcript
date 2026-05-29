#!/usr/bin/env python3
"""Build demo assets: find best 2s window, trim audio, generate spectrograms, write data.json."""

import pickle, json, shutil, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
PKL  = Path("/supernova/data/home/george/codeBase/birdtranscript/"
            "notebooks/bpa_pipeline_final/results/bpa_debug/matches_temporal.pkl")
DEMO = Path(__file__).parent
AUDIO_DIR = DEMO / "assets/audio"
SPEC_DIR  = DEMO / "assets/spectrograms"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 2.0

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading pickle…")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with open(PKL, "rb") as f:
        data = pickle.load(f)

recs = data["matched_records"]
by_file = defaultdict(list)
for r in recs:
    by_file[r["audio_file"]].append(r)

# ── Hand-picked source files ───────────────────────────────────────────────────
SELECTED = [
    "/supernova/data/home/george/codeBase/birdtranscript/data/raw/"
    "llb3_data/llb3_songs/llb3_2773_2018_04_29_12_46_03.wav",
    "/supernova/data/home/george/codeBase/birdtranscript/data/raw/"
    "llb3_data/llb3_songs/llb3_2750_2018_04_29_12_14_15.wav",
    "/supernova/data/home/george/codeBase/birdtranscript/data/raw/"
    "llb11_data/llb11_songs/llb11_05137_2018_05_14_13_57_13.wav",
]

BIRD_NAMES = {
    "llb3":  "Canary #3",
    "llb11": "Canary #11",
}

PALETTE = [
    "#4e9af1","#f15a5a","#6abf69","#f0a500","#ab6fe0",
    "#4ecdc4","#ff8c42","#a8d8ea","#e88de0","#c5f0a4",
    "#ff6b9d","#7b9ea8","#ffd93d","#95e1d3","#f38181",
]

def label_color(label, color_map):
    if label not in color_map:
        color_map[label] = PALETTE[len(color_map) % len(PALETTE)]
    return color_map[label]


def best_window(syllables, win=WINDOW_SEC):
    """Return (t_start, t_end) of the 2s window with most diverse syllables."""
    onsets = np.array([r["onset_s"] for r in syllables])
    best_score, best_t = -1, onsets[0]
    for t in onsets:
        mask   = (onsets >= t) & (onsets < t + win)
        in_win = [syllables[i] for i in np.where(mask)[0]]
        labels = [r["bpa_label"] for r in in_win]
        unique = len(set(labels))
        total  = len(labels)
        # runs = label transitions (penalise monotone runs)
        runs = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1]) + 1
        score = unique * 2 + runs - total * 0.2  # favour variety over sheer count
        if score > best_score:
            best_score, best_t = score, t
    t_start = max(0.0, best_t - 0.05)  # tiny pad before first syllable
    return t_start, t_start + win


def trim_audio(src_path, t_start, t_end, dst_path):
    y, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    s0 = int(t_start * sr)
    s1 = min(int(t_end   * sr), len(y))
    sf.write(str(dst_path), y[s0:s1], sr)


def make_spectrogram(audio_path, t_start, t_end, syllables, out_png):
    """Spectrogram of [t_start, t_end] with per-syllable coloured boxes."""
    y, sr = librosa.load(str(audio_path), sr=None, mono=True,
                         offset=t_start, duration=t_end - t_start)

    fig, ax = plt.subplots(figsize=(8, 2.2))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80,
                                          fmax=8000, n_fft=512, hop_length=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=128,
                             x_axis="time", y_axis="mel", ax=ax,
                             cmap="magma", vmin=-60, vmax=0, fmax=8000)

    ymax = ax.get_ylim()[1]
    color_map = {}
    for syl in syllables:
        onset  = syl["onset_s"]  - t_start
        offset = syl["offset_s"] - t_start
        col    = label_color(syl["bpa_label"], color_map)
        rect   = mpatches.FancyBboxPatch(
            (onset, 0), offset - onset, ymax,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor=col, facecolor="none", alpha=0.85,
        )
        ax.add_patch(rect)
        mid = (onset + offset) / 2
        ax.text(mid, ymax * 0.93, syl["bpa_label"],
                color=col, fontsize=6, ha="center", va="top",
                fontweight="bold", fontfamily="monospace")

    ax.set_xlabel("Time (s)", color="#9ca3af", fontsize=7)
    ax.set_ylabel("", visible=False)
    ax.tick_params(colors="#6b7280", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    plt.tight_layout(pad=0.2)
    plt.savefig(str(out_png), dpi=130, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    plt.close()
    print(f"  Spectrogram → {out_png.name}")


demo_data = []

for idx, audio_path in enumerate(SELECTED, 1):
    ap = Path(audio_path)
    if not ap.exists():
        print(f"  SKIP (missing): {ap.name}"); continue

    slug   = f"sample{idx}"
    bird   = by_file[audio_path][0]["bird"]
    syls   = sorted(by_file.get(audio_path, []), key=lambda r: r["onset_s"])

    t_start, t_end = best_window(syls)
    win_syls = [r for r in syls if r["onset_s"] >= t_start and r["offset_s"] <= t_end]
    # If window captured <3 syllables, fall back to all that start in window
    if len(win_syls) < 3:
        win_syls = [r for r in syls if r["onset_s"] >= t_start and r["onset_s"] < t_end]

    print(f"[{idx}] {ap.name}  window=[{t_start:.2f}s, {t_end:.2f}s]  {len(win_syls)} syl")

    # Trim audio
    dest_wav = AUDIO_DIR / f"{slug}.wav"
    trim_audio(ap, t_start, t_end, dest_wav)

    # Spectrogram
    spec_png = SPEC_DIR / f"{slug}.png"
    make_spectrogram(ap, t_start, t_end, win_syls, spec_png)

    # Tokens with times relative to window start
    tokens = [{
        "label":    r["bpa_label"],
        "onset_s":  round(r["onset_s"]  - t_start, 3),
        "offset_s": round(r["offset_s"] - t_start, 3),
        "dur_ms":   round(r["duration_ms"], 1),
        "cv_label": r["cluster_cv"],
    } for r in win_syls]

    demo_data.append({
        "id":          slug,
        "bird":        bird,
        "bird_name":   BIRD_NAMES.get(bird, bird),
        "filename":    ap.name,
        "audio":       f"assets/audio/{slug}.wav",
        "spectrogram": f"assets/spectrograms/{slug}.png",
        "window_sec":  WINDOW_SEC,
        "n_syllables": len(win_syls),
        "transcript":  " ".join(t["label"] for t in tokens),
        "tokens":      tokens,
    })

out_json = DEMO / "assets/data.json"
with open(out_json, "w") as f:
    json.dump(demo_data, f, indent=2, ensure_ascii=False)
print(f"\nWrote {out_json}  ({len(demo_data)} examples)")
