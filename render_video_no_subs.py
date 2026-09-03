#!/usr/bin/env python3
"""Render the Canary Transcript player as MP4 videos without subtitles.

Reproduces the web demo's animation: a scrolling playhead over the spectrogram
and the currently-sounding syllable highlighted in the token timeline.

Usage:
    python render_video.py                # all samples + combined reel
    python render_video.py sample1        # one sample
"""
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "video"

FPS = 60
W, H = 1280, 720
TAIL_SEC = 0.6  # freeze frames after the audio ends

BG = (15, 17, 23)
SURFACE = (26, 29, 39)
SURFACE2 = (34, 37, 54)
BORDER = (46, 51, 71)
TEXT = (229, 231, 235)
MUTED = (107, 114, 128)
ACCENT = (99, 102, 241)
ACCENT2 = (129, 140, 248)

PALETTE = [
    "#4e9af1", "#f15a5a", "#6abf69", "#f0a500", "#ab6fe0",
    "#4ecdc4", "#ff8c42", "#a8d8ea", "#e88de0", "#c5f0a4",
    "#ff6b9d", "#7b9ea8", "#ffd93d", "#95e1d3", "#f38181",
]

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
F = lambda name, size: ImageFont.truetype(str(FONT_DIR / name), size)

# Card geometry (extended CARD_Y1 to fill vertical space nicely)
CARD_X0, CARD_X1 = 64, W - 64
CARD_Y0, CARD_Y1 = 88, 656
PAD = 22
SPEC_X0 = CARD_X0 + PAD
SPEC_X1 = CARD_X1 - PAD
SPEC_W = SPEC_X1 - SPEC_X0
SPEC_Y0, SPEC_H = 148, 300
SPEC_Y1 = SPEC_Y0 + SPEC_H
TOK_Y0 = SPEC_Y1 + 18
TOK_ROW_H = 28
TOK_ROWS_MAX = 3
PROG_Y = 600

f_label = F("DejaVuSans-Bold.ttf", 13)
f_badge = F("DejaVuSans-Bold.ttf", 14)
f_meta = F("DejaVuSans.ttf", 14)
f_tok = F("DejaVuSansMono-Bold.ttf", 15)
f_time = F("DejaVuSansMono.ttf", 14)
f_cv = F("DejaVuSansMono.ttf", 15)
f_tick = F("DejaVuSans.ttf", 11)


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(fg, bg, a):
    return tuple(int(round(f * a + b * (1 - a))) for f, b in zip(fg, bg))


def text_w(draw, s, font):
    return draw.textbbox((0, 0), s, font=font)[2]


def build_spectrogram(wav_path, width, height):
    """Mel spectrogram rendered straight to an RGB array — no axes, so the
    time axis maps exactly onto the pixel width."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=64, n_mels=192, fmin=250, fmax=12000, power=2.0
    )
    db = librosa.power_to_db(S, ref=np.max, top_db=66)
    norm = np.clip((db + 66) / 66, 0, 1)
    rgb = (colormaps["magma"](norm)[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb[::-1])  # low frequencies at the bottom
    return img.resize((width, height), Image.LANCZOS), len(y) / sr


def layout_tokens(draw, tokens, dur):
    """Place token chips at their syllable midpoint, packed into rows so
    neighbouring chips do not overlap."""
    rows = [[] for _ in range(TOK_ROWS_MAX)]
    placed = []
    for tok in tokens:
        w = text_w(draw, tok["label"], f_tok) + 14
        mid = (tok["onset_s"] + tok["offset_s"]) / 2
        cx = SPEC_X0 + mid / dur * SPEC_W
        x0 = min(max(cx - w / 2, SPEC_X0), SPEC_X1 - w)
        row = next(
            (r for r in range(TOK_ROWS_MAX)
             if all(x0 + w + 5 <= a or x0 >= b + 5 for a, b in rows[r])),
            TOK_ROWS_MAX - 1,
        )
        rows[row].append((x0, x0 + w))
        placed.append({"x0": x0, "w": w, "row": row})
    n_rows = max(p["row"] for p in placed) + 1 if placed else 1
    return placed, n_rows


def draw_static(sample, spec_img, dur, colors, boxes, n_rows):
    """Everything that does not change between frames."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((CARD_X0, 26), "CANARY TRANSCRIPT", font=f_label, fill=ACCENT2)
    d.text((CARD_X0, 46), "Birdsong → phonetic transcript · MFCC + kNN against TIMIT",
           font=f_meta, fill=MUTED)

    d.rounded_rectangle([CARD_X0, CARD_Y0, CARD_X1, CARD_Y1], 14,
                        fill=SURFACE, outline=BORDER, width=1)

    # Card header: bird badge + file metadata
    bx, by = SPEC_X0, CARD_Y0 + 18
    bw = text_w(d, sample["bird_name"], f_badge) + 20
    d.rounded_rectangle([bx, by, bx + bw, by + 26], 6,
                        fill=mix(ACCENT, SURFACE, 0.16), outline=mix(ACCENT, SURFACE, 0.42))
    d.text((bx + 10, by + 5), sample["bird_name"], font=f_badge, fill=ACCENT2)
    meta = f"{sample['filename']}   ·   {sample['n_syllables']} syllables   ·   {dur:.2f} s"
    d.text((bx + bw + 14, by + 6), meta, font=f_meta, fill=MUTED)

    img.paste(spec_img, (SPEC_X0, SPEC_Y0))
    d.rectangle([SPEC_X0 - 1, SPEC_Y0 - 1, SPEC_X1, SPEC_Y1], outline=BORDER, width=1)

    # Frequency ticks drawn over the spectrogram's left edge
    mel_lo, mel_hi = librosa.hz_to_mel(250), librosa.hz_to_mel(12000)
    for hz, lab in ((1000, "1k"), (2000, "2k"), (4000, "4k"), (8000, "8k")):
        frac = (librosa.hz_to_mel(hz) - mel_lo) / (mel_hi - mel_lo)
        y = SPEC_Y1 - frac * SPEC_H
        d.line([SPEC_X0, y, SPEC_X0 + 8, y], fill=(255, 255, 255), width=1)
        d.text((SPEC_X0 + 12, y - 7), lab, font=f_tick, fill=(210, 210, 220))

    # Progress track baseline
    d.rounded_rectangle([SPEC_X0 + 46, PROG_Y + 15, SPEC_X1 - 92, PROG_Y + 19], 2, fill=BORDER)
    return img


def render(sample, out_path):
    audio_path = ROOT / sample["audio"]
    spec_img, dur = build_spectrogram(audio_path, SPEC_W, SPEC_H)
    tokens = sample["tokens"]
    dur = max(dur, max(t["offset_s"] for t in tokens))

    colors = {}
    for t in tokens:
        colors.setdefault(t["label"], hex2rgb(PALETTE[len(colors) % len(PALETTE)]))

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    boxes, n_rows = layout_tokens(probe, tokens, dur)

    base = draw_static(sample, spec_img, dur, colors, boxes, n_rows)

    n_frames = int(round((dur + TAIL_SEC) * FPS))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(audio_path),
         "-filter_complex", f"[1:a]apad=pad_dur={TAIL_SEC}[a]", "-map", "0:v", "-map", "[a]",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
         str(out_path)],
        stdin=subprocess.PIPE,
    )

    for fi in range(n_frames):
        t = fi / FPS
        frame = base.copy()
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        active = next((i for i, tk in enumerate(tokens)
                       if tk["onset_s"] <= t < tk["offset_s"]), None)

        # Syllable brackets on the spectrogram — faint, bright when sounding
        for i, tk in enumerate(tokens):
            col = colors[tk["label"]]
            x0 = SPEC_X0 + tk["onset_s"] / dur * SPEC_W
            x1 = SPEC_X0 + tk["offset_s"] / dur * SPEC_W
            if i == active:
                d.rectangle([x0, SPEC_Y0, x1, SPEC_Y1 - 1], fill=col + (46,),
                            outline=col + (255,), width=2)
            else:
                d.rectangle([x0, SPEC_Y0, x1, SPEC_Y1 - 1], outline=col + (70,), width=1)

        # Playhead with a soft glow
        px = SPEC_X0 + min(t, dur) / dur * SPEC_W
        for k, a in ((3, 26), (2, 60)):
            d.rectangle([px - k, SPEC_Y0, px + k, SPEC_Y1 - 1], fill=(255, 255, 255, a))
        d.rectangle([px - 1, SPEC_Y0, px + 1, SPEC_Y1 - 1], fill=(255, 255, 255, 235))

        # Token chips, each tethered to its syllable on the spectrogram
        for i, (tk, bx) in enumerate(zip(tokens, boxes)):
            col = colors[tk["label"]]
            y0 = TOK_Y0 + bx["row"] * TOK_ROW_H
            on = i == active
            mid_x = SPEC_X0 + (tk["onset_s"] + tk["offset_s"]) / 2 / dur * SPEC_W
            d.line([mid_x, SPEC_Y1 + 1, bx["x0"] + bx["w"] / 2, y0 - (3 if on else 0)],
                   fill=col + (215 if on else 42,), width=2 if on else 1)
            if on:
                y0 -= 3
                d.rounded_rectangle([bx["x0"], y0, bx["x0"] + bx["w"], y0 + 23], 5,
                                    fill=col + (56,), outline=col + (235,), width=1)
                fill = tuple(min(255, int(c * 1.25 + 30)) for c in col) + (255,)
            else:
                d.rounded_rectangle([bx["x0"], y0, bx["x0"] + bx["w"], y0 + 23], 5,
                                    fill=col + (18,), outline=col + (48,), width=1)
                fill = col + (128,)
            d.text((bx["x0"] + 7, y0 + 3), tk["label"], font=f_tok, fill=fill)

        # Transport row: play glyph, progress fill, clock, live CV label
        cx, cy = SPEC_X0 + 17, PROG_Y + 17
        d.ellipse([cx - 17, cy - 17, cx + 17, cy + 17], fill=ACCENT + (255,))
        d.polygon([(cx - 5, cy - 8), (cx - 5, cy + 8), (cx + 8, cy)], fill=(255, 255, 255, 255))
        tx0, tx1 = SPEC_X0 + 46, SPEC_X1 - 92
        d.rounded_rectangle([tx0, PROG_Y + 15, tx0 + (tx1 - tx0) * min(t, dur) / dur, PROG_Y + 19],
                            2, fill=ACCENT + (255,))
        d.text((SPEC_X1 - 84, PROG_Y + 9), f"{min(t, dur):.2f} / {dur:.2f}",
               font=f_time, fill=TEXT + (255,))
        if active is not None:
            tk = tokens[active]
            d.text((SPEC_X0 + 46, PROG_Y - 12),
                   f"{tk['cv_label']}   ·   {tk['dur_ms']:.0f} ms",
                   font=f_cv, fill=colors[tk["label"]] + (255,))

        frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
        ff.stdin.write(frame.tobytes())

    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")
    print(f"  wrote {out_path.relative_to(ROOT)}  ({n_frames} frames, {dur + TAIL_SEC:.2f}s)")


def main():
    samples = json.loads((ROOT / "assets" / "data.json").read_text())
    wanted = sys.argv[1:]
    if wanted:
        samples = [s for s in samples if s["id"] in wanted]
    OUT_DIR.mkdir(exist_ok=True)

    paths = []
    for s in samples:
        print(f"rendering {s['id']} …")
        p = OUT_DIR / f"{s['id']}.mp4"
        render(s, p)
        paths.append(p)

    if len(paths) > 1:
        listing = OUT_DIR / "concat.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in paths))
        combined = OUT_DIR / "all_samples.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", str(listing), "-c", "copy", str(combined)],
                       check=True)
        listing.unlink()
        print(f"  wrote {combined.relative_to(ROOT)}")


if __name__ == "__main__":
    main()