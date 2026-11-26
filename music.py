"""Original, procedurally generated show music (no samples, no licences): celesta/music-box melody, warm pad, soft room reverb.
theme()   ~16 s  opening theme (rising lullaby motif over C - Am - F - G)
stinger() ~5 s   short transition phrase
outro()   ~16 s  slower closing theme that fades out
bed(sec, density, seed)   sparse celesta + pad underneath the narration (the caller ducks it under the voice)"""
import numpy as np
from scipy.signal import fftconvolve

SR = 24000
N = {"C3": 130.81, "G3": 196.0, "A3": 220.0, "F3": 174.61, "E4": 329.63, "D4": 293.66, "C4": 261.63, "G4": 392.0, "A4": 440.0,
     "C5": 523.25, "D5": 587.33, "E5": 659.25, "G5": 783.99, "A5": 880.0, "B4": 493.88, "F4": 349.23, "B3": 246.94, "D3": 146.83,
     "E3": 164.81, "F5": 698.46}
PENTA = ["C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5"]
CHORDS = [["C3", "G3", "E4", "C5"], ["A3", "E4", "C5", "E5"], ["F3", "C4", "A4", "C5"], ["G3", "D4", "B4", "D5"]]   # C, Am, F, G
_rev = None


def reverb(x, wet=0.28):
    global _rev
    if _rev is None:
        rng = np.random.default_rng(7)
        n = int(SR * 2.2); t = np.arange(n) / SR
        _rev = rng.standard_normal(n) * np.exp(-t / 0.55) * (1 - np.exp(-t / 0.02))
        _rev /= np.abs(_rev).sum()
    y = fftconvolve(x, _rev)[: len(x)]
    return (1 - wet) * x + wet * y * (np.abs(x).max() / max(np.abs(y).max(), 1e-9)) * 1.0


def note(f, dur, vel=1.0, tau=1.1):
    n = int(SR * dur); t = np.arange(n) / SR
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / tau) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / (tau * 0.6))
         + 0.12 * np.sin(2 * np.pi * 4.1 * f * t) * np.exp(-t / (tau * 0.2)))
    a = int(SR * 0.004); s[:a] *= np.linspace(0, 1, a)
    return (s * vel).astype(np.float32)


def place(buf, x, at):
    i = int(at * SR)
    if i >= len(buf): return
    e = min(len(buf), i + len(x)); buf[i:e] += x[: e - i]


def pad_chord(freqs, dur, vel=1.0):
    n = int(SR * dur); t = np.arange(n) / SR; s = np.zeros(n)
    for k, f in enumerate(freqs):
        for det in (-0.5, 0.0, 0.6):
            s += np.sin(2 * np.pi * (f + det) * t + det) * (1.0 / (1 + 0.3 * k))
    fa = int(SR * min(2.5, dur / 2.5))
    env = np.ones(n); env[:fa] = np.linspace(0, 1, fa) ** 1.5; env[-fa:] = np.linspace(1, 0, fa) ** 1.5
    return (s / np.abs(s).max() * env * vel).astype(np.float32)


def _norm(x, peak=0.9):
    return (x / max(np.abs(x).max(), 1e-9) * peak).astype(np.float32)


def theme(sec=16.0, slow=1.0):
    q = 0.72 * slow    # one beat
    buf = np.zeros(int(SR * (sec + 3)), dtype=np.float32)
    phrase = [("E4", 1), ("G4", 1), ("A4", 1.5), ("G4", 0.5), ("E4", 1), ("D4", 1), ("C4", 2),
              ("E4", 1), ("G4", 1), ("C5", 1.5), ("A4", 0.5), ("G4", 1), ("E4", 1), ("D4", 0.5), ("E4", 0.5), ("C4", 2.5)]
    t = 0.6
    for nn, b in phrase:
        place(buf, note(N[nn], 3.0, 0.9), t); t += b * q
    for k, ch in enumerate(CHORDS):
        place(buf, pad_chord([N[c] for c in ch], q * 4 + 1.5, 0.35), k * q * 4)
    # soft bell echo of the first notes an octave up, like a music box
    for k, nn in enumerate(["E5", "G5", "A5"]):
        place(buf, note(N[nn], 2.0, 0.25, 0.8), 0.6 + k * q * 1.0 + 0.36)
    buf = reverb(buf, 0.3)[: int(SR * sec)]
    f = int(SR * 2.0); buf[-f:] *= np.linspace(1, 0, f)
    return _norm(buf, 0.85)


def stinger(sec=5.0, variant=0):
    buf = np.zeros(int(SR * (sec + 2)), dtype=np.float32)
    seqs = [["G5", "E5", "D5", "C5"], ["C5", "D5", "E5", "G5"], ["E5", "D5", "C5", "G4"]]
    for k, nn in enumerate(seqs[variant % 3]):
        place(buf, note(N[nn], 2.4, 0.9, 1.0), 0.2 + k * 0.55)
    place(buf, pad_chord([N[c] for c in CHORDS[variant % 4]], sec, 0.3), 0)
    buf = reverb(buf, 0.35)[: int(SR * sec)]
    f = int(SR * 1.5); buf[-f:] *= np.linspace(1, 0, f)
    return _norm(buf, 0.8)


def outro(sec=16.0):
    q = 0.95
    buf = np.zeros(int(SR * (sec + 3)), dtype=np.float32)
    phrase = [("G4", 1), ("E4", 1), ("D4", 1.5), ("E4", 0.5), ("C4", 3),
              ("E4", 1), ("D4", 1), ("C4", 4)]
    t = 0.4
    for nn, b in phrase:
        place(buf, note(N[nn], 3.5, 0.85, 1.4), t); t += b * q
    for k, ch in enumerate([CHORDS[0], CHORDS[2], CHORDS[3], CHORDS[0]]):
        place(buf, pad_chord([N[c] for c in ch], q * 4 + 2.0, 0.35), k * q * 4)
    buf = reverb(buf, 0.35)[: int(SR * sec)]
    f = int(SR * 5.0); buf[-f:] *= np.linspace(1, 0, f) ** 1.3
    return _norm(buf, 0.8)


def bed(sec, density=0.45, seed=1, tempo=0.9):
    """pad chords (C Am F G, 8 s each) with sparse pentatonic celesta notes drifting on a random walk"""
    rng = np.random.default_rng(seed)
    n = int(SR * (sec + 3)); buf = np.zeros(n, dtype=np.float32)
    k = 0; t = 0.0
    while t < sec:
        place(buf, pad_chord([N[c] for c in CHORDS[k % 4]], 10.0, 0.55), t); k += 1; t += 8.0
    idx = 4; t = 1.0; beat = tempo
    while t < sec:
        if rng.random() < density:
            idx = int(np.clip(idx + rng.choice([-2, -1, -1, 1, 1, 2]), 0, len(PENTA) - 1))
            place(buf, note(N[PENTA[idx]], 2.5, rng.uniform(0.35, 0.75), 0.9), t)
        t += beat * rng.choice([1, 1, 2])
    buf = reverb(buf, 0.35)[: int(SR * sec)]
    return _norm(buf, 0.8)


if __name__ == "__main__":
    import soundfile as sf
    for name, x in (("theme", theme()), ("stinger", stinger()), ("outro", outro()), ("bed", bed(30))):
        sf.write(f"samples/music_{name}.wav", x, SR); print(name, round(len(x) / SR, 1), "s")
