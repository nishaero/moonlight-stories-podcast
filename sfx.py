"""Procedural sound world for the show: soft ambiences (rain, wind, crickets, water, fire, birds, night) and short cues (splash, drip, owl, bird, bee, purr, tick, whoosh).
All synthesized with numpy (no samples, no licences). Levels are returned at about -12 dB peak; the mixer sets how loud they sit under the voice."""
import re
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 24000


def _bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def _lp(x, f, order=2):
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x)


def _hp(x, f, order=2):
    return sosfilt(butter(order, f, btype="high", fs=SR, output="sos"), x)


def _norm(x, peak=0.25):
    return (x / max(np.abs(x).max(), 1e-9) * peak).astype(np.float32)


_rev = None


def _reverb(x, wet=0.3):
    global _rev
    if _rev is None:
        rng = np.random.default_rng(11); n = int(SR * 1.6); t = np.arange(n) / SR
        _rev = rng.standard_normal(n) * np.exp(-t / 0.4) * (1 - np.exp(-t / 0.015)); _rev /= np.abs(_rev).sum()
    y = fftconvolve(x, _rev)[: len(x)]
    return (1 - wet) * x + wet * y * (np.abs(x).max() / max(np.abs(y).max(), 1e-9))


def _lfo(n, f, depth, seed, base=1.0):
    rng = np.random.default_rng(seed); t = np.arange(n) / SR
    return base - depth + depth * (0.5 + 0.5 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)))


def _fade(x, a=2.0, b=2.0):
    x = x.copy(); fa, fb = int(SR * a), int(SR * b)
    fa = min(fa, len(x) // 2); fb = min(fb, len(x) // 2)
    if fa: x[:fa] *= np.linspace(0, 1, fa)
    if fb: x[-fb:] *= np.linspace(1, 0, fb)
    return x


def _clicks(n, rate, seed, decay=0.01, lo=1500, hi=9000, amp=(0.2, 1.0)):
    rng = np.random.default_rng(seed); out = np.zeros(n)
    for p in np.where(rng.random(n) < rate / SR)[0]:
        L = int(SR * decay * 4); t = np.arange(L) / SR
        h = rng.standard_normal(L) * np.exp(-t / decay) * rng.uniform(*amp)
        e = min(n, p + L); out[p:e] += h[: e - p]
    return _bp(out, lo, hi)


# ------------------------------------------------------------ ambiences
def rain(sec, seed=1):
    n = int(SR * sec); rng = np.random.default_rng(seed)
    base = _bp(rng.standard_normal(n), 900, 6500) * 0.5 * _lfo(n, 0.07, 0.25, seed)
    return _norm(base + _clicks(n, 55, seed + 1, 0.004, 2500, 9000, (0.3, 1.0)) * 1.2)


def wind(sec, seed=2):
    n = int(SR * sec); rng = np.random.default_rng(seed)
    return _norm(_lp(rng.standard_normal(n), 520) * _lfo(n, 0.08, 0.7, seed) + _bp(rng.standard_normal(n), 300, 900) * 0.3 * _lfo(n, 0.05, 0.6, seed + 1))


def crickets(sec, seed=3):
    n = int(SR * sec); t = np.arange(n) / SR; rng = np.random.default_rng(seed); out = np.zeros(n)
    for k in range(4):
        f = rng.uniform(4100, 4900); per = rng.uniform(0.9, 1.4); ph = rng.uniform(0, per)
        gate = (((t + ph) % per) < 0.32).astype(float)
        am = (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * rng.uniform(26, 34) * t))) * gate
        am = _lp(am, 120, 1)
        out += np.sin(2 * np.pi * f * t) * am * rng.uniform(0.5, 1.0)
    return _norm(_lp(out, 6000) * _lfo(n, 0.04, 0.3, seed))


def water(sec, seed=4):
    n = int(SR * sec); rng = np.random.default_rng(seed)
    sw = _lp(rng.standard_normal(n), 900) * _lfo(n, 0.22, 0.65, seed)
    bub = np.zeros(n)
    for p in rng.integers(0, n - SR // 2, int(sec * 0.5)):
        L = int(SR * 0.12); t = np.arange(L) / SR; f = rng.uniform(500, 1200) * (1 + 6 * t)
        bub[p:p + L] += np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.04) * rng.uniform(0.05, 0.2)
    return _norm(sw * 0.5 + bub)


def fire(sec, seed=5):
    n = int(SR * sec); rng = np.random.default_rng(seed)
    rumble = _lp(rng.standard_normal(n), 140) * 0.6 * _lfo(n, 0.3, 0.4, seed)
    return _norm(rumble + _clicks(n, 14, seed + 1, 0.006, 700, 5000, (0.3, 1.0)) * 1.4 + _clicks(n, 40, seed + 2, 0.002, 2000, 8000, (0.05, 0.3)) * 0.6)


def birds(sec, seed=6):
    n = int(SR * sec); rng = np.random.default_rng(seed); out = np.zeros(n); t = 1.0
    while t < sec - 1:
        for k in range(rng.integers(2, 5)):
            c = cue_chirp(rng.uniform(2800, 4600), rng.uniform(0.08, 0.18), rng)
            p = int((t + k * 0.17) * SR); e = min(n, p + len(c)); out[p:e] += c[: e - p] * rng.uniform(0.4, 1.0)
        t += rng.uniform(3, 8)
    return _norm(_reverb(out, 0.3))


def night(sec, seed=7):
    return _norm(crickets(sec, seed) * 1.0 + wind(sec, seed + 1) * 0.5, 0.25)


AMBIENCE = {"rain": rain, "wind": wind, "crickets": crickets, "water": water, "fire": fire, "birds": birds, "night": night}


def ambience(kind, sec, seed=1, fade=2.0):
    f = AMBIENCE.get(kind)
    return _fade(f(sec, seed), fade, fade) if f else np.zeros(int(SR * sec), dtype=np.float32)


# ------------------------------------------------------------ cues
def cue_chirp(f0, dur, rng):
    n = int(SR * dur); t = np.arange(n) / SR
    f = f0 * (1 + 0.35 * np.sin(np.pi * t / dur) + 0.15 * t / dur)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) ** 1.5


def splash(seed=1):
    rng = np.random.default_rng(seed); n = int(SR * 0.9); t = np.arange(n) / SR
    x = _bp(rng.standard_normal(n), 700, 5000) * np.exp(-t / 0.12) * (1 - np.exp(-t / 0.004))
    for k, d in enumerate((0.08, 0.2, 0.33)):
        p = int(d * SR); L = int(SR * 0.1); tt = np.arange(L) / SR
        x[p:p + L] += np.sin(2 * np.pi * np.cumsum(rng.uniform(500, 900) * (1 + 5 * tt)) / SR) * np.exp(-tt / 0.035) * 0.4
    return _norm(_reverb(x, 0.25), 0.5)


def drip(seed=2):
    n = int(SR * 0.7); t = np.arange(n) / SR
    x = np.sin(2 * np.pi * np.cumsum(1700 * np.exp(-t / 0.06) + 500) / SR) * np.exp(-t / 0.05)
    return _norm(_reverb(x, 0.35), 0.45)


def owl(seed=3):
    def hoot(dur, f0, f1):
        n = int(SR * dur); t = np.arange(n) / SR; f = np.linspace(f0, f1, n) + 6 * np.sin(2 * np.pi * 5 * t)
        env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.8
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.25 * np.sin(4 * np.pi * np.cumsum(f) / SR)) * env
    x = np.concatenate([hoot(0.28, 400, 380), np.zeros(int(SR * 0.12)), hoot(0.85, 390, 330)])
    return _norm(_reverb(_lp(x, 1800), 0.4), 0.45)


def bird(seed=4):
    rng = np.random.default_rng(seed); out = np.zeros(int(SR * 1.2))
    for k in range(3):
        c = cue_chirp(rng.uniform(3000, 4200), 0.14, rng); p = int(k * 0.22 * SR); out[p:p + len(c)] += c
    return _norm(_reverb(out, 0.3), 0.35)


def bee(seed=5):
    n = int(SR * 2.2); t = np.arange(n) / SR; f = 190 + 12 * np.sin(2 * np.pi * 3 * t)
    x = (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(4 * np.pi * np.cumsum(f) / SR)) * (0.6 + 0.4 * np.sin(2 * np.pi * 42 * t))
    return _norm(_fade(_lp(x, 1500), 0.5, 0.9), 0.3)


def purr(seed=6):
    rng = np.random.default_rng(seed); n = int(SR * 3.0); t = np.arange(n) / SR
    x = (_lp(rng.standard_normal(n), 400) * 0.7 + np.sin(2 * np.pi * 110 * t) * 0.3) * (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 24 * t)) * 0.5 + 0.0)
    return _norm(_fade(_lp(x, 700), 0.7, 1.2), 0.3)


def tick(seed=7):
    out = np.zeros(int(SR * 2.4)); rng = np.random.default_rng(seed)
    for k in range(4):
        p = int(k * 0.6 * SR); L = int(SR * 0.05); t = np.arange(L) / SR
        c = _bp(rng.standard_normal(L), 900, 3200) * np.exp(-t / 0.008) * (1.0 if k % 2 == 0 else 0.7)
        out[p:p + L] += c
    return _norm(_reverb(out, 0.25), 0.35)


def whoosh(seed=8):
    rng = np.random.default_rng(seed); n = int(SR * 0.9); x = rng.standard_normal(n)
    y = np.zeros(n); blocks = 12
    for b in range(blocks):
        s, e = b * n // blocks, (b + 1) * n // blocks; c = 400 + 2200 * np.sin(np.pi * (b + 0.5) / blocks)
        y[s:e] = _bp(x[s:e], c * 0.7, c * 1.3, 1)
    return _norm(_fade(y, 0.3, 0.4), 0.3)


CUES = {"splash": splash, "drip": drip, "owl": owl, "bird": bird, "bee": bee, "purr": purr, "tick": tick, "whoosh": whoosh}

# ------------------------------------------------------------ what the story asks for
AMB_WORDS = {   # first match wins in order of the list; words in German and English
    "rain": r"regen|pfütze|tropf|rain|puddle|drizzle",
    "water": r"wasser|teich|see\b|meer|fluss|bach|welle|bad\b|badewanne|strand|schaum|water|pond|lake|sea|river|stream|wave|bath|beach|bubble",
    "fire": r"feuer|kamin|ofen|kerze|lagerfeuer|fire|fireplace|candle|stove",
    "wind": r"wind|sturm|schnee|schnee\w*|snow|breeze",
    "birds": r"vogel|vögel|garten|wald|baum|wiese|zwitscher|bird|garden|forest|wood|tree|meadow",
    "night": r"nacht|mond|sterne|stern\b|dunkel|abend|night|moon|stars?\b|dark|evening",
}
CUE_WORDS = {
    "splash": r"plitsch|platsch|splash|plätscher",
    "drip": r"tropf|drip",
    "owl": r"eule|uhu|huhu|owl|hoot",
    "bird": r"zwitscher|piep|tschirp|tweet|chirp",
    "bee": r"summ\b|summt|biene|bumblebee|buzz",
    "purr": r"schnurr|purr",
    "tick": r"tick.?tack|uhr\b|clock|ticktock",
    "whoosh": r"wusch|hui\b|whoosh|swoosh",
}


def ambience_for(text):
    t = text.lower(); scores = {k: len(re.findall(p, t)) for k, p in AMB_WORDS.items()}
    k = max(scores, key=scores.get)
    return k if scores[k] > 0 else None


def cues_for(text):
    t = text.lower()
    return [k for k, p in CUE_WORDS.items() if re.search(p, t)]


if __name__ == "__main__":
    import soundfile as sf
    for k in AMBIENCE: sf.write(f"samples/sfx_amb_{k}.wav", ambience(k, 12), SR)
    for k, f in CUES.items(): sf.write(f"samples/sfx_cue_{k}.wav", f(), SR)
    print("ok")
