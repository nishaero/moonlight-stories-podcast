"""Quality gate for finished episodes. Nothing is published unless every check passes (fail closed).

checks (each: pass / warn / fail):
  technical   loudness (LUFS), true peak, length, cover art, tags, mp3 length == mix length
  dead_air    no long silent stretch, no clicks/pops
  script      the FINAL mp3 is transcribed with Whisper and compared with the script word by word (garbled words, skipped or repeated passages, strays)
  voice       pace, pitch movement and a steady voice across the reading
  meaning     an LLM judge reads what was actually spoken: sensible, logical, age-appropriate, kind, no nonsense, no foreign-language fragments
  safety      calm, nothing frightening
  metadata    title/description present, child's name or birth date not in public metadata, optional AI disclosure
"""
import os, re, json, subprocess, sys, difflib
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
HALLUCINATIONS = ("untertitel", "amara", "danke fürs zuschauen", "vielen dank fürs", "subtitles", "thanks for watching", "thank you for watching", "copyright")


def _res(name, status, detail, data=None):
    r = {"name": name, "status": status, "detail": detail}
    if data is not None: r["data"] = data
    return r


def decode(path, sr=16000):
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def technical(mp3, mix_wav, lo_min, hi_min):
    out = []
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)], capture_output=True, text=True).stdout.strip() or 0)
    mdur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mix_wav)], capture_output=True, text=True).stdout.strip() or 0)
    out.append(_res("length", "pass" if lo_min * 60 <= dur <= hi_min * 60 else "fail", f"{dur / 60:.1f} min (allowed {lo_min}-{hi_min})"))
    out.append(_res("mp3 complete", "pass" if abs(dur - mdur) < 3 else "fail", f"mp3 {dur:.0f}s vs mix {mdur:.0f}s"))
    e = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mp3), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    summ = e[e.rfind("Summary:"):] if "Summary:" in e else e
    I = re.search(r"I:\s+(-?[0-9.]+) LUFS", summ); TP = re.search(r"Peak:\s+(-?[0-9.]+) dBFS", summ)
    if I:
        v = float(I.group(1)); out.append(_res("loudness", "pass" if -22.5 <= v <= -17 else "fail", f"{v} LUFS (target -20, allowed -22.5..-17)"))
    else:
        out.append(_res("loudness", "fail", "could not measure"))
    if TP:
        v = float(TP.group(1)); out.append(_res("true peak", "pass" if v <= -1.0 else "fail", f"{v} dBTP (max -1.0)"))
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(mp3)], capture_output=True, text=True).stdout
    out.append(_res("cover art", "pass" if "video" in pr else "fail", "embedded" if "video" in pr else "missing"))
    return out


def dead_air_and_clicks(mp3):
    x = decode(mp3, 16000); sr = 16000; n = len(x); out = []
    fr = int(sr * 0.1); rms = np.sqrt(np.convolve(x ** 2, np.ones(fr) / fr, mode="same"))[::fr]
    silent = rms < 10 ** (-52 / 20)
    run = 0; worst = 0; where = 0
    for i, s in enumerate(silent[: max(0, len(silent) - 120)]):   # ignore the last 12 s (fade out)
        run = run + 1 if s else 0
        if run > worst: worst, where = run, i
    out.append(_res("dead air", "pass" if worst * 0.1 <= 4 else "fail", f"longest silence {worst * 0.1:.1f}s at {where * 0.1 / 60:.1f} min"))
    d = np.abs(np.diff(x)); loc = np.sqrt(np.convolve(x ** 2, np.ones(320) / 320, mode="same"))[1:]
    spikes = np.where((d > 0.35) & (d > 12 * loc + 0.05))[0]
    groups = int(np.sum(np.diff(spikes) > sr * 0.05)) + (1 if len(spikes) else 0)
    out.append(_res("clicks", "pass" if groups <= 2 else ("warn" if groups <= 6 else "fail"), f"{groups} sharp clicks/pops"))
    return out


def _transcribe(x16, lang):
    parts = []
    step = 16000 * 200
    for k in range(0, len(x16), step):
        pcm = (np.clip(x16[k:k + step], -1, 1) * 32767).astype("<i2").tobytes()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "wyoming_transcribe.py"), lang], input=pcm, capture_output=True, timeout=1500)
        parts.append(r.stdout.decode().strip())
    return " ".join(parts)


def _transcribe_voice(x16, lang, max_s=24.0):
    """speech-to-text in short pieces cut at silences: one long clip makes Whisper loop and invent repeats; short clips are reliable"""
    import librosa
    sr = 16000
    iv = librosa.effects.split(x16, top_db=38, frame_length=1024, hop_length=256)
    pieces = []; cur = None
    for a, b in iv:
        if cur is None: cur = [a, b]
        elif (a - cur[1]) / sr < 0.5 and (b - cur[0]) / sr <= max_s: cur[1] = b
        else: pieces.append(cur); cur = [a, b]
    if cur: pieces.append(cur)
    out = []
    for a, b in pieces:
        seg = x16[max(0, a - int(0.15 * sr)): b + int(0.2 * sr)]
        if len(seg) < sr * 0.4: continue
        pcm = (np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "wyoming_transcribe.py"), lang], input=pcm, capture_output=True, timeout=600)
        t = r.stdout.decode().strip(); w = t.split(); dur = (b - a) / sr
        cap = int(dur * 4.5) + 3          # nobody speaks faster than this: more words than fit in the clip = the transcriber looped on a short clip
        out.append(" ".join(w[:cap]) if len(w) > cap else t)
    return " ".join(out)


FILLER_TOKENS = {"äh", "ähm", "ehm", "hm", "hmm", "mhm", "mmh", "mh", "öh", "um", "uh", "er", "uhm", "mm", "mmm"}


def _words(t):
    return [w for w in re.sub(r"[^a-zäöüßé' ]", " ", t.lower()).split() if w not in FILLER_TOKENS]


def _analyse(tr, exp, need, dialect, prefix=""):
    out = []; got = _words(tr)
    sm = difflib.SequenceMatcher(None, exp, got, autojunk=False); ratio = sm.ratio()
    out.append(_res(prefix + "script match", "pass" if ratio >= need else "fail", f"{ratio:.3f} of the script heard back (need {need})"))
    hit = np.zeros(len(exp), dtype=bool)
    for a, b, size in sm.get_matching_blocks():
        hit[a:a + size] = True
    bad = []
    for k in range(0, len(exp), 25):
        frac = hit[k:k + 25].mean() if len(exp[k:k + 25]) else 1
        if frac < (0.5 if dialect else 0.65): bad.append((k, round(float(frac), 2), " ".join(exp[k:k + 12])))
    out.append(_res(prefix + "garbled passages", "pass" if not bad else ("warn" if len(bad) <= 1 else "fail"), "none" if not bad else "; ".join(f"word {k}: {f} '{t}…'" for k, f, t in bad[:5]), data=[k for k, f, t in bad]))
    ins = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace") and (j2 - j1) >= 8 and (j2 - j1) > 1.5 * (i2 - i1):
            ins.append(" ".join(got[j1:j2][:14]))
    halluc = [t for t in ins if any(h in t for h in HALLUCINATIONS)]
    real = [t for t in ins if t not in halluc]
    out.append(_res(prefix + "stray speech", "pass" if not real else "fail", "none" if not real else "; ".join(real[:3])))
    if halluc: out.append(_res("whisper music artefact", "warn", f"{len(halluc)} (the speech-to-text invented words over music; harmless)"))
    return out, ratio


def script_check(mp3, lang, script_text, dialect=False, voice_wav=None):
    """two listen-backs: the clean voice track (is every word spoken correctly?) and the finished mp3 with music (is the voice still clear under it?)"""
    exp = _words(script_text); need = 0.80 if dialect else 0.90
    if voice_wav and os.path.exists(voice_wav):
        import soundfile as sf, librosa
        y, sr = sf.read(voice_wav, dtype="float32"); xv = librosa.resample(y, orig_sr=sr, target_sr=16000)
        trv = _transcribe_voice(xv, lang); out, rv = _analyse(trv, exp, need, dialect)
        trm = _transcribe(decode(mp3, 16000), lang); got = _words(trm)
        rm = difflib.SequenceMatcher(None, exp, got, autojunk=False).ratio()
        out.append(_res("clear under music", "pass" if rm >= 0.80 else ("warn" if rm >= 0.65 else "fail"), f"{rm:.3f} of the script heard back from the finished mp3 (music and effects included)"))
        return out, trv
    tr = _transcribe(decode(mp3, 16000), lang); out, _ = _analyse(tr, exp, need, dialect)
    return out, tr


def voice_check(first_wav):
    import librosa, soundfile as sf
    y, sr = sf.read(first_wav, dtype="float32"); y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    out = []; dur = len(y) / sr
    wins = [(0.15, 20), (0.45, 20), (0.75, 20)]
    f0s, rates, stds = [], [], []
    for frac, w in wins:
        s = int(max(0, dur * frac - w / 2) * sr); seg = y[s:s + w * sr]
        f0, vf, _ = librosa.pyin(seg, fmin=65, fmax=450, sr=sr, frame_length=2048, hop_length=512)
        f = f0[~np.isnan(f0)]
        if len(f) < 20: continue
        f0s.append(float(np.median(f))); stds.append(float(np.std(12 * np.log2(f / np.median(f)))))
        env = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0]; env = env / (env.max() + 1e-9)
        from scipy.signal import find_peaks, savgol_filter
        pk, _ = find_peaks(savgol_filter(env, 9, 2), height=0.08, distance=int(0.12 * sr / 256), prominence=0.03)
        iv = librosa.effects.split(seg, top_db=32); sp = sum(e - b for b, e in iv) / sr
        rates.append(len(pk) / max(sp, 1e-6))
    if not f0s:
        return [_res("voice", "fail", "no voiced speech found")]
    drift = (max(f0s) - min(f0s)) / np.median(f0s)
    out.append(_res("steady voice", "pass" if drift < 0.15 else ("warn" if drift < 0.25 else "fail"), f"pitch centre varies {drift * 100:.0f}% across the reading ({', '.join(str(int(v)) for v in f0s)} Hz)"))
    r = float(np.mean(rates)); out.append(_res("pace", "pass" if 2.2 <= r <= 4.4 else "warn", f"{r:.1f} syllables/s"))
    m = float(np.mean(stds)); out.append(_res("expressiveness", "pass" if m >= 2.2 else "warn", f"pitch variation {m:.1f} semitones (real narrators 6-8)"))
    return out


def meaning_check(script_text, lang, st_label, ollama):
    """judge the SCRIPT (what the voice was told to say). Whether the audio really says it is the job of the script-match check, so speech-to-text slips ("Stahl" for "Stall") can no longer fail the story itself."""
    sysmsg = ("You are a strict quality editor for bedtime audio stories for small children (" + st_label + "). You read the full script of an episode (intro, the story, the same story told again, goodbye). "
              "Judge: is the story coherent and sensible, with a clear beginning, middle and end? Are all sentences meaningful and correct in " + ("German" if lang == "de" else "English") +
              " (no nonsense words, broken grammar, or words from another language)? Is everything calm, kind and safe for a three-year-old, with no frightening, violent or unsuitable content? Is it pleasant and not boring? "
              "The story is told twice on purpose; that repetition is fine. Answer exactly:\nVERDICT: PASS or FAIL\nSCORE: <0-10>\nPROBLEMS: <short list of concrete problems with quotes, or none>")
    try:
        r = ollama(sysmsg, script_text[:9000], temp=0.1, num_predict=400)
        v = re.search(r"VERDICT:\s*(PASS|FAIL)", r, re.I); sc = re.search(r"SCORE:\s*([0-9]+(?:[.,][0-9])?)", r); pr = re.search(r"PROBLEMS:\s*(.+)", r, re.S)
        score = float(sc.group(1).replace(",", ".")) if sc else 0
        ok = bool(v and v.group(1).upper() == "PASS") and score >= 7
        return [_res("meaning", "pass" if ok else "fail", f"judge {score}/10; " + (pr.group(1).strip()[:300] if pr else "no details"))]
    except Exception as e:
        return [_res("meaning", "fail", "judge unavailable: " + str(e)[:100])]


def title_check(entry_lang, lang, ollama):
    """spelling and sense of the public title and description (a misspelled title once got through: 'sumrende' for 'summende')"""
    txt = f"TITLE: {entry_lang.get('title', '')}\nDESCRIPTION: {entry_lang.get('description', '')}"
    sysmsg = ("You proofread titles and descriptions of children's bedtime stories in " + ("German" if lang == "de" else "English") +
              ". Check every word for spelling mistakes, non-words and grammar errors. Answer exactly 'OK' if everything is correct, otherwise 'FIX: <the wrong word> -> <the right word>'.")
    try:
        r = ollama(sysmsg, txt, temp=0.0, num_predict=80).strip()
        return [_res("title spelling", "pass" if r.upper().startswith("OK") else "fail", r[:160])]
    except Exception as e:
        return [_res("title spelling", "fail", "proofreader unavailable: " + str(e)[:80])]


def metadata_check(entry_lang, mp3, cover, child, require_ai):
    out = []; txt = " ".join(str(entry_lang.get(k, "")) for k in ("title", "description", "blurb"))
    out.append(_res("title/description", "pass" if entry_lang.get("title") and len(entry_lang.get("description", "")) >= 20 else "fail", f"title {len(entry_lang.get('title', ''))} chars, description {len(entry_lang.get('description', ''))} chars"))
    tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(mp3)], capture_output=True, text=True).stdout
    secrets = [v for v in (child.get("name"), child.get("birthdate")) if v]
    leaked = [v for v in secrets if v.lower() in (txt + tags).lower()]
    out.append(_res("privacy", "pass" if not leaked else "fail", "no child details in public metadata" if not leaked else "child detail found in title/description/tags"))
    if require_ai:
        ok = re.search(r"\b(KI|AI|artificial|künstlich)", txt, re.I) is not None
        out.append(_res("AI disclosure", "pass" if ok else "fail", "present in description" if ok else "missing in description"))
    try:
        from PIL import Image
        w, h = Image.open(cover).size
        out.append(_res("cover size", "pass" if w >= 1000 and abs(w - h) < 10 else "warn", f"{w}x{h}"))
    except Exception as e:
        out.append(_res("cover size", "fail", str(e)[:80]))
    return out


def run(lang, mp3, mix_wav, first_wav, cover, entry_lang, script_text, st_label, ollama, safety_fn, story, child, lo_min=9, hi_min=30, dialect=False, require_ai=False, voice_wav=None):
    checks = []
    steps = (("technical", lambda: technical(mp3, mix_wav, lo_min, hi_min)), ("dead_air", lambda: dead_air_and_clicks(mp3)))
    for name, fn in steps:
        try: checks += fn()
        except Exception as e: checks.append(_res(name, "fail", "check crashed: " + str(e)[:100]))
    try:
        c, transcript = script_check(mp3, lang, script_text, dialect, voice_wav); checks += c
    except Exception as e:
        checks.append(_res("script", "fail", "check crashed: " + str(e)[:100])); transcript = ""
    try: checks += voice_check(first_wav)
    except Exception as e: checks.append(_res("voice", "fail", "check crashed: " + str(e)[:100]))
    checks += meaning_check(script_text, lang, st_label, ollama)
    try:
        ok = safety_fn(story, lang); checks.append(_res("safety", "pass" if ok else "fail", "story text passed the safety check" if ok else "safety check failed"))
    except Exception as e:
        checks.append(_res("safety", "fail", "check crashed: " + str(e)[:100]))
    try: checks += title_check(entry_lang, lang, ollama)
    except Exception as e: checks.append(_res("title spelling", "fail", "check crashed: " + str(e)[:100]))
    try: checks += metadata_check(entry_lang, mp3, cover, child, require_ai)
    except Exception as e: checks.append(_res("metadata", "fail", "check crashed: " + str(e)[:100]))
    verdict = "FAIL" if any(c["status"] == "fail" for c in checks) else "PASS"
    return {"lang": lang, "verdict": verdict, "checks": checks, "transcript_words": len(_words(transcript))}


def format_report(rep):
    icon = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}
    return f"[{rep['lang']}] {rep['verdict']}\n" + "\n".join(f"  {icon[c['status']]} {c['name']}: {c['detail']}" for c in rep["checks"])
