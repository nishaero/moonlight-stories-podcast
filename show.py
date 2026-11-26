#!/usr/bin/env python3
"""Mondlichtgeschichten / Moonlight Stories - a daily bilingual (German + English) bedtime story podcast, generated fully locally.

Pipeline: pick a theme -> qwen3.8:27b (Ollama) writes the story in German, tuned to the child's age (child.json) -> the same model
translates it to English -> XTTS-v2 reads both (same calm female voice, both languages) -> breathing intro + story + the SAME story again,
slower + outro over a soft generated music bed -> loudness-normalised MP3 per language -> two RSS feeds.
Format follows Yoto Dreams / Lumi.

usage: show.py [--theme TEXT] [--short] [--dry-story] [--no-qa] [--langs de,en] [--age-months N]
Env: VOICE (XTTS speaker, default 'Ana Florence'), OLLAMA, MODEL, THREADS
"""
import os, re, sys, json, time, random, hashlib, argparse, datetime as dt, subprocess, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
PUB = ROOT / "public"
STAGE = ROOT / "staging"   # finished episodes wait here until the quality gate (qc.py) has passed
HIST = ROOT / "history.json"
CHILD = ROOT / "child.json"          # {"birthdate": "YYYY-MM-DD", "name": null}  (private, chmod 600, never published)
DIALECT = os.environ.get("DIALECT_DE", "")      # "" = standard German (default). "hessisch" = mild Hessian colour in the German feed (kept as an option, not used)
HESS_FIXED = dict(   # mild Hessian colour: standard German with a few local touches (is, des, widder, noch emol, ei, gell)
    hello="Ei, willkommen bei den Mondlichtgeschichten. Es is Zeit, dich ganz groß zu strecken, es dir gemütlich zu machen und den Tag loszulassen.",
    breath="Mach des Licht aus und kuschel dich in dein Bett. Wir atmen zusammen tief ein, durch die Nase. Und langsam widder aus, durch den Mund. Spür, wie sich dein Bauch hebt und senkt. Noch emol tief ein. Und ganz langsam aus.",
    relaxed="Wunderbar. Jetzt bist du ganz entspannt.",
    twice="Ich erzähl dir die Geschichte zweimal. Beim zweiten Mal erzähl ich noch langsamer, und du darfst dabei einschlafen. Mach die Augen zu. Ab geht's.",
    second="Jetzt erzähl ich dir die Geschichte noch emol, ganz langsam. Du darfst die Augen zulassen.",
    bye="Die Geschichte is aus. Du hast es so schön warm und weich. Schlaf gut. Bis morgen Abend. Gut Nacht.")
HESS_TITLE_LINE = "Heut erzähl ich dir die Geschichte: {title}."
ENGINE = os.environ.get("TTS", "qwen")            # qwen = Qwen3-TTS voice clone of the chosen reference (GPU); xtts = XTTS-v2 (CPU)
QWEN_REF = os.environ.get("QWEN_REF", "C_erzaehlerin")      # the voice of the first episode ("Der schaukelnde Kutter") that was chosen for its voice and modulation
QWEN_SLOW = float(os.environ.get("QWEN_SLOW", "0.82"))   # extra slow-down on top of the stage speed (the clone speaks ~4.2 syll/s, bedtime wants ~3.5)
VOICE = os.environ.get("VOICE", "Nova Hogarth") if ENGINE == "xtts" else "qwen-" + QWEN_REF
OLLAMA = os.environ.get("OLLAMA", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "qwen3.8:27b")
SR = 24000
TARGET_LUFS = -20.0

LANG = {
    "de": dict(show="Mondlichtgeschichten", xtts="de", sub="zum Einschlafen",
               fixed=dict(
                   hello="Willkommen bei den Mondlichtgeschichten. Diese Geschichte wurde mit Hilfe von KI geschrieben, und eine KI-Stimme erzählt sie dir. Es ist Zeit, dich ganz groß zu strecken, es dir gemütlich zu machen und den Tag loszulassen.",
                   breath="Mach das Licht aus und kuschle dich in dein Bett. Wir atmen zusammen tief ein, durch die Nase. Und langsam wieder aus, durch den Mund. Spür, wie sich dein Bauch hebt und senkt. Noch einmal tief ein. Und ganz langsam aus.",
                   relaxed="Wunderbar. Jetzt bist du ganz entspannt.",
                   twice="Ich erzähle dir die Geschichte zweimal. Beim zweiten Mal erzähle ich noch langsamer, und du darfst dabei einschlafen. Schließ die Augen. Los geht's.",
                   second="Jetzt erzähle ich dir die Geschichte noch einmal, ganz langsam. Du darfst die Augen geschlossen lassen.",
                   bye="Die Geschichte ist zu Ende. Du hast es so schön warm und weich. Schlaf gut. Bis morgen Abend. Gute Nacht."),
               title_line="Heute erzähle ich dir die Geschichte: {title}."),
    "en": dict(show="Moonlight Stories", xtts="en", sub="bedtime stories",
               fixed=dict(
                   hello="Welcome to Moonlight Stories. This story was written with the help of AI, and an AI voice is telling it to you. It's time to give a big stretch, snuggle down, and let go of the day.",
                   breath="Turn off the light and cuddle into your bed. Let's take a deep breath together, in through your nose. And slowly out through your mouth. Feel your tummy rise and fall. One more deep breath in. And very slowly out.",
                   relaxed="Wonderful. Now you are all relaxed.",
                   twice="I will tell you the story twice. The second time I will tell it even slower, and you can drift off to sleep as you listen. Close your eyes. Here we go.",
                   second="Now I will tell you the story once more, very slowly. You can keep your eyes closed.",
                   bye="The story is over. You are so warm and cosy. Sleep well. See you tomorrow evening. Good night."),
               title_line="Tonight's story is called: {title}."),
}

# ------------------------------------------------------------------ age stages (the show grows with the child)
STAGES = [  # (up to age in months, settings)
    (42, dict(label="3-3.5 Jahre", words=(420, 560), sent=9, speed=0.80, pause=1.25, tempo=0.88,
              kind="Sehr einfach: ein kleines Tier oder Kuscheltier und das Kind-Du, vertraute Dinge (Decke, Bad, Mond, Teddy, Tiere). Viele sanfte Wiederholungen (\"Und leise, leise ...\"), Lautmalerei (Plitsch, Schnurr, Summ), ein einziges ganz kleines, friedliches Ereignis.",
              learn="Zähle sanft bis drei und nenne ein paar Farben.")),
    (54, dict(label="3.5-4.5 Jahre", words=(560, 720), sent=11, speed=0.82, pause=1.15, tempo=0.88,
              kind="Einfach und vertraut: eine freundliche Hauptfigur mit Namen und ein Begleiter, ein kleines Alltagsabenteuer, viele Wiederholungen und Lautmalerei.",
              learn="Benenne sanft Gefühle (müde, froh, geborgen) und eine kleine freundliche Geste.")),
    (72, dict(label="4.5-6 Jahre", words=(780, 920), sent=14, speed=0.84, pause=1.0, tempo=0.87,
              kind="Eine kleine Fantasiereise oder Geschichte mit Hauptfigur, Begleiter und einer friedlichen Aufgabe.",
              learn="Ein kleines Naturwunder (warum der Mond leuchtet, wie Tiere schlafen) kindgerecht und ruhig erklärt.")),
    (999, dict(label="6+ Jahre", words=(950, 1100), sent=16, speed=0.85, pause=0.9, tempo=0.86,
               kind="Eine ausführlichere Traumreise oder Geschichte mit schönen Bildern und einer friedlichen Aufgabe.",
               learn="Ein kleines Naturwunder oder eine freundliche Idee, ruhig erzählt.")),
]


def child():
    try:
        return json.loads(CHILD.read_text())
    except Exception:
        return {"birthdate": None, "name": None}


def age_months(override=None):
    if override:
        return override
    b = child().get("birthdate")
    if not b:
        return 42
    d = dt.date.fromisoformat(b); t = dt.date.today()
    return (t.year - d.year) * 12 + t.month - d.month - (1 if t.day < d.day else 0)


def stage_for(months):
    for lim, st in STAGES:
        if months < lim:
            return st
    return STAGES[-1][1]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ story (German master, then English translation)
STRUCTURES = {   # story shapes kids love: each gives the tale a hook, a pattern and a small surprise
    "Sammelgeschichte": "Die Hauptfigur trifft nacheinander drei Freunde. Jeder bringt etwas Besonderes mit (jedes Mal mit einem lustigen Geräusch), und am Ende passt alles zusammen. Ein kurzer Refrain kehrt dreimal wieder.",
    "Kleine Suche": "Etwas Wichtiges ist weg. Die Figur sucht an drei Orten; jeder Ort gibt einen lustigen Hinweis (ein Geräusch, eine Spur, ein Freund). Beim dritten Ort wird es gefunden, und alle freuen sich.",
    "Überraschender Besuch": "Ein winziger, unerwarteter Gast taucht auf, und niemand weiß, warum. Die Figur rät dreimal falsch (lustig), dann kommt die liebe Wahrheit heraus.",
    "Warum-Geschichte": "Ein kleines Kind fragt, warum etwas so ist (zum Beispiel warum der Mond mal klein, mal groß ist, warum Igel stachelig sind). Die Geschichte erklärt es fantasievoll, liebevoll und mit einem Augenzwinkern.",
    "Kleiner Mut": "Die Figur möchte etwas Kleines schaffen, das sie sich noch nicht traut. Zwei Versuche gehen lustig daneben, beim dritten klappt es mit Hilfe eines Freundes, und alle sind stolz.",
    "Geheimnis im Garten": "Ein neues, geheimnisvolles Geräusch oder eine Spur ist da. Die Figur und ein Freund gehen neugierig nachsehen. Die Auflösung ist überraschend, freundlich und lustig.",
    "Tauschgeschichte": "Die Figur braucht etwas und tauscht mit drei Freunden hin und her (jedes Mal ein Spaß). Am Ende hat jeder genau das, was er braucht, und noch eine Umarmung dazu.",
    "Kleine Reise": "Die Figur oder ein Fahrzeug fährt, läuft oder fliegt zu einem Ziel und erlebt unterwegs drei kleine Begegnungen mit je einem eigenen Geräusch oder Lied. Am Ziel wartet etwas Schönes.",
}


def style_de(st, name):
    namerule = (f"Die Hauptfigur oder das Kind in der Du-Geschichte heißt {name}." if name else "Gib der Hauptfigur einen kurzen, schönen Namen oder erzähle in der Du-Form.")
    return f"""Du bist eine erfahrene Autorin von Einschlafgeschichten für kleine Kinder (Altersstufe: {st['label']}). Dein Stil ist der von „Yoto Dreams“ und „Lumi“: sehr ruhig, warm, langsam, voller sanfter Sinneseindrücke (Licht, Wärme, weiche Stoffe, leise Geräusche, Gerüche), ohne Spannung und ohne Handlungssprünge.
Regeln:
- Hochdeutsch, einfache kurze Sätze: höchstens {st['sent']} Wörter pro Satz, höchstens 180 Zeichen.
- Altersgerecht: {st['kind']}
- Sanfter Lernfaden (unauffällig): {st['learn']}
- {namerule}
- Nichts Gruseliges, keine Bösewichte, keine Gefahr, kein Streit, keine Eile, keine Bildschirme, keine Marken, keine Werbung.
- Zahlen ausschreiben, keine Abkürzungen, keine englischen Wörter, keine Aufzählungen, keine Überschriften im Text.
- Aufbau in sechs bis acht kurzen Absätzen (Leerzeile dazwischen): 1) Ort und Figur, 2) ein kleiner Wunsch oder eine kleine Aufgabe, 3) und 4) gemeinsam tun, was das Thema verlangt (das Thema muss wirklich vorkommen und gelingen), 5) Ruhe und Gefühle, 6) und 7) Müdigkeit, Einkuscheln, Einschlafen.
- Sie muss SPANNEND UND LUSTIG sein, nicht nur ruhig: Der erste Satz weckt Neugier (eine kleine Frage, ein seltsames Geräusch, ein Wunsch). Die Hauptfigur will etwas ganz Bestimmtes und hat eine kleine Eigenart (zum Beispiel niest sie, wenn sie aufgeregt ist, oder sie summt immer dasselbe Lied). Es gibt eine Dreierfolge, bei der es jedes Mal etwas lustiger wird, einen kleinen Refrain, den Kinder mitsprechen können, einen Moment zum Schmunzeln und eine kleine Überraschung im letzten Drittel. Die Spannung ist nur kindgerecht klein (kein Streit, keine echte Gefahr) und löst sich liebevoll auf. Erst dann wird alles langsam ruhig und müde. Konkrete, bunte Einzelheiten statt allgemeiner Worte.
- Die Geschichte hat einen kleinen, nachvollziehbaren Kern, den ein Kind in diesem Alter kennt und versteht: ein kleines Gefühl oder Problem (zum Beispiel vermissen, Angst vor dem Dunkeln, teilen, mutig sein, etwas Neues probieren, etwas suchen) und eine liebe Lösung durch Freundschaft, Trost, Geduld oder Mut. Klarer Anfang, Mitte und Ende; alles muss logisch zusammenhängen. Nur Dinge, die ein dreijähriges Kind aus seinem Alltag kennt (Tiere, Fahrzeuge, Kuscheltiere, Familie, Essen, Bad, Wetter), keine abstrakten oder poetisch-rätselhaften Bilder. Das Thema ist die Handlung, nicht nur die Kulisse. Die Hauptfigur hat einen Namen.
- Jeder Satz sagt etwas Klares und Schönes. Keine seltsamen Vergleiche (nichts Kaltes, Hartes oder Scharfes), keine Widersprüche. Das Wort "leise" höchstens fünfmal.
- SCHREIBE WIE MAN SPRICHT, nicht wie ein Buch: gesprochenes, warmes Deutsch mit kurzen Nebensätzen, kleinen Füllwörtern und Abtönungspartikeln (ja, doch, mal, na, nun, schon, eben, so), Ausrufen (Ach! Oh! Mmh!), unvollständigen Sätzen („Ganz weich. Ganz warm.“), Wiederholungen („und dann, und dann …“) und einem kleinen Nachsatz („…, der Igel“). Keine langen Schachtelsätze, keine Schriftsprache wie „sodann“, „indes“, „jedoch“. Wörtliche Rede in „…“.
- Lebendig wie eine echte Erzählerin, die einem Kind auf dem Schoß vorliest: spreche das Kind ab und zu direkt an („Weißt du, was er da sah?“, „Na, hörst du das?“), stelle kleine Fragen und beantworte sie gleich, baue wörtliche Rede ein (die Figuren sagen kurze, liebe Sätze), nutze Klangwörter (Plitsch, Platsch, Schschsch, Mmmh, Hui) und wechsle die Satzlänge: manche Sätze nur zwei oder drei Wörter, andere etwas länger. Gelegentlich ein Ausruf („Oh!“) oder ein geflüstertes Geheimnis („Pssst…“).
- Ruhiger Rhythmus: sanfte Wiederholungen; Bilder, die langsam kleiner, weicher und leiser werden.
- Das Ende: die Hauptfigur (oder das Du) wird ganz müde, kuschelt sich ein und schläft ein. Die letzten Sätze werden immer kürzer und leiser.
Antworte genau in diesem Format und sonst nichts:
TITEL: <kurzer Titel, höchstens fünf Wörter>
BESCHREIBUNG: <ein Satz für Eltern>
GESCHICHTE:
<Absätze, durch Leerzeilen getrennt>"""


STYLE_EN = """You translate German bedtime stories for small children into natural, warm, simple English (not word by word: the way a loving parent would tell it).
Keep the same paragraphs, the same calm rhythm, the same very short sentences, repetitions and sound words. No digits (write numbers as words), no markdown.
Answer exactly in this format and nothing else:
TITLE: <short title>
DESCRIPTION: <one sentence for parents>
STORY:
<paragraphs separated by blank lines>"""


def ollama(system, user, temp=0.8, num_predict=3500):
    body = {"model": MODEL, "stream": False, "think": False, "keep_alive": "10m",
            "options": {"temperature": temp, "num_ctx": 8192, "num_predict": num_predict, "top_p": 0.92},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(OLLAMA + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["message"]["content"].strip()


def ollama_unload():
    """free the GPU memory of the story model before the voice runs: two heavy models on this one integrated GPU at the same time can wedge it (it did, twice)"""
    try:
        urllib.request.urlopen(urllib.request.Request(OLLAMA + "/api/generate", json.dumps({"model": MODEL, "keep_alive": 0}).encode(), {"Content-Type": "application/json"}), timeout=60).read()
    except Exception as e:
        log("ollama unload failed:", str(e)[:80])


def tts_unload():
    """release the voice model so the story/judge model can use the GPU alone"""
    global _qwen, _tts
    _qwen = None; _tts = None
    try:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass


def clean_text(t):
    t = re.sub(r"[*_#>`]+", "", t).replace("…", "...").replace("–", ", ").replace("—", ", ")
    return re.sub(r"\s+\n", "\n", t).strip()


def parse(raw, keys):
    m = re.search(rf"{keys[0]}:\s*(.+)\n+{keys[1]}:\s*(.+)\n+{keys[2]}:\s*\n*(.+)", raw, re.S)
    if not m:
        return None
    paras = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", clean_text(m.group(3))) if p.strip()]
    return m.group(1).strip().strip('"„“'), m.group(2).strip(), paras


NUM_DE = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf"]
NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def words_out(paras, table):
    return [re.sub(r"\d+", lambda x: table[int(x.group())] if int(x.group()) < len(table) else x.group(), p) for p in paras]


def make_story_de(theme, recent, st, name, short, form, structure=None):
    mode = form or ("du" if dt.date.today().toordinal() % 2 else "figur")
    form_txt = ("Erzähle als sanfte Traumreise in der Du-Form (\"du spürst, du hörst ...\")." if mode == "du"
                else "Erzähle von einer sanften Hauptfigur und einem freundlichen Begleiter.")
    lo, hi = (90, 220) if short else st["words"]
    shape = f"\nGeschichtenform \"{structure}\": {STRUCTURES[structure]}" if structure else ""
    user = f"Thema: {theme}.{shape}\n{form_txt}\nLänge: zwischen {lo} und {hi} Wörtern.\nVermeide diese schon erzählten Titel: {', '.join(recent[-12:]) or 'keine'}."
    for attempt in range(4):
        p = parse(ollama(style_de(st, name), user, temp=0.8 + 0.04 * attempt), ("TITEL", "BESCHREIBUNG", "GESCHICHTE"))
        if not p:
            log("format miss"); continue
        title, desc, paras = p
        paras = words_out(paras, NUM_DE)
        paras = regroup(paras)
        n = len(" ".join(paras).split())
        english = re.search(r"\b(the|and|with|you are)\b", " ".join(paras), re.I) is not None
        longest = max((len(s.split()) for q in paras for s in re.split(r"(?<=[.!?])\s+", q)), default=0)
        ok = lo * 0.85 <= n <= hi * 1.2 and not english and longest <= st["sent"] + 8
        log(f"story attempt {attempt+1}: {n} words (want {lo}-{hi}), longest sentence {longest}, ok={ok}")
        if ok:
            paras = interest_pass(title, paras, theme, st)
            paras = edit_pass(title, paras, theme, st)
            return dict(title=title, description=desc, paragraphs=paras, words=len(' '.join(paras).split()), mode=mode)
    raise SystemExit("no acceptable German story from the model")



def regroup(paras):
    """models sometimes return one long block: cut it into paragraphs of about five sentences so the reading gets breathing pauses"""
    if len(paras) >= 4:
        return paras
    sents = [x for p in paras for x in re.split(r"(?<=[.!?])\s+", p) if x]
    k = max(4, round(len(sents) / 7))
    return [" ".join(sents[i:i + k]) for i in range(0, len(sents), k)]


def interest_pass(title, paras, theme, st):
    """a critic scores how interesting the story is for a small child (hook, character, surprise, humour, pattern); weak stories are rewritten once or twice, the best version wins"""
    crit = ("Du bist eine strenge Kinderbuch-Lektorin. Bewerte die Einschlafgeschichte für ein Kind dieser Altersstufe: " + st["label"] + ". "
            "Kriterien (je 0 bis 10): Einstieg weckt Neugier; Hauptfigur mit Wunsch und Eigenart; Dreierfolge oder Refrain zum Mitsprechen; Moment zum Schmunzeln; kleine Überraschung; klarer Anfang, Mitte, Ende. "
            "Antworte genau so: SCORE: <Durchschnitt 0-10>\nSCHWACH: <die zwei schwächsten Punkte, ein Satz>")
    def rate(ps):
        r = ollama(crit, "\n\n".join(ps), temp=0.1, num_predict=200)
        m = re.search(r"SCORE:\s*([0-9]+(?:[.,][0-9])?)", r); w = re.search(r"SCHWACH:\s*(.+)", r)
        return (float(m.group(1).replace(",", ".")) if m else 7.0), (w.group(1).strip() if w else "")
    try:
        best, (best_score, weak) = paras, rate(paras)
        log(f"interest score {best_score}: {weak[:120]}")
        for k in range(2):
            if best_score >= 8.0:
                break
            sysmsg = ("Du bist Autorin von Einschlafgeschichten für Kleinkinder. Mach die Geschichte SPANNENDER, LUSTIGER und lebendiger, ohne sie unruhig oder gefährlich zu machen. "
                      f"Das sind die Schwächen: {weak} Baue gezielt ein: einen neugierig machenden ersten Satz; eine Figur mit Wunsch und kleiner Eigenart; eine Dreierfolge, bei der es jedes Mal lustiger wird; "
                      "einen kurzen Refrain; einen Moment zum Schmunzeln; eine kleine, liebe Überraschung im letzten Drittel. Danach wird es langsam ruhig und müde, und die Figur schläft ein. "
                      f"Behalte das Thema \"{theme}\", die Absätze (Leerzeilen), die Länge (±15 Prozent) und kurze, einfache Sätze (höchstens {st['sent']} Wörter, keine englischen Wörter). "
                      "Gib NUR den neuen Text aus, ohne Titel und Kommentar.")
            new = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", clean_text(ollama(sysmsg, "\n\n".join(best), temp=0.7 + 0.1 * k))) if p.strip()]
            new = regroup(words_out(new, NUM_DE))
            n0, n1 = len(" ".join(paras).split()), len(" ".join(new).split())
            longest = max((len(x.split()) for q in new for x in re.split(r"(?<=[.!?])\s+", q)), default=0)
            if not (0.8 * n0 <= n1 <= 1.3 * n0) or re.search(r"\b(the|and|with)\b", " ".join(new), re.I) or longest > st["sent"] + 8:
                log(f"interest rewrite {k+1} rejected (words {n1}, longest sentence {longest})"); continue
            sc, wk = rate(new); log(f"interest rewrite {k+1}: score {sc} ({n0} -> {n1} words)")
            if sc > best_score: best, best_score, weak = new, sc, wk
        return best
    except Exception as e:
        log("interest pass failed:", str(e)[:80]); return paras


def edit_pass(title, paras, theme, st):
    """second pass: fix odd phrases, make sure the theme really happens, keep length and the calm ending"""
    sysmsg = ("Du bist Lektorin für Einschlafgeschichten für Kleinkinder. Verbessere den Text: Entferne seltsame oder unpassende Vergleiche, Widersprüche und zu häufige Wiederholungen "
              f"(besonders das Wort \"leise\"). Das Thema \"{theme}\" muss in der Mitte wirklich vorkommen und gelingen. Behalte die Länge, die einfachen kurzen Sätze (höchstens {st['sent']} Wörter), "
              "die Absätze (Leerzeilen) und das ruhige Einschlafen am Ende. Gib NUR den verbesserten Text aus, ohne Titel, ohne Kommentar.")
    try:
        new = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", clean_text(ollama(sysmsg, "\n\n".join(paras), temp=0.4)))
               if p.strip()]
        new = regroup(words_out(new, NUM_DE))
        n0, n1 = len(" ".join(paras).split()), len(" ".join(new).split())
        if 0.8 * n0 <= n1 <= 1.25 * n0 and not re.search(r"\b(the|and|with)\b", " ".join(new), re.I):
            log(f"edit pass applied ({n0} -> {n1} words, {len(new)} paragraphs)")
            return new
    except Exception as e:
        log("edit pass failed:", str(e)[:80])
    return paras


def translate(story):
    """paragraph by paragraph, so the English keeps the same paragraphs and pauses as the German"""
    sysmsg = ("You translate German bedtime-story text for a three-year-old into natural, warm, simple English, the way a loving parent would tell it (not word by word). "
              "Keep very short sentences, repetitions and sound words. Write numbers as words, no digits, no markdown. Answer with the translation only, no comments.")
    def tr(x, temp=0.3):
        return clean_text(ollama(sysmsg, x, temp=temp, num_predict=900)).strip().strip('"')
    for attempt in range(2):
        paras = [tr(p, 0.3 + 0.1 * attempt).replace("\n", " ") for p in story["paragraphs"]]
        title = tr(story["title"]).split("\n")[0]; desc = tr(story["description"]).split("\n")[0]
        paras = words_out(paras, NUM_EN)
        n = len(" ".join(paras).split()); ratio = n / max(story["words"], 1)
        bad = re.search(r"[äöüß]", " ".join(paras)) or any(len(p) < 5 for p in paras)
        log(f"translation attempt {attempt+1}: {len(paras)} paragraphs, length ratio {ratio:.2f}, bad={bool(bad)}")
        if 0.7 <= ratio <= 1.5 and not bad:
            return dict(title=title, description=desc, paragraphs=paras, words=n, mode=story["mode"])
    raise SystemExit("no acceptable English translation")



def to_hessisch(story):
    """standard German master story -> Hessian dialect (light spelling that a German voice can read), paragraph by paragraph"""
    sysmsg = ("Du schreibst Einschlafgeschichten für ein dreijähriges Kind aus Hessen. Gib dem Text nur einen MILDEN hessischen Anstrich: "
              "Er bleibt gut verständliches, normal geschriebenes Hochdeutsch mit Wortwahl und Rechtschreibung des Standarddeutschen. Nur an wenigen Stellen (etwa ein- bis zweimal pro Absatz) kommen sanfte hessische Färbungen vor: "
              "'is' statt 'ist', 'des' statt 'das' (als Pronomen), 'widder' statt 'wieder', 'noch emol', 'gell', 'ei', 'e bissje', und Verkleinerungen auf '-je' oder '-sche' bei kleinen Tieren und Dingen (Häsje, Kätzsche, Liedche). "
              "KEINE starken Formen: kein 'se' für 'zu', kein 'uff', kein 'net', keine verschluckten Endungen, keine Apostrophe. Gleiche Anzahl Sätze, gleiche Reihenfolge, gleicher Inhalt, gleiche kurze Sätze und Lautmalerei. "
              "Zahlen als Wörter, keine Anführungszeichen um den Text. Antworte nur mit dem übertragenen Text, ohne Kommentar.")
    def tr(x, temp=0.4):
        return clean_text(ollama(sysmsg, x, temp=temp, num_predict=900)).strip().strip('"').replace("\n", " ")
    for attempt in range(3):
        paras = [tr(p, 0.35 + 0.1 * attempt) for p in story["paragraphs"]]
        paras = words_out(paras, NUM_DE)
        ratio = len(" ".join(paras).split()) / max(story["words"], 1)
        bad = any(len(p) < 5 for p in paras) or re.search(r"\b(the|and|with)\b", " ".join(paras), re.I)
        log(f"hessisch attempt {attempt+1}: ratio {ratio:.2f}, bad={bool(bad)}")
        if 0.75 <= ratio <= 1.3 and not bad:
            title = tr(story["title"]).split("\n")[0]
            return paras, title
    log("hessian conversion failed, using standard German"); return None, None


def episode_blurb(story, lang, st):
    """warm 2-3 sentence description for parents, written by the model from the finished story (no spoilers of the ending needed)"""
    if lang == "de":
        sysmsg = ("Du schreibst die Episodenbeschreibung für einen Einschlaf-Podcast für Kleinkinder. Schreibe 2 bis 3 warme, freundliche Sätze für Eltern: "
                  "worum es in der Geschichte geht und welche ruhige Stimmung sie hat. Kein Spoiler vom Ende, keine Werbung, keine Anführungszeichen, nur der Text.")
    else:
        sysmsg = ("You write the episode description for a bedtime podcast for toddlers. Write 2 to 3 warm, friendly sentences for parents: "
                  "what the story is about and the calm mood it has. No spoiler of the ending, no advertising, no quotation marks, just the text.")
    try:
        t = clean_text(ollama(sysmsg, f"{story['title']}\n\n" + " ".join(story["paragraphs"])[:1800], temp=0.5, num_predict=160)).strip().strip('"')
        t = re.sub(r"\s+", " ", t)
        if 40 <= len(t) <= 600:
            return t
    except Exception as e:
        log("blurb failed:", str(e)[:80])
    return story["description"]


def scene_prompts(story):
    """two English lines for the image generator: a calm picture-book scene, and one simple object for a 16x16 pixel icon"""
    ans = ollama("You write prompts for an image generator for a children's bedtime story. Answer with exactly two lines. "
                 "The picture MUST show the main character named in the story TITLE (if the title is 'The humming pig', the picture shows a pig, not another animal from the story). "
                 "Line 1: one concrete picture-book scene in English (that main character in a gentle setting, soft night light), no people faces, no text. "
                 "Line 2: ONE simple object for a tiny pixel icon that shows the same main character, 3 to 5 words, e.g. 'a pink sleeping pig'.",
                 f"Title: {story['title']}\n" + " ".join(story["paragraphs"])[:1500], temp=0.5, num_predict=120)
    lines = [l.strip(" -*0123456789.:") for l in ans.splitlines() if l.strip()]
    lines = [re.sub(r"^(line \d+:?)", "", l, flags=re.I).strip() for l in lines]
    scene = lines[0] if lines else "a sleepy little animal under a glowing moon"
    obj = lines[1] if len(lines) > 1 else "a crescent moon"
    return scene, obj


def safety_check(story, lang):
    txt = " ".join(story["paragraphs"])
    if lang == "de":
        q, s = "Ist diese Einschlafgeschichte für ein dreijähriges Kind völlig unbedenklich (nichts Gruseliges, keine Gewalt, keine Gefahr, kein Streit, keine Werbung), ruhig erzählt und in korrektem Deutsch?", "Du prüfst Kindergeschichten. Antworte nur mit OK oder NICHT OK und einem kurzen Grund."
    else:
        q, s = "Is this bedtime story completely safe for a three-year-old (nothing scary, no violence, no danger, no conflict, no advertising), calmly told, in correct English?", "You review children's stories. Answer only OK or NOT OK with a short reason."
    ans = ollama(s, q + "\n\n" + txt, temp=0.1, num_predict=60)
    log(f"safety check [{lang}]:", ans[:110])
    return ans.upper().startswith("OK")


# ------------------------------------------------------------------ audio
def silence(sec):
    return np.zeros(int(SR * sec), dtype=np.float32)


def split_sentences(par):
    out = []
    for p in re.split(r"(?<=[.!?])\s+", par.strip()):
        while len(p) > 220:
            cut = max(p.rfind(",", 0, 210), p.rfind(" ", 0, 210))
            out.append(p[:cut].strip()); p = p[cut:].strip(" ,")
        if p:
            out.append(p)
    return out


_tts = None


def tts_model():
    global _tts
    if _tts is None:
        os.environ.setdefault("COQUI_TOS_AGREED", "1")   # XTTS CPML licence: personal, non-commercial use only
        import torch
        from TTS.api import TTS
        torch.set_num_threads(int(os.environ.get("THREADS", "20")))
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    return _tts


def hear(wav, lang):
    """what does the speech actually say? German/English Whisper on the workstation (Wyoming)"""
    import socket, librosa
    y = librosa.resample(wav, orig_sr=SR, target_sr=16000)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    s = socket.create_connection(("127.0.0.1", 10300), timeout=120)
    def send(t, data=None, payload=b""):
        h = {"type": t}
        if data: h["data"] = data
        if payload: h["payload_length"] = len(payload)
        s.sendall((json.dumps(h) + "\n").encode() + payload)
    fmt = {"rate": 16000, "width": 2, "channels": 1}
    send("transcribe", {"language": lang}); send("audio-start", fmt)
    for k in range(0, len(pcm), 32000): send("audio-chunk", fmt, pcm[k:k + 32000])
    send("audio-stop")
    f = s.makefile("rb"); h = json.loads(f.readline())
    body = f.read(h.get("data_length", 0)) if h.get("data_length") else b""
    s.close()
    return json.loads(body)["text"] if body else ""


FILLER_TOKENS = {"äh", "ähm", "ehm", "hm", "hmm", "mhm", "mmh", "mh", "öh", "um", "uh", "er", "uhm", "mm", "mmm"}


def _norm(t):
    return [w for w in re.sub(r"[^a-zäöüß ]", "", t.lower()).split() if w not in FILLER_TOKENS]


_qwen = None


def qwen_model():
    global _qwen
    if _qwen is None:
        import torch, soundfile as sf
        from qwen_tts import Qwen3TTSModel
        m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
        refs = json.load(open(ROOT / "voices" / "refs.json"))
        wav, sr = sf.read(ROOT / "voices" / f"ref_{QWEN_REF}.wav", dtype="float32")
        _qwen = (m, m.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=refs.get("texts", {}).get(QWEN_REF, refs["ref_text"])))
    return _qwen


def say_once(text, lang, speed, seed):
    import torch
    torch.manual_seed(seed)
    if ENGINE == "qwen":
        m, prompt = qwen_model()
        w, sr = m.generate_voice_clone(text=text, language={"de": "German", "en": "English"}[lang], voice_clone_prompt=prompt, **({"temperature": 1.0, "top_k": 60, "top_p": 0.97} if os.environ.get("QWEN_HOT") == "1" else {}))
        w = np.asarray(w[0], dtype=np.float32)
        if sr != SR:
            import librosa; w = librosa.resample(w, orig_sr=sr, target_sr=SR)
        f = QWEN_SLOW
        if abs(f - 1) > 0.02: w = tempo(w, max(f, 0.5))
        return w * (0.9 / max(np.abs(w).max(), 1e-6))
    w = np.asarray(tts_model().tts(text=text, speaker=VOICE, language=LANG[lang]["xtts"], speed=speed), dtype=np.float32)
    return w * (0.9 / max(np.abs(w).max(), 1e-6))


STATS = {"chunks": 0, "retries": 0, "gave_up": 0}
SALTS_FILE = WORK / "salts.json"


def _salts():
    try: return json.loads(SALTS_FILE.read_text())
    except Exception: return {}


def salt_key(lang, text):
    return hashlib.md5((lang + "|" + text).encode()).hexdigest()[:12]


def bump_salt(lang, text):
    """make the next build re-record this passage with a different random take (the auto-repair uses this)"""
    d = _salts(); k = salt_key(lang, text); d[k] = d.get(k, 0) + 1
    WORK.mkdir(exist_ok=True); SALTS_FILE.write_text(json.dumps(d))


def say(text, lang, speed, verify=True):
    """synthesize, listen back with Whisper, and re-record (different random seed) when the take does not match the text; takes are cached per passage"""
    import difflib
    salt = _salts().get(salt_key(lang, text), 0)
    cdir = WORK / "chunks" / lang; cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / (hashlib.md5(f"{VOICE}|{lang}|{text}|{salt}|{QWEN_SLOW}|{speed}".encode()).hexdigest() + ".npy")
    if verify and cfile.exists():
        return np.load(cfile)
    base = int(hashlib.md5((lang + text + str(salt)).encode()).hexdigest()[:8], 16)
    best, best_score = None, -1
    dial = lang == "de" and DIALECT == "hessisch"   # Whisper writes dialect speech in standard German, and the text is mostly standard German with a few local words, so the bar is only slightly lower
    for attempt in range(4 if verify else 1):
        w = say_once(text, lang, speed, base + attempt)
        if not verify:
            return w
        score = difflib.SequenceMatcher(None, _norm(text), _norm(hear(w, lang))).ratio()
        if score > best_score:
            best, best_score = w, score
        if score >= (0.85 if dial else 0.92):
            break
        STATS["retries"] += 1
    STATS["chunks"] += 1
    if best_score < (0.70 if dial else 0.80):
        STATS["gave_up"] += 1; log(f"  !! weak take ({best_score:.2f}): {text[:60]}")
    np.save(cfile, best)
    return best


def chunks_of(par):
    """XTTS is steadier on 60-200 characters than on one very short sentence: merge short sentences, the pause comes after the chunk"""
    out, cur = [], ""
    for s in split_sentences(par):
        if cur and len(cur) + len(s) + 1 > 200:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
        if len(cur) >= 70:
            out.append(cur); cur = ""
    if cur:
        if out and len(cur) < 40 and len(out[-1]) + len(cur) < 230:
            out[-1] += " " + cur
        else:
            out.append(cur)
    return out


def read_paragraphs(paras, lang, st, label):
    """returns (audio, spans): spans[i] = dict(par, start, end, chunks=[(text, start_sample)]) so sound effects can follow the story"""
    out = []; total = sum(len(chunks_of(p)) for p in paras); k = 0; pos = 0; spans = []
    def put(x):
        nonlocal pos
        out.append(x); pos += len(x)
    for i, p in enumerate(paras):
        sp = dict(par=i, start=pos, chunks=[])
        for c in chunks_of(p):
            k += 1; sp["chunks"].append((c, pos))
            put(say(c, lang, st["speed"])); put(silence(random.Random(k).uniform(0.85, 1.15) * st["pause"]))
            if k % 5 == 0 or k == total:
                log(f"  {lang} {label}: {k}/{total} chunks, retries so far {STATS['retries']}")
        sp["end"] = pos; spans.append(sp)
        put(silence(1.8))
    return np.concatenate(out), spans


def tempo(wav, factor):
    import soundfile as sf
    a, b = WORK / "tmp_in.wav", WORK / "tmp_out.wav"
    sf.write(a, wav, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-filter:a", f"atempo={factor}", str(b)], check=True)
    return sf.read(b, dtype="float32")[0]


def pad(seconds):
    """soft ambient pad generated from sine waves (A minor add9): no samples, so no licence questions"""
    t = np.arange(int(SR * seconds)) / SR
    sig = np.zeros_like(t)
    for i, f in enumerate([110.0, 164.81, 220.0, 261.63, 329.63, 493.88]):
        for det in (-0.6, 0.0, 0.7):
            lfo = 0.55 + 0.45 * np.sin(2 * np.pi * (0.03 + 0.011 * i) * t + i * 1.3)
            sig += (1.0 / (1 + 0.35 * i)) * lfo * np.sin(2 * np.pi * (f + det) * t + det)
    sig /= np.abs(sig).max(); fade = int(SR * 4)
    sig[:fade] *= np.linspace(0, 1, fade); sig[-fade:] *= np.linspace(1, 0, fade)
    return sig.astype(np.float32)


def normalise(wav):
    import pyloudnorm as pyln
    m = pyln.Meter(SR)
    return np.clip(pyln.normalize.loudness(wav, m.integrated_loudness(wav), TARGET_LUFS), -0.97, 0.97).astype(np.float32)


def fixed_parts(lang, title):
    hess = lang == "de" and DIALECT == "hessisch"
    texts = HESS_FIXED if hess else LANG[lang]["fixed"]
    key = hashlib.md5((VOICE + lang + json.dumps(texts, sort_keys=True)).encode()).hexdigest()[:10]
    cache = WORK / f"fixed_{lang}_{key}.npz"
    if cache.exists():
        z = np.load(cache); fx = {k: z[k] for k in z.files}
    else:
        fx = {k: say(v, lang, 0.82) for k, v in texts.items()}; np.savez(cache, **fx)
    fx["title"] = say((HESS_TITLE_LINE if hess else LANG[lang]["title_line"]).format(title=title), lang, 0.82)
    return fx


def trim_silences(w, max_s=2.6):
    """cut silent stretches in the voice longer than max_s down to max_s (a take that left a hole)"""
    fr = int(SR * 0.05); n = len(w) // fr
    rms = np.sqrt((w[:n * fr].reshape(n, fr) ** 2).mean(axis=1)); quiet = rms < 0.004
    keep = np.ones(n, dtype=bool); run = 0
    for i in range(n):
        run = run + 1 if quiet[i] else 0
        if run > max_s / 0.05: keep[i] = False
    return np.concatenate([w[i * fr:(i + 1) * fr] for i in range(n) if keep[i]] + [w[n * fr:]])


def sfx_layers(spans, paras, marks, n, tempo2, lang, second=True):
    """ambience that matches each part of the story (rain, water, night ...) plus short cues on sound words (splash, owl, purr ...). ambience layer, cue layer."""
    import sfx
    amb = np.zeros(n, dtype=np.float32); cue = np.zeros(n, dtype=np.float32)
    f0 = int(marks["first"] * SR); s0 = int(marks["second"] * SR)
    runs = []   # merge consecutive paragraphs with the same ambience
    for sp in spans:
        k = sfx.ambience_for(paras[sp["par"]])
        if runs and runs[-1][0] == k and k: runs[-1][2] = sp["end"]
        else: runs.append([k, sp["start"], sp["end"]])
    for gi, (k, a, b) in enumerate(runs):
        if not k: continue
        for base, scale, lvl, fade in ((f0, 1.0, 1.0, 2.5),) + (((s0, 1.0 / tempo2, 0.55, 4.0),) if second else ()):
            st_, en_ = base + int(a * scale), base + int(b * scale)
            seg = sfx.ambience(k, (en_ - st_) / SR + 3.0, seed=gi + 1 + (7 if base == s0 else 0), fade=fade) * lvl
            p0 = max(0, st_ - int(1.5 * SR)); e = min(n, p0 + len(seg)); amb[p0:e] += seg[: e - p0]
    last = -99.0; count = 0
    for sp in spans:       # cues only in the first reading, at most one every 15 s
        for text, start in sp["chunks"]:
            kinds = sfx.cues_for(text)
            t = (f0 + start) / SR
            if kinds and t - last > 15 and count < 8:
                x = sfx.CUES[kinds[0]](seed=count + 1); p = int(t * SR + 0.4 * SR); e = min(n, p + len(x)); cue[p:e] += x[: e - p]; last = t; count += 1
    return amb, cue


def intimate(w):
    """close-microphone warmth for a bedtime voice: a little low-end (proximity effect), softer sibilance and top end, gentle compression"""
    import soundfile as sf
    a, b = WORK / "fx_in.wav", WORK / "fx_out.wav"
    sf.write(a, w, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-af",
                    "equalizer=f=150:t=q:w=0.9:g=2.5,equalizer=f=4200:t=q:w=1.2:g=-2.5,highshelf=f=8500:g=-4,acompressor=threshold=0.09:ratio=2.2:attack=15:release=300:makeup=1.4", str(b)], check=True)
    return sf.read(b, dtype="float32")[0]


def build(story, lang, st, opts=None):
    """voice timeline with musical gaps; the music (theme, stingers, bed that ducks under the voice, outro) is generated by music.py"""
    import soundfile as sf
    import music
    from scipy.ndimage import uniform_filter1d
    fx = fixed_parts(lang, story.get("title_hess") or story["title"])
    paras = spoken_paragraphs(story, lang)
    opts = opts or {}
    cache = WORK / f"first_{lang}.wav"
    sidecar = WORK / f"first_{lang}.json"
    if os.environ.get("REUSE_FIRST") and cache.exists():
        first = sf.read(cache, dtype="float32")[0]
        try: spans = json.loads(sidecar.read_text())
        except Exception: spans = None
    else:
        first, spans = read_paragraphs(paras, lang, st, "first reading")
        if opts.get("trim"):
            n0 = len(first); first = trim_silences(first); r = len(first) / max(n0, 1)   # keep the sound-effect timing in step (proportional)
            for sp in spans:
                sp["start"] = int(sp["start"] * r); sp["end"] = int(sp["end"] * r); sp["chunks"] = [(t, int(a * r)) for t, a in sp["chunks"]]
        sf.write(cache, first, SR); sidecar.write_text(json.dumps(spans))
    second = tempo(first, opts.get("tempo2") or st["tempo"])
    parts, marks, pos = [], {}, 0

    def add(name, x):
        nonlocal pos
        marks[name] = pos / SR; parts.append(x); pos += len(x)
    add("lead", silence(11.0))
    for name, gap in (("hello", 1.5), ("breath", 3.0), ("relaxed", 1.2)):
        add(name, fx[name]); add(name + "_gap", silence(gap))
    add("st0", silence(6.0))
    add("title", fx["title"]); add("g1", silence(0.8)); add("twice", fx["twice"]); add("g2", silence(3.5))
    add("first", first); add("g3", silence(2.0))
    add("st1", silence(7.0))
    add("second_ann", fx["second"]); add("g4", silence(3.0)); add("second", second); add("g5", silence(2.0))
    add("st2", silence(6.0))
    add("bye", fx["bye"]); add("g6", silence(1.5))
    add("outro", silence(16.0))
    voice = np.concatenate(parts)
    if os.environ.get("VOICE_FX", "0") == "1": voice = intimate(voice)
    voice = voice / max(np.abs(voice).max(), 1e-6) * 0.8
    sf.write(WORK / f"voice_{lang}.wav", voice, SR)   # clean voice track: the quality gate listens to it separately from the music mix
    total = len(voice) / SR
    # sidechain: music is quiet while the voice speaks and swells in the gaps
    env = uniform_filter1d(np.abs(voice), int(SR * 0.35)); mask = (env > 0.06 * np.percentile(env, 95)).astype(np.float32)
    duck = uniform_filter1d(mask, int(SR * 0.9)); duck = 1.0 - np.clip(duck, 0, 1)
    gain = (0.09 + 0.26 * duck) * opts.get("music", 1.0)
    bed = music.bed(total, 0.5, seed=hash(lang + story["title"]) % 1000, tempo=0.9)
    sl = int(marks["second_ann"] * SR)   # quieter, sparser bed for the slow second reading
    bed2 = music.bed(total, 0.28, seed=5, tempo=1.3); bed[sl:] = bed2[sl:] * 0.8
    fin = int(SR * 3); bed[:int(marks["hello"] * SR)] = 0   # bed starts under the voice, faded in
    a0 = int(marks["hello"] * SR); bed[a0:a0 + fin] *= np.linspace(0, 1, fin)
    out = np.zeros(len(voice), dtype=np.float32)
    out += bed * gain
    if spans and os.environ.get("SFX", "1") == "1":
        amb, cuel = sfx_layers(spans, paras, marks, len(voice), opts.get("tempo2") or st["tempo"], lang, opts.get("sfx2", True))
        out += amb * (0.045 + 0.11 * duck) * opts.get("music", 1.0)
        out += cuel * 0.5 * opts.get("music", 1.0)
    th = music.theme(16.0); out[:len(th)] += th * (0.35 + 0.25 * duck[:len(th)])
    for i, (key, var) in enumerate((("st0", 0), ("st1", 1), ("st2", 2))):
        x = music.stinger(float(5.5), var); p = int((marks[key] + 0.3) * SR); out[p:p + len(x)] += x[:len(out) - p] * 0.55
    ot = music.outro(16.0); p = int(marks["outro"] * SR); out[p:p + len(ot)] += ot[:len(out) - p] * 0.6
    mix = normalise(voice + out)
    sf.write(WORK / f"mix_{lang}.wav", mix, SR)
    return mix


def encode(lang, out_mp3, story, number, cover=None, opts=None):
    """maximum-compatibility MP3 for small players (Yoto): CBR 128 kbps, 44.1 kHz stereo, plain ID3v2.3 text tags and NO picture inside the file (the cover lives in the RSS feed; some importers choke on a big embedded image)"""
    opts = opts or {}; show = LANG[lang]["show"]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(WORK / f"mix_{lang}.wav")]
    cover = cover or PUB / lang / "cover.jpg"
    embed = os.environ.get("EMBED_COVER") == "1" and cover.exists()
    if embed:
        cmd += ["-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
    else:
        cmd += ["-map", "0:a", "-vn"]
    flt = []
    if opts.get("declick"): flt.append("adeclick")
    if opts.get("norm"): flt.append("loudnorm=I=-20:TP=-1.5:LRA=11")
    if flt: cmd += ["-af", ",".join(flt)]
    cmd += ["-ac", "2", "-ar", "44100", "-codec:a", "libmp3lame", "-b:a", "128k", "-id3v2_version", "3", "-write_id3v1", "1", "-metadata", f"title={story['title']}",
            "-metadata", f"artist={show}", "-metadata", f"album={show}", "-metadata", f"track={number}", "-metadata", f"comment={story['description']}", str(out_mp3)]
    subprocess.run(cmd, check=True)


def qa(lang, story):
    """judge the SOUND (pace, pitch movement, clipping, length) and, separately, that the speech matches the script"""
    res = {"flags": []}
    py = os.environ.get("AUDIO_PY", sys.executable)
    out = subprocess.run([py, str(ROOT / "audio_profile.py"), str(WORK / f"first_{lang}.wav"), "--start", "20", "--dur", "100", "--json"], capture_output=True, text=True)
    try:
        s = json.loads(out.stdout); res["sound"] = s
        if s.get("syllable_rate_per_s", 0) > 3.9: res["flags"].append("reading too fast for sleep")
        if s.get("pitch_std_semitones", 9) < 2.0: res["flags"].append("voice sounds monotone")
    except Exception:
        pass
    import soundfile as sf
    y, _ = sf.read(WORK / f"mix_{lang}.wav", dtype="float32")
    res["peak"] = round(float(np.abs(y).max()), 3); res["minutes"] = round(len(y) / SR / 60, 1)
    if res["peak"] >= 0.97: res["flags"].append("clipping")
    if not (9 <= res["minutes"] <= 40): res["flags"].append(f"unusual length {res['minutes']} min")
    try:
        import difflib
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(WORK / f"first_{lang}.wav"), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
        tr = subprocess.run([sys.executable, str(ROOT / "wyoming_transcribe.py"), lang], input=pcm, capture_output=True, timeout=1500).stdout.decode()
        norm = lambda t: re.sub(r"[^a-zäöüß ]", "", t.lower()).split()
        res["script_match"] = round(difflib.SequenceMatcher(None, norm(" ".join(story.get("paragraphs_hess") or story["paragraphs"]) if lang == "de" and DIALECT == "hessisch" else " ".join(story["paragraphs"])), norm(tr)).ratio(), 3)
        if res["script_match"] < (0.70 if lang == "de" and DIALECT == "hessisch" else 0.80): res["flags"].append(f"speech does not match the script ({res['script_match']})")
    except Exception as e:
        res["script_match_error"] = str(e)[:100]
    return res


def slug(date, title):
    t = title.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    return f"{date}-{re.sub(r'[^a-z0-9]+', '-', t).strip('-')}"


FILLERS = {"de": ["Ähm, ", "Na, ", "Hmm, ", "Also, ", "Naja, ", "Äh, "], "en": ["Um, ", "Well, ", "Hmm, ", "So, ", "Uh, ", "You know, "]}
MID_FILLERS = {"de": ["ähm", "äh", "na ja"], "en": ["um", "uh", "you know"]}
NO_FILLER_START = re.compile(r"^(ach|oh|na|hmm|mmm|mmh|psst|pssst|uh|um|well|so|ähm|äh|naja|also|ja|doch|aha|ssch|schsch|aaah|\W)", re.I)


def add_fillers(paras, lang, seed, per_words=None):
    """a real storyteller hesitates a little: sprinkle spoken fillers (ähm, na, hmm / um, well, hmm) at sentence starts and now and then after a comma. Deterministic per story, never in the first paragraph's first sentence."""
    per_words = per_words or float(os.environ.get("FILLER_EVERY", "0"))
    if per_words <= 0:
        return list(paras)
    rng = random.Random(seed); out = []; since = 0; first = True
    for p in paras:
        sents = re.split(r"(?<=[.!?…])\s+", p.strip()); new = []
        for i, sen in enumerate(sents):
            n = len(sen.split()); done = False
            if not (first and i == 0) and n >= 4 and not NO_FILLER_START.match(sen) and not sen.startswith(("„", '"')) and since >= per_words * rng.uniform(0.6, 1.3):
                if rng.random() < 0.25 and "," in sen[: len(sen) // 2]:
                    k = sen.index(",") + 1; sen = sen[:k] + " " + rng.choice(MID_FILLERS[lang]) + "," + sen[k:]
                else:
                    f = rng.choice(FILLERS[lang]); sen = f + sen[0].lower() + sen[1:] if not re.match(r"^[A-ZÄÖÜ][A-ZÄÖÜ]", sen) else f + sen
                since = 0; done = True
            if not done: since += n
            new.append(sen)
        first = False; out.append(" ".join(new))
    return out


def spoken_paragraphs(story, lang):
    hess = lang == "de" and DIALECT == "hessisch" and story.get("paragraphs_hess")
    paras = story["paragraphs_hess"] if hess else story["paragraphs"]
    return add_fillers(paras, lang, seed=hashlib.md5((story["title"] + lang).encode()).hexdigest())


def script_segments(story, lang, st):
    """the episode script as ordered parts: (kind, index, text). kind: fixed | title | para1 (first reading) | para2 (slow second reading)"""
    hess = lang == "de" and DIALECT == "hessisch"
    fixed = HESS_FIXED if hess else LANG[lang]["fixed"]
    paras = spoken_paragraphs(story, lang)
    title = story.get("title_hess") or story["title"]
    tl = (HESS_TITLE_LINE if hess else LANG[lang]["title_line"]).format(title=title)
    segs = [("fixed", "hello", fixed["hello"]), ("fixed", "breath", fixed["breath"]), ("fixed", "relaxed", fixed["relaxed"]), ("title", "title", tl), ("fixed", "twice", fixed["twice"])]
    segs += [("para1", i, p) for i, p in enumerate(paras)] + [("fixed", "second", fixed["second"])] + [("para2", i, p) for i, p in enumerate(paras)] + [("fixed", "bye", fixed["bye"])]
    return segs


def expected_script(story, lang, st):
    """the exact words the finished episode should contain (used by the quality gate to listen back to the final mp3)"""
    return " ".join(t for _, _, t in script_segments(story, lang, st))


def _env_file(path):
    try:
        for l in Path(path).expanduser().read_text().splitlines():
            if "=" in l and not l.startswith("#"):
                k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
    except Exception:
        pass


def notify(title, msg):
    """ntfy-style POST (NTFY_URL) or a Home Assistant notify service (HA_URL + HA_TOKEN, NOTIFY_SERVICE); never raises"""
    try:
        _env_file(os.environ.get("NOTIFY_ENV_FILE", ""))
        if os.environ.get("NTFY_URL"):
            urllib.request.urlopen(urllib.request.Request(os.environ["NTFY_URL"], msg.encode(), {"Title": title.encode("ascii", "ignore").decode() or "podcast"}), timeout=15)
        elif os.environ.get("HA_URL") and os.environ.get("HA_TOKEN"):
            svc = os.environ.get("NOTIFY_SERVICE", "notify.notify").replace(".", "/")
            urllib.request.urlopen(urllib.request.Request(os.environ["HA_URL"].rstrip("/") + "/api/services/" + svc, json.dumps({"title": title, "message": msg}).encode(),
                                   {"Authorization": "Bearer " + os.environ["HA_TOKEN"], "Content-Type": "application/json"}), timeout=15)
    except Exception as e:
        log("notify failed:", str(e)[:80])


def publish(day_dir):
    """move a staged, quality-checked episode into public/, add it to the history and rebuild the feeds"""
    import shutil
    entry = json.loads((day_dir / "entry.json").read_text())
    hist = json.loads(HIST.read_text()) if HIST.exists() else []
    for l in ("de", "en"):
        if l not in entry: continue
        for sub in ("episodes", "icons"):
            src = day_dir / l / sub
            if src.exists():
                (PUB / l / sub).mkdir(parents=True, exist_ok=True)
                for f in src.iterdir(): shutil.move(str(f), PUB / l / sub / f.name)
    entry["n"] = max([h["n"] for h in hist], default=0) + 1
    hist.append(entry); HIST.write_text(json.dumps(hist, ensure_ascii=False, indent=1))
    subprocess.run([sys.executable, str(ROOT / "feed.py")], check=False)
    (day_dir / "PUBLISHED").write_text(dt.datetime.now().isoformat())
    log("published", day_dir.name)


AI_LINE = {"de": "Hinweis: Die Geschichten werden von einer KI geschrieben und von einer KI-generierten Stimme gelesen. Auch die Musik ist KI-generiert.",
           "en": "Note: the stories are written by an AI and read by an AI-generated voice. The music is also AI-generated."}


def fix_text(story_l, lang, problems, st):
    """rewrite the story so that the concrete problems the judge named are gone (same length, same calm ending)"""
    sysmsg = (f"Du bist Lektorin für Einschlafgeschichten für Kleinkinder ({st['label']}). Der Text hat diese Probleme: {problems}\n"
              "Behebe genau diese Probleme: unsinnige oder falsche Sätze, Grammatikfehler, fremdsprachige Wörter, Schleifen, Unpassendes oder Gruseliges. "
              f"Behalte Handlung, Länge (±10 Prozent), Absätze (Leerzeilen), einfache kurze Sätze (höchstens {st['sent']} Wörter) und das ruhige Einschlafen am Ende. "
              + ("Sprache: Deutsch. " if lang == "de" else "Language: English only, simple and warm. ") + "Gib NUR den verbesserten Text aus, ohne Titel, ohne Kommentar.")
    new = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", clean_text(ollama(sysmsg, "\n\n".join(story_l["paragraphs"]), temp=0.4))) if p.strip()]
    new = regroup(words_out(new, NUM_DE if lang == "de" else NUM_EN)) if lang == "de" else words_out(new, NUM_EN)
    n0, n1 = len(" ".join(story_l["paragraphs"]).split()), len(" ".join(new).split())
    if 0.8 * n0 <= n1 <= 1.25 * n0 and len(new) >= 3:
        story_l = dict(story_l, paragraphs=new, words=n1); story_l.pop("paragraphs_hess", None)
        return story_l
    return None


def repair(lang, rep, story, st, state, entry, art_big):
    """turn a failed quality report into concrete fixes, cheapest first. returns what was changed ([] = nothing left to try, the episode is held)"""
    import qc
    fails = {c["name"]: c for c in rep["checks"] if c["status"] == "fail"}
    o = state["opts"]; done = []
    if "loudness" in fails or "true peak" in fails:
        if not o.get("norm"): o["norm"] = True; done.append("loudness-normalise with a true-peak limiter")
    if "clicks" in fails and not o.get("declick"):
        o["declick"] = True; done.append("remove clicks")
    if "dead air" in fails and not o.get("trim"):
        o["trim"] = True; done.append("cut long silent stretches in the voice")
    if "mp3 complete" in fails or "cover art" in fails:
        done.append("re-encode the mp3 with the cover")
    if "clear under music" in fails and not any(n in fails for n in ("script match", "garbled passages", "stray speech")):
        o["music"] = min(o.get("music", 1.0), 0.45); o["sfx2"] = False; done.append("quieter music and no sound effects in the second reading (the voice itself is fine)")
    if any(n in fails for n in ("script match", "garbled passages", "stray speech")):
        bad = (fails.get("garbled passages") or {}).get("data") or []
        if not state.get("quiet_music"):
            o["music"] = 0.55; o["tempo2"] = 0.94; state["quiet_music"] = True; done.append("quieter music under the voice and a less slowed second reading")
        elif not state.get("quieter2"):
            o["music"] = 0.4; o["sfx2"] = False; state["quieter2"] = True
            done.append("even quieter music and no sound effects in the slow second reading (the voice alone is clear)")
        elif bad:
            segs = script_segments(story[lang], lang, st); pos = 0; spans = []
            for kind, idx, text in segs:
                n = len(qc._words(text)); spans.append((pos, pos + n, kind, idx, text)); pos += n
            touched = set()
            for k in bad:
                for a0, a1, kind, idx, text in spans:
                    if a0 < k + 25 and k < a1:
                        if kind in ("para1", "para2"):
                            for ch in chunks_of(text):       # re-record only the passages around the bad words
                                touched.add(ch)
                        else:
                            touched.add(text)
            for t in touched: bump_salt(lang, t)
            for f in WORK.glob(f"fixed_{lang}_*.npz"): f.unlink()
            done.append(f"re-record {len(touched)} passage(s) with a new take")
        else:
            for kind, idx, text in script_segments(story[lang], lang, st):
                if kind != "fixed":
                    for ch in chunks_of(text): bump_salt(lang, ch)
            done.append("re-record the whole reading with new takes")
    if ("steady voice" in fails) and not state.get("rerecorded"):
        state["rerecorded"] = True
        for kind, idx, text in script_segments(story[lang], lang, st):
            if kind == "para1":
                for ch in chunks_of(text): bump_salt(lang, ch)
        done.append("re-record the reading (the voice drifted)")
    if ("meaning" in fails or "safety" in fails) and not state.get("text_fixed"):
        state["text_fixed"] = True
        problems = (fails.get("meaning") or fails.get("safety"))["detail"]
        new = fix_text(story[lang], lang, problems, st)
        if new:
            story[lang] = new
            if lang == "de" and "en" in story and story["en"]:   # keep the English translation in step with the fixed German story
                try: story["en"] = translate(new)
                except SystemExit: pass
            entry[lang]["blurb"] = episode_blurb(story[lang], lang, st)
            done.append("rewrite the story to remove the judge's problems")
    # metadata
    child_d = child()
    if "privacy" in fails:
        for k in ("title", "description", "blurb"):
            for v in (child_d.get("name"), child_d.get("birthdate")):
                if v: entry[lang][k] = re.sub(re.escape(v), "dein Kind" if lang == "de" else "your child", entry[lang][k], flags=re.I)
        done.append("remove the child's details from the public text")
    if "AI disclosure" in fails:
        for k in ("description", "blurb"):
            if AI_LINE[lang] not in entry[lang].get(k, ""): entry[lang][k] = (entry[lang].get(k, "") + " " + AI_LINE[lang]).strip()
        done.append("add the AI disclosure line")
    if "title spelling" in fails:
        fix = fails["title spelling"]["detail"]
        m = re.search(r"FIX:\s*(.+?)\s*->\s*(.+)", fix)
        if m:
            wrong, right = m.group(1).strip(" '\""), m.group(2).strip(" '\"")
            for key in ("title",):
                story[lang][key] = story[lang][key].replace(wrong, right)
            for k in ("description", "blurb"): entry[lang][k] = entry[lang].get(k, "").replace(wrong, right)
            entry[lang]["title"] = story[lang]["title"]
            done.append(f"correct the spelling '{wrong}' -> '{right}'")
    if "title/description" in fails:
        entry[lang]["description"] = story[lang].get("description") or episode_blurb(story[lang], lang, st); done.append("rewrite the description")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme"); ap.add_argument("--short", action="store_true"); ap.add_argument("--dry-story", action="store_true")
    ap.add_argument("--no-qa", action="store_true"); ap.add_argument("--form", choices=["du", "figur"])
    ap.add_argument("--langs", default="de,en"); ap.add_argument("--from-last", action="store_true", help="reuse work/last_story.json (skip story writing)"); ap.add_argument("--age-months", type=int)
    a = ap.parse_args()
    langs = a.langs.split(",")
    for l in langs:
        (PUB / l / "episodes").mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    hist = json.loads(HIST.read_text()) if HIST.exists() else []
    months = age_months(a.age_months); st = stage_for(months)
    month = dt.date.today().month; themes = []
    for l in (ROOT / "themes.txt").read_text().splitlines():
        l = l.strip()
        if not l or l.startswith("#"): continue
        if "|" in l.split(" ")[0]:   # "11,12|theme": only in those months
            m, l = l.split("|", 1)
            if month not in [int(x) for x in m.split(",")]: continue
        themes.append(l)
    used = {h["theme"] for h in hist[-(len(themes) - 1):]}
    theme = a.theme or random.Random(dt.date.today().toordinal()).choice([t for t in themes if t not in used] or themes)
    used_s = [h.get("structure") for h in hist[-(len(STRUCTURES) - 1):]]
    structure = random.Random(dt.date.today().toordinal() * 7).choice([k for k in STRUCTURES if k not in used_s] or list(STRUCTURES))
    log(f"child age {months} months -> stage {st['label']} | theme: {theme} | form: {structure}")
    name = child().get("name")
    story = {}
    if a.from_last:
        story = json.loads((WORK / "last_story.json").read_text()); theme = story.get("theme", theme)
    for _ in range(0 if a.from_last else 3):
        s_de = make_story_de(theme, [h["de"]["title"] for h in hist], st, name, a.short, a.form, structure)
        s_en = translate(s_de) if "en" in langs else None
        if safety_check(s_de, "de") and (s_en is None or safety_check(s_en, "en")):
            story = {"de": s_de, "en": s_en}; break
    if not story:
        raise SystemExit("no story passed the safety checks")
    log("title:", story["de"]["title"], "/", (story["en"] or {}).get("title"), "|", story["de"]["words"], "words")
    if DIALECT == "hessisch" and "de" in langs and "paragraphs_hess" not in story["de"]:
        ph, th = to_hessisch(story["de"])
        if ph: story["de"]["paragraphs_hess"] = ph; story["de"]["title_hess"] = th
    (WORK / "last_story.json").write_text(json.dumps(dict(story, theme=theme), ensure_ascii=False, indent=1))
    if a.dry_story:
        for l in langs:
            print(f"\n=== {l}: {story[l]['title']}\n" + "\n\n".join(story[l]["paragraphs"]))
        return
    art_big = art_icon = None
    try:
        import art
        scene, obj = scene_prompts(story["de"]); log("art:", scene[:90], "|", obj)
        art_big, art_icon = art.episode_art(scene, obj)
    except Exception as e:
        log("artwork failed (show cover is used instead):", str(e)[:120])
    number = max([h["n"] for h in hist], default=0) + 1; today = str(dt.date.today())
    entry = dict(n=number, date=today, theme=theme, structure=structure, age_months=months, stage=st["label"], voice=VOICE)
    day = STAGE / today; import shutil
    if day.exists(): shutil.rmtree(day)
    for l in langs:
        (day / l / "episodes").mkdir(parents=True, exist_ok=True)
    reports = {}; MAX_TRIES = int(os.environ.get("QC_TRIES", "4"))
    ollama_unload()   # story work is done: the voice gets the GPU alone
    for l in langs:
        state = {"opts": {}}
        for attempt in range(1, MAX_TRIES + 1):
            ollama_unload()
            log(f"=== building {l} (attempt {attempt})")
            mix = build(story[l], l, st, state["opts"])
            mp3 = day / l / "episodes" / f"{slug(today, story[l]['title'])}.mp3"
            cover = None
            if art_big is not None:
                cover = mp3.with_suffix(".jpg"); art_big.save(cover, quality=90)
                (day / l / "icons").mkdir(parents=True, exist_ok=True)
                art_icon.save(day / l / "icons" / (mp3.stem + ".png")); art_icon.resize((256, 256), __import__("PIL.Image", fromlist=["Image"]).NEAREST).save(day / l / "icons" / (mp3.stem + ".preview.png"))
            encode(l, mp3, story[l], number, cover, state["opts"])
            if l not in entry:
                entry[l] = dict(description=story[l]["description"], blurb=episode_blurb(story[l], l, st), stage=st["label"])
            entry[l].update(title=story[l]["title"], file=mp3.name, bytes=mp3.stat().st_size, seconds=round(len(mix) / SR), qa=[], script_match=None)
            (day / "entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1))
            if a.no_qa:
                break
            log(f"quality gate [{l}] ...")
            tts_unload()
            import qc
            reports[l] = qc.run(l, mp3, WORK / f"mix_{l}.wav", WORK / f"first_{l}.wav", cover or (PUB / l / "cover.jpg"), entry[l], expected_script(story[l], l, st), st["label"], ollama,
                                safety_check, story[l], child(), lo_min=6 if a.short else 9, hi_min=40, dialect=(l == "de" and DIALECT == "hessisch"),
                                require_ai=bool(os.environ.get("QC_REQUIRE_AI")), voice_wav=WORK / f"voice_{l}.wav")
            (day / f"qc_{l}_attempt{attempt}.json").write_text(json.dumps(reports[l], ensure_ascii=False, indent=1))
            (day / f"qc_{l}.json").write_text(json.dumps(reports[l], ensure_ascii=False, indent=1))
            log(qc.format_report(reports[l]))
            if reports[l]["verdict"] == "PASS":
                break
            if attempt == MAX_TRIES:
                break
            fixes = repair(l, reports[l], story, st, state, entry, art_big)
            ollama_unload()
            if not fixes:
                log(f"auto-repair [{l}]: nothing more to try"); break
            log(f"auto-repair [{l}]: " + "; ".join(fixes))
            (WORK / "last_story.json").write_text(json.dumps(dict(story, theme=theme), ensure_ascii=False, indent=1))
    failed = [l for l, r in reports.items() if r["verdict"] != "PASS"]
    if failed:
        why = "; ".join(f"{l}: " + ", ".join(c["name"] for c in reports[l]["checks"] if c["status"] == "fail") for l in failed)
        log(f"HELD: quality gate failed ({why}). Episode stays in {day}; nothing was published.")
        notify("Podcast episode held back", f"Today's episode did not pass the quality check ({why}). Nothing was published. Report: {day}/qc_*.json")
        raise SystemExit(2)
    if os.environ.get("MANUAL_APPROVAL"):
        log(f"quality gate passed; waiting for your approval: python publish.py {today}")
        notify("Podcast episode ready for review", f"Passed all automatic checks. Listen to it in {day} and approve with: python publish.py {today}")
        return
    publish(day)
    notify("Podcast episode published", f"{entry['de']['title'] if 'de' in entry else ''} / {entry['en']['title'] if 'en' in entry else ''} passed the quality gate and is live.")
    log("done")


if __name__ == "__main__":
    main()
