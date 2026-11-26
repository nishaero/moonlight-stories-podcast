#!/usr/bin/env python3
"""build public/<lang>/feed.rss (de, en) from history.json. Base URL: PODCAST_BASE_URL (default: LAN address)."""
import json, os, html, email.utils, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parent; PUB = ROOT / "public"
BASE = os.environ.get("PODCAST_BASE_URL", "https://podcast.example.com").rstrip("/")
hist = json.loads((ROOT / "history.json").read_text()) if (ROOT / "history.json").exists() else []
KEEP_DAYS = 3
if len(hist) > KEEP_DAYS:   # one entry per day: drop the older ones and delete their audio, art and icons
    for h in hist[:-KEEP_DAYS]:
        for lang in ("de", "en"):
            if lang in h:
                stem = h[lang]["file"][:-4]
                for p in (PUB/lang/"episodes"/f"{stem}.mp3", PUB/lang/"episodes"/f"{stem}.jpg", PUB/lang/"icons"/f"{stem}.png", PUB/lang/"icons"/f"{stem}.preview.png"):
                    p.unlink(missing_ok=True)
    hist = hist[-KEEP_DAYS:]
    (ROOT / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1))
META = {
 "de": ("Mondlichtgeschichten",
        "Jeden Abend eine neue, ruhige Einschlafgeschichte für kleine Kinder.\n\n"
        "So ist jede Folge aufgebaut: Zuerst eine kleine Dehn- und Atemübung zum Runterkommen. Dann erzählt eine warme, ruhige Stimme die Geschichte. "
        "Danach folgt dieselbe Geschichte noch einmal, noch langsamer, damit dein Kind dabei in den Schlaf gleiten kann. Sanfte, selbst erzeugte Musik begleitet das Ganze.\n\n"
        "Die Geschichten wachsen mit deinem Kind: Anfangs sehr kurz und einfach, mit vielen Wiederholungen und vertrauten Dingen, später länger, mit mehr Fantasie und kleinen Naturwundern. "
        "Es gibt nichts Gruseliges, keine Bösewichte und keine Werbung.\n\n"
        "Es gibt jeden Tag eine neue Folge, auch auf Englisch (Moonlight Stories). Die Geschichten und die Stimme werden lokal mit KI erzeugt.",
        "Ruhige Einschlafgeschichten für kleine Kinder, jeden Abend neu - mit Atemübung und zweimal erzählt.", "de"),
 "en": ("Moonlight Stories",
        "A new, calm bedtime story for little children every evening.\n\n"
        "Every episode follows the same gentle routine: a short stretch and breathing exercise to settle down, then a warm, quiet voice tells the story. "
        "After that the very same story is told once more, even more slowly, so your child can drift off to sleep while listening. Soft, generated music plays underneath.\n\n"
        "The stories grow with your child: very short and simple at first, with lots of repetition and familiar things, later longer, with more imagination and little wonders of nature. "
        "Nothing scary, no villains and no advertising.\n\n"
        "A new episode every day, also in German (Mondlichtgeschichten). The stories and the voice are created locally with AI.",
        "Calm bedtime stories for little children, new every evening - with a breathing exercise and told twice.", "en")}
FORMAT = {"de": "Aufbau: Atemübung, die Geschichte, danach noch einmal langsamer. Dauer etwa {m} Minuten. Geeignet für Kinder {stage}.",
          "en": "Format: breathing exercise, the story, then the story again, slower. About {m} minutes long. Suited to children aged {stage}."}
for lang, (title, desc, sub, code) in META.items():
    items = []
    for h in reversed(hist):
        e = h.get(lang)
        if not e: continue
        d = dt.datetime.fromisoformat(h["date"] + "T18:00:00+01:00")
        body = (e.get('blurb') or e['description']) + "\n\n" + FORMAT[lang].format(m=max(1, round(e['seconds'] / 60)), stage=e.get('stage', h.get('stage', '')).replace('Jahre', 'Jahren') if lang == 'de' else e.get('stage', h.get('stage', '')).replace('Jahre', 'years'))
        items.append(f"""<item><title>{html.escape(e['title'])}</title><description>{html.escape(body)}</description><itunes:subtitle>{html.escape((e.get('blurb') or e['description'])[:200])}</itunes:subtitle><itunes:summary>{html.escape(body)}</itunes:summary>
<pubDate>{email.utils.format_datetime(d)}</pubDate><guid isPermaLink="false">mondlicht-{lang}-{h['date']}-{h['n']}</guid>
<enclosure url="{BASE}/{lang}/episodes/{e['file']}" length="{e['bytes']}" type="audio/mpeg"/>
<itunes:image href="{BASE}/{lang}/episodes/{e['file'][:-4]}.jpg"/><itunes:duration>{e['seconds']}</itunes:duration><itunes:episode>{h['n']}</itunes:episode><itunes:explicit>false</itunes:explicit></item>""")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>
<title>{title}</title><link>{BASE}/{lang}/</link><language>{code}</language><description>{html.escape(desc)}</description><itunes:subtitle>{html.escape(sub)}</itunes:subtitle><itunes:summary>{html.escape(desc)}</itunes:summary>
<itunes:author>{title}</itunes:author><itunes:explicit>false</itunes:explicit><itunes:category text="Kids &amp; Family"/>
<itunes:image href="{BASE}/{lang}/cover.jpg"/><image><url>{BASE}/{lang}/cover.jpg</url><title>{title}</title><link>{BASE}/{lang}/</link></image>
{''.join(items)}
</channel></rss>"""
    (PUB / lang / "feed.rss").write_text(xml); print(f"{lang}/feed.rss: {len(items)} episodes")
