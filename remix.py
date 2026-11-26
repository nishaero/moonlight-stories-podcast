"""Rebuild the music mix of the newest episode from the cached voice (no re-reading): REUSE_FIRST=1 remix.py"""
import os, sys, json, subprocess
os.environ["REUSE_FIRST"] = "1"; sys.argv = [sys.argv[0]]
import show
story = json.load(open(show.WORK / "last_story.json")); hist = json.load(open(show.HIST)); h = hist[-1]
st = show.stage_for(h["age_months"])
for l in ("de", "en"):
    mix = show.build(story[l], l, st)
    mp3 = show.PUB / l / "episodes" / h[l]["file"]
    show.encode(l, mp3, story[l], h["n"], mp3.with_suffix(".jpg"))
    h[l]["bytes"] = mp3.stat().st_size; h[l]["seconds"] = round(len(mix) / show.SR); print(l, h[l]["seconds"], "s", h[l]["bytes"])
json.dump(hist, open(show.HIST, "w"), ensure_ascii=False, indent=1)
subprocess.run([sys.executable, str(show.ROOT / "feed.py")])
