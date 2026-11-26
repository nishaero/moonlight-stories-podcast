"""re-run the quality gate on a staged episode without rebuilding: recheck.py [lang ...]  (uses work/last_story.json and staging/<newest>)"""
import os, sys, json, glob
args=sys.argv[1:]; sys.argv=[sys.argv[0]]; os.environ.setdefault("TTS","qwen")
import show, qc
day=sorted(p for p in show.STAGE.glob("*") if p.is_dir())[-1]
entry=json.load(open(day/"entry.json")); story=json.load(open(show.WORK/"last_story.json")); st=show.stage_for(entry["age_months"])
for l in (args or ["de","en"]):
    mp3=day/l/"episodes"/entry[l]["file"]
    rep=qc.run(l, mp3, show.WORK/f"mix_{l}.wav", show.WORK/f"first_{l}.wav", mp3.with_suffix(".jpg"), entry[l], show.expected_script(story[l],l,st), st["label"], show.ollama, show.safety_check, story[l], show.child(), voice_wav=show.WORK/f"voice_{l}.wav",
               require_ai=bool(os.environ.get("QC_REQUIRE_AI")))
    (day/f"qc_{l}.json").write_text(json.dumps(rep,ensure_ascii=False,indent=1)); print(qc.format_report(rep),flush=True)
