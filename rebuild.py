"""rebuild one staged language with given mix options from cached voice takes, then re-run the quality gate. usage: rebuild.py LANG [music=0.4] [sfx2=0] [tempo2=0.94]"""
import os, sys, json
args=sys.argv[1:]; sys.argv=[sys.argv[0]]; os.environ.setdefault("TTS","qwen")
import show, qc
l=args[0]; opts={}
for a in args[1:]:
    k,v=a.split("="); opts[k]=(bool(int(v)) if k in ("sfx2","trim","norm","declick") else float(v))
day=sorted(p for p in show.STAGE.glob("*") if p.is_dir())[-1]
entry=json.load(open(day/"entry.json")); story=json.load(open(show.WORK/"last_story.json")); st=show.stage_for(entry["age_months"])
mix=show.build(story[l], l, st, opts)
mp3=day/l/"episodes"/entry[l]["file"]; show.encode(l, mp3, story[l], entry["n"], mp3.with_suffix(".jpg"), opts)
entry[l].update(bytes=mp3.stat().st_size, seconds=round(len(mix)/show.SR)); json.dump(entry,open(day/"entry.json","w"),ensure_ascii=False,indent=1)
show.ollama_unload()
rep=qc.run(l, mp3, show.WORK/f"mix_{l}.wav", show.WORK/f"first_{l}.wav", mp3.with_suffix(".jpg"), entry[l], show.expected_script(story[l],l,st), st["label"], show.ollama, show.safety_check, story[l], show.child(), voice_wav=show.WORK/f"voice_{l}.wav")
(day/f"qc_{l}.json").write_text(json.dumps(rep,ensure_ascii=False,indent=1)); print(qc.format_report(rep),flush=True)
