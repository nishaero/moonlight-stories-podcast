"""approve (publish) or reject a held episode.  usage: publish.py [YYYY-MM-DD]   (default: newest staged day that is not yet published)
after a human listen when MANUAL_APPROVAL=1, or after fixing a held episode.  --reject deletes the staged episode."""
import sys, shutil
sys.argv, args = [sys.argv[0]], sys.argv[1:]
import show
days = sorted(d for d in show.STAGE.glob("*") if d.is_dir() and not (d / "PUBLISHED").exists())
pick = next((d for d in days if d.name in args), days[-1] if days else None)
if not pick: raise SystemExit("nothing staged")
if "--reject" in args:
    shutil.rmtree(pick); print("rejected", pick.name)
else:
    show.publish(pick); print("published", pick.name)
