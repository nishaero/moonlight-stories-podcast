import sys,torch,librosa,glob,json
try:
    pred=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True)
except Exception as e:
    print("UTMOS load failed:",str(e)[:200]); sys.exit(1)
res={}
for f in sorted(sys.argv[1:]):
    y,sr=librosa.load(f,sr=16000,mono=True)
    with torch.no_grad(): s=pred(torch.from_numpy(y).unsqueeze(0),sr).item()
    res[f]=round(s,2); print(f"{s:5.2f}  {f}",flush=True)
json.dump(res,open("mos_scores.json","w"),indent=1)
