#!/usr/bin/env python3
"""Describe how an audio file SOUNDS from the signal itself (no transcript): pace, pauses, pitch movement, loudness, brightness.
usage: audio_profile.py FILE [--start SEC] [--dur SEC] [--json]
Runs in ~/.venvs/audiofeat or ~/.venvs/yoto-show (librosa)."""
import sys, json, argparse, numpy as np, librosa
ap=argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--start",type=float,default=0); ap.add_argument("--dur",type=float,default=180); ap.add_argument("--json",action="store_true")
a=ap.parse_args()
y,sr=librosa.load(a.file,sr=22050,mono=True,offset=a.start,duration=a.dur)
dur=len(y)/sr
# speech vs pause (relative to the clip's own loudness)
iv=librosa.effects.split(y,top_db=32,frame_length=1024,hop_length=256)
speech=sum((e-s) for s,e in iv)/sr
gaps=[(iv[i+1][0]-iv[i][1])/sr for i in range(len(iv)-1)]
gaps=np.array(gaps) if gaps else np.array([0.0])
# syllable-rate proxy: energy-envelope peaks inside speech
env=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)[0]; env=env/(env.max()+1e-9)
from scipy.signal import find_peaks, savgol_filter
sm=savgol_filter(env,9,2); pk,_=find_peaks(sm,height=0.08,distance=int(0.12*sr/256),prominence=0.03)
syl_rate=len(pk)/max(speech,1e-6)
# pitch (voiced frames only), 65-450 Hz covers female+male
f0,vf,_=librosa.pyin(y,fmin=65,fmax=450,sr=sr,frame_length=2048,hop_length=512)
f=f0[~np.isnan(f0)]
if len(f)>20:
    st=12*np.log2(f/np.median(f))
    pitch=dict(f0_median_hz=float(np.median(f)),f0_p10_hz=float(np.percentile(f,10)),f0_p90_hz=float(np.percentile(f,90)),
               pitch_std_semitones=float(np.std(st)),pitch_range_semitones=float(np.percentile(st,95)-np.percentile(st,5)),
               voiced_ratio=float(np.mean(~np.isnan(f0))))
    # pitch movement speed: mean abs change per second (semitones/s) - flat/monotone voices are low
    d=np.abs(np.diff(st))*(sr/512); pitch["pitch_movement_st_per_s"]=float(np.mean(d[d<40]))
else: pitch={}
# loudness
rms=librosa.feature.rms(y=y)[0]; db=20*np.log10(rms+1e-9)
cent=librosa.feature.spectral_centroid(y=y,sr=sr)[0]
out=dict(file=a.file,seconds=round(dur,1),speech_ratio=round(speech/dur,3),
  pause_count_per_min=round(len(gaps)/dur*60,1),median_pause_s=round(float(np.median(gaps)),2),long_pauses_over_1s_per_min=round(float((gaps>1.0).sum())/dur*60,1),
  syllable_rate_per_s=round(float(syl_rate),2),loudness_mean_db=round(float(db.mean()),1),loudness_dyn_range_db=round(float(np.percentile(db,95)-np.percentile(db,10)),1),
  brightness_hz=round(float(cent.mean())),**{k:round(v,2) for k,v in pitch.items()})
print(json.dumps(out) if a.json else "\n".join(f"{k:28s} {v}" for k,v in out.items()))
