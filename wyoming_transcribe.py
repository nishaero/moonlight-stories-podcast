#!/usr/bin/env python3
"""read raw 16 kHz mono s16le PCM on stdin, print the Whisper (Wyoming, workstation) transcript. usage: wyoming_transcribe.py LANG"""
import socket, json, sys
pcm = sys.stdin.buffer.read(); lang = sys.argv[1] if len(sys.argv) > 1 else "de"
s = socket.create_connection(("127.0.0.1", 10300), timeout=1500)
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
print(json.loads(body)["text"] if body else "")
