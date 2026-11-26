# Moonlight Stories: a local, bilingual bedtime-story podcast

A daily bedtime podcast for small children, generated entirely on your own machine and served as an ordinary RSS feed, so it works with apps and players that accept a podcast URL (for example the Yoto player). German and English feeds, a new story every day, written for the age of your child and read in a calm human-sounding voice with original music.

Not affiliated with Yoto or any podcast it takes inspiration from.

## What it does
1. **Story:** a local LLM (via Ollama) writes a short, relatable, calm story for the child's age stage (word count, sentence length and learning thread grow with the age), then an edit pass and a safety check.
2. **Translation:** the German master story is translated paragraph by paragraph into English (same pauses).
3. **Voice:** Qwen3-TTS voice cloning from a designed reference voice (a "design once, clone for every sentence" workflow). Every sentence is transcribed back with Whisper and re-recorded when it does not match the text.
4. **Show format:** breathing intro, the story, the same story again more slowly, goodbye. Original generated music: theme, transitions, a sparse bed that ducks under the voice, an outro.
5. **Quality control:** the audio is profiled from the signal (pace, pitch movement, clipping, length), so a flat or rushed episode is flagged without anyone listening.
6. **Artwork:** show covers, per-episode art and 16x16 pixel icons (for players with a small display) via a local ComfyUI SDXL-Turbo workflow.
7. **Feed:** `feed.py` writes `public/<lang>/feed.rss` (RSS 2.0 + iTunes tags) and only the last 3 days of episodes are kept.

## Requirements
- A GPU machine (developed on an AMD Strix Halo with ROCm; CUDA works with the matching torch), Python 3.12+, ffmpeg
- [Ollama](https://ollama.com) with a capable instruction model (set `MODEL`, default `qwen3.8:27b`)
- A Wyoming-protocol Whisper service on `127.0.0.1:10300` (for the listen-back check; e.g. `rhasspy/wyoming-whisper`)
- Optional: ComfyUI with an SDXL-Turbo checkpoint on `127.0.0.1:8188` (artwork); the workflow is `comfy_workflow_sdxl_turbo.json`

## Setup
```bash
./install_gpu.sh                      # venv + torch + dependencies (edit for your GPU)
cp child.example.json child.json      # optional: child's first name and birth date (stays local, git-ignored)
python design_voices.py               # designs candidate narrator voices into voices/cand/ (listen, pick one)
# copy the chosen clip to voices/ref_<name>.wav and add its text under "texts" in voices/refs.json,
# then set QWEN_REF=<name>
PODCAST_BASE_URL=https://podcast.example.com python show.py     # one episode in both languages
```
Useful flags: `--theme "..."`, `--short`, `--langs de`, `--age-months N`, `--dry-story` (story only), `--from-last`.
Settings through environment variables: `QWEN_REF`, `QWEN_SLOW` (extra slow-down), `TTS=xtts` (CPU fallback engine), `MODEL`, `OLLAMA`, `DIALECT_DE=hessisch` (optional mild regional colouring of the German text).

## Publishing the feed
Serve `public/` over HTTPS at a stable URL (see `docker-compose.example.yml`) and give players `https://<your-host>/de/feed.rss` and `https://<your-host>/en/feed.rss`. Some podcast fetchers cannot pass bot-protection pages, so do not put a login or JavaScript challenge in front of the feed paths. Run daily with a systemd timer (`mondlicht-podcast.service.example`).


## Quality gate and daily operation
Every episode is built into `staging/<date>/` and published only after `qc.py` passes (loudness and peak, dead air and clicks, two listen-backs with Whisper of the clean voice track and of the finished mp3, steady voice, an LLM judge on the script, safety, title spelling, privacy). Failures are repaired automatically (up to 4 tries): normalise, de-click, trim silences, quieter music, re-record only the bad passages with a new take, rewrite the story, correct metadata. `MANUAL_APPROVAL=1` holds even passing episodes until `python publish.py` (or `--reject`). `rebuild.py` and `recheck.py` re-run mixing and the gate from cached voice takes.

## Notes for small players such as the Yoto Mini
- Serve the feed and audio over HTTPS with an RSA certificate and a short chain; plain 128 kbps stereo MP3 without an embedded picture streams fine.
- A device on an isolated (IoT/guest) Wi-Fi may not reach your own public address through the router (NAT loopback). Put it on the main network or host the audio elsewhere.
- Podcast apps and players cache by episode GUID and URL: when you rename files, remove and re-add the podcast in the player.
- AI-generated shows: Apple Podcasts asks for disclosure in the content and metadata; `feed.py` adds a note to every episode and the spoken intro says it.

## Privacy and safety
- The child's name and birth date live only in `child.json` (git-ignored, mode 600 recommended); stories never contain a surname, address or other details.
- Everything runs locally; the only outbound traffic is downloading models and serving your feed.
- Stories pass a safety check (no scary or violent content) before anything is voiced; nothing is published automatically except through your own web server.

## Contributing
Improvements are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are accepted under the MIT licence.

## Licences
Code: MIT. Check each model's licence before use: Qwen3-TTS, your Ollama model, Whisper and SDXL-Turbo each have their own terms. The optional XTTS-v2 fallback is under a personal non-commercial licence. The music is generated procedurally by `music.py` from original melodies.
