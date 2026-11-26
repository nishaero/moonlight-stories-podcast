# Contributing

Improvements are welcome: better voices, more natural delivery, new story themes, other languages, other TTS engines, bug fixes, docs.

## Licence of contributions
This project is under the [MIT licence](LICENSE). By opening a pull request you agree that your contribution is released under the same licence (inbound = outbound), and that you have the right to contribute it.

## Ground rules
- **No personal data in the repo:** no real child names or birth dates, no addresses, hostnames, IP addresses, tokens or `.env` files. `child.json`, `history.json`, generated episodes and audio are git-ignored on purpose.
- **Child safety first:** story prompts, themes and changes to the safety check must keep content calm and age-appropriate (no scary, violent or manipulative content, no brands or ads).
- **Licences of assets:** do not add voice recordings, music samples or images you do not own. Music in this project is generated procedurally from original melodies; keep it that way or state the licence clearly.
- **Models and engines:** note the licence of any new model or dataset you add (some TTS models are non-commercial only).

## How to contribute
1. Open an issue first for larger changes (new engine, new language, structure changes).
2. Fork, create a branch, keep changes focused, run `python -m py_compile *.py`.
3. For voice or story changes, describe how you checked them (listen-back match score, `audio_profile.py` numbers, a short sample).
4. Open a pull request with a short description of what changed and why.

## Ideas that would help most
- A more human, expressive narrator (pitch variation, breathing, pacing) and a way to measure it.
- More relatable story themes for ages 2 to 6, and other languages.
- Support for other TTS engines and GPUs (CUDA, Apple silicon).
- Tests for the feed builder and the story checks.
