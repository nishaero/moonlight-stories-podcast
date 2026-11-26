#!/usr/bin/env python3
"""Artwork for the show, generated locally with ComfyUI (SDXL-Turbo).
  art.py show            -> public/<lang>/cover.jpg (1400x1400 podcast cover), yoto_card.jpg (638x1011 MYO card cover), icons/show.png (16x16)
  art.py episode LANG SLUG "scene in English" "pixel object" -> public/<lang>/episodes/SLUG.jpg (square art) + icons/SLUG.png (16x16 Yoto Mini icon)
Yoto Mini / Player show a 16x16 pixel icon per track; podcast apps and the Yoto card use the big art."""
import json, os, sys, time, urllib.request, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
ROOT = Path(__file__).resolve().parent; PUB = ROOT / "public"
U = "http://127.0.0.1:8188"; WF = Path(os.environ.get("COMFY_WORKFLOW", Path(__file__).resolve().parent / "comfy_workflow_sdxl_turbo.json"))
STYLE = ("gentle children's picture-book illustration, soft watercolor and paper-cut layers, deep indigo night sky, soft glowing light, "
         "warm cozy colors, calm and dreamy, very simple shapes, high quality, no text, no letters")
NEG = "text, letters, words, watermark, logo, scary, dark shadows, creepy, realistic photo, ugly, deformed, blurry, noisy, clutter, people faces"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"; FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


def comfy(prompt, neg, w, h, seed, steps=6, cfg=1.2):
    wf = json.load(open(WF))
    wf["2"]["inputs"]["text"] = prompt; wf["3"]["inputs"]["text"] = neg
    wf["4"]["inputs"].update(width=w, height=h); wf["5"]["inputs"].update(seed=seed, steps=steps, cfg=cfg); wf["7"]["inputs"]["filename_prefix"] = "yotoshow"
    pid = json.load(urllib.request.urlopen(urllib.request.Request(U + "/prompt", json.dumps({"prompt": wf}).encode(), {"Content-Type": "application/json"})))["prompt_id"]
    for _ in range(240):
        hist = json.load(urllib.request.urlopen(U + "/history/" + pid))
        if pid in hist: break
        time.sleep(1)
    im = hist[pid]["outputs"]["7"]["images"][0]
    data = urllib.request.urlopen(U + "/view?filename=%s&subfolder=%s&type=%s" % (im["filename"], im["subfolder"], im["type"])).read()
    from io import BytesIO
    return Image.open(BytesIO(data)).convert("RGB")


def finish(im, size):
    im = im.resize(size, Image.LANCZOS)
    im = ImageEnhance.Color(im).enhance(1.08)
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))


def title_overlay(im, lines, y_frac=0.70):
    w, h = im.size; ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    top = int(h * (y_frac - 0.06))
    for y in range(top, h):                       # soft dark-blue gradient so the title is readable on any picture
        a = int(200 * ((y - top) / (h - top)) ** 1.2); d.line([(0, y), (w, y)], fill=(8, 14, 44, a))
    im = Image.alpha_composite(im.convert("RGBA"), ov); d = ImageDraw.Draw(im)
    big = ImageFont.truetype(FONT_B, int(w * 0.095)); small = ImageFont.truetype(FONT_R, int(w * 0.045))
    y = int(h * y_frac)
    for text, is_big in lines:
        f = big if is_big else small; tw = d.textlength(text, font=f)
        d.text(((w - tw) / 2 + 3, y + 3), text, font=f, fill=(0, 0, 20, 160)); d.text(((w - tw) / 2, y), text, font=f, fill=(252, 242, 205, 255))
        y += int(f.size * 1.25)
    return im.convert("RGB")


def pixel_icon(prompt, seed):
    """16x16 icon for the Yoto Mini display: chunky pixel-art object on a navy background (black is invisible on the dark screen)"""
    im = comfy(f"pixel art icon of {prompt}, 16-bit retro game sprite, very simple chunky shapes, thick dark outline, flat bright colors, centered, plain dark blue background",
               "blurry, photo, realistic, gradient, text, watermark, multiple objects, busy, black background", 512, 512, seed)
    im = ImageEnhance.Color(im).enhance(1.3)
    small = im.resize((16, 16), Image.BOX)
    q = small.quantize(colors=12, method=Image.MEDIANCUT, dither=Image.NONE).convert("RGB")
    return q


def save_icon(icon, path):
    path.parent.mkdir(parents=True, exist_ok=True); icon.save(path)
    icon.resize((256, 256), Image.NEAREST).save(path.with_suffix(".preview.png"))


def make_show():
    scene = {"de": "a sleepy white snow hare curled up on a snowy hill under a huge glowing crescent moon, tiny stars, a cozy little cabin with a warm yellow window in the distance",
             "en": "a sleepy white snow hare curled up on a snowy hill under a huge glowing crescent moon, tiny stars, a cozy little cabin with a warm yellow window in the distance"}
    names = {"de": [("Mondlicht", True), ("Geschichten", True), ("zum Einschlafen", False)], "en": [("Moonlight", True), ("Stories", True), ("bedtime stories", False)]}
    for lang in ("de", "en"):
        out = PUB / lang; out.mkdir(parents=True, exist_ok=True)
        for i, seed in enumerate((11, 23, 37)):
            comfy(scene[lang] + ", " + STYLE, NEG, 768, 768, seed).save(ROOT / "work" / f"cover_raw_{lang}_{i}.png")
    print("raw covers in work/ (cover_raw_<lang>_<n>.png) - pick one with: art.py use LANG N")


def use_cover(lang, n):
    names = {"de": [("Mondlicht", True), ("Geschichten", True), ("zum Einschlafen", False)], "en": [("Moonlight", True), ("Stories", True), ("bedtime stories", False)]}[lang]
    raw = Image.open(ROOT / "work" / f"cover_raw_{lang}_{n}.png")
    title_overlay(finish(raw, (1400, 1400)), names, 0.70).save(PUB / lang / "cover.jpg", quality=92)
    # portrait cover for the Yoto Make-Your-Own card (638x1011): a real portrait picture, so nothing is stretched
    scene = ("a sleepy white snow hare sitting on a snowy cabin roof, huge glowing full moon, tiny stars, snowy fir trees, warm yellow windows")
    port = comfy(scene + ", " + STYLE, NEG, 640, 1024, 37)
    title_overlay(finish(port, (638, 1011)), names, 0.74).save(PUB / lang / "yoto_card.jpg", quality=92)
    icon = pixel_icon("a yellow crescent moon with two small stars", 5)
    save_icon(icon, PUB / lang / "icons" / "show.png")
    print(lang, "cover.jpg, yoto_card.jpg, icons/show.png written")


def episode_art(scene, obj, seed=None):
    """one picture + one 16x16 icon for a story; returns (1400x1400 PIL image, 16x16 PIL image)"""
    seed = seed or random.randint(1, 10**6)
    big = finish(comfy(scene + ", " + STYLE, NEG, 768, 768, seed), (1400, 1400))
    return big, pixel_icon(obj, seed + 1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "show": make_show()
    elif cmd == "use": use_cover(sys.argv[2], int(sys.argv[3]))
