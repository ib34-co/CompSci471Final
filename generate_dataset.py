"""
generate_dataset.py
-------------------
Creates a small, fully synthetic (self-generated, cartoon-style) dataset of
side-view military vehicles for the Tank vs. Not-Tank CNN project.

Output:
    data/tank/tank_01.png ... tank_20.png          (label 1 = TANK)
    data/not_tank/not_tank_01.png ... _20.png      (label 0 = NOT TANK)

Tanks     : low wide hull, tracks with many small road wheels, turret, long gun.
Not tanks : trucks, wheeled APCs, and jeeps (big separate wheels, no turret/gun).

Every image gets random background, camo color, size, position and direction
so the CNN has to learn shapes instead of memorizing one picture.

Usage:
    python generate_dataset.py            # 20 + 20 images (default)
    python generate_dataset.py --per-class 50 --seed 7
"""

import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFilter

CANVAS = 128  # draw at 128x128; the CNN resizes to 64x64 when loading

BACKGROUNDS = [  # (sky, ground)
    ((150, 200, 235), (190, 170, 120)),  # desert
    ((170, 210, 240), (90, 130, 70)),    # forest / grass
    ((210, 220, 230), (235, 235, 240)),  # snow
    ((120, 140, 170), (110, 100, 90)),   # overcast / mud
    ((240, 190, 140), (160, 120, 80)),   # sunset / dirt
]

CAMO = [
    (85, 100, 60), (120, 110, 70), (70, 80, 55), (150, 135, 95),
    (100, 105, 100), (60, 70, 50), (170, 155, 110), (90, 85, 75),
]


def jitter(color, amount=15):
    return tuple(max(0, min(255, c + random.randint(-amount, amount))) for c in color)


def darker(color, f=0.6):
    return tuple(int(c * f) for c in color)


def background():
    sky, ground = random.choice(BACKGROUNDS)
    img = Image.new("RGB", (CANVAS, CANVAS), jitter(sky))
    d = ImageDraw.Draw(img)
    horizon = random.randint(70, 90)
    d.rectangle([0, horizon, CANVAS, CANVAS], fill=jitter(ground))
    # a few random ground speckles so backgrounds are never perfectly flat
    for _ in range(random.randint(5, 20)):
        x, y = random.randint(0, CANVAS), random.randint(horizon, CANVAS)
        r = random.randint(1, 3)
        d.ellipse([x - r, y - r, x + r, y + r], fill=jitter(ground, 30))
    return img, horizon


def wheel(d, cx, cy, r, color):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 30, 30))
    hr = max(1, int(r * 0.45))
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=color)


def add_camo_spots(d, box, body):
    x0, y0, x1, y1 = box
    for _ in range(random.randint(0, 4)):
        w = random.randint(4, 10)
        h = random.randint(2, 6)
        x = random.randint(int(x0), max(int(x0), int(x1) - w))
        y = random.randint(int(y0), max(int(y0), int(y1) - h))
        d.ellipse([x, y, x + w, y + h], fill=darker(body, 0.75))


# ----------------------------------------------------------------- TANK
def draw_tank(d, x, ground_y, s, body):
    L = 70 * s                      # hull length
    track_h = 12 * s
    hull_h = 10 * s
    # tracks (rounded band)
    ty0 = ground_y - track_h
    d.rounded_rectangle([x, ty0, x + L, ground_y], radius=track_h / 2, fill=(40, 40, 40))
    # many small road wheels -> a strong "tank" cue
    n = random.randint(5, 7)
    r = track_h * 0.33
    for i in range(n):
        cx = x + track_h / 2 + i * (L - track_h) / (n - 1)
        wheel(d, cx, ground_y - track_h / 2, r, darker(body, 0.8))
    # hull (sloped front)
    hy1 = ty0
    hy0 = hy1 - hull_h
    d.polygon([(x + 2 * s, hy1), (x + L - 2 * s, hy1), (x + L + 3 * s, hy0 + 3 * s),
               (x + L - 8 * s, hy0), (x + 6 * s, hy0)], fill=body)
    add_camo_spots(d, (x + 6 * s, hy0, x + L - 8 * s, hy1), body)
    # turret
    tw = random.uniform(26, 34) * s
    th = random.uniform(9, 12) * s
    tx0 = x + L * random.uniform(0.25, 0.38)
    d.rounded_rectangle([tx0, hy0 - th, tx0 + tw, hy0 + 1], radius=th * 0.45, fill=jitter(body, 8))
    # long main gun
    gun_len = random.uniform(30, 42) * s
    gy = hy0 - th * 0.55
    gh = max(2, 2.2 * s)
    d.rectangle([tx0 + tw - 2, gy - gh / 2, tx0 + tw + gun_len, gy + gh / 2], fill=darker(body, 0.7))
    # muzzle brake
    d.rectangle([tx0 + tw + gun_len - 4 * s, gy - gh, tx0 + tw + gun_len, gy + gh], fill=darker(body, 0.6))
    # commander hatch
    d.rectangle([tx0 + tw * 0.3, hy0 - th - 3 * s, tx0 + tw * 0.5, hy0 - th], fill=darker(body, 0.8))


# ----------------------------------------------------------------- NOT TANKS
def draw_truck(d, x, ground_y, s, body):
    L = 78 * s
    r = 7 * s
    base = ground_y - r
    # cargo box
    box_x0 = x
    box_x1 = x + L * 0.65
    d.rectangle([box_x0, base - 26 * s, box_x1, base], fill=body)
    # canvas cover stripes
    for i in range(1, 4):
        xx = box_x0 + i * (box_x1 - box_x0) / 4
        d.line([xx, base - 26 * s, xx, base], fill=darker(body, 0.8), width=max(1, int(s)))
    # cab
    cab_x0 = box_x1 + 2 * s
    cab_x1 = x + L
    d.rectangle([cab_x0, base - 20 * s, cab_x1, base], fill=jitter(body, 10))
    d.rectangle([cab_x0 + 4 * s, base - 17 * s, cab_x1 - 3 * s, base - 10 * s], fill=(140, 180, 200))
    # chassis
    d.rectangle([x, base - 2 * s, x + L, base + 2 * s], fill=(45, 45, 45))
    # 2-3 big wheels
    for cx in random.choice([[0.15, 0.4, 0.85], [0.2, 0.85], [0.12, 0.3, 0.85]]):
        wheel(d, x + cx * L, ground_y - r, r, (120, 120, 120))


def draw_apc(d, x, ground_y, s, body):
    L = 72 * s
    r = 7.5 * s
    base = ground_y - r * 0.8
    h = 22 * s
    d.polygon([(x, base), (x + L, base), (x + L + 4 * s, base - h * 0.55),
               (x + L - 10 * s, base - h), (x + 4 * s, base - h), (x - 2 * s, base - h * 0.4)], fill=body)
    add_camo_spots(d, (x + 4 * s, base - h, x + L - 10 * s, base), body)
    # vision slits
    for i in range(random.randint(2, 4)):
        sx = x + 12 * s + i * 12 * s
        d.rectangle([sx, base - h * 0.7, sx + 5 * s, base - h * 0.6], fill=(40, 40, 40))
    # optional small roof machine gun (short, thin, not a tank cannon)
    if random.random() < 0.5:
        mx = x + L * 0.5
        d.rectangle([mx, base - h - 4 * s, mx + 5 * s, base - h], fill=darker(body, 0.7))
        d.rectangle([mx + 5 * s, base - h - 3 * s, mx + 14 * s, base - h - 2 * s], fill=(40, 40, 40))
    # 3-4 big separate wheels
    n = random.choice([3, 4])
    for i in range(n):
        cx = x + 10 * s + i * (L - 20 * s) / (n - 1)
        wheel(d, cx, ground_y - r, r, (110, 110, 110))


def draw_jeep(d, x, ground_y, s, body):
    s = s * 1.1
    L = 55 * s
    r = 7 * s
    base = ground_y - r
    d.rectangle([x, base - 12 * s, x + L, base], fill=body)            # body
    d.polygon([(x + L * 0.35, base - 12 * s), (x + L * 0.45, base - 22 * s),
               (x + L * 0.75, base - 22 * s), (x + L * 0.8, base - 12 * s)], fill=jitter(body, 10))
    d.polygon([(x + L * 0.48, base - 20 * s), (x + L * 0.72, base - 20 * s),
               (x + L * 0.76, base - 13 * s), (x + L * 0.42, base - 13 * s)], fill=(140, 180, 200))
    d.rectangle([x - 4 * s, base - 14 * s, x + 2 * s, base - 4 * s], fill=(50, 50, 50))  # spare tire
    wheel(d, x + L * 0.18, ground_y - r, r, (120, 120, 120))
    wheel(d, x + L * 0.82, ground_y - r, r, (120, 120, 120))


NON_TANK_DRAWERS = [draw_truck, draw_apc, draw_jeep]


def make_image(is_tank):
    img, horizon = background()
    d = ImageDraw.Draw(img)
    s = random.uniform(0.85, 1.25)
    body = jitter(random.choice(CAMO))
    ground_y = random.randint(horizon + 8, min(CANVAS - 4, horizon + 30))
    x = random.uniform(4, max(5, CANVAS - 110 * s))
    if is_tank:
        draw_tank(d, x, ground_y, s, body)
    else:
        random.choice(NON_TANK_DRAWERS)(d, x, ground_y, s, body)
    if random.random() < 0.5:  # face left or right
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
    return img


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic tank / not-tank images")
    ap.add_argument("--out", default="data", help="output folder (default: data)")
    ap.add_argument("--per-class", type=int, default=20, help="images per class (default: 20)")
    ap.add_argument("--seed", type=int, default=471, help="random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    for cls, is_tank in (("tank", True), ("not_tank", False)):
        folder = os.path.join(args.out, cls)
        os.makedirs(folder, exist_ok=True)
        for i in range(1, args.per_class + 1):
            make_image(is_tank).save(os.path.join(folder, f"{cls}_{i:02d}.png"))
        print(f"Wrote {args.per_class} images to {folder}/")


if __name__ == "__main__":
    main()
