"""
Generate color variants for car images.
Smart recoloring: only changes body paint, preserves glass/chrome/tires/lights.
"""

import os, io
from pathlib import Path
from PIL import Image
from rembg import remove
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb

IMAGES_DIR = Path("images")
CARS = [
    "toyota-camry",
    "bmw-3-series",
    "ford-explorer",
    "tesla-model-3",
    "honda-civic",
    "ram-1500",
]

# Target HSV values for each color (hue 0-360, sat 0-1, val_mult)
COLORS = {
    "white":  {"hue": None, "sat": 0.03, "val_mult": 1.4,  "name": "Arctic White"},
    "black":  {"hue": None, "sat": 0.05, "val_mult": 0.2,  "name": "Midnight Black"},
    "red":    {"hue": 0,    "sat": 0.85, "val_mult": 0.7,  "name": "Crimson Red"},
    "blue":   {"hue": 215,  "sat": 0.80, "val_mult": 0.55, "name": "Deep Ocean Blue"},
    "silver": {"hue": None, "sat": 0.04, "val_mult": 0.75, "name": "Sterling Silver"},
    "green":  {"hue": 145,  "sat": 0.70, "val_mult": 0.45, "name": "Emerald Green"},
    "gold":   {"hue": 40,   "sat": 0.65, "val_mult": 0.7,  "name": "Champagne Gold"},
}


def compute_paint_mask(img_rgba):
    """
    Create a mask of which pixels are likely body paint vs glass/chrome/tires/lights.
    Returns a float mask 0.0-1.0 where 1.0 = definitely paint, 0.0 = definitely not paint.
    """
    arr = np.array(img_rgba, dtype=np.float64)
    r, g, b, a = arr[:,:,0]/255.0, arr[:,:,1]/255.0, arr[:,:,2]/255.0, arr[:,:,3]

    # Compute HSV per pixel
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    val = cmax
    sat = np.where(cmax == 0, 0, diff / (cmax + 1e-10))

    # Start with full mask for visible pixels
    visible = (a > 30).astype(np.float64)
    mask = visible.copy()

    # === Exclude very dark pixels (tires, dark trim, black rubber) ===
    # Tires and dark trim: V < 0.12
    too_dark = val < 0.12
    mask = np.where(too_dark, mask * 0.05, mask)

    # Dark-ish pixels with very low saturation (dark gray trim): reduce
    dark_gray = (val < 0.25) & (sat < 0.15)
    mask = np.where(dark_gray, mask * 0.15, mask)

    # === Exclude very bright, low-saturation pixels (chrome, metallic highlights, reflections) ===
    chrome = (val > 0.92) & (sat < 0.08)
    mask = np.where(chrome, mask * 0.1, mask)

    # Bright highlights
    bright_highlight = (val > 0.85) & (sat < 0.12)
    mask = np.where(bright_highlight, mask * 0.2, mask)

    # === Exclude glass/windows: typically dark-medium, very low saturation ===
    glass = (val > 0.15) & (val < 0.7) & (sat < 0.08)
    mask = np.where(glass, mask * 0.1, mask)

    # === Headlights/taillights: very bright spots ===
    headlights = val > 0.95
    mask = np.where(headlights, mask * 0.05, mask)

    # === Keep body paint: moderate value, any saturation ===
    # Body paint typically: 0.2 < V < 0.9, can have any saturation
    good_paint = (val > 0.2) & (val < 0.9) & visible.astype(bool)
    # Boost these slightly
    mask = np.where(good_paint & (sat > 0.05), np.minimum(mask * 1.2, 1.0), mask)

    # Smooth the mask slightly to avoid harsh edges
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=1.0)

    return np.clip(mask, 0, 1)


def recolor_car(img_rgba, color_config, paint_mask):
    """Recolor only the paint areas of the car using the paint mask."""
    arr = np.array(img_rgba, dtype=np.float64)
    original = arr.copy()

    r, g, b = arr[:,:,0]/255.0, arr[:,:,1]/255.0, arr[:,:,2]/255.0
    a = arr[:,:,3]

    # Compute original HSV
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin + 1e-10

    orig_val = cmax
    orig_sat = np.where(cmax == 0, 0, (cmax - cmin) / (cmax + 1e-10))

    # Compute original hue
    orig_hue = np.zeros_like(r)
    # When cmax == r
    rm = (cmax == r)
    orig_hue = np.where(rm, (60 * ((g - b) / diff) + 360) % 360, orig_hue)
    # When cmax == g
    gm = (cmax == g)
    orig_hue = np.where(gm, (60 * ((b - r) / diff) + 120) % 360, orig_hue)
    # When cmax == b
    bm = (cmax == b)
    orig_hue = np.where(bm, (60 * ((r - g) / diff) + 240) % 360, orig_hue)

    target_hue = color_config["hue"]
    target_sat = color_config["sat"]
    val_mult = color_config["val_mult"]

    if target_hue is not None:
        # Chromatic color: set hue, set saturation, adjust value
        new_hue = np.full_like(r, target_hue / 360.0)
        new_sat = np.clip(target_sat + orig_sat * 0.15, 0, 1)  # base target + slight original variation
        new_val = np.clip(orig_val * val_mult, 0, 1)
    else:
        # Achromatic (white, black, silver): desaturate heavily, adjust brightness
        new_hue = orig_hue / 360.0  # keep original hue (doesn't matter much)
        new_sat = np.full_like(r, target_sat)
        new_val = np.clip(orig_val * val_mult, 0, 1)

    # HSV to RGB conversion (vectorized)
    h6 = new_hue * 6.0
    hi = h6.astype(int) % 6
    f = h6 - hi.astype(float)
    p = new_val * (1 - new_sat)
    q = new_val * (1 - f * new_sat)
    t = new_val * (1 - (1 - f) * new_sat)

    new_r = np.zeros_like(r)
    new_g = np.zeros_like(g)
    new_b = np.zeros_like(b)

    for i, (rv, gv, bv) in enumerate([(new_val,t,p), (q,new_val,p), (p,new_val,t),
                                        (p,q,new_val), (t,p,new_val), (new_val,p,q)]):
        m = hi == i
        new_r = np.where(m, rv, new_r)
        new_g = np.where(m, gv, new_g)
        new_b = np.where(m, bv, new_b)

    # Blend recolored with original using paint mask
    pm = paint_mask[:,:,np.newaxis] if paint_mask.ndim == 2 else paint_mask
    pm3 = np.stack([paint_mask, paint_mask, paint_mask], axis=2)

    recolored = np.stack([new_r * 255, new_g * 255, new_b * 255], axis=2)
    orig_rgb = np.stack([r * 255, g * 255, b * 255], axis=2)

    # Blend: paint_mask=1 means use recolored, paint_mask=0 means use original
    blended = recolored * pm3 + orig_rgb * (1 - pm3)

    arr[:,:,0] = np.clip(blended[:,:,0], 0, 255)
    arr[:,:,1] = np.clip(blended[:,:,1], 0, 255)
    arr[:,:,2] = np.clip(blended[:,:,2], 0, 255)

    return Image.fromarray(arr.astype(np.uint8), 'RGBA')


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    colors_dir = IMAGES_DIR / "colors"
    colors_dir.mkdir(exist_ok=True)

    for car in CARS:
        src = IMAGES_DIR / f"{car}.jpg"
        if not src.exists():
            print(f"  SKIP {car} - source not found")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {car}")
        print(f"{'='*50}")

        # Step 1: Remove background (reuse if already done)
        nobg_path = colors_dir / f"{car}-nobg.png"
        if nobg_path.exists():
            print(f"  Loading existing background-removed image...")
            car_nobg = Image.open(nobg_path).convert("RGBA")
        else:
            print(f"  Removing background...")
            with open(src, "rb") as f:
                input_data = f.read()
            output_data = remove(input_data)
            car_nobg = Image.open(io.BytesIO(output_data)).convert("RGBA")
            car_nobg.save(nobg_path)
            print(f"  Saved: {nobg_path}")

        # Resize to web size
        w, h = car_nobg.size
        if w > 800:
            ratio = 800 / w
            car_nobg = car_nobg.resize((800, int(h * ratio)), Image.LANCZOS)

        # Step 2: Compute paint mask once per car
        print(f"  Computing paint mask...")
        paint_mask = compute_paint_mask(car_nobg)

        # Step 3: Generate color variants
        for color_key, color_config in COLORS.items():
            out_path = colors_dir / f"{car}-{color_key}.png"
            print(f"  Generating {color_config['name']}...")

            recolored = recolor_car(car_nobg, color_config, paint_mask)
            recolored.save(out_path, optimize=True)
            print(f"    Saved: {out_path}")

    print(f"\nDone! Generated {len(CARS) * len(COLORS)} color variants.")


if __name__ == "__main__":
    main()
