"""Flag XM3600 images that contain readable text (signs, labels, plates), using EasyOCR.

Text copied from a photo passes through any translator unchanged, which makes the image
game too easy and hides translator differences, so text-bearing images are excluded as
targets. Writes <DATA_ROOT>/derived/xm3600/ocr_text.json: key -> detected boxes with
confidence and area fraction.

Usage: python scripts/xm3600_detect_text.py
"""

import json
import sys
from pathlib import Path

import easyocr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.paths import DATA_ROOT, XM3600_DIR  # noqa: E402

OUT = DATA_ROOT / "derived" / "xm3600" / "ocr_text.json"


def main():
    done = json.load(open(OUT)) if OUT.exists() else {}
    # Latin covers most signage in the set; scripts the reader lacks still yield boxes
    # (detection runs before recognition), which is what the filter uses.
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    paths = sorted((XM3600_DIR / "images").glob("*.jpg"))
    for i, p in enumerate(paths):
        if p.stem in done:
            continue
        w, h = Image.open(p).size
        boxes = []
        for box, text, conf in reader.readtext(str(p)):
            xs, ys = [pt[0] for pt in box], [pt[1] for pt in box]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys)) / (w * h)
            boxes.append({"text": text, "conf": round(float(conf), 3), "area": round(area, 5)})
        done[p.stem] = boxes
        if i % 200 == 0:
            print(f"{i}/{len(paths)}", flush=True)
            json.dump(done, open(OUT, "w"))
    json.dump(done, open(OUT, "w"))
    print(f"wrote {OUT} ({len(done)} images)")


if __name__ == "__main__":
    main()
