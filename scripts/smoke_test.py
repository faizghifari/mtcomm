"""End-to-end smoke test with free backends only: identity and local NLLB channels,
scripted describers (native XM3600 captions) and dry-run LLM players. No paid calls.

Usage: python scripts/smoke_test.py [--nllb facebook/nllb-200-distilled-600M] [--n 2]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm import lid  # noqa: E402
from mtcomm.channels import (AmazonChannel, GoogleWebChannel, IdentityChannel, NLLBChannel, NoChannel,
                             PivotChannel)  # noqa: E402
from mtcomm.data import load_blend_items, load_image_items  # noqa: E402
from mtcomm.game import Move, run_episode  # noqa: E402
from mtcomm.games.blend_lv import BlendLocalVisitorGame  # noqa: E402
from mtcomm.games.image_ref import ImageRefGame  # noqa: E402
from mtcomm.paths import OUTPUT_ROOT  # noqa: E402
from mtcomm.players import LLMPlayer, ScriptedPlayer  # noqa: E402
from mtcomm.vision import GeminiVision  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nllb", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    nllb = NLLBChannel(args.nllb)
    channels = [IdentityChannel(), NoChannel(), nllb, PivotChannel(nllb, "en"), GoogleWebChannel(),
                AmazonChannel(dry_run=True)]
    vision = GeminiVision(dry_run=True)
    records = []

    # PoC-1 static: native ko caption -> guesser in id, and id caption -> guesser in ko.
    for item in load_image_items("hard")[: args.n]:
        for src, tgt in [("ko", "id"), ("id", "ko")]:
            for ch in channels:
                if not ch.supports(src, tgt):
                    continue
                gm = ImageRefGame(item, src, tgt, allow_message=not isinstance(ch, NoChannel))
                players = {"describer": ScriptedPlayer([Move("message", item["captions"][src])]),
                           "guesser": LLMPlayer(vision, "gemini-dry")}
                rec = run_episode(gm, players, ch, lid=lid.check)
                records.append({"task": "image_ref", "pair": f"{src}->{tgt}", **rec})

    # PoC-2: visitor ko asking a West Java (su) local, dry-run players on both sides.
    blend = [b for b in load_blend_items("ko") if b["local_culture"] == "West_Java"][: args.n]
    for item in blend:
        for ch in [nllb, PivotChannel(nllb, "id")]:
            gm = BlendLocalVisitorGame(item, "su", "ko", max_questions=1)
            players = {"visitor": LLMPlayer(vision, "gemini-dry"), "local": LLMPlayer(vision, "gemini-dry")}
            rec = run_episode(gm, players, ch)
            records.append({"task": "blend_lv", "pair": "su(local)<-ko(visitor)", **rec})

    out = OUTPUT_ROOT / "smoke" / "smoke_records.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    for r in records:
        msgs = [t for t in r["turns"] if t["kind"] == "message"]
        first = msgs[0] if msgs else None
        print(f"[{r['task']}] {r['pair']:24s} {r['channel']:40s} finished={r['finished']} score={r['score']}")
        if first and r["task"] == "image_ref":
            print(f"    src: {first['source'][:90]}\n    out: {str(first['delivered'])[:90]}  lid={first['lid']}")
    print(f"dry-run vision calls logged: {len(vision.calls)}; records -> {out}")


if __name__ == "__main__":
    main()
