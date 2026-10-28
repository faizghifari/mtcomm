"""Reference-free COMET scoring of (source, translation) pairs; runs in the separate .venv-metrics env.

Mirrors mt-eval's research/mt_compare/compute_comet.py run_model (same checkpoints, src+mt only, XCOMET
error-span counts), which cannot run in mt-eval's own env at the moment (setuptools 82 has no
pkg_resources). Standalone on purpose: it imports nothing from mtcomm.

Usage: .venv-metrics/bin/python scripts/comet_score.py IN.jsonl OUT.jsonl --models cometkiwi xcomet
  IN rows: {"record_id", "source_text", "post_edit"}; OUT rows: {"record_id", "<model>_score", ...}
"""

import argparse
import gc
import json

MODEL_REGISTRY = {"cometkiwi": "Unbabel/wmt22-cometkiwi-da", "xcomet": "Unbabel/XCOMET-XL"}


def run_model(name, rows, batch_size):
    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(MODEL_REGISTRY[name]))
    out = model.predict([{"src": r["source_text"], "mt": r["post_edit"]} for r in rows],
                        batch_size=batch_size, gpus=1)
    spans = None
    meta = getattr(out, "metadata", None)
    if meta is not None and getattr(meta, "error_spans", None) is not None:
        spans = [len(s) if s else 0 for s in meta.error_spans]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return list(out.scores), spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--models", nargs="*", default=["cometkiwi", "xcomet"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8") if line.strip()]
    res = [{"record_id": r["record_id"]} for r in rows]
    for name in args.models:
        scores, spans = run_model(name, rows, args.batch_size)
        for i, s in enumerate(scores):
            res[i][f"{name}_score"] = s
            if spans is not None:
                res[i][f"{name}_n_error_spans"] = spans[i]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"scored {len(rows)} segments with {args.models} -> {args.out}")


if __name__ == "__main__":
    main()
