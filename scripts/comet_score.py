"""Reference-free COMET scoring of (source, translation) pairs; runs in the separate .venv-metrics env.

Mirrors mt-eval's research/mt_compare/compute_comet.py run_model (same checkpoints, src+mt only, XCOMET
error-span counts), which cannot run in mt-eval's own env at the moment (setuptools 82 has no
pkg_resources). Standalone on purpose: it imports nothing from mtcomm.

XCOMET-XL (3.5B, fp32) does not load on this 23 GB-RAM machine the default way (killed while loading);
`--low-mem` builds the model in bf16 and memory-maps the checkpoint, so it fits in RAM and on a 16 GB GPU.

Usage: .venv-metrics/bin/python scripts/comet_score.py IN.jsonl OUT.jsonl --models cometkiwi [xcomet]
  IN rows: {"record_id", "source_text", "post_edit"}; OUT rows: {"record_id", "<model>_score", ...}
"""

import argparse
import gc
import json

MODEL_REGISTRY = {"cometkiwi": "Unbabel/wmt22-cometkiwi-da", "xcomet": "Unbabel/XCOMET-XL"}


def run_model(name, rows, batch_size, gpus=1, low_mem=False):
    import torch
    from comet import download_model, load_from_checkpoint

    path = download_model(MODEL_REGISTRY[name])
    if low_mem:
        load, dtype = torch.load, torch.get_default_dtype()
        # Lightning hands torch.load an open file; mmap needs the path
        torch.load = lambda f, *a, **k: load(getattr(f, "name", f), *a, **{**k, "mmap": True})
        torch.set_default_dtype(torch.bfloat16)  # fp16 overflows a -1e20 fill buffer
        try:
            model = load_from_checkpoint(path).to(torch.bfloat16)  # torch.FloatTensor(...) params stay fp32
        finally:
            torch.load = load
            torch.set_default_dtype(dtype)
    else:
        model = load_from_checkpoint(path)
    out = model.predict([{"src": r["source_text"], "mt": r["post_edit"]} for r in rows],
                        batch_size=batch_size, gpus=gpus)
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
    ap.add_argument("--gpus", type=int, default=1, help="0 = CPU (XCOMET-XL on taco, whose GPUs serve Qwen)")
    ap.add_argument("--low-mem", action="store_true", help="bf16 weights, memory-mapped checkpoint (XCOMET-XL here)")
    ap.add_argument("--limit", type=int, help="score only the first N segments (checks)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8") if line.strip()][:args.limit]
    res = [{"record_id": r["record_id"]} for r in rows]
    for name in args.models:
        scores, spans = run_model(name, rows, args.batch_size, args.gpus, args.low_mem)
        for i, s in enumerate(scores):
            res[i][f"{name}_score"] = s
            if spans is not None:
                res[i][f"{name}_n_error_spans"] = spans[i]
        with open(args.out, "w", encoding="utf-8") as f:  # after each model, so a later crash keeps it
            for r in res:
                f.write(json.dumps(r) + "\n")
        print(f"scored {len(rows)} segments with {name} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
