"""Round-trip translation (RTT) as a cheap proxy for one translator on one language pair.

Each opening message of the main runs (the batched first message, identical across translators)
is translated src -> tgt -> src -> ... for `--trips` round trips with the same translator, and every
back-translation is compared with the original (chrF and LaBSE cosine; same language, so no
reference is needed). Drift over trips shows how much meaning the translator loses on this pair.

Known confounds, handled here: an intermediate that is not in the target language (copied or
untranslated) would round-trip perfectly, so each forward hop is language-checked and a failed
hop ends the chain with score 0 from that trip on. The pivots (via Indonesian) count as one
translator per hop.

LLM translators are batched per hop (the hops of a chain are sequential, the chains are not);
NLLB and Google web run directly.

Usage: python scripts/roundtrip.py run --translators google sol flash-lite nllb google-id nllb-id [--trips 3]
       python scripts/roundtrip.py score     # chrF + LaBSE, then AUC / Spearman against task success
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtcomm import lid  # noqa: E402
from mtcomm.backends import make_backend  # noqa: E402
from mtcomm.channels import GoogleWebChannel, NLLBChannel, PivotChannel  # noqa: E402
from mtcomm.languages import get as lang  # noqa: E402
from mtcomm.paths import OUTPUT_ROOT  # noqa: E402

METRICS = OUTPUT_ROOT / "metrics"
OUT = OUTPUT_ROOT / "roundtrip"
LLM = {"sol": "gpt-6.1-sol", "flash-lite": "gemini-3.1-flash-lite"}
PIVOTS = {"google-id", "nllb-id"}


def chains():
    """One chain per opening message: (task, pair, item) -> source text, with src/tgt languages."""
    segs = {json.loads(l)["record_id"]: json.loads(l) for l in open(METRICS / "segments.jsonl")}
    out = {}
    for r in map(json.loads, open(METRICS / "messages.jsonl")):
        if r["turn"] != 0:
            continue
        s = segs[r["seg"]]
        item = r["episode"].split("|")[-1]
        cid = f"{r['task']}|{r['pair']}|{item}"
        out.setdefault(cid, {"id": cid, "task": r["task"], "pair": r["pair"], "src": s["src_lang"],
                             "tgt": s["tgt_lang"], "text": s["source_text"]})
    return out


class Recorder:
    """Stands in for the LLM provider to capture InterpreterAgent's prompt instead of calling it."""

    def generate(self, prompt, system_prompt=None, **kw):
        self.last = (prompt, system_prompt)
        return ""

    def get_provider_name(self):
        return "recorder"


def llm_request(backend, text, src, tgt):
    from interpreter_agent_eval.interpreter import InterpreterAgent

    rec = Recorder()
    InterpreterAgent(rec, lang(src).iso3, lang(tgt).iso3).translate(text)
    return backend.build(rec.last[0], rec.last[1], [])


def direct_channel(name, made={}):
    if name not in made:
        if name == "nllb":
            made[name] = NLLBChannel("facebook/nllb-200-3.3B")
        elif name == "google":
            made[name] = GoogleWebChannel()
        elif name in PIVOTS:
            made[name] = PivotChannel(direct_channel(name.split("-")[0]), "id")
    return made[name]


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    hops = defaultdict(dict)  # (translator, chain) -> hop index -> record
    for path in OUT.glob("hops_*.jsonl"):
        for r in map(json.loads, open(path)):
            hops[(r["translator"], r["chain"])][r["hop"]] = r
    cs = chains()
    for tr in args.translators:
        log = open(OUT / f"hops_{tr}.jsonl", "a")  # one file per translator, so runs can go in parallel
        todo = [c for c in cs.values() if tr not in PIVOTS or {c["src"], c["tgt"]} == {"su", "jv"}]
        for hop in range(2 * args.trips):
            jobs = []
            for c in todo:
                h = hops[(tr, c["id"])]
                if hop in h or any(x.get("failed") for x in h.values()):
                    continue
                prev = h[hop - 1]["out"] if hop else c["text"]
                src, tgt = (c["src"], c["tgt"]) if hop % 2 == 0 else (c["tgt"], c["src"])
                jobs.append((c, prev, src, tgt))
            if not jobs:
                continue
            print(f"{tr} hop {hop}: {len(jobs)} translations", flush=True)
            if tr in LLM:
                be = make_backend(LLM[tr], json_mode=False, **({"reasoning_effort": "low"} if tr == "sol" else {}))
                reqs = [(f"{c['id']}#{hop}", llm_request(be, prev, src, tgt)) for c, prev, src, tgt in jobs]
                wd = OUT / f"batch_{tr}_{hop}"
                wd.mkdir(exist_ok=True)
                job = be.submit(reqs, wd)
                while (st := be.status(job)) == "running":
                    time.sleep(30)
                if st != "succeeded":
                    raise RuntimeError(f"{tr} hop {hop}: batch {job} {st}")
                res = be.results(job)
                outs = [(res.get(f"{c['id']}#{hop}") or {}) for c, *_ in jobs]
                outs = [(o.get("text", "").strip() or None, o.get("usage")) for o in outs]
            else:
                ch = direct_channel(tr)
                with ThreadPoolExecutor(1 if tr.startswith("nllb") else 2) as ex:
                    outs = list(ex.map(lambda j: (ch.translate(j[1], j[2], j[3]).text, None), jobs))
            for (c, prev, src, tgt), (text, usage) in zip(jobs, outs):
                check = lid.check(text, tgt) if text else None
                failed = text is None or (hop % 2 == 0 and not check["ok"])
                r = {"translator": tr, "chain": c["id"], "hop": hop, "src": src, "tgt": tgt, "in": prev, "out": text,
                     "lid": check, "failed": failed, "usage": usage}
                hops[(tr, c["id"])][hop] = r
                log.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.flush()


def labse(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE").eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            vecs.append(torch.nn.functional.normalize(model(**b).pooler_output, dim=-1))
    return torch.cat(vecs)


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def score(args):
    from sacrebleu.metrics import CHRF
    from scipy.stats import spearmanr

    cs = chains()
    hops = defaultdict(dict)
    for r in (json.loads(l) for p in OUT.glob("hops_*.jsonl") for l in open(p)):
        hops[(r["translator"], r["chain"])][r["hop"]] = r
    trips = max(h for v in hops.values() for h in v) // 2 + 1
    rows, pairs = [], []  # one row per (translator, chain); pairs to embed: (original, back-translation)
    for (tr, cid), h in hops.items():
        row = {"translator": tr, "chain": cid, "task": cs[cid]["task"], "pair": cs[cid]["pair"]}
        for k in range(1, trips + 1):
            back = h.get(2 * k - 1)
            ok = back and back["out"] and not any(h[j]["failed"] for j in range(2 * k) if j in h)
            row[f"back{k}"] = back["out"] if ok else None
            if ok:
                pairs.append((cs[cid]["text"], back["out"]))
        rows.append(row)
    uniq = sorted({t for p in pairs for t in p})
    idx = {t: i for i, t in enumerate(uniq)}
    emb = labse(uniq)
    chrf = CHRF()
    for row in rows:
        orig = cs[row["chain"]]["text"]
        for k in range(1, trips + 1):
            b = row[f"back{k}"]
            row[f"chrf{k}"] = chrf.sentence_score(b, [orig]).score / 100 if b else 0.0
            row[f"labse{k}"] = float(emb[idx[orig]] @ emb[idx[b]]) if b else 0.0
    with open(OUT / "scores.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # task success of the matching episodes (same task, pair, item, translator)
    success = {}
    for r in map(json.loads, open(METRICS / "messages.jsonl")):
        if r["turn"] == 0:
            item = r["episode"].split("|")[-1]
            if r["task"] == "route" and not item.startswith("e"):
                continue  # v1 hard maps: ~0% everywhere
            success[(r["cond"], f"{r['task']}|{r['pair']}|{item}")] = bool(r["correct"])
    keys = [f"{m}{k}" for m in ("chrf", "labse") for k in range(1, trips + 1)]
    summary = {"trips": trips, "episode_auc": {}, "cell_spearman": {}, "cells": {}}
    for task in ("image", "mcq", "route"):
        sel = [r for r in rows if r["task"] == task and (r["translator"], r["chain"]) in success]
        summary["episode_auc"][task] = {
            key: auc([r[key] for r in sel if success[(r["translator"], r["chain"])]],
                     [r[key] for r in sel if not success[(r["translator"], r["chain"])]]) for key in keys}
        cells = defaultdict(list)
        for r in sel:
            cells[(r["translator"], r["pair"])].append(r)
        stats = {f"{t}|{p}": {"n": len(v), "success": sum(success[(x["translator"], x["chain"])] for x in v) / len(v),
                              **{key: sum(x[key] for x in v) / len(v) for key in keys}} for (t, p), v in cells.items()}
        summary["cells"][task] = stats
        summary["cell_spearman"][task] = {
            key: spearmanr([s[key] for s in stats.values()], [s["success"] for s in stats.values()]).correlation
            for key in keys}
    # drift: mean score by trip, per translator and pair (all tasks)
    drift = defaultdict(list)
    for r in rows:
        drift[f"{r['translator']}|{r['pair']}"].append(r)
    summary["drift"] = {k: {key: round(sum(x[key] for x in v) / len(v), 3) for key in keys} | {"n": len(v)}
                        for k, v in sorted(drift.items())}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    for k, v in summary["drift"].items():
        print(f"{k:22s} " + "  ".join(f"{key} {v[key]:.3f}" for key in keys))
    for part in ("episode_auc", "cell_spearman"):
        for task, v in summary[part].items():
            print(part, task, {k: (round(x, 3) if x is not None else None) for k, x in v.items()})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--translators", nargs="+", required=True, choices=["google", "nllb", "sol", "flash-lite",
                                                                       "google-id", "nllb-id"])
    r.add_argument("--trips", type=int, default=3)
    sub.add_parser("score")
    args = ap.parse_args()
    {"run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
