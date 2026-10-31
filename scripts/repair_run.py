"""Copy a run's cache into a new run directory, minus the games the 2026-10-31 fixes change, so that
re-running there replays every unchanged call for free and regenerates only:
  - games in which a player said it could not understand (they now continue with the refused message
    made unreadable, GameMaster.redact_received), and
  - priors (no monolingual prompt any more; their opening call is keyed by request hash, so the new
    request simply misses the cache).

Usage: python scripts/repair_run.py main2_route main2x_route
"""

import json
import sys

from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import NOT_UNDERSTOOD, is_not_understood


def main(src, dst):
    s, d = OUTPUT_ROOT / src, OUTPUT_ROOT / dst
    d.mkdir(parents=True, exist_ok=True)
    affected = set()
    for line in open(s / "records.jsonl"):
        r = json.loads(line)
        if r["status"] != "done" or any(t["kind"] == "message" and any(is_not_understood(t["source"], k)
                                                                        for k in NOT_UNDERSTOOD)
                                        for t in r["turns"]):
            affected.add(r["episode"])
    kept = dropped = 0
    with open(d / "cache.jsonl", "w") as out:
        for line in open(s / "cache.jsonl"):
            rec = json.loads(line)
            if any(f"|{e}|" in rec["key"] for e in affected if e in rec["key"]):
                dropped += 1
                continue
            out.write(line)
            kept += 1
    print(f"{src} -> {dst}: {len(affected)} games to regenerate; cache kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
