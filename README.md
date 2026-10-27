# mtcomm

MT-mediated communication benchmark harness: two players who speak different languages
complete a grounded task, and every message passes through the translation channel
under test. The primary score is task success.

Design and positioning: `../docs/benchmark_positioning.md`, `../docs/poc_plan.md`.

## Layout

- `src/mtcomm/channels.py`: translation channels. Every control is a channel:
  - identity (same-language ceiling);
  - none (no-communication prior);
  - NLLB;
  - Google (free web endpoint for the PoC; official API class kept for release);
  - Amazon Translate;
  - LLM via mt-eval's InterpreterAgent;
  - pivot (composition).

  Paid backends are dry-run by default.
- `src/mtcomm/game.py`: `GameMaster` interface and `run_episode`. Games hold task state and scoring; the runner routes messages through the channel.
- `src/mtcomm/games/`: `image_ref` (describe 1, pick 1 of N) and `blend_lv` (BLEnD local–visitor).
- `src/mtcomm/players.py`: scripted players (native human text) and LLM players (JSON action format).
- `src/mtcomm/vision.py`: Gemini vision backend for image players.
- `src/mtcomm/lid.py`: GlotLID check, reusing mt-eval's, with a NumPy 2 workaround.
- `src/mtcomm/data.py`: loaders for the PoC-1 manifest and the BLEnD local–visitor items.
- `scripts/`: data preparation (XM3600 subset, BLEnD pairs, rendering sheets) and `smoke_test.py`.

## Data

Data lives outside the repo, under `../data` and `../outputs/mtcomm`; see `src/mtcomm/paths.py`. Override with `MTCOMM_DATA` / `MTCOMM_OUTPUTS`.

## Setup

```bash
uv sync                      # Python 3.11; mt-eval is an editable path dependency
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/smoke_test.py   # free backends only
```
