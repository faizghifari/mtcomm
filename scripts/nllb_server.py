"""One copy of NLLB for every run on this machine. Each process that loads it holds about 19 GB of RAM, which allowed one at a time.

Runs find it through mtcomm.channels.nllb_channel. Answers only after the weights are on the GPU. Translations go one at a time, as they
do inside a single process, so output is the same as a local copy.

Usage: python scripts/nllb_server.py [--port 10090] [--host 0.0.0.0]
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_server(channel, model, port, host="127.0.0.1"):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"status": "ok", "model": model})

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                out = channel.translate(q["text"], q["src"], q["tgt"])
                self._send(200, {"text": out.text, "meta": out.meta})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:400]})

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    from mtcomm.channels import NLLBChannel

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=10090)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside a container, reached through an SSH tunnel")
    ap.add_argument("--model", default="facebook/nllb-200-3.3B")
    args = ap.parse_args()
    channel = NLLBChannel(args.model)
    channel._load()
    print(f"NLLB ready on port {args.port}", flush=True)
    make_server(channel, args.model, args.port, args.host).serve_forever()


if __name__ == "__main__":
    main()
