"""Run kev's TypeSafe server (kev.serve) inside the container, configured by environment.

kev.serve.main() binds 127.0.0.1 and takes the checkpoint and port as CLI flags, which a
container cannot publish or an .env cannot set. This mirrors its setup (device, bf16 on
GPU, backend) and serves the same app on KEV_HOST:KEV_PORT with the checkpoint in
KEV_MODEL. It also adds an unauthenticated GET /health, like laya-serve's, and answers one
warm-up request before listening: Triton compiles the Gated DeltaNet kernels on first use,
which takes a minute or more, and that should not land on the first real request.
"""
import os
import time
from dataclasses import replace

import torch
import uvicorn
from kev.api import SystemOneRequest
from kev.checkpoint import Checkpoint, LoadOptions
from kev.device import default_device
from kev.serve import Server, app


# One question of each type, so every readout path is compiled.
WARMUP = SystemOneRequest(state="I was charged twice. Please refund me.", questions={
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "support": "other"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
})


@app.get("/health")
def health():
    s = app.state.server
    return {"status": "ok", "model": s.checkpoint.requested, "device": s.device}


def main():
    dev = default_device()
    if os.environ.get("KEV_REQUIRE_CUDA", "1") == "1" and dev != "cuda":
        raise SystemExit("CUDA is not available (set KEV_REQUIRE_CUDA=0 to serve on CPU)")
    # The same serving defaults as kev.serve.main; KEV_DTYPE / KEV_BACKEND / ... still override.
    opts = LoadOptions.from_env()
    if dev != "cpu" and opts.dtype is None:
        opts = replace(opts, dtype=torch.bfloat16)
    if opts.backend is None:
        opts = replace(opts, backend="auto")
    ck = Checkpoint(os.environ.get("KEV_MODEL") or "jaredpalmer/kev-0.8b")
    tok, model = ck.load(dev, opts)
    app.state.server = server = Server(ck, tok, model, dev)
    t = time.time()
    server.answer(WARMUP)
    server.prefix_cache.clear(); server.prefix_hits = server.prefix_misses = 0
    print(f"warm-up done in {time.time() - t:.1f}s", flush=True)
    print(f"serving {ck.requested} on {dev} via {model.backend} ({model.dtype})", flush=True)
    uvicorn.run(
        app,
        host=os.environ.get("KEV_HOST", "0.0.0.0"),
        port=int(os.environ.get("KEV_PORT", "8001")),
        log_level=os.environ.get("KEV_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
