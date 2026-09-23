# laya-kev-docker

Two interchangeable decision servers in CUDA-enabled containers, configured entirely
through one `.env` file. Both speak the TypeSafe `POST /v1/systemone` protocol, so a client
switches between them by changing the port.

| Compose file | Server | Model | Port |
| --- | --- | --- | --- |
| `compose.laya.yaml` | [Laya](https://github.com/NandhaKishorM/laya)'s `laya-serve` | ModernBERT / mmBERT encoders (~0.4B), language-routed | 8000 |
| `compose.kev.yaml` | [kev](https://github.com/jaredpalmer/kev)'s `kev.serve` | Qwen3.5-based Kev checkpoints (0.8B–9B) | 8001 |

Pick one or both with `COMPOSE_FILE` in `.env`, e.g.
`COMPOSE_FILE=compose.laya.yaml:compose.kev.yaml`. Without it, `compose.yaml` runs both. Each
file also works on its own: `docker compose -f compose.kev.yaml up -d`.

- **Upstream servers**: `laya[serve]` from PyPI and kev from GitHub, both pinned. Kev gets
  a small wrapper (`kev/kev_serve.py`); see [Kev notes](#kev-notes).
- **CUDA via PyTorch wheels**: a slim Python base plus torch from the `cu128` index. The
  wheels bundle the CUDA runtime, so the host only needs the NVIDIA driver.
- **One `.env`**: build args, ports, GPU selection and every `LAYA_*` / `KEV_*` setting.
- **Prebuilt images**: CI publishes both to GHCR, so building locally is optional; see
  [Images and CI](#images-and-ci).
- Non-root (uid 10001), container healthcheck, secrets can be loaded from files
  (`*_FILE`), and weights are kept in a named volume.

## Prerequisites

- NVIDIA driver whose "CUDA Version" (`nvidia-smi`) is at least `TORCH_INDEX`, e.g.
  12.8 for `cu128`
- Docker with Compose v2.24+ and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quickstart

```bash
make env        # cp .env.example .env
make pull       # optional: fetch the published images instead of building
make up         # build (unless pulled) and start the selected servers
make logs       # wait for "Application startup complete"
make smoke      # /health + a sample /v1/systemone request, per server
```

`make gpu-check` confirms that torch sees the GPU inside each container.

## Configuration (`.env`)

| Variable | Default | Effect |
| --- | --- | --- |
| `COMPOSE_FILE` | `compose.laya.yaml` | servers to run, colon-separated compose files |
| `IMAGE_PREFIX` | `ghcr.io/andimajore/laya-kev-docker` | registry path of the published images |
| `LAYA_VERSION` | `0.3.9` | **build**: laya release from PyPI |
| `KEV_REF` | `557598f…` | **build**: kev commit, tag or branch on GitHub |
| `TORCH_INDEX` | `cu128` | **build**: PyTorch wheel index (`cu126`, `cu128`, `cu129`, ...) |
| `BUILD_NETWORK` | `default` | **build**: set to `host` if pip cannot resolve DNS during the build |
| `LAYA_GPU_ID` | `0` | host GPU index or UUID (appears as `cuda:0` inside the container) |
| `KEV_GPU_ID` | `LAYA_GPU_ID` | host GPU for kev |
| `LAYA_BIND` | `0.0.0.0` | host interface to publish both servers on; `127.0.0.1` for local-only |
| `LAYA_CACHE_VOLUME` | `laya-gpu-hf-cache` | Docker volume holding the checkpoints |
| `LAYA_PORT` | `8000` | server port, published under the same number on the host |
| `LAYA_DEVICE` | `cuda` | torch device; empty = auto |
| `LAYA_PRELOAD` | `1` | load checkpoints at startup instead of on first request |
| `LAYA_MODELS` | `english,multilingual` | checkpoints to preload; empty = all three |
| `LAYA_AUTO_TASK` | `0` | auto-route recognised workflows to `typed-decisions` |
| `LAYA_LOG_LEVEL` | `info` | uvicorn log level |
| `LAYA_THREADS` | empty | cap torch CPU threads |
| `LAYA_API_KEY` / `LAYA_API_KEY_FILE` | empty | laya: require `Authorization: Bearer <key>` |
| `KEV_PORT` | `8001` | kev's port, published under the same number on the host |
| `KEV_MODEL` | `jaredpalmer/kev-0.8b` | Hub id (optionally `@revision`) or local run directory |
| `KEV_DTYPE` | empty (bf16) | `bf16`, `fp16`, or `fp32` (the exact path kev's evaluations use) |
| `KEV_DATE_FACTS` | `0` | append day counts between dates in the state |
| `KEV_TEMPERATURE` | empty | override the checkpoint's calibration temperature |
| `KEV_REQUIRE_CUDA` | `1` | refuse to start without CUDA instead of serving on CPU |
| `KEV_LOG_LEVEL` | `info` | uvicorn log level |
| `KEV_API_KEY` / `KEV_API_KEY_FILE` | empty | kev: require `Authorization: Bearer <key>` on `/v1/*` |
| `HF_TOKEN` / `HF_TOKEN_FILE` | empty | Hugging Face token (public checkpoints need none) |
| `HF_HUB_OFFLINE` | `0` | `1` = use cached weights only, never contact the Hub |

Compose uses `.env` for `${...}` interpolation (build, port, GPU) **and** as the
containers' `env_file`, so every server setting reaches the servers without being listed
in `compose.yaml`. kev's other `KEV_*` options (`KEV_BACKEND`, `KEV_ATTN`, `KEV_PREFIX_CACHE`, ...)
work the same way; see kev's README. Build settings need a rebuild; `make up` always rebuilds, and cached
layers make that fast.

For secret files, uncomment the mounts in `compose.yaml` and set the matching `*_FILE`
variable. The entrypoint reads the file into the environment at startup.

## API

```bash
curl localhost:8000/health
# {"status":"ok","loaded":["english","multilingual"],"device":"cuda"}

curl localhost:8000/v1/systemone -H 'Content-Type: application/json' \
  --data @examples/request.json

curl localhost:8001/health
# {"status":"ok","model":"jaredpalmer/kev-0.8b","device":"cuda"}
curl localhost:8001/v1/systemone -H 'Content-Type: application/json' \
  --data @examples/request.json
```

The responses share `answers` and `usage`. On top of that, laya adds `routing` (which
checkpoint and why) and kev adds `latency_ms`. kev also serves `GET /v1/models` and the
demo routes `POST /v1/systemone/permute` and `/v1/systemone/separate`.

The request body is `{"state": ..., "questions": {...}, "model"?: ...}`, with
`choice` / `score` / `noul` questions. It is wire-compatible with the TypeSafe Jev
`/v1/systemone` protocol. For the request format, see the
[Laya README](https://github.com/NandhaKishorM/laya).

## Images and CI

`.github/workflows/images.yml` builds both images on every push and pull request. For each
image it:
- reads `LAYA_VERSION`, `KEV_REF` and `TORCH_INDEX` from `.env.example`, so CI and local
  builds always use the same versions;
- imports the server inside the image, since hosted runners have no GPU;
- on `main`, tags `v*` and manual runs, pushes to GHCR. Pull requests only build and check.

| Image | Tags |
| --- | --- |
| `ghcr.io/andimajore/laya-kev-docker/laya` | `<LAYA_VERSION>-<TORCH_INDEX>` (what compose pulls), `latest`, `sha-<commit>`, `v*` |
| `ghcr.io/andimajore/laya-kev-docker/kev` | `<KEV_REF>-<TORCH_INDEX>` (what compose pulls), `latest`, `sha-<commit>`, `v*` |

Build layers are cached in the registry (`:buildcache`), so a change that doesn't touch
the dependencies rebuilds in minutes. To move to a new laya release or kev commit, change
`.env.example`: CI publishes the new tag and the compose files pull it.

`make up` always builds locally. To use the published images, run `make pull` and then
`docker compose up -d`. GHCR packages start out private; make them public under the
package settings, or `docker login ghcr.io` first.

## Kev notes

- **Separate image**: kev pins `torch<2.9`, laya's image uses a newer torch, so kev builds
  from `kev/Dockerfile`. It installs kev from a GitHub archive because the `kev` package on
  PyPI is a different project. It adds `flash-linear-attention` for fast Qwen3.5 kernels,
  and `gcc` because Triton compiles its launcher at runtime.
- **Wrapper**: `kev.serve.main()` binds `127.0.0.1` and takes its settings as CLI flags.
  `kev/kev_serve.py` makes the same serving choices but reads `KEV_MODEL` / `KEV_PORT`,
  binds `0.0.0.0` and adds `GET /health`.
- **First start compiles kernels**: the wrapper sends one warm-up request before listening.
  On a fresh cache Triton compiles for about 75 s; the compiled kernels are kept in the
  cache volume (`TRITON_CACHE_DIR`), so later starts warm up in about 1 s.
- **Model size vs. VRAM**: `kev-0.8b` ~2 GB of weights (~3.7 GB of GPU memory in use),
  `kev-4b` ~9 GB, `kev-9b` ~20 GB. On an 8 GB GPU, `kev-0.8b` runs next to laya; the 4B does
  not fit.
- **Latency** on an RTX 4060 Laptop GPU, sample request: laya ~55 ms, kev-0.8b ~100–150 ms.

## Behind a VPN

Some VPN clients (e.g. Cisco AnyConnect) cut off Docker's bridge networks, so the
container cannot download weights. The fix:

1. Set `BUILD_NETWORK=host` so the image build can reach PyPI.
2. Run `make download` to prefetch the checkpoints of the selected servers (`LAYA_MODELS`,
   `KEV_MODEL` and its Qwen base) into the cache volume over the host network.
3. Set `HF_HUB_OFFLINE=1` so the server loads straight from the cache instead of waiting
   on Hub timeouts.

## Housekeeping

- `make down` stops the server and keeps the weights.
- `docker compose down --volumes` also deletes the weights and the compiled kernels.
- VRAM measured on an RTX 4060 Laptop GPU (8 GB): laya `english` + `multilingual`
  ~3.9 GB, kev-0.8b ~3.7 GB. Both fit side by side.
