-include .env

COMPOSE_FILE ?= compose.yaml
IMAGE_PREFIX ?= ghcr.io/andimajore/laya-kev-docker
LAYA_PORT ?= 8000
KEV_PORT ?= 8001
LAYA_VERSION ?= 0.3.9
KEV_REF ?= 557598fced1dada75dfbf36ed144dce309ac6ceb
KEV_MODEL ?= jaredpalmer/kev-0.8b
TORCH_INDEX ?= cu128
LAYA_CACHE_VOLUME ?= laya-gpu-hf-cache
LAYA_IMAGE := $(IMAGE_PREFIX)/laya:$(LAYA_VERSION)-$(TORCH_INDEX)
KEV_IMAGE := $(IMAGE_PREFIX)/kev:$(KEV_REF)-$(TORCH_INDEX)
# Servers in use: laya and/or kev, from the compose.<name>.yaml files in COMPOSE_FILE
# (compose.yaml includes both).
SERVERS := $(strip $(if $(filter compose.yaml,$(subst :, ,$(COMPOSE_FILE))),laya kev,\
	$(patsubst compose.%.yaml,%,$(filter compose.laya.yaml compose.kev.yaml,$(subst :, ,$(COMPOSE_FILE))))))

# Host networking because some VPN clients (e.g. Cisco AnyConnect) cut off Docker's
# bridge networks. Once cached, the servers load the weights without network access.
FETCH := docker run --rm --network host --env-file .env -e HF_HUB_OFFLINE=0 \
	-v $(LAYA_CACHE_VOLUME):/data/hf --entrypoint python

.PHONY: help env build pull up down logs download gpu-check smoke

help:
	@echo "Servers come from COMPOSE_FILE in .env (now: $(SERVERS))."
	@echo "make env       - create .env from .env.example (keeps an existing one)"
	@echo "make build     - build the images locally"
	@echo "make pull      - pull the published images instead of building"
	@echo "make up        - build and start the servers (detached)"
	@echo "make down      - stop them (weights stay in the cache volume)"
	@echo "make logs      - follow container logs"
	@echo "make download  - prefetch the checkpoints into the cache (host network, works behind VPNs)"
	@echo "make gpu-check - verify torch sees the GPU inside each container"
	@echo "make smoke     - call /health and /v1/systemone with examples/request.json"

env:
	@test -f .env && echo ".env exists, leaving it alone" || (cp .env.example .env && echo "created .env")

build:
	docker compose build

pull:
	docker compose pull

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

download: build
	docker compose create --quiet-pull 2>/dev/null || docker compose create
	$(if $(filter laya,$(SERVERS)),$(FETCH) $(LAYA_IMAGE) -c \
		"import os; from laya import Router; Router(device='cpu').preload([m for m in os.environ.get('LAYA_MODELS', '').split(',') if m.strip()] or None)")
	$(if $(filter kev,$(SERVERS)),$(FETCH) $(KEV_IMAGE) -c \
		"from kev.checkpoint import Checkpoint, resolve_run; from kev.model import load_tokenizer; \
		ck = Checkpoint('$(KEV_MODEL)'); resolve_run(f'{ck.meta.base}@{ck.meta.base_revision or \"\"}'); \
		load_tokenizer(ck.meta.base, revision=ck.meta.base_revision); print('cached', ck.requested, '+', ck.meta.base)")

gpu-check:
	@for s in $(SERVERS); do \
		docker compose run --rm --no-deps $$s python -c \
			"import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('$$s:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" || exit 1; \
	done

# $(1) = name, $(2) = port, $(3) = API key
smoke-one = \
	echo "== $(1)"; \
	curl -fsS http://localhost:$(2)/health && echo && \
	curl -fsS http://localhost:$(2)/v1/systemone $(if $(3),-H 'Authorization: Bearer $(3)') \
		-H 'Content-Type: application/json' --data @examples/request.json && echo

smoke:
	@$(if $(filter laya,$(SERVERS)),$(call smoke-one,laya,$(LAYA_PORT),$(LAYA_API_KEY)))
	@$(if $(filter kev,$(SERVERS)),$(call smoke-one,kev,$(KEV_PORT),$(KEV_API_KEY)))
