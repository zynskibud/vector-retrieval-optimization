# Every target runs inside the Docker container (docker-compose.yml). Nothing here
# runs a toolchain on the host.
RUN   := docker compose run --rm bench
SETUP := docker compose run --rm setup

.PHONY: image setup build test bench report shell clean-raw

image:
	docker compose build

## one-time: image, Python deps (network), then all four language builds (no network)
setup: image
	$(SETUP) sh -c 'uv sync --frozen && cd indexes/rust && cargo fetch'
	$(MAKE) build

build:
	$(RUN) sh -c 'cd indexes/go && go vet ./... && go build -o bin/bench ./cmd/bench'
	$(RUN) sh -c 'cmake -S indexes/cpp -B indexes/cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build indexes/cpp/build -j'
	$(RUN) sh -c 'cd indexes/rust && cargo build --release'

test:
	$(RUN) uv run --frozen python -m tools.data.verify --data data/processed/dev
	$(RUN) uv run --frozen pytest indexes/python/tests tools/bench/tests -q
	$(RUN) sh -c 'cd indexes/go && go test -timeout 60m ./...'
	$(RUN) ctest --test-dir indexes/cpp/build --output-on-failure
	$(RUN) sh -c 'cd indexes/rust && cargo test --release'

## e.g. make bench ARGS="--data data/processed/dev --languages rust,cpp --indexes flat,ivf --repeat 3"
bench:
	$(RUN) uv run --frozen python -m tools.bench.runner $(ARGS)

## e.g. make report ARGS="--data data/processed/dev"
report:
	$(RUN) uv run --frozen python -m tools.bench.report $(ARGS)

shell:
	$(RUN) bash

## delete the raw results volume (results/summary on the host is kept)
clean-raw:
	docker compose down -v --remove-orphans
