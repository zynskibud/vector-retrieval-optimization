# Every target runs inside the Docker container (docker-compose.yml). Nothing here
# runs a toolchain on the host.
RUN   := docker compose run --rm bench
SETUP := docker compose run --rm setup

.PHONY: image deps setup build test bench report shell clean-raw db-up db-down dbbench db-test bench-db load load-db changes changes-db cache cache-up cache-down cache-test backup-db backup-test vro-test

image:
	docker compose build

## one-time: image, Python deps (network), then all four language builds (no network)
## deps: fill the venv volume from uv.lock (network). Run after every uv.lock change.
deps:
	$(SETUP) uv sync --frozen --all-groups

setup: image
	$(SETUP) sh -c 'uv sync --frozen --all-groups && cd indexes/rust && cargo fetch'
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

## Phase 2: one database at a time. e.g. make db-up DB=qdrant; make db-test DB=qdrant; make db-down DB=qdrant
db-up:
	docker compose --profile $(DB) up -d --wait
db-down:
	docker compose --profile $(DB) down
## run the database bench (tools.db.bench) inside the container on the internal network
## e.g. make dbbench ARGS="--db qdrant --index hnsw --data data/processed/dev --out results/raw/dev/qdrant-hnsw.json --search ef=64"
dbbench:
	docker compose --profile db run --rm dbbench uv run --frozen python -m tools.db.bench $(ARGS)
## the runner for databases, inside dbbench: make bench-db ARGS="--data data/processed/dev --languages qdrant --indexes hnsw"
bench-db:
	docker compose --profile db run --rm dbbench uv run --frozen python -m tools.bench.runner $(ARGS)
db-test:
	docker compose --profile db run --rm dbbench uv run --frozen pytest tools/db/tests/test_$(DB).py -q

## Phase 4 load sweep (CONTRACT section 12): the runner in load mode. Two targets:
## languages inside bench:   make load ARGS="--data data/processed/dev --languages python,go,cpp,rust"
## databases inside dbbench: make load-db ARGS="--data data/processed/dev --languages qdrant"  (after make db-up DB=qdrant)
load:
	$(RUN) uv run --frozen python -m tools.bench.runner --load $(ARGS)
load-db:
	docker compose --profile db run --rm dbbench uv run --frozen python -m tools.bench.runner --load $(ARGS)

## Phase 5 changes sweep (CONTRACT section 13): deletes with and without compaction, and updates.
## languages inside bench:   make changes ARGS="--data data/processed/dev --languages python,go,cpp,rust"
## databases inside dbbench: make changes-db ARGS="--data data/processed/dev --languages qdrant"  (after make db-up DB=qdrant)
changes:
	$(RUN) uv run --frozen python -m tools.bench.runner --changes $(ARGS)
changes-db:
	docker compose --profile db run --rm dbbench uv run --frozen python -m tools.bench.runner --changes $(ARGS)

## Phase 6 embedding cache (CONTRACT section 14): Redis under the cache profile, the sweep inside
## dbbench (on the Redis network). e.g. make cache-up; make cache ARGS="--data data/processed/dev"; make cache-down
cache-up:
	docker compose --profile cache up -d redis
cache-down:
	docker compose --profile cache down
cache:
	docker compose --profile db run --rm dbbench uv run --frozen python -m tools.bench.runner --cache $(ARGS)
## the cache tests, Redis included (after make cache-up)
cache-test:
	docker compose --profile db run --rm dbbench uv run --frozen pytest tools/cache/tests -q

## Phase 7 backup and restore (CONTRACT section 15.3), one database at a time (after make db-up DB=...).
## Runs on the host: scripts/backup_db.sh starts the dbbench stages and the docker steps between them.
## e.g. make backup-db DB=pgvector ARGS="--data data/processed/dev [--indexes hnsw] [--limit 20000] [--force]"
backup-db:
	scripts/backup_db.sh $(DB) $(ARGS)
## the database round-trip test: 20,000 rows, every index of DB, then the checks on the outputs
backup-test:
	scripts/backup_db.sh $(DB) --data data/processed/dev --limit 20000 --force
	docker compose --profile db run --rm dbbench uv run --frozen pytest tools/backup/tests/test_db_roundtrip.py -q -k $(DB)
## the cross-language .vro test (Rust writes, Go, C++, Python load), inside bench
vro-test:
	$(RUN) uv run --frozen pytest tools/backup/tests/test_vro_cross.py -q
