# On Windows use Git Bash + `mingw32-make <target>` (or WSL `make`).
SHELL := bash
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif
CONFIG ?= configs/baselines/b3.yaml
CLIP ?= data/videos/accident/cctv_compilation_acc_ved.mp4

.PHONY: venv infra up down smoke test logs case-library experiments evaluate figures demo dashboard

venv:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

infra:         ## stream backbone only: docker compose if available, else Kafka+Prometheus in WSL
	@if command -v docker >/dev/null 2>&1; then docker compose up -d --wait; \
	elif command -v wsl.exe >/dev/null 2>&1; then wsl.exe -d Ubuntu -- bash scripts/infra_wsl.sh up; \
	else bash scripts/infra_wsl.sh up; fi
	$(PY) -c "from xstream.config import load_config; from xstream.bus import ensure_topics; ensure_topics(load_config())"

up:            ## backbone + all five pipeline services (live namespace)
	bash scripts/run_all.sh $(CONFIG)

down:
	-bash scripts/stop_services.sh
	-@if command -v docker >/dev/null 2>&1; then docker compose down; \
	elif command -v wsl.exe >/dev/null 2>&1; then wsl.exe -d Ubuntu -- bash scripts/infra_wsl.sh down; \
	else bash scripts/infra_wsl.sh down; fi

smoke:         ## M0 acceptance: produce + consume one message
	$(PY) scripts/smoke_test.py

test:
	$(PY) -m pytest -q

case-library:  ## offline; falls back to the dev clips when UCF train videos are missing
	$(PY) scripts/build_case_library.py --source $(if $(wildcard data/ucf_crime/videos/*/*.mp4),ucf_train,devset)

experiments:   ## B1, B2 (W=2,4,8), B3 + ablations, then figures
	for c in configs/baselines/b1.yaml configs/baselines/b2.yaml configs/baselines/b3.yaml; do \
	  $(PY) scripts/run_experiment.py --config $$c || exit 1; done
	$(PY) scripts/make_figures.py

figures:
	$(PY) scripts/make_figures.py

dashboard:
	$(PY) -m streamlit run dashboard/app.py

demo:          ## services + dashboard + replay one accident clip at native speed
	bash scripts/run_all.sh configs/demo.yaml
	$(PY) -m streamlit run dashboard/app.py --server.headless true & echo $$! >> runs/.pids
	sleep 20
	$(PY) scripts/replay_producer.py $(CLIP) --speed 1

logs:
	tail -f runs/live/logs/*.log
