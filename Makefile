VAST_HOST ?= ssh3.vast.ai
VAST_PORT ?= 38507
VAST_USER ?= root
VAST_DIR ?= /workspace/bumblebee
VAST_REPO ?= https://github.com/AfreedHassan/bumblebee.git
VAST_BRANCH ?= main
VAST_CUDA_HOST_CXX ?= g++-13
CMD ?= ./build/bumblebee

.PHONY: main run_on_vast

main:
	./scripts/build.sh ./build/bumblebee

run_on_vast:
	ssh -p $(VAST_PORT) $(VAST_USER)@$(VAST_HOST) 'set -e; if [ -d "$(VAST_DIR)/.git" ]; then git -C "$(VAST_DIR)" fetch origin "$(VAST_BRANCH)"; git -C "$(VAST_DIR)" switch "$(VAST_BRANCH)" 2>/dev/null || git -C "$(VAST_DIR)" switch --track -c "$(VAST_BRANCH)" "origin/$(VAST_BRANCH)"; git -C "$(VAST_DIR)" merge --ff-only "origin/$(VAST_BRANCH)"; elif [ -d "$(VAST_DIR)" ]; then git -C "$(VAST_DIR)" init -b "$(VAST_BRANCH)"; git -C "$(VAST_DIR)" remote add origin "$(VAST_REPO)"; git -C "$(VAST_DIR)" fetch origin "$(VAST_BRANCH)"; git -C "$(VAST_DIR)" reset "origin/$(VAST_BRANCH)"; git -C "$(VAST_DIR)" branch --set-upstream-to="origin/$(VAST_BRANCH)" "$(VAST_BRANCH)"; else git clone --branch "$(VAST_BRANCH)" "$(VAST_REPO)" "$(VAST_DIR)"; fi; cd "$(VAST_DIR)"; rm -rf build; CC=gcc-16 CXX=g++-16 CUDAHOSTCXX=$(VAST_CUDA_HOST_CXX) ./scripts/build.sh $(CMD)'
