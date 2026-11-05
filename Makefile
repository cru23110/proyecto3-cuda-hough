NVCC   ?= nvcc
ARCH   ?= native
FLAGS   = -O3 -arch=$(ARCH) -std=c++17 -Icommon -Ithird_party
COMMON  = common/pgm.cpp common/hough.cpp common/image_io.cpp common/lines.cpp
HEADERS = $(wildcard common/*.h)
BINS    = hough_global hough_const hough_shared

all: $(BINS)

hough_%: src/hough_%.cu $(COMMON) $(HEADERS)
	$(NVCC) $(FLAGS) $< $(COMMON) -o $@

variants: src/experiments/variants.cu $(COMMON) $(HEADERS)
	$(NVCC) $(FLAGS) $< $(COMMON) -o $@

run: all
	mkdir -p output
	for b in $(BINS); do ./$$b runway.pgm || exit 1; done
	-python3 scripts/compare_outputs.py

bench: hough_global
	bash scripts/run_benchmarks.sh

sweep: all
	bash scripts/block_sweep.sh

clean:
	rm -f $(BINS) variants *.exe *.exp *.lib

.PHONY: all run bench sweep clean
