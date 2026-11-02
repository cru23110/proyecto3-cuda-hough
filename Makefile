NVCC   ?= nvcc
ARCH   ?= native
FLAGS   = -O3 -arch=$(ARCH) -std=c++17 -Icommon -Ithird_party
COMMON  = common/pgm.cpp common/hough.cpp common/image_io.cpp common/lines.cpp
HEADERS = $(wildcard common/*.h)

all: hough_global hough_shared

hough_global: src/hough_global.cu $(COMMON) $(HEADERS)
	$(NVCC) $(FLAGS) src/hough_global.cu $(COMMON) -o $@

hough_shared: src/hough_shared.cu $(COMMON) $(HEADERS)
	$(NVCC) $(FLAGS) src/hough_shared.cu $(COMMON) -o $@

run: hough_global
	mkdir -p output
	./hough_global runway.pgm

bench: hough_global
	bash scripts/run_benchmarks.sh

clean:
	rm -f hough_global hough_shared *.exe *.exp *.lib

.PHONY: all run bench clean
