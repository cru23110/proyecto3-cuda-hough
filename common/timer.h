#pragma once
#include <chrono>
#include "cuda_utils.h"

struct GpuTimer {
  cudaEvent_t begin, end;

  void create() {
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
  }
  void start() { CUDA_CHECK(cudaEventRecord(begin)); }
  float stop() {
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    return ms;
  }
  void destroy() {
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
  }
};

struct CpuTimer {
  std::chrono::high_resolution_clock::time_point begin;

  void start() { begin = std::chrono::high_resolution_clock::now(); }
  double stop() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }
};
