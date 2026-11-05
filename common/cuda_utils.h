#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

inline void printDeviceInfo() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
}

// Bloques activos por SM
template <class Kernel>
void printOccupancy(Kernel kernel, int block) {
  cudaDeviceProp p;
  int perSM = 0;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kernel, block, 0));
  printf("Ocupancia: %d bloques/SM, %d%% de hilos\n", perSM,
         100 * perSM * block / p.maxThreadsPerMultiProcessor);
}
