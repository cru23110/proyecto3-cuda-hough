// Hough: global, constante, compartida
#include <cstdio>
#include <vector>
#include "cuda_utils.h"
#include "hough.h"
#include "lines.h"
#include "pgm.h"
#include "timer.h"

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

static_assert(accSize * sizeof(int) <= 48 * 1024, "localAcc excede 48KB");

__global__ void GPU_HoughTranShared(const unsigned char *__restrict__ pic, int w, int h, int *acc,
                                    float rMax, float rScale) {
  __shared__ int localAcc[accSize];
  int locID = threadIdx.x;
  int gloID = blockIdx.x * blockDim.x + locID;

  for (int i = locID; i < accSize; i += blockDim.x) localAcc[i] = 0;
  __syncthreads();

  // Sin return: hay barreras
  if (gloID < w * h && pic[gloID] > 0) {
    int xCoord = gloID % w - w / 2;
    int yCoord = h / 2 - gloID / w;
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
      int rIdx = rIndex(xCoord, yCoord, d_Cos[tIdx], d_Sin[tIdx], rMax, rScale);
      atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
    }
  }
}

int main(int argc, char **argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt)) return 1;

  PGMImage img(opt.input);
  int w = img.x_dim, h = img.y_dim, n = w * h;
  float rMax = sqrtf((float)(w * w + h * h)) / 2;
  float rScale = 2 * rMax / rBins;
  printf("Imagen %dx%d\n", w, h);
  printDeviceInfo();

  float *pcCos = new float[degreeBins];
  float *pcSin = new float[degreeBins];
  buildTrigTables(pcCos, pcSin);

  int *cpuAcc = new int[accSize];
  int *gpuAcc = new int[accSize];
  CpuTimer cpuTimer;
  cpuTimer.start();
  CPU_HoughTran(img.pixels, w, h, rMax, rScale, pcCos, pcSin, cpuAcc);
  double cpuMs = cpuTimer.stop();

  unsigned char *d_in;
  int *d_acc;
  CUDA_CHECK(cudaMalloc(&d_in, n));
  CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * accSize));
  CUDA_CHECK(cudaMemcpy(d_in, img.pixels, n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins));
  CUDA_CHECK(cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins));

  int blocks = (n + opt.block - 1) / opt.block;
  GpuTimer timer;
  timer.create();
  std::vector<float> times;

  // Corrida 0: calentamiento
  for (int run = 0; run <= opt.runs; run++) {
    CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * accSize));
    timer.start();
    GPU_HoughTranShared<<<blocks, opt.block>>>(d_in, w, h, d_acc, rMax, rScale);
    float ms = timer.stop();
    CUDA_CHECK(cudaGetLastError());
    if (run > 0) times.push_back(ms);
  }
  CUDA_CHECK(cudaMemcpy(gpuAcc, d_acc, sizeof(int) * accSize, cudaMemcpyDeviceToHost));

  reportTimes("shared", opt, times, cpuMs);
  bool ok = validate(cpuAcc, gpuAcc);
  saveOutputs(opt.out, "shared", img.pixels, w, h, gpuAcc, rMax, rScale, opt.k);

  timer.destroy();
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_acc));
  delete[] pcCos;
  delete[] pcSin;
  delete[] cpuAcc;
  delete[] gpuAcc;
  CUDA_CHECK(cudaDeviceReset());
  return ok ? 0 : 1;
}
