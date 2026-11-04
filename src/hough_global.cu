// Hough: memoria global
// Estilo: [ETIQUETA] nota corta
#include <cstdio>
#include <vector>
#include "cuda_utils.h"
#include "hough.h"
#include "lines.h"
#include "pgm.h"
#include "timer.h"

__global__ void GPU_HoughTran(const unsigned char *__restrict__ pic, int w, int h, int *acc,
                              float rMax, float rScale,
                              const float *__restrict__ d_Cos, const float *__restrict__ d_Sin) {
  // [ID] un hilo, un píxel
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h || pic[gloID] == 0) return;

  // [COORD] origen al centro
  int xCoord = gloID % w - w / 2;
  int yCoord = h / 2 - gloID / w;

  for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
    int rIdx = rIndex(xCoord, yCoord, d_Cos[tIdx], d_Sin[tIdx], rMax, rScale);
    // [ATOMIC] varios píxeles, mismo bin
    atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
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

  // [HOST] tablas y referencia CPU
  float *pcCos = new float[degreeBins];
  float *pcSin = new float[degreeBins];
  buildTrigTables(pcCos, pcSin);

  int *cpuAcc = new int[accSize];
  int *gpuAcc = new int[accSize];
  CpuTimer cpuTimer;
  cpuTimer.start();
  CPU_HoughTran(img.pixels, w, h, rMax, rScale, pcCos, pcSin, cpuAcc);
  double cpuMs = cpuTimer.stop();

  // [GPU] reservar y copiar
  unsigned char *d_in;
  int *d_acc;
  float *d_Cos, *d_Sin;
  CUDA_CHECK(cudaMalloc(&d_in, n));
  CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * accSize));
  CUDA_CHECK(cudaMalloc(&d_Cos, sizeof(float) * degreeBins));
  CUDA_CHECK(cudaMalloc(&d_Sin, sizeof(float) * degreeBins));
  CUDA_CHECK(cudaMemcpy(d_in, img.pixels, n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice));

  int blocks = (n + opt.block - 1) / opt.block;
  GpuTimer timer;
  timer.create();
  std::vector<float> times;

  // [TIEMPO] corrida 0: calentamiento
  for (int run = 0; run <= opt.runs; run++) {
    CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * accSize));
    timer.start();
    GPU_HoughTran<<<blocks, opt.block>>>(d_in, w, h, d_acc, rMax, rScale, d_Cos, d_Sin);
    float ms = timer.stop();
    CUDA_CHECK(cudaGetLastError());
    if (run > 0) times.push_back(ms);
  }
  // [SALIDA] validar y dibujar
  CUDA_CHECK(cudaMemcpy(gpuAcc, d_acc, sizeof(int) * accSize, cudaMemcpyDeviceToHost));

  reportTimes("global", opt, times, cpuMs);
  bool ok = validate(cpuAcc, gpuAcc);
  saveOutputs(opt, "global", img.pixels, w, h, gpuAcc, rMax, rScale);

  // [LIMPIEZA] GPU, CPU y reset
  timer.destroy();
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_acc));
  CUDA_CHECK(cudaFree(d_Cos));
  CUDA_CHECK(cudaFree(d_Sin));
  delete[] pcCos;
  delete[] pcSin;
  delete[] cpuAcc;
  delete[] gpuAcc;
  CUDA_CHECK(cudaDeviceReset());
  return ok ? 0 : 1;
}
