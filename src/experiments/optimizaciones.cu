// Optimizaciones de la compartida
// Estilo: [ETIQUETA] nota corta
#include <algorithm>
#include <cstdio>
#include <vector>
#include "cuda_utils.h"
#include "hough.h"
#include "pgm.h"
#include "timer.h"

// [CONST] tablas cos/sin
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// [SHARED] MULTI: varios píxeles; DIV: θ desfasado
template <bool MULTI, bool DIV, bool LDG, bool VOTE = true>
__global__ void kOpt(const unsigned char *__restrict__ pic, int w, int h, int *acc, float rMax,
                     float rScale, const float *__restrict__ gCos, const float *__restrict__ gSin) {
  __shared__ int localAcc[accSize];
  int locID = threadIdx.x;
  for (int i = locID; i < accSize; i += blockDim.x) localAcc[i] = 0;
  // [BARRERA 1] init terminada
  __syncthreads();

  // [ID] grid-stride si MULTI
  int stride = MULTI ? gridDim.x * blockDim.x : w * h;
  // [VOLCADO] sin VOTE: costo fijo
  for (int p = blockIdx.x * blockDim.x + locID; VOTE && p < w * h; p += stride) {
    if (pic[p] == 0) continue;
    int xCoord = p % w - w / 2;
    int yCoord = h / 2 - p / w;
    for (int i = 0; i < degreeBins; i++) {
      // [DIV] θ distinto por hilo
      int t = DIV ? (i + locID) % degreeBins : i;
      // [CONST] o tabla global
      float c = LDG ? __ldg(gCos + t) : d_Cos[t];
      float s = LDG ? __ldg(gSin + t) : d_Sin[t];
      atomicAdd(&localAcc[rIndex(xCoord, yCoord, c, s, rMax, rScale) * degreeBins + t], 1);
    }
  }
  // [BARRERA 2] votos terminados
  __syncthreads();

  // [VOLCADO] solo bins con votos
  for (int i = locID; i < accSize; i += blockDim.x)
    if (localAcc[i] > 0) atomicAdd(&acc[i], localAcc[i]);
}

using KernelFn = void (*)(const unsigned char *, int, int, int *, float, float, const float *, const float *);
struct Config {
  const char *name;
  KernelFn kernel;
  int block;
  bool multi;
};

int main(int argc, char **argv) {
  Options opt;
  opt.runs = 30;
  opt.csv = "results/bitacora_optimizaciones.csv";
  if (!parseArgs(argc, argv, opt)) return 1;

  PGMImage img(opt.input);
  int w = img.x_dim, h = img.y_dim, n = w * h;
  float rMax = sqrtf((float)(w * w + h * h)) / 2;
  float rScale = 2 * rMax / rBins;
  printDeviceInfo();

  // [HOST] tablas y referencia CPU
  float pcCos[degreeBins], pcSin[degreeBins];
  buildTrigTables(pcCos, pcSin);
  std::vector<int> cpuAcc(accSize), gpuAcc(accSize);
  CPU_HoughTran(img.pixels, w, h, rMax, rScale, pcCos, pcSin, cpuAcc.data());

  // [GPU] reservar y copiar
  unsigned char *d_in;
  int *d_acc;
  float *d_gCos, *d_gSin;
  CUDA_CHECK(cudaMalloc(&d_in, n));
  CUDA_CHECK(cudaMalloc(&d_gCos, sizeof(pcCos)));
  CUDA_CHECK(cudaMalloc(&d_gSin, sizeof(pcSin)));
  CUDA_CHECK(cudaMemcpy(d_gCos, pcCos, sizeof(pcCos), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gSin, pcSin, sizeof(pcSin), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * accSize));
  CUDA_CHECK(cudaMemcpy(d_in, img.pixels, n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(pcCos)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(pcSin)));

  std::vector<Config> cfg = {
      {"base_256", kOpt<false, false, false>, 256, false},
      {"base_512", kOpt<false, false, false>, 512, false},
      {"ldg_512", kOpt<false, false, true>, 512, false},
      {"div_const_512", kOpt<false, true, false>, 512, false},
      {"div_ldg_512", kOpt<false, true, true>, 512, false},
      {"multi_256", kOpt<true, false, false>, 256, true},
      {"multi_512", kOpt<true, false, false>, 512, true},
      {"fijo_256", kOpt<false, false, false, false>, 256, false},
      {"fijo_512", kOpt<false, false, false, false>, 512, false},
      {"fijo_multi_512", kOpt<true, false, false, false>, 512, true},
  };

  // [ID] grid: un píxel o lleno
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::vector<int> grids;
  for (auto &c : cfg) {
    int perSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, c.kernel, c.block, 0));
    grids.push_back(c.multi ? perSM * prop.multiProcessorCount : (n + c.block - 1) / c.block);
  }

  GpuTimer timer;
  timer.create();
  std::vector<std::vector<float>> ms(cfg.size());
  std::vector<int> bad(cfg.size());

  // [TIEMPO] intercaladas, sin calentamiento
  for (int run = 0; run <= opt.runs; run++)
    for (size_t v = 0; v < cfg.size(); v++) {
      CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * accSize));
      timer.start();
      cfg[v].kernel<<<grids[v], cfg[v].block>>>(d_in, w, h, d_acc, rMax, rScale, d_gCos, d_gSin);
      float t = timer.stop();
      CUDA_CHECK(cudaGetLastError());
      if (run > 0) ms[v].push_back(t);
      if (run == opt.runs) {
        CUDA_CHECK(cudaMemcpy(gpuAcc.data(), d_acc, sizeof(int) * accSize, cudaMemcpyDeviceToHost));
        for (int i = 0; i < accSize; i++) bad[v] += cpuAcc[i] != gpuAcc[i];
      }
    }

  // [SALIDA] tabla y CSV
  FILE *f = fopen(opt.csv, "w");
  if (f) fprintf(f, "variant,block,grid,run,kernel_ms,mismatches\n");
  printf("%-14s %6s %6s %10s %10s %9s\n", "variante", "bloque", "grid", "mediana", "min_ms", "mismatch");
  for (size_t v = 0; v < cfg.size(); v++) {
    std::vector<float> s = ms[v];
    std::sort(s.begin(), s.end());
    printf("%-14s %6d %6d %10.4f %10.4f %9d\n", cfg[v].name, cfg[v].block, grids[v], s[s.size() / 2], s[0], bad[v]);
    for (int r = 0; f && r < opt.runs; r++)
      fprintf(f, "%s,%d,%d,%d,%.6f,%d\n", cfg[v].name, cfg[v].block, grids[v], r + 1, ms[v][r], bad[v]);
  }
  if (f) fclose(f);

  // [LIMPIEZA] GPU, CPU y reset
  timer.destroy();
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_acc));
  CUDA_CHECK(cudaFree(d_gCos));
  CUDA_CHECK(cudaFree(d_gSin));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
