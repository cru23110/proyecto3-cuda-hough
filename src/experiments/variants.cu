// Experimentos: trig y compartida
// Estilo: [ETIQUETA] nota corta
#include <cstdio>
#include <cstring>
#include <vector>
#include "cuda_utils.h"
#include "hough.h"
#include "pgm.h"
#include "timer.h"

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

enum Variant {
  OnTheFly, Intrinsic, Global, GlobalLdg, Constant, ConstantDiv, GlobalLdgDiv,
  SharedSkip, SharedAll, SharedFixed, NumVariants
};
const char *groups[NumVariants] = {"trig", "trig", "trig", "trig", "trig", "trig", "trig",
                                   "shared", "shared", "shared"};
const char *names[NumVariants] = {"cosf_kernel", "sincosf_intrin", "global", "global_ldg", "constant",
                                  "constant_div", "global_ldg_div",
                                  "skip_ceros", "volcar_todo", "solo_init_volcado"};

// [TRIG] solo cambia cos/sin
template <int V>
__global__ void kTrig(const unsigned char *__restrict__ pic, int w, int h, int *acc, float rMax,
                      float rScale, const float *gCos, const float *gSin) {
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h || pic[gloID] == 0) return;
  int xCoord = gloID % w - w / 2;
  int yCoord = h / 2 - gloID / w;

  for (int i = 0; i < degreeBins; i++) {
    // [DIV] θ distinto por hilo
    int t = (V == ConstantDiv || V == GlobalLdgDiv) ? (i + threadIdx.x) % degreeBins : i;
    float c, s;
    if constexpr (V == OnTheFly) { c = cosf(t * radInc); s = sinf(t * radInc); }
    else if constexpr (V == Intrinsic) __sincosf(t * radInc, &s, &c);
    else if constexpr (V == Global) { c = gCos[t]; s = gSin[t]; }
    else if constexpr (V == Constant || V == ConstantDiv) { c = d_Cos[t]; s = d_Sin[t]; }
    else { c = __ldg(gCos + t); s = __ldg(gSin + t); }
    atomicAdd(&acc[rIndex(xCoord, yCoord, c, s, rMax, rScale) * degreeBins + t], 1);
  }
}

// [SHARED] solo cambia volcado
template <int V>
__global__ void kShared(const unsigned char *__restrict__ pic, int w, int h, int *acc, float rMax,
                        float rScale, const float *, const float *) {
  __shared__ int localAcc[accSize];
  int locID = threadIdx.x;
  int gloID = blockIdx.x * blockDim.x + locID;
  for (int i = locID; i < accSize; i += blockDim.x) localAcc[i] = 0;
  // [BARRERA 1] init terminada
  __syncthreads();

  if (V != SharedFixed && gloID < w * h && pic[gloID] > 0) {
    int xCoord = gloID % w - w / 2;
    int yCoord = h / 2 - gloID / w;
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
      int rIdx = rIndex(xCoord, yCoord, d_Cos[tIdx], d_Sin[tIdx], rMax, rScale);
      atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
    }
  }
  // [BARRERA 2] votos terminados
  __syncthreads();

  // [VOLCADO] todo o solo >0
  for (int i = locID; i < accSize; i += blockDim.x)
    if (V == SharedAll || localAcc[i] > 0) atomicAdd(&acc[i], localAcc[i]);
}

using KernelFn = void (*)(const unsigned char *, int, int, int *, float, float, const float *, const float *);
KernelFn kernels[NumVariants] = {kTrig<OnTheFly>, kTrig<Intrinsic>, kTrig<Global>, kTrig<GlobalLdg>,
                                 kTrig<Constant>, kTrig<ConstantDiv>, kTrig<GlobalLdgDiv>,
                                 kShared<SharedSkip>, kShared<SharedAll>, kShared<SharedFixed>};

int main(int argc, char **argv) {
  Options opt;
  opt.runs = 30;
  opt.csv = "results/bitacora_variantes.csv";
  if (!parseArgs(argc, argv, opt)) return 1;

  PGMImage img(opt.input);
  int w = img.x_dim, h = img.y_dim, n = w * h;
  float rMax = sqrtf((float)(w * w + h * h)) / 2;
  float rScale = 2 * rMax / rBins;
  printDeviceInfo();

  float pcCos[degreeBins], pcSin[degreeBins];
  buildTrigTables(pcCos, pcSin);
  std::vector<int> cpuAcc(accSize), gpuAcc(accSize);
  CPU_HoughTran(img.pixels, w, h, rMax, rScale, pcCos, pcSin, cpuAcc.data());

  unsigned char *d_in;
  int *d_acc;
  float *d_gCos, *d_gSin;
  CUDA_CHECK(cudaMalloc(&d_in, n));
  CUDA_CHECK(cudaMalloc(&d_acc, sizeof(int) * accSize));
  CUDA_CHECK(cudaMalloc(&d_gCos, sizeof(pcCos)));
  CUDA_CHECK(cudaMalloc(&d_gSin, sizeof(pcSin)));
  CUDA_CHECK(cudaMemcpy(d_in, img.pixels, n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gCos, pcCos, sizeof(pcCos), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gSin, pcSin, sizeof(pcSin), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(pcCos)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(pcSin)));

  int blocks = (n + opt.block - 1) / opt.block;
  GpuTimer timer;
  timer.create();
  std::vector<float> ms[NumVariants];
  int bad[NumVariants];

  // [TIEMPO] intercaladas, sin calentamiento
  for (int run = 0; run <= opt.runs; run++)
    for (int v = 0; v < NumVariants; v++) {
      CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(int) * accSize));
      timer.start();
      kernels[v]<<<blocks, opt.block>>>(d_in, w, h, d_acc, rMax, rScale, d_gCos, d_gSin);
      float t = timer.stop();
      CUDA_CHECK(cudaGetLastError());
      if (run > 0) ms[v].push_back(t);
      if (run == opt.runs) {
        CUDA_CHECK(cudaMemcpy(gpuAcc.data(), d_acc, sizeof(int) * accSize, cudaMemcpyDeviceToHost));
        bad[v] = 0;
        for (int i = 0; i < accSize; i++) bad[v] += cpuAcc[i] != gpuAcc[i];
      }
    }

  FILE *f = fopen(opt.csv, "w");
  if (f) fprintf(f, "group,variant,block,run,kernel_ms,mismatches\n");
  printf("%-8s %-20s %10s %10s %10s\n", "grupo", "variante", "media_ms", "min_ms", "mismatch");
  for (int v = 0; v < NumVariants; v++) {
    double sum = 0;
    float mn = ms[v][0];
    for (float t : ms[v]) sum += t, mn = t < mn ? t : mn;
    printf("%-8s %-20s %10.4f %10.4f %10d\n", groups[v], names[v], sum / opt.runs, mn, bad[v]);
    for (int r = 0; f && r < opt.runs; r++)
      fprintf(f, "%s,%s,%d,%d,%.6f,%d\n", groups[v], names[v], opt.block, r + 1, ms[v][r], bad[v]);
  }
  if (f) fclose(f);

  timer.destroy();
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_acc));
  CUDA_CHECK(cudaFree(d_gCos));
  CUDA_CHECK(cudaFree(d_gSin));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
