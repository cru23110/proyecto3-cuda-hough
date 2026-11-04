#include "hough.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

bool parseArgs(int argc, char **argv, Options &opt) {
  if (argc < 2) {
    printf("Uso: %s imagen.pgm [--runs N] [--block B] [--k K] [--top N] [--nms 0|1] [--csv f] [--out prefijo]\n", argv[0]);
    return false;
  }
  opt.input = argv[1];
  for (int i = 2; i < argc; i += 2) {
    if (i + 1 >= argc) {
      printf("Falta el valor de %s\n", argv[i]);
      return false;
    }
    if (!strcmp(argv[i], "--runs")) opt.runs = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--block")) opt.block = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--k")) opt.k = (float)atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--top")) opt.top = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--nms")) opt.nms = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--csv")) opt.csv = argv[i + 1];
    else if (!strcmp(argv[i], "--out")) opt.out = argv[i + 1];
    else {
      printf("Opcion desconocida: %s\n", argv[i]);
      return false;
    }
  }
  if (opt.runs < 1 || opt.block < 32 || opt.block > 1024) {
    printf("Requiere runs >= 1 y 32 <= block <= 1024\n");
    return false;
  }
  return true;
}

void buildTrigTables(float *cosT, float *sinT) {
  for (int t = 0; t < degreeBins; t++) {
    cosT[t] = cosf(t * radInc);
    sinT[t] = sinf(t * radInc);
  }
}

void CPU_HoughTran(const unsigned char *pic, int w, int h, float rMax, float rScale,
                   const float *cosT, const float *sinT, int *acc) {
  memset(acc, 0, sizeof(int) * accSize);
  int xCent = w / 2, yCent = h / 2;

  for (int j = 0; j < h; j++)
    for (int i = 0; i < w; i++) {
      if (pic[j * w + i] == 0) continue;
      int x = i - xCent, y = yCent - j;
      for (int t = 0; t < degreeBins; t++)
        acc[rIndex(x, y, cosT[t], sinT[t], rMax, rScale) * degreeBins + t]++;
    }
}

bool validate(const int *cpu, const int *gpu) {
  int bad = 0;
  long long sumCpu = 0, sumGpu = 0;
  for (int i = 0; i < accSize; i++) {
    sumCpu += cpu[i];
    sumGpu += gpu[i];
    if (cpu[i] != gpu[i] && ++bad <= 5)
      printf("  Mismatch bin %d: CPU=%d GPU=%d\n", i, cpu[i], gpu[i]);
  }
  printf("Validacion: %d mismatches, votos CPU=%lld GPU=%lld -> %s\n",
         bad, sumCpu, sumGpu, bad ? "ERROR" : "OK");
  return bad == 0;
}

void reportTimes(const char *version, const Options &opt, const std::vector<float> &ms, double cpuMs) {
  double sum = 0, sq = 0;
  for (float v : ms) sum += v;
  double mean = sum / ms.size();
  for (float v : ms) sq += (v - mean) * (v - mean);
  double sd = ms.size() > 1 ? sqrt(sq / (ms.size() - 1)) : 0;
  float mn = *std::min_element(ms.begin(), ms.end());
  float mx = *std::max_element(ms.begin(), ms.end());

  printf("Kernel %s (%zu corridas, bloque %d): media %.4f ms, sd %.4f, min %.4f, max %.4f\n",
         version, ms.size(), opt.block, mean, sd, mn, mx);
  printf("CPU: %.3f ms -> speedup %.1fx\n", cpuMs, cpuMs / mean);

  if (!opt.csv) return;
  FILE *f = fopen(opt.csv, "a+");
  if (!f) {
    printf("No se pudo abrir %s\n", opt.csv);
    return;
  }
  fseek(f, 0, SEEK_END);
  if (ftell(f) == 0) fprintf(f, "version,block,run,kernel_ms\n");
  for (size_t i = 0; i < ms.size(); i++)
    fprintf(f, "%s,%d,%zu,%.6f\n", version, opt.block, i + 1, ms[i]);
  fclose(f);
}
