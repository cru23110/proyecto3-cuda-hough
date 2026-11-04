#pragma once
#include <cmath>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr int degreeInc = 2;
constexpr int degreeBins = 180 / degreeInc;
constexpr int rBins = 100;
constexpr int accSize = degreeBins * rBins;
constexpr float radInc = degreeInc * 3.14159265358979f / 180;

struct Options {
  const char *input = nullptr;
  const char *csv = nullptr;
  const char *out = "output/runway";
  int runs = 10;
  int block = 256;
  float k = 2.0f;
  int top = 0;
  int nms = 1;
};

// Misma fórmula CPU/GPU
HD inline int rIndex(int x, int y, float c, float s, float rMax, float rScale) {
  float r = fmaf((float)x, c, y * s);
  int idx = (int)((r + rMax) / rScale);
  return idx < rBins ? idx : rBins - 1;
}

bool parseArgs(int argc, char **argv, Options &opt);
void buildTrigTables(float *cosT, float *sinT);
void CPU_HoughTran(const unsigned char *pic, int w, int h, float rMax, float rScale,
                   const float *cosT, const float *sinT, int *acc);
bool validate(const int *cpu, const int *gpu);
void reportTimes(const char *version, const Options &opt, const std::vector<float> &ms, double cpuMs);
