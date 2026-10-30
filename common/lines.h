#pragma once
#include <vector>
#include "image_io.h"

struct Line {
  int rIdx, tIdx, votes;
};

float lineThreshold(const int *acc, float k);
std::vector<Line> selectLines(const int *acc, float threshold);
void drawLines(RGBImage &img, const std::vector<Line> &lines, float threshold, float rMax, float rScale);
bool saveHoughImage(const char *path, const int *acc);
void saveOutputs(const char *prefix, const char *version, const unsigned char *pic, int w, int h,
                 const int *acc, float rMax, float rScale, float k);
