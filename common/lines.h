#pragma once
#include <vector>
#include "hough.h"
#include "image_io.h"

struct Line {
  int rIdx, tIdx, votes;
};

float lineThreshold(const int *acc, float k);
std::vector<Line> selectLines(const int *acc, float threshold, bool nms, int top);
void drawLines(RGBImage &img, const std::vector<Line> &lines, float threshold, float rMax, float rScale);
bool saveHoughImage(const char *path, const int *acc);
void saveOutputs(const Options &opt, const char *version, const unsigned char *pic, int w, int h,
                 const int *acc, float rMax, float rScale);
