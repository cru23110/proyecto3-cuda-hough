#pragma once
#include <vector>

struct RGB {
  unsigned char r, g, b;
};

struct RGBImage {
  int w, h;
  std::vector<unsigned char> data;

  RGBImage(int w, int h) : w(w), h(h), data(w * h * 3, 0) {}
  void put(int x, int y, RGB c);
};

RGBImage grayToRGB(const unsigned char *gray, int w, int h);
bool writePNG(const char *path, const RGBImage &img);
