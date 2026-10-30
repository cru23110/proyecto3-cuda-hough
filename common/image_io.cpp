#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image_io.h"

void RGBImage::put(int x, int y, RGB c) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  unsigned char *p = &data[(y * w + x) * 3];
  p[0] = c.r;
  p[1] = c.g;
  p[2] = c.b;
}

RGBImage grayToRGB(const unsigned char *gray, int w, int h) {
  RGBImage img(w, h);
  int maxV = 1;
  for (int i = 0; i < w * h; i++) maxV = gray[i] > maxV ? gray[i] : maxV;

  // Estira el contraste
  for (int i = 0; i < w * h; i++)
    img.data[i * 3] = img.data[i * 3 + 1] = img.data[i * 3 + 2] = (unsigned char)(gray[i] * 255 / maxV);
  return img;
}

bool writePNG(const char *path, const RGBImage &img) {
  return stbi_write_png(path, img.w, img.h, 3, img.data.data(), img.w * 3) != 0;
}
