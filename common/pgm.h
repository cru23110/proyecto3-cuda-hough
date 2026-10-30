#pragma once

class PGMImage {
public:
  int x_dim = 0, y_dim = 0, num_colors = 0;
  unsigned char *pixels = nullptr;

  PGMImage(const char *fname);
  ~PGMImage();
  bool write(const char *fname) const;
};
