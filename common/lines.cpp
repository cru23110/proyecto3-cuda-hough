#include "lines.h"
#include "hough.h"
#include <algorithm>
#include <cstdio>

// Media + k*sd (bins>0)
float lineThreshold(const int *acc, float k) {
  double sum = 0, sq = 0;
  int n = 0;
  for (int i = 0; i < accSize; i++)
    if (acc[i] > 0) {
      sum += acc[i];
      n++;
    }
  if (n == 0) return 0;
  double mean = sum / n;
  for (int i = 0; i < accSize; i++)
    if (acc[i] > 0) sq += (acc[i] - mean) * (acc[i] - mean);
  return (float)(mean + k * sqrt(sq / n));
}

// Máximo local 5x5
static bool isPeak(const int *acc, int r, int t) {
  int v = acc[r * degreeBins + t];
  for (int dr = -2; dr <= 2; dr++)
    for (int dt = -2; dt <= 2; dt++) {
      int rr = r + dr, tt = t + dt;
      // θ circular: r invertido
      if (tt < 0 || tt >= degreeBins) {
        tt = (tt + degreeBins) % degreeBins;
        rr = rBins - 1 - rr;
      }
      if (rr < 0 || rr >= rBins || (dr == 0 && dt == 0)) continue;
      int u = acc[rr * degreeBins + tt];
      if (u > v || (u == v && rr * degreeBins + tt < r * degreeBins + t)) return false;
    }
  return true;
}

std::vector<Line> selectLines(const int *acc, float threshold, bool nms, int top) {
  std::vector<Line> lines;
  for (int r = 0; r < rBins; r++)
    for (int t = 0; t < degreeBins; t++)
      if (acc[r * degreeBins + t] > threshold && (!nms || isPeak(acc, r, t)))
        lines.push_back({r, t, acc[r * degreeBins + t]});
  std::sort(lines.begin(), lines.end(), [](const Line &a, const Line &b) { return a.votes > b.votes; });
  if (top > 0 && (int)lines.size() > top) lines.resize(top);
  return lines;
}

void drawLines(RGBImage &img, const std::vector<Line> &lines, float threshold, float rMax, float rScale) {
  if (lines.empty()) return;
  int xCent = img.w / 2, yCent = img.h / 2;
  float maxVotes = (float)lines.front().votes;

  // Débiles primero
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    float c = cosf(it->tIdx * radInc), s = sinf(it->tIdx * radInc);
    float r = (it->rIdx + 0.5f) * rScale - rMax;
    float t = (it->votes - threshold) / std::max(maxVotes - threshold, 1.0f);
    RGB color = {255, (unsigned char)(220 * (1 - t)), 0};

    if (fabsf(s) > fabsf(c)) {
      for (int i = 0; i < img.w; i++) {
        int j = yCent - (int)lroundf((r - (i - xCent) * c) / s);
        for (int d = -1; d <= 1; d++) img.put(i, j + d, color);
      }
    } else {
      for (int j = 0; j < img.h; j++) {
        int i = xCent + (int)lroundf((r - (yCent - j) * s) / c);
        for (int d = -1; d <= 1; d++) img.put(i + d, j, color);
      }
    }
  }
}

bool saveHoughImage(const char *path, const int *acc) {
  const int scale = 4;
  int maxV = *std::max_element(acc, acc + accSize);
  RGBImage img(degreeBins * scale, rBins * scale);

  for (int y = 0; y < img.h; y++)
    for (int x = 0; x < img.w; x++) {
      float v = 3 * sqrtf((float)acc[(y / scale) * degreeBins + x / scale] / std::max(maxV, 1));
      auto ch = [](float f) { return (unsigned char)(255 * std::min(std::max(f, 0.0f), 1.0f)); };
      img.put(x, y, {ch(v), ch(v - 1), ch(v - 2)});
    }
  return writePNG(path, img);
}

void saveOutputs(const Options &opt, const char *version, const unsigned char *pic, int w, int h,
                 const int *acc, float rMax, float rScale) {
  const char *prefix = opt.out;
  float threshold = lineThreshold(acc, opt.k);
  std::vector<Line> lines = selectLines(acc, threshold, opt.nms, opt.top);
  printf("Umbral (media + %.1f sd): %.1f votos -> %zu lineas\n", opt.k, threshold, lines.size());

  RGBImage img = grayToRGB(pic, w, h);
  drawLines(img, lines, threshold, rMax, rScale);

  char path[512];
  snprintf(path, sizeof(path), "%s_%s_lines.png", prefix, version);
  printf(writePNG(path, img) ? "Imagen: %s\n" : "No se pudo escribir %s\n", path);
  snprintf(path, sizeof(path), "%s_%s_hough.png", prefix, version);
  printf(saveHoughImage(path, acc) ? "Espacio Hough: %s\n" : "No se pudo escribir %s\n", path);

  // Para comparar versiones
  snprintf(path, sizeof(path), "%s_%s_acc.bin", prefix, version);
  FILE *f = fopen(path, "wb");
  if (f) {
    fwrite(acc, sizeof(int), accSize, f);
    fclose(f);
  }
}
