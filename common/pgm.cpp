#include "pgm.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>

// Salta espacios y comentarios
static int readInt(FILE *f) {
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (c == '#')
      while ((c = fgetc(f)) != '\n' && c != EOF);
    else if (!isspace(c))
      break;
  }
  ungetc(c, f);
  int v;
  return fscanf(f, "%d", &v) == 1 ? v : -1;
}

static void fail(const char *msg, const char *fname) {
  fprintf(stderr, "%s: %s\n", msg, fname);
  exit(EXIT_FAILURE);
}

PGMImage::PGMImage(const char *fname) {
  FILE *f = fopen(fname, "rb");
  if (!f) fail("No se pudo abrir", fname);

  char magic[2];
  if (fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || (magic[1] != '5' && magic[1] != '2'))
    fail("No es PGM (P5/P2)", fname);

  x_dim = readInt(f);
  y_dim = readInt(f);
  num_colors = readInt(f);
  if (x_dim <= 0 || y_dim <= 0 || num_colors <= 0 || num_colors > 255)
    fail("Encabezado PGM invalido", fname);

  int n = x_dim * y_dim;
  pixels = new unsigned char[n];

  if (magic[1] == '5') {
    fgetc(f);
    if ((int)fread(pixels, 1, n, f) != n) fail("PGM incompleto", fname);
  } else {
    for (int i = 0; i < n; i++) {
      int v = readInt(f);
      if (v < 0) fail("PGM incompleto", fname);
      pixels[i] = (unsigned char)v;
    }
  }
  fclose(f);
}

PGMImage::~PGMImage() { delete[] pixels; }

bool PGMImage::write(const char *fname) const {
  FILE *f = fopen(fname, "wb");
  if (!f) return false;
  fprintf(f, "P5\n%d %d\n%d\n", x_dim, y_dim, num_colors);
  bool ok = (int)fwrite(pixels, 1, x_dim * y_dim, f) == x_dim * y_dim;
  fclose(f);
  return ok;
}
