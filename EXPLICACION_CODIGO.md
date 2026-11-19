# Explicación del código: Transformada de Hough en CUDA

Guía para estudiar el código antes de la presentación. Recorre el programa en el orden en que se ejecuta, explica cada función importante y dedica una sección completa a la **sincronización**.

Los números de ejemplo usan la imagen de prueba `runway.pgm` (800 × 600 píxeles, 138 406 encendidos).

---

## Índice

1. [La idea en 30 segundos](#1-la-idea-en-30-segundos)
2. [Mapa de archivos](#2-mapa-de-archivos)
3. [Constantes y parámetros (`hough.h`)](#3-constantes-y-parámetros-houghh)
4. [Flujo del `main`, paso a paso](#4-flujo-del-main-paso-a-paso)
5. [Los tres kernels](#5-los-tres-kernels)
6. [Sincronización: todo lo que hay que saber](#6-sincronización-todo-lo-que-hay-que-saber)
7. [Funciones de apoyo en CPU](#7-funciones-de-apoyo-en-cpu)
8. [Detección y dibujo de líneas (`lines.cpp`)](#8-detección-y-dibujo-de-líneas-linescpp)
9. [Medición de tiempo (`timer.h`)](#9-medición-de-tiempo-timerh)
10. [Manejo de errores y liberación de memoria](#10-manejo-de-errores-y-liberación-de-memoria)
11. [Preguntas probables del catedrático](#11-preguntas-probables-del-catedrático)

---

## 1. La idea en 30 segundos

Una recta se puede describir con dos números: un ángulo **θ** y una distancia **r** al origen:

```
r = x · cos(θ) + y · sin(θ)
```

La Transformada de Hough es una **votación**:

1. Cada píxel encendido (un borde) prueba los 90 ángulos posibles.
2. Para cada ángulo calcula su `r` y le da **un voto** a la casilla `(r, θ)` de una matriz llamada **acumulador**.
3. Las casillas con muchos votos son rectas que de verdad existen en la imagen.

**Por qué paraleliza bien:** cada píxel vota sin depender de los demás, así que se asigna **un hilo de GPU por píxel**. El único punto delicado es que muchos hilos votan en la **misma casilla** al mismo tiempo, y por eso se necesitan operaciones atómicas.

---

## 2. Mapa de archivos

| Archivo | Qué contiene |
|---|---|
| `src/hough_global.cu` | Versión 1: todo en memoria **global** |
| `src/hough_const.cu` | Versión 2: tablas cos/sin en memoria **constante** |
| `src/hough_shared.cu` | Versión 3 (final): constante + acumulador local en memoria **compartida** |
| `common/hough.h` / `hough.cpp` | Constantes, `rIndex`, opciones de línea de comandos, versión CPU, validación, reporte de tiempos |
| `common/timer.h` | `GpuTimer` (CUDA events) y `CpuTimer` |
| `common/cuda_utils.h` | Macro `CUDA_CHECK`, info de la GPU, ocupación |
| `common/lines.cpp` | Umbral, detección de picos, dibujo de líneas, imágenes de salida |
| `common/pgm.cpp` | Lectura de la imagen PGM de entrada |
| `common/image_io.cpp` | Imagen RGB y escritura PNG (usa `stb_image_write`) |
| `src/experiments/` | Experimentos extra (variantes de acceso a cos/sin y optimizaciones probadas) |

Las tres versiones tienen **el mismo `main`**. Solo cambian el kernel y la forma de pasar `d_Cos`/`d_Sin`. Todo lo demás está compartido en `common/`.

### Estilo de comentarios

Cada comentario empieza con una etiqueta, igual en los tres archivos. Buscar una etiqueta lleva a la misma etapa en cualquier versión:

| Etiqueta | Marca |
|---|---|
| `[ID]` | cálculo de índices del hilo |
| `[COORD]` | coordenadas centradas |
| `[ATOMIC]` | sumas atómicas |
| `[CONST]` | memoria constante |
| `[SHARED]` | memoria compartida |
| `[BARRERA 1]`, `[BARRERA 2]` | `__syncthreads()` |
| `[VOLCADO]` | paso del acumulador local al global |
| `[HOST]`, `[GPU]`, `[TIEMPO]`, `[SALIDA]`, `[LIMPIEZA]` | etapas del `main` |

---

## 3. Constantes y parámetros (`hough.h`)

```cpp
constexpr int degreeInc = 2;                    // paso de 2 grados
constexpr int degreeBins = 180 / degreeInc;     // 90 ángulos: 0°, 2°, …, 178°
constexpr int rBins = 100;                      // 100 intervalos de distancia
constexpr int accSize = degreeBins * rBins;     // 9 000 casillas en el acumulador
constexpr float radInc = degreeInc * π / 180;   // paso en radianes
```

- **¿Por qué θ solo va de 0° a 178°?** Una recta con ángulo θ + 180° es la misma que con θ, solo con `r` de signo opuesto. Con 180° se cubren todas las rectas.
- **El acumulador** es un arreglo 1D de 9 000 enteros que representa una matriz de 100 filas (r) × 90 columnas (θ). La casilla `(rIdx, tIdx)` está en `acc[rIdx * degreeBins + tIdx]`.

### `rIndex`: de distancia a casilla

```cpp
HD inline int rIndex(int x, int y, float c, float s, float rMax, float rScale) {
  float r = fmaf((float)x, c, y * s);     // r = x·cos + y·sin
  int idx = (int)((r + rMax) / rScale);   // [-rMax, rMax] → [0, rBins)
  return idx < rBins ? idx : rBins - 1;   // r == rMax cae en la última casilla
}
```

- `r` puede ser negativa, así que se le suma `rMax` para que empiece en 0 y se divide entre `rScale` (el ancho de cada casilla).
- Para `runway.pgm`: `rMax = √(800² + 600²) / 2 = 500` y `rScale = 2 · 500 / 100 = 10`. Cada casilla cubre 10 píxeles de distancia.
- **`HD`** = `__host__ __device__`: la misma función se compila para la CPU **y** para la GPU.
- **`fmaf`** (*fused multiply-add*) calcula `x·c + y·s` con un solo redondeo. Se usa en las dos versiones para que la CPU y la GPU den **exactamente** el mismo resultado. Sin esto, una diferencia de redondeo puede mandar un voto a la casilla vecina y la validación falla.

---

## 4. Flujo del `main`, paso a paso

El `main` es igual en las tres versiones. Usando `hough_shared.cu` como referencia:

### Paso 1: leer opciones e imagen

```cpp
Options opt;
if (!parseArgs(argc, argv, opt)) return 1;
PGMImage img(opt.input);
float rMax = sqrtf((float)(w * w + h * h)) / 2;
float rScale = 2 * rMax / rBins;
```

`rMax` es la mitad de la diagonal: la distancia máxima posible desde el centro de la imagen a una esquina.

### Paso 2: `[HOST]` tablas trigonométricas y versión CPU

```cpp
float *pcCos = new float[degreeBins];
float *pcSin = new float[degreeBins];
buildTrigTables(pcCos, pcSin);                // los 90 cos y sin, una sola vez
CPU_HoughTran(..., cpuAcc);                   // resultado de referencia
```

La CPU calcula el acumulador correcto para después **comparar** el de la GPU, y mide su tiempo para calcular el speedup.

### Paso 3: `[GPU]` reservar memoria y copiar datos

```cpp
cudaMalloc(&d_in, n);                                   // imagen en la GPU
cudaMalloc(&d_acc, sizeof(int) * accSize);              // acumulador global
cudaMemcpy(d_in, img.pixels, n, cudaMemcpyHostToDevice);
cudaMemcpyToSymbol(d_Cos, pcCos, ...);                  // versiones const y shared
cudaMemcpyToSymbol(d_Sin, pcSin, ...);
```

En la versión global, `d_Cos` y `d_Sin` se reservan con `cudaMalloc` y se copian con `cudaMemcpy`. Ver [sección 5.2](#52-versión-constante-hough_constcu).

### Paso 4: geometría del grid

```cpp
int blocks = (n + opt.block - 1) / opt.block;
```

Es una **división hacia arriba**: hacen falta suficientes bloques para cubrir todos los píxeles. Con 480 000 píxeles y bloques de 256 hilos salen **1 875 bloques**. Si `n` no fuera múltiplo de 256, el último bloque tendría hilos sobrantes, y por eso el kernel revisa `gloID < w * h`.

### Paso 5: `[TIEMPO]` lanzar y medir el kernel

```cpp
for (int run = 0; run <= opt.runs; run++) {
  cudaMemset(d_acc, 0, sizeof(int) * accSize);     // acumulador a cero
  timer.start();
  GPU_HoughTranShared<<<blocks, opt.block>>>(d_in, w, h, d_acc, rMax, rScale);
  float ms = timer.stop();
  cudaGetLastError();
  if (run > 0) times.push_back(ms);                // corrida 0 = calentamiento
}
```

- **`<<<blocks, opt.block>>>`** es la configuración de lanzamiento: 1 875 bloques de 256 hilos, un grid 1D.
- **Corrida 0 descartada:** la primera ejecución incluye la carga del contexto y del módulo CUDA y sale más lenta. No representa el tiempo real del kernel.
- **`cudaMemset` fuera del intervalo medido:** se mide solo el kernel, no la limpieza.
- **`cudaGetLastError()`:** los errores de lanzamiento (por ejemplo, demasiados hilos por bloque) no se reportan solos. Hay que preguntarlos.

### Paso 6: `[SALIDA]` traer resultado, validar y dibujar

```cpp
cudaMemcpy(gpuAcc, d_acc, ..., cudaMemcpyDeviceToHost);
reportTimes("shared", opt, times, cpuMs);
bool ok = validate(cpuAcc, gpuAcc);
saveOutputs(opt, "shared", img.pixels, w, h, gpuAcc, rMax, rScale);
```

### Paso 7: `[LIMPIEZA]` liberar todo

Ver [sección 10](#10-manejo-de-errores-y-liberación-de-memoria).

---

## 5. Los tres kernels

### 5.1 Versión global (`hough_global.cu`)

```cpp
__global__ void GPU_HoughTran(const unsigned char *__restrict__ pic, int w, int h, int *acc,
                              float rMax, float rScale,
                              const float *__restrict__ d_Cos, const float *__restrict__ d_Sin) {
  // [ID] un hilo, un píxel
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h || pic[gloID] == 0) return;

  // [COORD] origen al centro
  int xCoord = gloID % w - w / 2;
  int yCoord = h / 2 - gloID / w;

  for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
    int rIdx = rIndex(xCoord, yCoord, d_Cos[tIdx], d_Sin[tIdx], rMax, rScale);
    // [ATOMIC] varios píxeles, mismo bin
    atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
  }
}
```

#### `gloID`: el índice global del hilo

```
gloID = blockIdx.x * blockDim.x + threadIdx.x
        └─ bloque ─┘  └ hilos ┘   └ posición ┘
                      por bloque   dentro del bloque
```

Ejemplo con bloques de 256: el hilo 10 del bloque 3 tiene `gloID = 3 · 256 + 10 = 778`. Como el grid es 1D y la imagen está guardada fila por fila, `gloID` es directamente el índice del píxel.

#### Filtro de entrada

`if (gloID >= w * h || pic[gloID] == 0) return;`

- `gloID >= w * h`: hilos sobrantes del último bloque, que no tienen píxel.
- `pic[gloID] == 0`: píxel apagado (no es borde), así que no vota.

Aquí el `return` temprano **sí es válido**, porque este kernel no tiene barreras. En la versión compartida **no** se puede hacer así (ver [sección 6.2](#62-la-trampa-del-return-temprano)).

#### Coordenadas centradas (`xCoord`, `yCoord`), pregunta 3 del enunciado

```cpp
int xCoord = gloID % w - w / 2;   // columna, desplazada al centro
int yCoord = h / 2 - gloID / w;   // fila, desplazada e invertida
```

1. **De índice lineal a 2D:** `gloID % w` es la columna y `gloID / w` es la fila. La imagen está "cortada" en filas puestas una tras otra.
2. **Origen al centro:** se resta `w/2` y `h/2` para que el centro de la imagen sea `(0, 0)`. Así `r` va de `−rMax` a `rMax` y el acumulador queda simétrico. Con el origen en una esquina, las distancias serían casi todas positivas y se desperdiciaría media matriz.
3. **Eje Y invertido:** en una imagen la fila 0 está **arriba** y crece hacia abajo. En el plano cartesiano `y` crece hacia **arriba**. Por eso es `h/2 − fila` y no `fila − h/2`.

Ejemplo (800 × 600): el píxel de la esquina superior izquierda (`gloID = 0`) queda en `(−400, 300)` y el del centro en `(0, 0)`.

#### Votación con `atomicAdd`

Cada píxel recorre los 90 ángulos y suma 1 en la casilla `(rIdx, tIdx)`. **¿Por qué atómico?** Miles de hilos pueden votar en la misma casilla al mismo tiempo. Un `acc[i]++` normal son tres pasos (leer, sumar, escribir), y si dos hilos leen el mismo valor antes de escribir, se pierde un voto. `atomicAdd` hace los tres pasos como una sola operación indivisible. Detalle en [sección 6.4](#64-atomicadd-coordinación-sin-barrera).

#### `__restrict__`

Le promete al compilador que esos punteros no apuntan a la misma memoria. Así puede usar la caché de solo lectura para `pic`, `d_Cos` y `d_Sin`.

---

### 5.2 Versión constante (`hough_const.cu`)

**Lo único que cambia** es dónde viven `d_Cos` y `d_Sin`:

```cpp
// [CONST] tablas cos/sin  (fuera del main, alcance global)
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTranConst(const unsigned char *__restrict__ pic, int w, int h, int *acc,
                                   float rMax, float rScale) {   // ya no recibe d_Cos ni d_Sin
  ...
  for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
    // [CONST] mismo tIdx: broadcast
    int rIdx = rIndex(xCoord, yCoord, d_Cos[tIdx], d_Sin[tIdx], rMax, rScale);
    atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
  }
}
```

Y en el `main`:

```cpp
// [CONST] host a símbolo
cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);
```

| | Global | Constante |
|---|---|---|
| Declaración | `float *d_Cos` + `cudaMalloc` | `__constant__ float d_Cos[90]` en el encabezado |
| Copia desde CPU | `cudaMemcpy` | `cudaMemcpyToSymbol` |
| Paso al kernel | como argumento | no se pasa: es global al programa |
| Liberación | `cudaFree` | no hace falta: es estática |

#### Por qué la memoria constante encajaba en teoría

- Son datos **de solo lectura**, iguales para todos los píxeles.
- Ocupan **720 bytes** (2 × 90 × 4) de los 64 KB disponibles.
- Todos los hilos de un warp leen el **mismo** `tIdx` al mismo tiempo, porque van en el mismo paso del ciclo. La memoria constante hace **broadcast**: una lectura sirve a los 32 hilos.

#### Por qué no mejoró el tiempo (pregunta 7a)

Medido: global 0.507 ms y constante 0.507 ms (mediana). No hay diferencia significativa (Welch p = 0.065).

1. **Leer cos/sin nunca fue el cuello de botella.** Son solo 720 bytes. En la versión global ya caben en la caché L1/L2 y se leen casi igual de rápido.
2. **El cuello de botella son las 12.46 millones de sumas atómicas** sobre solo 9 000 casillas. Los hilos hacen fila para escribir en la misma casilla (*contención*).
3. **Advertencia:** la memoria constante es rápida solo si todos los hilos del warp leen la misma dirección. Si leen direcciones distintas, la petición se divide y se serializa. El experimento `src/experiments/optimizaciones.cu` lo muestra: al desfasar el ángulo por hilo, la versión constante se vuelve 4.4× más lenta.

---

### 5.3 Versión compartida (`hough_shared.cu`), la versión final

```cpp
static_assert(accSize * sizeof(int) <= 48 * 1024, "localAcc excede 48KB");

__global__ void GPU_HoughTranShared(const unsigned char *__restrict__ pic, int w, int h, int *acc,
                                    float rMax, float rScale) {
  // [SHARED] acumulador del bloque
  __shared__ int localAcc[accSize];
  int locID = threadIdx.x;
  // [ID] local y global
  int gloID = blockIdx.x * blockDim.x + locID;

  for (int i = locID; i < accSize; i += blockDim.x) localAcc[i] = 0;
  // [BARRERA 1] init terminada
  __syncthreads();

  // [BARRERA] sin return temprano
  if (gloID < w * h && pic[gloID] > 0) {
    int xCoord = gloID % w - w / 2;
    int yCoord = h / 2 - gloID / w;
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
      int rIdx = rIndex(xCoord, yCoord, d_Cos[tIdx], d_Sin[tIdx], rMax, rScale);
      // [ATOMIC] voto en shared
      atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
    }
  }
  // [BARRERA 2] votos terminados
  __syncthreads();

  // [VOLCADO] solo bins con votos
  for (int i = locID; i < accSize; i += blockDim.x)
    if (localAcc[i] > 0) atomicAdd(&acc[i], localAcc[i]);
}
```

#### La idea

En lugar de que **todos** los hilos de la GPU compitan por el mismo acumulador en memoria global, cada **bloque** tiene su propio acumulador privado (`localAcc`) en memoria compartida, que está dentro del chip y es mucho más rápida. Al final, cada bloque suma su resultado parcial al acumulador global.

```
         Bloque 0               Bloque 1              …   Bloque 1874
  ┌────────────────────┐ ┌────────────────────┐       ┌────────────────────┐
  │ 256 hilos votan en │ │ 256 hilos votan en │       │ 256 hilos votan en │
  │ localAcc (shared)  │ │ localAcc (shared)  │       │ localAcc (shared)  │
  └─────────┬──────────┘ └─────────┬──────────┘       └─────────┬──────────┘
            │ volcado              │ volcado                    │ volcado
            └──────────────────────┼────────────────────────────┘
                                   ▼
                     acc (memoria global, 9 000 casillas)
```

#### Las 5 etapas del kernel

| # | Etapa | Código | Qué hace |
|---|---|---|---|
| 1 | IDs | `locID = threadIdx.x`, `gloID = …` | `locID` es la posición dentro del bloque (0–255); `gloID` es el píxel |
| 2 | Inicialización | `for (i = locID; …; i += blockDim.x) localAcc[i] = 0;` | Los hilos del bloque ponen en cero `localAcc` repartiéndose el trabajo |
| 3 | **Barrera 1** | `__syncthreads()` | Espera a que **todo** `localAcc` esté en cero |
| 4 | Votación | `atomicAdd(&localAcc[…], 1)` | Cada píxel vota en el acumulador **del bloque** |
| 5 | **Barrera 2** | `__syncthreads()` | Espera a que **todos** los hilos del bloque terminen de votar |
| 6 | Volcado | `atomicAdd(&acc[i], localAcc[i])` | Suma el acumulador del bloque al global |

#### Etapa 2: inicialización en paralelo (patrón *block-stride loop*)

`localAcc` tiene 9 000 casillas y el bloque tiene 256 hilos, así que cada hilo limpia varias:

```
hilo 0   → casillas 0, 256, 512, …, 8704
hilo 1   → casillas 1, 257, 513, …, 8705
…
hilo 255 → casillas 255, 511, …, 8959
```

Cada hilo hace unas 35 casillas. Hilos consecutivos tocan casillas consecutivas, lo que evita conflictos de banco en la memoria compartida. **La memoria compartida empieza con basura**, y solo se puede inicializar desde el kernel; la CPU no tiene acceso a ella.

#### Etapa 6: volcado

Mismo patrón de reparto que la inicialización. Dos detalles:

- **`if (localAcc[i] > 0)`**: solo se suman las casillas con votos. Un bloque cubre 256 píxeles, menos de una fila de la imagen, así que la mayoría de sus 9 000 casillas quedan en 0. Sumar ceros sería trabajo inútil y más contención.
- **Sigue siendo `atomicAdd`**: los 1 875 bloques escriben en el **mismo** acumulador global, así que también aquí puede haber choques. La diferencia es que ahora son a lo sumo **una suma por casilla y por bloque**, en lugar de una por voto.

#### `static_assert`: el límite de la memoria compartida

```cpp
static_assert(accSize * sizeof(int) <= 48 * 1024, "localAcc excede 48KB");
```

`localAcc` ocupa 9 000 × 4 = **36 000 bytes (~35 KB)**. Un bloque puede usar como máximo **48 KB** de memoria compartida estática. Si alguien sube `rBins` o baja `degreeInc`, este `assert` detiene la **compilación** con un mensaje claro en lugar de fallar en ejecución.

| Cambio | Tamaño de `localAcc` | ¿Compila la versión shared? |
|---|---|---|
| Actual (90 × 100) | 35.2 KB | Sí |
| `rBins = 136` (máximo con 90 ángulos) | 47.8 KB | Sí |
| `rBins = 200` | 70.3 KB | **No** |
| `degreeInc = 1` (180 ángulos) | 70.3 KB | **No** |

Esto es justamente lo que dice el enunciado: el número de casillas "en el caso de CUDA en realidad está restringido por el tamaño de la memoria Compartida".

#### Resultado

| Versión | Mediana | Speedup vs global |
|---|---|---|
| Global | 0.507 ms | 1.00× |
| Global + constante | 0.507 ms | 1.00× |
| **Global + constante + compartida** | **0.113 ms** | **4.49×** |

La compartida gana porque ataca el problema real: las sumas atómicas en memoria compartida son mucho más rápidas, y la contención se reparte entre 1 875 acumuladores independientes en lugar de concentrarse en uno.

---

## 6. Sincronización: todo lo que hay que saber

En este proyecto hay **dos niveles** de sincronización:

- **Dentro de la GPU**, entre hilos: `__syncthreads()` y `atomicAdd`.
- **Entre la CPU y la GPU**: eventos CUDA, `cudaMemcpy` y el orden del *stream*.

### 6.1 `__syncthreads()`: barreras de bloque

`__syncthreads()` es una **barrera**: ningún hilo del bloque pasa de esa línea hasta que **todos** los hilos del mismo bloque hayan llegado a ella. Además, garantiza que todo lo que se escribió en memoria compartida antes de la barrera es visible para todos los hilos después.

**Importante:** solo sincroniza hilos **del mismo bloque**. No hay forma de sincronizar bloques distintos con `__syncthreads()`. Por eso el volcado al acumulador global usa `atomicAdd` y no una barrera.

#### Barrera 1: después de inicializar (línea 27)

```cpp
for (int i = locID; i < accSize; i += blockDim.x) localAcc[i] = 0;
__syncthreads();   // [BARRERA 1]
```

**Qué protege:** que nadie vote antes de que todo `localAcc` esté en cero.

**Qué pasaría sin ella:** cada hilo limpia casillas distintas. El hilo 0 podría terminar de limpiar las suyas y empezar a votar en la casilla 4 500, que le toca limpiar al hilo 148. Si el hilo 148 todavía no la limpió, pasa una de dos cosas:

- El voto se suma sobre basura, y la casilla queda con un valor incorrecto.
- El hilo 148 limpia la casilla **después** del voto, y el voto se pierde.

```
SIN barrera 1:
  hilo 0:    limpia sus casillas → vota en casilla 4500 (+1)
  hilo 148:                                              → limpia casilla 4500 (= 0)   ← voto perdido
```

#### Barrera 2: después de votar (línea 40)

```cpp
atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);   // votación
__syncthreads();   // [BARRERA 2]
if (localAcc[i] > 0) atomicAdd(&acc[i], localAcc[i]);   // volcado
```

**Qué protege:** que el volcado al acumulador global se haga con `localAcc` **completo**.

**Qué pasaría sin ella:** un hilo que ya terminó de votar empezaría a volcar. Si vuelca la casilla 4 500 cuando tiene 3 votos, y después otro hilo del bloque le suma un cuarto voto, ese cuarto voto nunca llega al global.

```
SIN barrera 2:
  hilo 0:    termina de votar → vuelca casilla 4500 (envía 3)
  hilo 200:  todavía votando              → suma +1 a casilla 4500 (ahora 4)   ← nunca se envía
```

#### Cómo se detecta si algo falla

La función `validate` compara casilla por casilla contra la CPU. Si se quita una barrera, lo normal es ver `Validacion: N mismatches … -> ERROR`. **Pero puede no fallar siempre**: es una *condición de carrera*, y depende de en qué orden se ejecutan los warps. Que el resultado salga bien una vez sin barrera no significa que sea correcto.

### 6.2 La trampa del `return` temprano

En la versión global:

```cpp
if (gloID >= w * h || pic[gloID] == 0) return;   // OK: no hay barreras
```

En la versión compartida **no se puede hacer así**. Se usa un `if` que envuelve solo la votación:

```cpp
// [BARRERA] sin return temprano
if (gloID < w * h && pic[gloID] > 0) {
  ... votación ...
}
__syncthreads();   // TODOS los hilos llegan aquí
```

**Por qué:** si un hilo con un píxel apagado hiciera `return`, nunca llegaría a la barrera 2. El resto del bloque esperaría en un `__syncthreads()` al que ese hilo nunca llega, y el comportamiento queda **indefinido**: puede colgarse o dar resultados incorrectos. Además, ese hilo tampoco participaría en el volcado, que **sí** debe hacer aunque su píxel esté apagado, porque le toca su parte de las casillas.

**Regla:** todo `__syncthreads()` debe ser alcanzado por **todos** los hilos del bloque. Nunca va dentro de un `if` que unos hilos cumplen y otros no, ni después de un `return` que solo algunos ejecutan.

### 6.3 Por qué no hace falta una tercera barrera al final

Después del volcado el kernel termina. Ningún hilo vuelve a leer `localAcc`, y la memoria compartida se libera sola al terminar el bloque. No queda nada que proteger.

### 6.4 `atomicAdd`: coordinación sin barrera

`atomicAdd(&x, v)` lee `x`, le suma `v` y escribe el resultado **como una sola operación indivisible**. Ningún otro hilo puede meterse en medio.

```
SIN atómico (acc[i]++):                    CON atomicAdd:
  hilo A lee  acc[i] = 5                     hilo A: 5 → 6  (indivisible)
  hilo B lee  acc[i] = 5                     hilo B: 6 → 7  (indivisible)
  hilo A escribe 6                           resultado: 7 ✔
  hilo B escribe 6
  resultado: 6 ✘ (se perdió un voto)
```

**Dónde se usa:**

| Lugar | Quiénes compiten | Por qué hace falta |
|---|---|---|
| Votación en global (versiones 1 y 2) | Todos los hilos de la GPU | Muchos píxeles caen en la misma casilla |
| Votación en `localAcc` (versión 3) | Los 256 hilos de un bloque | Igual, pero dentro del bloque |
| Volcado a `acc` (versión 3) | Los 1 875 bloques | Varios bloques suman a la misma casilla global |

**Diferencia con la barrera:** `atomicAdd` garantiza que cada suma individual sea correcta, pero **no** que ya se hayan hecho todas. `__syncthreads()` garantiza que todos hayan terminado, pero **no** protege de escrituras simultáneas. Por eso la versión compartida necesita **las dos cosas**.

**El costo:** cuando muchos hilos hacen `atomicAdd` sobre la misma dirección, el hardware los **serializa**: hacen fila. Ese es el cuello de botella de este algoritmo, y lo que la memoria compartida reduce.

### 6.5 Sincronización CPU ↔ GPU

**El lanzamiento del kernel es asíncrono.** `kernel<<<…>>>(…)` regresa de inmediato y la CPU sigue con la siguiente línea mientras la GPU todavía trabaja. Por eso no se puede medir el tiempo con un reloj de CPU alrededor del lanzamiento: mediría casi cero.

Mecanismos que usa el programa:

| Mecanismo | Dónde | Qué sincroniza |
|---|---|---|
| **Orden del stream por defecto** | `cudaMemset` → `timer.start` → kernel → `timer.stop` | Todo lo que se manda al mismo stream se ejecuta **en orden**. El `cudaMemset` termina antes de que empiece el kernel, sin esperar explícitamente. |
| **`cudaEventRecord`** | `GpuTimer::start` / `stop` | Pone una "marca" en el stream. La GPU registra la hora exacta cuando llega a ella. |
| **`cudaEventSynchronize(end)`** | `GpuTimer::stop` | **La CPU se bloquea** hasta que la GPU llega al evento `end`, es decir, hasta que el kernel terminó. Sin esto, `cudaEventElapsedTime` leería un evento que todavía no ocurrió. |
| **`cudaMemcpy` (device → host)** | `[SALIDA]` | Es **bloqueante**: no regresa hasta que la GPU terminó todo el trabajo previo y la copia está completa. Garantiza que `gpuAcc` tiene el resultado final. |
| **`cudaGetLastError()`** | después de cada lanzamiento | No sincroniza, pero recoge errores del lanzamiento, que de otra forma pasarían en silencio. |

```
CPU:  memset ─ record(begin) ─ lanzar kernel ─ record(end) ─ [espera en cudaEventSynchronize] ─ ElapsedTime
GPU:          memset ─────── begin ──── kernel ejecutándose ──── end ─┘
                             └──────── tiempo medido ─────────┘
```

---

## 7. Funciones de apoyo en CPU

### `parseArgs` (`hough.cpp`)

Lee la línea de comandos: `./hough_shared runway.pgm --runs 30 --block 512 …`. Valida que `runs ≥ 1` y que `32 ≤ block ≤ 1024`, porque 1 024 es el máximo de hilos por bloque en CUDA.

| Opción | Default | Uso |
|---|---|---|
| `--runs N` | 10 | corridas medidas (más una de calentamiento) |
| `--block B` | 256 | hilos por bloque |
| `--k K` | 2.0 | umbral = media + K · desviación estándar |
| `--top N` | 0 (todas) | dibujar solo las N líneas más fuertes |
| `--nms 0/1` | 1 | supresión de no máximos 5×5 |
| `--csv f` | — | agrega los tiempos a un CSV (bitácora) |
| `--out p` | `output/runway` | prefijo de las imágenes de salida |

### `buildTrigTables` (`hough.cpp`)

Calcula `cos` y `sin` de los 90 ángulos **una sola vez** en la CPU. Sin estas tablas, cada hilo calcularía 90 cosenos y 90 senos: unos 25 millones de llamadas trigonométricas en total.

### `CPU_HoughTran` (`hough.cpp`)

La versión **secuencial** de referencia (Listado 1 del enunciado): dos ciclos sobre filas y columnas, y para cada píxel encendido, un ciclo de 90 ángulos. Usa la **misma** función `rIndex` que la GPU, y por eso los resultados coinciden casilla por casilla. Tarda unos 32 ms contra 0.113 ms de la versión compartida (**~288×**).

### `validate` (`hough.cpp`)

Compara las 9 000 casillas de la CPU y la GPU, muestra hasta 5 diferencias y la suma total de votos (12 456 540). Retorna `true` solo si hay **0 diferencias**. El programa termina con código 1 si la validación falla.

### `reportTimes` (`hough.cpp`)

Calcula media, desviación estándar (muestral, n − 1), mínimo y máximo de las corridas, y el speedup contra la CPU. Si se pasa `--csv`, agrega cada corrida al archivo y **continúa la numeración** si el archivo ya tenía corridas de esa versión y ese tamaño de bloque. Así se construyen las bitácoras intercaladas.

### `PGMImage` (`pgm.cpp`)

Lee imágenes PGM en formato binario (`P5`) o de texto (`P2`). `readInt` salta espacios y comentarios (`#`) del encabezado. El destructor libera `pixels` automáticamente.

---

## 8. Detección y dibujo de líneas (`lines.cpp`)

Esto responde a la **pregunta 4** del enunciado: la imagen de salida con las líneas en color.

### `lineThreshold`: el umbral

```
umbral = media + k · desviación estándar     (k = 2 por defecto)
```

Se calcula **solo sobre casillas con votos** (> 0). Si se incluyeran los miles de ceros, la media bajaría tanto que pasarían demasiadas líneas.

### `isPeak`: supresión de no máximos (NMS)

Una recta real no da votos solo a una casilla: también "contamina" a las vecinas con casi tantos votos. Sin NMS, una sola recta se dibujaría como un manojo de líneas casi iguales (ver `results/figures/sin_nms_global_lines.png`).

`isPeak` acepta una casilla solo si es **la mayor en su vecindario de 5 × 5**.

- **Detalle circular:** θ = 178° y θ = 0° son ángulos vecinos. Al pasar de 180° a 0°, la recta es la misma pero con `r` de signo opuesto, y por eso se invierte el índice de r: `rr = rBins - 1 - rr`.
- **Empates:** si dos casillas tienen los mismos votos, gana la de índice menor. Así no se dibujan las dos.

### `selectLines`

Recorre el acumulador, se queda con las casillas que superan el umbral y son picos, y las ordena de más a menos votos. Con `--top N` se queda solo con las N primeras.

### `drawLines`

Convierte cada casilla `(rIdx, tIdx)` de vuelta a una recta y la dibuja:

1. `r = (rIdx + 0.5) · rScale − rMax`: el **centro** de la casilla, deshaciendo lo que hizo `rIndex`.
2. Despeja de `r = x·cos + y·sin`:
   - Si la recta es más horizontal (`|sin| > |cos|`): recorre cada columna `x` y calcula `y`.
   - Si es más vertical: recorre cada fila `y` y calcula `x`. Esto evita dividir entre un número cercano a 0 y que la línea salga punteada.
3. Dibuja 3 píxeles de grosor.
4. **Color según votos:** de amarillo (justo sobre el umbral) a rojo (la más votada). Se dibujan primero las débiles para que las fuertes queden encima.

Las líneas cruzan toda la imagen porque Hough detecta **rectas infinitas**, no segmentos.

### `saveOutputs`

Genera tres archivos por versión:

| Archivo | Contenido |
|---|---|
| `output/runway_<versión>_lines.png` | La imagen de entrada con las líneas en color |
| `output/runway_<versión>_hough.png` | El acumulador visto como imagen (el "espacio de Hough"), escalado ×4 |
| `output/runway_<versión>_acc.bin` | El acumulador crudo, que usa `compare_outputs.py` para verificar que las 3 versiones coinciden |

---

## 9. Medición de tiempo (`timer.h`)

Responde a la **pregunta 2** del enunciado.

```cpp
struct GpuTimer {
  cudaEvent_t begin, end;
  void create()  { cudaEventCreate(&begin); cudaEventCreate(&end); }
  void start()   { cudaEventRecord(begin); }
  float stop() {
    cudaEventRecord(end);
    cudaEventSynchronize(end);                     // espera a que el kernel termine
    float ms = 0;
    cudaEventElapsedTime(&ms, begin, end);         // diferencia en milisegundos
    return ms;
  }
  void destroy() { cudaEventDestroy(begin); cudaEventDestroy(end); }
};
```

**Por qué eventos CUDA y no un reloj de CPU:** los eventos los marca la **propia GPU** cuando llega a ese punto de su cola de trabajo. Miden solo lo que pasó en la GPU entre las dos marcas, con resolución de ~0.5 µs, sin contar la latencia de lanzamiento desde la CPU.

`CpuTimer` usa `std::chrono::high_resolution_clock` y sirve solo para medir la versión de CPU, que es síncrona.

### Protocolo de medición

1. Se descarta la corrida 0 (calentamiento).
2. El acumulador se pone en cero **fuera** del intervalo medido.
3. Las tres versiones se corren **intercaladas** (3 rondas × 10 corridas) para que la temperatura o la carga no favorezcan a ninguna.
4. Se reportan **mediana** y mínimo además de la media, porque hay saltos aleatorios de hasta 1.1 ms causados por interferencia externa (ver `results/bitacora_control_ruido.csv`).

---

## 10. Manejo de errores y liberación de memoria

### `CUDA_CHECK` (`cuda_utils.h`)

```cpp
#define CUDA_CHECK(call)                                       \
  do {                                                         \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while (0)
```

Envuelve **cada** llamada a CUDA. Si algo falla, imprime el archivo, la línea y el mensaje de error y termina. Sin esto, las funciones CUDA fallan en silencio y el programa sigue con datos basura. El `do { … } while (0)` permite usar la macro como una instrucción normal terminada en `;`.

### `[LIMPIEZA]`: pregunta 1b y rúbrica

```cpp
timer.destroy();                 // eventos CUDA
CUDA_CHECK(cudaFree(d_in));      // memoria de la GPU
CUDA_CHECK(cudaFree(d_acc));
// (versión global también: cudaFree(d_Cos), cudaFree(d_Sin))
delete[] pcCos;                  // memoria de la CPU
delete[] pcSin;
delete[] cpuAcc;
delete[] gpuAcc;
CUDA_CHECK(cudaDeviceReset());   // reinicio del dispositivo
```

| Recurso | Cómo se libera |
|---|---|
| Eventos CUDA | `cudaEventDestroy` (en `timer.destroy()`) |
| Memoria global de la GPU | `cudaFree` |
| Memoria constante (`d_Cos`, `d_Sin` en versiones 2 y 3) | No se libera: es estática y vive mientras viva el programa |
| Memoria compartida (`localAcc`) | No se libera: se libera sola al terminar cada bloque |
| Memoria de la CPU (`new[]`) | `delete[]` |
| Imagen PGM | El destructor de `PGMImage` |
| El dispositivo completo | `cudaDeviceReset()`: destruye el contexto CUDA y libera todo lo que quede |

---

## 11. Preguntas probables del catedrático

**¿Qué es `gloID` y cómo se calcula?**
El índice global del hilo: `blockIdx.x * blockDim.x + threadIdx.x`. Como el grid es 1D y la imagen se guarda fila por fila, cada hilo procesa el píxel número `gloID`.

**¿Qué diferencia hay entre `gloID` y `locID`?**
`locID = threadIdx.x` es la posición del hilo **dentro de su bloque** (0 a 255) y se usa para repartir el trabajo sobre `localAcc`. `gloID` es la posición en **toda la imagen** y dice qué píxel procesa el hilo.

**¿Por qué se centran las coordenadas?**
Para que el origen sea el centro de la imagen, como pide el enunciado. Así `r` va de `−rMax` a `rMax` y el acumulador se aprovecha completo. La Y se invierte porque en imágenes la fila 0 está arriba.

**¿Para qué sirve cada `__syncthreads()`?**
La barrera 1 asegura que `localAcc` esté en cero antes de votar. La barrera 2 asegura que todos terminaron de votar antes de volcar. Sin ellas hay condiciones de carrera y se pierden votos.

**¿Qué pasa si quito una barrera?**
El resultado puede salir mal: `validate` muestra diferencias contra la CPU. Pero no siempre falla, porque depende del orden en que se ejecutan los warps. Eso lo hace peligroso.

**¿Por qué no se usa `return` en el kernel compartido?**
Porque un hilo que sale antes de tiempo no llega a la barrera, y el resto del bloque se queda esperando. Además, ese hilo tiene que participar en la inicialización y el volcado aunque su píxel esté apagado.

**¿Por qué se necesita `atomicAdd` si ya hay barreras?**
Las barreras solo dicen "esperen a que todos terminen". No impiden que dos hilos escriban en la misma casilla **al mismo tiempo** durante la votación. `atomicAdd` evita que se pierdan votos en esos choques.

**¿Por qué el volcado también es atómico?**
Porque los 1 875 bloques suman al **mismo** acumulador global, y `__syncthreads()` no sincroniza bloques entre sí.

**¿Por qué la memoria constante no mejoró?**
Leer cos/sin nunca fue el cuello de botella: son 720 bytes que ya estaban en caché. Lo lento son los 12.46 millones de `atomicAdd` sobre 9 000 casillas.

**¿Por qué la compartida sí mejoró 4.5×?**
Porque reparte la contención: cada bloque vota en su propio acumulador, en memoria dentro del chip, y al final hace como máximo una suma global por casilla.

**¿Se puede subir la resolución del acumulador?**
En las versiones global y constante sí. En la compartida, `localAcc` no puede pasar de 48 KB por bloque. Con 90 ángulos, el máximo es `rBins = 136`. Si se pasa, el `static_assert` detiene la compilación.

**¿Cómo se mide el tiempo?**
Con dos eventos CUDA (`cudaEventRecord`) antes y después del kernel, y `cudaEventSynchronize` para esperar a que termine. Se descarta la primera corrida porque incluye la carga del contexto.

**¿Cómo saben que el resultado es correcto?**
`validate` compara las 9 000 casillas contra la versión CPU y da 0 diferencias en las tres versiones. Además, `compare_outputs.py` confirma que las tres generan el mismo acumulador y la misma imagen.

**¿Cómo se eligen las líneas que se dibujan?**
Las casillas con más votos que media + 2 desviaciones estándar que además son máximos locales en un vecindario de 5 × 5.
