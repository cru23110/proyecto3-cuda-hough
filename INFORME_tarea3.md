# Tarea 3: xCoord y yCoord

Cada hilo procesa un pixel y `gloID` es su posición en la imagen guardada fila por fila (ancho `w`). Para recuperar la posición 2D:

- `gloID % w` es la columna `i` (cuántos pixeles lleva en su fila).
- `gloID / w` es la fila `j` (cuántas filas completas hay antes).

La Transformada de Hough asume el origen en el centro de la imagen (Figura 2 del enunciado), así que se desplazan las coordenadas:

- `xCoord = i - xCent`: la columna se mide desde el centro. Es negativa a la izquierda y positiva a la derecha.
- `yCoord = yCent - j`: en una imagen la fila 0 está arriba y `j` crece hacia abajo, mientras que en el plano cartesiano `y` crece hacia arriba. Por eso se invierte el eje: `yCent - j` es positivo arriba del centro y negativo abajo.

Con el origen en el centro, `r = xCoord·cos(θ) + yCoord·sin(θ)` queda en `[-rMax, rMax]`, donde `rMax` es la distancia del centro a una esquina. Al sumar `rMax` y dividir entre `rScale` se obtiene `rIdx` en `[0, rBins)`.

Ejemplo (`w = 4`, `h = 4`, `xCent = yCent = 2`): `gloID = 6` da `i = 2`, `j = 1`, por lo que `xCoord = 0` y `yCoord = 1`. Es el pixel justo encima del centro.
