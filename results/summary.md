| Versión | n | Media ± sd (ms) | Mediana (ms) | Mín (ms) | IC 95 % | Speedup vs global | Speedup vs CPU |
|---|---|---|---|---|---|---|---|
| Global | 30 | 0.653 ± 0.223 | 0.507 | 0.497 | ± 0.083 | 1.00× | 64× |
| Global + Constante | 30 | 0.565 ± 0.123 | 0.507 | 0.500 | ± 0.046 | 1.00× | 64× |
| Global + Constante + Compartida | 30 | 0.113 ± 0.000 | 0.113 | 0.113 | ± 0.000 | 4.49× | 288× |

CPU secuencial: 32.50 ms. Speedup calculado con la mediana.

| Prueba t de Welch | t | p | ¿Significativa (α = 0.05)? |
|---|---|---|---|
| global vs const | 1.89 | 6.45e-02 | No |
| global vs shared | 13.25 | 7.85e-14 | Sí |
| const vs shared | 20.13 | 1.37e-18 | Sí |
