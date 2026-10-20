# LightTracker

**LightTracker** es un software ligero, moderno y de alto rendimiento para el seguimiento de objetos (*object tracking*) y análisis cinemático a partir de videos, diseñado especialmente para experimentos de física de laboratorio y modelado computacional.

Inspirado en la funcionalidad clásica de **Tracker (Open Source Physics)** de Douglas Brown, **LightTracker** reimplementa los flujos de trabajo científicos esenciales sustituyendo la pesada máquina virtual Java por un núcleo rápido en Python 3.12 y OpenCV (C++ con aceleración SIMD).

---

## Características Principales

- **Ultra Liviano y Rápido**:
  - Tiempo de arranque inferior a 1 segundo.
  - Consumo de memoria RAM de ~40-60 MB (frente a los >500 MB del Tracker original).
  - Decodificación y búsqueda de plantillas multihilo a >120 FPS.
- **Calibración y Geometría Física**:
  - Vara de calibración de 2 puntos para definir la escala real (píxeles $\to$ metros).
  - Sistema de ejes cartesianos con origen $(x_0, y_0)$ y ángulo de inclinación $\theta$ configurables.
- **Múltiples Modos de Seguimiento**:
  - **Manual**: Click por fotograma con avance automático por paso configurable.
  - **AutoTracker (NCC)**: Coeficiente de correlación cruzada normalizada (`TM_CCOEFF_NORMED`) con subpíxel parabólico y predicción cinemática *look-ahead* de velocidad.
  - **Color / Centroide HSV**: Detección ultrarrápida de marcadores o esferas de colores mediante momentos espaciales.
- **Motor Cinemático Integrado**:
  - Tiempo $t$, posiciones $(x, y, r, \theta)$, componentes de velocidad $(v_x, v_y, v)$ y aceleraciones $(a_x, a_y, a)$.
  - Derivación numérica por diferencias centradas con filtro de suavizado opcional (Savitzky-Golay).
- **Visualización y Exportación**:
  - Tabla de datos interactiva en tiempo real.
  - Gráficos integrados en Matplotlib ($x(t)$, $y(t)$, $y(x)$ trayectoria, $v_y(t)$, etc.).
  - Exportación a CSV, JSON (`.trk.json`) y tablas LaTeX para reportes de laboratorio.

---

## Instalación y Ejecución

```bash
# Crear o activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e .

# Iniciar la interfaz gráfica
light-tracker

# O ejecutar el CLI para análisis en lote / headless
light-tracker-cli --help
```
