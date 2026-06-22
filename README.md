# ASCII Art Pipeline 🎨

Conversión de imágenes a ASCII Art mediante un pipeline de procesamiento de imagen que combina
luminancia y detección de bordes estructurales.

## ¿Qué hace?

El pipeline implementa dos capas que se superponen:

- **Capa base**: mapea el brillo de cada bloque de píxeles a un carácter de una paleta de 70 símbolos,
  de más oscuro (` `) a más brillante (`@`).
- **Capa de bordes**: detecta los contornos de la imagen usando Difference of Gaussians + filtro Sobel,
  y los representa con los caracteres `/`, `\`, `|`, `-` según la dirección del gradiente.

El resultado es una imagen ASCII que preserva tanto la textura y el brillo de la imagen original
como sus bordes y contornos estructurales.

## Estructura del proyecto

ProyectoASCII_Art/

├── ASCII_Art.ipynb       # Notebook principal con el pipeline completo

├── ascii_live.py         # Script en vivo con la cámara del computador

├── requirements.txt      # Dependencias del proyecto

├── .gitignore

└── README.md

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/jccaldas23/ProyectoASCII_Art.git
cd ProyectoASCII_Art

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Notebook
Abre `ASCII_Art.ipynb` en Jupyter y ejecuta las celdas en orden. Puedes cambiar la imagen
de entrada modificando la variable `image` al inicio del notebook.

### Script en vivo
Ejecuta el pipeline en tiempo real usando la cámara del computador:

```bash
python ascii_live.py
```

| Tecla | Acción |
|-------|--------|
| `Q`   | Salir  |
| `S`   | Guardar snapshot como `snapshot_ascii.png` |

## Parámetros clave

### `block_size`
Controla el tamaño de cada bloque de píxeles que se mapea a un carácter.
- Valores más **bajos** (ej: `8`) → más detalle, más lento
- Valores más **altos** (ej: `16`) → menos detalle, más rápido

### `threshold` ⚠️
Este es el parámetro más importante para la calidad de los bordes. Controla qué tan fuerte
debe ser el gradiente de una zona para ser considerada borde.

- Valores **bajos** (ej: `0.02` - `0.05`) → detecta muchos bordes, incluyendo ruido y texturas finas.
  La imagen se verá más densa y detallada pero puede verse saturada de caracteres.
- Valores **altos** (ej: `0.15` - `0.25`) → solo detecta los bordes más pronunciados.
  La imagen se verá más limpia y minimalista, con más espacios vacíos entre contornos.

**Recomendación**: empieza con `threshold=0.08` y ajusta según el resultado:
- Si hay demasiados bordes y la imagen se ve llena → **súbelo**
- Si los bordes desaparecen o se ven muy pocos → **bájalo**

El valor óptimo depende de la imagen — fotografías con mucho contraste toleran thresholds más
altos, mientras que imágenes con iluminación suave requieren valores más bajos.

## Resultados

Los resultados generados para cada imagen de prueba se encuentran dentro de la carpeta `imagenes/`
del repositorio. Para cada imagen se incluye:

- La imagen original
- El resultado del ASCII Art base (luminancia)
- El resultado de bordes estructurales
- El resultado combinado final

├── ascii_base/

├── ascii_bordes/

└── ascii_combinado/

## Demostración en vivo

<!-- Agrega aquí un GIF o video de la cámara en vivo -->
<!-- Puedes grabar la pantalla con QuickTime en Mac y convertirlo a GIF con ffmpeg: -->
<!-- ffmpeg -i demo.mov -vf "fps=10,scale=800:-1" demo.gif -->

![Demo en vivo](imagenes/demo.gif)

## Dependencias

- `opencv-python` — captura de cámara y renderizado en vivo
- `pillow` — renderizado de texto ASCII a imagen
- `scipy` — filtros gaussianos y convolución para detección de bordes
- `numpy` — operaciones matriciales sobre los píxeles
- `matplotlib` — visualización en el notebook
