import cv2
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image, ImageDraw, ImageFont
import copy

# ==============================================================================
# 1. ANÁLISIS TONAL (LUMINANCIA Y SOMBREADO)
# ==============================================================================

def calculate_luminance(rgb_image):
    """
    Calcula la luminancia percibida de una imagen en formato RGB o RGBA
    utilizando el estándar de la fórmula relativa ITU-R BT.709.
    
    Ajusta el peso de cada canal según la sensibilidad del ojo humano:
    Verde (71.52%), Rojo (21.26%) y Azul (7.22%).
    """
    if rgb_image.shape[-1] == 3:  # Formato RGB estándar
        return 0.2126 * rgb_image[:,:,0] + 0.7152 * rgb_image[:,:,1] + 0.0722 * rgb_image[:,:,2]
    elif rgb_image.shape[-1] == 4:  # Formato RGBA (Ignora el canal Alpha)
        return 0.2126 * rgb_image[:,:,0] + 0.7152 * rgb_image[:,:,1] + 0.0722 * rgb_image[:,:,2]
    return rgb_image  # Si ya es escala de grises, la devuelve intacta


def convert_to_ascii(downscaled_image):
    """
    Convierte una imagen reducida (downscaled) a cadenas de texto ASCII tonales.
    Mapea el brillo de cada píxel a un índice dentro de una paleta de 70 caracteres,
    donde los espacios ' ' son el negro absoluto y '@' es la densidad máxima de luz.
    """
    ascii_chars = " .'`^\";:Il!i><~+_-?][}{1)(tfjrxnuvczsXYGAUJCLQ0OZmwqpdbkhao*#MHW&8%B@"
    num_chars = len(ascii_chars)
    
    # Obtiene la matriz de brillo (valores entre 0.0 y 1.0)
    luminance_matrix = calculate_luminance(downscaled_image)
    
    # Mapea los decimales a índices enteros uniformes [0, 69]
    char_indices = np.floor(luminance_matrix * (num_chars - 1)).astype(int)
    char_indices = np.clip(char_indices, 0, num_chars - 1)
    
    # Construye el bloque de texto fila por fila
    return ["".join([ascii_chars[idx] for idx in row]) for row in char_indices]


# ==============================================================================
# 2. ANÁLISIS ESTRUCTURAL (EXTRACCIÓN VECTORIAL DE BORDES)
# ==============================================================================

def apply_difference_of_gaussians(image_gray, sigma1=1.0, sigma2=2.0):
    """
    Preprocesamiento por Diferencia de Gaussianos (DoG).
    Resta dos versiones desenfocadas de la imagen para aislar las frecuencias altas.
    Funciona como un filtro de paso de banda que elimina el ruido de fondo y 
    resalta los contornos puros antes de pasarlos al detector de Sobel.
    """
    g1 = ndimage.gaussian_filter(image_gray, sigma=sigma1)
    g2 = ndimage.gaussian_filter(image_gray, sigma=sigma2)
    return g1 - g2


def apply_sobel_and_angles(dog_image, threshold=0.02):
    """
    Aplica el operador Sobel para calcular el vector gradiente (Gx, Gy) en cada píxel.
    Determina la magnitud del cambio y calcula su dirección exacta mediante atan2.
    
    Normaliza el ángulo al rango [0.0, 1.0] con la fórmula: (atan2 / pi) * 0.5 + 0.5
    Los píxeles que no superan el umbral (Threshold) se marcan con una bandera (-1.0).
    """
    # Máscaras de convolución Sobel tradicionales
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
    
    # Convolución para obtener variaciones horizontales y verticales
    Gx = ndimage.convolve(dog_image, kernel_x)
    Gy = ndimage.convolve(dog_image, kernel_y)
    
    # Evaluación del umbral de corte
    magnitude = np.sqrt(Gx**2 + Gy**2)
    edge_mask = magnitude > threshold
    
    # Obtención y normalización del ángulo del vector
    angles = np.arctan2(Gx, Gy)
    normalized_angles = (angles / np.pi) * 0.5 + 0.5
    
    # Si es borde guarda el ángulo, de lo contrario guarda la bandera -1.0
    return np.where(edge_mask, normalized_angles, -1.0)


def convert_angles_to_ascii_edges(angle_matrix):
    """
    Clasifica los ángulos continuos del gradiente en 4 direcciones discretas,
    asignando el carácter ASCII geométrico que mejor representa la línea:
    - '-' Horizontal   |   '|' Vertical   |   '/' Diagonal Creciente   |   '\\' Diagonal Decreciente
    """
    h, w = angle_matrix.shape
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            val = angle_matrix[y, x]
            if val == -1.0:
                row.append(" ")  # Región plana (sin borde estructural)
            elif (val < 0.125) or (val >= 0.875):
                row.append("-")  # Flujo horizontal
            elif val >= 0.375 and val < 0.625:
                row.append("|")  # Flujo vertical
            elif val >= 0.125 and val < 0.375:
                row.append("/")  # Pendiente positiva
            else:
                row.append("\\") # Pendiente negativa
        rows.append("".join(row))
    return rows


# ==============================================================================
# 3. COMPOSICIÓN HÍBRIDA Y MÁSCARAS POR FRAME
# ==============================================================================

def frame_to_ascii_combined(frame, block_size=8, threshold=0.08):
    """
    Procesa un fotograma individual de video, dividiéndolo y entrelazándolo
    en las dos capas ASCII calculadas por bloque espacial.
    
    Aplica una máscara lógica para que los bordes estructurales tengan prioridad
    sobre el sombreado tonal sin crear colisiones de caracteres.
    """
    # Normalización del frame de BGR (OpenCV) a RGB flotante [0.0, 1.0]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w  = image.shape[:2]
    new_h, new_w = h // block_size, w // block_size

    # --- PROCESO CAPA BASE (SOMBREADO) ---
    truncated  = image[:new_h * block_size, :new_w * block_size]
    # Reduce bloques de N x N a un único píxel promedio
    downscaled = truncated.reshape(new_h, block_size, new_w, block_size, 3).mean(axis=(1, 3))
    rows_base  = convert_to_ascii(downscaled)

    # --- PROCESO CAPA BORDES (CONTORNOS) ---
    gray         = calculate_luminance(image)
    dog          = apply_difference_of_gaussians(gray, sigma1=1.2, sigma2=2.4)
    angles       = apply_sobel_and_angles(dog, threshold=threshold)
    trunc_angles = angles[:new_h * block_size, :new_w * block_size]
    down_angles  = trunc_angles.reshape(new_h, block_size, new_w, block_size).mean(axis=(1, 3))
    rows_edges   = convert_angles_to_ascii_edges(down_angles)

    # --- OPERACIÓN DE ENMASCARAMIENTO DE CAPAS ---
    num_cols = max(len(r) for r in rows_base)
    rows_base_masked = []
    rows_edges_only  = []

    for i in range(new_h):
        # Asegura la alineación uniforme rellenando con espacios si es necesario
        base = rows_base[i].ljust(num_cols)
        edge = rows_edges[i].ljust(num_cols) if i < len(rows_edges) else " " * num_cols
        mb, eo = "", ""
        
        for j, (b, e) in enumerate(zip(base, edge)):
            if e != " ":
                mb += " "  # Deja un hueco en la capa base para que el borde no colisione
                eo += e    # Agrega el carácter estructural (/, \, |, -)
            else:
                mb += b    # Si no hay borde estructural, se imprime el carácter tonal
                eo += " "
                
        rows_base_masked.append(mb)
        rows_edges_only.append(eo)

    return rows_base_masked, rows_edges_only


# ==============================================================================
# 4. MOTOR DE RENDERIZACIÓN GRÁFICA OPTIMIZADA (PIL)
# ==============================================================================

def render_ascii_to_image(rows_base, rows_edges, char_w=10, char_h=18):
    """
    Construye un lienzo bitmap RGB a partir de las dos capas de texto.
    
    Optimización:
    Dibuja la Capa Base como cadenas completas por fila (ultra rápido)
    y luego inyecta la Capa de Bordes en alta intensidad carácter por carácter
    únicamente donde existen datos, evitando sobreescribir con espacios en blanco.
    """
    num_rows = len(rows_base)
    num_cols = max(len(r) for r in rows_base)

    # Dimensiones exactas de la imagen final basadas en el tamaño de la tipografía
    img_w = num_cols * char_w
    img_h = num_rows * char_h

    img  = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Carga de tipografía Monoespaciada nativa
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except:
        font = ImageFont.load_default()

    # -- PASADA 1: Renderizado veloz de la Capa Tonal (Gris tenue) --
    for i, base_row in enumerate(rows_base):
        y = i * char_h
        draw.text((0, y), base_row, fill=(160, 160, 160), font=font)

    # -- PASADA 2: Inyección de Contornos (Blanco brillante de alta definición) --
    for i, edge_row in enumerate(rows_edges):
        if not edge_row.strip():  # Si la fila entera está vacía, se salta el bucle de píxeles
            continue
        y = i * char_h
        for x_idx, ch in enumerate(edge_row):
            if ch != " ":  # Dibuja únicamente donde se detectó el trazo estructural
                draw.text((x_idx * char_w, y), ch, fill=(255, 255, 255), font=font)

    return np.array(img)


# ==============================================================================
# 5. LOOP PRINCIPAL Y CAPTURA DE VIDEO (OPENCV)
# ==============================================================================

def live_ascii_camera(block_size=12, threshold=0.12):
    """
    Inicializa el ciclo de captura de video por hardware (Cámara Web).
    Instancia una ventana nativa de OpenCV en modo Pantalla Completa
    y procesa el flujo en tiempo real cuadro por cuadro.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("Controles de la aplicación:")
    print("  [Q] -> Salir del programa")
    print("  [S] -> Guardar una captura de pantalla de alta resolución (.png)")

    # Configuración de ventana de despliegue nativa en modo Pantalla Completa
    cv2.namedWindow("ASCII Live", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ASCII Live", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Efecto espejo horizontal para que actúe de forma natural ante el usuario
        frame = cv2.flip(frame, 1)

        # 1. Procesa las capas combinadas del fotograma actual
        rows_b, rows_e = frame_to_ascii_combined(frame, block_size, threshold)
        
        # Telemetría de depuración en consola (Ejecutada una sola vez en el arranque)
        if frame_count == 0:
            chars_en_bordes = set(ch for row in rows_e for ch in row if ch != " ")
            print("\n--- Diagnóstico de Inicialización ---")
            print("Caracteres detectados en vectores de bordes:", chars_en_bordes)
            print("Muestra de datos de la fila 10 (Edge):", rows_e[10])
            print("--------------------------------------\n")
        frame_count += 1
    
        # 2. Transforma el mapa de caracteres en una matriz gráfica de píxeles
        ascii_img = render_ascii_to_image(rows_b, rows_e)

        # 3. Conversión de espacio de color para la visualización correcta en OpenCV (RGB a BGR)
        cv2.imshow("ASCII Live", cv2.cvtColor(ascii_img, cv2.COLOR_RGB2BGR))

        # --- CAPTURA DE EVENTOS DE TECLADO ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Cierre del programa
            break
        elif key == ord('s'):  # Exportar Snapshot
            cv2.imwrite("snapshot_ascii.png", cv2.cvtColor(ascii_img, cv2.COLOR_RGB2BGR))
            print("¡Snapshot guardado con éxito como 'snapshot_ascii.png'!")

    # Liberación controlada de los recursos de hardware y ventanas del sistema operativo
    cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados correctamente. Programa finalizado.")

if __name__ == "__main__":
    # Parámetros recomendados para balance óptimo de rendimiento y definición
    live_ascii_camera(block_size=8, threshold=0.08)