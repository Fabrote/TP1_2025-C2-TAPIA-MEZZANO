### TRABAJO PRACTICO 1 - PROBLEMA 2 ###
### VALIDACION DE FORMULARIOS ###

### TRABAJO PRACTICO 1 - PROBLEMA 2 ###
### VALIDACION DE FORMULARIOS ###

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PASO 1: DETECTAR LINEAS DEL FORMULARIO
# ==============================================================================

def detectar_lineas(img):
    """
    Detecta las lineas horizontales del formulario.
    Retorna las posiciones de las lineas horizontales.
    """
    # Umbralizar (invertir para que lineas sean blancas)
    _, img_bin = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)
    
    # Sumar pixeles por fila (lineas horizontales)
    suma_rows = np.sum(img_bin, axis=1)
    
    # Definir umbral (50% del maximo)
    umbral_row = np.max(suma_rows) * 0.5
    
    # Detectar donde hay lineas
    lineas_h = suma_rows > umbral_row
    
    # Encontrar posiciones centrales de las lineas
    pos_horizontales = encontrar_posiciones(lineas_h)
    
    return pos_horizontales


def detectar_columnas_por_fila(img, h_lines, fila_idx, min_altura_pct=0.7):
    """
    Detecta las lineas verticales dentro de una fila especifica.
    Filtra lineas que no tengan suficiente altura (descarta palitos de letras).
    
    Parametros:
    - img: imagen en escala de grises
    - h_lines: lista de posiciones de lineas horizontales
    - fila_idx: indice de la fila (0 = primera fila)
    - min_altura_pct: porcentaje minimo de altura que debe tener la linea (0.7 = 70%)
    
    Retorna: lista de posiciones de columnas en esa fila
    """
    if fila_idx >= len(h_lines) - 1:
        return []
    
    # Extraer solo esa fila
    y1 = h_lines[fila_idx]
    y2 = h_lines[fila_idx + 1]
    fila = img[y1:y2, :]
    altura_fila = y2 - y1
    
    # Umbralizar
    _, fila_bin = cv2.threshold(fila, 200, 1, cv2.THRESH_BINARY_INV)
    
    # Para cada columna, contar cuantas filas tienen pixeles
    altura_por_columna = np.sum(fila_bin > 0, axis=0)
    
    # Una linea vertical real debe tener pixeles en al menos min_altura_pct de la altura
    umbral_altura = altura_fila * min_altura_pct
    
    # Detectar columnas que cumplan el criterio de altura
    lineas_v = altura_por_columna > umbral_altura
    
    # Encontrar posiciones
    pos_verticales = encontrar_posiciones(lineas_v)
    
    return pos_verticales

def encontrar_posiciones(mascara):
    """
    Encuentra las posiciones centrales de las lineas.
    """
    posiciones = []
    en_linea = False
    inicio = 0
    
    for i, val in enumerate(mascara):
        if val and not en_linea:
            inicio = i
            en_linea = True
        elif not val and en_linea:
            centro = (inicio + i - 1) // 2
            posiciones.append(centro)
            en_linea = False
    
    if en_linea:
        centro = (inicio + len(mascara) - 1) // 2
        posiciones.append(centro)
    
    return posiciones


# ==============================================================================
# PASO 3: EXTRAER Y CONTAR CARACTERES
# ==============================================================================

def extraer_celda(img, h_lines, v_lines, fila_idx, col_idx):
    """
    Extrae una celda especifica del formulario.
    
    Parametros:
    - img: imagen en escala de grises
    - h_lines: lineas horizontales
    - v_lines: lineas verticales de esa fila
    - fila_idx: indice de fila
    - col_idx: indice de columna
    
    Retorna: imagen de la celda
    """
    if fila_idx >= len(h_lines) - 1 or col_idx >= len(v_lines) - 1:
        return None
    
    margen = 5  # Margen para evitar bordes
    
    y1 = h_lines[fila_idx] + margen
    y2 = h_lines[fila_idx + 1] - margen
    x1 = v_lines[col_idx] + margen
    x2 = v_lines[col_idx + 1] - margen
    
    celda = img[y1:y2, x1:x2]
    return celda


def contar_caracteres(celda):
    """
    Cuenta los caracteres en una celda usando componentes conectadas.
    
    Retorna: numero de caracteres detectados
    """
    if celda is None or celda.size == 0:
        return 0
    
    # Umbralizar
    _, celda_bin = cv2.threshold(celda, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, 8, cv2.CV_32S)
    
    # Filtrar componentes pequeñas (ruido)
    area_minima = 20
    caracteres = 0
    
    for i in range(1, num_labels):  # Ignorar fondo (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_minima:
            caracteres += 1
    
    return caracteres


def esta_vacia(celda):
    """
    Determina si una celda esta vacia.
    """
    return contar_caracteres(celda) == 0


# ==============================================================================
# PRUEBA COMPLETA
# ==============================================================================

# Cargar formulario con datos
img = cv2.imread(r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\TP1\formulario_05.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo cargar la imagen")
    exit()

print(f"Imagen cargada: {img.shape}")

# Detectar lineas horizontales
h_lines = detectar_lineas(img)
print(f"\nLineas horizontales detectadas: {len(h_lines)}")

# Detectar columnas en todas las filas
print("\nDetectando columnas en todas las filas:")
columnas_por_fila = {}

for i in range(len(h_lines) - 1):
    v_lines = detectar_columnas_por_fila(img, h_lines, i)
    columnas_por_fila[i] = v_lines
    print(f"  Fila {i}: {len(v_lines)} columnas")

# Visualizar deteccion de lineas
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Dibujar lineas horizontales en azul
for y in h_lines:
    cv2.line(img_color, (0, y), (img.shape[1], y), (255, 0, 0), 2)

# Dibujar columnas de cada fila en rojo
for fila_idx, v_lines in columnas_por_fila.items():
    if fila_idx >= len(h_lines) - 1:
        continue
    
    y1 = h_lines[fila_idx]
    y2 = h_lines[fila_idx + 1]
    
    for x in v_lines:
        cv2.line(img_color, (x, y1), (x, y2), (0, 0, 255), 2)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Deteccion de celdas')
plt.axis('off')
plt.show()

# Extraer y contar caracteres en celda de ejemplo
print("\n" + "="*70)
print("PRUEBA: Conteo de caracteres")
print("="*70)

celda_nombre = extraer_celda(img, h_lines, v_lines, 1, 1)

if celda_nombre is not None:
    num_chars = contar_caracteres(celda_nombre)
    vacia = esta_vacia(celda_nombre)
    
    print(f"\nCelda 'Nombre y apellido':")
    print(f"  - Caracteres detectados: {num_chars}")
    print(f"  - Esta vacia: {vacia}")
    
    plt.figure(figsize=(10, 3))
    plt.imshow(celda_nombre, cmap='gray')
    plt.title(f'Celda Nombre - {num_chars} caracteres')
    plt.axis('off')
    plt.show()

print("\n[OK] Listo para implementar validaciones")
print(v_lines)
print(v_lines_fila1)