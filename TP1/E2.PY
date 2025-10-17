### TRABAJO PRACTICO 1 - PROBLEMA 2 ###
### VALIDACION DE FORMULARIOS ###

import cv2
from cv2.typing import Prim
import numpy as np
import matplotlib.pyplot as plt







# -------------------------------
# PASO 1: DETECTAR LINEAS DEL FORMULARIO
# -------------------------------








def detectar_lineas(img):
    """
    Detecta las lineas horizontales del formulario.
    Retorna las posiciones de las lineas horizontales.
    """
    # Umbralizar (invertir para que lineas sean blancas)
    _, img_bin = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)
    
    # Sumar pixeles por fila (lineas horizontales)
    suma_rows = np.sum(img_bin, axis=1)
    
    # Definir umbral
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






# -------------------------------
# PASO 3: EXTRAER Y CONTAR CARACTERES
# -------------------------------






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







# -------------------------------
# PASO 4: FUNCIONES DE VALIDACION
# -------------------------------




def contar_palabras(celda):  ## funcion con mas problemas para encontrar . _, tuve que cambiar el umbral multiplicador
    """    
    Encuentra las palabras en una celda.
    Retorna: cantidad de palabras detectadas
    """

    if celda is None or celda.size == 0:
        return 0
    
    # Umbralizar
    _, celda_bin = cv2.threshold(celda, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, 8, cv2.CV_32S)
    
    # Filtrar componentes pequeñas (ruido, _, .)
    # Los caracteres _ y . tienen area < 76, las letras normales > 100
    area_minima = 10  # No filtrar componentes pequeñas como _ y . a costa de tomar ruido
    #debido a la naturaleza del trabajo veo correcto usar un umbral de 10 
    # ya que no hay minusculas y tampoco hay ruido, por lo que el umbral es correcto y puedo detectar correctamente los "." y "_"


    componentes = []
    umbral_multiplicador = 0.6
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Solo considerar componentes con area suficiente
        if area > area_minima:
            x = stats[i, cv2.CC_STAT_LEFT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            componentes.append((x, w))
    
    if len(componentes) == 0:
        return 0
    
    # Ordenar por posicion horizontal
    componentes.sort()

    # Calcular ancho promedio de caracteres
    ancho_promedio = 0
    for i in range(len(componentes)):
        ancho_promedio += componentes[i][1]
    ancho_promedio /= len(componentes)
    
    # Umbral para detectar espacios entre palabras
    umbral_espacio = ancho_promedio * umbral_multiplicador  #Tuve que debuggear para encontrar el umbral correcto
    
    # Contar espacios entre palabras
    num_palabras = 1  # Empezamos con 1 palabra

    for i in range(len(componentes) - 1):
        x_fin_actual = componentes[i][0] + componentes[i][1]
        x_inicio_siguiente = componentes[i + 1][0]
        gap = x_inicio_siguiente - x_fin_actual
        
        if gap > umbral_espacio:
            num_palabras += 1

    return num_palabras




def validar_nombre_apellido(celda):
    """
    Valida Nombre y apellido:
    - Minimo 2 palabras
    - No mas de 25 caracteres
    """
    if celda is None or celda.size == 0: #Esto significa que la celda no tiene pixels
        return False
    
    num_chars = contar_caracteres(celda)
    

    # Verificar rango de caracteres
    if  num_chars > 25:
        return False
    
    num_palabras = encontrar_palabras(celda)
    
    # Verificar rango de palabras
    if num_palabras < 2:
        return False

    # Si tiene al menos 2 caracteres, asumimos que hay nombre y apellido
    # (simplificacion: confiamos en que si escribieron algo, pusieron ambos)
    return True


def validar_edad(celda):
    """
    Debe contener 2 o 3 caracteres consecutivos (no deben existir espacios 
    entre ellos)
    """
    
    if celda is None or celda.size == 0:
        return False
    
    num_chars = contar_caracteres(celda)
    
    # Verificar rango de caracteres
    if num_chars < 2 or num_chars > 3:
        return False
    
    #verificar que los caracteres sean consecutivos
    if encontrar_palabras(celda) != 1:
        return False
    
    return True

def validar_mail(celda): # problema, mi separador no detectaBA caracteres de poca altura como caracteres reales, por lo tanto no sabIA separar palabras
    """
     Debe contener una palabra y no más de 25 caracteres.
    """
    
    if celda is None or celda.size == 0: 
        return False
    
    num_chars = contar_caracteres(celda)
    
    # Verificar rango de caracteres
    if num_chars > 25:
        return False
    
    num_palabras = contar_palabras(celda)
    
    # Verificar rango de palabras
    if num_palabras != 1:
        return False
    
    return True



def validar_legajo(celda):
    """
    Debe contener sólo 8 caracteres en total, formando una única palabra. 
    """
    
    if celda is None or celda.size == 0: 
        return False
    
    num_chars = contar_caracteres(celda)
    
    # Verificar rango de caracteres
    if num_chars != 8:
        return False
    
    # Para legajos, usar umbral alto para ignorar espacios pequeños
    num_palabras = encontrar_palabras(celda, umbral_multiplicador=2.5)
    
    # Verificar rango de palabras
    if num_palabras != 1:
        return False
    
    return True
    




def validar_pregunta(celda_si, celda_no):
    """
    Valida que haya exactamente UNA celda marcada (SI o NO).
    - No pueden estar ambas marcadas
    - No pueden estar ambas vacias
    """
    si_marcada = not esta_vacia(celda_si)
    no_marcada = not esta_vacia(celda_no)
    
    # XOR: una y solo una debe estar marcada
    return si_marcada != no_marcada








#----------------
# PRUEBA COMPLETA
#----------------







# Cargar formulario con datos
img = cv2.imread(r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\TP1\formulario_01.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo cargar la imagen")
    exit()

print(f"Imagen cargada: {img.shape}")

# Detectar lineas horizontales
h_lines = detectar_lineas(img)
print(h_lines)
print(f"\nLineas horizontales detectadas: {len(h_lines)}")

# Detectar columnas en todas las filas
columnas_por_fila = {}

for i in range(len(h_lines) - 1):
    v_lines = detectar_columnas_por_fila(img, h_lines, i)
    columnas_por_fila[i] = v_lines


# Extraer todas las celdas en un diccionario

#Probe lo de abajo manualmente y decidi hacer un diccionario para hacerlo mas legible y usable
celdas = {
    'nombre': extraer_celda(img, h_lines, columnas_por_fila[1], 1, 1),
    'edad': extraer_celda(img, h_lines, columnas_por_fila[2], 2, 1),
    'mail': extraer_celda(img, h_lines, columnas_por_fila[3], 3, 1),
    'legajo': extraer_celda(img, h_lines, columnas_por_fila[4], 4, 1),
    'pregunta1_si': extraer_celda(img, h_lines, columnas_por_fila[6], 6, 1),
    'pregunta1_no': extraer_celda(img, h_lines, columnas_por_fila[6], 6, 2),
    'pregunta2_si': extraer_celda(img, h_lines, columnas_por_fila[7], 7, 1),
    'pregunta2_no': extraer_celda(img, h_lines, columnas_por_fila[7], 7, 2),
    'pregunta3_si': extraer_celda(img, h_lines, columnas_por_fila[8], 8, 1),
    'pregunta3_no': extraer_celda(img, h_lines, columnas_por_fila[8], 8, 2),
    'comentarios': extraer_celda(img, h_lines, columnas_por_fila[9], 9, 1),
}


"""
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

print("\n Listo para implementar validaciones")
print(v_lines)

## PRUEBA DE VALIDACION DE NOMBRE Y APELLIDO ##
print("\n")
print("PRUEBA: Validacion de nombre y apellido")
print("-----------------------------------------")

if validar_nombre_apellido(celda_nombre):
    print("\nCelda 'Nombre y apellido':")
    print("  - Validacion exitosa")
else:
    print("\nCelda 'Nombre y apellido':")
    print("  - Validacion fallida")
"""



print(contar_palabras(celdas['mail']))