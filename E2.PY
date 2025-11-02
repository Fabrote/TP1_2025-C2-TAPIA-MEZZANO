### TRABAJO PRACTICO 1 - PROBLEMA 2 ###
### VALIDACION DE FORMULARIOS ###

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

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
        return [] # si la fila es la ultima, no hay columnas
    
    # Extraer solo esa fila
    y1 = h_lines[fila_idx]
    y2 = h_lines[fila_idx + 1]
    fila = img[y1:y2, : ]
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
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_minima:
            caracteres += 1
    
    return caracteres



### debugeando encontramos que el contador de letras no detectaba correctamente los caracteres "LA" 



def esta_vacia(celda):
    """
    Determina si una celda esta vacia.
    """
    return contar_caracteres(celda) == 0



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


# -------------------------------
# PASO 4: FUNCIONES DE VALIDACION
# -------------------------------


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
    
    num_palabras = contar_palabras(celda)
    
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
    if contar_palabras(celda) != 1:
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
    num_palabras = contar_palabras(celda)
    
    # Verificar rango de palabras
    if num_palabras != 1:
        return False
    
    return True
    


def validar_pregunta(celda_si, celda_no):
    """
    En cada pregunta debe haber una única celda marcada 
    (referenciadas a las columnas de Si y No), con un único caracter. Es decir, en 
    cada pregunta, ambas celdas no pueden estar marcadas ni ambas celdas pueden 
    quedar vacías.
    """
    if contar_caracteres(celda_si) > 1 or contar_caracteres(celda_no) > 1:
        return False

    celda_si = not esta_vacia(celda_si)
    celda_no = not esta_vacia(celda_no)
    # XOR: una y solo una debe estar marcada
    return celda_si != celda_no

def validar_comentarios(celda):
    """
    Debe contener al menos una palabra y no más de 25 caracteres.
    """
    
    if celda is None or celda.size == 0: 
        return False
    
    num_chars = contar_caracteres(celda)
    
    # Verificar rango de caracteres
    if num_chars > 25:
        return False
    
    num_palabras = contar_palabras(celda)
    
    # Verificar rango de palabras
    if num_palabras < 1:
        return False
    
    return True
    
#--------------------------------------------
# CLASIFICACION DE FORMULARIOS "A", "B" O "C"
#--------------------------------------------


def detectar_tipo_formulario(path_img):
    """
    Detecta el tipo de formulario (A, B o C) a partir del encabezado superior derecho.
    Método simple: se recorta la zona del encabezado y se analiza su intensidad/media.
    """
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Desconocido"

    alto, ancho = img.shape
    # Recorte superior derecho (ajustá estos valores según tu imagen)
    roi = img[30:58, 557:580]

    # Umbralizar para resaltar la letra
    _, binaria = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)

    # Componentes conectadas para detectar la letra
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, 8, cv2.CV_32S)

    if num_labels <= 1:
        return "Desconocido"

    # Tomar la componente más grande (probable letra)
    max_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    area_letra = stats[max_idx, cv2.CC_STAT_AREA]

    # Clasificación por área absoluta (calibrado con datos reales)
    # A = 142 px, B = 166 px
    # Umbral intermedio: 154 px
    if area_letra < 154:
        tipo = "A"
    elif area_letra >= 154 and area_letra < 200:
        tipo = "B"
    else:
        tipo = "C"
    
    return tipo

#----------------
# PRUEBA COMPLETA
#----------------


def validar_formulario(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: No se pudo cargar la imagen")
        exit()

    # Detectar lineas horizontales
    h_lines = detectar_lineas(img)

    # Detectar columnas en todas las filas
    columnas_por_fila = {}

    for i in range(len(h_lines) - 1):
        v_lines = detectar_columnas_por_fila(img, h_lines, i)
        columnas_por_fila[i] = v_lines

    formulario = {
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


    convertir = lambda x: "OK" if x else "MAL"
    
    respuesta_nombre = convertir(validar_nombre_apellido(formulario['nombre']))
    respuesta_edad = convertir(validar_edad(formulario['edad']))
    respuesta_mail = convertir(validar_mail(formulario['mail']))
    respuesta_legajo = convertir(validar_legajo(formulario['legajo']))
    respuesta_pregunta1 = convertir(validar_pregunta(formulario['pregunta1_si'], formulario['pregunta1_no']))
    respuesta_pregunta2 = convertir(validar_pregunta(formulario['pregunta2_si'], formulario['pregunta2_no']))
    respuesta_pregunta3 = convertir(validar_pregunta(formulario['pregunta3_si'], formulario['pregunta3_no']))
    respuesta_comentarios = convertir(validar_comentarios(formulario['comentarios']))

    respuesta  = f"""
        '> Nombre y apellido': {respuesta_nombre},
        '> Edad': {respuesta_edad},
        '> Mail': {respuesta_mail},
        '> Legajo': {respuesta_legajo},
        '> Pregunta1': {respuesta_pregunta1},
        '> Pregunta2': {respuesta_pregunta2},
        '> Pregunta3': {respuesta_pregunta3},
        '> Comentarios': {respuesta_comentarios},
        """
    return respuesta

    # Extraer todas las celdas en un diccionario
    

# Cargar formulario con datos
img1 = r'C:\Users\Usuario\OneDrive\Documentos\TUIA\4to Cuatri\PDI\TP\formulario_01.png'
img2 = r'C:\Users\Usuario\OneDrive\Documentos\TUIA\4to Cuatri\PDI\TP\formulario_02.png'
img3 = r'C:\Users\Usuario\OneDrive\Documentos\TUIA\4to Cuatri\PDI\TP\formulario_03.png'
img4 = r'C:\Users\Usuario\OneDrive\Documentos\TUIA\4to Cuatri\PDI\TP\formulario_04.png'
img5 = r'C:\Users\Usuario\OneDrive\Documentos\TUIA\4to Cuatri\PDI\TP\formulario_05.png'


#--------------------------------------------
# PROCESAMIENTO CICLICO Y RESUMEN FINAL
#--------------------------------------------

def procesar_formularios():
    """
    Procesa de forma cíclica los 5 formularios e informa resultados
    agrupados por tipo (A, B o C).
    """
    # Lista de formularios
    formularios = [
        ('formulario_01.png', img1),
        ('formulario_02.png', img2),
        ('formulario_03.png', img3),
        ('formulario_04.png', img4),
        ('formulario_05.png', img5),
    ]
    
    # Diccionario para agrupar resultados por tipo
    resultados_por_tipo = defaultdict(list)
    
    print("="*70)
    print("PROCESAMIENTO CÍCLICO DE FORMULARIOS")
    print("="*70)
    
    # Procesar cada formulario
    for nombre, path in formularios:
        # Clasificar tipo de formulario
        tipo = detectar_tipo_formulario(path)
        
        # Validar formulario
        resultado_validacion = validar_formulario(path)
        
        # Guardar resultado
        resultados_por_tipo[tipo].append({
            'nombre': nombre,
            'validacion': resultado_validacion
        })
    
    
    for tipo in sorted(resultados_por_tipo.keys()):
        print(f"\n{'-'*70}")
        print(f"# FORMULARIOS TIPO {tipo}")
        print(f"{'-'*70}\n")
        
        for resultado in resultados_por_tipo[tipo]:
            print(f"Formulario: {resultado['nombre']}")
            print("-" * 70)
            print(resultado['validacion'])
            print()
    
    # Resumen final
    print("="*70)
    print("RESUMEN FINAL")
    print("="*70)
    for tipo in sorted(resultados_por_tipo.keys()):
        cantidad = len(resultados_por_tipo[tipo])
        print(f"Tipo {tipo}: {cantidad} formulario(s)")
    print("="*70)


#--------------------------------------------
# GENERACION DE IMAGEN DE SALIDA
#--------------------------------------------

def generar_imagen_salida():
    """
    Genera una imagen con los nombres de todos los formularios
    y un indicador visual (verde=OK, rojo=MAL).
    """
    formularios = [
        ('Formulario 1', img1),
        ('Formulario 2', img2),
        ('Formulario 3', img3),
        ('Formulario 4', img4),
        ('Formulario 5', img5),
    ]
    
    crops_nombres = []
    estados = []
    
    # Extraer nombres y validar cada formulario
    for nombre, path in formularios:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h_lines = detectar_lineas(img)
        columnas_por_fila = {}
        
        for i in range(len(h_lines) - 1):
            v_lines = detectar_columnas_por_fila(img, h_lines, i)
            columnas_por_fila[i] = v_lines
        
        # Extraer celda de nombre (fila 1, columna 1)
        celda_nombre = extraer_celda(img, h_lines, columnas_por_fila[1], 1, 1)
        
        # Validar formulario completo
        resultado = validar_formulario(path)
        es_correcto = "MAL" not in resultado
        
        crops_nombres.append(celda_nombre)
        estados.append(es_correcto)
    
    # Crear imagen de salida
    margen = 20
    altura_fila = 60
    ancho_indicador = 40
    
    # Calcular dimensiones
    max_ancho = max([c.shape[1] for c in crops_nombres])
    ancho_total = margen + max_ancho + margen + ancho_indicador + margen
    alto_total = margen + (altura_fila * len(crops_nombres)) + margen
    
    # Crear imagen blanca
    imagen_salida = np.ones((alto_total, ancho_total, 3), dtype=np.uint8) * 255
    
    # Colocar cada nombre con su indicador
    y_actual = margen
    
    for i, (crop, es_ok) in enumerate(zip(crops_nombres, estados)):
        # Convertir crop a BGR
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        
        # Redimensionar si es muy alto
        if crop.shape[0] > altura_fila - 10:
            factor = (altura_fila - 10) / crop.shape[0]
            nuevo_ancho = int(crop.shape[1] * factor)
            nuevo_alto = altura_fila - 10
            crop_bgr = cv2.resize(crop_bgr, (nuevo_ancho, nuevo_alto))
        
        # Pegar nombre
        h, w = crop_bgr.shape[:2]
        imagen_salida[y_actual:y_actual+h, margen:margen+w] = crop_bgr
        
        # Dibujar indicador (verde=OK, rojo=MAL)
        x_indicador = margen + max_ancho + margen
        color = (0, 255, 0) if es_ok else (0, 0, 255)  # Verde o Rojo
        cv2.rectangle(imagen_salida, 
                     (x_indicador, y_actual), 
                     (x_indicador + ancho_indicador, y_actual + altura_fila - 10),
                     color, -1)
        
        # Texto OK/MAL
        texto = "OK" if es_ok else "MAL"
        cv2.putText(imagen_salida, texto,
                   (x_indicador + 5, y_actual + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_actual += altura_fila
    
    # Guardar imagen
    cv2.imwrite(r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\TP1\resultado_formularios.png', imagen_salida)
    print("\n✓ Imagen de salida generada: resultado_formularios.png")
    
    # Mostrar
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(imagen_salida, cv2.COLOR_BGR2RGB))
    plt.title('Resultado de Formularios')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


#--------------------------------------------
# GENERACION DE CSV
#--------------------------------------------

def generar_csv():
    """
    Genera un archivo CSV con los resultados de validación.
    """
    
    formularios = [
        ('01', img1),
        ('02', img2),
        ('03', img3),
        ('04', img4),
        ('05', img5),
    ]
    
    resultados = []
    
    for id_form, path in formularios:
        # Extraer celdas usando la misma lógica de validar_formulario
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h_lines = detectar_lineas(img)
        columnas_por_fila = {}
        
        for i in range(len(h_lines) - 1):
            v_lines = detectar_columnas_por_fila(img, h_lines, i)
            columnas_por_fila[i] = v_lines
        
        formulario = {
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
        
        # Validar cada campo
        nombre = validar_nombre_apellido(formulario['nombre'])
        edad = validar_edad(formulario['edad'])
        mail = validar_mail(formulario['mail'])
        legajo = validar_legajo(formulario['legajo'])
        pregunta1 = validar_pregunta(formulario['pregunta1_si'], formulario['pregunta1_no'])
        pregunta2 = validar_pregunta(formulario['pregunta2_si'], formulario['pregunta2_no'])
        pregunta3 = validar_pregunta(formulario['pregunta3_si'], formulario['pregunta3_no'])
        comentarios = validar_comentarios(formulario['comentarios'])
        

        # Agregar fila
        resultados.append({
            'ID': id_form,
            'Nombre y Apellido': 'OK' if nombre else 'MAL',
            'Edad': 'OK' if edad else 'MAL',
            'Mail': 'OK' if mail else 'MAL',
            'Legajo': 'OK' if legajo else 'MAL',
            'Pregunta1': 'OK' if pregunta1 else 'MAL',
            'Pregunta2': 'OK' if pregunta2 else 'MAL',
            'Pregunta3': 'OK' if pregunta3 else 'MAL',
            'Comentarios': 'OK' if comentarios else 'MAL'
        })
    
    # Crear DataFrame y guardar
    df = pd.DataFrame(resultados)
    df.to_csv(r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\TP1\resultados_validacion.csv', index=False)
    print("\n✓ CSV generado: resultados_validacion.csv")
    print(df.to_string(index=False))



#--------------------------------------------
# EJECUCIÓN
#--------------------------------------------

if __name__ == "__main__":
    procesar_formularios()
    print("\n" + "="*70)
    print("GENERANDO IMAGEN DE SALIDA...")
    print("="*70)
    generar_imagen_salida()
    print("\n" + "="*70)
    print("GENERANDO CSV...")
    print("="*70)
    generar_csv()
