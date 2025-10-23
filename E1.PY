### TRABAJO PRACTICO 1 - PROBLEMA 1 ###
### ECUALIZACION LOCAL DE HISTOGRAMA ###

import cv2
import matplotlib.pyplot as plt
import numpy as np

def ecualizacion_local(img, window_size):  ####   HACER CON M X N 
    """
    Implementa ecualizacion local de histograma.
    Parametros:
    - img: imagen en escala de grises
    - window_size: tamano de ventana (impar)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    img_output = np.zeros_like(img, dtype=np.uint8)
    pad = window_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    rows, cols = img.shape
    
    for i in range(rows):
        for j in range(cols):
            window = img_padded[i:i+window_size, j:j+window_size]
            window_eq = cv2.equalizeHist(window)
            img_output[i, j] = window_eq[pad, pad]
    
    return img_output

img = cv2.imread(r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\TP1\Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo cargar la imagen")
    exit()

# Aplicar ecualizacion local con ventana 51x51
print("Procesando ecualizacion local con ventana 51x51...")
img_eq_51 = ecualizacion_local(img, 51)


print("\nAnalizando diferentes tamanos de ventana...")
window_sizes = [11, 21, 41, 101]
results = {}

for ws in window_sizes:
    print(f"    Procesando ventana {ws}x{ws}...")
    results[ws] = ecualizacion_local(img, ws)

# Mostrar resultados comparativos
fig = plt.figure(figsize=(16, 8))

# Fila 1: Imagen original + resultados
plt.subplot(2, 5, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

for idx, ws in enumerate(window_sizes):
    plt.subplot(2, 5, idx + 2)
    plt.imshow(results[ws], cmap='gray')
    plt.title(f'Ventana {ws}x{ws}')
    plt.axis('off')

# Histogramas
plt.subplot(2, 5, 6)
plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Histograma Original')
plt.xlabel('Intensidad')

for idx, ws in enumerate(window_sizes):
    plt.subplot(2, 5, idx + 7)
    plt.hist(results[ws].ravel(), bins=256, range=(0, 256), alpha=0.7)
    plt.title(f'Hist {ws}x{ws}')
    plt.xlabel('Intensidad')

plt.tight_layout()
plt.show()
