import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread(r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\UNIDAD 1\img_calculadora.tif',cv2.IMREAD_GRAYSCALE)

# --- Matplotlib -------------------------------------------------
plt.figure(1)
h = plt.imshow(img1, cmap='gray')
plt.colorbar(h)

np.histogram(img1, bins=256, range=(0,256))

# Histograma
plt.figure(2)
plt.title("Histograma de Intensidades")
plt.xlabel("Valor de intensidad")
plt.ylabel("Número de píxeles")
plt.xticks(np.arange(0, 256, 10))

# Mostrar histograma
plt.hist(img1.ravel(), bins=256, range=(0, 256), color='gray')
plt.grid(True)


plt.show() 
"""

 f)
 Recortar las teclas con las etiquetas: ‘SIN’, ‘COS’ y ‘TAN’. Los tres recortes deben
 ser del mismo tamaño. Mostrar los recortes en una nueva figura utilizando subplots
 (uno para cada región recortada), con los títulos acordes.
"""

