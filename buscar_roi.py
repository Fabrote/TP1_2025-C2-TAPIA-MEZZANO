import cv2
import matplotlib.pyplot as plt

# Cargar imagen
img_path = r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\TP1\formulario_01.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo cargar la imagen")
else:
    alto, ancho = img.shape

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, cmap='gray')
    ax.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.show()
