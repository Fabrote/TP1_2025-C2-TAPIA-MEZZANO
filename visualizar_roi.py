import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar una imagen de ejemplo
img_path = r'D:\FABRO\TUIA\PROCESAMIENTO DE IMAGENES\TP1\formulario_05.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo cargar la imagen")
else:
    alto, ancho = img.shape
    print(f"Dimensiones imagen: {ancho} x {alto}")
    
    # Recorte superior derecho
    roi = img[30:58, 557:580]
    
    print(f"Dimensiones ROI: {roi.shape[1]} x {roi.shape[0]}")
    
    # Umbralizar
    _, binaria = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Componentes conectadas para calcular área
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, 8, cv2.CV_32S)
    
    print(f"\n--- ANÁLISIS DE LA LETRA ---")
    print(f"Componentes detectados: {num_labels - 1}")
    
    if num_labels > 1:
        # Tomar la componente más grande (la letra)
        max_idx = 1
        max_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            print(f"  Componente {i}: Área = {area} px")
            if area > max_area:
                max_area = area
                max_idx = i
        
        print(f"\nComponente más grande (letra): Área = {max_area} px")
        
        # Calcular área relativa
        letra_crop = binaria[labels == max_idx]
        area_rel = np.sum(letra_crop) / letra_crop.size if letra_crop.size > 0 else 0
        print(f"Área relativa: {area_rel:.3f}")
        
        # Crear visualización con la letra marcada
        letra_visual = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        x = stats[max_idx, cv2.CC_STAT_LEFT]
        y = stats[max_idx, cv2.CC_STAT_TOP]
        w = stats[max_idx, cv2.CC_STAT_WIDTH]
        h = stats[max_idx, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(letra_visual, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        print("No se detectó ninguna letra")
        letra_visual = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(roi, cmap='gray')
    axes[0].set_title('ROI extraído (zona de la letra)')
    axes[0].axis('off')
    
    axes[1].imshow(binaria, cmap='gray')
    axes[1].set_title('ROI binarizado')
    axes[1].axis('off')
    
    axes[2].imshow(letra_visual)
    axes[2].set_title(f'Letra detectada (Área: {max_area if num_labels > 1 else 0} px)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
