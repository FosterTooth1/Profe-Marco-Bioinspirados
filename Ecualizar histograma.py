import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

def ecualizar_histograma(imagen_path):
    # Leer la imagen
    imagen = cv2.imread(imagen_path)
    
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen en la ruta {imagen_path}")
        return
    
    # Convertir a RGB para visualización correcta de colores
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    # Ecualización para diferentes tipos de imágenes
    if len(imagen.shape) == 2:
        # Imagen en escala de grises
        ecualizada = cv2.equalizeHist(imagen)
        titulo_original = 'Imagen Original (Grises)'
        titulo_ecualizada = 'Imagen Ecualizada (Grises)'
    else:
        # Imagen a color (ecualizamos el canal V en HSV)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_ecualizado = cv2.equalizeHist(v)
        hsv_ecualizado = cv2.merge([h, s, v_ecualizado])
        ecualizada = cv2.cvtColor(hsv_ecualizado, cv2.COLOR_HSV2RGB)
        titulo_original = 'Imagen Original (Color)'
        titulo_ecualizada = 'Imagen Ecualizada (Color)'

    # Mostrar resultados
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    ax[0].imshow(imagen_rgb)
    ax[0].set_title(titulo_original)
    ax[0].axis('off')
    
    if len(imagen.shape) == 2:
        ax[1].imshow(ecualizada, cmap='gray')
    else:
        ax[1].imshow(ecualizada)
    ax[1].set_title(titulo_ecualizada)
    ax[1].axis('off')
    
    plt.show()

if __name__ == "__main__":
    ecualizar_histograma("kodim23.png") 