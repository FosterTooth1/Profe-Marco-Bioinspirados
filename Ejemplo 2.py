import numpy as np
import cv2
import matplotlib.pyplot as plt

# Leemos la imagen y la convertimos a escala de grises
img_color = cv2.imread("kodim23.png")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
# Preparamos una versión normalizada para el cálculo de entropía
b = img_gray.astype(np.float32) / 255.0

# Parámetros del algoritmo
Num_iteraciones = int(input('Ingrese el número de iteraciones: '))
Num_Generaciones = int(input('Ingrese el número de generaciones por iteración: '))
Num_pob = int(input('Ingrese el tamaño de la población: '))

# PSO hiperparámetros
c1 = 3.0
c2 = 3.0
w = 0.6

# Sólo una variable: clipLimit
Num_var = 1
Limite_Inferior = np.array([1.0], dtype=np.float32)
Limite_Superior = np.array([40.0], dtype=np.float32)

# Para guardar resultados
Resultados_Generales = np.zeros(Num_iteraciones)
Mejor_Individuo_Iteracion = np.zeros((Num_iteraciones, Num_var))

def calcular_entropia(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag /= grad_mag.max()  # normalizamos a [0,1]

    hist_counts, _ = np.histogram(grad_mag.flatten(), bins=256, range=(0,1))
    p = hist_counts / hist_counts.sum()
    p = p[p>0]
    return -np.sum(p * np.log2(p))

# Convertimos la imagen gris a uint8 para usar CLAHE
gray_uint8 = img_gray.copy()

for it in range(Num_iteraciones):
    # Inicializamos posiciones y velocidades
    x = np.random.uniform(Limite_Inferior, Limite_Superior, (Num_pob, Num_var))
    v = np.random.uniform(-(Limite_Superior-Limite_Inferior),
                          (Limite_Superior-Limite_Inferior),
                          (Num_pob, Num_var))
    # Topología en anillo
    nbh = [[i, (i-1)%Num_pob, (i+1)%Num_pob] for i in range(Num_pob)]

    x_p_best = x.copy()
    best_apt = np.full(Num_pob, np.inf)

    for gen in range(Num_Generaciones):
        aptitud = np.zeros(Num_pob)

        # Evaluamos cada partícula
        for i in range(Num_pob):
            clip = x[i,0]
            # Creamos y aplicamos CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
            img_clahe = clahe.apply(gray_uint8)
            # Normalizamos a [0,1] para el cálculo de entropía
            img_norm = img_clahe.astype(np.float32) / 255.0

            aptitud[i] = -calcular_entropia(img_norm)

        # Actualizamos mejores personales
        mejor_mask = aptitud < best_apt
        x_p_best[mejor_mask] = x[mejor_mask]
        best_apt[mejor_mask] = aptitud[mejor_mask]

        # Actualizamos velocidades y posiciones
        for i in range(Num_pob):
            vecinos = nbh[i]
            apt_vec = [aptitud[vx] for vx in vecinos]
            mejor_vecino = vecinos[np.argmin(apt_vec)]
            x_g_best = x[mejor_vecino]

            r1, r2 = np.random.rand(2)
            v[i] = (w * v[i]
                    + c1 * r1 * (x_p_best[i] - x[i])
                    + c2 * r2 * (x_g_best   - x[i]))
            x[i] += v[i]

            # Aplicamos límites
            for j in range(Num_var):
                if x[i,j] < Limite_Inferior[j] or x[i,j] > Limite_Superior[j]:
                    x[i,j] = np.random.uniform(Limite_Inferior[j], Limite_Superior[j])
                    v[i,j] = np.random.uniform(-(Limite_Superior[j]-Limite_Inferior[j]),
                                              (Limite_Superior[j]-Limite_Inferior[j]))

    # Guardamos el mejor de la iteración
    idx = np.argmin(best_apt)
    Mejor_Individuo_Iteracion[it] = x_p_best[idx]
    Resultados_Generales[it] = best_apt[idx]
    print(f"Iteración {it+1}: Mejor clipLimit = {x_p_best[idx,0]:.4f}, Costo = {best_apt[idx]:.4f}")

# Estadísticas finales
mejor = Resultados_Generales.min()
media = Resultados_Generales.mean()
peor = Resultados_Generales.max()
std = Resultados_Generales.std()
print("\n=== Resultados Finales ===")
print(f"Mejor: {mejor:.4f}   Media: {media:.4f}   Peor: {peor:.4f}   Desv.: {std:.4f}")

# Aplicación con la mejor solución
best_idx = np.argmin(Resultados_Generales)
best_clip = Mejor_Individuo_Iteracion[best_idx,0]

clahe_best = cv2.createCLAHE(clipLimit=best_clip, tileGridSize=(8,8))
img_best = clahe_best.apply(gray_uint8)

# Entropías
ent_orig = calcular_entropia(b)
ent_mejor = calcular_entropia(img_best.astype(np.float32)/255.0)
print(f"\nEntropía Original: {ent_orig:.4f}")
print(f"Entropía Mejorada: {ent_mejor:.4f}")

# Mostrar comparación
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(gray_uint8, cmap='gray')
plt.title('Original (Grayscale)')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_best, cmap='gray')
plt.title(f'CLAHE (clipLimit={best_clip:.2f})')
plt.axis('off')
plt.show()