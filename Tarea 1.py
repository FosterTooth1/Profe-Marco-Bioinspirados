import numpy as np
import matplotlib.pyplot as plt

# Solicitar parámetros al usuario
num_generaciones = int(input('Ingrese el número de generaciones en el algoritmo: '))
num_pob = int(input('Ingrese el número de individuos dentro de la población: '))
num_var = int(input('Ingrese la cantidad de variables que tiene cada individuo: '))

limite_inferior = np.zeros(num_var)
for i in range(num_var):
    limite_inferior[i] = float(input(f'Ingrese el limite inferior de la variable {i+1}: '))

limite_superior = np.zeros(num_var)
for i in range(num_var):
    limite_superior[i] = float(input(f'Ingrese el limite superior de la variable {i+1}: '))

Pc = float(input('Ingrese la probabilidad de cruza del algoritmo: '))
Pm = float(input('Ingrese la probabilidad de mutacion del algoritmo: '))

# Parámetros para el operador SBX y la mutación polinomial
Nc = 2    # valor base
Nm = 20   # valor base

# Inicializar población
poblacion = np.random.uniform(
    low=limite_inferior,
    high=limite_superior,
    size=(num_pob, num_var)
)

# Función objetivo
def evaluar_individuo(individuo):
    x1, x2 = individuo[0], individuo[1]
    a = [3, 5, 2, 1, 7]
    b = [5, 2, 1, 4, 9]
    c = [1, 2, 5, 2, 3]
    suma = 0.0
    for i in range(5):
        d = (x1 - a[i])**2 + (x2 - b[i])**2
        termino = (c[i] * np.cos(np.pi * d)) / np.exp(d / np.pi)
        suma += termino
    return -suma

def evaluar_poblacion(poblacion):
    fitness = np.array([evaluar_individuo(individuo) for individuo in poblacion])
    return fitness

# Evaluamos la población inicial
fitness = evaluar_poblacion(poblacion)

# Ciclo de generaciones
for generacion in range(num_generaciones):

    # Determinamos parametros de nc y nm dependiendo de la generación para fomentar la exploración y explotación
    if generacion < num_generaciones * 0.5:
        Nc, Nm = 2, 20
    elif generacion < num_generaciones * 0.75:
        Nc, Nm = 5, 50
    elif generacion < num_generaciones * 0.80:
        Nc, Nm = 10, 75
    elif generacion < num_generaciones * 0.95:
        Nc, Nm = 15, 85
    else:
        Nc, Nm = 20, 100
        
    # Seleccionar padres por torneo
    padres = np.zeros_like(poblacion)
    # Para cada posición se escogen dos individuos al azar y se selecciona el mejor
    indices = np.arange(num_pob)
    torneo_indices = np.array([np.random.permutation(num_pob) for _ in range(2)]).T
    for i in range(num_pob):
        a, b = torneo_indices[i]
        if fitness[a] < fitness[b]:
            padres[i] = poblacion[a]
        else:
            padres[i] = poblacion[b]

    # Cruzamiento SBX
    hijos = np.zeros_like(padres)
    for i in range(0, num_pob, 2):
        # Si no hay par, se copia el último
        if i+1 >= num_pob:
            hijos[i] = padres[i]
            break

        if np.random.rand() <= Pc:
            hijo1, hijo2 = [], []
            for j in range(num_var):
                p1 = padres[i, j]
                p2 = padres[i+1, j]
                
                # Evitar división por cero
                if abs(p2 - p1) < 1e-14:
                    hijo1.append(p1)
                    hijo2.append(p2)
                    continue

                # Cálculo de beta para SBX
                delta = p2 - p1
                beta = 1 + (2 / delta) * min(p1 - limite_inferior[j], limite_superior[j] - p2)
                alpha = 2 - beta**(-(Nc + 1))
                U = np.random.rand()

                if U <= 1/alpha:
                    beta_c = (U * alpha)**(1/(Nc+1))
                else:
                    beta_c = (1 / (2 - U*alpha))**(1/(Nc+1))

                # Generación de los dos descendientes
                c1 = 0.5*((p1+p2) - beta_c*abs(delta))
                c2 = 0.5*((p1+p2) + beta_c*abs(delta))
                # Limitar para que no excedan los límites
                c1 = np.clip(c1, limite_inferior[j], limite_superior[j])
                c2 = np.clip(c2, limite_inferior[j], limite_superior[j])
                hijo1.append(c1)
                hijo2.append(c2)

            hijos[i] = hijo1
            hijos[i+1] = hijo2
        else:
            hijos[i] = padres[i]
            hijos[i+1] = padres[i+1]

    # Mutación polinomial
    for i in range(num_pob):
        for j in range(num_var):
            if np.random.rand() <= Pm:
                delta = min(limite_superior[j] - hijos[i,j],
                            hijos[i,j] - limite_inferior[j]) / (limite_superior[j] - limite_inferior[j])
                r = np.random.rand()
                if r <= 0.5:
                    deltaq = (2*r + (1 - 2*r)*(1 - delta)**(Nm + 1))**(1/(Nm + 1)) - 1
                else:
                    deltaq = 1 - (2*(1 - r) + 2*(r - 0.5)*(1 - delta)**(Nm + 1))**(1/(Nm + 1))
                hijos[i,j] += deltaq * (limite_superior[j] - limite_inferior[j])
                # Asegurar que la variable se mantenga dentro de los límites
                hijos[i,j] = np.clip(hijos[i,j], limite_inferior[j], limite_superior[j])

    # Evaluar la nueva población de hijos
    fitness_hijos = evaluar_poblacion(hijos)

    # Sustitución de la población (elitismo)
    poblacion_extendida = np.vstack((poblacion, hijos))
    fitness_extendida = np.hstack((fitness, fitness_hijos))
    
    # Se ordena la población extendida de mejor a peor
    orden = np.argsort(fitness_extendida)
    poblacion_extendida = poblacion_extendida[orden]
    fitness_extendida = fitness_extendida[orden]

    # Se conserva la mitad superior: los 'num_pob' mejores individuos
    poblacion = poblacion_extendida[:num_pob]
    fitness = fitness_extendida[:num_pob]

    # Graficar la población de la generación actual
    if num_var >= 2:
        plt.clf()
        plt.scatter(poblacion[:, 0], poblacion[:, 1], s=50, c='blue')
        plt.title(f'Generación {generacion+1}')
        plt.xlim(limite_inferior[0], limite_superior[0])
        plt.ylim(limite_inferior[1], limite_superior[1])
        plt.grid(True)
        plt.pause(0.1)

# Se determina el mejor individuo encontrado
mejor_idx = np.argmin(fitness)
mejor = fitness[mejor_idx]
mejor_x = poblacion[mejor_idx]

print('\nResultados finales del algoritmo genético:')
print(f'Mejor aptitud: {mejor}')
print(f'Variables del mejor individuo: {mejor_x}')

plt.show()
