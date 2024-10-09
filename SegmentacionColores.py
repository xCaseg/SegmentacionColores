
# ---------------------------------------- SEGMENTACIÓN DE COLORES POR DISTANCIA ----------------------------------- #

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------------------- Carga de imagen para ambos métodos ----------------------------------- #

def cargar_imagen(ruta):
    imagen = cv2.imread(ruta)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen_rgb

# ................................................................................................................... #

# +------------------------------------------------------------------------------------------------------------------+ 
# |                                                                                                                  |
# |                                          +---------------------------------+                                     |
# |                                          | MÉTODO POR DISTANCIA EUCLIDIANA |                                     |
# |                                          +---------------------------------+                                     |
# |                                                                                                                  |
# +------------------------------------------------------------------------------------------------------------------+ 

# ................................................................................................................... #

def detectar_formato(imagen):
    if len(imagen.shape) == 2:
        return "Escala de grises (PGB)"
    elif len(imagen.shape) == 3:
        if imagen.shape[2] == 3:
            return "RGB"
        elif imagen.shape[2] == 4:
            if (imagen[:, :, 3] == 255).all():
                return "RGBA"
            else:
                return "RGBA (con transparencia)"
    else:
        return "Formato de color no reconocido"

# ................................................................................................................... #

def mostrar_informacion_imagen(imagen):
    formato = detectar_formato(imagen)
    print("El formato de color de la imagen es:", formato)
    alto, ancho, profundidad = imagen.shape
    print("Altura de la imagen:", alto)
    print("Anchura de la imagen:", ancho)
    print("Profundidad de la imagen (número de canales de color):", profundidad)
    print("Cantidad de pixeles:", imagen.size)

# ................................................................................................................... #


def seleccionar_pixeles_muestra(imagen_rgb, coord1, coord2): # De imagen se toman las dimensiones 0 y 1; ancho y alto (x, y)
    muestra1 = imagen_rgb[coord1[0], coord1[1]]
    muestra2 = imagen_rgb[coord2[0], coord2[1]]
    return muestra1, muestra2

# ................................................................................................................... #

def calcular_distancia_euclidiana(pixel1, pixel2):
    distancia = math.sqrt((pixel1[0] - pixel2[0])**2 + (pixel1[1] - pixel2[1])**2 + (pixel1[2] - pixel2[2])**2)
    return distancia

# ................................................................................................................... #

def segmentar_colores_por_distancia(imagen, muestra1, muestra2):
    alto, ancho, _ = imagen.shape
    d_rango = calcular_distancia_euclidiana(muestra1, muestra2)

    matriz_segmentada = np.empty((alto, ancho, 3), dtype=np.uint8) #Crear una nueva matriz con las mismas dimensiones 
    # que la imagen

    for i in range(alto):
        for j in range(ancho):
            distancia_entre_pixel_y_muestra = calcular_distancia_euclidiana(imagen[i, j], muestra1)
            if distancia_entre_pixel_y_muestra < d_rango:
                matriz_segmentada[i, j] = imagen[i, j]
            else:
                matriz_segmentada[i, j] = [255, 255, 255]
    return matriz_segmentada

# +------------------------------------------------------------------------------------------------------------------+ 
# |                                                                                                                  |
# |                                          +----------------------------------+                                    |
# |                                          | MÉTODO POR DISTANCIA MAHALANOBIS |                                    |
# |                                          +----------------------------------+                                    |
# |                                                                                                                  |
# +------------------------------------------------------------------------------------------------------------------+ 

# ................................................................................................................... #

def d_mahalanobis(x, y, covarianza):
    dif = x - y
    covarianza_max = np.max(np.abs(covarianza))  # Encontrar el valor máximo absoluto en la matriz de covarianza
    covarianza_normalizada = covarianza / covarianza_max
    covarianza_inversa = np.linalg.inv(covarianza_normalizada)
    dist_mahalanobis = np.sqrt(np.dot(np.dot(dif, covarianza_inversa), dif.T))
    return dist_mahalanobis

'''

    La fórmula es la siguiente:  D= sqrt (vector x- vector y)^t * C^-1 * ( vector x- vector y). 

    # En python se interpreta de la siguiente manera:

    # dif = x - y : calcula la diferencia elemento a elemento entre el vector  x y el vector y

    # covarianza_inversa = np.linalg.inv(covarianza_normalizada):  Aquí se calcula la inversa de la matriz de covarianza C^-1

    # np.dot(np.dot(diff, covarianza_inversa), dif.T): Es el producto punto entre la diferencia de vectores x−y 

    y la inversa de la matriz de covarianza C^1 donde donde .T realiza la transposición.

    # (.dot() en NumPy se utiliza para calcular el producto entre dos arrays)

    # El primer .dot() se encarga de realizar la multiplicación matricial entre dif y covarianza_inversa

    # El segundo .dot() calcula una segunda multiplicación matricial entre la salida del primer .dot() 

    y la transpuesta de la matriz dif

 
 '''   

# ................................................................................................................... #

def seleccionar_puntos_muestra(imagen_rgb):
    plt.imshow(imagen_rgb)
    plt.title('Selecciona 10 píxeles de muestra')
    tupla_coordenadas = plt.ginput(10, timeout=0)
    plt.close()
    print("\nPíxeles seleccionados:\n")
    for pixel in tupla_coordenadas:
        print(pixel)
    return np.asarray(tupla_coordenadas, dtype=int)

# ................................................................................................................... #

def obtener_valores_pixeles(imagen_rgb, matriz_coordenadas):
    valores_pixeles = []
    for coord in matriz_coordenadas:
        x, y = coord[0], coord[1]
        valor_pixel = imagen_rgb[y, x]
        valores_pixeles.append(valor_pixel)
    return np.array(valores_pixeles)

# ................................................................................................................... #

def calcular_matriz_covarianza(valores_pixeles):
    return np.cov(valores_pixeles, rowvar=False)

# ................................................................................................................... #

def calcular_media_pixeles(valores_pixeles):
    return np.mean(valores_pixeles, axis=0)

# ................................................................................................................... #

def distancia_maxima_entre_puntos(valores_pixeles):
    distancia_maxima = 0
    for i in range(len(valores_pixeles)):
        for j in range(i+1, len(valores_pixeles)):
            distancia = np.linalg.norm(valores_pixeles[i] - valores_pixeles[j])
            if distancia > distancia_maxima:
                distancia_maxima = distancia
    return distancia_maxima

# ................................................................................................................... #

def segmentar_por_mahalanobis_con_distancia_maxima(imagen_rgb, media_pixeles, matriz_covarianza, distancia_maxima):
    imagen_resultante = np.zeros_like(imagen_rgb)
    for i in range(imagen_rgb.shape[0]):
        for j in range(imagen_rgb.shape[1]):
            x = imagen_rgb[i, j]
            distancia_mahalanobis = d_mahalanobis(x, media_pixeles, matriz_covarianza)
            if distancia_mahalanobis < distancia_maxima:
                imagen_resultante[i, j] = imagen_rgb[i, j]
            else:
                imagen_resultante[i, j] = [255, 255, 255]
    return imagen_resultante

# ................................................................................................................... #


# +------------------------------------------------------------------------------------------------------------------+ 
# |                                                                                                                  |
# |                                          +----------------------+                                                |
# |                                          | EJECUCIÓN DEL CÓDIGO |                                                |
# |                                          +----------------------+                                                |
# |                                                                                                                  |
# +------------------------------------------------------------------------------------------------------------------+ 


# ---------------------------------------- FUNCIONES PARA AMBOS MÉTODOS -------------------------------------------- #

# Ruta de la imagen
ruta_imagen = r'C:\Users\DELL\Desktop\Imagen.jpg'

# Cargar la imagen
imagen = cargar_imagen(ruta_imagen)

# Mostrar información de la imagen
mostrar_informacion_imagen(imagen)

# ---------------------------------------- MÉTODO DISTANCIA EUCLIDIANA --------------------------------------------- #

# Coordenadas de los píxeles de muestra
coord1 = (187, 62)
coord2 = (460, 224)

# Seleccionar los píxeles de muestra
muestra1, muestra2 = seleccionar_pixeles_muestra(imagen, coord1, coord2)

# Segmentar los colores por distancia
matriz_segmentada = segmentar_colores_por_distancia(imagen, muestra1, muestra2)


# ---------------------------------------- MÉTODO DISTANCIA MAHALANOBIS -------------------------------------------- #

# Paso 1: Seleccionar los píxeles de muestra
matriz_coordenadas = seleccionar_puntos_muestra(imagen)

# Paso 2: Obtener los valores de los píxeles seleccionados
valores_pixeles = obtener_valores_pixeles(imagen, matriz_coordenadas)

# Paso 3: Calcular la matriz de covarianza
matriz_covarianza = calcular_matriz_covarianza(valores_pixeles)

# Paso 4: Calcular la media de los píxeles seleccionados
media_pixeles = calcular_media_pixeles(valores_pixeles)

# Paso 5: Calcular la distancia máxima entre los 10 píxeles de muestra
distancia_maxima = distancia_maxima_entre_puntos(valores_pixeles)

# Paso 6: Segmentar por distancia de Mahalanobis con la distancia máxima
imagen_resultante = segmentar_por_mahalanobis_con_distancia_maxima(imagen, media_pixeles, matriz_covarianza, distancia_maxima)


# -------------------------------------------- MOSTRAR RESULTADOS -------------------------------------------------- #

plt.figure(figsize=(14, 12))

# Subplot 1: Imagen Original - Euclidiana
plt.subplot(2, 2, 1)
plt.imshow(imagen)
plt.title('Imagen Original - Euclidiana')
plt.axis('off')

# Subplot 2: Píxeles dentro del rango Euclidiano
plt.subplot(2, 2, 2)
plt.imshow(matriz_segmentada)
plt.title('Píxeles dentro del rango Euclidiano')
plt.axis('off')

# Subplot 3: Imagen Original - Mahalanobis
plt.subplot(2, 2, 3)
plt.imshow(imagen)
plt.title('Imagen Original - Mahalanobis')
plt.axis('off')

# Subplot 4: Píxeles dentro del rango de Mahalanobis
plt.subplot(2, 2, 4)
plt.imshow(imagen_resultante)
plt.title('Píxeles dentro del rango de Mahalanobis')
plt.axis('off')

plt.tight_layout()  # Ajustar automáticamente los espacios entre las subgráficas
plt.show()


# ................................................................................................................... #


