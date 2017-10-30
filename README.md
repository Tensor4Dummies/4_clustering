# CLUSTERING

- [Introducción](#introducción)
- [Algoritmos de clustering](#algoritmos-de-clustering)
- [Clustering con k-means](#clustering-con-k-medias-k-means)
- [Ejemplo con TensorFlow](#ejemplo-con-tensorflow)  
        1. [Crear los datos de muestra](#1-crear-los-datos-de-muestra)  
        2. [Elegir los centroides iniciales](#2-elegir-los-centroides-iniciales)  
        3. [Asignar el centroide más cercano a cada punto](#3-asignar-el-centroide-más-cercano-a-cada-punto)  
        4. [Actualizar los centroides](#4-actualizar-los-centroides)  
        5. [Mostrar el resultado de forma gráfica](#5-mostrar-el-resultado-de-forma-gráfica)  


## Introducción
El clustering consiste en la división de un conjunto de datos X en K subconjuntos (clusters) distintos C<sub>1</sub>, C<sub>2</sub>…C<sub>K</sub> tales que los objetos dentro de cada subconjunto son similares y objetos en distintos subconjuntos son diferentes. Es decir, que los documentos, imágenes o vectores de características en un cluster deberían ser parecidos, y los de clusters diferentes deberían ser distintos.  
El objetivo es particionar los datos en clases con alta similitud intraclase y baja similitud interclase.

En la siguiente imagen se pueden apreciar varias formas geométricas, que formarían tres clusters. En cada cluster los elementos son  parecidos, pudiendo variar de tamaño o de color. Los elementos de cada uno de los grupos se diferencian claramente de los elementos de otras agrupaciones.

![Ejemplo de clusters](/ejemplo_clusters.png)

<span>El clustering es la forma más común del **aprendizaje no supervisado**. En el aprendizaje supervisado, el proceso de aprendizaje se lleva a cabo a través de entrenamiento controlado por un agente externo que indica la respuesta que debería generar la red a partir de una entrada determinada. Sin embargo, en el aprendizaje no supervisado, no se proporcionan estos ejemplos y son los propios algoritmos los que tienen que descubrir características de los datos y encontrar alguna manera de organizarlos.</span>
</br>

## Algoritmos de clustering
1.	**Clustering basado en centroides**. Construyen k particiones de los datos, donde cada partición representa un cluster. Cada grupo tiene al menos un elemento y cada elemento pertenece a un solo grupo. Dentro de esta categoría, los más usados son: k-medianas y k-medias (k-means), siendo este último en el que nos centraremos para el ejemplo. 
2.	**Clustering jerárquico o basado en conectividad**. Está basado en la idea principal de un objeto está más relacionado con objetos cercanos que con los lejanos.
3.	**Clustering basado en la distribución**. Los clusters se definen como objetos que pertenecen con más probabilidad a la misma distribución.  EM (expectativa de maximización), mezcla de gaussianas
4.	**Clustering basado en la densidad**. Se agrupan objetos en clusters mientras su número de elementos (densidad) en el cluster más cercano esté dentro de un cierto umbral.
</br>


## Clustering con k-medias (k-means)
El objetivo es minimizar la diferencia intra-cluster y maximizar la diferencia inter-cluster, es decir, que los elementos dentro de un mismo cluster sean lo más parecido posible, y que, comparándolos con elementos de otros clusters, sean lo más distinto posible.
Para ello, se utiliza un algoritmo llamado **algoritmo de Lloyd**, que consiste en:
1.	Inicialmente, se define K, que será el número de clusters. Este número se puede escoger de manera aleatoria, pero dependiendo del número elegido el resultado variará.</br>Lo que se suele hacer es tomar K observaciones de la muestra al azar. Estos k datos elegidos serán los centroides iniciales.
2.	Sabiendo que en total hay N muestras, para cada uno de los N-K datos restantes se calcula la distancia entre ese dato y cada uno de los centroides.</br>Para calcular esta distancia, se pueden utilizar una variedad de fórmulas, pero normalmente la que se usa es la distancia euclídea o euclidiana. Los datos están formados por coordenadas cartesianas de n dimensiones. Para hacer esta operación, hay que restar las coordenadas del punto menos las coordenadas del centroide.  
Suponiendo que estamos en un espacio n-dimensional, la distancia euclídea entre los puntos ![equation](http://www.sciweavers.org/upload/Tex2Img_1509306482/render.png) y ![equation](http://www.sciweavers.org/upload/Tex2Img_1509306540/render.png) sería así:  
![equation](http://www.sciweavers.org/upload/Tex2Img_1509306370/render.png)

3.	Una vez obtenida la distancia para cada dato de la muestra, se asigna cada uno de estos datos al centroide cuya distancia euclídea sea la mínima.
4.	Al terminar, se tienen K grupos de observaciones.
5.	Conociendo los nuevos elementos de cada cluster, ahora se calcula el nuevo centroide de cada grupo basándose en los nuevos miembros.</br> Lo que se hace es sumar las coordenadas de cada punto y dividirlo por el número de elementos del cluster. Suponiendo de nuevo que estamos en un espacio n-dimensional, y que m es el nuevo número de elementos del cluster, los nuevos centroides se calcularían así:  
![equation](http://www.sciweavers.org/upload/Tex2Img_1509306768/render.png)
6.	Repetir el proceso hasta que no haya reasignaciones de puntos a clusters distintos.
</br>
Como resultado, el algoritmo devuelve los centroides definitivos y los vectores de etiquetas, que asignan cada punto de la muestra a una de las clases.


## Ejemplo con TensorFlow
Una vez entendido el funcionamiento de clustering con el método k-medias, pasamos a implementar un ejemplo con TensorFlow.
Para ello usaremos Python y la librería de TensorFlow, además de la librería NumPy para realizar algunos cálculos complejos y la librería MatPlotLib, para poder ver el resultado final de manera gráfica.
Para empezar, vamos a definir algunos valores que necesitará el algoritmo para poder realizar todos sus pasos.
###  1. Crear los datos de muestra

###  2. Elegir los centroides iniciales

###  3. Asignar el centroide más cercano a cada punto

###  4. Actualizar los centroides

###  5. Mostrar el resultado de forma gráfica
