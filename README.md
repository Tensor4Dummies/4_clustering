# CLUSTERING

## Introducción
El clustering consiste en la división de un conjunto de datos X en K subconjuntos (clusters) distintos C<sub>1</sub>, C<sub>2</sub>…C<sub>K</sub> tales que los objetos dentro de cada subconjunto son similares y objetos en distintos subconjuntos son diferentes. Es decir, que los documentos, imágenes o vectores de características en un cluster deberían ser parecidos, y los de clusters diferentes deberían ser distintos.
El objetivo es particionar los datos en clases con alta similitud intraclase y baja similitud interclase.

En la siguiente imagen se pueden apreciar varias formas geométricas, que formarían tres clusters. En cada cluster los elementos son  parecidos, pudiendo variar de tamaño o de color. Los elementos de cada uno de los grupos se diferencian claramente de los elementos de otras agrupaciones.

![Ejemplo de clusters](/ejemplo_clusters.png)

<span>Es la forma más común del **aprendizaje no supervisado**. En el aprendizaje supervisado, el proceso de aprendizaje se lleva a cabo a través de entrenamiento controlado por un agente externo que indica la respuesta que debería generar la red a partir de una entrada determinada. Sin embargo, en el aprendizaje no supervisado, no se proporcionan estos ejemplos, y son los propios algoritmos los que tienen que descubrir características de los datos y encontrar alguna manera de organizarlos.</span>


## Algoritmos de clustering
1.	**Clustering basado en centroides**. Construyen k particiones de los datos, donde cada partición representa un cluster. Cada grupo tiene al menos un elemento y cada elemento pertenece a un solo grupo. Dentro de esta categoría, los más usados son: k-medias (k-means) y k-medianas.
2.	**Clustering jerárquico o basado en conectividad**. Está basado en la idea principal de un objeto está más relacionado con objetos cercanos que con los lejanos.
3.	**Clustering basado en la distribución**. Los clusters se definen como objetos que pertenecen con más probabilidad a la misma distribución.  EM (expectativa de maximización), mezcla de gaussianas
4.	**Clustering basado en la densidad**. Se agrupan objetos en clusters mientras su número de elementos (densidad) en el cluster más cercano esté dentro de un cierto umbral.



## Clustering k-medias (k-means)
El objetivo es minimizar la diferencia intra-cluster y maximizar la diferencia inter-cluster, es decir, que los elementos dentro de un mismo cluster sean lo más parecido posible, y que, comparándolos con elementos de otros clusters, sean lo más distinto posible.
Para ello, se utiliza un algoritmo llamado **algoritmo de Lloyd**, que consiste en:
1.	Inicialmente, se define K, que será el número de clusters. Este número se puede escoger de manera aleatoria, pero dependiendo del número elegido el resultado variará.</br>Lo que se suele hacer es tomar K observaciones de la muestra al azar. Estos k datos elegidos serán los centroides iniciales.
2.	Sabiendo que en total hay N muestras, para cada uno de los N-K datos restantes se calcula la distancia entre ese dato y cada uno de los centroides.</br>Para calcular esta distancia, se usa la fórmula de la distancia euclídea o euclidiana. Los datos están formados por coordenadas cartesianas de n dimensiones. Para hacer esta operación, hay que restar las coordenadas del punto menos las coordenadas del centroide.</br>
Suponiendo que estamos en un espacio n-dimensional, la distancia euclídea entre los puntos P=(p<sub>1</sub>, p<sub>2</sub>, ..., p<sub>n</sub>) y Q=(q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>n</sub>) sería así:</br>


3.	Una vez obtenida la distancia para cada dato de la muestra, se asigna cada uno de estos datos al centroide cuya distancia euclídea sea la mínima.
4.	Al terminar, se tienen K grupos de observaciones.
5.	Conociendo los nuevos elementos de cada cluster, ahora se calcula el nuevo centroide de cada grupo basándose en los nuevos miembros.</br> Lo que se hace es sumar las coordenadas de cada punto y dividirlo por el número de elementos del cluster. 
6.	Repetir el proceso hasta que no haya reasignaciones de puntos a clusters distintos.

