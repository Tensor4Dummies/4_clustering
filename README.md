# CLUSTERING

## Introducción
El clustering consiste en la división de un conjunto de datos X en K subconjuntos (clusters) distintos C<sub>1</sub>, C<sub>2</sub>…C<sub>K</sub> tales que los objetos dentro de cada subconjunto son similares y objetos en distintos subconjuntos son diferentes. Es decir, que los documentos, imágenes o vectores de características en un cluster deberían ser parecidos, y los de clusters diferentes deberían ser distintos.
El objetivo es repartir los datos en clases con alta similitud intraclase y baja similitud interclase.

En la siguiente imagen se pueden apreciar varias formas geométricas, que formarían tres clusters. En cada cluster los elementos son  parecidos, pudiendo variar de tamaño o de color. Los elementos de cada uno de los grupos se diferencian claramente de los elementos de otras agrupaciones.

![Ejemplo de clusters](/ejemplo_clusters.png)

<span>Es la forma más común del **aprendizaje no supervisado**. En el aprendizaje supervisado, el proceso de aprendizaje se lleva a cabo a través de entrenamiento controlado por un agente externo que indica la respuesta que debería generar la red a partir de una entrada determinada. Sin embargo, en el aprendizaje no supervisado, no se proporcionan estos ejemplos, y son los propios algoritmos los que tienen que descubrir características de los datos y encontrar alguna manera de organizarlos.</span>


## Algoritmos de clustering
1.	**Clustering basado en centroides**. Construyen k particiones de los datos, donde cada partición representa un cluster. Cada grupo tiene al menos un elemento y cada elemento pertenece a un solo grupo. Dentro de esta categoría, los más usados son: k-medias (k-means) y k-medianas.
2.	**Clustering jerárquico o basado en conectividad**. Está basado en la idea principal de un objeto está más relacionado con objetos cercanos que con los lejanos.
3.	**Clustering basado en la distribución**. Los clusters se definen como objetos que pertenecen con más probabilidad a la misma distribución.  EM (expectativa de maximización), mezcla de gaussianas
4.	**Clustering basado en la densidad**. Se agrupan objetos en clusters mientras su número de elementos (densidad) en el cluster más cercano esté dentro de un cierto umbral.


## Clustering k-medias (k-means)
Se establecen k clases a priori, todas separadas por fronteras de decisión, que se establece como la mitad de la distancia entre centroides de distintas clases.
