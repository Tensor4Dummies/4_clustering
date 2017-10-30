# CLUSTERING

- [Introducción](#introducción)
- [Algoritmos de clustering](#algoritmos-de-clustering)
- [Clustering con k-means](#clustering-con-k-medias-k-means)
- [Ejemplo con TensorFlow](#ejemplo-con-tensorflow)  
     1. [Generación de datos y selección de centroides iniciales](#i-generación-de-datos-y-selección-de-centroides-iniciales)
     2. [Cálculo de la distancia entre puntos y centroides](#ii-cálculo-de-la-distancia-entre-puntos-y-centroides)
     3. [Cálculo y actualización de los nuevos centroides](#iii-cálculo-y-actualización-de-los-nuevos-centroides)
     4. [Ejecución del algoritmo](#iv-ejecución-del-algoritmo)
     5. [Mostrar el resultado gráficamente](#v-mostrar-el-resultado-gráficamente)


___



## Introducción
El clustering consiste en la división de un conjunto de datos X en K subconjuntos (clusters) distintos C<sub>1</sub>, C<sub>2</sub>…C<sub>K</sub> tales que los objetos dentro de cada subconjunto son similares y objetos en distintos subconjuntos son diferentes. Es decir, que los documentos, imágenes o vectores de características en un cluster deberían ser parecidos, y los de clusters diferentes deberían ser distintos.  
El objetivo es particionar los datos en clases con alta similitud intraclase y baja similitud interclase.

En la siguiente imagen se pueden apreciar varias formas geométricas, que formarían tres clusters. En cada cluster los elementos son  parecidos, pudiendo variar de tamaño o de color. Los elementos de cada uno de los grupos se diferencian claramente de los elementos de otras agrupaciones.

![Ejemplo de clusters](/ejemplo_clusters.png)

El clustering es la forma más común del **aprendizaje no supervisado**. En el aprendizaje supervisado, el proceso de aprendizaje se lleva a cabo a través de entrenamiento controlado por un agente externo que indica la respuesta que debería generar la red a partir de una entrada determinada. Sin embargo, en el aprendizaje no supervisado, no se proporcionan estos ejemplos y son los propios algoritmos los que tienen que descubrir características de los datos y encontrar alguna manera de organizarlos.  
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
</br>


## Ejemplo con TensorFlow
Una vez entendido el funcionamiento de clustering con el método k-medias, pasamos a implementar un ejemplo con TensorFlow.
Para ello usaremos Python y la librería de TensorFlow, además de la librería NumPy para realizar algunos cálculos complejos y la librería MatPlotLib, para poder ver el resultado final de manera gráfica.
Para empezar, vamos a definir algunos valores que necesitará el algoritmo para poder realizar todos sus pasos:
   
   - `num_puntos` = número de puntos que tendrá la muestra en total
   - `num_clusters` = número de clusters en los que se dividirán los datos
   - `num_iteraciones` = número de veces que se repetirá el algoritmo

### i. Generación de datos y selección de centroides iniciales
Empezamos definiendo los puntos y los centroides de manera aleatoria.  
Los puntos serán de tipo constante porque se van a mantener en la misma posición durante todo el proceso. 
```python
puntos = tf.constant(np.random.uniform(0, 10, (num_puntos, 2)))
```  
Para crear puntos de manera aleatoria, se usa el método random.uniform de la librería NumPy, que usa una distribución uniforme con tres parámetros:
- El número más pequeño a generar
- El mayor número a generar
- Las dimensiones del número. En este caso, será de una matriz de num_puntos por dos dimensiones.


Los centroides, sin embargo, son de tipo variable porque se actualizan con cada iteración del algoritmo.
```python
centroides = tf.Variable(tf.slice(tf.random_shuffle(puntos), [0, 0], [num_clusters, -1]))
```

Para seleccionar centroides dentro de la muestra de los puntos generados de manera aleatoria, se usa el método `slice` con los siguientes parámetros.
- Los datos de los que se extraerán los centroides. Este método `tf.random_suffle(puntos)` a su vez “baraja” o mezcla los datos anteriormente obtenidos.
- El punto de partida.
- El tamaño del dato extraído. El -1 indica que el tamaño de esa dimensión se computa de manera que el tamaño total sea constante.


### ii. Cálculo de la distancia entre puntos y centroides
Para poder calcular la distancia, hay que hacer una resta elemento por elemento de los puntos y de los centroides que sean tensores de dos dimensiones. Si se imprimen las variables `puntos` y `centroides`, vemos que no son iguales.
```python
Tensor("Const:0", shape=(800, 2), dtype=float64)
<tf.Variable 'Variable:0' shape=(4, 2) dtype=float64_ref>
```
Debido a que los tensores tienen una forma diferente, lo que hay que hacer es expandirlos a tres dimensiones. Lo que hace esto es que el array más pequeño se "difunde" a través del más grande para que tengan formas compatibles y así poder operarse elemento por elemento.
Por ejemplo, un vector de 3 elementos pasaría a ser una matriz de 3x1.  
Para hacer esta expansión a una dimensión más, lo que se hace es ejecutar:
```python
puntos_expand = tf.expand_dims(puntos, 0)
centroides_expand = tf.expand_dims(centroides, 1)
```
Donde el primer parámetro es la variable a expandir y el segundo parámatro es la posición en la que se añadirá la nueva dimensión.  
Así, al volver a imprimir estas dos variables, el resultado será:
```python
Tensor("ExpandDims:0", shape=(1, 800, 2), dtype=float64)
Tensor("ExpandDims_1:0", shape=(4, 1, 2), dtype=float64)
```


Ahora se calcula la distancia entre centroides y puntos con la distancia euclídea mencionada anteriormente, y se obtiene la distancia mínima de todas las calculadas.
```python
distancias = tf.reduce_sum(tf.square(tf.subtract(puntos_expand, centroides_expand)), 2)
dist_minima = tf.argmin(distancias, 0)
```


### iii. Cálculo y actualización de los nuevos centroides
Pasamos a comparar cada cluster con el vector de etiquetas de un cluster, asignaremos los puntos a cada cluster y calcularemos los valores medios. Estas medias serán los nuevos centroides, así que habrá que actualizar la variable `centroides` con los nuevos valores obtenidos.
```python
lista_centroides = tf.dynamic_partition(puntos, tf.cast(dist_minima, tf.int32), num_clusters)
nuevos_centroides  = [tf.reduce_mean(punto, 0) for punto in lista_centroides]
centroides_actualizados = tf.assign(centroides, nuevos_centroides)
```
Lo que se hace es crear una lista mediante el método de partición dinámica, que indica qué es lo que se quiere dividir,  y el número de particiones que se quieren. En este caso, serían los puntos, la distancia mínima y el número de clusters, respectivamente.  
Después se calcula la media de estos puntos con el método `reduce_mean` por cada punto o dato en la lista. Al pasarle 0 como segundo parámetro, hace las operaciones con las coordenadas *x* por un lado y las operaciones con las coordenadas *y* por otro.
Por último, se asignan los centroides calculados a la variable `centroides`.



### iv. Ejecución del algoritmo

```python
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_iteraciones):
        [_, valores_centroides, valores_puntos, valores_asignaciones] = sess.run(
            [centroides_actualizados, centroides, puntos, dist_minima])

    print ("Centroides finales: \n{}".format(valores_centroides))
```


### v. Mostrar el resultado gráficamente
Por último se utilizan métodos de la librería MatPlotLib para mostrar el resultado final en un grafo.   
Primero se crea un grafo de dispersión para representar todos los puntos de la muestra, configurando tamaño, transparencia y colores.  
Después se dibujan los puntos donde están situados los centroides que ha devuelto el algoritmo en forma de cruces negras.
```python
plt.scatter(valores_puntos[:, 0], valores_puntos[:, 1], c=valores_asignaciones, s=40, alpha=1, cmap=plt.cm.rainbow)
plt.plot(valores_centroides[:, 0], valores_centroides[:, 1], 'kx', markersize=15, mew=2)
plt.show()
```


La siguiente imagen sería un posible resultado obtenido al indicar 800 como número total de puntos de la muestra, 4 como número de clusters y 500 como número de iteraciones.

![Posible resultado del ejemplo](/posible_resultado.png)
 
 Además, los centroides finales quedarían así:
 ```pyhton
 Centroides finales:
[[ 7.69324076  2.35547593]
 [ 2.52013846  2.6992541 ]
 [ 2.42233639  7.82168145]
 [ 7.23525408  7.67087173]]
 ```
