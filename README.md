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
Para ello usaremos Python y la librería de TensorFlow, además de la librería NumPy para realizar algún cálculo más complejo y la librería MatPlotLib, para poder ver el resultado final de manera gráfica.
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
```
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
```
Tensor("ExpandDims:0", shape=(1, 800, 2), dtype=float64)
Tensor("ExpandDims_1:0", shape=(4, 1, 2), dtype=float64)
```


Ahora se calcula la distancia entre centroides y puntos con la distancia euclídea mencionada anteriormente, obteniendo las distancias de cada punto con los centroides. Del resultado obtenido se escoge la distancia menor para cada punto y se obtiene la asignación de cada punto al número de cluster de cuyo centroide esté más cerca.
```python
distancias = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(puntos_expand, centroides_expand)), 2))
vector_dist_minimas = tf.argmin(distancias, 0)
```


### iii. Cálculo y actualización de los nuevos centroides
Pasamos a comparar cada cluster con el vector de etiquetas de un cluster, es decir, el último array obtenido. Asignaremos los puntos a cada cluster y calcularemos los valores medios. Estas medias serán los nuevos centroides, así que habrá que actualizar la variable `centroides` con los nuevos valores obtenidos.
```python
lista = tf.dynamic_partition(puntos, tf.cast(vector_dist_minimas, tf.int32), num_clusters)
nuevos_centroides  = [tf.reduce_mean(punto, 0) for punto in lista]

centroides_actualizados = tf.assign(centroides, nuevos_centroides)
```
Lo que se hace es crear una lista mediante el método de partición dinámica, al que se le pasa como parámetros qué es lo que se quiere dividir, el índice de la lista resultante en el que irá el elemento y el número de particiones deseadas. En este caso, los parámetros serían los puntos, el vector de etiquetas de cada punto y el número de clusters, respectivamente.  
Esta lista que se obtiene tiene tantas particiones bidimensionales como número de clusters.  
Para aclarar esto, supongamos que el número de clusters es 4 y que el número de puntos es 100. Si en el vector de asignaciones de distancias mínimas se indica un 2 en la sexta posición, significa que el punto de la sexta posición se colocaría en el elemento \[2] de la nueva lista. Así quedaría reflejado que este punto pertenece al tercer cluster. En el vector de asignaciones de distancia mínima, el resto de las posiciones que sean un 2 también irán a este tercer elemento o tercer cluster. De esta manera se irían repartiendo los elementos en los distintos clusters.  
Después se calcula la media de estos puntos con el método `reduce_mean` por cada punto o dato en la lista. Al pasarle 0 como segundo parámetro, hace las operaciones con las coordenadas *x* por un lado y las operaciones con las coordenadas *y* por otro.
Por último, se asignan los centroides calculados, `nuevos_centoides`, a la variable `centroides_actualizados`, de manera que quedan actualizados los centroides.



### iv. Ejecución del algoritmo
Se inicializan todas las variables y se inicia una sesión para poder ejecutar el algoritmo. Dentro de esta sesión, un bucle se ejecutará tantas veces como `num_iteraciones` se haya indicado al principio.
```python
    for i in range(num_iteraciones):
        [_, valores_centroides, valores_puntos, valores_distancias] = sesion.run(
            [centroides_actualizados, centroides, puntos, vector_dist_minimas])

    print ("Centroides finales:\n", valores_centroides)
```
Vemos que `sesion` está ejecutando cuatro de los tensores que habíamos definido antes:
- `sesion.run(centroides actualizados)` va a reasignar los nuevos centroides. Esta variable usa `nuevos_centroides`, que a su vez llama a la lista de particiones de `vector_dist_minimas`. Este array viene de `vector_distancias`, que contiene `centroides`. Una vez se tiene esta variable, se hacen todas las operaciones y se obtendrían los puntos de los nuevos centroides.  
Al iterar por primera vez, se utilizan los centroides iniciales. El resto de iteraciones, toma los últimos centroides obtenidos y los recalcula. De esta manera, la variable `centroides` va cambiando en cada iteración.  
El símbolo `-` indica que este resultado se guardará en otra variable y se ignora.
- Al llamar a `centroides_actualizados`, la variable `centroides` ha sido reasignada con los nuevos puntos que servirán como centro de los clusters. El resultado de esta reasignación con `sesion.run(centroides)` es `valores_centroides`.
- `sesion.run(puntos)` son los puntos de la muestra. Se mantienen iguales durante todo el algoritmo, de ahí que sean de tipo constante.
- `sesion.run(vector_dist_minimas)` actualiza el vector de distancias mínimas con los nuevos centroides calculados. Se guarda en `valores_distancia`, que especifica el número de cluster al que pertenece cada punto de la muestra después de haber recalculado los centroides.

Al terminar de iterar, se imprime el resultado, que son las coordenadas de los centroides finales.



### v. Mostrar el resultado gráficamente
Por último se utilizan métodos de la librería MatPlotLib para mostrar el resultado final en un grafo.   
Primero se crea un grafo de dispersión para representar todos los puntos de la muestra, configurando los valores en los ejes, los valores a representar y, de manera opcional, el tamaño, la transparencia y los colores de los puntos.  
Después se dibujan los puntos donde están situados los centroides que ha devuelto el algoritmo, especificando de la misma manera los ejes y configurando para que aparezcan estos puntos en forma de cruces negras.
```python
plt.scatter(valores_puntos[:, 0], valores_puntos[:, 1], c=valores_distancias, s=40, alpha=1, cmap=plt.cm.rainbow)
plt.plot(valores_centroides[:, 0], valores_centroides[:, 1], 'kx', markersize=15, mew=2)
plt.show()
```


La siguiente imagen sería un posible resultado obtenido al indicar 800 como número total de puntos de la muestra, 4 como número de clusters y 500 como número de iteraciones.

![Posible resultado del ejemplo](/posible_resultado.png)
 
 Además, los centroides finales quedarían así:
 ```pyhton
Centroides finales:
 [[ 7.81670316  7.57256505]
 [ 2.57910718  7.19336677]
 [ 2.38124039  2.17226728]
 [ 7.18812771  2.52348183]]
 ```
