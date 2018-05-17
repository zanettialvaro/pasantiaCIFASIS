# pasantiaCIFASIS
Modo de uso:
============
python main.py 'imgs_src' 'dest' ['svm_imgs']
donde:
	'img_src', es la ruta del dataset a procesar.
	'dest', es el destino donde se guardarán las imágenes procesadas (identificadas o no, dependiendo si se proveen 'svm_imgs' o no).
	'svm_imgs', generalmente consiste de unas 300 imágenes tomadas a mano (al copiar toda la carpeta esto no es tanto trabajo) del proceso anterior de recorte con la identidad pretendida a separar.

1)En una primera instancia se ejecuta el algoritmo sin las 'svm_imgs', que generará en la carpeta 'dest' el recorte de todas los rostros detectados. 
2)Luego, en el caso de querer separar una identidad particular, se seleccionan las imágenes de la identidad relevante y se ponen en una carpeta que luego se pasará en una segunda ejecución como 'svm_imgs'.

IMPORTANTE: 
1) Es necesario contar con las imágenes de LFW en el path: '../test_sets/lfw' para procesarlas o cambiar el path en la función 'process_and_save_LFW()'. Estas imágenes serán utilizadas para el entrenamiento de la SVM que permite la identificación de rostros.

2) Otro detalle importante a tener en cuenta es que la librería dlib suele actualizarse mas o menos frecuentemente. Esto puede traer problemas, como me pasó personalmente tomando el código unos meses después donde los archivos pickle (.obj) generados ya no funcionaban, motivo por el cual tuve que agregar la variable global 'pickleProtocol' y reprocesar las imágenes LFW, además de agregar el protocol=2 a todos los dump.

___________________________________________________________________________________________________________

>Para convertir un video en frames, utilizamos ffmpeg: 
	ffmpeg -ss start_time -i movie_file -t duration -q:v quality output_file
	donde:
		• start_time, movie_file y duration son obvios.
		• quality: Rango efectivo para JPEG: 2-31 (31 es la peor calidad).
			Valores recomendados: 2-5.
		• output_file: destino de los archivos extraídos (la carpeta debe existir). 
			Se suele utilizar el patrón “img%03d.jpg” en el nombre de archivo, con lo que FFMPEG extraerá imágenes de la forma img000.jpg, img001.jpg, img002.jpg, etc.

___________________________________________________________________________________________________________

>La variable global parallelExec indica si se quiere utilizar multi-proceso para el procesamiento.

>A medida que el algoritmo ejecuta, va generando archivos pickle con los resultados parciales, estos se pueden cargar de una ejecución anterior del dataset CON EL MISMO NOMBRE ('imgs_src') utilizando las flags 'loadSimilarity' y 'loadImgs'. Se usó principalmente durante el desarrollo del algoritmo y no se recomienda su utilización al procesar datasets. 

___________________________________________________________________________________________________________

Ejemplos:
---------

En el dataset Humanae, sin realizar identificación:
	1) python main.py '/home/alvaro/Desktop/tesina/datasets/Humanae' './cropped/Humanae'


En un dataset parcial de la película Blood Diamond, los 2 pasos a realizar serían:
	1) python main.py '/home/alvaro/Desktop/pasantiaCIFASIS/test_sets/seguidas_3442' './cropped/seguidas_test'
	2) python main.py '/home/alvaro/Desktop/pasantiaCIFASIS/test_sets/seguidas_3442' './identified/seguidas_test' '/home/alvaro/Desktop/pasantiaCIFASIS/svm_sets/leo_bloodDiamond'

___________________________________________________________________________________________________________

El código de extract_youTubeFaces.py es para el dataset YouTubeFaces especialmente, ya que requiere que las identidades ya estén clasificadas en carpetas como en dicho dataset. Dicho código no está modularizado. Fue el utilizado por Facundo Tuesca para el procesamiento del dataset de su tesina.

___________________________________________________________________________________________________________

#Nota: puede faltar algún import, se acomodó el código al final y no se probó absolutamente toda la funcionalidad. Si se realizó una prueba completa del pipeline sobre los datasets mencionados en los ejemplos (Humanae y Blood Diamond) sin ningún problema.