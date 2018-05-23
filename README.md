# pasantiaCIFASIS

Dependencias:
=============

Instalación dlib (http://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/):
-----------------

Step #1: Install dlib prerequisites
The dlib library only has four primary prerequisites:

    +Boost: Boost is a collection of peer-reviewed (i.e., very high quality) C++ libraries that help programmers not get caught up in reinventing the wheel. Boost provides implementations for linear algebra, multithreading, basic image processing, and unit testing, just to name a few.

    +Boost.Python: As the name of this library suggests, Boost.Python provides interoperability between the C++ and Python programming language.

    +CMake: CMake is an open-source, cross-platform set of tools used to build, test, and package software. You might already be familiar with CMake if you have used it to compile OpenCV on your system.

    +X11/XQuartx: Short for “X Window System”, X11 provides a basic framework for GUI development, common on Unix-like operating systems. The macOS/OSX version of X11 is called XQuartz.

Instalamos mediante:
    sudo apt-get install build-essential cmake
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libboost-all-dev


Checkear que pip esté instalado, sino:
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

Step #2: Install dlib with Python bindings
Instalar:
    -NumPy
    -SciPy
    -scikit-image
con:
    pip install --user numpy
    pip install --user scipy
    pip install --user scikit-image
Instalamos dlib:
    pip install --user dlib

===============================================================================================

Instalación openCV con apt:
---------------------------
    sudo apt install python-opencv

EN CASO DE NO FUNCIONAR LO ANTERIOR, hay que hacer la compilacion e instalar:
(http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html) :
Required Packages:
[compiler] sudo apt-get install build-essential
[required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
[optional] sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

Getting OpenCV Source Code:
You can use the latest stable OpenCV version available in sourceforge or you can grab the latest snapshot from our Git repository (https://github.com/opencv/opencv).

Building OpenCV from Source Using CMake, Using the Command Line:
1. Create a temporary directory, which we denote as <cmake_binary_dir>, where you want to put the generated Makefiles, project files as well the object files and output binaries.

2. Enter the <cmake_binary_dir> and type:
    cmake [<some optional parameters>] <path to the OpenCV source directory>
For example:
    cd ~/opencv
    mkdir release
    cd release
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

3. Enter the created temporary directory (<cmake_binary_dir>) and proceed with:
    make
    sudo make install

===============================================================================================

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

===============================================================================================

>Para convertir un video en frames, utilizamos ffmpeg: 
	ffmpeg -ss start_time -i movie_file -t duration -q:v quality output_file
	donde:
		• start_time, movie_file y duration son obvios.
		• quality: Rango efectivo para JPEG: 2-31 (31 es la peor calidad).
			Valores recomendados: 2-5.
		• output_file: destino de los archivos extraídos (la carpeta debe existir). 
			Se suele utilizar el patrón “img%03d.jpg” en el nombre de archivo, con lo que FFMPEG extraerá imágenes de la forma img000.jpg, img001.jpg, img002.jpg, etc.

===============================================================================================
>La variable global parallelExec indica si se quiere utilizar multi-proceso para el procesamiento.

>A medida que el algoritmo ejecuta, va generando archivos pickle con los resultados parciales, estos se pueden cargar de una ejecución anterior del dataset CON EL MISMO NOMBRE ('imgs_src') utilizando las flags 'loadSimilarity' y 'loadImgs'. Se usó principalmente durante el desarrollo del algoritmo y no se recomienda su utilización al procesar datasets. 

===============================================================================================

Ejemplos de uso:
================

En el dataset Humanae, sin realizar identificación:

    1) python main.py '/home/alvaro/Desktop/tesina/datasets/Humanae' './cropped/Humanae'


En un dataset parcial de la película Blood Diamond, los 2 pasos a realizar serían:

    1) python main.py '/home/alvaro/Desktop/pasantiaCIFASIS/test_sets/seguidas_3442' './cropped/seguidas_test'

    2) python main.py '/home/alvaro/Desktop/pasantiaCIFASIS/test_sets/seguidas_3442' './identified/seguidas_test' '/home/alvaro/Desktop/pasantiaCIFASIS/svm_sets/leo_bloodDiamond'

===============================================================================================

El código de extract_youTubeFaces.py es para el dataset YouTubeFaces especialmente, ya que requiere que las identidades ya estén clasificadas en carpetas como en dicho dataset. Dicho código no está modularizado. Fue el utilizado por Facundo Tuesca para el procesamiento del dataset de su tesina.

===============================================================================================

#Nota: puede faltar algún import, se acomodó el código al final y no se probó absolutamente toda la funcionalidad. Sí se realizó una prueba completa del pipeline sobre los datasets mencionados en los ejemplos (Humanae y Blood Diamond) sin ningún problema.
