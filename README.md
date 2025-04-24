---

#  Clasificador de figuras por camara

Alumno: Emanuel Castillo Ibarra  
Asignatura: Sistemas de Visión Artificial  
Proyecto: Detección y clasificación de frutas por visión computacional

---

## ¿Que hace?

Este repositorio contiene una solución basada en redes neuronales convolucionales para identificar si tenemos con nosotros ciertas figuras, por el momento esto solo es capaz de diferencia entre circulo y triangulo

### Funcionalidades:

- Registro de datos personalizado: Permite capturar fotos de frutas directamente desde la cámara.
- Clasificación en vivo: Visualización instantánea de resultados desde la webcam.

---

## Requisitos

Para correr este proyecto necesitas tener instalados:

- Python 3.11 o superior
- TensorFlow y Keras (modelado de redes neuronales)
- NumPy (manipulación de datos numéricos)
- OpenCV (procesamiento de imágenes y video)


Instálar con:

```bash
pip install -r Requirements.txt
```

---

## Estructura del Proyecto

```plaintext
Convolutive_neuronal_network/
│
├── src/
│   ├──  Shape_classifier.py   
│   ├──  Trained_model.py
│   ├── Capture_Circulo.py          
│   └──  Capture_Triangulo.py    
│
├── Trained_model/
│   └── model.h5               
│
├── .gitignore
├── main.py                   
├── Requirements.txt        
└── README.md                  

---

### Guía Rápida

### 1. Clonar el repositorio
Abre una terminal y ejecuta:

```bash
git clone *direccion del repositorio*
```

### 2. Crear un entorno virtual
Para un manejo más limpio de las dependencias.

### 3. Instalar los requerimientos

```bash
pip install -r Requirements.txt
```

### 4. Ejecutar el flujo del proyecto

#### Capturar tus datos:

Asegúrate de tener buena luz y el objeto enfocado.

```bash
python main.py --step capture_Triangulo
python main.py --step capture_circulo
```

Teclear `s` para guardar y `q` para salir.

#### Entrenamiento del modelo:

```bash
python main.py --step train
```

Entrena la red neuronal y guarda el modelo en `Trained_model/`.

#### Clasificación en tiempo real:

```bash
python main.py --step run
```

El modelo detecta la figura en cámara si la confianza es del 70%  más.


## Logros

- Reconocimiento de frutas con interfaz visual en directo.
- Generación de dataset propio.

---