import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

def build_model(input_shape=(100, 100, 3), num_classes=2):
    model = Sequential()
    
    # Capa convolucional con 32 filtros
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda capa convolucional con 64 filtros
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Tercera capa convolucional con 128 filtros
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Aplanar la salida antes de pasar a la capa densa
    model.add(Flatten())

    # Capa densa de 128 unidades
    model.add(Dense(128, activation='relu'))

    # Capa densa de 64 unidades
    model.add(Dense(64, activation='relu'))

    # Capa de salida con softmax para clasificación en 2 clases
    model.add(Dense(num_classes, activation='softmax'))

    return model


def train_model():
    data_dir = "dataset"
    model_save_path = "Trained_model/model.h5"

    img_size = (100, 100)
    batch_size = 16

    # Generador de imágenes con normalización y división en conjunto de entrenamiento/validación
    data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Generador de datos de entrenamiento
    train_generator = data_generator.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Generador de datos de validación
    validation_generator = data_generator.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Construir el modelo
    model = build_model(input_shape=(100, 100, 3), num_classes=2)

    # Compilación del modelo con optimizador Adam
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenamiento del modelo
    model.fit(train_generator, validation_data=validation_generator, epochs=200)

    # Guardar el modelo entrenado
    model.save(model_save_path)
    print(f"Modelo guardado en {model_save_path}")
