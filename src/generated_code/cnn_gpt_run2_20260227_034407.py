# Generado por: GPT
# Run: 2
# Fecha: 20260227_034407
# Dataset: LeLePhid
==================================================

A continuación, te proporciono un código completo en Python para una red neuronal convolucional (CNN) utilizando TensorFlow y Keras, que clasifica imágenes de hojas de limón en dos categorías: con áfidos y sin áfidos. Incluye data augmentation, early stopping y métricas de precisión y pérdida.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Definir rutas y parámetros
data_dir = '/content/Images'
img_height, img_width = 224, 224
batch_size = 32

# Preprocesamiento de datos y Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Usamos el 20% de los datos para validación
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Usar para entrenamiento
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Usar para validación
)

# Construcción del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Clasificación binaria
])

# Compilar el modelo
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Evaluación final
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")

# Guardar el modelo
model.save('lemon_leaf_classifier.h5')
```

### Explicación del Código:

1. **Preprocesamiento de Datos**: Utilizamos `ImageDataGenerator` para aplicar técnicas de aumento de datos, como rotaciones, desplazamientos de ancho/alto, y volteos horizontales. Esto ayuda a mejorar la generalización del modelo.

2. **Modelo CNN**: La red está compuesta por varias capas de convolución y agrupamiento (pooling) para extraer características, seguidas de una capa completamente conectada y una capa de salida con activación sigmoide para la clasificación binaria.

3. **Compilación y Entrenamiento**: Utilizamos `binary_crossentropy` como función de pérdida y `adam` como optimizador. Implementamos `EarlyStopping` para detener el entrenamiento si la pérdida de validación no mejora en 5 épocas consecutivas.

4. **Evaluación y Guardado**: Después del entrenamiento, evaluamos el modelo en el conjunto de validación y guardamos el modelo entrenado para su uso futuro.

Asegúrate de que el directorio `/content/Images` esté correctamente estructurado con subdirectorios para cada clase: uno para "con áfidos" y otro para "sin áfidos".