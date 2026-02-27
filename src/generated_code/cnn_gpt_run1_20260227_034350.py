# Generado por: GPT
# Run: 1
# Fecha: 20260227_034350
# Dataset: LeLePhid
==================================================

Para este caso, vamos a utilizar TensorFlow y Keras para construir una CNN que clasifique imágenes de hojas de limón en dos categorías: con áfidos y sin áfidos. El código incluirá técnicas de aumento de datos, early stopping y cálculo de métricas de precisión. Aquí tienes el código completo:

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

# Rutas del dataset
train_dir = '/content/Images/train'
validation_dir = '/content/Images/validation'

# Generador de imágenes con aumento de datos para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador de imágenes para el conjunto de validación
validation_datagen = ImageDataGenerator(rescale=1./255)

# Cargar imágenes del directorio de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Cargar imágenes del directorio de validación
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Definición del modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilación del modelo
model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

# Callbacks para el entrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=[early_stopping, checkpoint]
)

# Evaluación del modelo
loss, accuracy, precision, recall = model.evaluate(validation_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
```

### Notas importantes:
- El directorio `/content/Images` debe tener dos subdirectorios: `train` y `validation`, cada uno con dos subdirectorios: `con_afidos` y `sin_afidos`.
- Asegúrate de que las imágenes en las carpetas estén correctamente clasificadas antes de entrenar el modelo.
- Este código utiliza `ImageDataGenerator` para aumentar los datos de entrenamiento, lo que ayuda a mejorar la robustez del modelo.
- El modelo se guarda como `best_model.h5` cuando se alcanza la mejor precisión en el conjunto de validación.
- Se emplean métricas de precisión y recall para evaluar el rendimiento del modelo, además de la exactitud estándar.

Asegúrate de ajustar las rutas y parámetros según tus necesidades específicas y la estructura de tus datos.