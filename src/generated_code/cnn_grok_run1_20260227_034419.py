# Generado por: GROK
# Run: 1
# Fecha: 20260227_034419
# Dataset: LeLePhid
==================================================

```python
# Código Python COMPLETO para CNN de clasificación de hojas de limón con/sin áfidos
# Framework: TensorFlow/Keras (elegido por simplicidad y eficiencia en prototipado)
# Asume estructura de dataset en /content/Images:
# - /content/Images/con_afidos/   (imágenes con áfidos)
# - /content/Images/sin_afidos/   (imágenes sanas)
# Ejecutable directamente en Google Colab.
# Incluye: Data Augmentation, Transfer Learning (MobileNetV2), Early Stopping, ReduceLROnPlateau,
# Métricas (accuracy, precision, recall), Gráficos de entrenamiento, Guardado de modelo.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = '/content/Images'
CLASSES = ['sin_afidos', 'con_afidos']  # Orden: 0=sana, 1=áfidos

# 1. Data Augmentation y Generadores de Datos
# Train: con augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80% train, 20% val
)

# Validation: solo rescale (sin augmentation)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # 0=sin_afidos (sana), 1=con_afidos
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# 2. Modelo CNN con Transfer Learning (MobileNetV2 preentrenado en ImageNet)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False  # Congelar base inicialmente

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compilación
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

print("Modelo resumido:")
model.summary()

# 4. Callbacks: Early Stopping, Reduce LR, Checkpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('/content/lemon_aphids_model_best.h5', monitor='val_accuracy', save_best_only=True)
]

# 5. Entrenamiento
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 6. Fine-tuning (opcional: descongelar top layers)
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20  # Fine-tune últimas 20 layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # LR más bajo
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=20,  # Menos epochs para fine-tuning
    callbacks=callbacks
)

# 7. Guardar modelo final
model.save('/content/lemon_aphids_model_final.h5')

# 8. Evaluación y Visualización
# Gráficos de entrenamiento
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'])
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])

plt.subplot(1, 3, 3)
plt.plot(history.history['precision'] + history_fine.history['precision'])
plt.plot(history.history['val_precision'] + history_fine.history['val_precision'])
plt.title('Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])
plt.tight_layout()
plt.show()

# Predicciones en validación
val_preds = model.predict(val_generator)
val_preds_binary = (val_preds > 0.5).astype(int).flatten()
val_labels = val_generator.classes[:len(val_preds_binary)]

# Reporte de métricas
print("\nClassification Report:")
print(classification_report(val_labels, val_preds_binary, target_names=CLASSES))

# Matriz de confusión
cm = confusion_matrix(val_labels, val_preds_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("¡Entrenamiento completado! Modelos guardados en /content/")
print(f"Clases mapeadas: {train_generator.class_indices}")
```

### Instrucciones de uso en Google Colab:
1. Sube tu dataset a `/content/Images/` con subcarpetas `con_afidos/` y `sin_afidos/` (imágenes JPG/PNG).
2. Ejecuta el código completo (¡es autónomo!).
3. Monitorea métricas: accuracy, precision, recall.
4. Modelos guardados: `lemon_aphids_model_best.h5` y `lemon_aphids_model_final.h5`.

Este código es **robusto, eficiente y listo para producción**. Usa transfer learning para pocos datos, augmentation para generalización, y callbacks para evitar overfitting. ¡Ajusta hiperparámetros si necesitas!