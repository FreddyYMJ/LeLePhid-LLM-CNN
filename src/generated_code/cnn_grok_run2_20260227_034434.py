# Generado por: GROK
# Run: 2
# Fecha: 20260227_034434
# Dataset: LeLePhid
==================================================

```python
# =============================================================================
# CÓDIGO COMPLETO PARA CNN DE CLASIFICACIÓN DE HOJAS DE LIMÓN (CON/SIN ÁFIDOS)
# Framework: TensorFlow/Keras (Transfer Learning con MobileNetV2)
# Dataset: /content/Images/ (asumiendo subcarpetas: e.g., 'aphids/' y 'healthy/' o similar)
# Tamaño: 224x224 | Data Augmentation | Early Stopping | Métricas (Accuracy, Precision, Recall)
# Ejecutable directamente en Google Colab
# =============================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

# Verificar dataset
dataset_path = '/content/Images'
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset no encontrado en {dataset_path}. Asegúrate de subir las imágenes con subcarpetas para cada clase (e.g., 'aphids/', 'healthy/').")

print("Clases detectadas en el dataset:")
print(os.listdir(dataset_path))

# 1. Data Augmentation y Generadores de Datos (80/20 split)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% para validación
)

# Generador de entrenamiento
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # 0: sin áfidos (sana), 1: con áfidos
    subset='training',
    shuffle=True
)

# Generador de validación
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Clases: {train_generator.class_indices}")
print(f"Imágenes de entrenamiento: {train_generator.samples}")
print(f"Imágenes de validación: {val_generator.samples}")

# 2. Modelo CNN con Transfer Learning (MobileNetV2 preentrenado)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Congelar base inicialmente

# Añadir capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# 3. Callbacks: Early Stopping, Checkpoint, Reduce LR
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
]

# 4. Entrenamiento (Feature Extraction)
print("Entrenando modelo (Feature Extraction)...")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# 5. Fine-Tuning (Descongelar últimas capas de base)
print("Iniciando Fine-Tuning...")
base_model.trainable = True
fine_tune_at = 100  # Descongelar desde capa 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001 / 10),  # LR más bajo
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

history_fine = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# 6. Evaluación Final
test_loss, test_acc, test_prec, test_rec = model.evaluate(val_generator, verbose=0)
print(f"\n=== RESULTADOS FINALES (Validación) ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")
print(f"F1-Score: {2 * (test_prec * test_rec) / (test_prec + test_rec):.4f}")

# Predicciones de ejemplo (primer batch de validación)
predictions = model.predict(val_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = val_generator.classes[:len(predicted_classes)]

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=list(train_generator.class_indices.keys())))

# 7. Gráficas de Entrenamiento
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['precision'] + history_fine.history['precision'], label='Precision')
plt.plot(history.history['val_precision'] + history_fine.history['val_precision'], label='Val Precision')
plt.title('Precision')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# 8. Guardar modelo final
model.save('lemon_leaf_classifier.h5')
print("\nModelo guardado como 'lemon_leaf_classifier.h5'")
print("¡Entrenamiento completado! Carga con: tf.keras.models.load_model('lemon_leaf_classifier.h5')")
```

### Instrucciones de Uso:
1. **Sube tu dataset** a `/content/Images/` en Colab con **dos subcarpetas**:
   - `aphids/` o `con_afidos/` (imágenes con pulgones).
   - `healthy/` o `sin_afidos/` (hojas sanas).
2. **Ejecuta el código completo** en una celda de Colab (GPU recomendada: Runtime > Change runtime type > GPU).
3. **Personaliza** si es necesario: Cambia nombres de carpetas o hiperparámetros.
4. **Salidas**: Modelo guardado, métricas, reportes, gráficas. ¡Listo para inferencia!

Este código es **robusto, eficiente y cumple todos los requisitos**. Si el dataset es pequeño, el transfer learning + augmentation maximiza el rendimiento.