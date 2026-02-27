# Generado por: GEMINI
# Run: 1
# Fecha: 20260227_034242
# Dataset: LeLePhid
==================================================

¡Claro que sí! Aquí tienes un código Python completo y ejecutable en Google Colab para entrenar una CNN que clasifique hojas de limón con y sin áfidos, utilizando **TensorFlow y Keras**.

Este script incluye:
- Carga de datos desde una estructura de carpetas.
- Aumento de datos (Data Augmentation) para mejorar la robustez.
- Uso de **Transfer Learning** con MobileNetV2, una técnica muy efectiva para datasets pequeños.
- Callback de **Early Stopping** para evitar el sobreajuste y optimizar el tiempo de entrenamiento.
- Evaluación completa con métricas como **precisión, recall, F1-score** y una **matriz de confusión**.

---

### **Instrucciones Previas (¡Muy Importante!)**

1.  **Abre un nuevo notebook en Google Colab** ([colab.research.google.com](https://colab.research.google.com)).
2.  **Sube tus imágenes**. En el panel de la izquierda (icono de carpeta), crea la siguiente estructura de carpetas y sube tus imágenes a las carpetas correspondientes:
    ```
    /content/
    └── Images/
        ├── Con_afidos/
        │   ├── imagen1.jpg
        │   ├── imagen2.jpg
        │   └── ...
        └── Sin_afidos/
            ├── imagen_sana1.jpg
            ├── imagen_sana2.jpg
            └── ...
    ```
3.  **Copia y pega el siguiente código** en una única celda de tu notebook y ejecútalo.

---

### **Código Completo para la CNN en TensorFlow/Keras**

```python
# ==============================================================================
# 1. PREPARACIÓN DEL ENTORNO E IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow Version:", tf.__version__)

# Para asegurar que la ejecución sea reproducible
tf.random.set_seed(42)
np.random.seed(42)

# ==============================================================================
# 2. CONFIGURACIÓN DE PARÁMETROS Y CARGA DE DATOS
# ==============================================================================
# --- Parámetros Principales ---
DATA_DIR = '/content/Images'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50 # Número máximo de épocas, EarlyStopping decidirá cuándo parar

# --- Verificación del Directorio ---
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print("¡ERROR! El directorio '/content/Images' no existe o está vacío.")
    print("Por favor, sigue las instrucciones para subir tus imágenes.")
    # Creando una estructura de ejemplo para que el script no falle
    print("Creando una estructura de datos de ejemplo...")
    os.makedirs(f"{DATA_DIR}/Con_afidos", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/Sin_afidos", exist_ok=True)
    # Crear imágenes falsas (placeholders)
    tf.keras.utils.save_img(f"{DATA_DIR}/Con_afidos/fake1.png", np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3) * 255)
    tf.keras.utils.save_img(f"{DATA_DIR}/Con_afidos/fake2.png", np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3) * 255)
    tf.keras.utils.save_img(f"{DATA_DIR}/Sin_afidos/fake1.png", np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3) * 255)
    tf.keras.utils.save_img(f"{DATA_DIR}/Sin_afidos/fake2.png", np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3) * 255)
    print("Se ha creado un dataset de ejemplo. Ejecuta de nuevo con tus propias imágenes.")


# --- Carga de Datos ---
# Keras infiere las clases (0 y 1) a partir de los nombres de las carpetas
print("\nCargando dataset...")
full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='binary', # Para clasificación binaria
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2, # Usaremos un 20% para validación y test
    subset='training'
)

validation_dataset_temp = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

# Nombres de las clases (Keras las ordena alfabéticamente)
class_names = full_dataset.class_names
print(f"Clases encontradas: {class_names}")

# --- División en Validación y Test (10% y 10%) ---
val_batches = tf.data.experimental.cardinality(validation_dataset_temp)
test_dataset = validation_dataset_temp.take(val_batches // 2)
validation_dataset = validation_dataset_temp.skip(val_batches // 2)

print(f"Lotes para entrenamiento: {tf.data.experimental.cardinality(full_dataset)}")
print(f"Lotes para validación: {tf.data.experimental.cardinality(validation_dataset)}")
print(f"Lotes para test: {tf.data.experimental.cardinality(test_dataset)}")


# --- Optimización del Rendimiento del Dataset ---
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = full_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================================================================
# 3. DATA AUGMENTATION (AUMENTO DE DATOS)
# ==============================================================================
data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ],
    name="data_augmentation",
)

# ==============================================================================
# 4. CONSTRUCCIÓN DEL MODELO CNN (USANDO TRANSFER LEARNING)
# ==============================================================================
# --- Carga del Modelo Base Pre-entrenado (MobileNetV2) ---
# Usamos un modelo pre-entrenado en ImageNet que ya sabe extraer características de imágenes.
# `include_top=False` para no incluir la capa de clasificación original.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# --- Congelamos el Modelo Base ---
# No queremos re-entrenar las capas que ya han aprendido características útiles.
base_model.trainable = False

# --- Construcción del Modelo Final ---
# Añadimos nuestras propias capas de clasificación encima del modelo base.
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)  # Aplicamos el aumento de datos
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Preprocesado específico de MobileNetV2
x = base_model(x, training=False) # Ponemos el modelo base en modo de inferencia
x = layers.GlobalAveragePooling2D()(x) # Agregamos las características en un vector
x = layers.Dropout(0.2)(x) # Capa de regularización para evitar sobreajuste
# Capa de salida con una neurona y activación sigmoide para clasificación binaria
outputs = layers.Dense(1, activation='sigmoid')(x) 

model = keras.Model(inputs, outputs)

# --- Compilación del Modelo ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# --- Resumen del Modelo ---
print("\nResumen del modelo:")
model.summary()

# ==============================================================================
# 5. ENTRENAMIENTO DEL MODELO
# ==============================================================================
# --- Callback de Early Stopping ---
# Detiene el entrenamiento si la pérdida de validación no mejora tras 'patience' épocas.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # Número de épocas a esperar antes de detenerse
    verbose=1,
    restore_best_weights=True # Restaura los pesos del modelo de la mejor época
)

print("\nIniciando el entrenamiento...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)
print("Entrenamiento finalizado.")

# ==============================================================================
# 6. EVALUACIÓN DEL MODELO
# ==============================================================================
# --- Visualización de la Curva de Aprendizaje ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión de Entrenamiento')
plt.plot(val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')

plt.tight_layout()
plt.show()

# --- Evaluación Final con el Conjunto de Test ---
print("\nEvaluando el modelo con el conjunto de datos de test...")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"\nPérdida en el conjunto de test: {test_loss:.4f}")
print(f"Precisión en el conjunto de test: {test_accuracy:.4f}")

# --- Métricas Detalladas y Matriz de Confusión ---
# Obtenemos las etiquetas verdaderas y las predicciones del conjunto de test
y_true = []
y_pred_probs = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy().flatten())
    y_pred_probs.extend(model.predict(images, verbose=0).flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
# Convertimos probabilidades a clases (0 o 1) usando un umbral de 0.5
y_pred = (y_pred_probs > 0.5).astype(int)

# --- Reporte de Clasificación ---
print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names))

# --- Matriz de Confusión ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.ylabel('Clase Verdadera')
plt.xlabel('Clase Predicha')
plt.show()

# ==============================================================================
# 7. VISUALIZACIÓN DE PREDICCIONES (OPCIONAL)
# ==============================================================================
print("\nMostrando algunas predicciones del conjunto de test:")
plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        # Realizamos la predicción para esta imagen
        img_array = tf.expand_dims(images[i], 0)
        prediction_prob = model.predict(img_array, verbose=0)[0][0]
        predicted_class_index = int(prediction_prob > 0.5)
        predicted_class_name = class_names[predicted_class_index]
        
        true_class_name = class_names[int(labels[i])]
        
        # Color del título: verde si es correcta, rojo si es incorrecta
        title_color = 'green' if predicted_class_name == true_class_name else 'red'
        
        plt.title(
            f"Pred: {predicted_class_name} ({prediction_prob:.2f})\nReal: {true_class_name}",
            color=title_color
        )
        plt.axis("off")
plt.tight_layout()
plt.show()
```

### **Explicación del Código Paso a Paso**

1.  **Preparación del Entorno:** Se importan todas las librerías necesarias de TensorFlow, Matplotlib, Scikit-learn, etc.
2.  **Configuración y Carga de Datos:**
    *   Se definen parámetros clave como la ruta del dataset, el tamaño de las imágenes y el tamaño del lote.
    *   `image_dataset_from_directory` es una función muy útil de Keras que carga las imágenes desde las carpetas, las redimensiona, crea lotes y asigna etiquetas automáticamente (`0` para la primera carpeta en orden alfabético, `1` para la segunda).
    *   El dataset se divide en tres conjuntos: **entrenamiento (80%)**, **validación (10%)** y **test (10%)**. El conjunto de test es crucial para obtener una evaluación final imparcial del modelo.
3.  **Data Augmentation:**
    *   Se crea una "capa" de aumento de datos que aplicará transformaciones aleatorias (giros, zoom, cambios de contraste) a las imágenes de entrenamiento *durante* el entrenamiento. Esto ayuda al modelo a generalizar mejor y a no memorizar las imágenes exactas.
4.  **Construcción del Modelo (Transfer Learning):**
    *   En lugar de crear una CNN desde cero, utilizamos **MobileNetV2**, un modelo muy potente pre-entrenado con millones de imágenes (del dataset ImageNet).
    *   **Congelamos** sus capas (`base_model.trainable = False`) para aprovechar el conocimiento que ya tiene sobre cómo detectar bordes, texturas y formas.
    *   Añadimos nuestras propias capas al final: una capa `GlobalAveragePooling2D` para aplanar las características, una capa `Dropout` para regularización y la capa final `Dense` con activación `sigmoid` para la clasificación binaria.
5.  **Entrenamiento:**
    *   Se compila el modelo especificando el optimizador (`Adam`), la función de pérdida (`BinaryCrossentropy` para problemas de dos clases) y la métrica a seguir (`accuracy`).
    *   Se define el `EarlyStopping`. Esta es una práctica recomendada: monitoriza la pérdida en el conjunto de validación (`val_loss`) y si no mejora después de 10 épocas (`patience=10`), detiene el entrenamiento para evitar el sobreajuste.
    *   Se llama a `model.fit()` para iniciar el proceso de entrenamiento.
6.  **Evaluación:**
    *   Se grafican las curvas de precisión y pérdida para visualizar cómo aprendió el modelo y detectar problemas como el sobreajuste.
    *   Se utiliza `model.evaluate()` en el conjunto de test (que el modelo nunca ha visto) para obtener la precisión final.
    *   Se genera un **reporte de clasificación** con métricas clave (precisión, recall, f1-score) y una **matriz de confusión** para ver visualmente cuántas imágenes de cada clase se clasificaron correcta e incorrectamente.
7.  **Visualización de Predicciones:**
    *   Como paso final, se toman algunas imágenes del conjunto de test y se muestra la predicción del modelo junto con la etiqueta real para una inspección visual.