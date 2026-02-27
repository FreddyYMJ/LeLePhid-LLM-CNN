# Generado por: GEMINI
# Run: 2
# Fecha: 20260227_034334
# Dataset: LeLePhid
==================================================

¡Claro que sí! Aquí tienes un código Python completo y ejecutable en Google Colab para entrenar una Red Neuronal Convolucional (CNN) con TensorFlow y Keras que clasifica hojas de limón.

Este script incluye todo lo solicitado:
*   Uso de un modelo pre-entrenado (MobileNetV2) para transfer learning, lo cual es muy eficaz para datasets pequeños.
*   Carga de datos desde la estructura de carpetas especificada.
*   Aumento de datos (Data Augmentation) para mejorar la robustez.
*   Parada temprana (Early Stopping) para evitar el sobreajuste.
*   Métricas de evaluación detalladas (accuracy, loss, matriz de confusión y reporte de clasificación).
*   Un ejemplo de cómo predecir una nueva imagen.

**Importante:** Para que el código sea directamente ejecutable, he incluido una primera sección que **crea un dataset de imágenes falsas**. Cuando uses tus propias imágenes, simplemente **borra o comenta esa primera sección** y sube tu carpeta `Images` a `/content/`.

---

### Código Completo para la Clasificación de Hojas de Limón

Puedes copiar y pegar este código directamente en una celda de un notebook de Google Colab.

```python
# -*- coding: utf-8 -*-

#==============================================================================
# PARTE 0: IMPORTACIONES Y CONFIGURACIÓN
#==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TensorFlow Version:", tf.__version__)

#==============================================================================
# PARTE 1: CREACIÓN DE UN DATASET DE EJEMPLO (PARA PRUEBAS)
#==============================================================================
# --- ¡¡ATENCIÓN!! ---
# ESTA SECCIÓN CREA IMÁGENES FALSAS PARA QUE EL CÓDIGO SEA EJECUTABLE.
# CUANDO TENGAS TU PROPIO DATASET, BORRA O COMENTA TODA ESTA SECCIÓN
# Y SUBE TUS IMÁGENES A /content/Images/Con_afidos y /content/Images/Sin_afidos

def create_dummy_dataset(base_path='/content/Images', num_images_per_class=50):
    print(f"Creando dataset de ejemplo en '{base_path}'...")
    if os.path.exists(base_path):
        import shutil
        shutil.rmtree(base_path)
    
    # Clases
    class_names = ['Con_afidos', 'Sin_afidos']
    
    for class_name in class_names:
        class_path = os.path.join(base_path, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        for i in range(num_images_per_class):
            # Crear una imagen base verde (simulando una hoja)
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            img_array[:, :, 1] = np.random.randint(100, 200, size=(224, 224)) # Componente verde
            
            # Si es 'Con_afidos', añadir puntos negros (simulando pulgones)
            if class_name == 'Con_afidos':
                num_afidos = np.random.randint(10, 50)
                for _ in range(num_afidos):
                    x, y = np.random.randint(0, 224, size=2)
                    size = np.random.randint(2, 5)
                    img_array[x:x+size, y:y+size, :] = np.random.randint(0, 50) # Puntos oscuros

            img = PIL.Image.fromarray(img_array)
            img.save(os.path.join(class_path, f'dummy_image_{i}.png'))
            
    print("Dataset de ejemplo creado con éxito.")
    print(f"Total de imágenes en '{os.path.join(base_path, class_names[0])}': {len(os.listdir(os.path.join(base_path, class_names[0])))}")
    print(f"Total de imágenes en '{os.path.join(base_path, class_names[1])}': {len(os.listdir(os.path.join(base_path, class_names[1])))}")

# Llamar a la función para crear el dataset
create_dummy_dataset()

#==============================================================================
# PARTE 2: DEFINICIÓN DE PARÁMETROS Y CARGA DE DATOS
#==============================================================================
# Parámetros
DATA_DIR = '/content/Images'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 25 # Aumentado para dar tiempo al early stopping

# Cargar los datos usando la utilidad de Keras
# Divide los datos en 80% para entrenamiento y 20% para validación
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# Obtener los nombres de las clases
class_names = train_ds.class_names
print("Clases encontradas:", class_names)
# Asignamos nombres más descriptivos para los gráficos
class_names_map = { 'Con_afidos': 'Con Áfidos', 'Sin_afidos': 'Sin Áfidos (Sana)' }
mapped_class_names = [class_names_map[name] for name in class_names]

# Visualizar algunas imágenes del dataset de entrenamiento
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(mapped_class_names[labels[i]])
    plt.axis("off")
plt.suptitle("Muestra del Dataset de Entrenamiento", fontsize=16)
plt.show()

# Optimizar el rendimiento del pipeline de datos
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#==============================================================================
# PARTE 3: AUMENTO DE DATOS (DATA AUGMENTATION)
#==============================================================================
data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal_and_vertical", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
  ],
  name="data_augmentation"
)

#==============================================================================
# PARTE 4: CONSTRUCCIÓN DEL MODELO (TRANSFER LEARNING CON MOBILENETV2)
#==============================================================================
# Pre-procesamiento específico de MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Cargar el modelo base pre-entrenado en ImageNet sin la capa de clasificación final
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, # No incluir la capa final
                                               weights='imagenet')

# Congelar el modelo base para no re-entrenar sus pesos
base_model.trainable = False

# Construir el modelo final
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)       # 1. Aplicar aumento de datos
x = preprocess_input(x)             # 2. Pre-procesar para MobileNetV2
x = base_model(x, training=False)   # 3. Pasar por el modelo base (en modo inferencia)
x = layers.GlobalAveragePooling2D()(x) # 4. Reducir dimensiones
x = layers.Dropout(0.2)(x)             # 5. Regularización con Dropout
# 6. Capa de salida. Una neurona con activación sigmoide para clasificación binaria.
outputs = layers.Dense(1, activation='sigmoid')(x) 
model = tf.keras.Model(inputs, outputs)

#==============================================================================
# PARTE 5: COMPILACIÓN DEL MODELO
#==============================================================================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

#==============================================================================
# PARTE 6: ENTRENAMIENTO DEL MODELO
#==============================================================================
# Definir el callback de Early Stopping
# Monitorea la pérdida en el conjunto de validación (`val_loss`)
# y detiene el entrenamiento si no mejora después de `patience` épocas.
early_stopping_callback = EarlyStopping(
    monitor='val_loss', 
    patience=5,          # Número de épocas sin mejora antes de detenerse
    verbose=1, 
    restore_best_weights=True # Restaura los pesos del modelo de la mejor época
)

print("\nIniciando entrenamiento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping_callback]
)
print("Entrenamiento finalizado.")

#==============================================================================
# PARTE 7: EVALUACIÓN DEL MODELO
#==============================================================================
# --- 7.1: Gráficas de Accuracy y Loss ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Accuracy de Entrenamiento')
plt.plot(val_acc, label='Accuracy de Validación')
plt.legend(loc='lower right')
plt.title('Accuracy de Entrenamiento y Validación')
plt.xlabel('Época')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.suptitle('Métricas de Entrenamiento', fontsize=16)
plt.show()

# --- 7.2: Métricas Detalladas (Matriz de Confusión y Reporte) ---
print("\nEvaluando el modelo con el conjunto de validación...")

# Obtener predicciones y etiquetas verdaderas
y_pred_probs = model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

y_true = []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=mapped_class_names))

# Matriz de Confusión
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=mapped_class_names, yticklabels=mapped_class_names)
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()

#==============================================================================
# PARTE 8: PREDICCIÓN CON UNA NUEVA IMAGEN
#==============================================================================
def predict_new_image(image_path):
    # Cargar la imagen
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
    # Convertir a array y añadir una dimensión de batch
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Realizar la predicción
    predictions = model.predict(img_array)
    score = predictions[0][0]

    # Interpretar el resultado
    predicted_class_index = 0 if score > 0.5 else 1
    predicted_class_name = mapped_class_names[predicted_class_index]
    confidence = score if score > 0.5 else 1 - score

    # Mostrar la imagen y el resultado
    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class_name}\nConfianza: {confidence:.2%}")
    plt.axis('off')
    plt.show()

# Probar con una imagen del dataset de ejemplo
# NOTA: Ajusta la ruta si usas tu propio dataset
try:
    example_image_path = os.path.join(DATA_DIR, 'Con_afidos', 'dummy_image_0.png')
    if os.path.exists(example_image_path):
        print("\n--- Ejemplo de Predicción con una Imagen de la clase 'Con Áfidos' ---")
        predict_new_image(example_image_path)

    example_image_path = os.path.join(DATA_DIR, 'Sin_afidos', 'dummy_image_0.png')
    if os.path.exists(example_image_path):
        print("\n--- Ejemplo de Predicción con una Imagen de la clase 'Sin Áfidos' ---")
        predict_new_image(example_image_path)
except Exception as e:
    print(f"No se pudo realizar la predicción de ejemplo: {e}")
```

### Cómo Utilizar este Código

1.  **Abre Google Colab:** Ve a [colab.research.google.com](https://colab.research.google.com) y crea un nuevo notebook.
2.  **Configura el Acelerador de Hardware:** En el menú, ve a `Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` y selecciona `GPU` como acelerador de hardware. Esto acelerará enormemente el entrenamiento.
3.  **Ejecución Inicial (con datos de prueba):**
    *   Copia y pega todo el código en una celda del notebook.
    *   Ejecuta la celda. El script creará primero un dataset de imágenes falsas en `/content/Images/` y luego procederá con el entrenamiento y la evaluación. Esto te permite verificar que todo el flujo funciona correctamente.
4.  **Ejecución con tus Propias Imágenes:**
    *   **Comenta o elimina la "PARTE 1"** del código, la que contiene la función `create_dummy_dataset()`.
    *   En el panel de la izquierda de Colab, haz clic en el icono de la carpeta para ver los archivos.
    *   Sube tu carpeta `Images`, que debe contener dos subcarpetas: `Con_afidos` y `Sin_afidos`. La estructura final debe ser:
        ```
        /content/
        └── Images/
            ├── Con_afidos/
            │   ├── img1.jpg
            │   ├── img2.png
            │   └── ...
            └── Sin_afidos/
                ├── sano1.jpg
                ├── sano2.png
                └── ...
        ```
    *   Vuelve a ejecutar la celda. Ahora el código usará tus imágenes reales para entrenar el modelo.
    *   Para la predicción de ejemplo en la **PARTE 8**, asegúrate de que la ruta `example_image_path` apunte a una de tus imágenes.

### Explicación de las Partes Clave

*   **Transfer Learning (PARTE 4):** Usamos `MobileNetV2`, un modelo muy potente ya entrenado con millones de imágenes (ImageNet). "Congelamos" sus capas (`base_model.trainable = False`) para aprovechar su conocimiento sobre cómo detectar bordes, texturas y formas. Solo entrenamos las nuevas capas que añadimos al final, lo que es mucho más rápido y efectivo que entrenar una red desde cero, especialmente con pocos datos.
*   **Data Augmentation (PARTE 3):** Las capas `RandomFlip`, `RandomRotation`, etc., modifican aleatoriamente las imágenes de entrenamiento en cada época. Esto "enseña" al modelo a reconocer las hojas aunque estén giradas, con diferente zoom o volteadas, haciéndolo más robusto y evitando que memorice las imágenes de entrenamiento (sobreajuste).
*   **Early Stopping (PARTE 6):** Es una técnica de regularización fundamental. Vigila la métrica `val_loss` (el error en los datos de validación). Si esta métrica deja de mejorar durante 5 épocas (`patience=5`), el entrenamiento se detiene automáticamente. Además, con `restore_best_weights=True`, el modelo final que obtienes es el que tuvo el mejor rendimiento en validación, no necesariamente el de la última época.
*   **Salida Sigmoide y Loss Binaria (PARTE 4 y 5):** Como es un problema de clasificación binaria (dos clases), usamos una única neurona en la capa de salida con activación `sigmoid`. Esta función devuelve un valor entre 0 y 1, que interpretamos como la probabilidad de que la imagen pertenezca a la clase "positiva" (en este caso, 'Con_afidos', que TensorFlow suele asignar al índice 0). La función de pérdida `BinaryCrossentropy` es la pareja perfecta para esta configuración.