# LeLePhid-LLM-CNN

Evaluación de Modelos de Lenguaje de Gran Escala (LLMs) para la generación automática de código CNN aplicado a la clasificación de hojas de limón con áfidos.

## 📋 Descripción

Este repositorio contiene el código, los prompts, las figuras y los resultados del experimento realizado para la tesis de maestría **"Evaluación de LLMs para la generación de código CNN en la clasificación de imágenes de hojas de limón con áfidos"**. El objetivo es comparar el rendimiento de cuatro LLMs (GPT-4o, Gemini, Grok y Claude) en la generación de código Python para una CNN, evaluando tanto la calidad estática del código como su ejecutabilidad en un pipeline automatizado.

## 📁 Estructura del Repositorio


## 🔧 Requisitos

Para ejecutar el pipeline, necesitas:

- Python 3.12 o superior
- Google Colab Pro (recomendado) o un entorno con GPU
- GPU con al menos 16 GB de VRAM (probado en Tesla T4)
- Las librerías listadas en `requirements.txt`

```bash
pip install -r requirements.txt

from google.colab import userdata
openai_key = userdata.get('OPENAI_KEY')
gemini_key = userdata.get('GEMINI_KEY')
grok_key = userdata.get('GROKAI_KEY')
claude_key = userdata.get('CLAUDE_KEY')

# Reorganizar el dataset en train/val/test (70/15/15)
# con estratificación y semilla fija (42)

# Para cada corrida (1..10):
#   Para cada prompt (v1_base, v2_pytorch, v3_tensorflow):
#     Para cada réplica (1..2):
#       Para cada LLM (GPT-4o, Gemini, Grok, Claude):
#         Generar código CNN

# Extraer bloques de código Python
# Corregir errores comunes (rutas, imports, np.Inf, etc.)
# Aplicar correcciones específicas para Gemini

# Calcular Pylint score
# Verificar sintaxis
# Calcular imports ratio
# Verificar ejecutabilidad (1 época)

# Seleccionar modelos con Pylint >= 7
# Entrenar por 10 épocas (prueba rápida)
# Entrenar por 50 épocas (prueba completa)
# Registrar accuracy, F1, tiempo, GPU y RAM

@mastersthesis{maldonado2026llm,
  title={Evaluación de Modelos de Lenguaje de Gran Escala (LLMs) para la generación automática de código CNN aplicado a la clasificación de hojas de limón con áfidos},
  author={Maldonado Jiménez, Freddy Yovanny},
  year={2026},
  school={Yachay Tech University}
}

---

1. **`[freddy.maldonado@yachaytech.edu.ec]`**
2. **`Enero 2026`** 
3. **`pipeline_completo.ipynb`** 

---




