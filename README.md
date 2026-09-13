# LeLePhid-LLM-CNN

Evaluación de Modelos de Lenguaje de Gran Escala (LLMs) para la generación automática de código CNN aplicado a la clasificación de hojas de limón con áfidos.

## 📋 Descripción

Este repositorio contiene el código, los prompts, las figuras y los resultados del experimento realizado para la tesis de maestría **"Evaluación de LLMs para la generación de código CNN en la clasificación de imágenes de hojas de limón con áfidos"**. El objetivo es comparar el rendimiento de cuatro LLMs (GPT-4o, Gemini, Grok y Claude) en la generación de código Python para una CNN, evaluando tanto la calidad estática del código como su ejecutabilidad en un pipeline automatizado.

## 📁 Estructura del Repositorio
LeLePhid-LLM-CNN/
├── src/
│ └── prompts/
│ ├── prompt.txt # Prompt base (v1_base)
│ ├── prompt_v2_pytorch.txt # Prompt forzando PyTorch
│ └── prompt_v3_tensorflow.txt # Prompt forzando TensorFlow
├── resultados_multiple_runs/ # Resultados de la fase repaired (10 corridas)
│ ├── run1_quality.csv
│ ├── run1_quick_eval.csv
│ ├── run1_full_eval.csv
│ └── ...
├── resultados_raw_executability/ # Resultados de la fase raw (sin limpieza)
│ ├── comparacion_raw_vs_repaired.csv
│ └── raw_test_inicio.csv
├── requirements.txt # Dependencias del proyecto
├── pipeline_completo.ipynb # Notebook con el pipeline completo
└── README.md # Este archivo


## 🔧 Requisitos

Para ejecutar el pipeline, necesitas:

- Python 3.12 o superior
- Google Colab Pro (recomendado) o un entorno con GPU
- GPU con al menos 16 GB de VRAM (probado en Tesla T4)
- Las librerías listadas en `requirements.txt`

```bash
pip install -r requirements.txt

🔑 Configuración de API Keys
El pipeline requiere las siguientes API keys para generar código:

OpenAI (GPT-4o): OPENAI_KEY

Google (Gemini): GEMINI_KEY

xAI (Grok): GROKAI_KEY

Anthropic (Claude): CLAUDE_KEY

En Google Colab, puedes configurarlas usando userdata:

from google.colab import userdata
openai_key = userdata.get('OPENAI_KEY')
gemini_key = userdata.get('GEMINI_KEY')
grok_key = userdata.get('GROKAI_KEY')
claude_key = userdata.get('CLAUDE_KEY')

🚀 Instrucciones de Ejecución
Paso 1: Preparar el Dataset

# Reorganizar el dataset en train/val/test (70/15/15)
# con estratificación y semilla fija (42)

Paso 2: Generar Código
# Para cada corrida (1..10):
#   Para cada prompt (v1_base, v2_pytorch, v3_tensorflow):
#     Para cada réplica (1..2):
#       Para cada LLM (GPT-4o, Gemini, Grok, Claude):
#         Generar código CNN

Paso 3: Limpiar Código

# Extraer bloques de código Python
# Corregir errores comunes (rutas, imports, np.Inf, etc.)
# Aplicar correcciones específicas para Gemini

Paso 4: Evaluar Calidad

# Calcular Pylint score
# Verificar sintaxis
# Calcular imports ratio
# Verificar ejecutabilidad (1 época)

Paso 5: Entrenar (10 y 50 épocas)

# Seleccionar modelos con Pylint >= 7
# Entrenar por 10 épocas (prueba rápida)
# Entrenar por 50 épocas (prueba completa)
# Registrar accuracy, F1, tiempo, GPU y RAM

📊 Resultados Esperados
Al finalizar el pipeline, se generan los siguientes archivos:

Archivo	Contenido
run{N}_quality.csv	Métricas de calidad estática (Pylint, imports, sintaxis)
run{N}_quick_eval.csv	Resultados de la prueba de 10 épocas
run{N}_full_eval.csv	Resultados de la prueba de 50 épocas (accuracy, F1, tiempo, recursos)

Resumen de Resultados
Tras ejecutar las 10 corridas, se espera obtener:

240 modelos planificados (60 por LLM)

183 códigos generados (los 57 restantes corresponden a timeouts de Gemini)

131 códigos con Pylint ≥ 7

37 modelos que inician 50 épocas

18 modelos que completan 50 épocas

Los resultados principales son:

GPT-4o: mayor consistencia (90% en Top 4) y tasa de finalización (21.7%)

Claude: alta calidad estática (Pylint 8.23) pero baja ejecutabilidad (5%)

Grok: baja tasa de finalización (1.7%)

Gemini: no comparable (solo 3 modelos generados por timeouts)

📊 Raw-code vs. Repaired-code Executability
Además de la evaluación principal (repaired-code executability), se realizó una segunda fase experimental para medir la raw-code executability, es decir, la ejecutabilidad del código tal como fue generado por los LLMs, sin aplicar el proceso de limpieza.

Resultados
LLM	Raw (Pylint ≥ 7)	Repaired (Pylint ≥ 7)	Mejora (pp)
GPT-4o	0.0% (0/60)	60.0% (36/60)	+60.0
Grok	0.0% (0/60)	66.7% (40/60)	+66.7
Claude	0.0% (0/60)	90.0% (54/60)	+90.0
Gemini	0.0% (0/0)*	66.7% (2/3)*	N/A
* Gemini no generó códigos en la fase raw debido a timeouts en la API.

Conclusión
Sin el proceso de limpieza, ningún código logró superar el filtro de calidad estática (Pylint ≥ 7). Esto confirma que la limpieza automática es un requisito indispensable para que el código generado por LLMs sea evaluable en un pipeline automatizado.

Los archivos de esta fase se encuentran en la carpeta resultados_raw_executability/.

📈 Figuras Incluidas
Figura	Descripción
fig_pylint_distribucion.png	Distribución de puntuaciones Pylint por LLM
fig_filtros_modelos.png	Modelos que superaron cada filtro del pipeline
fig_consistencia_top4.png	Consistencia de los LLMs en el Top 4
fig_radar_rendimiento.png	Perfil de rendimiento por LLM (radar)
cnn_architecture.png	Arquitectura de una CNN típica
📝 Cómo Citar
Si usas este código o los resultados en tu investigación, por favor cita:

bibtex
@mastersthesis{maldonado2026llm,
  title={Evaluación de Modelos de Lenguaje de Gran Escala (LLMs) para la generación automática de código CNN aplicado a la clasificación de hojas de limón con áfidos},
  author={Maldonado Jiménez, Freddy Yovanny},
  year={2026},
  school={Yachay Tech University}
}
📄 Licencia
Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

📧 Contacto
Para preguntas o comentarios, contactar a:

Freddy Yovanny Maldonado Jiménez

Email: freddy.maldonado@yachaytech.edu.ec

Repositorio: https://github.com/FreddyYMJ/LeLePhid-LLM-CNN

Última actualización: Enero 2026
