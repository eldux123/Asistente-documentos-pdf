# Ejercicio 05 — Asistente de Documentos PDF con Gemini y Gradio

Aplicación que permite hacer preguntas sobre un PDF usando la API de Gemini. Las respuestas se generan **exclusivamente** a partir del contenido del documento cargado, implementando el patrón conceptual de RAG (Retrieval-Augmented Generation).

---

## Requisitos previos

### Dependencias

```bash
pip install -r requirements.txt
```

| Librería | Versión | Para qué se usa |
|---|---|---|
| `pypdf` | cualquiera | Extraer texto de archivos PDF |
| `gradio` | cualquiera | Interfaz web del chat |
| `google-genai` | cualquiera | Cliente de la API de Gemini |
| `python-dotenv` | cualquiera | Cargar variables de entorno desde `.env` |

### Archivo `.env`

Crea un archivo llamado `.env` en la **misma carpeta** que el script:

```
GEMINI_API_KEY="tu_key_aqui"
```

Obtén tu key gratis en: https://aistudio.google.com/apikey

### PDF por defecto

Descarga el paper "Attention is All You Need" y guárdalo con este nombre exacto en la misma carpeta:

```
attention_is_all_you_need.pdf
```

URL de descarga: https://arxiv.org/pdf/1706.03762

---

## Cómo ejecutar

```bash
python ejercicio_5.py
```

Luego abre en el navegador:

```
http://localhost:8080
```

---

## Estructura del código

El archivo está dividido en 7 pasos secuenciales:

```
Paso 0 → Instalación de dependencias
Paso 1 → Configuración inicial (API key, modelo, rutas)
Paso 2 → Extracción de texto del PDF
Paso 3 → System prompts (estándar y con citas)
Paso 4 → Mejora A: Conteo de tokens
Paso 5 → Función de chat con streaming
Paso 6 → Mejora B: Función integradora con las 3 mejoras
Paso 7 → Interfaz Gradio
```

---

## Referencia de funciones

### `extract_text_from_pdf(pdf_path)`

Extrae el texto de todas las páginas de un PDF.

```python
text = extract_text_from_pdf("mi_documento.pdf")
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `pdf_path` | `str` | Ruta al archivo PDF |

**Retorna:** `str` — Texto completo del PDF, con páginas separadas por doble salto de línea.

> **Nota:** Solo extrae texto incrustado. No funciona con PDFs escaneados (imágenes).

---

### `build_system_prompt(document_text)`

Genera el system prompt estándar. Le indica al modelo que responda únicamente con información del documento.

```python
prompt = build_system_prompt(document_text)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `document_text` | `str` | Texto completo del documento |

**Retorna:** `str` — System prompt formateado.

---

### `build_system_prompt_con_citas(document_text)`

**Mejora C.** Genera un system prompt que obliga al modelo a estructurar cada respuesta en tres partes:

- **Respuesta:** explicación basada en el documento
- **Cita del documento:** fragmento textual exacto del paper
- **Ubicación aproximada:** sección del paper donde se encuentra la cita

```python
prompt = build_system_prompt_con_citas(document_text)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `document_text` | `str` | Texto completo del documento |

**Retorna:** `str` — System prompt con formato de citas.

---

### `contar_tokens_system_prompt(document_text)`

**Mejora A.** Usa el endpoint `count_tokens` de Gemini para contar exactamente cuántos tokens ocupa el system prompt antes de hacer la llamada real.

```python
info = contar_tokens_system_prompt(document_text)
print(info["tokens"])      # ej: 32,450
print(info["porcentaje"])  # ej: 3.245 (% del límite de 1M)
print(info["disponible"])  # tokens restantes
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `document_text` | `str` | Texto completo del documento |

**Retorna:** `dict` con las claves:

| Clave | Tipo | Descripción |
|---|---|---|
| `tokens` | `int` | Tokens usados por el system prompt |
| `porcentaje` | `float` | Porcentaje del límite de 1,000,000 tokens |
| `disponible` | `int` | Tokens restantes disponibles |

---

### `chat_con_documento(message, history, document_text, usar_citas)`

Función generadora que envía el mensaje a Gemini con streaming y va acumulando la respuesta token a token.

Maneja el historial en los dos formatos que puede entregar Gradio:
- **Formato nuevo:** lista de dicts `{"role": "...", "content": "..."}`
- **Formato antiguo:** lista de listas `[mensaje_usuario, mensaje_asistente]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `message` | `str` | Mensaje actual del usuario |
| `history` | `list` | Historial de la conversación |
| `document_text` | `str` | Texto del PDF ya extraído |
| `usar_citas` | `bool` | `True` activa el sistema de citas (Mejora C) |

**Retorna:** generador de `str` — cada `yield` es la respuesta acumulada hasta ese momento.

---

### `chat_completo(message, history, pdf_file, usar_citas)`

**Función principal.** Es la que Gradio llama en cada mensaje. Integra las tres mejoras:

- **Mejora A** — Imprime en consola el conteo de tokens del system prompt
- **Mejora B** — Usa el PDF subido por el usuario, o el PDF por defecto si no se subió ninguno
- **Mejora C** — Activa las citas textuales según el checkbox

```python
# Internamente Gradio la llama así:
yield from chat_completo(message, history, pdf_file, usar_citas)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `message` | `str` | Mensaje del usuario |
| `history` | `list` | Historial del chat |
| `pdf_file` | `str \| None` | Ruta al PDF subido por el usuario, o `None` |
| `usar_citas` | `bool` | Estado del checkbox de citas |

**Retorna:** generador de `str`. En caso de error, hace `yield` de un mensaje con ❌.

---

## Variables globales

| Variable | Valor | Descripción |
|---|---|---|
| `MODELO` | `"gemini-2.5-flash-lite"` | Modelo de Gemini usado |
| `LIMITE_TOKENS` | `1_000_000` | Límite oficial de contexto del modelo |
| `PDF_PATH` | `"attention_is_all_you_need.pdf"` | Ruta al PDF por defecto |
| `BASE_DIR` | ruta del script | Directorio base para resolver rutas relativas |

---

## Interfaz de usuario

La interfaz tiene los siguientes controles:

| Elemento | Tipo | Descripción |
|---|---|---|
| Campo de chat | `gr.ChatInterface` | Cuadro principal para escribir preguntas |
| Sube tu PDF | `gr.File` | Permite subir cualquier PDF (opcional) |
| Modo con citas | `gr.Checkbox` | Activa/desactiva las citas textuales (Mejora C) |

### Ejemplos precargados

La interfaz incluye estos ejemplos para probar rápidamente:

| Pregunta | Citas |
|---|---|
| ¿Cuál es la arquitectura principal propuesta en el paper? | ✅ |
| ¿Qué es el mecanismo de atención multi-cabeza? | ✅ |
| ¿Cuántas capas tiene el encoder del modelo base? | ✅ |
| ¿Cuál es el resultado en WMT 2014 English-to-German? | ✅ |
| ¿Quiénes son los autores? | ❌ |
| ¿Qué es GPT-4? *(pregunta trampa)* | ❌ |

> La última pregunta es una **trampa**: GPT-4 no está en el paper. El modelo debe responder que la información no se encuentra en el documento.

---

## Flujo de ejecución

```
Usuario escribe pregunta
        ↓
chat_completo()
        ↓
¿Se subió un PDF?
    Sí → extract_text_from_pdf(pdf_file)
    No → extract_text_from_pdf(PDF_PATH por defecto)
        ↓
contar_tokens_system_prompt()  ← imprime en consola (Mejora A)
        ↓
chat_con_documento()
        ↓
¿Modo con citas activado?
    Sí → build_system_prompt_con_citas()
    No → build_system_prompt()
        ↓
generate_content_stream()  ← llamada a Gemini con streaming
        ↓
yield token a token → Gradio muestra la respuesta en tiempo real
```

---

## Solución de problemas

| Error | Causa | Solución |
|---|---|---|
| `GEMINI_API_KEY not found` | El `.env` no existe o está en otra carpeta | Crea `.env` junto al script |
| `403 PERMISSION_DENIED - suspended` | La API key fue suspendida | Genera una nueva en aistudio.google.com |
| `ERR_ADDRESS_INVALID` en el navegador | Se usó `0.0.0.0` como URL | Usa `http://localhost:8080` |
| `No se encontró el PDF por defecto` | El PDF no está en la carpeta del script | Descarga el paper y guárdalo con el nombre correcto |
| Respuestas lentas | El paper es largo (~32K tokens) | Es normal; el modelo procesa el contexto completo cada vez |

---

## Conceptos relacionados

**RAG (Retrieval-Augmented Generation):** patrón de arquitectura donde el contenido relevante se inyecta en el contexto del modelo antes de generar una respuesta. Este ejercicio implementa la versión más simple: inyectar el documento completo. En producción, RAG divide el documento en fragmentos y solo recupera los más relevantes para cada pregunta, lo que permite manejar documentos de miles de páginas.

**Ventana de contexto:** Gemini 2.5 Flash Lite soporta hasta 1,000,000 tokens de contexto. El paper "Attention is All You Need" ocupa aproximadamente 32,000 tokens, usando solo el 3.2% del límite disponible.

**Streaming:** en lugar de esperar a que el modelo genere toda la respuesta, `generate_content_stream` devuelve fragmentos (chunks) a medida que se generan. Gradio los muestra en tiempo real usando `yield`.
