"""
Ejercicio 05 - Asistente de Documentos con Gemini y Gradio
Asistente que responde preguntas sobre un PDF usando RAG conceptual.
"""

# ─────────────────────────────────────────────
# PASO 0: Instalación (ejecutar una vez en terminal)
# pip install pypdf gradio google-genai python-dotenv
# ─────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
import gradio as gr

# ─────────────────────────────────────────────
# PASO 1: Configuración inicial
# ─────────────────────────────────────────────

# Carga el .env desde la misma carpeta del script para evitar problemas
# cuando se ejecuta el archivo desde otro directorio de trabajo.
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
if not GEMINI_API_KEY:
    raise ValueError(
        "No se encontró GEMINI_API_KEY. "
        "Verifica el archivo .env junto a este script con: GEMINI_API_KEY='tu_key_aqui'"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

MODELO = "gemini-2.5-flash-lite"
LIMITE_TOKENS = 1_000_000
PDF_PATH = str(BASE_DIR / "attention_is_all_you_need.pdf")  # PDF por defecto


# ─────────────────────────────────────────────
# PASO 2: Extracción de texto del PDF
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Lee un PDF y retorna su contenido como texto plano.
    Filtra páginas vacías o con solo imágenes.
    """
    reader = PdfReader(pdf_path)

    pages_text = [
        page.extract_text()
        for page in reader.pages
        if page.extract_text()
    ]

    return "\n\n".join(pages_text)


# ─────────────────────────────────────────────
# PASO 3: System prompts
# ─────────────────────────────────────────────

def build_system_prompt(document_text: str) -> str:
    """System prompt estándar: responde solo con info del documento."""
    return f"""Eres un asistente experto en el paper de investigación que se te proporciona.

Tu única fuente de información para responder es el siguiente documento:

---INICIO DEL DOCUMENTO---
{document_text}
---FIN DEL DOCUMENTO---

Reglas que debes seguir estrictamente:
1. Responde ÚNICAMENTE con información presente en el documento.
2. Si la respuesta no está en el documento, responde exactamente:
   "Esta información no se encuentra en el documento proporcionado."
3. Responde en el idioma en que se te haga la pregunta.
"""


def build_system_prompt_con_citas(document_text: str) -> str:
    """
    MEJORA C — System prompt con citas textuales.
    Obliga al modelo a respaldar cada afirmación con un fragmento literal del paper.
    """
    return f"""Eres un asistente experto en el paper de investigación que se te proporciona.

Tu única fuente de información para responder es el siguiente documento:

INICIO DEL DOCUMENTO
{document_text}
FIN DEL DOCUMENTO

Reglas que debes seguir estrictamente:

1. Responde ÚNICAMENTE con información presente en el documento.

2. Cada respuesta debe tener exactamente este formato:

   **Respuesta:**
   [Tu explicación clara y concisa basada en el documento]

   **Cita del documento:**
   > "[Fragmento textual EXACTO del paper que respalda tu respuesta]"

   **Ubicación aproximada:**
   [Sección o parte del paper, por ejemplo: "Sección 3.2", "Abstract", "Tabla 2"]

3. La cita debe ser un fragmento LITERAL del documento, entre comillas.
   No la parafrasees ni la modifiques.

4. Si la pregunta no puede responderse con el documento, responde:
   "Esta información no se encuentra en el documento proporcionado."
   En ese caso, omite los campos de Cita y Ubicación.

5. Responde en el idioma en que se te haga la pregunta.
"""


# ─────────────────────────────────────────────
# PASO 4: MEJORA A — Conteo de tokens
# ─────────────────────────────────────────────

def contar_tokens_system_prompt(document_text: str) -> dict:
    """
    Cuenta los tokens exactos del system prompt usando el endpoint
    count_tokens de Gemini, y calcula el porcentaje del límite de 1M.
    """
    system_prompt = build_system_prompt(document_text)

    resultado = client.models.count_tokens(
        model=MODELO,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=system_prompt)]
            )
        ]
    )

    tokens_usados = resultado.total_tokens
    porcentaje = (tokens_usados / LIMITE_TOKENS) * 100

    return {
        "tokens": tokens_usados,
        "porcentaje": round(porcentaje, 4),
        "disponible": LIMITE_TOKENS - tokens_usados
    }


# ─────────────────────────────────────────────
# PASO 5: Función de chat con streaming
# ─────────────────────────────────────────────

def chat_con_documento(
    message: str,
    history: list,
    document_text: str,
    usar_citas: bool = False
):
    """
    Genera una respuesta con streaming basada en el documento.
    Compatible con formato nuevo (dict) y antiguo (lista) de Gradio.
    """
    gemini_history = []

    for turn in history:
        if isinstance(turn, dict):
            role = "model" if turn["role"] == "assistant" else "user"
            content = turn["content"]
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            gemini_history.append(
                types.Content(role=role, parts=[types.Part(text=content)])
            )
        else:
            # Formato antiguo de Gradio: [mensaje_usuario, mensaje_asistente]
            if turn[0]:
                texto = turn[0]
                if isinstance(texto, list):
                    texto = texto[0].get("text", "") if texto else ""
                gemini_history.append(
                    types.Content(role="user", parts=[types.Part(text=texto)])
                )
            if turn[1]:
                texto = turn[1]
                if isinstance(texto, list):
                    texto = texto[0].get("text", "") if texto else ""
                gemini_history.append(
                    types.Content(role="model", parts=[types.Part(text=texto)])
                )

    # Normaliza el mensaje actual si llega en formato lista
    if isinstance(message, list):
        message = message[0].get("text", "") if message else ""

    gemini_history.append(
        types.Content(role="user", parts=[types.Part(text=message)])
    )

    # Elige el system prompt según el modo activado (Mejora C)
    system_prompt = (
        build_system_prompt_con_citas(document_text)
        if usar_citas
        else build_system_prompt(document_text)
    )

    response = client.models.generate_content_stream(
        model=MODELO,
        contents=gemini_history,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        )
    )

    # yield acumula el texto para que Gradio lo muestre token a token
    accumulated = ""
    for chunk in response:
        if chunk.text:
            accumulated += chunk.text
            yield accumulated


# ─────────────────────────────────────────────
# PASO 6: MEJORA B — Función integradora con las 3 mejoras
# ─────────────────────────────────────────────

def chat_completo(
    message: str,
    history: list,
    pdf_file,           # gr.File → filepath o None  (Mejora B)
    usar_citas: bool    # gr.Checkbox               (Mejora C)
):
    """
    Función principal que Gradio llama en cada mensaje.
    Combina las tres mejoras:
      A) Conteo de tokens del system prompt
      B) PDF dinámico subido por el usuario
      C) Respuestas con citas textuales
    """
    try:
        # MEJORA B: usa el PDF subido por el usuario, o el PDF por defecto
        if pdf_file is not None:
            document_text = extract_text_from_pdf(pdf_file)
            print(f"[PDF] Cargado: {pdf_file}")
        else:
            if not os.path.exists(PDF_PATH):
                yield (
                    f"⚠️ No se encontró el PDF por defecto '{PDF_PATH}'. "
                    "Por favor sube un PDF usando el campo de archivo."
                )
                return
            document_text = extract_text_from_pdf(PDF_PATH)
            print(f"[PDF] Usando PDF por defecto: {PDF_PATH}")

        # MEJORA A: imprime el informe de tokens en consola
        info = contar_tokens_system_prompt(document_text)
        print(f"[Tokens] System prompt: {info['tokens']:,} tokens "
              f"({info['porcentaje']}% del límite) — "
              f"{info['disponible']:,} tokens disponibles")

        # Delega el streaming a chat_con_documento (Mejora C incluida)
        yield from chat_con_documento(message, history, document_text, usar_citas)

    except Exception as e:
        yield f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────
# PASO 7: Interfaz Gradio
# ─────────────────────────────────────────────

demo = gr.ChatInterface(
    fn=chat_completo,
    title="📄 Asistente de Documentos PDF",
    description=(
        "Sube cualquier PDF y haz preguntas sobre su contenido. "
        "Las respuestas se basan **exclusivamente** en el documento. "
        "Activa **Modo con citas** para que cada respuesta incluya "
        "un fragmento textual del documento que la respalda."
    ),
    additional_inputs=[
        # MEJORA B: el usuario sube cualquier PDF
        gr.File(
            label="Sube tu PDF (opcional — si no subes uno, se usa el paper por defecto)",
            file_types=[".pdf"],
            type="filepath"
        ),
        # MEJORA C: toggle para activar citas textuales
        gr.Checkbox(
            label="Modo con citas (incluye fragmentos textuales del documento)",
            value=True
        ),
    ],
    examples=[
        ["¿Cuál es la arquitectura principal propuesta en el paper?", None, True],
        ["¿Qué es el mecanismo de atención multi-cabeza?", None, True],
        ["¿Cuántas capas tiene el encoder del modelo base?", None, True],
        ["¿Cuál es el resultado en WMT 2014 English-to-German?", None, True],
        ["¿Quiénes son los autores?", None, False],
        ["¿Qué es GPT-4?", None, False],  # Pregunta trampa — no está en el paper
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)