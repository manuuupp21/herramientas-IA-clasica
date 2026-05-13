"""
FitCore RAG — Fase 2: Webhook RAG (proceso en tiempo real)

Dialogflow llama a este webhook cuando detecta intents que requieren
respuesta dinámica (Consultar_Plan_Nutricional, Consultar_Plan_Entrenamiento,
y cualquier intent con webhookUsed=true que apunte a acciones RAG).

Pipeline por llamada:
  pregunta usuario → embedding → búsqueda Qdrant → top-K chunks
  → prompt con contexto → Deepseek (Ollama) → fulfillmentText → Dialogflow

Requiere:
  pip install fastapi uvicorn qdrant-client ollama
  uvicorn fase2_webhook_rag:app --host 0.0.0.0 --port 3001
  + ngrok http 3001  (para exponer a Dialogflow)
"""

import os
import re
from typing import Any

import ollama
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

# ─── Configuración ───────────────────────────────────────────────────────────

OLLAMA_HOST     = os.getenv("OLLAMA_HOST",     "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "deepseek-ocr:3b")
QDRANT_HOST     = os.getenv("QDRANT_HOST",     "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "fitcore_kb"
TOP_K           = 4    # chunks recuperados por consulta
MAX_CTX_CHARS   = 2000 # límite de caracteres de contexto para el prompt

app    = FastAPI(title="FitCore RAG Webhook")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
olm    = ollama.Client(host=OLLAMA_HOST)

# ─── Helpers RAG ─────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Genera embedding de la pregunta del usuario."""
    return olm.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Busca en Qdrant los chunks más relevantes para la query.
    Devuelve lista de {text, source, page, score}.
    """
    vector = embed(query)
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text":   h.payload.get("text", ""),
            "source": h.payload.get("source", "desconocido"),
            "page":   h.payload.get("page", 1),
            "score":  round(h.score, 3),
        }
        for h in hits
    ]


def build_prompt(question: str, chunks: list[dict], extra_context: str = "") -> str:
    """
    Construye el prompt RAG con el contexto recuperado.
    """
    # Truncar contexto si es muy largo
    context_parts = []
    total = 0
    for c in chunks:
        snippet = c["text"].strip()
        if total + len(snippet) > MAX_CTX_CHARS:
            snippet = snippet[:MAX_CTX_CHARS - total]
            context_parts.append(f"[{c['source']} p.{c['page']}]\n{snippet}")
            break
        context_parts.append(f"[{c['source']} p.{c['page']}]\n{snippet}")
        total += len(snippet)

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No se encontró información relevante."

    extra = f"\nContexto adicional: {extra_context}" if extra_context else ""

    return f"""Eres el asistente virtual de FitCore, especializado en soluciones deportivas y nutrición personal.
Responde de forma natural, concisa y en español, basándote ÚNICAMENTE en el contexto proporcionado.
Si el contexto no contiene información suficiente, dilo claramente y ofrece contactar con un especialista.
No inventes datos ni menciones que estás usando documentos.{extra}

CONTEXTO RECUPERADO:
{context_str}

PREGUNTA DEL USUARIO:
{question}

RESPUESTA (máximo 3 frases, natural y conversacional):"""


def generate(prompt: str) -> str:
    """Genera respuesta con Deepseek via Ollama."""
    response = olm.generate(
        model=GEN_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.3,   # baja temperatura: respuestas más consistentes
            "num_predict": 250,   # máximo de tokens generados
            "stop": ["\n\n\n"],   # evita respuestas interminables
        }
    )
    text = response["response"].strip()
    # Limpiar el bloque <think>...</think> que genera deepseek-r1
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ─── Router de acciones Dialogflow ──────────────────────────────────────────

def handle_plan_nutricional(params: dict, question: str) -> str:
    objetivo = params.get("objetivo_deportivo", "")
    extra    = f"El usuario busca información sobre nutrición para: {objetivo}" if objetivo else ""
    chunks   = retrieve(question or f"plan nutricional {objetivo}", top_k=TOP_K)
    prompt   = build_prompt(question, chunks, extra_context=extra)
    return generate(prompt)


def handle_plan_entrenamiento(params: dict, question: str) -> str:
    objetivo  = params.get("objetivo_deportivo", "")
    dias      = params.get("dias_semana", "")
    extra     = f"Objetivo deportivo: {objetivo}. Días disponibles: {dias} días/semana." if objetivo else ""
    chunks    = retrieve(question or f"rutina entrenamiento {objetivo}", top_k=TOP_K)
    prompt    = build_prompt(question, chunks, extra_context=extra)
    return generate(prompt)


def handle_generic_rag(params: dict, question: str) -> str:
    """Fallback RAG genérico para cualquier pregunta compleja."""
    chunks = retrieve(question, top_k=TOP_K)
    if not chunks or chunks[0]["score"] < 0.35:
        return ("Lo siento, no tengo información suficiente sobre eso en mi base de conocimiento. "
                "Te recomiendo contactar con nuestro equipo en soporte@fitcore.es.")
    prompt = build_prompt(question, chunks)
    return generate(prompt)


# ─── Endpoint principal del webhook ─────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request):
    body: dict[str, Any] = await request.json()

    query_result = body.get("queryResult", {})
    action       = query_result.get("action", "")
    params       = query_result.get("parameters", {})
    question     = query_result.get("queryText", "")

    print(f"\n[Webhook RAG] acción={action} | pregunta='{question}'")

    # ── Dispatcher por acción ─────────────────────────────────────────────────
    if action == "consultar.plan.nutricional":
        response_text = handle_plan_nutricional(params, question)

    elif action == "consultar.plan.entrenamiento":
        response_text = handle_plan_entrenamiento(params, question)

    # Añade aquí más acciones RAG según necesites:
    # elif action == "consultar.suplementacion":
    #     response_text = handle_suplementacion(params, question)

    elif action.startswith("rag."):
        # Captura cualquier acción que empiece por "rag." como RAG genérico
        response_text = handle_generic_rag(params, question)

    else:
        # Acción no RAG — respuesta vacía para que Dialogflow use la estática
        return JSONResponse(content={})

    print(f"[Webhook RAG] respuesta generada ({len(response_text)} chars)")
    return JSONResponse(content={"fulfillmentText": response_text})


# ─── Health check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Verifica que Qdrant y Ollama están accesibles."""
    status = {"webhook": "ok"}
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        status["qdrant"] = f"ok — {info.points_count} vectores"
    except Exception as e:
        status["qdrant"] = f"error: {e}"
    try:
        models = [m["name"] for m in olm.list()["models"]]
        status["ollama_models"] = models
    except Exception as e:
        status["ollama"] = f"error: {e}"
    return status


# ─── Arranque ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fase2_webhook_rag:app", host="0.0.0.0", port=3001, reload=True)
