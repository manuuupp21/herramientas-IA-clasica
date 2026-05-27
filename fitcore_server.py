"""
FitCore — Servidor unificado (versión corregida)
Puerto único 3001:
  GET  /webhook  → verificación de Dialogflow al guardar URL
  POST /webhook  → webhook RAG llamado por Dialogflow
  WS   /ws       → pipeline de voz (micrófono navegador → Azure STT → Dialogflow → Azure TTS)
  GET  /         → cliente web HTML
  GET  /health   → estado servicios

Ejecutar:
  uvicorn fitcore_server_final:app --host 0.0.0.0 --port 3001 --reload
  ngrok http 3001
  → Pegar https://xxx.ngrok-free.app/webhook en Dialogflow Fulfillment
  → Abrir https://xxx.ngrok-free.app en navegador para la voz
"""

import os
import re
import uuid
import json
import asyncio
from typing import Any

import ollama
import azure.cognitiveservices.speech as speechsdk
import dialogflow
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, HTMLResponse
from qdrant_client import QdrantClient

# ─── Configuración ───────────────────────────────────────────────────────────

SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY",             "TU_AZURE_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION",          "westeurope")
PROJECT_ID    = os.getenv("DIALOGFLOW_PROJECT_ID",        "TU_PROJECT_ID")
LANGUAGE      = "es-ES"
TTS_VOICE     = "es-ES-AlvaroNeural"

# Credenciales Google — ruta absoluta recomendada para evitar problemas
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds

OLLAMA_HOST     = os.getenv("OLLAMA_HOST",          "http://localhost:11434")
GEN_MODEL       = os.getenv("OLLAMA_GEN_MODEL",     "deepseek-ocr:3b")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL",   "nomic-embed-text")
QDRANT_HOST     = os.getenv("QDRANT_HOST",          "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT",      "6333"))
COLLECTION_NAME = "fitcore_kb"
TOP_K           = 4
MAX_CTX_CHARS   = 2000

app    = FastAPI(title="FitCore Servidor Unificado")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
olm    = ollama.Client(host=OLLAMA_HOST)


# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — RAG  (POST /webhook llamado por Dialogflow)
# ═════════════════════════════════════════════════════════════════════════════

def embed(text: str) -> list[float]:
    return olm.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def retrieve(query: str) -> list[dict]:
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed(query),
        limit=TOP_K,
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


def build_prompt(question: str, chunks: list[dict], extra: str = "") -> str:
    parts, total = [], 0
    for c in chunks:
        snippet = c["text"].strip()
        if total + len(snippet) > MAX_CTX_CHARS:
            snippet = snippet[:MAX_CTX_CHARS - total]
            parts.append(f"[{c['source']} p.{c['page']}]\n{snippet}")
            break
        parts.append(f"[{c['source']} p.{c['page']}]\n{snippet}")
        total += len(snippet)

    context_str = "\n\n---\n\n".join(parts) or "Sin información relevante."
    extra_str   = f"\nContexto adicional: {extra}" if extra else ""

    return f"""Eres el asistente virtual de FitCore, especializado en soluciones deportivas y nutrición personal.
Responde de forma natural, concisa y en español, basándote ÚNICAMENTE en el contexto proporcionado.
Si el contexto no contiene información suficiente, dilo claramente.{extra_str}

CONTEXTO:
{context_str}

PREGUNTA: {question}

RESPUESTA (máximo 3 frases, natural y conversacional):"""


def generate(prompt: str) -> str:
    resp = olm.generate(
        model=GEN_MODEL,
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": 250, "stop": ["\n\n\n"]},
    )
    text = resp["response"].strip()
    # Eliminar bloque <think>...</think> de deepseek-r1
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def rag_response(question: str, extra: str = "") -> str:
    chunks = retrieve(question)
    if not chunks or chunks[0]["score"] < 0.35:
        return ("No tengo información suficiente sobre eso en mi base de conocimiento. "
                "Puedes contactar con nuestro equipo en soporte@fitcore.es.")
    return generate(build_prompt(question, chunks, extra))


def dispatch_rag(action: str, params: dict, question: str) -> str | None:
    """Devuelve texto RAG o None si la acción no le corresponde."""
    if action == "consultar.plan.nutricional":
        objetivo = params.get("objetivo_deportivo", "")
        return rag_response(
            question or f"plan nutricional {objetivo}",
            extra=f"Objetivo deportivo: {objetivo}" if objetivo else ""
        )
    if action == "consultar.plan.entrenamiento":
        objetivo = params.get("objetivo_deportivo", "")
        dias     = params.get("dias_semana", "")
        extra    = f"Objetivo: {objetivo}. Días disponibles: {dias} días/semana." if objetivo else ""
        return rag_response(question or f"rutina entrenamiento {objetivo}", extra=extra)
    if action.startswith("rag."):
        return rag_response(question)
    return None  # acción no RAG → Dialogflow usa respuesta estática del intent


# ── GET /webhook — Dialogflow verifica la URL con GET al guardarla ────────────
@app.get("/webhook")
async def webhook_verify():
    """
    FIX método no permitido: Dialogflow hace GET para verificar el webhook.
    Sin este endpoint respondía 405 y Fulfillment no se podía guardar.
    """
    return JSONResponse(content={"status": "ok", "service": "FitCore webhook activo"})


# ── POST /webhook — llamada real de Dialogflow ────────────────────────────────
@app.post("/webhook")
async def webhook(request: Request):
    body: dict[str, Any] = await request.json()
    qr       = body.get("queryResult", {})
    action   = qr.get("action", "")
    params   = qr.get("parameters", {})
    question = qr.get("queryText", "")

    print(f"\n[Webhook POST] acción={action!r} | pregunta={question!r}")

    response_text = dispatch_rag(action, params, question)

    if response_text is None:
        print("[Webhook POST] acción no RAG → respuesta estática de Dialogflow")
        return JSONResponse(content={})

    print(f"[Webhook POST] RAG OK → {len(response_text)} chars")
    return JSONResponse(content={"fulfillmentText": response_text})


# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Pipeline de voz  (WebSocket /ws)
# ═════════════════════════════════════════════════════════════════════════════

def make_speech_config() -> speechsdk.SpeechConfig:
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.speech_recognition_language = LANGUAGE
    cfg.speech_synthesis_voice_name = TTS_VOICE
    # Evitar cortes prematuros al hablar
    cfg.set_property(
        speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1200"
    )
    return cfg


def stt_from_pcm(speech_cfg: speechsdk.SpeechConfig, pcm_bytes: bytes) -> str | None:
    """
    Azure STT desde PCM int16 enviado por el navegador (16000 Hz, mono).
    Usa PushAudioInputStream para audio pregrabado — NO necesita micrófono del servidor.
    """
    print(f"[STT] PCM recibido: {len(pcm_bytes)} bytes "
          f"= {len(pcm_bytes)/2/16000:.2f}s de audio")

    if len(pcm_bytes) < 3200:   # menos de 0.1s → descartar
        print("[STT] Audio demasiado corto, descartado.")
        return None

    audio_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=16000,
        bits_per_sample=16,
        channels=1
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    audio_cfg   = speechsdk.audio.AudioConfig(stream=push_stream)

    recognizer  = speechsdk.SpeechRecognizer(
        speech_config=speech_cfg, audio_config=audio_cfg
    )

    # PhraseList mejora el reconocimiento de términos del dominio
    phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
    for term in ["FitCore", "PED-", "INC-", "incidencia",
                 "plan nutricional", "plan de entrenamiento",
                 "pérdida de peso", "ganancia muscular", "creatina"]:
        phrase_list.addPhrase(term)

    push_stream.write(pcm_bytes)
    push_stream.close()

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"[STT] Reconocido: {result.text!r}")
        return result.text

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"[STT] Cancelado: {details.reason}")
        if details.reason == speechsdk.CancellationReason.Error:
            print(f"[STT] Error detalle: {details.error_details}")
            # Error común: credenciales incorrectas o región equivocada
            if "401" in str(details.error_details):
                print("[STT] ⚠️  Verifica AZURE_SPEECH_KEY y AZURE_SPEECH_REGION")
            if "1006" in str(details.error_details):
                print("[STT] ⚠️  Región incorrecta — usa el código corto: westeurope, eastus, etc.")
    else:
        print(f"[STT] Sin resultado: {result.reason}")

    return None


def detect_intent_text(df_client, session_path: str, text: str) -> tuple[str, str]:
    """Dialogflow detect_intent → (fulfillment_text, intent_name)."""
    query_input = dialogflow.types.QueryInput(
        text=dialogflow.types.TextInput(text=text, language_code=LANGUAGE)
    )
    resp   = df_client.detect_intent(session=session_path, query_input=query_input)
    qr     = resp.query_result
    intent = qr.intent.display_name if qr.intent else "desconocido"
    text_r = qr.fulfillment_text or "No pude procesar tu solicitud."
    print(f"[Dialogflow] intent={intent!r} | respuesta={text_r[:80]!r}...")
    return text_r, intent


def tts_to_bytes(speech_cfg: speechsdk.SpeechConfig, text: str) -> bytes:
    """Azure TTS con SSML → bytes WAV en memoria."""
    stream      = speechsdk.audio.PullAudioOutputStream()
    audio_cfg   = speechsdk.audio.AudioOutputConfig(stream=stream)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_cfg, audio_config=audio_cfg
    )
    ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="es-ES">
        <voice name="{TTS_VOICE}">
            <prosody rate="0.95" pitch="+2%">{text}</prosody>
        </voice>
    </speak>"""
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"[TTS] Error: {details.error_details}")
    return b""


async def _process_turn(
    ws: WebSocket, loop: asyncio.AbstractEventLoop,
    df_client, session_path: str,
    speech_cfg: speechsdk.SpeechConfig,
    user_text: str
) -> bool:
    """
    Turno completo: texto → Dialogflow → TTS → enviar audio.
    Devuelve True si el intent es de despedida (cerrar sesión).
    """
    EXIT_INTENTS = {"Despedida"}

    await ws.send_json({"type": "stt",    "text": user_text})
    await ws.send_json({"type": "status", "text": "Consultando FitCore..."})

    response_text, intent_name = await loop.run_in_executor(
        None, detect_intent_text, df_client, session_path, user_text
    )
    await ws.send_json({"type": "response", "text": response_text})

    audio = await loop.run_in_executor(None, tts_to_bytes, speech_cfg, response_text)
    if audio:
        await ws.send_bytes(audio)
    else:
        # Si TTS falla, al menos desbloquear el cliente
        await ws.send_json({"type": "status", "text": "Pulsa para hablar"})

    return intent_name in EXIT_INTENTS


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    df_client    = dialogflow.SessionsClient()
    session_path = df_client.session_path(PROJECT_ID, str(uuid.uuid4()))
    speech_cfg   = make_speech_config()
    loop         = asyncio.get_event_loop()
    waiting_audio = False

    print(f"[WS] Nueva sesión: {session_path.split('/')[-1][:8]}...")

    # ── Saludo inicial ────────────────────────────────────────────────────────
    saludo = "¡Hola! Bienvenido a FitCore, tu asistente de soluciones deportivas y nutrición. ¿En qué puedo ayudarte?"
    audio  = await loop.run_in_executor(None, tts_to_bytes, speech_cfg, saludo)
    await ws.send_json({"type": "tts", "text": saludo})
    if audio:
        await ws.send_bytes(audio)

    try:
        while True:
            data = await ws.receive()

            # ── Mensaje de texto (JSON) ───────────────────────────────────────
            if "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "audio_start":
                    # Próximo mensaje será audio binario del navegador
                    waiting_audio = True
                    await ws.send_json({"type": "status", "text": "Recibiendo audio..."})

            # ── Audio binario PCM del navegador ───────────────────────────────
            elif "bytes" in data:
                if not waiting_audio:
                    continue
                waiting_audio = False
                pcm_bytes = data["bytes"]

                await ws.send_json({"type": "status", "text": "Reconociendo voz..."})

                user_text = await loop.run_in_executor(
                    None, stt_from_pcm, speech_cfg, pcm_bytes
                )

                if not user_text:
                    await ws.send_json({
                        "type": "status",
                        "text": "No entendí bien. ¿Puedes repetirlo?"
                    })
                    # Desbloquear el cliente
                    await ws.send_bytes(b"")
                    continue

                farewell = await _process_turn(
                    ws, loop, df_client, session_path, speech_cfg, user_text
                )
                if farewell:
                    break

    except WebSocketDisconnect:
        print("[WS] Cliente desconectado.")
    except Exception as e:
        print(f"[WS] Error inesperado: {e}")
        try:
            await ws.send_json({"type": "status", "text": f"Error: {e}"})
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Health check y cliente web
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    status: dict[str, Any] = {"server": "ok", "region": SPEECH_REGION}
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        status["qdrant"] = f"ok — {info.points_count} vectores"
    except Exception as e:
        status["qdrant"] = f"error: {e}"
    try:
        models = [m["name"] for m in olm.list()["models"]]
        status["ollama"] = models
    except Exception as e:
        status["ollama"] = f"error: {e}"
    return status


@app.get("/", response_class=HTMLResponse)
async def get_client():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitcore_voz_cliente.html")
    try:
        return HTMLResponse(open(html_path, encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse("<h2>fitcore_voz_cliente.html no encontrado en la misma carpeta que el servidor.</h2>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fitcore_server_final:app", host="0.0.0.0", port=3001, reload=True)