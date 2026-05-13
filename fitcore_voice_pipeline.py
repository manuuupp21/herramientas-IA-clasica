"""
FitCore - Pipeline de voz end-to-end
Azure Speech STT → Dialogflow ES detect_intent → Azure TTS

Dependencias:
    pip install azure-cognitiveservices-speech google-cloud-dialogflow

Variables de entorno necesarias:
    AZURE_SPEECH_KEY      → clave de Azure Speech Services
    AZURE_SPEECH_REGION   → región, ej: westeurope
    GOOGLE_APPLICATION_CREDENTIALS → ruta al JSON de cuenta de servicio GCP
    DIALOGFLOW_PROJECT_ID → ID del proyecto de Google Cloud
"""

import os
import uuid
import azure.cognitiveservices.speech as speechsdk
from google.cloud import dialogflow

# ─── Configuración ───────────────────────────────────────────────────────────

SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY", "TU_AZURE_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "westeurope")
PROJECT_ID    = os.getenv("DIALOGFLOW_PROJECT_ID", "TU_PROJECT_ID")
LANGUAGE      = "es-ES"
TTS_VOICE     = "es-ES-AlvaroNeural"   # Cambia a "es-ES-ElviraNeural" si prefieres voz femenina

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", "service_account.json"
)

# ─── Clase principal del pipeline ────────────────────────────────────────────

class FitCoreVoiceBot:
    """
    Pipeline de voz end-to-end para FitCore.

    Flujo por turno:
      1. Azure STT  → escucha el micrófono y devuelve texto
      2. Dialogflow → detecta la intención y devuelve fulfillment_text
      3. Azure TTS  → sintetiza la respuesta con voz neural y la reproduce
    """

    def __init__(self):
        # Dialogflow: crear sesión única por conversación
        self.df_client  = dialogflow.SessionsClient()
        self.session_id = str(uuid.uuid4())
        self.session    = self.df_client.session_path(PROJECT_ID, self.session_id)

        # Azure Speech: configuración compartida para STT y TTS
        self.speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY,
            region=SPEECH_REGION
        )
        self.speech_config.speech_recognition_language = LANGUAGE
        self.speech_config.speech_synthesis_voice_name = TTS_VOICE

        # ── Mejoras de precisión STT ──────────────────────────────────────────
        # Fuerza el modelo "detailed" en lugar del base (mayor precisión en es-ES)
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed

        # Activa el modo de dictado: entiende puntuación y pausas naturales mejor
        # que el modo "conversation" por defecto
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_RecoMode, "DICTATION"
        )

        # Habilita la segmentación por silencio más larga (ms):
        # por defecto 500ms, subimos a 1200ms para evitar cortes prematuros
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1200"
        )

        # Puntuación automática: el modelo inserta comas y puntos correctamente
        self.speech_config.set_service_property(
            "punctuation", "explicit",
            speechsdk.ServicePropertyChannel.UriQueryParameter
        )

        print(f"FitCore VoiceBot listo | Sesión: {self.session_id[:8]}...")

    # ── 1. Azure STT ─────────────────────────────────────────────────────────

    def listen(self) -> str | None:
        """
        Escucha el micrófono y devuelve el texto reconocido.
        Usa PhraseList para mejorar el reconocimiento de términos del dominio.
        """
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        recognizer   = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # Ayudas de dominio: cuantas más frases del dominio, mejor el reconocimiento
        phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        domain_phrases = [
            # Entidades con Regexp - importantes para el slot-fill
            "PED-00123", "PED-00456", "PED-00789",
            "INC-0042", "INC-0099", "INC-0010",
            # Términos de negocio
            "FitCore", "proteína", "creatina", "suplemento",
            "incidencia", "duplicado", "hipertrofia", "definición",
            "L-Carnitina", "beta-alanina", "citrulina", "whey",
            # Frases de intent frecuentes
            "estado de mi pedido", "abrir una incidencia",
            "plan nutricional", "plan de entrenamiento",
            "pérdida de peso", "ganancia muscular", "resistencia",
        ]
        for term in domain_phrases:
            phrase_list.addPhrase(term)

        print("\nEscuchando... (habla ahora)")
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"  Tú: {result.text}")
            return result.text

        if result.reason == speechsdk.ResultReason.NoMatch:
            print("  No se detectó voz.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            print(f"  STT cancelado: {details.reason}")
            if details.reason == speechsdk.CancellationReason.Error:
                print(f"  Error STT: {details.error_details}")
        return None

    # ── 2. Dialogflow ES ─────────────────────────────────────────────────────

    def detect_intent(self, text: str) -> tuple[str, str]:
        """
        Envía el texto a Dialogflow y devuelve (fulfillment_text, intent_name).
        La sesión mantiene los contextos activos entre llamadas.
        """
        response = self.df_client.detect_intent(
            request={
                "session": self.session,
                "query_input": {
                    "text": {
                        "text": text,
                        "language_code": LANGUAGE,
                    }
                },
            }
        )

        query_result  = response.query_result
        fulfillment   = query_result.fulfillment_text
        intent_name   = query_result.intent.display_name if query_result.intent else "desconocido"
        confidence    = query_result.intent_detection_confidence

        print(f"  Intent: {intent_name} ({confidence:.0%})")
        print(f"  Bot: {fulfillment}")
        return fulfillment, intent_name

    # ── 3. Azure TTS ─────────────────────────────────────────────────────────

    def speak(self, text: str, use_ssml: bool = True) -> bool:
        """
        Sintetiza y reproduce el texto con Azure Neural TTS.
        Con use_ssml=True aplica prosodia optimizada para respuestas del bot.
        """
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        synthesizer  = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        if use_ssml:
            # Prosodia ligeramente más lenta y tono medio-alto: natural para un asistente
            content = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="es-ES">
                <voice name="{TTS_VOICE}">
                    <prosody rate="0.95" pitch="+2%">
                        {text}
                    </prosody>
                </voice>
            </speak>"""
            result = synthesizer.speak_ssml_async(content).get()
        else:
            result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return True

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            print(f"  Error TTS: {details.reason} - {details.error_details}")
        return False

    # ── Bucle principal ───────────────────────────────────────────────────────

    def run(self):
        """
        Bucle de conversación.
        - Escucha al usuario
        - Consulta Dialogflow
        - Responde con voz
        - Termina al detectar intención de despedida
        """
        EXIT_INTENTS = {"Despedida"}
        EXIT_WORDS   = {"adiós", "hasta luego", "salir", "terminar", "exit"}

        print("=" * 52)
        print("   FitCore — Asistente de voz")
        print("=" * 52)

        # Saludo inicial: reproduce directamente sin pasar por Dialogflow
        greeting = ("¡Hola! Bienvenido a FitCore, tu asistente de soluciones "
                    "deportivas y nutrición personal. ¿En qué puedo ayudarte?")
        print(f"\n  Bot: {greeting}")
        self.speak(greeting)

        while True:
            user_text = self.listen()

            if user_text is None:
                self.speak("No te he escuchado bien. ¿Puedes repetirlo, por favor?")
                continue

            # Salida por palabra clave (por si el intent no se detecta)
            if any(w in user_text.lower() for w in EXIT_WORDS):
                farewell = "¡Hasta pronto! Ha sido un placer ayudarte. ¡Mucho ánimo!"
                print(f"\n  Bot: {farewell}")
                self.speak(farewell)
                break

            response_text, intent_name = self.detect_intent(user_text)

            if not response_text:
                response_text = "Lo siento, no he podido procesar tu solicitud. ¿Puedes intentarlo de nuevo?"

            self.speak(response_text)

            # Salida por intent de despedida
            if intent_name in EXIT_INTENTS:
                break

        print("\nConversación finalizada.")


# ─── Punto de entrada ────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = FitCoreVoiceBot()
    bot.run()
