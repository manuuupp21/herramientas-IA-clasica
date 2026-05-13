# FitCore – Agente Conversacional Dialogflow ES
## Documentación de Arquitectura y Guía de Importación

---

## 1. Visión General

**FitCore** es un agente de IA conversacional de voz end-to-end para una empresa de soluciones deportivas y nutrición personal. La capa NLU está implementada en **Dialogflow ES** y se integra con un webhook que simula tanto lógica de negocio como consultas a una base de conocimiento técnica mediante **RAG (Retrieval-Augmented Generation)**.

```
Usuario (voz) → STT → Dialogflow ES (NLU) → Webhook (lógica + RAG) → TTS → Usuario (voz)
```

---

## 2. Estructura de Archivos

```
fitcore_agent/
├── entities/
│   ├── Codigo_Cliente.json                  ← Entidad Regexp: [A-Z]{2}\d{6}
│   ├── Codigo_Cliente_entries_es.json
│   ├── Numero_Pedido.json                   ← Entidad Regexp: PED-\d{5}
│   ├── Numero_Pedido_entries_es.json
│   ├── Codigo_Incidencia.json               ← Entidad Regexp: INC-\d{4}
│   ├── Codigo_Incidencia_entries_es.json
│   ├── Tipo_Incidencia.json                 ← Entidad Enum (averia, retraso, etc.)
│   ├── Tipo_Incidencia_entries_es.json
│   ├── Objetivo_Deportivo.json              ← Entidad Enum (perdida_de_peso, etc.)
│   └── Objetivo_Deportivo_entries_es.json
│
├── intents/
│   ├── Default_Welcome_Intent.json          ← Saludo inicial, abre ctx sesion_activa
│   ├── Default_Fallback_Intent.json         ← Respuestas de error multi-variante
│   │
│   ├── Consultar_Estado_Pedido.json         ★ WEBHOOK — slot-fill Numero_Pedido + Codigo_Cliente
│   ├── Consultar_Estado_Pedido_-_yes.json   ← Followup positivo (estático)
│   ├── Consultar_Estado_Pedido_-_no.json    ← Followup negativo (estático)
│   │
│   ├── Abrir_Incidencia.json                ★ WEBHOOK — slot-fill Codigo_Cliente + Tipo_Incidencia
│   ├── Abrir_Incidencia_-_yes.json          ★ WEBHOOK — confirma apertura y genera código INC
│   ├── Abrir_Incidencia_-_no.json           ← Cancela (estático)
│   │
│   ├── Consultar_Incidencia.json            ★ WEBHOOK — slot-fill Codigo_Incidencia
│   │
│   ├── Cerrar_Incidencia.json               ← Slot-fill + confirmación (estático)
│   ├── Cerrar_Incidencia_-_yes.json         ★ WEBHOOK — ejecuta el cierre
│   │
│   ├── Consultar_Productos.json             ← Catálogo estático
│   ├── Consultar_Plan_Nutricional.json      ★ WEBHOOK RAG — slot-fill Objetivo_Deportivo
│   ├── Consultar_Plan_Entrenamiento.json    ★ WEBHOOK RAG — slot-fill Objetivo_Deportivo
│   ├── Informacion_Envios.json              ← Respuesta estática
│   ├── Politica_Devoluciones.json           ← Respuesta estática (enlaza a Abrir_Incidencia)
│   ├── Cambiar_Tema.json                    ← Interrupción: resetContexts=true
│   └── Despedida.json                       ← Cierre de sesión
│
├── webhook.js                               ← Servidor Express con lógica simulada
└── package.json
```

---

## 3. Intents Personalizados (resumen)

| # | Intent | Tipo | Webhook | Entidades usadas |
|---|--------|------|---------|-----------------|
| 1 | Consultar_Estado_Pedido | Multi-turno | ✅ | @Numero_Pedido, @Codigo_Cliente |
| 2 | Consultar_Estado_Pedido - yes/no | Followup | ❌ | — |
| 3 | Abrir_Incidencia | Multi-turno | ✅ | @Codigo_Cliente, @Tipo_Incidencia, @Numero_Pedido |
| 4 | Abrir_Incidencia - yes | Followup | ✅ | — (usa contexto) |
| 5 | Abrir_Incidencia - no | Followup | ❌ | — |
| 6 | Consultar_Incidencia | Simple | ✅ | @Codigo_Incidencia |
| 7 | Cerrar_Incidencia | Multi-turno | ❌ | @Codigo_Incidencia |
| 8 | Cerrar_Incidencia - yes | Followup | ✅ | — (usa contexto) |
| 9 | Consultar_Productos | Simple | ❌ | — |
| 10 | Consultar_Plan_Nutricional | Slot-fill | ✅ RAG | @Objetivo_Deportivo, @sys.number |
| 11 | Consultar_Plan_Entrenamiento | Slot-fill | ✅ RAG | @Objetivo_Deportivo, @sys.number |
| 12 | Informacion_Envios | Simple | ❌ | — |
| 13 | Politica_Devoluciones | Simple | ❌ | — |
| 14 | Cambiar_Tema | Interrupción | ❌ | — |
| 15 | Despedida | Simple | ❌ | — |

**Total intents personalizados: 15** (+ Default Welcome + Default Fallback = 17 totales)

---

## 4. Entidades Personalizadas

| Entidad | Tipo | Patrón / Valores |
|---------|------|-----------------|
| Codigo_Cliente | **Regexp** | `[A-Z]{2}\d{6}` ej: AB123456 |
| Numero_Pedido | **Regexp** | `PED-\d{5}` ej: PED-00123 |
| Codigo_Incidencia | **Regexp** | `INC-\d{4}` ej: INC-0042 |
| Tipo_Incidencia | Enum | averia, retraso, producto_incorrecto, consulta_tecnica |
| Objetivo_Deportivo | Enum | perdida_de_peso, ganancia_muscular, resistencia, definicion |

Entidades de sistema también utilizadas: `@sys.number` (peso, días/semana).

---

## 5. Gestión de Contextos

| Contexto | Origen | Uso |
|----------|--------|-----|
| `sesion_activa` (lifespan: 99) | Default Welcome Intent | Protege todos los intents personalizados; garantiza que el usuario pasó por el saludo |
| `consulta_pedido_activa` (lifespan: 3) | Consultar_Estado_Pedido | Habilita los followups yes/no |
| `incidencia_abierta` (lifespan: 5) | Abrir_Incidencia | Pasa parámetros al followup de confirmación |
| `Abrir_Incidencia-followup` (lifespan: 2) | Abrir_Incidencia | Activa yes/no del flujo de apertura |
| `consulta_incidencia_activa` (lifespan: 3) | Consultar_Incidencia | Reservado para followups futuros |
| `Cerrar_Incidencia-followup` (lifespan: 2) | Cerrar_Incidencia | Activa el yes de cierre |
| `plan_nutricional_activo` (lifespan: 5) | Consultar_Plan_Nutricional | Permite preguntas de seguimiento |
| `plan_entrenamiento_activo` (lifespan: 5) | Consultar_Plan_Entrenamiento | Permite preguntas de seguimiento |

### Gestión de interrupciones
El intent **Cambiar_Tema** tiene `resetContexts: true` y **no** requiere ningún contexto de entrada, por lo que puede activarse desde cualquier punto del flujo. Al ejecutarse, limpia todos los contextos activos y restaura únicamente `sesion_activa`, permitiendo al usuario retomar una conversación limpia.

---

## 6. Flujos Conversacionales Principales

### Flujo A – Consulta de pedido
```
Usuario: "¿dónde está mi pedido?"
  → Slot-fill: número de pedido (PED-XXXXX)
  → Slot-fill: código de cliente (ABXXXXXX)
  → Webhook → respuesta con estado y ETA
  → Followup yes/no para continuar o cerrar
```

### Flujo B – Apertura de incidencia
```
Usuario: "tengo un retraso en mi pedido"
  → Extrae @Tipo_Incidencia = retraso automáticamente
  → Slot-fill: código de cliente
  → Slot-fill: número de pedido (opcional)
  → Confirmación yes/no
  → Webhook → genera código INC-XXXX y lo comunica
```

### Flujo C – Plan personalizado (RAG simulado)
```
Usuario: "quiero un plan nutricional"
  → Slot-fill: objetivo deportivo
  → Webhook → consulta base de conocimiento → respuesta personalizada
  → Sugerencia cruzada: "¿quieres también un plan de entrenamiento?"
```

### Flujo D – Interrupción a mitad de flujo
```
[Usuario en medio de abrir incidencia]
Usuario: "espera, primero quiero saber cuánto tarda el envío"
  → Intent Cambiar_Tema (resetContexts=true) o
  → Intent Informacion_Envios (prioridad 500000, sin contexto requerido)
  → Responde y puede retomar cualquier flujo
```

---

## 7. Guía de Importación en Dialogflow ES

### Paso 1 – Crear el agente
1. Accede a [dialogflow.cloud.google.com](https://dialogflow.cloud.google.com)
2. Crea un agente nuevo → nombre: `FitCore` → idioma: `Spanish (es)` → zona: `europe-west1`

### Paso 2 – Importar entidades
Para cada entidad en la carpeta `entities/`:
1. Ve a **Entities** → botón de menú ⋮ → **Import Entities**
2. Sube el archivo `.json` de definición (ej. `Codigo_Cliente.json`)
3. Sube el archivo `_entries_es.json` correspondiente como entradas

### Paso 3 – Importar intents
Para cada intent en la carpeta `intents/`:
1. Ve a **Intents** → botón de menú ⋮ → **Import Intents** (o restaurar desde ZIP)
2. **Alternativa recomendada**: Usa la API de Dialogflow o el botón de restauración de agente para importar todos de una vez comprimiendo la estructura en un ZIP.

> **Estructura ZIP para restauración completa:**
> ```
> agent.json          ← (metadatos del agente, opcional)
> intents/            ← todos los JSON de intents
> entities/           ← todos los JSON de entidades
> ```

### Paso 4 – Configurar el Webhook
1. Ve a **Fulfillment** → activa **Webhook**
2. Lanza el webhook localmente: `npm install && npm start`
3. Usa **ngrok** para exponer el puerto: `ngrok http 3000`
4. Pega la URL de ngrok en el campo Webhook URL: `https://XXXX.ngrok.io/webhook`

### Paso 5 – Probar en el simulador
Prueba estos utterances de ejemplo:
- `"hola"` → Bienvenida
- `"quiero consultar mi pedido PED-00123"` → Slot-fill código cliente → Webhook
- `"quiero abrir una incidencia"` → Slot-fill datos → Confirmación → Webhook
- `"quiero un plan de entrenamiento para ganar músculo"` → Webhook RAG
- `"olvida lo anterior"` → Reset de contextos
- `"adiós"` → Despedida

---

## 8. Integración de Voz End-to-End (arquitectura sugerida)

```
[Micrófono]
    ↓
[STT: Google Speech-to-Text / Whisper]
    ↓
[Dialogflow ES Detect Intent API]
    ↓ (si webhookUsed=true)
[Webhook FitCore → lógica + RAG]
    ↓
[fulfillmentText]
    ↓
[TTS: Google Text-to-Speech / ElevenLabs]
    ↓
[Altavoz]
```

Para RAG real, sustituir el objeto `planesNutricionales` en `webhook.js` por llamadas a:
- **Vector DB**: Pinecone / Chroma / Weaviate con embeddings de documentos técnicos
- **LLM**: OpenAI GPT-4 / Anthropic Claude para generación de respuesta basada en contexto recuperado

---

*Versión 1.0 — FitCore Conversational Agent — Mayo 2026*
