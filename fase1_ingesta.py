"""
FitCore RAG — Fase 1: Ingesta de conocimiento (proceso offline)

Pipeline:
  documentos/imágenes → Deepseek OCR (Ollama) → chunks → embeddings
  (nomic-embed-text via Ollama) → Qdrant (vector DB local)

Requiere:
  pip install qdrant-client ollama pillow pymupdf python-docx
  docker run -p 6333:6333 qdrant/qdrant          ← Qdrant local
  ollama pull deepseek-ocr:3b
  ollama pull nomic-embed-text                    ← embeddings (384 dims, rápido)
"""

import os
import re
import json
import base64
from pathlib import Path

import fitz                          # pymupdf — extrae páginas de PDF como imagen
import ollama                        
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
import io
import uuid

# ─── Configuración ───────────────────────────────────────────────────────────

OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL     = os.getenv("OLLAMA_VISION_MODEL", "deepseek-ocr:3b")  
EMBED_MODEL      = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME  = "fitcore_kb"
CHUNK_SIZE       = 500     # caracteres por chunk
CHUNK_OVERLAP    = 80      # solapamiento entre chunks para no perder contexto
DOCS_DIR         = Path(os.getenv("DOCS_DIR", "./documentos"))  # carpeta con docs

# ─── Cliente Qdrant ───────────────────────────────────────────────────────────

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
olm    = ollama.Client(host=OLLAMA_HOST)


def setup_collection():
    """
    Crea la colección en Qdrant si no existe.
    nomic-embed-text produce vectores de 768 dimensiones.
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Colección '{COLLECTION_NAME}' creada.")
    else:
        print(f"Colección '{COLLECTION_NAME}' ya existe.")


# ─── OCR con Deepseek (Ollama vision) ────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
    """Convierte una imagen PIL a base64 para enviarla a Ollama."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_image(img: Image.Image, source_hint: str = "") -> str:
    """
    Extrae texto de una imagen usando Deepseek vision via Ollama.
    Incluye prompt optimizado para documentos técnicos, tablas y manuscritos.
    """
    prompt = """Eres un motor OCR experto. Extrae TODO el texto visible en esta imagen con máxima fidelidad.
Reglas:
- Preserva la estructura: tablas con | separadores, listas con guiones, títulos en mayúsculas.
- Si hay texto manuscrito, transcríbelo entre [MANUSCRITO: ...].
- Si hay una tabla, conviértela a formato Markdown.
- Si hay gráficos con valores, extrae los datos numéricos como lista.
- No añadas interpretaciones ni resúmenes, solo el texto extraído.
- Si no hay texto legible, responde exactamente: [SIN_TEXTO]
"""
    response = olm.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_to_base64(img)]
        }]
    )
    text = response["message"]["content"].strip()

    # Validación con Regex: descarta respuestas vacías o de error
    if re.fullmatch(r"\[SIN_TEXTO\]|\s*", text):
        print(f"  [AVISO] Sin texto legible en {source_hint}")
        return ""

    # Limpiar artefactos comunes del OCR
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # quita reasoning de deepseek-r1
    text = re.sub(r"\n{3,}", "\n\n", text)                            # normaliza saltos
    return text


# ─── Extracción de texto por tipo de archivo ─────────────────────────────────

def extract_from_pdf(path: Path) -> list[dict]:
    """
    Extrae texto de un PDF.
    - Si la página tiene texto nativo (PDF digital), lo usa directamente.
    - Si la página es imagen escaneada, aplica OCR con Deepseek.
    Devuelve lista de {text, page, source}.
    """
    pages = []
    doc = fitz.open(str(path))

    for i, page in enumerate(doc):
        native_text = page.get_text("text").strip()

        if len(native_text) > 50:
            # Página con texto nativo — no necesita OCR
            print(f"  PDF nativo p.{i+1}: {len(native_text)} chars")
            pages.append({"text": native_text, "page": i + 1, "source": path.name})
        else:
            # Página escaneada o con poco texto — aplicar OCR
            continue   # omitimos OCR para acelerar el proceso           
            #print(f"  PDF escaneado p.{i+1}: aplicando OCR...")
            #pix  = page.get_pixmap(dpi=200)
            #img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            #text = ocr_image(img, source_hint=f"{path.name} p.{i+1}")
            #if text:
                #pages.append({"text": text, "page": i + 1, "source": path.name})

    doc.close()
    return pages


def extract_from_image(path: Path) -> list[dict]:
    """Aplica OCR directamente sobre una imagen (jpg, png, webp)."""
    print(f"  Imagen: aplicando OCR a {path.name}...")
    img  = Image.open(path).convert("RGB")
    text = ocr_image(img, source_hint=path.name)
    return [{"text": text, "page": 1, "source": path.name}] if text else []


def extract_from_txt(path: Path) -> list[dict]:
    """Lee archivos .txt o .md directamente."""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"text": text, "page": 1, "source": path.name}] if text else []


def extract_text(path: Path) -> list[dict]:
    """Router: selecciona el método de extracción según la extensión."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_from_pdf(path)
    elif ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
        return extract_from_image(path)
    elif ext in {".txt", ".md"}:
        return extract_from_txt(path)
    else:
        print(f"  [IGNORADO] Formato no soportado: {path.name}")
        return []


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide el texto en chunks con solapamiento para no perder contexto en los bordes.
    Intenta dividir en párrafos primero; si son muy largos, divide por caracteres.
    """
    # Dividir en párrafos respetando la estructura del documento
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) <= size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
                # Solapamiento: incluir el final del chunk anterior
                current = current[-overlap:] + "\n\n" + para if len(current) > overlap else para
            else:
                # Párrafo más largo que el chunk — dividir por fuerza
                for i in range(0, len(para), size - overlap):
                    chunks.append(para[i:i + size])
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) > 30]  # descartar chunks triviales


# ─── Embeddings ──────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Genera embedding con nomic-embed-text via Ollama (768 dims)."""
    response = olm.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# ─── Ingesta en Qdrant ───────────────────────────────────────────────────────

def ingest_document(path: Path):
    """
    Pipeline completo para un documento:
      extracción → chunking → embedding → upsert en Qdrant
    """
    print(f"\n{'='*55}")
    print(f"Procesando: {path.name}")

    pages = extract_text(path)
    if not pages:
        print("  Sin contenido extraíble.")
        return

    all_chunks = []
    for page_data in pages:
        chunks = chunk_text(page_data["text"])
        for chunk in chunks:
            all_chunks.append({
                "text":   chunk,
                "page":   page_data["page"],
                "source": page_data["source"],
            })

    print(f"  → {len(pages)} páginas, {len(all_chunks)} chunks")

    points = []
    for i, chunk_data in enumerate(all_chunks):
        print(f"  Embedding chunk {i+1}/{len(all_chunks)}...", end="\r")
        vector = embed(chunk_data["text"])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text":   chunk_data["text"],
                "source": chunk_data["source"],
                "page":   chunk_data["page"],
                "chunk":  i,
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\n  ✓ {len(points)} chunks insertados en Qdrant.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("FitCore RAG — Fase 1: Ingesta de conocimiento")
    print(f"Documentos en: {DOCS_DIR.resolve()}")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT} | Colección: {COLLECTION_NAME}")
    print(f"Modelos: OCR={VISION_MODEL} | Embed={EMBED_MODEL}\n")

    setup_collection()

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"Carpeta '{DOCS_DIR}' creada. Añade tus documentos y vuelve a ejecutar.")
        return

    supported = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".txt", ".md"}
    docs = [p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in supported]

    if not docs:
        print("No se encontraron documentos. Añade PDFs, imágenes o .txt en la carpeta.")
        return

    print(f"Documentos encontrados: {len(docs)}")
    for doc in docs:
        ingest_document(doc)

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\n{'='*55}")
    print(f"Ingesta completada. Total vectores en Qdrant: {info.points_count}")


if __name__ == "__main__":
    main()
