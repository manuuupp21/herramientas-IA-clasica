"""
FitCore RAG — Fase 1: Ingesta de conocimiento (proceso offline) — versión optimizada

Pipeline:
  PDFs digitales  → PyMuPDF extracción nativa (instantáneo)
  PDFs escaneados → Tesseract OCR (10-20x más rápido que Deepseek)
  Imágenes        → Tesseract OCR
  .txt / .md      → lectura directa

chunks → nomic-embed-text (Ollama) → Qdrant

Requiere:
  pip install qdrant-client ollama pillow pymupdf pytesseract
  apt install tesseract-ocr tesseract-ocr-spa   ← Linux
  brew install tesseract                         ← Mac
  https://github.com/UB-Mannheim/tesseract/wiki  ← Windows
  docker run -p 6333:6333 qdrant/qdrant
  ollama pull nomic-embed-text
"""

import os
import re
import uuid
import io
from pathlib import Path

import fitz                    # pymupdf
import pytesseract             # tesseract OCR binding
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

# ─── Configuración ───────────────────────────────────────────────────────────

OLLAMA_HOST     = os.getenv("OLLAMA_HOST",        "http://localhost:11434")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")
QDRANT_HOST     = os.getenv("QDRANT_HOST",         "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT",     "6333"))
COLLECTION_NAME = "fitcore_kb"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80
DOCS_DIR        = Path(os.getenv("DOCS_DIR",       "./documentos"))

# Umbral: si una página tiene menos de 50 chars de texto nativo → tratar como escaneada
NATIVE_TEXT_MIN = 50

# Configuración Tesseract: español + inglés, modo página automático
TESS_CONFIG     = "--oem 3 --psm 3"
TESS_LANG       = "spa+eng"

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
olm    = ollama.Client(host=OLLAMA_HOST)


# ─── Setup Qdrant ────────────────────────────────────────────────────────────

def setup_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Colección '{COLLECTION_NAME}' creada.")
    else:
        print(f"Colección '{COLLECTION_NAME}' ya existe.")


# ─── OCR con Tesseract (rápido) ──────────────────────────────────────────────

def ocr_with_tesseract(img: Image.Image, source_hint: str = "") -> str:
    """
    Tesseract es 10-20x más rápido que un modelo LLM para OCR.
    Funciona muy bien en documentos impresos estándar.
    Para manuscritos o tablas complejas, deepseek sigue siendo mejor opción.
    """
    # Preprocesado ligero: escala de grises mejora la precisión de Tesseract
    img_gray = img.convert("L")

    text = pytesseract.image_to_string(
        img_gray,
        lang=TESS_LANG,
        config=TESS_CONFIG
    ).strip()

    if not text or len(text) < 10:
        print(f"  [AVISO] Tesseract sin resultado en {source_hint}")
        return ""

    # Limpiar caracteres basura frecuentes en Tesseract
    text = re.sub(r"[^\S\n]+", " ", text)   # múltiples espacios → uno
    text = re.sub(r"\n{3,}", "\n\n", text)   # múltiples saltos → dos
    return text


# ─── Extracción por tipo de archivo ──────────────────────────────────────────

def extract_from_pdf(path: Path) -> list[dict]:
    """
    Para cada página del PDF:
    - texto nativo (>50 chars) → extracción directa con PyMuPDF (ms)
    - página escaneada         → Tesseract OCR (segundos, no minutos)
    """
    pages = []
    doc   = fitz.open(str(path))

    for i, page in enumerate(doc):
        native = page.get_text("text").strip()

        if len(native) >= NATIVE_TEXT_MIN:
            # ── Texto nativo: instantáneo ────────────────────────────────────
            print(f"  p.{i+1} nativa  → {len(native)} chars")
            pages.append({"text": native, "page": i + 1, "source": path.name})
        else:
            # ── Página escaneada: Tesseract ──────────────────────────────────
            print(f"  p.{i+1} escaneada → Tesseract OCR...")
            pix  = page.get_pixmap(dpi=200)
            img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = ocr_with_tesseract(img, source_hint=f"{path.name} p.{i+1}")
            if text:
                pages.append({"text": text, "page": i + 1, "source": path.name})

    doc.close()
    return pages


def extract_from_image(path: Path) -> list[dict]:
    print(f"  Imagen → Tesseract OCR: {path.name}")
    img  = Image.open(path).convert("RGB")
    text = ocr_with_tesseract(img, source_hint=path.name)
    return [{"text": text, "page": 1, "source": path.name}] if text else []


def extract_from_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"text": text, "page": 1, "source": path.name}] if text else []


def extract_text(path: Path) -> list[dict]:
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
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) <= size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
                current = current[-overlap:] + "\n\n" + para if len(current) > overlap else para
            else:
                for i in range(0, len(para), size - overlap):
                    chunks.append(para[i:i + size])
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) > 30]


# ─── Embeddings + Qdrant ─────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    return olm.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def ingest_document(path: Path):
    print(f"\n{'='*55}")
    print(f"Procesando: {path.name}")

    pages = extract_text(path)
    if not pages:
        print("  Sin contenido extraíble.")
        return

    all_chunks = []
    for page_data in pages:
        for chunk in chunk_text(page_data["text"]):
            all_chunks.append({
                "text":   chunk,
                "page":   page_data["page"],
                "source": page_data["source"],
            })

    print(f"  → {len(pages)} páginas | {len(all_chunks)} chunks")

    points = []
    for i, chunk_data in enumerate(all_chunks):
        print(f"  Embedding {i+1}/{len(all_chunks)}...", end="\r")
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(chunk_data["text"]),
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
    print("FitCore RAG — Fase 1 (optimizada: PyMuPDF + Tesseract)")
    print(f"Documentos: {DOCS_DIR.resolve()}")
    print(f"Embed: {EMBED_MODEL} | Qdrant: {QDRANT_HOST}:{QDRANT_PORT}\n")

    setup_collection()

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"Carpeta '{DOCS_DIR}' creada. Añade documentos y vuelve a ejecutar.")
        return

    supported = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".txt", ".md"}
    docs = [p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in supported]

    if not docs:
        print("No se encontraron documentos.")
        return

    print(f"Documentos encontrados: {len(docs)}")
    for doc in docs:
        ingest_document(doc)

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\n{'='*55}")
    print(f"Ingesta completada. Vectores en Qdrant: {info.points_count}")


if __name__ == "__main__":
    main()