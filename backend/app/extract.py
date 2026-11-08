"""Turn uploaded photos, handwriting, PDFs and documents into plain question text.

The extracted text is returned to the user to review and edit before it is sent,
then flows through the normal solve-and-verify pipeline. Reading images needs a
Groq vision model (LLM_PROVIDER=groq and GROQ_API_KEY). Plain text, .docx and PDFs
with a text layer work without any AI call.

Environment variables:
    GROQ_VISION_MODEL   default "meta-llama/llama-4-scout-17b-16e-instruct"
                        (check the Groq console for current vision models)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET

import requests
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

MAX_FILES = 5
MAX_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 10
MAX_VISION_PAGES = 6
MAX_CHARS = 6000
TEXT_EXTENSIONS = {".txt", ".md", ".tex", ".csv"}

Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard (errors above 2x this)

_PROMPT = (
    "You are an OCR engine for mathematics. Transcribe ALL text and mathematics in this image exactly as "
    "written, including handwriting.\n"
    "- Write mathematics in plain, readable text format that a student can easily read (e.g., x^2, sqrt(x), pi, integral from 0 to x, fraction 1/2).\n"
    "- Do NOT use LaTeX commands (like \\frac, \\int, \\sqrt) or dollar signs. Just write the math naturally.\n"
    "- Keep question numbers and multiple-choice options such as (A), (B), (C), (D) on separate lines.\n"
    "- Put a brief description of any diagram or figure in [square brackets].\n"
    "- Write [illegible] for anything you cannot read; never guess missing parts.\n"
    "- Do NOT solve the problem, explain, or add commentary. Output only the transcription."
)


class ExtractError(Exception):
    """A problem with one uploaded file, safe to show to the user."""


def vision_available() -> bool:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "groq" and bool(os.getenv("GROQ_API_KEY"))


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

def _pil_to_jpeg(img: Image.Image) -> bytes:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, "white")
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    img.thumbnail((1800, 1800), Image.LANCZOS)
    for quality in (88, 75, 60):
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality, optimize=True)
        if buffer.tell() <= 3_000_000:  # Groq limits base64 images to about 4 MB
            break
    return buffer.getvalue()


def _image_bytes_to_jpeg(data: bytes) -> bytes:
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        raise ExtractError("HEIC photos aren't supported. Take a screenshot or share it as JPG/PNG.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)  # treat "suspiciously large" as an error
            img = Image.open(io.BytesIO(data))
            img.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ExtractError("This image is too large. Please use a smaller photo.") from None
    except Exception:  # noqa: BLE001 - corrupt or unsupported image
        raise ExtractError("This image could not be opened. Try JPG or PNG.") from None
    return _pil_to_jpeg(img)


def _clean_transcript(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    return text


def _vision(jpeg: bytes) -> str:
    if not vision_available():
        raise ExtractError("Reading images needs the Groq provider (set LLM_PROVIDER=groq and GROQ_API_KEY).")
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
        ]}],
        "temperature": 0,
        "max_completion_tokens": 2000,
    }
    try:
        response = requests.post(GROQ_URL, json=payload, timeout=(5, 60),
                                 headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
    except requests.RequestException:
        raise ExtractError("Could not reach the image reader. Please try again.") from None
    if response.status_code == 429:
        raise ExtractError("The image reader is busy (rate-limited). Try again in a minute.")
    if response.status_code in (400, 404):
        logger.error("Vision request rejected (%s): %s", response.status_code, response.text[:300])
        raise ExtractError("The vision model rejected this request. The image may be too large, or "
                           "GROQ_VISION_MODEL may not be a current vision model.")
    if response.status_code >= 400:
        raise ExtractError(f"The image reader failed (error {response.status_code}).")
    try:
        content = response.json()["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, ValueError):
        raise ExtractError("The image reader returned an unexpected response.") from None
    return _clean_transcript(content)


# --------------------------------------------------------------------------- #
# PDF and DOCX and text
# --------------------------------------------------------------------------- #

# Private-use glyphs and math symbols usually mean the PDF text layer garbles formulas.
_MATHY = re.compile("[\ue000-\uf8ff∫∑∏√∂∞≤≥≠±∈∉⊂⊃∪∩⇒∴∵αβγθλμπσφωΔΣΩ]")


def _extract_pdf(data: bytes) -> tuple[str, list[str]]:
    import pypdfium2 as pdfium

    notes: list[str] = []
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception:  # noqa: BLE001 - encrypted or corrupt
        raise ExtractError("This PDF could not be opened (it may be password-protected or damaged).") from None
    try:
        total = len(pdf)
        if total > MAX_PDF_PAGES:
            notes.append(f"Only the first {MAX_PDF_PAGES} of {total} pages were read.")
        parts: list[str] = []
        vision_pages = 0
        for index in range(min(total, MAX_PDF_PAGES)):
            page = pdf[index]
            text = ""
            try:
                text = page.get_textpage().get_text_range().strip()
            except Exception:  # noqa: BLE001
                pass
            scanned = len(text) < 40
            mathy = len(_MATHY.findall(text)) >= 3
            if (scanned or mathy) and vision_available() and vision_pages < MAX_VISION_PAGES:
                vision_pages += 1
                jpeg = _pil_to_jpeg(page.render(scale=2.0).to_pil())
                parts.append(_vision(jpeg))
            elif scanned:
                if vision_pages >= MAX_VISION_PAGES:
                    notes.append(f"Page {index + 1} was skipped (too many scanned pages).")
                else:
                    raise ExtractError("This looks like a scanned PDF. Reading it needs the Groq provider.")
            else:
                if mathy:
                    notes.append(f"Page {index + 1}: formulas may be garbled; check them or upload a photo.")
                parts.append(text)
        return "\n\n".join(p for p in parts if p), notes
    finally:
        pdf.close()


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def _extract_docx(data: bytes) -> tuple[str, list[str]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        info = archive.getinfo("word/document.xml")
    except (zipfile.BadZipFile, KeyError):
        raise ExtractError("This isn't a valid .docx file.") from None
    if info.file_size > 20 * 1024 * 1024:
        raise ExtractError("This document is too large to read.")
    try:
        root = ET.fromstring(archive.read(info))
    except ET.ParseError:
        raise ExtractError("This .docx file is damaged.") from None
    paragraphs = []
    for para in root.iter(_W + "p"):
        pieces = []
        for node in para.iter():
            if node.tag in (_W + "t", _M + "t") and node.text:
                pieces.append(node.text)
            elif node.tag == _W + "tab":
                pieces.append("\t")
            elif node.tag == _W + "br":
                pieces.append("\n")
        line = "".join(pieces).strip()
        if line:
            paragraphs.append(line)
    notes = ["Images inside the document were not read."] if any(n.startswith("word/media/") for n in archive.namelist()) else []
    return "\n".join(paragraphs), notes


def _extract_text(data: bytes) -> tuple[str, list[str]]:
    if b"\x00" in data[:4096]:
        raise ExtractError("This file isn't plain text.")
    return data.decode("utf-8", errors="replace").strip(), []


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _sniff(name: str, data: bytes) -> str:
    """Decide the file type from its CONTENT, not its name."""
    ext = os.path.splitext(name.lower())[1]
    if data[:4] == b"%PDF":
        return "pdf"
    if (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[:6] in (b"GIF87a", b"GIF89a")
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP") or data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1")):
        return "image"
    if data[:2] == b"PK":
        if ext == ".docx":
            return "docx"
        raise ExtractError("Only .docx Word files are supported (not .xlsx, .pptx or .zip).")
    if ext in TEXT_EXTENSIONS:
        return "text"
    raise ExtractError("Unsupported file type. Use a photo (JPG/PNG/WEBP), PDF, .docx, .txt, .md, .tex or .csv.")


def extract_one(name: str, data: bytes) -> tuple[str, str, list[str]]:
    """Return (kind, text, notes) for one file."""
    if not data:
        raise ExtractError("This file is empty.")
    if len(data) > MAX_BYTES:
        raise ExtractError(f"This file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    kind = _sniff(name, data)
    if kind == "image":
        return kind, _vision(_image_bytes_to_jpeg(data)), []
    if kind == "pdf":
        text, notes = _extract_pdf(data)
        return kind, text, notes
    if kind == "docx":
        text, notes = _extract_docx(data)
        return kind, text, notes
    text, notes = _extract_text(data)
    return kind, text, notes


def extract_files(files: list[tuple[str, bytes]]) -> dict:
    """Extract text from up to MAX_FILES files. Never raises for a bad file; reports it per item."""
    if not files:
        raise ExtractError("No file was uploaded.")
    if len(files) > MAX_FILES:
        raise ExtractError(f"Upload at most {MAX_FILES} files at a time.")
    items: list[dict] = []
    blocks: list[tuple[str, str]] = []
    for name, data in files:
        item: dict = {"name": name, "ok": False, "kind": None, "chars": 0, "error": None, "notes": []}
        try:
            kind, text, notes = extract_one(name, data)
            item["kind"], item["notes"] = kind, notes
            if not text.strip():
                raise ExtractError("No text was found in this file.")
            item["ok"], item["chars"] = True, len(text)
            blocks.append((name, text.strip()))
        except ExtractError as error:
            item["error"] = str(error)
        except Exception:  # noqa: BLE001 - never leak internals
            logger.exception("Unexpected error extracting %s", name)
            item["error"] = "Could not read this file."
        items.append(item)

    if len(blocks) > 1:
        combined = "\n\n".join(f"[{name}]\n{text}" for name, text in blocks)
    else:
        combined = blocks[0][1] if blocks else ""
    truncated = len(combined) > MAX_CHARS
    if truncated:
        cut = combined[:MAX_CHARS]
        combined = cut[:cut.rfind("\n")] if "\n" in cut[MAX_CHARS // 2:] else cut
    return {"text": combined, "items": items, "truncated": truncated, "vision": vision_available()}
