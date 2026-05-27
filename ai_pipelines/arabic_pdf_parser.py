"""
Extract Arabic curriculum PDFs and convert them to structured JSON via Gemini.

Primary extraction uses pdf2text-arabic (RTL layout repair, ligature fixes).
Falls back to pdfplumber (layout + RTL word order) and PyMuPDF block sorting.

Usage:
    python ai_pipelines/arabic_pdf_parser.py --input curriculum_pdfs/book.pdf
    python ai_pipelines/arabic_pdf_parser.py --input curriculum_pdfs/book.pdf --extract-only --output out.txt
    python ai_pipelines/arabic_pdf_parser.py --input curriculum_pdfs/book.pdf --output database_sync/structured.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Literal

import fitz
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from pdf2text_arabic import extract_page as extract_arabic_page
except ImportError:  # pragma: no cover - optional at import time, required at runtime
    extract_arabic_page = None

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "database_sync" / "structured_arabic_curriculum.json"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_CHUNK_CHARS = 12_000
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LAM_ALEF = "\u0644\u0627"
LAM_ALEF_LIGATURE = "\uFEFB"

OcrStrategy = Literal["never", "warn", "auto", "force"]


# ---------------------------------------------------------------------------
# Structured output schema (Gemini response)
# ---------------------------------------------------------------------------
class InteractiveExample(BaseModel):
    scenario: str = Field(
        description="سيناريو تفاعلي من الحياة اليومية (مثال: في السعودية) يطبق المفهوم بشكل مشوق للطلاب."
    )
    question: str = Field(description="سؤال تفاعلي ذكي بناءً على السيناريو المطروح.")
    interactive_step: str = Field(
        description="تلميح أو خطوة يقوم بها الطالب لاكتشاف الإجابة الصحيحة."
    )


class ConceptNode(BaseModel):
    concept_title: str = Field(description="عنوان المفهوم التعليمي الفرعي.")
    simplified_text: str = Field(
        description="شرح مبسط وممتع جداً للمفهوم، بأسلوب تفاعلي يناسب الطلاب الصغار بعيداً عن التعقيد."
    )
    examples: list[InteractiveExample]
    suggested_visual_prompt: str = Field(
        description="وصف تفصيلي باللغة الإنجليزية لإنشاء صورة أو رسمة توضيحية تعبر عن هذا المفهوم بدقة عبر الذكاء الاصطناعي التوليدي."
    )


class ChapterStructure(BaseModel):
    chapter_number: int
    chapter_title: str
    summary: str = Field(description="ملخص سريع وممتع للمنهاج أو الفصل (أسلوب ألعاب/تفاعلي).")
    concepts: list[ConceptNode]


# ---------------------------------------------------------------------------
# Arabic text normalization
# ---------------------------------------------------------------------------
def light_normalize_arabic(text: str) -> str:
    """Unicode cleanup safe for text already in logical Arabic reading order."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\u0640+", "\u0640", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def repair_visual_order_arabic(text: str) -> str:
    """
    Repair PDFs that return Arabic in visual (right-to-left glyph) order.

    Uses the lam-alef ligature + reverse + NFKD approach for legacy extractors.
    """
    if not text:
        return ""

    repaired = text.replace(LAM_ALEF, LAM_ALEF_LIGATURE)
    repaired = unicodedata.normalize("NFKD", repaired[::-1])
    return light_normalize_arabic(repaired)


def _arabic_quality_score(text: str) -> float:
    """Higher is better: rewards Arabic density and penalizes broken spacing/ligatures."""
    if not text.strip():
        return 0.0

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0

    arabic_count = sum(1 for ch in letters if ARABIC_RE.match(ch))
    arabic_ratio = arabic_count / len(letters)

    broken_lam_alef = text.count("لا ") + text.count(" ال ")
    isolated_forms = sum(1 for ch in text if "\uFE70" <= ch <= "\uFEFF")
    short_tokens = sum(
        1
        for token in text.split()
        if len(token) == 1 and ARABIC_RE.search(token) is not None
    )

    penalty = (broken_lam_alef * 0.5) + (isolated_forms * 0.3) + (short_tokens * 0.2)
    return (arabic_ratio * 100.0) - penalty


def _iter_page_indexes(total_pages: int, start_page: int | None, end_page: int | None) -> Iterable[int]:
    start_idx = 0 if start_page is None else max(0, start_page - 1)
    end_idx = total_pages if end_page is None else min(total_pages, end_page)
    for idx in range(start_idx, end_idx):
        yield idx


# ---------------------------------------------------------------------------
# Extraction backends
# ---------------------------------------------------------------------------
def _extract_page_pdf2text_arabic(page: fitz.Page, ocr_strategy: OcrStrategy) -> str:
    if extract_arabic_page is None:
        return ""
    try:
        text, _ = extract_arabic_page(page, ocr_strategy=ocr_strategy)
        return text or ""
    except Exception:
        # Some PDFs trigger upstream extraction edge cases; fall back to other engines.
        return ""


def _extract_page_pdfplumber(page: pdfplumber.page.Page) -> str:
    # layout=True preserves spatial reading order; horizontal_ltr=False favors RTL runs.
    text = page.extract_text(layout=True, horizontal_ltr=False) or ""
    return text.strip()


def _extract_page_pymupdf(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    # Sort top-to-bottom, then right-to-left within each row bucket.
    sorted_blocks = sorted(blocks, key=lambda block: (round(block[1], 1), -block[0]))
    lines = [block[4].strip() for block in sorted_blocks if block[4] and block[4].strip()]
    return "\n".join(lines)


def _extract_best_page_text(
    fitz_page: fitz.Page,
    plumber_page: pdfplumber.page.Page | None,
    ocr_strategy: OcrStrategy,
) -> str:
    candidates: list[tuple[str, str]] = []

    arabic_engine_text = light_normalize_arabic(_extract_page_pdf2text_arabic(fitz_page, ocr_strategy))
    if arabic_engine_text:
        candidates.append(("pdf2text-arabic", arabic_engine_text))

    if plumber_page is not None:
        plumber_text = repair_visual_order_arabic(_extract_page_pdfplumber(plumber_page))
        if plumber_text:
            candidates.append(("pdfplumber", plumber_text))

    pymupdf_text = repair_visual_order_arabic(_extract_page_pymupdf(fitz_page))
    if pymupdf_text:
        candidates.append(("pymupdf", pymupdf_text))

    if not candidates:
        return ""

    _, best_text = max(candidates, key=lambda item: _arabic_quality_score(item[1]))
    return best_text


def extract_arabic_pdf(
    pdf_path: Path,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
    max_pages: int | None = None,
    ocr_strategy: OcrStrategy = "auto",
) -> str:
    """
    Extract Arabic text with RTL-aware layout handling.

    Uses pdf2text-arabic first, then compares pdfplumber/pymupdf fallbacks per page.
    """
    chunks: list[str] = []

    with fitz.open(str(pdf_path)) as doc, pdfplumber.open(str(pdf_path)) as pdf:
        page_indexes = list(_iter_page_indexes(len(doc), start_page, end_page))
        if max_pages is not None:
            page_indexes = page_indexes[:max_pages]

        for page_idx in page_indexes:
            fitz_page = doc[page_idx]
            plumber_page = pdf.pages[page_idx] if page_idx < len(pdf.pages) else None
            page_text = _extract_best_page_text(fitz_page, plumber_page, ocr_strategy)
            if page_text:
                chunks.append(f"--- صفحة {page_idx + 1} ---\n{page_text}")

    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Chunking + Gemini structured analysis
# ---------------------------------------------------------------------------
def _split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    page_sections = re.split(r"(?=--- صفحة \d+ ---)", text)
    page_sections = [section.strip() for section in page_sections if section.strip()]
    if not page_sections:
        page_sections = [text]

    chunks: list[str] = []
    current = ""

    for section in page_sections:
        candidate = f"{current}\n\n{section}".strip() if current else section
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(section) <= max_chars:
            current = section
        else:
            for start in range(0, len(section), max_chars):
                chunks.append(section[start : start + max_chars])
            current = ""

    if current:
        chunks.append(current)

    return chunks


def _merge_chapter_structures(chapters: list[ChapterStructure]) -> ChapterStructure:
    if not chapters:
        raise ValueError("Gemini returned no structured chapter data.")

    first = chapters[0]
    merged_concepts: list[ConceptNode] = []
    for chapter in chapters:
        merged_concepts.extend(chapter.concepts)

    summaries = [chapter.summary.strip() for chapter in chapters if chapter.summary.strip()]
    merged_summary = summaries[0] if len(summaries) == 1 else " ".join(summaries[:3])

    return ChapterStructure(
        chapter_number=first.chapter_number,
        chapter_title=first.chapter_title,
        summary=merged_summary,
        concepts=merged_concepts,
    )


class GeminiAnalysisError(RuntimeError):
    pass


def _coerce_chapter_structure(raw_json: str) -> ChapterStructure:
    """Accept minor model schema drift and coerce into ChapterStructure."""
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise GeminiAnalysisError("Gemini response is not a JSON object.")

    # Exact schema path first.
    try:
        return ChapterStructure.model_validate(data)
    except ValidationError:
        pass

    # Common alternative shapes observed from generative models.
    chapter_number = data.get("chapter_number") or data.get("chapterNo") or data.get("number") or 1
    chapter_title = data.get("chapter_title") or data.get("title") or data.get("chapter") or "الفصل"
    summary = data.get("summary") or data.get("overview") or data.get("intro") or ""

    raw_concepts = (
        data.get("concepts")
        or data.get("modules")
        or data.get("sections")
        or data.get("topics")
        or []
    )
    concepts: list[ConceptNode] = []
    if isinstance(raw_concepts, list):
        for item in raw_concepts:
            if not isinstance(item, dict):
                continue
            examples_raw = item.get("examples") or item.get("activities") or item.get("questions") or []
            examples: list[InteractiveExample] = []
            if isinstance(examples_raw, list):
                for ex in examples_raw:
                    if not isinstance(ex, dict):
                        continue
                    examples.append(
                        InteractiveExample(
                            scenario=str(ex.get("scenario") or ex.get("context") or "مثال تفاعلي"),
                            question=str(ex.get("question") or ex.get("prompt") or "ما الإجابة الصحيحة؟"),
                            interactive_step=str(ex.get("interactive_step") or ex.get("step") or "جرّب ثم تحقق من الحل."),
                        )
                    )

            concepts.append(
                ConceptNode(
                    concept_title=str(item.get("concept_title") or item.get("title") or item.get("name") or "مفهوم"),
                    simplified_text=str(item.get("simplified_text") or item.get("explanation") or item.get("description") or ""),
                    examples=examples if examples else [
                        InteractiveExample(
                            scenario="موقف يومي مرتبط بالدرس.",
                            question="كيف نطبق هذا المفهوم؟",
                            interactive_step="طبّق الفكرة بخطوة بسيطة ثم راجع الناتج.",
                        )
                    ],
                    suggested_visual_prompt=str(
                        item.get("suggested_visual_prompt")
                        or item.get("visual_prompt")
                        or "Educational illustration for this concept in a Saudi classroom context."
                    ),
                )
            )

    coerced = {
        "chapter_number": int(chapter_number) if str(chapter_number).isdigit() else 1,
        "chapter_title": str(chapter_title),
        "summary": str(summary),
        "concepts": [concept.model_dump(mode="json") for concept in concepts],
    }
    return ChapterStructure.model_validate(coerced)


@retry(
    retry=retry_if_exception_type(GeminiAnalysisError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _analyze_chunk_with_gemini(
    client: genai.Client,
    raw_arabic_text: str,
    *,
    model: str,
    chunk_index: int,
    chunk_total: int,
) -> ChapterStructure:
    prompt = f"""
    قم بتحليل هذا الجزء ({chunk_index}/{chunk_total}) من المنهج الدراسي المدرسي باللغة العربية.
    أعد هيكلة المحتوى إلى مكونات تفاعلية، مبسطة، وممتعة للغاية لتناسب عقول الطلاب.
    يجب أن تكون جميع النصوص المستخرجة والشروحات والأسئلة باللغة العربية الفصحى المبسطة والمشوقة.

    النص المراد تحليله:
    {raw_arabic_text}
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "أنت مهندس تعليمي وخبير في تبسيط المناهج العربية وجعلها تفاعلية كالألعاب الرقمية. "
                "تقوم بصياغة المفاهيم والملخصات بدقة وبأسلوب قصصي تفاعلي باللغة العربية."
            ),
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )

    if not response.text:
        raise GeminiAnalysisError("Gemini returned an empty response.")

    try:
        return _coerce_chapter_structure(response.text)
    except (ValidationError, json.JSONDecodeError, GeminiAnalysisError) as exc:
        raise GeminiAnalysisError(f"Gemini response failed schema validation: {exc}") from exc


def analyze_with_gemini(
    raw_arabic_text: str,
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
) -> ChapterStructure:
    client = genai.Client(api_key=api_key)
    chunks = _split_text_into_chunks(raw_arabic_text)
    chapter_parts: list[ChapterStructure] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"Sending chunk {idx}/{len(chunks)} to Gemini...")
        chapter_parts.append(
            _analyze_chunk_with_gemini(
                client,
                chunk,
                model=model,
                chunk_index=idx,
                chunk_total=len(chunks),
            )
        )

    return _merge_chapter_structures(chapter_parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Arabic PDF text and optionally convert it to structured curriculum JSON."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input PDF file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output path (.json for structured output, .txt for raw extraction).",
    )
    parser.add_argument("--extract-only", action="store_true", help="Only extract text; skip Gemini structuring.")
    parser.add_argument("--start-page", type=int, default=None, help="Start page (1-based, inclusive).")
    parser.add_argument("--end-page", type=int, default=None, help="End page (1-based, inclusive).")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages processed.")
    parser.add_argument(
        "--ocr-strategy",
        choices=["never", "warn", "auto", "force"],
        default="auto",
        help="OCR strategy passed to pdf2text-arabic for scanned/image pages.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help="Gemini model name for structuring.",
    )
    return parser.parse_args()


def _resolve_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Set it in your environment or in a .env file at the project root."
        )
    return api_key


def main() -> None:
    args = parse_args()
    pdf_path = args.input if args.input.is_absolute() else (PROJECT_ROOT / args.input)
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file: {pdf_path}")
    if extract_arabic_page is None:
        raise ImportError(
            "pdf2text-arabic is required. Install dependencies with: "
            "pip install -r ai_pipelines/requirements.txt"
        )

    print("Extracting Arabic text from PDF (RTL-aware)...")
    raw_text = extract_arabic_pdf(
        pdf_path,
        start_page=args.start_page,
        end_page=args.end_page,
        max_pages=args.max_pages,
        ocr_strategy=args.ocr_strategy,
    )

    if not raw_text.strip():
        raise RuntimeError("No extractable Arabic text found. Try --ocr-strategy auto or force for scanned PDFs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.extract_only or output_path.suffix.lower() == ".txt":
        output_path.write_text(raw_text, encoding="utf-8")
        print(f"Saved extracted text to: {output_path}")
        return

    api_key = _resolve_api_key()
    print("Converting extracted text to structured JSON via Gemini...")
    structured = analyze_with_gemini(raw_text, api_key=api_key, model=args.model)

    output_path.write_text(
        json.dumps(structured.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Success! Saved structured curriculum to: {output_path}")


if __name__ == "__main__":
    main()
