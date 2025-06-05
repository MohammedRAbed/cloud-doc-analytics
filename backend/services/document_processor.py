import fitz  # PyMuPDF
import docx
from typing import Tuple

def extract_pdf_content(file_path: str) -> Tuple[str, str]:
    """
    Extract the title and full text from a PDF file.
    Title = first line of text (for simplicity).
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    title = text.split('\n')[0].strip() if text else "Untitled PDF"
    return title, text

def extract_docx_content(file_path: str) -> Tuple[str, str]:
    """
    Extract the title and full text from a Word (.docx) file.
    Title = first line of text (for simplicity).
    """
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    title = text.split('\n')[0].strip() if text else "Untitled DOCX"
    return title, text
