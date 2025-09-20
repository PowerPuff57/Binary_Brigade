import os
import logging
import PyPDF2
import pdfplumber
import docx2txt
from docx import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document extraction and parsing"""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
        return text.strip()

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        try:
            text = docx2txt.process(file_path)
            if not text:
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            text = ""
        return text.strip()

    @staticmethod
    def extract_text(file_path: str) -> str:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            return DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return DocumentProcessor.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
