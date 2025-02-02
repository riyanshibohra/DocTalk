import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    """
    try:
        with open(pdf_path, 'rb') as file:
            # Try to create PDF reader
            try:
                reader = PyPDF2.PdfReader(file)
            except Exception as e:
                logger.error(f"Error creating PDF reader: {e}")
                raise ValueError(f"Could not read PDF file: {e}")

           