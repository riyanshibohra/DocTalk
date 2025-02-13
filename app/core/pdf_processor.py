from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file
    """
    try:
        logger.info(f"Opening PDF file: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

# Function to chunk text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks of specified size
    """
    try:
        if not text:
            logger.error("No text provided for chunking")
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise 