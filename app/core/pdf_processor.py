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
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
            logger.info(f"Page {i+1}: extracted {len(page_text)} characters")
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        if len(text) < 100:  # Log full text if it's very short
            logger.info(f"Extracted text: {text}")
        else:
            logger.info(f"Text sample: {text[:100]}...")
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
            
        # Adjust chunk size for small documents
        text_length = len(text)
        if text_length < chunk_size:
            chunk_size = max(100, text_length // 2)  # Use smaller chunks for small documents
            chunk_overlap = chunk_size // 4  # Adjust overlap accordingly
            
        logger.info(f"Using chunk_size: {chunk_size}, overlap: {chunk_overlap}")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")
        
        # Log first chunk for debugging
        if chunks:
            logger.info(f"First chunk sample: {chunks[0][:100]}...")
            
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise 