import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file using pdfplumber
    """
    try:
        logger.info(f"Opening PDF file: {pdf_path}")
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    logger.info(f"Page {i+1}: extracted {len(page_text)} characters")
                    if len(page_text) > 0:
                        logger.info(f"Sample from page {i+1}: {page_text[:100]}...")
                except Exception as e:
                    logger.error(f"Error on page {i+1}: {e}")
                    continue

        if not text.strip():
            logger.error("No text could be extracted from the PDF")
            raise ValueError("PDF appears to be empty or unreadable")

        logger.info(f"Total extracted: {len(text)} characters")
        logger.info(f"Sample of extracted text: {text[:200]}...")
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
        if not text or not text.strip():
            logger.error("No text provided for chunking")
            return []
            
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Adjust chunk size for small documents
        text_length = len(text)
        if text_length < chunk_size:
            chunk_size = max(100, text_length // 2)
            chunk_overlap = chunk_size // 4
            
        logger.info(f"Using chunk_size: {chunk_size}, overlap: {chunk_overlap}")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Validate chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        logger.info(f"Created {len(chunks)} non-empty chunks")
        
        if chunks:
            logger.info(f"First chunk sample: {chunks[0][:100]}...")
            
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise 