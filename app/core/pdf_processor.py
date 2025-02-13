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
            
        # Clean the text
        text = " ".join(text.split())
        
        # Adjust chunk size for document length
        text_length = len(text)
        if text_length < chunk_size:
            chunk_size = max(100, text_length // 3)  # Smaller chunks for small docs
            chunk_overlap = chunk_size // 3
        
        logger.info(f"Using chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]  # More granular separators
        )
        
        chunks = text_splitter.split_text(text)
        
        # Post-process chunks
        processed_chunks = []
        for chunk in chunks:
            # Clean chunk
            chunk = chunk.strip()
            if len(chunk) > 50:  # Only keep substantial chunks
                processed_chunks.append(chunk)
        
        logger.info(f"Created {len(processed_chunks)} non-empty chunks")
        
        if processed_chunks:
            logger.info(f"First chunk sample: {processed_chunks[0][:200]}...")
            
        return processed_chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise 