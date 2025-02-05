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

            # Extract text from all pages
            text = ''
            for page in reader.pages:
                try:
                    text += page.extract_text() + '\n'
                except Exception as e:
                    logger.error(f"Error extracting text from page: {e}")
                    continue

            if not text.strip():
                logger.warning("No text was extracted from the PDF")
                
            logger.info(f"Extracted text: {text[:100]}...")  # Log the first 100 characters
            return text
            
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

# Function to chunk text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks of specified size
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text successfully split into {len(chunks)} chunks")
        logger.info(f"Number of chunks created: {len(chunks)}")  # Log the number of chunks
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise 