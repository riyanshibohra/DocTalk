import PyPDF2
from typing import List, Dict
import os

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of approximately chunk_size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extract basic metadata from PDF."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            info = pdf_reader.metadata
            return {
                'title': info.get('/Title', os.path.basename(pdf_path)),
                'author': info.get('/Author', 'Unknown'),
                'creation_date': info.get('/CreationDate', 'Unknown')
            }

    def process_document(self, pdf_path: str) -> Dict:
        """Process a PDF document and return text chunks and metadata."""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        metadata = self.extract_metadata(pdf_path)
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'source': pdf_path
        } 