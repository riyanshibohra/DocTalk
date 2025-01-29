from pdf_processor import extract_text_from_pdf, chunk_text
from pinecone_manager import initialize_pinecone_index, store_embeddings

# Example usage
if __name__ == "__main__":
    pdf_path = 'sample.pdf'  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(extracted_text)
    vectorstore = initialize_pinecone_index()
    store_embeddings(chunks, vectorstore) 