from pdf_processor import extract_text_from_pdf, chunk_text
from pinecone_manager import initialize_pinecone_index, store_embeddings
from query_manager import setup_retrieval_chain, get_answer
from langchain_core.documents import Document

# Example usage
if __name__ == "__main__":
    # Process PDF and store embeddings
    pdf_path = 'sample.pdf'  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(extracted_text)

    # Initialize Pinecone index
    vectorstore = initialize_pinecone_index()

    # Convert chunks to Document objects
    docs = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]

    # Store embeddings in Pinecone
    result = store_embeddings(docs, vectorstore)
    print(result)  # Log the result of storing embeddings

    # Set up retrieval chain
    qa_chain = setup_retrieval_chain(vectorstore)

    # Example Q&A
    chat_history = []
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        answer, sources = get_answer(qa_chain, question, chat_history)
        print("\nAnswer:", answer)
        print("\nSources:", [doc.metadata['source'] for doc in sources])
        
        # Update chat history
        chat_history.append((question, answer)) 