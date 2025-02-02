from pdf_processor import extract_text_from_pdf, chunk_text
from pinecone_manager import initialize_pinecone_index, store_embeddings
from query_manager import setup_retrieval_chain, get_answer
from speech_to_text import WhisperTranscriber
from text_to_speech import ElevenLabsTTS
from langchain_core.documents import Document
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize PDF processing
    pdf_path = 'sample.pdf'
    
    if not Path(pdf_path).is_file():
        logger.error(f"PDF file not found: {pdf_path}")
        exit(1)
        
    if os.path.getsize(pdf_path) == 0:
        logger.error(f"PDF file is empty: {pdf_path}")
        exit(1)
        
    try:
        extracted_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(extracted_text)
        
        # Initialize Pinecone and store embeddings
        vectorstore = initialize_pinecone_index()
        docs = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]
        store_embeddings(docs, vectorstore)
        
        # Initialize QA chain
        qa_chain = setup_retrieval_chain(vectorstore)
        
        # Initialize speech components
        transcriber = WhisperTranscriber(model_name="base")
        eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        tts = ElevenLabsTTS(api_key=eleven_labs_api_key) if eleven_labs_api_key else None

        # Interactive QA loop
        chat_history = []
        while True:
            input_method = input("\nDo you want to type or speak? (type/speak/quit): ").lower()
            
            if input_method == 'quit':
                break
                
            if input_method == 'speak':
                print("\nListening... (speak for 5 seconds)")
                question = transcriber.transcribe_microphone(duration=5)
                print(f"\nTranscribed question: {question}")
            elif input_method == 'type':
                question = input("\nEnter your question: ")
            else:
                print("Invalid input method. Please choose 'type' or 'speak'")
                continue

            if question.lower() == 'quit':
                break
                
            answer, sources = get_answer(qa_chain, question, chat_history)
            print("\nAnswer:", answer)
            print("\nSources:", [doc.metadata.get('source', 'No source available') for doc in sources])
            
            if tts:
                tts.synthesize_speech(answer)

            chat_history.append((question, answer))
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        exit(1) 