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

# Example usage
if __name__ == "__main__":
    # Process PDF and store embeddings
    pdf_path = 'sample.pdf'  # Replace with your PDF file path
    
    # Check if file exists
    if not Path(pdf_path).is_file():
        logger.error(f"PDF file not found: {pdf_path}")
        print(f"Error: Could not find the PDF file at {pdf_path}")
        exit(1)
        
    # Check if file is empty
    if os.path.getsize(pdf_path) == 0:
        logger.error(f"PDF file is empty: {pdf_path}")
        print(f"Error: The PDF file at {pdf_path} is empty")
        exit(1)
        
    try:
        extracted_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        print(f"Error: Could not read the PDF file. Please ensure it is a valid PDF file.")
        exit(1)

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

    # Initialize Whisper transcriber
    transcriber = WhisperTranscriber(model_name="base")

    # Initialize ElevenLabs TTS
    eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not eleven_labs_api_key:
        logger.warning("ELEVEN_LABS_API_KEY not found in .env file. Text-to-speech will be disabled.")
        tts = None
    else:
        try:
            tts = ElevenLabsTTS(api_key=eleven_labs_api_key)
        except ValueError as e:
            logger.error(f"Error initializing ElevenLabs TTS: {e}")
            tts = None

    # Example Q&A with voice input
    chat_history = []
    while True:
        # Ask if user wants to type or speak
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
        
        # Synthesize speech from the answer
        if tts:
            tts.synthesize_speech(answer)

        # Update chat history
        chat_history.append((question, answer)) 