from .core.pdf_processor import extract_text_from_pdf, chunk_text
from .core.pinecone_manager import PineconeManager
from .core.query_manager import setup_retrieval_chain
from .core.speech_to_text import WhisperTranscriber
from .core.text_to_speech import ElevenLabsTTS
from langchain_core.documents import Document
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from pathlib import Path
import uvicorn
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="DocTalk API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    pinecone_manager = PineconeManager()
    vectorstore = pinecone_manager.vectorstore
    qa_chain = setup_retrieval_chain(vectorstore)
    transcriber = WhisperTranscriber(model_name="base")
    tts = ElevenLabsTTS(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    raise

class Question(BaseModel):
    text: str

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        logger.info(f"Processing PDF file: {file.filename}")
        with open("temp.pdf", "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Process PDF
        extracted_text = extract_text_from_pdf("temp.pdf")
        logger.info(f"Extracted text length: {len(extracted_text)}")
        
        if not extracted_text:
            logger.error("No text extracted from PDF")
            return {"error": "Could not extract text from PDF"}
            
        chunks = chunk_text(extracted_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks created from text")
            return {"error": "Could not create chunks from PDF text"}

        # Store chunks in Pinecone
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file.filename,
                    "chunk_id": i,
                    "text": chunk
                }
            )
            documents.append(doc)
        
        try:
            # Try to delete existing vectors
            pinecone_manager.delete_all_vectors()
            logger.info("Deleted existing vectors")
            
            # Add new documents
            ids = vectorstore.add_documents(documents)
            logger.info(f"Successfully stored {len(ids)} documents in Pinecone")
            
            # Clean up
            os.remove("temp.pdf")
            
            return {
                "message": "PDF processed successfully",
                "chunks": len(chunks),
                "stored_documents": len(ids),
                "text_length": len(extracted_text)
            }
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        return {"error": str(e)}

@app.post("/api/ask")
async def ask_question(question: Question):
    logger.info(f"Query sent to AI: {question.text}")
    try:
        # First verify documents exist in vectorstore
        index = pinecone_manager.pc.Index(pinecone_manager.index_name)
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        # Get relevant documents
        relevant_docs = vectorstore.similarity_search(
            question.text,
            k=5
        )
        
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        
        if len(relevant_docs) == 0:
            # Try direct query with embeddings
            query_embedding = pinecone_manager.embeddings.embed_query(question.text)
            query_response = index.query(
                vector=query_embedding,
                top_k=5,
                namespace="",
                include_metadata=True
            )
            
            if query_response.matches:
                relevant_docs = [
                    Document(
                        page_content=match.metadata.get('text', ''),
                        metadata=match.metadata
                    )
                    for match in query_response.matches
                ]
                logger.info(f"Retrieved {len(relevant_docs)} documents through direct query")
        
        if len(relevant_docs) == 0:
            return {"answer": "I'm sorry, I couldn't find any relevant information in the document. Please try rephrasing your question."}
            
        # Log retrieved documents for debugging
        for i, doc in enumerate(relevant_docs):
            logger.info(f"Document {i+1} content: {doc.page_content[:200]}")
        
        # Use the chain to get the answer
        response = qa_chain.invoke({
            "question": question.text,
            "chat_history": [],
            "context": "\n\n".join(doc.page_content for doc in relevant_docs)
        })
        
        # Format the response
        formatted_answer = format_response(response["answer"])
        return {"answer": formatted_answer}
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        logger.exception("Full traceback:")
        return {"error": str(e)}

def format_response(answer: str) -> str:
    # Split the answer into sections
    sections = answer.split("\n")
    formatted = "<h3>Summary of the Document:</h3><ul>"
    
    for section in sections:
        if section.strip():  # Avoid empty lines
            formatted += f"<li>{section.strip()}</li>"
    
    formatted += "</ul>"
    return formatted

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded audio temporarily
        with open("temp_audio.wav", "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Transcribe
        text = transcriber.transcribe_audio("temp_audio.wav")
        
        # Clean up
        os.remove("temp_audio.wav")
        
        return {"text": text}
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return {"error": str(e)}

@app.post("/api/synthesize")
async def synthesize_speech(text: str):
    try:
        audio_path = tts.synthesize_speech(text)
        return {"audio_path": str(audio_path)}
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    return {
        "message": "DocTalk API is running",
        "endpoints": {
            "PDF Processing": "/api/process-pdf",
            "Ask Questions": "/api/ask",
            "Speech to Text": "/api/transcribe",
            "Text to Speech": "/api/synthesize"
        },
        "documentation": "/docs"
    }

@app.get("/api/test-vectorstore")
async def test_vectorstore():
    try:
        # Perform a simple similarity search with a test query
        test_query = "test"
        results = vectorstore.similarity_search(test_query, k=1)
        return {
            "status": "success",
            "document_found": len(results) > 0,
            "sample_content": results[0].page_content if results else None
        }
    except Exception as e:
        logger.error(f"Error testing vectorstore: {e}")
        return {"error": str(e)}

@app.delete("/api/delete-all-documents")
async def delete_all_documents():
    """Delete all documents from the vector store"""
    try:
        success = pinecone_manager.delete_all_vectors()
        return {
            "message": "Successfully deleted all documents from the vector store",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete-documents/{document_id}")
async def delete_documents(document_id: str):
    """Delete specific document by ID"""
    try:
        success = pinecone_manager.delete_vectors_by_ids([document_id])
        return {
            "message": f"Successfully deleted document {document_id}",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)