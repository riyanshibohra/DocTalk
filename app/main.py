from .core.pdf_processor import extract_text_from_pdf, chunk_text
from .core.pinecone_manager import initialize_pinecone_index
from .core.query_manager import setup_retrieval_chain
from .core.speech_to_text import WhisperTranscriber
from .core.text_to_speech import ElevenLabsTTS
from langchain_core.documents import Document
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="DocTalk API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    vectorstore = initialize_pinecone_index()
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
        with open("temp.pdf", "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Process PDF
        extracted_text = extract_text_from_pdf("temp.pdf")
        chunks = chunk_text(extracted_text)
        
        # Clean up
        os.remove("temp.pdf")
        
        return {"message": "PDF processed successfully", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": str(e)}

@app.post("/api/ask")
async def ask_question(question: Question):
    try:
        response = qa_chain({"question": question.text, "chat_history": []})
        return {"answer": response["answer"]}
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {"error": str(e)}

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