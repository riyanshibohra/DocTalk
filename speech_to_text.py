import whisper
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        """
        Initialize Whisper model
        model_name options: "tiny", "base", "small", "medium", "large"
        """
        try:
            logger.info(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text
        """
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            result = self.model.transcribe(audio_path)
            logger.info("Audio transcription completed")
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def transcribe_microphone(self, duration=5):
        """
        Record audio from microphone and transcribe it
        duration: recording duration in seconds
        """
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav
            import numpy as np

            # Record audio
            logger.info(f"Recording audio for {duration} seconds...")
            sample_rate = 16000
            recording = sd.rec(int(duration * sample_rate), 
                            samplerate=sample_rate, 
                            channels=1)
            sd.wait()
            
            # Save recording to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                wav.write(temp_audio.name, sample_rate, recording)
                
            # Transcribe the temporary file
            text = self.transcribe_audio(temp_audio.name)
            
            # Clean up temporary file
            Path(temp_audio.name).unlink()
            
            return text
            
        except Exception as e:
            logger.error(f"Error recording/transcribing from microphone: {e}")
            raise 