import requests
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("ElevenLabs API key is required. Please set ELEVEN_LABS_API_KEY in your .env file")
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # Validate API key on initialization
        self.validate_api_key()

    def validate_api_key(self):
        """
        Validate the API key by making a test request
        """
        try:
            headers = {"xi-api-key": self.api_key}
            response = requests.get(f"{self.base_url}/voices", headers=headers)
            response.raise_for_status()
            logger.info("ElevenLabs API key validated successfully")
        except Exception as e:
            logger.error(f"Invalid ElevenLabs API key: {e}")
            raise ValueError("Invalid ElevenLabs API key. Please check your API key in the .env file")

    def synthesize_speech(self, text, voice_id="21m00Tcm4TlvDq8ikWAM"):  # Default voice ID (Rachel)
        """
        Convert text to speech using ElevenLabs API
        """
        if not text:
            logger.warning("Empty text provided, skipping speech synthesis")
            return

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        try:
            # Make request to ElevenLabs API
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            # Save the audio to a file
            with open("output.mp3", "wb") as audio_file:
                audio_file.write(response.content)
            logger.info("Speech synthesized successfully and saved to output.mp3")
            
            # Play the audio (optional)
            try:
                from playsound import playsound
                playsound("output.mp3")
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Unauthorized: Please check your ElevenLabs API key")
            else:
                logger.error(f"Error synthesizing speech: {e}")
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")