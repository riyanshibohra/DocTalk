import os
from dotenv import load_dotenv
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs import generate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from streamlit_chat import message

from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient

import warnings
warnings.filterwarnings("ignore")

# Load environment variables from the .env file
load_dotenv()

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Function to load embeddings and database

def load_embeddings_and_database():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Initialize Pinecone
    pc = PineconeClient()
    
    # Get the existing index
    db = Pinecone(
        index=pc.Index("voice-assistant"),
        embedding=embeddings,
        text_key="text"
    )
    
    return db

# Function that takes an audio file path and OpenAI API key as parameters
# Transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return response.text
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None
    
# Record audio using audio_recorder and transcribe using transcribe_audio
def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription

# Display the transcription of the audio on the app
def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")

# Get user input from Streamlit text input field
def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")

# Search the database for a response based on the user's query
def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    model = ChatOpenAI(model_name='gpt-3.5-turbo')
    template = """Answer the question based on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    response = chain.invoke(user_input)
    return {"result": response, "source_documents": []}  # Maintaining the expected return format

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))
        
        # Simple ElevenLabs generation
        text = history["generated"][i]
        audio = generate(
            text=text,
            voice="Bella",
            api_key=eleven_api_key
        )
        st.audio(audio, format='audio/mp3')

# Main function to run the app
def main():
    # Initialize Streamlit app with a title
    st.write("# VoiceGPT 🎙️")
   
    # Load embeddings and the Pinecone database
    db = load_embeddings_and_database()

    # Record and transcribe audio
    transcription = record_and_transcribe_audio()

    # Get user input from text input or audio transcription
    user_input = get_user_input(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
        
    # Search the database for a response based on user input and update the session state
    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

    print("successfully!")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()