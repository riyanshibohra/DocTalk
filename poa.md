Plan of action:


Backend-------

1. Set up virtual environment + dependencies
2. Get sample research data (pdfs), chunk them, generate embeddings, and store in vector store(Pinecone)
3. Implement a query pipeline that can query the vector store using Langchain to retrieve relevant chunks and synthesize answers
4. Add Whisper for speech-to-text
5. Add ElevenLabs for text-to-speech
6. Combine everything and test together
