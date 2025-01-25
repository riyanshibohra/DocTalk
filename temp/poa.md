Plan of action:


Backend-------

1. Set up virtual environment + dependencies
2. Get sample research data (pdfs), chunk them, generate embeddings, and store in vector store(Pinecone)
3. Implement a query pipeline that can query the vector store using Langchain to retrieve relevant chunks and synthesize answers
4. Add Whisper for speech-to-text
5. Add ElevenLabs for text-to-speech
6. Combine everything and test together


step 2: process pdfs

detail for step 2:
- get sample research data (pdfs)
- use pypdf2 to extract text from pdfs - text extraction 
- use langChain’s RecursiveCharacterTextSplitter to chunk the text while preserving context - chunking
- combine text extraction and chunking (function to process pdfs)

step 3: pinecone integration
- initialize pinecone
- generate embeddings
- store embeddings in pinecone

step 4: query retrieval with langchain
- setup a retrieval chain  (enable retrieval-based Q&A with Pinecone)
- synthesize answers

step 5: speech-to-text with whisper
step 6: text-to-speech with elevenlabs
step 7: combine everything and test together
