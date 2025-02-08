# DocTalk 🤖

An intelligent document interaction system that allows users to chat with their documents using advanced language models.

## 🛠️ Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI" />
  <img src="https://img.shields.io/badge/LangChain-339933?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white" alt="Pinecone" />
  <img src="https://img.shields.io/badge/Eleven_Labs-FF0000?style=for-the-badge&logo=elevenlabs&logoColor=white" alt="Eleven Labs" />
</div>

## 🚀 Features

- 📄 Document upload and processing
- 💬 Interactive chat with documents
- 🔍 Semantic search capabilities
- 🧠 AI-powered document analysis
- 🔗 Context-aware responses

## 🛠️ Technical Stack Details

### Frontend
- **Next.js**: React framework for production
- **React**: UI component library
- **Modern JavaScript**: ES6+ features

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **LangChain**: Framework for LLM applications
- **OpenAI**: Language model integration
- **Pinecone**: Vector database for embeddings
- **Whisper**: Speech-to-text capabilities
- **Eleven Labs**: Text-to-speech capabilities

## 🏃‍♂️ Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/riyanshibohra/DocTalk.git
```

2. **Install backend dependencies**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
npm install
```

4. **Start the backend server**
```bash
uvicorn app.main:app --reload
```

5. **Start the frontend development server**
```bash
npm run dev
```

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following variables:
```OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
ELEVEN_LABS_API_KEY=your_eleven_labs_api_key
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

