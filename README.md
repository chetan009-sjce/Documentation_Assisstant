# Documentation_Assistant
The LangChain Documentation Assisstant is a AI-powered web application.This intelligent documentation assistant provides accurate answers to questions about LangChain documentation using advanced Retrieval-Augmented Generation (RAG) techniques, enhanced with web crawling capabilities and conversational memory.

Prerequisites
Python 3.8 or higher
GOOGLE API key
Pinecone API key
Tavily API key (required - for documentation crawling and web search)

Installation
Clone the repository
```bash
git clone https://github.com/chetan009-sjce/Documentation-Assisstant.git
cd documentation-helper
```
Set up environment variables

Create a .env file in the root directory:
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # Required - for documentation crawling
```

Install dependencies
```bash
pipenv install
```
Ingest LangChain Documentation (Run the ingestion pipeline)

python ingestion.py  # Uses Tavily to crawl and index documentation
Run the application
```bash
streamlit run main.py
```
Open your browser and navigate to http://localhost:8501
