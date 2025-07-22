from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()
from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
# from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# # from documentation-helper.consts import INDEX_NAME
# from documentation-helper.consts import INDEX_NAME
import asyncio
import threading

# Fix: Ensure event loop is created for the current thread
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")

# embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2025", embedding=embeddings
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
    docsearch = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)
    chat = ChatGoogleGenerativeAI(verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
    docsearch = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)
    chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    rag_chain = (
            {
                "context": docsearch.as_retriever() | format_docs,
                "input": RunnablePassthrough(),
            }
            | retrieval_qa_chat_prompt
            | chat
            | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()

    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
        answer=rag_chain
    )

    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result
