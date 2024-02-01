from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

import google.generativeai as genai
GOOGLE_API_KEY='YOUR_API_HERE'

genai.configure(api_key=GOOGLE_API_KEY)

def chatbot(user_input1,user_input2):    
    loader = WebBaseLoader(user_input1)
    docs = loader.load()
    # Extract the text from the website data document
    text_content = docs[0].page_content

    # Convert the text to LangChain's `Document` format
    docs =  [Document(page_content=text_content, metadata={"source": "local"})]


    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    # Save to disk
    vectorstore = Chroma.from_documents(
                        documents=docs,                 # Data
                        embedding=gemini_embeddings,    # Embedding model
                        persist_directory="./chroma_db" # Directory to save data
                        )

    # Load from disk
    vectorstore_disk = Chroma(
                            persist_directory="./chroma_db",       # Directory of db
                            embedding_function=gemini_embeddings   # Embedding model
                    )

    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})


    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                    temperature=0.7, top_p=0.85,google_api_key=GOOGLE_API_KEY)

    # Prompt template to query Gemini
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    # Combine data from documents to readable string format.
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    res=rag_chain.invoke(user_input2)
    return res
