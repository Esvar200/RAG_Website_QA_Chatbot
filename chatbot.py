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
import requests

from streamlitapp import GOOGLE_API_KEY
import google.generativeai as genai# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY=GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

def chatbot(user_input1,user_input2):    
    loader= requests.get(user_input1)
    txt=loader.text

    # Convert the text to LangChain's `Document` format
    docs =  [Document(page_content=txt, metadata={"source": "local"})]

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    # Save to disk
    vectorstore = Chroma.from_documents(
                        documents=docs,                 # Data
                        embedding=gemini_embeddings,    # Embedding model
                        persist_directory="./chroma_db" # Directory to save data
                        )
    vectorstore.persist()
    # Load from disk
    vectorstore_disk = Chroma(
                            persist_directory="./chroma_db",       # Directory of db
                            embedding_function=gemini_embeddings   # Embedding model
                    )
    # Get the Retriever interface for the store to use later.
    # When an unstructured query is given to a retriever it will return documents.
    # Read more about retrievers in the following link.
    # https://python.langchain.com/docs/modules/data_connection/retrievers/
    #
    # Since only 1 document is stored in the Chroma vector store, search_kwargs `k`
    # is set to 1 to decrease the `k` value of chroma's similarity search from 4 to
    # 1. If you don't pass this value, you will get a warning.
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                    temperature=0.3, top_p=0.85,google_api_key=GOOGLE_API_KEY)

    # Prompt template to query Gemini
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.
    Provide answer in a neat format if it has points.\n
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
