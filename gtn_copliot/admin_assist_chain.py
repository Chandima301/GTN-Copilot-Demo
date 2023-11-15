from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

def get_admin_assist_chain():
    loader = PyPDFLoader("./data/at_help.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(loader.load())
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    template = """
                Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                Always Give step by step procedure.
                Always thank the customer for using GTN Copilot at the end of the answer. 
                
                {context}
                Question: {question}
                Helpful Answer:"""
    
    rag_prompt_custom = PromptTemplate.from_template(template)

    # memory = ConversationBufferMemory(memory_key="chat_history")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm