import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
import openai

url = "https://twelve-dragons-act.loca.lt/v1/"

openai.api_key = 'token-abc'  
llm = OpenAI(base_url=url, api_key="token-abc", model="meta-llama/Llama-3.2-1B-Instruct", max_tokens = 512)  # Replace with LLaMA or your preferred LLM
docs_path = "docs/"
docs = []
for file in os.listdir(docs_path):
    if file.endswith(".txt"):  # Ensure only text files are loaded
        loader = TextLoader(os.path.join(docs_path, file))
        docs.extend(loader.load())        
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(docs)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def chatbot(query, history):
    response = qa_chain.run(query)    
    return response

with gr.Blocks() as app:
    gr.Markdown("### 🧠 Smart LLM Chatbot with Auto-RAG")
    
    chat_interface = gr.ChatInterface(chatbot)
    
    app.launch(share=False)
