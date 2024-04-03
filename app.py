from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

PINECONE_API_KEY = "6970c088-24a7-44a7-a6c0-648146ecc094"
PINECONE_API_ENV = "gcp-starter"
INDEX_NAME="medical-chatbot"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_INDEX_NAME"] = INDEX_NAME


embeddings = download_hugging_face_embeddings()

#Loading the index
docsearch=PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)

 
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


