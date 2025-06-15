from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder=r"D:/chatbot/End-to-End-Medical-Chatbot-Generative-AI/template")

# Set up API keys
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# Download embeddings model
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone vector store
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up LLM using OpenRouter
llm = ChatOpenAI(
    temperature=0.4,
    max_tokens=500,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct:free"
)

# Create RAG pipeline
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)