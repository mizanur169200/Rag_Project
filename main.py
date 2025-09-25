from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import google.generativeai as genai

# ------------------- Setup ------------------- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GEMINI = bool(GOOGLE_API_KEY)

if USE_GEMINI:
    genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="Chat PDF: Local Embeddings + Gemini API")
templates = Jinja2Templates(directory="templates")  # folder for HTML templates

# ------------------- Helper Functions ------------------- #

def get_pdf_text(pdf_files: list[UploadFile]) -> str:
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks: list[str]):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return "Vector store saved locally."
    except Exception as e:
        return f"Error creating vector store: {str(e)}"

def get_conversational_chain():
    if USE_GEMINI:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    else:
        # Fallback to local HuggingFace model
        from transformers import pipeline
        from langchain.llms import HuggingFacePipeline
        hf_pipeline = pipeline("text-generation", model="google/flan-t5-small", max_length=512)
        model = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "answer is not available in the context".

Context:\n{context}\n
Question:\n{question}\n
Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def query_vector_store(user_question: str) -> str:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # ⚠️ safe because it's your local file
        )
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        if not chain:
            return "Error initializing conversational chain."
        
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "No response from model.")
    except Exception as e:
        return f"Error querying vector store: {str(e)}"

# ------------------- FastAPI Endpoints ------------------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_pdfs/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    try:
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        result = get_vector_store(text_chunks)
        return JSONResponse(content={"message": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/ask_question/")
async def ask_question(question: str = Form(...)):
    answer = query_vector_store(question)
    return JSONResponse(content={"question": question, "answer": answer})
