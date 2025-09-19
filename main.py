from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import weaviate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pdf2image import convert_from_bytes
import pytesseract
from langchain.docstore.document import Document
import tempfile
from dotenv import load_dotenv
import os
from weaviate.classes.config import Property, DataType

from fastapi.middleware.cors import CORSMiddleware
# ---------------- FASTAPI ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ✅ allow all origins
    allow_credentials=True,
    allow_methods=["*"],      # allow all HTTP methods
    allow_headers=["*"],      # allow all headers
)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- WEAVIATE ----------------
weaviate_client = weaviate.connect_to_local()

# Embeddings + LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# Ensure collection exists
if "PDFRAG" not in weaviate_client.collections.list_all():
    weaviate_client.collections.create(
        name="PDFRAG",
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
        ]
    )

collection = weaviate_client.collections.get("PDFRAG")


# ---------------- PDF UPLOAD ----------------
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(await file.read())
    temp_file.close()
    pdf_path = temp_file.name

    # --- 1. Extract text ---
    loader = PyPDFLoader(pdf_path)
    text_docs = loader.load()

    # --- 2. OCR for images ---
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = convert_from_bytes(pdf_bytes, poppler_path=r"C:\poppler\Library\bin")
    ocr_docs = []
    for i, page in enumerate(pages):
        ocr_text = pytesseract.image_to_string(page)
        if ocr_text.strip():
            ocr_docs.append(
                Document(
                    page_content=ocr_text,
                    metadata={"source": f"page_{i+1}_ocr"}
                )
            )

    all_docs = text_docs + ocr_docs

    # --- 3. Chunking ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(all_docs)

    # --- 4. Store in Weaviate ---
    for doc in docs:
        vec = embeddings.embed_query(doc.page_content)  # embed text
        collection.data.insert(
            properties={
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown")
            },
            vector=vec
        )

    return {"status": "success", "chunks_stored": len(docs)}


# ---------------- ASK QUESTIONS ----------------
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(payload: QuestionRequest):
    question = payload.question
    # Embed question
    query_vector = embeddings.embed_query(question)

    # Search top 3 chunks
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=3,
        return_properties=["text", "source"]
    )

    retrieved_chunks = [obj.properties["text"] for obj in response.objects]
    context = "\n\n".join(retrieved_chunks)

    # Pass context + question to LLM
    prompt = f"""You are a helpful assistant. 
Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    answer = llm.invoke(prompt)

    return {
        "answer": answer.content,
        "sources": [obj.properties for obj in response.objects]
    }
