import os
import google.generativeai as genai  # Google Gemini API
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document  # Required for handling documents
from langchain.llms.base import LLM
from typing import Optional, List, Any
import pdfplumber

# 🔹 Set up Google Gemini API Key
genai.configure(api_key="")  

# ✅ Custom LLM Wrapper for Google Gemini
class GeminiLLM(LLM):
    """Custom LLM wrapper for Google Gemini API."""
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> str:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response else ""

    @property
    def _llm_type(self) -> str:
        return "Google Gemini"

# 🔹 Load PDF Files and Extract Text
pdf_folder = "pdf_blue"
documents = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        with pdfplumber.open(os.path.join(pdf_folder, file)) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            documents.append(Document(page_content=text))  # ✅ Convert text to `Document` objects

# 🔹 Split Text into Chunks for FAISS
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
docs = text_splitter.split_documents(documents)  # ✅ Properly split `Document` objects

# 🔹 Load Embedding Model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

# 🔹 Try to Load FAISS, If Not Found, Create a New One
faiss_index_path = "faiss_index_blue_new"

if not os.path.exists(faiss_index_path):  
    print("FAISS index not found! Creating a new one...")  
    vectorstore = FAISS.from_documents(docs, embeddings)  
    vectorstore.save_local(faiss_index_path)  
else:
    print("FAISS index found. Loading from disk...")  
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# 🔹 Create a Retriever from FAISS
retriever = vectorstore.as_retriever()

# 🔹 Initialize Google Gemini LLM
gemini_llm = GeminiLLM()

# 🔹 Create RetrievalQA Chain with Google Gemini
qa = RetrievalQA.from_chain_type(llm=gemini_llm, chain_type="stuff", retriever=retriever)

# 🔹 Interactive Query Loop
while True:
    query = input("\nType your query (or type 'exit' to quit): \n")
    if query.lower() == "exit":
        print("Exiting the system. Goodbye! 👋")
        break
    result = qa.invoke({"query": query})  # ✅ Corrected API usage
    print("\n💡 Answer: ", result)
