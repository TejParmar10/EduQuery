from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import tempfile
from typing import List
# Tqdm is a popular Python library that provides a simple and convenient way to add 
# progress bars to loops and iterable objects. 
from tqdm import tqdm

def create_llm():
    """
    Create an instance of the Phi-1.5 model using Hugging Face Transformers.
    
    Returns:
    - model: The Phi-1.5 model loaded and ready to generate responses.
    - tokenizer: The tokenizer for the Phi-1.5 model.
    """

    # Load the Phi-1.5 model and tokenizer
    model_name = "stabilityai/phi-1_5"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure the model runs on CPU since we are on Mac with no GPU
    device = "cpu"
    model = model.to(device)

    return model, tokenizer
def generate_text(prompt, model, tokenizer):
    """
    Generate text based on the given prompt using the Phi-1.5 model.
    
    Args:
    - prompt: The input prompt for the model.
    - model: The Phi-1.5 model loaded using create_llm.
    - tokenizer: The tokenizer for the Phi-1.5 model.
    
    Returns:
    - response: Generated text based on the input prompt.
    """

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Generate text using the model
    outputs = model.generate(inputs.input_ids, max_length=100)

    # Decode the generated text back into a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def create_vector_store(pdf_files: List):
    """
    Create In-memory FAISS vetor store using uploaded Pdf

    Args:
    - pdf_files(List): PDF file uploaded
    retunrs:
    - vector_store: In-memory Vector store fo further processing at chat app

    """
    vector_store = None

    if pdf_files:
        text = []
        
        for file in tqdm(pdf_files, desc="Processing files"):
            #Get the file and check it's extension
            file_extension = os.path.splitext(file.name)[1]
            #Write the PDF file to temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            #Load the PDF files using PyPdf library 
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            #Load if text file
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        #Split the file to chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=10)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store and storing document chunks using embedding model
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store
    
