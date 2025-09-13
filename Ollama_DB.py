import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_BASE_PATH = "chroma"
DATA_PATH = "data/books"

embedding_model_name = "local_models/bge-large-en-v1.5"
chunk_size = 300
chunk_overlap = chunk_size // 3

def process_document(file_path, title, revision):
    # 1. Load the document
    documents = load_document(file_path)
    if not documents:
        print(f"Could not load document from: {file_path}. Aborting.")
        return

    # 2. Split document into chunks
    chunks = split_text(documents)
    if not chunks:
        print("Text splitting failed. Aborting.")
        return

    # 3. Saving chunks to chroma
    save_to_chroma(chunks, title, revision)
    print(f"Finished processing: {title} ({revision}) \n")

''' Breakdown of accepted formats'''

def load_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        print(f"Loading PDF: {file_path}")
        loader = PyMuPDFLoader(file_path)
    elif file_extension == ".docx":
        print(f"Loading Word Document: {file_path}")
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_extension in [".md", ".txt"]:
        print(f"Loading Markdown Document: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        print(f"Skipping unsupported file type: {file_path}")
        return []

    return loader.load()

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} source pages into {len(chunks)} chunks.")
    return calculate_chunk_ids(chunks)

def calculate_chunk_ids(chunks):
    # Create a unique ID for each chunk
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "N/A")
        start_index = chunk.metadata.get("start_index", "N/A")

        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{start_index}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks

def save_to_chroma(chunks: list[Document], title, revision, batch_size = 150):
    embedding_function = HuggingFaceEmbeddings(model_name  = embedding_model_name)

    chroma_path = os.path.join(CHROMA_BASE_PATH, title, revision)

    # Delete old database if one exists for fresh start
    if os.path.exists(chroma_path):
        print(f"Clearing existing database at: {chroma_path}")
        shutil.rmtree(chroma_path)

    os.makedirs(chroma_path, exist_ok=True)

    db = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_function,
        persist_directory = chroma_path
    )

    print(f"Successfully saved {len(chunks)} chunks to {chroma_path}")
