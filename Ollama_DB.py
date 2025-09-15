import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CONFIG = {
    "chroma_base_path": "chroma",
    "embedding_model_name": "local_models/bge-large-en-v1.5",
    "default_chunk_size": 512,
    "chunk_overlap_ratio": 0.2,
}

class DatabaseBuilder:
    def __init__(self, config):
        self.config = config
        self._embedding_function = None

    @property
    def embedding_function(self):
        # Load embedding model
        if self._embedding_function is None:
            print(f"Loading embedding model for DB Builder: {self.config['embedding_model_name']}...")
            self._embedding_function = HuggingFaceEmbeddings(model_name=self.config["embedding_model_name"])
            print("✅ DB Builder embedding model loaded.")
        return self._embedding_function

    # Process Document for chunking (Chunk size determined by user)
    def process_document(self, file_path: str, title: str, revision: str, chunk_size: int):
        print(f"--- Processing Document: {title} ({revision}) with chunk size {chunk_size} ---")
        documents = self._load_document(file_path)
        if not documents:
            return

        chunks = self._split_text(documents, chunk_size)
        if not chunks:
            return

        self._save_to_chroma(chunks, title, revision, chunk_size)
        print(f"--- Finished Processing: {title} ({revision}) ---\n")

    # Loads a document from supplied file path
    def _load_document(self, file_path: str) -> list[Document]:
        file_extension = os.path.splitext(file_path)[1].lower()
        loader = None
        if file_extension == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension in [".md", ".txt"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"Skipping unsupported file type: {file_path}")
            return []

        print(f"Loading file: {file_path}")
        return loader.load()

    # Splits documents into chunks using user specified chunk size
    def _split_text(self, documents: list[Document], chunk_size: int) -> list[Document]:

        # Chunk overlap fo preserve context
        chunk_overlap = int(chunk_size * self.config["chunk_overlap_ratio"])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} source pages into {len(chunks)} chunks.")
        return self._calculate_chunk_ids(chunks)

    # Create a unique ID for each chunk
    def _calculate_chunk_ids(self, chunks: list[Document]) -> list[Document]:
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
            chunk.metadata["id"] = f"{current_page_id}:{start_index}:{current_chunk_index}"
            last_page_id = current_page_id
        return chunks

    # Saves the chunked document to ChromaDB
    def _save_to_chroma(self, chunks: list[Document], title: str, revision: str, chunk_size: int):
        revision_folder_name = f"{revision} (Chunk {chunk_size})"
        chroma_path = os.path.join(self.config["chroma_base_path"], title, revision_folder_name)
        if os.path.exists(chroma_path):
            print(f"Clearing existing database at: {chroma_path}")
            shutil.rmtree(chroma_path)

        os.makedirs(chroma_path, exist_ok=True)
        print(f"Creating new database at: {chroma_path}")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
            persist_directory=chroma_path
        )
        print(f"✅ Successfully saved {len(chunks)} chunks to {chroma_path}")

