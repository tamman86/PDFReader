import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Centralize all settings into a single, easy-to-manage configuration dictionary.
# The chunk size here acts as a default if not provided.
CONFIG = {
    "chroma_base_path": "chroma",
    "embedding_model_name": "local_models/bge-large-en-v1.5",
    "default_chunk_size": 512,
    "chunk_overlap_ratio": 0.2,
}


class DatabaseBuilder:
    """
    Encapsulates the logic for creating and managing ChromaDB databases.
    Loads the embedding model only once for efficiency.
    """

    def __init__(self, config):
        self.config = config
        self._embedding_function = None

    @property
    def embedding_function(self):
        """Lazy loader for the embedding model."""
        if self._embedding_function is None:
            print(f"Loading embedding model for DB Builder: {self.config['embedding_model_name']}...")
            self._embedding_function = HuggingFaceEmbeddings(model_name=self.config["embedding_model_name"])
            print("✅ DB Builder embedding model loaded.")
        return self._embedding_function

    ### --- MODIFICATION: process_document now accepts chunk_size --- ###
    def process_document(self, file_path: str, title: str, revision: str, chunk_size: int):
        """
        Main method to load, split, and embed a single document.
        Now takes a specific chunk_size for this operation.
        """
        print(f"--- Processing Document: {title} ({revision}) with chunk size {chunk_size} ---")
        documents = self._load_document(file_path)
        if not documents:
            return

        ### --- MODIFICATION: Pass chunk_size to the splitter method --- ###
        chunks = self._split_text(documents, chunk_size)
        if not chunks:
            return

        self._save_to_chroma(chunks, title, revision)
        print(f"--- Finished Processing: {title} ({revision}) ---\n")

    def _load_document(self, file_path: str) -> list[Document]:
        """Loads a document from a file path."""
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

    ### --- MODIFICATION: _split_text now accepts and uses chunk_size --- ###
    def _split_text(self, documents: list[Document], chunk_size: int) -> list[Document]:
        """Splits documents into chunks using a specific chunk_size."""

        # Calculate overlap based on the provided chunk_size and configured ratio
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

    def _calculate_chunk_ids(self, chunks: list[Document]) -> list[Document]:
        """Creates a unique ID for each chunk."""
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

    def _save_to_chroma(self, chunks: list[Document], title: str, revision: str):
        """Saves document chunks to a ChromaDB database."""
        chroma_path = os.path.join(self.config["chroma_base_path"], title, revision)
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

