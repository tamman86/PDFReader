import os
import shutil
import torch
import math
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

CONFIG = {
    "chroma_base_path": "chroma",
    "embedding_model_name": "local_models/bge-large-en-v1.5",
    "default_chunk_size": 512,
    "chunk_overlap_ratio": 0.2,

    "generator_models": {
        "mistral-q3": {
            "path": "local_models/mistral-7b-instruct-v0.2-gguf",
            "filename": "mistral-7b-instruct-v0.2.Q3_K_M.gguf",
            "type": "gguf"
        }
    }
}

class DatabaseBuilder:
    def __init__(self, config):
        self.config = config
        self._embedding_function = None
        self._generator_llms = {}

    @property
    def embedding_function(self):
        self._clear_cached_generator()

        # Load embedding model
        if self._embedding_function is None:
            print(f"Loading embedding model for DB Builder: {self.config['embedding_model_name']}...")
            self._embedding_function = HuggingFaceEmbeddings(model_name=self.config["embedding_model_name"])
            print("✅ DB Builder embedding model loaded.")
        return self._embedding_function

    def _clear_cached_embedder(self):
        if self._embedding_function is None:
            return
        print("Clearing DB embedder model from VRAM...")
        try:
            if hasattr(self._embedding_function, 'client'):
                del self._embedding_function.client
            del self._embedding_function
        except Exception:
            pass
        self._embedding_function = None
        torch.cuda.empty_cache()

    def _clear_cached_generator(self):
        if not self._generator_llms:
            return
        print("Clearing DB generator model from VRAM...")
        keys = list(self._generator_llms.keys())
        for key in keys:
            del self._generator_llms[key]
        torch.cuda.empty_cache()

    def get_generator_llm(self, model_name="mistral-q3"):
        # Eject the embedder BEFORE loading the generator.
        self._clear_cached_embedder()

        if model_name not in self._generator_llms:
            # Eject any other generator that might be loaded.
            self._clear_cached_generator()

            model_info = self.config["generator_models"][model_name]
            print(f"Loading generator model for summarization: {model_name}...")

            gguf_path = model_info["path"]
            gguf_filename = model_info["filename"]
            gguf_file_path = os.path.join(gguf_path, gguf_filename)
            if not os.path.exists(gguf_file_path):
                raise RuntimeError(f"Model file not found: {gguf_file_path}")

            llm = LlamaCpp(
                model_path=gguf_file_path, n_gpu_layers=-1, n_batch=512,
                n_ctx=4096, max_tokens=512, verbose=False
            )
            self._generator_llms[model_name] = llm

        return self._generator_llms[model_name]

    def _generate_summary(self, chunks: list[Document]) -> str:
        print(f"  -> Generating summary using Map-Reduce for {len(chunks)} chunks...")

        # Summarize each chunk into a single sentence.

        map_prompt_template = """You are an expert archivist. Read the following text chunk and generate a *single, concise sentence* summarizing its main point.
        ---
        {chunk_text}
        ---
        SINGLE SENTENCE SUMMARY:
        """
        chunk_summaries = []
        try:
            # Load the LLM for the whole loop
            llm = self.get_generator_llm("mistral-q3")
            llm.temperature = 0.1  # Very low temp for factual, one-line summaries

            for i, chunk in enumerate(chunks):
                if (i + 1) % 50 == 0:
                    print(f"    -> Summarizing chunk {i + 1} of {len(chunks)}...")

                prompt = map_prompt_template.format(chunk_text=chunk.page_content)
                one_line_summary = llm.invoke(prompt).strip()

                # Only add non-empty summaries
                if one_line_summary:
                    chunk_summaries.append(one_line_summary)

            print(f"  -> Map step complete. Created {len(chunk_summaries)} one-line summaries.")
            if not chunk_summaries:
                print("  -> No chunk summaries generated. Aborting.")
                return "No summary could be generated."

            # Combine the one-line summaries and summarize *them*.
            combined_summaries_text = "\n".join(chunk_summaries)

            reduce_prompt = f"""You are an expert technical archivist. Read the following list of one-sentence summaries (one from each chunk of a document) and generate a final, concise 2-3 sentence summary.
        Focus *only* on the main topics, purpose, and key subjects of the entire document.
        Do not add any preamble like "This document is about...". Just output the final summary.

        LIST OF CHUNK SUMMARIES:
        ---
        {combined_summaries_text}
        ---

        FINAL CONCISE 2-3 SENTENCE SUMMARY:
        """

            # Re-use the same LLM, but maybe with slightly more creativity
            llm.temperature = 0.3
            final_summary = llm.invoke(reduce_prompt).strip()

            print(f"  -> Generated final summary: {final_summary}")
            return final_summary

        except Exception as e:
            print(f"  -> ERROR: Failed to generate summary: {e}")
            return "No summary could be generated."  # Fallback
        finally:
            # Clear generator from VRAM to make room for embedder in the next step
            self._clear_cached_generator()

    # Process Document for chunking (Chunk size determined by user)
    def process_document(self, file_path: str, title: str, revision: str, chunk_size: int):
        print(f"--- Processing Document: {title} ({revision}) with chunk size {chunk_size} ---")
        documents = self._load_document(file_path)
        if not documents:
            return

        # Chunking the document
        chunks = self._split_text(documents, chunk_size)
        if not chunks:
            return

        # Generate document summary from chunks
        summary = self._generate_summary(chunks)

        self._save_to_chroma(chunks, title, revision, chunk_size, summary)
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
    def _save_to_chroma(self, chunks: list[Document], title: str, revision: str, chunk_size: int, description: str):
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

        # Save generated document summary
        description_path = os.path.join(chroma_path, "description.txt")
        try:
            with open(description_path, 'w', encoding='utf-8') as f:
                f.write(description)
            print(f"✅ Successfully saved auto-generated summary.")
        except Exception as e:
            print(f"❌ Failed to save summary/description: {e}")

