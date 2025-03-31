import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_BASE_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith((".pdf", ".docx", ".md"))]

    if not files:
        print("No PDF or Word documents found in the directory.")
        return

    print("\nFound Documents:")
    for i, file in enumerate(files, start=1):
        print(f"  [{i}] {file}")

    for i, file in enumerate(files, start=1):
        print(f"\n🔍 Processing: {file}")

        title = input(f"Enter the title for '{file}': ").strip()
        revision = input(f"Enter the revision name (e.g., v1, v2, latest) for '{file}': ").strip()

        file_path = os.path.join(DATA_PATH, file)
        process_document(file_path, title, revision)


def process_document(file_path, title, revision):
    documents = load_document(file_path)
    chunks = split_text(documents)
    save_to_chroma(chunks, title, revision)


def load_document(file_path):
    if file_path.endswith(".pdf"):
        print(f"Loading PDF: {file_path}")
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        print(f"Loading Word Document: {file_path}")
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".md"):
        print(f"Loading Markdown Document: {file_path}")
        loader = DirectoryLoader(file_path)
    else:
        print(f"Skipping unsupported file type: {file_path}")
        return []

    return loader.load()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    return calculate_chunk_ids(chunks)


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "N/A")

        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks


def save_to_chroma(chunks: list[Document], title, revision, batch_size=150):
    embeddings = HuggingFaceEmbeddings(model_name="local_models/all-mpnet-base-v2")
    chroma_path = os.path.join(CHROMA_BASE_PATH, title, revision)
    os.makedirs(chroma_path, exist_ok=True)
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    existing_ids = set(db.get()["ids"]) if db._collection.count() > 0 else set()

    print(f"Storing in: {chroma_path}")
    print(f"Existing document chunks in {title} - {revision}: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new document chunks to {title} - {revision}")
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=batch_ids)
            print(f"Added batch {i // batch_size + 1} with {len(batch)} chunks")
    else:
        print("No new documents to add")


if __name__ == "__main__":
    main()
