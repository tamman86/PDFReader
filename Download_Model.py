import os
from huggingface_hub import snapshot_download, hf_hub_download
from dotenv import load_dotenv

# --- Configuration ---
LOCAL_MODEL_DIR = "local_models"

# Load environment variables from .env to get your token
load_dotenv()
HuggingFaceToken = os.getenv("HUGGINGFACE_TOKEN")

if not HuggingFaceToken:
    print("Hugging Face token not found in .env file. Downloads may be slower or fail for gated models.")
else:
    print("Hugging Face token loaded successfully.")

MODELS_TO_DOWNLOAD = [
    # --- Full Repository Snapshots ---
    {"type": "snapshot", "repo_id": "BAAI/bge-large-en-v1.5", "local_name": "bge-large-en-v1.5"},
    {"type": "snapshot", "repo_id": "BAAI/bge-reranker-large", "local_name": "bge-reranker-large"},
    {"type": "snapshot", "repo_id": "deepset/roberta-base-squad2", "local_name": "roberta-base-squad2"},
    {"type": "snapshot", "repo_id": "bert-large-uncased-whole-word-masking-finetuned-squad",
     "local_name": "bert-large-squad"},

    # --- Single GGUF File Downloads ---
    {
        "type": "single_file",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q3_K_M.gguf",
        "local_name": "mistral-7b-instruct-v0.2-gguf"
    },
{
        "type": "single_file",
        "repo_id": "TheBloke/Zephyr-7B-beta-GGUF",
        "filename": "zephyr-7b-beta.Q4_K_M.gguf",
        "local_name": "zephyr-7b-beta-gguf"
    },
    {
        "type": "single_file",
        "repo_id": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "filename": "codellama-7b-instruct.Q4_K_M.gguf",
        "local_name": "codellama-7b-instruct-gguf"
    },
]

def download_model_snapshot(repo_id, local_name):
    local_path = os.path.join(LOCAL_MODEL_DIR, local_name)
    if os.path.exists(local_path):
        print(f"✔️ Repo '{repo_id}' already exists at '{local_path}'. Skipping.")
        return

    print(f"⬇️  Downloading full repo '{repo_id}' to '{local_path}'...")
    try:
        snapshot_download(
            repo_id=repo_id, local_dir=local_path,
            local_dir_use_symlinks=False, resume_download=True, token=HuggingFaceToken
        )
        print(f"✅ Successfully downloaded '{repo_id}'.")
    except Exception as e:
        print(f"❌ Failed to download '{repo_id}'. Error: {e}")

def download_single_gguf_file(repo_id, filename, local_name):
    local_path = os.path.join(LOCAL_MODEL_DIR, local_name)
    file_path = os.path.join(local_path, filename)

    if os.path.exists(file_path):
        print(f"✔️ File '{filename}' already exists at '{file_path}'. Skipping.")
        return

    print(f"⬇️  Downloading single file '{filename}' from '{repo_id}'...")
    os.makedirs(local_path, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HuggingFaceToken
        )
        print(f"✅ Successfully downloaded '{filename}'.")
    except Exception as e:
        print(f"❌ Failed to download '{filename}'. Error: {e}")


if __name__ == "__main__":
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    print("--- Starting Model Download Process ---")

    for model_info in MODELS_TO_DOWNLOAD:
        download_type = model_info.get("type", "snapshot")

        if download_type == "snapshot":
            download_model_snapshot(model_info["repo_id"], model_info["local_name"])
        elif download_type == "single_file":
            download_single_gguf_file(model_info["repo_id"], model_info["filename"], model_info["local_name"])

    print("--- Model Download Process Finished ---")

