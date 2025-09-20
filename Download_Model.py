import os
from huggingface_hub import snapshot_download, hf_hub_download
from dotenv import load_dotenv

# --- Configuration ---
LOCAL_MODEL_DIR = "local_models"

MODELS_TO_DOWNLOAD = [
    # --- Full Repository Snapshots ---
    {"type": "snapshot", "repo_id": "BAAI/bge-large-en-v1.5", "local_name": "bge-large-en-v1.5"},
    {"type": "snapshot", "repo_id": "BAAI/bge-reranker-large", "local_name": "bge-reranker-large"},
    {"type": "snapshot", "repo_id": "deepset/roberta-base-squad2", "local_name": "roberta-base-squad2"},
    {"type": "snapshot", "repo_id": "bert-large-uncased-whole-word-masking-finetuned-squad",
     "local_name": "bert-large-squad"},

    # --- Single GGUF File Downloads ---
    {"type": "single_file", "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
     "filename": "mistral-7b-instruct-v0.2.Q3_K_M.gguf", "local_name": "mistral-7b-instruct-v0.2-gguf"},
    {"type": "single_file", "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
     "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf", "local_name": "mistral-7b-instruct-v0.2-gguf"},
    {"type": "single_file", "repo_id": "TheBloke/Zephyr-7B-beta-GGUF", "filename": "zephyr-7b-beta.Q4_K_M.gguf",
     "local_name": "zephyr-7b-beta-gguf"},
    {"type": "single_file", "repo_id": "TheBloke/CodeLlama-7B-Instruct-GGUF",
     "filename": "codellama-7b-instruct.Q4_K_M.gguf", "local_name": "codellama-7b-instruct-gguf"},
]


class ModelDownloader:
    def __init__(self, models_list, local_dir, token):
        self.models_to_download = models_list
        self.local_dir = local_dir
        self.token = token

    def run_downloads(self, status_callback=None):
        os.makedirs(self.local_dir, exist_ok=True)
        if status_callback: status_callback("Starting model download process...")

        for model_info in self.models_to_download:
            download_type = model_info.get("type", "snapshot")

            if download_type == "snapshot":
                self._download_snapshot(model_info, status_callback)
            elif download_type == "single_file":
                self._download_single_file(model_info, status_callback)

        if status_callback: status_callback("Model download process finished.")
        return True

    def _download_snapshot(self, model_info, status_callback=None):
        repo_id = model_info["repo_id"]
        local_name = model_info["local_name"]
        local_path = os.path.join(self.local_dir, local_name)

        if os.path.exists(local_path):
            message = f"✔️ Skipping repo '{local_name}' (already exists)."
            print(message)
            if status_callback: status_callback(message)
            return

        message = f"⬇️  Downloading repo '{local_name}'..."
        print(message)
        if status_callback: status_callback(message)

        try:
            snapshot_download(repo_id=repo_id, local_dir=local_path, resume_download=True, token=self.token)
            if status_callback: status_callback(f"✅ Downloaded '{local_name}'.")
        except Exception as e:
            error_message = f"❌ Failed to download '{local_name}'. Error: {e}"
            print(error_message)
            if status_callback: status_callback(error_message)

    def _download_single_file(self, model_info, status_callback=None):
        repo_id = model_info["repo_id"]
        filename = model_info["filename"]
        local_name = model_info["local_name"]
        local_path = os.path.join(self.local_dir, local_name)
        file_path = os.path.join(local_path, filename)

        if os.path.exists(file_path):
            message = f"✔️ Skipping file '{filename}' (already exists)."
            print(message)
            if status_callback: status_callback(message)
            return

        message = f"⬇️  Downloading file '{filename}'..."
        print(message)
        if status_callback: status_callback(message)

        os.makedirs(local_path, exist_ok=True)
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_path, resume_download=True,
                            token=self.token)
            if status_callback: status_callback(f"✅ Downloaded '{filename}'.")
        except Exception as e:
            error_message = f"❌ Failed to download '{filename}'. Error: {e}"
            print(error_message)
            if status_callback: status_callback(error_message)

if __name__ == "__main__":
    load_dotenv()
    HuggingFaceToken = os.getenv("HUGGINGFACE_TOKEN")
    downloader = ModelDownloader(MODELS_TO_DOWNLOAD, LOCAL_MODEL_DIR, HuggingFaceToken)
    downloader.run_downloads(status_callback=print)

