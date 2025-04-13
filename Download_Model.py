from huggingface_hub import login
import os
from dotenv import load_dotenv
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM
)
from auto_gptq import AutoGPTQForCausalLM
from sentence_transformers import SentenceTransformer

model = AutoGPTQForCausalLM.from_quantized("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ", device="cuda")
print("Model loaded successfully on GPU.")

load_dotenv()  # Load environment variables from .env

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HUGGINGFACE_TOKEN is None:
    raise ValueError("Hugging Face token is missing! Set it in the .env file.")

print("Token Loaded Successfully")  # Optional check
# Define local storage path
LOCAL_MODEL_DIR = "local_models"

# Define models to download
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2",
    "gpt2": "gpt2",
    #"HuggingFaceH4/zephyr-7b-beta": "zephyr-7b-beta",
    "deepset/roberta-base-squad2": "roberta-base-squad2",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "bert-large-squad",
    "mistralai/Mistral-7B-v0.1": "mistral-7b",
    "EleutherAI/gpt-j-6B": "gpt-j-6b",
    #"allenai/longformer-base-4096": "longformer-base-4096",
    #"distilbert-base-uncased": "distilbert-base-uncased",
    "roberta-base": "roberta-base",
    #"allenai/scibert_scivocab_uncased": "scibert-scivocab-uncased",
    #"google/bigbird-roberta-base": "bigbird-roberta-base",
    "TheBloke/gpt-j-6B-GPTQ": "gpt-j-6b-gptq",
    "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ": "mistral-7b-gptq",
}


def download_and_save_model(model_name, local_name):
    """Check if model exists locally; if not, download and save it."""
    local_path = os.path.join(LOCAL_MODEL_DIR, local_name)

    if os.path.exists(local_path):
        print(f"{model_name} already exists in {local_path}, skipping download.")
        return

    print(f"Downloading {model_name}...")

    os.makedirs(local_path, exist_ok=True)

    # Handle Sentence Transformers separately
    if "sentence-transformers" in model_name:
        model = SentenceTransformer(model_name)
        model.save(local_path)
    elif "GPTQ" in model_name:
        model = AutoGPTQForCausalLM.from_quantized(model_name, device="cpu")  # 👈 load quantized model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
    else:
        if "squad" in model_name or "longformer" in model_name:  # QA and extractive models
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif "gpt" in model_name or "zephyr" in model_name or "mistral" in model_name:  # Text generation models
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif "bigbird-pegasus" in model_name:  # BigBird-Pegasus is for summarization
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:  # General models
            model = AutoModel.from_pretrained(model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

    print(f"{model_name} saved to {local_path}.")


if __name__ == "__main__":
    for model, local_name in MODELS.items():
        download_and_save_model(model, local_name)
