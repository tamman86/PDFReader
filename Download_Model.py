import os
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM
)
from optimum.gptq import GPTQModel
from auto_gptq import AutoGPTQForCausalLM
from sentence_transformers import SentenceTransformer
import requests

LOCAL_MODEL_DIR = "local_models"

''' HuggingFace Token to download models '''

load_dotenv()  # Load environment variables from .env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HUGGINGFACE_TOKEN is None:
    raise ValueError("Hugging Face token is missing! Set it in the .env file.")

print("Token Loaded Successfully")

''' Define models to download '''
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2",
    "gpt2": "gpt2",

    "deepset/roberta-base-squad2": "roberta-base-squad2",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "bert-large-squad",
    "mistralai/Mistral-7B-v0.1": "mistral-7b",
    "EleutherAI/gpt-j-6B": "gpt-j-6b",
    "roberta-base": "roberta-base",
    "TheBloke/gpt-j-6B-GPTQ": "gpt-j-6b-gptq",
    "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ": "mistral-7b-gptq",
    "TheBloke/DeepSeek-LLM-7B-GPTQ": "deepseek-7b-gptq",
    "TheBloke/DeepSeek-LLM-7B-GGUF": "deepseek-7b-gguf",
    #"HuggingFaceH4/zephyr-7b-beta": "zephyr-7b-beta",
    #"allenai/longformer-base-4096": "longformer-base-4096",
    #"distilbert-base-uncased": "distilbert-base-uncased",
    #"allenai/scibert_scivocab_uncased": "scibert-scivocab-uncased",
    #"google/bigbird-roberta-base": "bigbird-roberta-base",
}

def download_and_save_model(model_name, local_name):
    ''' Check if model is already locally saved '''
    local_path = os.path.join(LOCAL_MODEL_DIR, local_name)

    if os.path.exists(local_path):
        print(f"{model_name} already exists in {local_path}, skipping download.")
        return

    print(f"Downloading {model_name}...")

    os.makedirs(local_path, exist_ok=True)

    if "sentence-transformers" in model_name:
        model = SentenceTransformer(model_name)
        model.save(local_path)
    else:
        if "squad" in model_name or "longformer" in model_name:                         # QA and extractive models
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif "gpt" in model_name or "zephyr" in model_name or "mistral" in model_name:  # Text generation models
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif "bigbird-pegasus" in model_name:                                           # BigBird-Pegasus is for summarization
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:                                                                           # General models
            model = AutoModel.from_pretrained(model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

    print(f"{model_name} saved to {local_path}.")

def download_gguf(model_name, local_path):
    os.makedirs(local_path, exist_ok=True)
    gguf_file = "deepseek-llm-7b.Q4_K_M.gguf"                                           # Customize this if needed

    url = f"https://huggingface.co/{model_name}/resolve/main/{gguf_file}"
    dest_path = os.path.join(local_path, gguf_file)

    if os.path.exists(dest_path):
        print(f"GGUF file {gguf_file} already exists at {dest_path}")
        return

    print(f"Downloading GGUF from {url}...")
    response = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved GGUF to {dest_path}")


def download_gptq(model_name, local_name):
    local_path = os.path.join(LOCAL_MODEL_DIR, local_name)
    if os.path.exists(local_path):
        print(f"{model_name} already exists at {local_path}, skipping...")
        return

    print(f"Downloading GPTQ model {model_name} to {local_path}...")
    os.makedirs(local_path, exist_ok=True)

    model = GPTQModel.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)

    print(f"Saved {model_name} to {local_path}")

if __name__ == "__main__":
    for model, local_name in MODELS.items():
        if "GGUF" in model:
            download_gguf(model, os.path.join(LOCAL_MODEL_DIR, local_name))
        elif "GPTQ" in model:
            download_gptq(model, local_name)
        else:
            download_and_save_model(model, local_name)
