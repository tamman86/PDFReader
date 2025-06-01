import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

# 1. Define your custom calibration dataset
# Replace this with text extracted from your engineering documents
engineering_calibration_data = [
    "The tensile strength of structural steel was determined to be 400 MPa, with a yield point at 250 MPa, in accordance with ASTM A36 standards.",
    "Finite element analysis of the cantilever beam indicated a maximum deflection of 0.05 meters under the applied load of 10 kN.",
    "The PID controller parameters were tuned as Kp=2.0, Ki=0.5, and Kd=0.1 to achieve optimal system response for temperature regulation.",
    "In fluid dynamics, Bernoulli's principle states that an increase in the speed of a fluid occurs simultaneously with a decrease in pressure or a decrease in the fluid's potential energy.",
    "Software engineering best practices include version control, automated testing, and continuous integration/continuous deployment pipelines.",
    "The semiconductor fabrication process involved photolithography, etching, and deposition steps to create the integrated circuit layers.",
    # Add more diverse engineering sentences/paragraphs from your documents
]

# 2. Choose your model and quantization parameters
model_name = "EleutherAI/gpt-j-6b" # The original GPT-J-6B model
quantized_model_dir = "local_models/gpt-j-6b-autogptq-engineering"

# Quantization configuration
# 'bits' refers to the target bit-width (e.g., 4-bit)
# 'group_size' defines how many weights share the same quantization parameters.
# Smaller group_size (e.g., 64, 32) can yield better accuracy but slightly larger model size.
# Larger group_size (e.g., 128) might be faster but could have minor accuracy drop.
# 'desc_act=False' is often recommended for faster inference, but you can try True for slightly better accuracy.
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128, # Or try 64 or 32 for potentially better accuracy
    desc_act=False,
    # You can add other parameters as needed, refer to AutoGPTQ documentation
)

# 3. Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# Load the model in half-precision (float16) to save memory during quantization
# device_map="auto" helps manage GPU memory for larger models
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto" # Distributes model across available GPUs/CPU if too large for one
)

# 4. Prepare the calibration dataset for quantization
# AutoGPTQ's quantize method expects a list of tokenized examples.
# You can tokenize the `engineering_calibration_data` directly.
# The examples should be in a format that the model can process, typically just input_ids.
# AutoGPTQ's `quantize` method can often take a list of strings directly for simpler cases.
# However, for explicit control and to prevent potential issues, it's good practice to tokenize them.
tokenized_calibration_data = [
    tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).input_ids[0]
    for text in engineering_calibration_data
]

# 5. Perform the quantization
print(f"Starting quantization of {model_name} with {quantize_config.bits}-bit to {quantized_model_dir}...")
model_quantized = AutoGPTQForCausalLM.from_pretrained(
    model, # Pass the loaded model instance
    quantize_config=quantize_config,
    examples=tokenized_calibration_data, # Use your tokenized custom data here
    # Use_triton is often faster on modern GPUs but can have compatibility issues, test it.
    # use_triton=True,
    # tune_mlp_fusion=True # Can optimize MLP layers, potentially improving performance
)

# 6. Save the quantized model and tokenizer
model_quantized.save_pretrained(quantized_model_dir)
tokenizer.save_pretrained(quantized_model_dir)

print(f"Quantized model and tokenizer saved to: {quantized_model_dir}")

# Now you can load this quantized model in your application
# model = AutoGPTQForCausalLM.from_quantized(
#     "local_models/gpt-j-6b-autogptq-engineering",
#     device="cuda:0",
#     use_safetensors=True,
#     local_files_only=True,
#     trust_remote_code=True,
#     disable_exllamav2=True,
#     inject_fused_attention=False,
#     inject_fused_mlp=False
# )