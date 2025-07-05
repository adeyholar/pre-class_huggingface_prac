# local_chatbot.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from accelerate import Accelerator
import os

# --- Configuration ---
# Set Hugging Face cache directory to avoid re-downloading models
os.environ["HF_HOME"] = r"D:\AI\Models\huggingface"

# Model to use: Mistral-7B-Instruct-v0.2
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LOCAL_MODEL_DIR = r"D:\AI\Models\mistral-7b-instruct-v0.2" # Directory to save/load the model locally

# --- Initialize Accelerator for GPU optimization ---
accelerator = Accelerator()

# Determine the device for model loading and inference
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Quantization Configuration (for memory efficiency on GPU) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16,
)

# --- Load Tokenizer and Model ---
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Mistral models often don't have a pad_token by default, but it's good for batching/generation.
# We set it to the EOS token as a common practice for causal LMs.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# IMPORTANT: Mistral Instruct models use a specific chat template.
# This ensures your prompts are formatted correctly for the model to understand instructions.
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"


print(f"Loading model {MODEL_NAME} with 4-bit quantization on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16,
    device_map=device
)

# Prepare model with Accelerator
model = accelerator.prepare(model)

print(f"Model loaded successfully on {next(model.parameters()).device}")

# --- Simple Chat Loop ---
print("\n--- Start Chatting! Type 'exit' to quit. ---")
chat_history = [] # Store messages as a list of dictionaries for the chat template

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Add user message to history
    chat_history.append({"role": "user", "content": user_input})

    # Apply the chat template to format the conversation for the model
    input_ids = tokenizer.apply_chat_template(
        chat_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # Generate a response
    generated_ids = model.generate(
        input_ids,
        attention_mask=input_ids.attention_mask, # ADDED: Pass the attention_mask
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Bot: {response}")

    chat_history.append({"role": "assistant", "content": response})

print("\n--- Conversation Ended ---")

# --- Optional: Save the loaded model and tokenizer locally ---
try:
    print(f"\nSaving model and tokenizer to {LOCAL_MODEL_DIR}...")
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print("Model and tokenizer saved successfully for faster future loading!")
except Exception as e:
    print(f"Error saving model: {e}")