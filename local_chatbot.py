# local_chatbot.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig # CORRECTED IMPORT
from accelerate import Accelerator
import os

# --- Configuration ---
os.environ["HF_HOME"] = r"D:\AI\Models\huggingface"

MODEL_NAME = "microsoft/DialoGPT-small"
LOCAL_MODEL_DIR = r"D:\AI\Models\dialoGPT-small"

# --- Initialize Accelerator for GPU optimization ---
accelerator = Accelerator()

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

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

print(f"Loading model {MODEL_NAME} with 4-bit quantization on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16,
    device_map=device
)

model = accelerator.prepare(model)

print(f"Model loaded successfully on {next(model.parameters()).device}")

# --- Simple Chat Loop ---
print("\n--- Start Chatting! Type 'exit' to quit. ---")
chat_history_ids = None

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Encode the new user input, adding the end of sentence token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    # Append the new input to the chat history
    # Ensure bot_input_ids is a dictionary if new_input_ids is
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=bot_input_ids.ne(tokenizer.pad_token_id), # ADDED: Pass attention_mask
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")

print("\n--- Conversation Ended ---")

# --- Optional: Save the loaded model and tokenizer locally ---
try:
    print(f"\nSaving model and tokenizer to {LOCAL_MODEL_DIR}...")
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print("Model and tokenizer saved successfully for faster future loading!")
except Exception as e:
    print(f"Error saving model: {e}")