# llm_manager.py

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline # CORRECTED IMPORT for pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
from accelerate import Accelerator

import config

class LLMManager:
    def __init__(self):
        os.environ["HF_HOME"] = config.HF_CACHE_DIR

        self.accelerator = Accelerator()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"LLM Manager using device: {self.device}")

        self.tokenizer = None # Initialize as None, will be set in _load_model_and_tokenizer
        self.model = None     # Initialize as None, will be set in _load_model_and_tokenizer
        self.llm_pipeline = None

        self._load_model_and_tokenizer()
        self._create_llm_pipeline()

    def _load_model_and_tokenizer(self):
        print(f"Loading tokenizer for {config.LLM_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"

        print(f"Loading model {config.LLM_MODEL_NAME} with 4-bit quantization on {self.device}...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.LOAD_IN_4BIT,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=config.COMPUTE_DTYPE,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            quantization_config=quantization_config,
            torch_dtype=config.COMPUTE_DTYPE,
            device_map=self.device
        )

        self.model = self.accelerator.prepare(self.model)
        print(f"Model loaded successfully on {next(self.model.parameters()).device}")

    def _create_llm_pipeline(self):
        # Ensure tokenizer and model are loaded before creating pipeline
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Tokenizer or Model not loaded. Cannot create LLM pipeline.")

        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=config.MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.pad_token_id, # Pylance should now infer type
            do_sample=config.DO_SAMPLE,
            top_k=config.TOP_K,
            top_p=config.TOP_P,
            temperature=config.TEMPERATURE,
            no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            return_full_text=False
        )

    def get_llm_pipeline(self):
        return self.llm_pipeline

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device

    def save_model_and_tokenizer(self):
        if self.tokenizer and self.model: # Check if objects exist before saving
            try:
                print(f"\nSaving LLM model and tokenizer to {config.LLM_LOCAL_DIR}...")
                self.tokenizer.save_pretrained(config.LLM_LOCAL_DIR) # Pylance should now infer type
                self.model.save_pretrained(config.LLM_LOCAL_DIR)     # Pylance should now infer type
                print("LLM model and tokenizer saved successfully!")
            except Exception as e:
                print(f"Error saving LLM model: {e}")
        else:
            print("Cannot save model/tokenizer: not loaded.")