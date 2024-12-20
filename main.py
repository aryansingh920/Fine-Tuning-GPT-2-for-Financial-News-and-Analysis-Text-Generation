# main.py
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TextDataset,
)
import pandas as pd
import torch
from dataclasses import dataclass
import os


class FinancialGPT2Generator:
    def __init__(self, model_dir: str):
        if torch.backends.mps.is_available():  # Check for MPS support
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")  # Fallback to CPU
        print(f"Using device: {self.device}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device)

    def generate_news(self, ticker: str = None, max_length: int = 150, temperature: float = 0.7):
        prompt = f"Ticker: {ticker}\n" if ticker else "Breaking financial news: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
