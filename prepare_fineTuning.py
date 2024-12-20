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


@dataclass
class FinancialGPT2Config:
    model_name: str = "gpt2"
    max_length: int = 512
    batch_size: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./financial-gpt2"
    logging_dir: str = "./logs"
    evaluation_strategy: str = "epoch"
    save_steps: int = 500


class FinancialGPT2Trainer:
    def __init__(self, config: FinancialGPT2Config, csv_path: str):
        self.config = config
        self.csv_path = csv_path
        if torch.backends.mps.is_available():  # Check for MPS support
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")  # Fallback to CPU
        print(f"Using device: {self.device}")

        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            config.model_name).to(self.device)

        # Pad token setup
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def prepare_dataset(self, max_length: int):
        """Convert CSV to text and prepare GPT-2 compatible dataset."""
        df = pd.read_csv(self.csv_path)

        # Combine text fields
        df['text'] = df.apply(lambda row: f"Ticker: {row['ticker']}\nTitle: {row['title']}\n"
                              f"Summary: {row['summary']}\nDescription: {row['description']}\n"
                              f"Date: {row['date']}\nProvider: {row['provider']}\n\n", axis=1)

        # Save to a single .txt file
        dataset_path = "financial_news.txt"
        with open(dataset_path, "w") as f:
            for text in df['text']:
                f.write(text + "\n")

        # Create TextDataset
        return TextDataset(
            tokenizer=self.tokenizer,
            file_path=dataset_path,
            block_size=max_length
        )

    def train(self):
        """Fine-tune GPT-2 on the dataset."""
        print("Preparing dataset...")
        dataset = self.prepare_dataset(self.config.max_length)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # GPT-2 doesn't use masked language modeling
        )

        print("Starting training...")
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=self.config.logging_dir,
            evaluation_strategy="no",  # Disable evaluation
            save_steps=self.config.save_steps
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )

        trainer.train()
        print("Training completed. Saving the model...")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)


def main():
    # Define the configuration
    config = FinancialGPT2Config()

    # Path to your CSV file (update this path if necessary)
    csv_path = "data/finance_news.csv"

    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file '{csv_path}' not found. Please provide a valid path.")

    # Initialize the trainer
    trainer = FinancialGPT2Trainer(config, csv_path)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
