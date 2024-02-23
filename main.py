import warnings

import pandas as pd
import torch

from data import ConversationDataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Load the data
data_path = "Data/train.csv"
data = pd.read_csv(data_path)

random_data = data.sample(n=5000, random_state=23)

# Preprocess the data
conversations = []
for index, row in random_data.iterrows():
    conversation = " [EOS] Assistant".join(
        row["texts"].replace("\n", "").split("Assistant")
    )
    conversations.append(conversation)


# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("Tokenizer/")
model = GPT2LMHeadModel.from_pretrained("pretrained_model/")
model.resize_token_embeddings(len(tokenizer)).to(device)

# Tokenize the data
tokenized_data = tokenizer(
    conversations, truncation=True, padding=True, max_length=128, return_tensors="pt"
).to(device)


# Create an instance of the custom dataset
dataset = ConversationDataset(tokenized_data)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./trained_model",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

if __name__ == "__main__":
    # Train the model
    trainer.train()
    # Save the trained model
    trainer.save_model("trained_model/")
