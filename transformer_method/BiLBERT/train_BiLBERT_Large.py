import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
import os

# Import the architecture you just created
# Adjust the import path based on where your file is located
from arq_BiLBERT_Large import BiLBERTLarge

def prepare_train_features(examples, tokenizer):
    """
    Tokenizes the texts and aligns the character-level start/end positions 
    of the answers to token-level start/end positions.
    """
    # Tokenize context and question
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second", # Only truncate the context, not the question
        max_length=384,           # Standard max length for QA training
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answers are given (impossible answer), set the cls index as answer.
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

def main():
    # 1. Configuration
    model_name = 'deepset/bert-large-uncased-whole-word-masking-squad2'
    batch_size = 16
    epochs = 2 # 2 epochs is usually enough for a QA head
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Initializing training on: {device}")

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BiLBERTLarge(model_name=model_name).to(device)

    # 3. Load and Preprocess SQuAD 2.0 Dataset
    print("📥 Downloading and preprocessing SQuAD 2.0...")
    # We use the HF datasets library to get the raw SQuAD v2
    datasets = load_dataset("squad_v2")
    
    # Map the preprocessing function to the dataset
    train_dataset = datasets["train"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing training dataset",
    )
    
    # Convert to PyTorch tensors
    train_dataset.set_format("torch")
    
    # Create the DataLoader to feed data in batches
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # 4. Setup Optimizer and Loss Function
    # We only pass the LSTM and QA output parameters to the optimizer 
    # because DistilBert is frozen in your architecture.
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # CrossEntropyLoss is perfect here because we are classifying WHICH token is the start/end
    loss_fn = nn.CrossEntropyLoss()

    # 5. Training Loop
    print("🔥 Starting training...")
    model.train()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1} / {epochs} ---")
        total_loss = 0
        
        # tqdm creates a nice progress bar
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch tensors to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass: Get predictions from your custom architecture
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            # Calculate Loss (Error)
            # We add the loss of the start prediction and the end prediction
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            total_batch_loss = (start_loss + end_loss) / 2

            # Backward pass: Calculate gradients
            total_batch_loss.backward()

            # Optimize: Update the LSTM weights
            optimizer.step()

            total_loss += total_batch_loss.item()
            progress_bar.set_postfix({"loss": total_batch_loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"✅ Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # 6. Save the trained model
    os.makedirs("trained_models", exist_ok=True)
    save_path = "trained_models/bilbert_large_qa_weights.pth"
    print(f"💾 Saving trained model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("🎉 Training Complete!")

if __name__ == "__main__":
    main()