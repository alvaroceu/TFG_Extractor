import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
import os

# Import the architecture you just created
# Adjust the import path based on where your file is located
from transformer_method.BiLBERT.arq_BiLBERT_Distil import BiLBERTDistil

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
    model_name = 'twmkn9/distilbert-base-uncased-squad2'
    batch_size = 16
    epochs = 3 # 2 epochs is usually enough for a QA head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BiLBERTDistil(model_name=model_name).to(device)

    print(f"Initializing training on: {device}")

    print("Downloading and preprocessing SQuAD 2.0...")
    datasets = load_dataset("squad_v2")
    
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
    
    # CrossEntropyLoss is perfect here because we are classifying WHICH token is the start/end
    loss_fn = nn.CrossEntropyLoss()

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1} / {epochs} ---")
        
        # Gradual unfreezing
        if epoch == 0:
            # EPOCH 1: Frozen BERT. Train only new layer
            print("1: Frozen BERT")
            for param in model.distilbert.parameters():
                param.requires_grad = False
            
            optimizer = AdamW([
                {"params": model.lstm.parameters(), "lr": 5e-4},
                {"params": model.qa_outputs.parameters(), "lr": 5e-4}
            ])
            
            total_steps = len(train_dataloader)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
            
        elif epoch == 1: 
            # EPOCH 2, 3: Unforzen BERT. Train everything
            print("2: Unfrozen BERT")
            for param in model.distilbert.parameters():
                param.requires_grad = True
            
            # Differential learning rates
            optimizer = AdamW([
                {"params": model.distilbert.parameters(), "lr": 1e-5}, 
                {"params": model.lstm.parameters(), "lr": 1e-4},
                {"params": model.qa_outputs.parameters(), "lr": 1e-4}
            ])
            
            total_steps = len(train_dataloader) * (epochs - 1)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            optimizer.zero_grad()

            start_logits, end_logits = model(input_ids, attention_mask)

            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            total_batch_loss = (start_loss + end_loss) / 2

            total_batch_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            scheduler.step()

            total_loss += total_batch_loss.item()
            progress_bar.set_postfix({"loss": total_batch_loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    os.makedirs("trained_models/complete", exist_ok=True)
    save_path = "trained_models/complete/bilbert_distil_qa_weights.pth"
    print(f"Saving trained model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Training Complete!")

if __name__ == "__main__":
    main()