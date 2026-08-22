"""
SQuAD Dataset Loader for training and evaluation.
Implements the standard HuggingFace SQuAD preprocessing with sliding windows.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_SEQUENCE_LENGTH
from typing import Dict, Optional

class SQuADDataset(Dataset):
    """PyTorch Dataset wrapper for processed SQuAD features."""
    
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        item = self.features[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.float),
            'start_positions': torch.tensor(item.get('start_positions', 0), dtype=torch.long),
            'end_positions': torch.tensor(item.get('end_positions', 0), dtype=torch.long)
        }


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Tokenize training examples with sliding windows and map answers to token positions.
    """
    # Tokenize with sliding windows
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # If no answers are given, set the cls_index as answer.
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

    # We also keep example_id for reference
    example_ids = []
    for i in range(len(tokenized_examples["input_ids"])):
        sample_index = sample_mapping[i]
        example_ids.append(examples["id"][sample_index])
    tokenized_examples["example_id"] = example_ids

    # Convert to list of dicts
    features = []
    for i in range(len(tokenized_examples["input_ids"])):
        features.append({
            "input_ids": tokenized_examples["input_ids"][i],
            "attention_mask": tokenized_examples["attention_mask"][i],
            "start_positions": tokenized_examples["start_positions"][i],
            "end_positions": tokenized_examples["end_positions"][i],
            "example_id": tokenized_examples["example_id"][i],
        })
        
    return features


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Tokenize validation examples. For evaluation, we need to map predictions back to the original context.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    
    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1
        
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    # Convert to list of dicts
    features = []
    for i in range(len(tokenized_examples["input_ids"])):
        features.append({
            "input_ids": tokenized_examples["input_ids"][i],
            "attention_mask": tokenized_examples["attention_mask"][i],
            "example_id": tokenized_examples["example_id"][i],
            "offset_mapping": tokenized_examples["offset_mapping"][i]
        })
        
    return features


def get_squad_dataloaders(
    batch_size: int = 16,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_length: int = MAX_SEQUENCE_LENGTH,
    doc_stride: int = 128
):
    """
    Loads SQuAD and returns train/val DataLoaders and raw datasets for evaluation.
    """
    print("Loading SQuAD dataset...")
    datasets = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_data = datasets["train"]
    if max_train_samples is not None:
        train_data = train_data.select(range(max_train_samples))
        
    val_data = datasets["validation"]
    if max_val_samples is not None:
        val_data = val_data.select(range(max_val_samples))
        
    print(f"Tokenizing {len(train_data)} training examples...")
    train_features = []
    # Process in batches to avoid memory issues
    for i in range(0, len(train_data), 1000):
        batch = train_data[i:i+1000]
        features = prepare_train_features(batch, tokenizer, max_length, doc_stride)
        train_features.extend(features)
        
    print(f"Tokenizing {len(val_data)} validation examples...")
    val_features = []
    for i in range(0, len(val_data), 1000):
        batch = val_data[i:i+1000]
        features = prepare_validation_features(batch, tokenizer, max_length, doc_stride)
        val_features.extend(features)
        
    train_dataset = SQuADDataset(train_features)
    val_dataset = SQuADDataset(val_features)
    
    # Custom collate fn is not strictly necessary if everything is a tensor of same shape
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, train_data, val_data, val_features
