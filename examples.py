"""
Example script demonstrating modular AMMR framework usage.

This is a simpler entry point showing how to use individual components
without running the full pipeline.
"""

import torch
from config import DEVICE, MODEL_NAME, SEED
from src import (
    initialize_reproducibility,
    print_header,
    QADataLoader,
    BaselineQAModel,
    count_parameters,
)


def example_1_basic_setup():
    """Example 1: Basic setup and model loading."""
    print_header("EXAMPLE 1: BASIC SETUP")
    
    initialize_reproducibility()
    
    # Load baseline model
    baseline = BaselineQAModel(MODEL_NAME, device=DEVICE)
    
    print(f"Model: {baseline.model_name}")
    print(f"Parameters: {baseline.num_parameters:,}")
    print(f"Device: {DEVICE}")


def example_2_tokenization():
    """Example 2: Tokenization and data loading."""
    print_header("EXAMPLE 2: TOKENIZATION")
    
    data_loader = QADataLoader(MODEL_NAME)
    
    question = "What is DistilBERT?"
    context = "DistilBERT is a smaller, faster, cheaper version of BERT."
    
    # Tokenize
    encoded = data_loader.tokenize_qa(question, context, device=DEVICE)
    
    input_ids = encoded["input_ids"]
    tokens = data_loader.get_tokens(input_ids)
    
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Sequence Length: {len(tokens)}")
    print(f"Tokens: {tokens}")


def example_3_hidden_states():
    """Example 3: Extracting hidden states."""
    print_header("EXAMPLE 3: HIDDEN STATES")
    
    data_loader = QADataLoader(MODEL_NAME)
    baseline = BaselineQAModel(MODEL_NAME, device=DEVICE)
    
    question = "What is AI?"
    context = "AI is artificial intelligence."
    
    encoded = data_loader.tokenize_qa(question, context, device=DEVICE)
    
    # Get hidden states
    hidden_states, all_layers = baseline.get_hidden_states(
        encoded["input_ids"],
        encoded["attention_mask"],
        return_all_layers=True
    )
    
    print(f"Final Hidden State Shape: {hidden_states.shape}")
    print(f"Number of Layers: {len(all_layers)}")
    
    for i, layer in enumerate(all_layers):
        print(f"  Layer {i}: {layer.shape}")


if __name__ == "__main__":
    print("=" * 80)
    print("AMMR FRAMEWORK - EXAMPLES")
    print("=" * 80)
    print()
    
    # Run examples
    example_1_basic_setup()
    print()
    
    example_2_tokenization()
    print()
    
    example_3_hidden_states()
    print()
    
    print("=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)
