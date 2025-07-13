GPT from Scratch
Overview
This project implements a lightweight, GPT-inspired Transformer model in PyTorch, designed for character-level text generation on the Tiny Shakespeare dataset. It incorporates key concepts from the seminal paper "Attention Is All You Need" (Vaswani et al., 2017), including the Transformer architecture and multi-head attention. The implementation prioritizes clarity, efficiency, and modularity, making it ideal for educational purposes and experimentation. The adoption of the paper's attention mechanisms enhances the model's ability to capture contextual relationships, significantly improving its text generation potential.
Key Features

Transformer Architecture: Custom-built with encoder and decoder blocks, inspired by "Attention Is All You Need".
Multi-Head Attention: Implements scaled dot-product attention for robust sequence modeling, as described in the paper.
Positional Encoding: Embeds positional information to maintain sequence order, per the original Transformer design.
Training Pipeline: Processes the Tiny Shakespeare dataset with a character-level tokenizer for efficient training.
Modular Codebase: Organized for easy modification and scalability.

Impact of "Attention Is All You Need"
The project leverages the multi-head attention mechanism and Transformer architecture from the "Attention Is All You Need" paper. This enables the model to focus on relevant parts of the input sequence dynamically, improving its ability to learn long-range dependencies in text. The result is a more coherent and context-aware model, even with the small-scale Tiny Shakespeare dataset, demonstrating the power of attention-based architectures.
Directory Structure
gpt_from_scratch/
├── dataset/
│   └── shakespeare/
│       └── input.txt          # Tiny Shakespeare dataset
├── src/
│   ├── decoder.py             # Transformer decoder with cross-attention
│   ├── encoder.py             # Transformer encoder with self-attention
│   ├── model.py               # Complete Transformer architecture
│   ├── multiheadattention.py  # Multi-head attention mechanism
│   ├── scaledotattention.py   # Scaled dot-product attention
│   └── training.py            # Training script with loss tracking
├── Utils/
│   ├── dataloader.py          # DataLoader for batch processing
│   ├── dataset.py             # Script to fetch Tiny Shakespeare dataset
│   ├── positional_encoding.py # Positional encoding module
│   └── tokenizer.py           # Character-level tokenization utilities
└── README.md                  # Project documentation

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd gpt_from_scratch


Install Dependencies:Requires Python 3.8+ and PyTorch. Install dependencies:
pip install torch


Download Dataset:Fetch the Tiny Shakespeare dataset:
python Utils/dataset.py


Train the Model:Start training with default settings (5 epochs, Adam optimizer, 1e-4 learning rate):
python training.py



Usage

Training: Trains on 70% of the Tiny Shakespeare dataset, logging loss every 100 batches and average loss per epoch.
Customization: Modify hyperparameters (e.g., embed_dim, num_layers, dropout) in model.py or training.py.
Device Support: Uses CUDA if available; otherwise, defaults to CPU.
Extending Functionality: Adaptable for text generation by adding a decoding function (not included).

Technical Details

Model Architecture: 4 Transformer layers, 8 attention heads, 512 embedding dimensions, and 2048 feed-forward dimensions.
Tokenizer: Character-level for simplicity and flexibility.
Dataset: Tiny Shakespeare (~1MB) for lightweight training.
Training: Uses teacher-forcing with cross-entropy loss and Adam optimizer.

Future Enhancements

Add model checkpointing to save weights after each epoch.
Implement text generation for inference.
Introduce evaluation metrics (e.g., perplexity) for performance analysis.
Optimize hyperparameters and explore larger datasets.

Why This Project?
This project excels in its clear implementation of the "Attention Is All You Need" principles, offering a hands-on way to understand Transformers. Its lightweight design runs efficiently on modest hardware, making it accessible for learning and experimentation.
License
Licensed under the MIT License.
