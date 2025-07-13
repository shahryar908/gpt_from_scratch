#  GPT from Scratch

A lightweight, GPT-style Transformer model implemented **from scratch** in **PyTorch** for **character-level text generation** using the Tiny Shakespeare dataset.  
It re-implements the core concepts from the groundbreaking paper [_"Attention Is All You Need"_](https://arxiv.org/abs/1706.03762).

---

##  Overview

This project implements a **GPT-inspired Transformer** architecture, incorporating:
- **Multi-Head Attention**
- **Scaled Dot-Product Attention**
- **Positional Encoding**
- A modular training pipeline for experimentation

It emphasizes **clarity**, **modularity**, and **educational value** — ideal for learning how Transformers work at a low level.

---

##  Key Features

-  **Transformer Architecture**: Built using custom encoder/decoder blocks
-  **Multi-Head Attention**: For learning long-range sequence dependencies
-  **Positional Encoding**: Preserves sequence order without recurrence
-  **Training Pipeline**: Tokenizes and processes Tiny Shakespeare dataset
-  **Modular Codebase**: Easily extendable and understandable

---

##  Paper Impact

This project is deeply rooted in the **"Attention Is All You Need"** paper by Vaswani et al.  
It demonstrates how **self-attention** and **multi-head attention** can be implemented from scratch and used effectively even on **small-scale datasets**, producing **coherent and context-aware outputs**.

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/shahryar908/gpt_from_scratch
cd gpt_from_scratch
```

### 2. Install Dependencies
> Requires **Python 3.8+** and **PyTorch**
```bash
pip install torch
```

### 3. Download the Dataset
```bash
python Utils/dataset.py
```

### 4. Train the Model
```bash
python training.py
```
> ⚙️ Default settings: `5 epochs`, `Adam optimizer`, `learning rate = 1e-4`

---

##  Usage

- 🏋️ **Training**: Trains on 70% of dataset, logs loss every 100 batches
- ⚙️ **Customization**: Tweak hyperparameters in `model.py` or `training.py`
- 💻 **Device Support**: Uses GPU (CUDA) if available, else CPU
- 🧵 **Extending**: Add decoding loop for text generation (currently not included)

---

##  Technical Details

| Component           | Value                              |
| ------------------- | ---------------------------------- |
| Transformer Layers  | 4                                  |
| Attention Heads     | 8                                  |
| Embedding Dimension | 512                                |
| Feedforward Dim     | 2048                               |
| Tokenizer           | Character-level                    |
| Dataset             | Tiny Shakespeare (~1MB)           |
| Loss Function       | Cross-Entropy with Teacher Forcing |
| Optimizer           | Adam                               |

---

##  Directory Structure

```
gpt_from_scratch/
├── dataset/
│   └── shakespeare/
│       └── input.txt
├── src/
│   ├── encoder.py
│   ├── decoder.py
│   ├── model.py
│   ├── multiheadattention.py
│   ├── scaledotattention.py
│   └── training.py
├── Utils/
│   ├── dataloader.py
│   ├── dataset.py
│   ├── positional_encoding.py
│   └── tokenizer.py
└── README.md
```

---

##  Future Enhancements

- 💾 Add checkpoint saving after each epoch
- 🧠 Implement text generation (sampling/greedy decoding)
- 📊 Add evaluation metrics (e.g., perplexity)
- 📈 Support larger datasets like WikiText or OpenWebText

---

##  Why This Project?

This project is built for learners, engineers, and researchers who want to:
- Understand Transformer internals from scratch
- Work with a clean, minimal PyTorch codebase
- Run and experiment on low-resource machines (e.g., laptops, Colab)

---

##  License

This project is licensed under the [MIT License](LICENSE).

---
