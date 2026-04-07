# Local Conversational Transformer

A locally trained **decoder-only Transformer language model** built using PyTorch. The model is designed to generate conversational responses by learning from tokenized dialogue data. This project implements the core components of modern language models from scratch, including tokenization, attention mechanisms, transformer blocks, training pipelines, and probabilistic text generation.

The goal of the project is to gain a practical understanding of how transformer-based language models work internally while building a lightweight conversational model that can be trained and run locally.

---

## Features

### Byte-Level BPE Tokenization
- Uses **ByteLevelBPETokenizer** for subword tokenization.
- Loads vocabulary and merge rules from `vocab.json` and `merges.txt`.
- Converts conversational text into token IDs for training and inference.

### Efficient Dataset Handling
- Training tokens are stored using **NumPy memmap (`tokens.bin`)**.
- Allows training on large token datasets without loading everything into memory.
- Random sequence sampling is used to generate training batches.

### Context Window Training
- Fixed context window of **256 tokens**.
- Model learns using **autoregressive next-token prediction**.

---

## Transformer Architecture

The model implements a **decoder-only GPT-style transformer**.

### Embeddings
- Token embeddings
- Sinusoidal positional encoding based on the formulation from *Attention Is All You Need*

### Multi-Head Self Attention
Custom implementation including:
- Query, Key, Value projections
- Scaled dot-product attention
- Causal masking for autoregressive generation
- Multi-head attention aggregation

### Feed Forward Network
Each transformer block includes a feed-forward network with:
- Linear expansion layer
- GELU activation
- Dropout regularization
- Linear projection back to model dimension

### Transformer Blocks
Each block consists of:
- Layer normalization
- Multi-head self-attention
- Residual connections
- Feed-forward network

### Model Configuration
- Transformer layers: **10**
- Hidden dimension (`d_model`): **384**
- Attention heads: **8**
- Context length: **256 tokens**

---

## Training Pipeline

The model is trained using **autoregressive next-token prediction**.

### Training Setup
- Optimizer: **AdamW**
- Learning rate: **2e-4**
- Weight decay: **0.01**
- Loss function: **CrossEntropyLoss**
- Training steps: **70,000**

### Training Process
1. Random token batches are sampled from the dataset.
2. Sequences are passed through the transformer.
3. The model predicts the probability distribution of the next token.
4. Cross entropy loss is computed between predicted tokens and ground truth.
5. Gradients are backpropagated and model weights are updated.

Model weights are saved to:


model_weights.pth


---

## Text Generation

The model generates responses using **autoregressive decoding**.

### Sampling Techniques
- **Temperature scaling** to control randomness.
- **Repetition penalty** to discourage repeating tokens.
- **Top-K sampling** to restrict predictions to the most probable tokens.
- **Top-P (nucleus) sampling** to dynamically filter the probability distribution.

### Token Filtering
Certain undesirable tokens and sequences are suppressed during generation to improve output quality.

### Conversation Formatting
Prompts are formatted as:


```
<start_convo>
user:
model:
```


Generation stops when:
- a new `user:` token is generated
- an `<end_convo>` token appears

---

## GPU Support

The model automatically detects and uses CUDA if available.

Example startup output:

```text
CUDA available: True
NVIDIA GeForce RTX 4050 Laptop GPU
Using device: CUDA
```

## Project Structure

```
project-root/
│
├── tokenizer/
│   ├── vocab.json
│   └── merges.txt
│
├── data/
│   └── tokens.bin
│
├── model_weights.pth
└── start.py
```


## Running the Project

### 1. Train the Model
```text
python start.py
```
If `model_weights.pth` already exists, the model will load the saved weights instead of retraining.

### 2. Run the Interactive Chatbot

After the model loads, you can interact with it in the terminal:

```text
You: hello
Model: ...
```

To exit the program:

```text
exit
```

## Purpose

This project was created to understand the internal mechanics of **transformer-based language models** by implementing the major components manually, including tokenization, positional encoding, multi-head attention, transformer blocks, autoregressive training, and probabilistic text generation.
