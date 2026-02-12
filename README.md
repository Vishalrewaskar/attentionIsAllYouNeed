# Transformer from Scratch (NumPy + PyTorch)

A simple implementation of the Transformer architecture from the paper:

**"Attention Is All You Need"**

🔗 Paper Link: https://arxiv.org/abs/1706.03762

---

## Project Structure

```
├── 1-Tokenization/
│   ├── bpe.ipynb              # Byte Pair Encoding tokenization
│   └── word_frequency.csv     # Sample word frequency data
│
└── Transformer/
    ├── main/
    │   ├── embedding.py       # Text embedding with positional encoding
    │   ├── encoder.py         # Multi-head attention and encoder block
    │   └── test_model.py      # Test script for the model
    │
    └── notebook/
        ├── 01-selfAttention.ipynb    # Self-attention mechanism (NumPy)
        ├── 02-multiHead.ipynb        # Multi-head attention (NumPy)
        ├── 03-encoder.ipynb          # Complete encoder block (NumPy)
        ├── 04-decoder.ipynb          # Decoder with causal masking (NumPy)
        ├── SA-Torch.ipynb            # Self-attention (PyTorch)
        └── MHA-Torch.ipynb           # Multi-head attention (PyTorch)
```

## Features

### Implemented Components

1. **Tokenization**
   - Word-level tokenization
   - Byte Pair Encoding (BPE) basics

2. **Embeddings**
   - Word embeddings (Word2Vec & nn.Embedding)
   - Positional encoding

3. **Attention Mechanisms**
   - Self-attention
   - Multi-head attention
   - Causal masking (for decoder)

4. **Transformer Blocks**
   - Encoder block (Multi-head attention + Feed-forward + Layer norm)
   - Layer normalization
   - Residual connections
   - Feed-forward network

## Quick Start

### Using NumPy (Notebooks)

The notebooks provide step-by-step implementation:

```python
# Example sentence
text = "The animal did not cross the street because it was tired."

# Steps covered:
# 1. Tokenization
# 2. Word embeddings
# 3. Positional encoding
# 4. Self-attention computation
# 5. Multi-head attention
# 6. Complete encoder block
```

### Using PyTorch (Main Module)

```python
from embedding import TextEmbedding
from encoder import EncoderBlock

# Create embedder
embedder = TextEmbedding(d_model=64)
embedder.build_vocab([text])

# Get embeddings
X = embedder(text)  # Shape: (1, seq_len, 64)

# Create encoder
encoder = EncoderBlock(d_model=64, num_heads=8)
output = encoder(X)
```

## Parameters

- **d_model**: 64 (embedding dimension)
- **num_heads**: 8 (number of attention heads)
- **d_ff**: 256 (feed-forward hidden dimension)
- **d_k**: 8 (key/query dimension per head = d_model / num_heads)

## Example: Attention Analysis

The notebooks include analysis showing which words "it" attends to:

```
Head 0: it → animal = 0.134
Head 0: it → did = 0.129
Head 0: it → not = 0.112
```

This demonstrates that "it" learns to attend to "animal" (the correct antecedent).

## Requirements

```
numpy
torch
torchtext
gensim
```

## Learning Path

1. Start with `01-selfAttention.ipynb` to understand basic attention
2. Move to `02-multiHead.ipynb` for multi-head mechanism
3. Study `03-encoder.ipynb` for the complete encoder architecture
4. Check `04-decoder.ipynb` for causal masking
5. Compare with PyTorch implementations in `SA-Torch.ipynb` and `MHA-Torch.ipynb`

## Key Concepts

- **Self-Attention**: Allows each word to attend to all other words
- **Multi-Head Attention**: Multiple attention mechanisms running in parallel
- **Positional Encoding**: Adds position information to embeddings
- **Layer Normalization**: Normalizes activations for stable training
- **Residual Connections**: Helps gradient flow in deep networks
- **Causal Masking**: Prevents decoder from attending to future tokens

## Notes

- NumPy implementations are for educational purposes
- PyTorch implementations are production-ready
- All examples use a simple sentence for clarity
- The code focuses on understanding rather than optimization
