import torch
from embedding import TextEmbedding
from encoder import EncoderBlock


text = "The animal did not cross the street because it was tired."

# embedding module
embedder = TextEmbedding(d_model=64)

# Build vocab first
embedder.build_vocab([text])

# Get embedded input
X = embedder(text)

print("Embedding shape:", X.shape)  # (1, seq_len, 64)

# Create encoder
encoder = EncoderBlock(d_model=64, num_heads=8)

# Forward pass
output = encoder(X)

print("Encoder output shape:", output.shape)