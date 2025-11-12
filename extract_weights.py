from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="pchlenski/gpt2-transcoders",
    allow_patterns=["*.pt"],
    local_dir="./gpt-2-small-transcoders",
)


import torch
import numpy as np
from unittest.mock import MagicMock
import sys
import os
import types

# Mock wandb BEFORE any imports
sys.modules["wandb"] = MagicMock()
sys.modules["wandb.util"] = MagicMock()
sys.modules["wandb.util"].generate_id = lambda: "dummy"

# Patch numpy for compatibility
import numpy.core.multiarray

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

# Create output directory
output_dir = "./gpt-2-small-transcoders-weights"
os.makedirs(output_dir, exist_ok=True)

# Process all layers (0-11 for GPT-2-small)
layers = list(range(12))
transcoder_template = "./gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576.pt"

print(f"Processing {len(layers)} layers...")
print(f"Output directory: {output_dir}\n")

for layer in layers:
    transcoder_path = transcoder_template.format(layer)

    if not os.path.exists(transcoder_path):
        print(f"⚠️  Layer {layer}: File not found, skipping")
        continue

    try:
        # Load checkpoint
        checkpoint = torch.load(transcoder_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        # Extract weights
        weights = {
            "W_enc": state_dict["W_enc"],
            "b_enc": state_dict["b_enc"],
            "W_dec": state_dict["W_dec"],
            "b_dec": state_dict["b_dec"],
        }

        # Add b_dec_out if it exists (for transcoders)
        if "b_dec_out" in state_dict:
            weights["b_dec_out"] = state_dict["b_dec_out"]

        # Save pure weights to new file
        output_path = os.path.join(output_dir, f"layer_{layer}_weights.pt")
        torch.save(weights, output_path)

        print(f"✅ Layer {layer}: Extracted and saved")
        print(f"   W_enc: {weights['W_enc'].shape}, W_dec: {weights['W_dec'].shape}")

    except Exception as e:
        print(f"❌ Layer {layer}: Error - {e}")
        continue

print(f"\n✨ Done! All weights saved to {output_dir}/")
print(f"   Files: layer_0_weights.pt through layer_11_weights.pt")

import torch

# Load weights for a specific layer
weights = torch.load("gpt-2-small-transcoders-weights/layer_0_weights.pt")

# Access the weight matrices
W_enc = weights["W_enc"]  # Encoder weights
W_dec = weights["W_dec"]  # Decoder weights
b_enc = weights["b_enc"]  # Encoder bias
b_dec = weights["b_dec"]  # Decoder bias

print(W_enc.shape, W_dec.shape, b_enc.shape, b_dec.shape)
