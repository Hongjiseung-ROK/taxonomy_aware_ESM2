import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys
import json
import random

# Import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import TaxonomyAwareESM

def visualize_phylum_embeddings(model_path, vocab_path, output_dir):
    print("Visualizing Random Phylum Embeddings (Rank 1)...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model Weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Vocab
    try:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    except FileNotFoundError:
        print(f"Error: Vocab file not found at {vocab_path}")
        return
    
    # Filter out <UNK> if possible, or keep if it's all there is
    candidates = [name for name in vocab.keys() if name != "<UNK>"]
    if not candidates:
        candidates = list(vocab.keys())
        
    # Select 4 random
    if len(candidates) >= 4:
        selected_names = random.sample(candidates, 4)
    else:
        selected_names = candidates
        print(f"Warning: Only found {len(candidates)} candidates in vocab. Using all of them.")
        
    print(f"Selected Phyla: {selected_names}")

    # 3. Visualize Rank 1 (Phylum)
    rank_idx = 1
    key = f"tax_embeddings.{rank_idx}.weight"
    if key not in state_dict:
        print(f"Missing weight for rank {rank_idx}")
        return
        
    weight = state_dict[key].numpy()
    
    # PCA
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(weight)
    
    # Plot
    plt.figure(figsize=(12, 12))
    
    # Background: All points
    plt.scatter(transformed[:, 0], transformed[:, 1], c='lightgrey', alpha=0.5, s=20, label='Others')
    
    # Highlight Selected
    colors = ['#FF0000', '#008000', '#0000FF', '#FFA500'] # Red, DarkGreen, Blue, Orange
    
    for i, name in enumerate(selected_names):
        idx = vocab.get(name)
        if idx is not None and idx < len(transformed):
            x, y = transformed[idx]
            color = colors[i % len(colors)]
            
            plt.scatter(x, y, c=color, s=200, edgecolor='black', zorder=10, marker='*')
            
            plt.annotate(name, 
                         xy=(x, y), 
                         xytext=(10, 10),
                         textcoords='offset points',
                         fontsize=14, 
                         fontweight='bold', 
                         color=color,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        else:
            print(f"Index for {name} ({idx}) out of bounds or not found.")

    plt.title("Phylum Embedding Space (Random 4)", fontsize=16)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    
    out_file = os.path.join(output_dir, "rank_1_phylum_random_4.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default="data/vocab/phylum_vocab.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    visualize_phylum_embeddings(args.model_path, args.vocab_path, args.output_dir)
